mod model;
mod store_adapters;

use std::path::PathBuf;

use anyhow::Context as _;
use burn::store::{ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore};
use clap::ValueEnum;
use clap::{Parser, Subcommand};
use image::imageops::FilterType;
use image::imageops::{rotate90, rotate180, rotate270};
use image::{GenericImage, RgbImage};
use std::os::unix::io::AsRawFd as _;
use tokenizers::Tokenizer;

const DEFAULT_OCR_PROMPT: &str = "<image>\nFree OCR.";

fn argmax_f32(values: &[f32], banned: Option<&[bool]>) -> (usize, f32, usize) {
    let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
    let mut nan = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if banned
            .map(|m| m.get(i).copied().unwrap_or(false))
            .unwrap_or(false)
        {
            continue;
        }
        if v.is_nan() {
            nan += 1;
            continue;
        }
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    (best_i, best_v, nan)
}

fn topk_f32(values: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(values.len());
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if best.len() < k {
            best.push((i, v));
            if best.len() == k {
                best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            continue;
        }
        // best is kept in descending order.
        if best.last().copied().is_some_and(|(_, min_v)| v <= min_v) {
            continue;
        }
        // Insert in sorted position (k is tiny, O(k) is fine).
        let mut pos = 0usize;
        while pos < best.len()
            && best[pos]
                .1
                .partial_cmp(&v)
                .unwrap_or(std::cmp::Ordering::Less)
                == std::cmp::Ordering::Greater
        {
            pos += 1;
        }
        best.insert(pos, (i, v));
        best.truncate(k);
    }
    best
}

fn nan_min_max(values: &[f32]) -> (usize, f32, f32) {
    let mut nan = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        min = min.min(v);
        max = max.max(v);
    }
    if nan == values.len() {
        min = f32::NAN;
        max = f32::NAN;
    }
    (nan, min, max)
}

fn no_repeat_ngram_banned_ids(ids: &[i64], n: usize) -> Vec<usize> {
    if n == 0 || ids.len() < n {
        return Vec::new();
    }
    let prefix_len = n - 1;
    let cur_prefix = &ids[ids.len() - prefix_len..];

    use std::collections::HashMap;
    let mut map: HashMap<Vec<i64>, Vec<i64>> = HashMap::new();
    for i in 0..=ids.len().saturating_sub(n) {
        let key = ids[i..i + prefix_len].to_vec();
        let next = ids[i + prefix_len];
        map.entry(key).or_default().push(next);
    }

    let mut banned: Vec<usize> = Vec::new();
    if let Some(nexts) = map.get(cur_prefix) {
        for &t in nexts.iter() {
            if let Ok(idx) = usize::try_from(t) {
                banned.push(idx);
            }
        }
    }
    banned
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Inspect a `.safetensors` file (list tensor names, shapes, and dtypes).
    Inspect {
        /// Path to a `.safetensors` file.
        #[arg(long)]
        weights: PathBuf,
        /// How many entries to print (0 = all).
        #[arg(long, default_value_t = 50)]
        take: usize,
    },

    /// Text-only generation (language backbone only).
    GenerateText {
        /// Backend to run on (vulkan GPU or ndarray CPU).
        #[arg(long, value_enum, default_value_t = BackendKind::Vulkan)]
        backend: BackendKind,
        /// Path to `model-00001-of-000001.safetensors`.
        #[arg(long)]
        weights: PathBuf,
        /// Path to `tokenizer.json`.
        #[arg(long)]
        tokenizer: PathBuf,
        /// Prompt text.
        #[arg(long)]
        prompt: String,
        /// Max new tokens.
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// EOS token id (defaults to 1 for DeepSeek-OCR-2).
        #[arg(long, default_value_t = 1)]
        eos_token_id: i64,
        /// Override number of hidden layers (debug helper).
        #[arg(long, default_value_t = 12)]
        num_hidden_layers: usize,
        /// Cast BF16 weights to F16 while loading (debug helper).
        #[arg(long)]
        cast_f16: bool,
        /// KV cache dtype (F32 is safest; F16 can reduce memory at the cost of potential numeric drift).
        #[arg(long, value_enum, default_value_t = KvCacheDtype::F32)]
        kv_cache: KvCacheDtype,
        /// Best-effort reduce CPU-side memory after loading weights (drop OS page cache + malloc_trim on glibc).
        #[arg(long)]
        trim_memory: bool,
    },

    /// End-to-end OCR (image + language) using DeepSeek-OCR-2 (global view only for now).
    GenerateOcr {
        /// Backend to run on (vulkan GPU or ndarray CPU).
        #[arg(long, value_enum, default_value_t = BackendKind::Vulkan)]
        backend: BackendKind,
        /// Path to `model-00001-of-000001.safetensors`.
        #[arg(long)]
        weights: PathBuf,
        /// Path to `tokenizer.json`.
        #[arg(long)]
        tokenizer: PathBuf,
        /// Image file path (png/jpg).
        #[arg(long)]
        image: PathBuf,
        /// Max new tokens.
        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
        /// EOS token id (defaults to 1 for DeepSeek-OCR-2).
        #[arg(long, default_value_t = 1)]
        eos_token_id: i64,
        /// Image placeholder token id (defaults to 128815 for DeepSeek-OCR-2 tokenizer).
        #[arg(long, default_value_t = 128_815)]
        image_token_id: i64,
        /// Square size for the global view (defaults to 1024).
        #[arg(long, default_value_t = 1024)]
        image_size: u32,
        /// Disable dynamic tiling into 768x768 crops for large images.
        #[arg(long)]
        no_crop: bool,
        /// Rotate the input image clockwise by the given degrees before preprocessing (0/90/180/270).
        #[arg(long, value_enum, default_value_t = Rotate::R0)]
        rotate: Rotate,
        /// Try to auto-rotate sideways images (only tries 90/270 using a cheap heuristic).
        ///
        /// Manual `--rotate` always wins over this flag.
        #[arg(long)]
        auto_rotate: bool,
        /// Local crop size (defaults to 768, matches HF reference).
        #[arg(long, default_value_t = 768)]
        crop_image_size: u32,
        /// Forbid repeating n-grams of this size during generation (HF example uses 20).
        #[arg(long, default_value_t = 20)]
        no_repeat_ngram_size: usize,
        /// KV cache dtype (F32 is safest; F16 can reduce memory at the cost of potential numeric drift).
        #[arg(long, value_enum, default_value_t = KvCacheDtype::F32)]
        kv_cache: KvCacheDtype,
        /// Best-effort reduce CPU-side memory after loading weights (drop OS page cache + malloc_trim on glibc).
        #[arg(long)]
        trim_memory: bool,
    },

    /// Sanity-check Burn's RoPE implementation on the selected backend.
    DebugRope {
        /// Max sequence length used to build the RoPE cache.
        #[arg(long, default_value_t = 16)]
        max_seq_len: usize,
        /// Head dimension (must be even).
        #[arg(long, default_value_t = 128)]
        head_dim: usize,
        /// Sequence length to apply RoPE to.
        #[arg(long, default_value_t = 4)]
        seq_len: usize,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum BackendKind {
    Vulkan,
    Ndarray,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum KvCacheDtype {
    F32,
    F16,
}

#[derive(Debug)]
struct GenerateTextOpts {
    backend: BackendKind,
    weights: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    eos_token_id: i64,
    num_hidden_layers: usize,
    cast_f16: bool,
    kv_cache: KvCacheDtype,
    trim_memory: bool,
}

#[derive(Debug)]
struct GenerateOcrOpts {
    backend: BackendKind,
    weights: PathBuf,
    tokenizer: PathBuf,
    image: PathBuf,
    max_new_tokens: usize,
    eos_token_id: i64,
    image_token_id: i64,
    image_size: u32,
    crop_mode: bool,
    rotate: Rotate,
    auto_rotate: bool,
    crop_image_size: u32,
    no_repeat_ngram_size: usize,
    kv_cache: KvCacheDtype,
    trim_memory: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Rotate {
    #[value(name = "0")]
    R0,
    #[value(name = "90")]
    R90,
    #[value(name = "180")]
    R180,
    #[value(name = "270")]
    R270,
}

fn rotate_apply(img: RgbImage, rotate: Rotate) -> RgbImage {
    match rotate {
        Rotate::R0 => img,
        Rotate::R90 => rotate90(&img),
        Rotate::R180 => rotate180(&img),
        Rotate::R270 => rotate270(&img),
    }
}

fn gray_u8(p: image::Rgb<u8>) -> u8 {
    // ITU-R BT.601 luma approximation (fast integer math).
    let r = p[0] as u32;
    let g = p[1] as u32;
    let b = p[2] as u32;
    ((77 * r + 150 * g + 29 * b) >> 8) as u8
}

fn downsample_for_heuristic(img: &RgbImage, max_side: u32) -> RgbImage {
    let (w, h) = img.dimensions();
    let max_dim = w.max(h).max(1);
    if max_dim <= max_side {
        return img.clone();
    }
    let scale = max_side as f32 / max_dim as f32;
    let new_w = ((w as f32) * scale).round().max(1.0) as u32;
    let new_h = ((h as f32) * scale).round().max(1.0) as u32;
    image::imageops::resize(img, new_w, new_h, FilterType::Triangle)
}

fn edge_ratio_dx_dy(img: &RgbImage) -> f64 {
    let (w, h) = img.dimensions();
    let w = w as usize;
    let h = h as usize;
    if w < 2 || h < 2 {
        return 1.0;
    }
    // Step to cap compute on huge images.
    let max_side = w.max(h);
    let step = (max_side / 256).max(1);

    let mut dx: u64 = 0;
    let mut dy: u64 = 0;
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            let g = gray_u8(*img.get_pixel(x as u32, y as u32)) as i16;
            if x + step < w {
                let g2 = gray_u8(*img.get_pixel((x + step) as u32, y as u32)) as i16;
                dx += (g - g2).unsigned_abs() as u64;
            }
            if y + step < h {
                let g2 = gray_u8(*img.get_pixel(x as u32, (y + step) as u32)) as i16;
                dy += (g - g2).unsigned_abs() as u64;
            }
        }
    }
    dx as f64 / (dy as f64 + 1e-9)
}

fn dark_top_minus_bottom(img: &RgbImage, thr: u8) -> f64 {
    let (w, h) = img.dimensions();
    let w = w as usize;
    let h = h as usize;
    if w == 0 || h == 0 {
        return 0.0;
    }
    let max_side = w.max(h);
    let step = (max_side / 256).max(1);
    let mid = h / 2;

    let mut top_cnt: u64 = 0;
    let mut top_dark: u64 = 0;
    let mut bot_cnt: u64 = 0;
    let mut bot_dark: u64 = 0;
    for y in (0..h).step_by(step) {
        for x in (0..w).step_by(step) {
            let g = gray_u8(*img.get_pixel(x as u32, y as u32));
            if y < mid {
                top_cnt += 1;
                if g < thr {
                    top_dark += 1;
                }
            } else {
                bot_cnt += 1;
                if g < thr {
                    bot_dark += 1;
                }
            }
        }
    }
    let top = top_dark as f64 / (top_cnt as f64 + 1e-9);
    let bot = bot_dark as f64 / (bot_cnt as f64 + 1e-9);
    top - bot
}

fn auto_rotate_choice(img: &RgbImage) -> Rotate {
    // Heuristic: if the image has much stronger x-gradient than y-gradient, it often means
    // horizontal text lines were captured sideways (lines become vertical -> dx grows).
    //
    // Only try 90/270 to keep the overhead minimal; users can always override with `--rotate`.
    let small = downsample_for_heuristic(img, 256);
    let r0 = edge_ratio_dx_dy(&small);
    let thresh = 1.35;
    if r0 <= thresh {
        return Rotate::R0;
    }

    let s90 = rotate90(&small);
    let s270 = rotate270(&small);
    let r90 = edge_ratio_dx_dy(&s90);
    let r270 = edge_ratio_dx_dy(&s270);

    // Prefer the candidate with the smaller dx/dy ratio (more horizontal structure).
    let mut best = if r90 <= r270 {
        Rotate::R90
    } else {
        Rotate::R270
    };
    // Tie-break: choose the one with more "ink" in the top half (dark pixels),
    // which helps disambiguate 90 vs 270 (upside-down) for common document photos.
    if (r90 - r270).abs() < 0.05 {
        let d90 = dark_top_minus_bottom(&s90, 100);
        let d270 = dark_top_minus_bottom(&s270, 100);
        best = if d90 >= d270 {
            Rotate::R90
        } else {
            Rotate::R270
        };
    }
    best
}

fn pad_to_square_rgb(image: &RgbImage, size: u32, pad_color: u8) -> anyhow::Result<RgbImage> {
    let (w, h) = image.dimensions();
    anyhow::ensure!(w > 0 && h > 0, "invalid image dimensions");

    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let new_w = (w as f32 * scale).round().max(1.0) as u32;
    let new_h = (h as f32 * scale).round().max(1.0) as u32;

    let resized = image::imageops::resize(image, new_w, new_h, FilterType::CatmullRom);

    let mut canvas =
        RgbImage::from_pixel(size, size, image::Rgb([pad_color, pad_color, pad_color]));
    let off_x = (size - new_w) / 2;
    let off_y = (size - new_h) / 2;
    canvas
        .copy_from(&resized, off_x, off_y)
        .context("failed to paste resized image")?;

    Ok(canvas)
}

fn trim_memory_after_weights_load(weights: &std::path::Path) {
    // This is best-effort memory hygiene for CPU-side allocations:
    // - Drop the safetensors file from the OS page cache (doesn't affect correctness; might slow *next* run).
    // - Ask the allocator to return freed pages back to the OS.
    //
    // Note: this doesn't touch GPU/unified memory allocations made by wgpu/Vulkan.
    fn read_self_status_kb(key: &str) -> Option<u64> {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix(key) {
                // Example: "VmRSS:\t  123456 kB"
                let n = rest
                    .split_whitespace()
                    .next()
                    .and_then(|v| v.parse::<u64>().ok())?;
                return Some(n);
            }
        }
        None
    }

    fn read_meminfo_kb() -> Option<(u64, u64)> {
        // (MemAvailable, Cached) in kB.
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut avail = None;
        let mut cached = None;
        for line in s.lines() {
            if line.starts_with("MemAvailable:") {
                avail = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<u64>().ok());
            } else if line.starts_with("Cached:") {
                cached = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<u64>().ok());
            }
        }
        Some((avail?, cached?))
    }

    let rss_before = read_self_status_kb("VmRSS:").unwrap_or(0);
    let (avail_before, cached_before) = read_meminfo_kb().unwrap_or((0, 0));

    #[cfg(target_os = "linux")]
    {
        let mut fadvise_ret: Option<i32> = None;
        if let Ok(file) = std::fs::File::open(weights) {
            // 0,0 => entire file.
            unsafe {
                fadvise_ret = Some(libc::posix_fadvise(
                    file.as_raw_fd(),
                    0,
                    0,
                    libc::POSIX_FADV_DONTNEED,
                ));
            }
        }
        if let Some(ret) = fadvise_ret.filter(|&ret| ret != 0) {
            eprintln!("trim-memory: posix_fadvise(DONTNEED) failed errno={ret}");
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let malloc_trim_ret: Option<i32> = Some(unsafe { libc::malloc_trim(0) });
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let malloc_trim_ret: Option<i32> = None;

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // If this returns 1, glibc released at least some free heap pages back to the OS.
        // 0 means "nothing to release" (common if most memory is still live or held by non-glibc allocators).
        let _ = malloc_trim_ret;
    }

    let rss_after = read_self_status_kb("VmRSS:").unwrap_or(0);
    let (avail_after, cached_after) = read_meminfo_kb().unwrap_or((0, 0));

    let d_rss = rss_after as i64 - rss_before as i64;
    let d_avail = avail_after as i64 - avail_before as i64;
    let d_cached = cached_after as i64 - cached_before as i64;

    eprintln!(
        "trim-memory: rss_kb {rss_before}->{rss_after} (d={d_rss}), mem_avail_kb {avail_before}->{avail_after} (d={d_avail}), cached_kb {cached_before}->{cached_after} (d={d_cached}), malloc_trim={:?}",
        malloc_trim_ret
    );
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Inspect { weights, take } => cmd_inspect(&weights, take),
        Command::GenerateText {
            backend,
            weights,
            tokenizer,
            prompt,
            max_new_tokens,
            eos_token_id,
            num_hidden_layers,
            cast_f16,
            kv_cache,
            trim_memory,
        } => {
            let opts = GenerateTextOpts {
                backend,
                weights,
                tokenizer,
                prompt,
                max_new_tokens,
                eos_token_id,
                num_hidden_layers,
                cast_f16,
                kv_cache,
                trim_memory,
            };
            cmd_generate_text(&opts)
        }
        Command::GenerateOcr {
            backend,
            weights,
            tokenizer,
            image,
            max_new_tokens,
            eos_token_id,
            image_token_id,
            image_size,
            no_crop,
            rotate,
            auto_rotate,
            crop_image_size,
            no_repeat_ngram_size,
            kv_cache,
            trim_memory,
        } => {
            let opts = GenerateOcrOpts {
                backend,
                weights,
                tokenizer,
                image,
                max_new_tokens,
                eos_token_id,
                image_token_id,
                image_size,
                crop_mode: !no_crop,
                rotate,
                auto_rotate,
                crop_image_size,
                no_repeat_ngram_size,
                kv_cache,
                trim_memory,
            };
            cmd_generate_ocr(&opts)
        }
        Command::DebugRope {
            max_seq_len,
            head_dim,
            seq_len,
        } => cmd_debug_rope(max_seq_len, head_dim, seq_len),
    }
}

fn cmd_debug_rope(max_seq_len: usize, head_dim: usize, seq_len: usize) -> anyhow::Result<()> {
    use burn::backend::Vulkan as B;
    use burn::nn::{RotaryEncoding, RotaryEncodingConfig};
    use burn::tensor::DType;
    use burn::tensor::Tensor;

    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    let rope: RotaryEncoding<B> = RotaryEncodingConfig::new(max_seq_len, head_dim).init(&device);

    // freq_complex[0, 0..4, 0..2] should be [cos=1,sin=0] at position 0 for all dims.
    let freq0 = rope
        .freq_complex
        .clone()
        .slice([0..1, 0..4, 0..2])
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()?;
    eprintln!("freq_complex[0, 0..4, 0..2] = {freq0:?}");

    // Apply to an all-zero tensor: output must stay zeros (no NaNs).
    let x = Tensor::<B, 4>::zeros([1, 2, seq_len, head_dim], &device);
    let y = rope
        .apply(x, 0)
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()?;
    let (best_i, best_v, nan) = argmax_f32(&y, None);
    let min = y
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::INFINITY, f32::min);
    let max = y
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f32::NEG_INFINITY, f32::max);

    println!("rope.apply(zeros): nan={nan}, min={min}, max={max}, argmax=({best_i},{best_v})");

    // Now try non-zero inputs to catch dtype-specific issues.
    let n = 2 * seq_len * head_dim;
    let mut data = vec![0f32; n];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32) * 0.01;
    }
    let x_f32 = Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(data.clone(), [1, 2, seq_len, head_dim]),
        &device,
    );
    let y = rope
        .apply(x_f32, 0)
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()?;
    let (nan, min, max) = nan_min_max(&y);
    println!("rope.apply(f32 ramp): nan={nan}, min={min}, max={max}");

    let x_f16 = Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(data.clone(), [1, 2, seq_len, head_dim]),
        &device,
    )
    .cast(DType::F16);
    let y = rope
        .apply(x_f16, 0)
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()?;
    let (nan, min, max) = nan_min_max(&y);
    println!("rope.apply(f16 ramp): nan={nan}, min={min}, max={max}");

    let x_bf16 = Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(data, [1, 2, seq_len, head_dim]),
        &device,
    )
    .cast(DType::BF16);
    let y = rope
        .apply(x_bf16, 0)
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()?;
    let (nan, min, max) = nan_min_max(&y);
    println!("rope.apply(bf16 ramp): nan={nan}, min={min}, max={max}");
    Ok(())
}

fn cmd_inspect(weights: &PathBuf, take: usize) -> anyhow::Result<()> {
    let mut store = SafetensorsStore::from_file(weights)
        .match_all()
        .validate(false);

    let mut keys = store.keys().context("failed to list tensors")?;
    keys.sort();

    println!("tensors: {}", keys.len());
    let iter: Box<dyn Iterator<Item = String>> = if take == 0 {
        Box::new(keys.into_iter())
    } else {
        Box::new(keys.into_iter().take(take))
    };

    for key in iter {
        let snap = store
            .get_snapshot(&key)
            .with_context(|| format!("failed to fetch snapshot for '{key}'"))?
            .context("snapshot missing")?;
        println!("{key}\t{:?}\t{:?}", snap.shape, snap.dtype);
    }

    Ok(())
}

fn cmd_generate_text(opts: &GenerateTextOpts) -> anyhow::Result<()> {
    match opts.backend {
        BackendKind::Vulkan => cmd_generate_text_vulkan(opts),
        BackendKind::Ndarray => cmd_generate_text_ndarray(opts),
    }
}

fn cmd_generate_text_vulkan(opts: &GenerateTextOpts) -> anyhow::Result<()> {
    use burn::backend::Vulkan as B;
    use burn::tensor::DType;
    use burn::tensor::Int;
    use burn::tensor::Tensor;
    use model::deepseek_v2::{DeepseekV2Config, DeepseekV2ForCausalLM, KvCache};

    let device = burn::backend::wgpu::WgpuDevice::default();
    // Explicitly select the Vulkan graphics API.
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    // Build the model skeleton (params are lazily initialized).
    let default_cfg = DeepseekV2Config::default();
    let config = DeepseekV2Config {
        num_hidden_layers: opts.num_hidden_layers,
        first_k_dense_replace: default_cfg
            .first_k_dense_replace
            .min(opts.num_hidden_layers),
        kv_cache_dtype: match opts.kv_cache {
            KvCacheDtype::F32 => DType::F32,
            KvCacheDtype::F16 => DType::F16,
        },
        ..default_cfg
    };
    let mut model = DeepseekV2ForCausalLM::<B>::new(&config, &device);

    // Load weights from HF safetensors (PyTorch layout).
    use store_adapters::{CastDTypeAdapter, ChainAdapter};
    let mut store = if opts.cast_f16 {
        SafetensorsStore::from_file(&opts.weights).with_from_adapter(ChainAdapter::new(
            PyTorchToBurnAdapter,
            CastDTypeAdapter::to_f16(),
        ))
    } else {
        SafetensorsStore::from_file(&opts.weights).with_from_adapter(PyTorchToBurnAdapter)
    }
    .skip_enum_variants(true)
    .allow_partial(true)
    // Avoid building snapshots for the vision tower (huge) when doing text-only inference.
    .with_regex(r"^model\.embed_tokens\.")
    .with_regex(r"^model\.layers\.")
    .with_regex(r"^model\.norm\.")
    .with_regex(r"^lm_head\.")
    .validate(false);

    let apply = model
        .load_from(&mut store)
        .context("failed to load model weights")?;
    eprintln!(
        "loaded: applied={}, missing={}, skipped={}, errors={}",
        apply.applied.len(),
        apply.missing.len(),
        apply.skipped.len(),
        apply.errors.len()
    );
    drop(store);
    if opts.trim_memory {
        trim_memory_after_weights_load(&opts.weights);
    }

    // Tokenizer.
    let tokenizer = Tokenizer::from_file(&opts.tokenizer)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to load tokenizer.json")?;
    let enc = tokenizer
        .encode(opts.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("tokenize failed")?;

    let mut input_ids: Vec<i64> = Vec::with_capacity(enc.len() + 1);
    input_ids.push(0); // BOS
    input_ids.extend(enc.get_ids().iter().map(|&v| v as i64));
    let prompt_len = input_ids.len();

    let mut caches: Vec<Option<KvCache<B>>> = Vec::new();

    // Initial forward on the full prompt.
    let mut logits = {
        let data = burn::tensor::TensorData::new(input_ids.clone(), [1, prompt_len]);
        let ids = Tensor::<B, 2, Int>::from_data(data, &device);

        if std::env::var("DEEPSEEK_DEBUG_LAYER0").is_ok() && config.num_hidden_layers >= 1 {
            let mut caches_dbg: Vec<Option<KvCache<B>>> = Vec::new();
            let mut hidden = model.model.embed_tokens.forward(ids.clone());
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: embed nan={nan} min={min} max={max}");

            caches_dbg.resize_with(model.model.layers.len(), || None);
            let layer0 = &model.model.layers[0];

            let residual = hidden.clone();
            hidden = layer0.input_layernorm.forward(hidden);
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: after input_layernorm nan={nan} min={min} max={max}");

            hidden = layer0.self_attn.forward(hidden, &mut caches_dbg[0]);
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: self_attn out nan={nan} min={min} max={max}");

            hidden = residual + hidden;
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: after attn residual nan={nan} min={min} max={max}");

            let residual = hidden.clone();
            hidden = layer0.post_attention_layernorm.forward(hidden);
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: after post_attention_layernorm nan={nan} min={min} max={max}");

            hidden = layer0.mlp.forward(hidden);
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: mlp out nan={nan} min={min} max={max}");

            hidden = residual + hidden;
            let (nan, min, max) = nan_min_max(
                &hidden
                    .clone()
                    .cast(DType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
            eprintln!("debug: after mlp residual nan={nan} min={min} max={max}");
        }

        model.forward(ids, &mut caches)
    };

    if std::env::var("DEEPSEEK_DEBUG_TOPK").is_ok() {
        let [_, seq, vocab] = logits.dims();
        let last = logits
            .clone()
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .context("failed to read logits")?;
        let best = topk_f32(&last, 10);
        eprintln!(
            "debug: step0 top10 ids={:?}",
            best.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        );
        let mut toks = Vec::new();
        for (i, _) in best.iter() {
            let s = tokenizer
                .decode(&[*i as u32], false)
                .unwrap_or_else(|_| "<decode_err>".to_string());
            toks.push(s);
        }
        eprintln!("debug: step0 top10 tok={toks:?}");
        eprintln!(
            "debug: step0 top10 logit={:?}",
            best.iter().map(|(_, v)| *v).collect::<Vec<_>>()
        );
    }

    // Greedy decode loop.
    for _ in 0..opts.max_new_tokens {
        let [_, seq, vocab] = logits.dims();
        let last_t = logits.clone().slice([0..1, (seq - 1)..seq, 0..vocab]);
        let next_id = {
            let idx = last_t.argmax(2);
            let idx = idx
                .into_data()
                // On Vulkan/WGPU, Burn uses i32 for Int tensors.
                .to_vec::<i32>()
                .context("failed to read argmax id")?;
            idx[0] as i64
        };

        input_ids.push(next_id);
        if next_id == opts.eos_token_id {
            break;
        }

        // One-step forward with KV cache.
        let data = burn::tensor::TensorData::new(vec![next_id], [1, 1]);
        let ids = Tensor::<B, 2, Int>::from_data(data, &device);
        logits = model.forward(ids, &mut caches);
    }

    let gen_ids_u32: Vec<u32> = input_ids[prompt_len..]
        .iter()
        .filter_map(|&v| u32::try_from(v).ok())
        .collect();
    let out = tokenizer
        .decode(&gen_ids_u32, false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("decode failed")?;
    // Match HF inference: strip the trailing EOS marker if present, then trim whitespace.
    let stop_str = "<｜end▁of▁sentence｜>";
    let out = out
        .strip_suffix(stop_str)
        .unwrap_or(&out)
        .trim()
        .to_string();

    println!("{out}");
    Ok(())
}

fn cmd_generate_text_ndarray(opts: &GenerateTextOpts) -> anyhow::Result<()> {
    use burn::tensor::DType;
    use burn::tensor::Int;
    use burn::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use model::deepseek_v2::{DeepseekV2Config, DeepseekV2ForCausalLM, KvCache};
    use store_adapters::{CastDTypeAdapter, ChainAdapter};

    type B = NdArray<f32, i32>;
    let device = NdArrayDevice::Cpu;

    anyhow::ensure!(
        opts.kv_cache == KvCacheDtype::F32,
        "ndarray backend only supports --kv-cache f32"
    );

    let default_cfg = DeepseekV2Config::default();
    let config = DeepseekV2Config {
        num_hidden_layers: opts.num_hidden_layers,
        first_k_dense_replace: default_cfg
            .first_k_dense_replace
            .min(opts.num_hidden_layers),
        kv_cache_dtype: DType::F32,
        ..default_cfg
    };
    let mut model = DeepseekV2ForCausalLM::<B>::new(&config, &device);

    // NdArray doesn't support BF16/F16 weights; cast to F32 while loading.
    let adapter = ChainAdapter::new(PyTorchToBurnAdapter, CastDTypeAdapter::to_f32());
    let mut store = SafetensorsStore::from_file(&opts.weights)
        .with_from_adapter(adapter)
        .skip_enum_variants(true)
        .allow_partial(true)
        .with_regex(r"^model\.embed_tokens\.")
        .with_regex(r"^model\.layers\.")
        .with_regex(r"^model\.norm\.")
        .with_regex(r"^lm_head\.")
        .validate(false);

    let apply = model
        .load_from(&mut store)
        .context("failed to load model weights")?;
    eprintln!(
        "loaded: applied={}, missing={}, skipped={}, errors={}",
        apply.applied.len(),
        apply.missing.len(),
        apply.skipped.len(),
        apply.errors.len()
    );
    drop(store);
    if opts.trim_memory {
        trim_memory_after_weights_load(&opts.weights);
    }

    let tokenizer = Tokenizer::from_file(&opts.tokenizer)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to load tokenizer.json")?;
    let enc = tokenizer
        .encode(opts.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("tokenize failed")?;

    let mut input_ids: Vec<i64> = Vec::with_capacity(enc.len() + 1);
    input_ids.push(0); // BOS
    input_ids.extend(enc.get_ids().iter().map(|&v| v as i64));
    let prompt_len = input_ids.len();

    let mut caches: Vec<Option<KvCache<B>>> = Vec::new();

    let mut logits = {
        let data = burn::tensor::TensorData::new(input_ids.clone(), [1, prompt_len]);
        let ids = Tensor::<B, 2, Int>::from_data(data, &device);
        model.forward(ids, &mut caches)
    };

    if std::env::var("DEEPSEEK_DEBUG_TOPK").is_ok() {
        let [_, seq, vocab] = logits.dims();
        let last = logits
            .clone()
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .context("failed to read logits")?;
        let best = topk_f32(&last, 10);
        eprintln!(
            "debug: step0 top10 ids={:?}",
            best.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        );
        let mut toks = Vec::new();
        for (i, _) in best.iter() {
            let s = tokenizer
                .decode(&[*i as u32], false)
                .unwrap_or_else(|_| "<decode_err>".to_string());
            toks.push(s);
        }
        eprintln!("debug: step0 top10 tok={toks:?}");
        eprintln!(
            "debug: step0 top10 logit={:?}",
            best.iter().map(|(_, v)| *v).collect::<Vec<_>>()
        );
    }

    for _ in 0..opts.max_new_tokens {
        let [_, seq, vocab] = logits.dims();
        let last = logits
            .clone()
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .context("failed to read logits")?;
        let (best_i, _best_v, nan) = argmax_f32(&last, None);
        if nan > 0 {
            eprintln!("warning: argmax saw {nan} NaNs in logits");
        }
        let next_id = best_i as i64;

        input_ids.push(next_id);
        if next_id == opts.eos_token_id {
            break;
        }

        let data = burn::tensor::TensorData::new(vec![next_id], [1, 1]);
        let ids = Tensor::<B, 2, Int>::from_data(data, &device);
        logits = model.forward(ids, &mut caches);
    }

    let gen_ids_u32: Vec<u32> = input_ids[prompt_len..]
        .iter()
        .filter_map(|&v| u32::try_from(v).ok())
        .collect();
    let out = tokenizer
        .decode(&gen_ids_u32, false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("decode failed")?;
    let stop_str = "<｜end▁of▁sentence｜>";
    let out = out
        .strip_suffix(stop_str)
        .unwrap_or(&out)
        .trim()
        .to_string();

    println!("{out}");
    Ok(())
}

fn tokenize_with_image(
    tokenizer: &Tokenizer,
    prompt: &str,
    image_token_id: i64,
    base_size: u32,
    crop_ratio: (u32, u32),
    crop_image_size: u32,
) -> anyhow::Result<(Vec<i64>, Vec<bool>)> {
    let parts: Vec<&str> = prompt.split("<image>").collect();
    anyhow::ensure!(
        parts.len() == 2,
        "prompt must contain exactly one '<image>' placeholder"
    );

    let mut ids: Vec<i64> = Vec::new();
    let mut mask: Vec<bool> = Vec::new();

    // BOS
    ids.push(0);
    mask.push(false);

    for (pi, part) in parts.iter().enumerate() {
        let part = *part;
        if !part.is_empty() {
            let enc = tokenizer
                .encode(part, false)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .context("tokenize failed")?;
            ids.extend(enc.get_ids().iter().map(|&v| v as i64));
            mask.extend(std::iter::repeat_n(false, enc.len()));
        }

        // After the first part, insert image tokens.
        if pi == 0 {
            // DeepSeek-OCR-2 uses patch_size=16 and downsample_ratio=4 for the vision token grid.
            let patch_size = 16u32;
            let downsample_ratio = 4u32;
            let num_queries_base = (base_size / patch_size).div_ceil(downsample_ratio);
            let mut n_img_tokens = (num_queries_base * num_queries_base + 1) as usize; // +1 view_seperator token

            if crop_ratio.0 > 1 || crop_ratio.1 > 1 {
                let num_queries = (crop_image_size / patch_size).div_ceil(downsample_ratio);
                let local_w = num_queries * crop_ratio.0;
                let local_h = num_queries * crop_ratio.1;
                n_img_tokens += (local_w * local_h) as usize;
            }

            ids.extend(std::iter::repeat_n(image_token_id, n_img_tokens));
            mask.extend(std::iter::repeat_n(true, n_img_tokens));
        }
    }

    Ok((ids, mask))
}

fn find_closest_aspect_ratio(
    aspect_ratio: f32,
    target_ratios: &[(u32, u32)],
    width: u32,
    height: u32,
    image_size: u32,
) -> (u32, u32) {
    let mut best_ratio_diff = f32::INFINITY;
    let mut best_ratio = (1u32, 1u32);
    let area = (width as f32) * (height as f32);

    for &(rw, rh) in target_ratios.iter() {
        let target_ar = (rw as f32) / (rh as f32);
        let diff = (aspect_ratio - target_ar).abs();
        if diff < best_ratio_diff {
            best_ratio_diff = diff;
            best_ratio = (rw, rh);
        } else if (diff - best_ratio_diff).abs() < f32::EPSILON {
            // Tie-breaker: prefer higher effective resolution.
            let thresh =
                0.5 * (image_size as f32) * (image_size as f32) * (rw as f32) * (rh as f32);
            if area > thresh {
                best_ratio = (rw, rh);
            }
        }
    }

    best_ratio
}

fn dynamic_preprocess(
    image: &RgbImage,
    min_num: u32,
    max_num: u32,
    image_size: u32,
) -> (Vec<RgbImage>, (u32, u32)) {
    let (orig_w, orig_h) = image.dimensions();
    let aspect_ratio = (orig_w as f32) / (orig_h as f32);

    // Enumerate candidate tilings (w_tiles, h_tiles).
    let mut ratios: Vec<(u32, u32)> = Vec::new();
    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                let blocks = i * j;
                if blocks <= max_num && blocks >= min_num {
                    ratios.push((i, j));
                }
            }
        }
    }
    ratios.sort_by_key(|(i, j)| i * j);
    ratios.dedup();

    let target_ratio = find_closest_aspect_ratio(aspect_ratio, &ratios, orig_w, orig_h, image_size);
    let (tiles_w, tiles_h) = target_ratio;

    let target_w = image_size * tiles_w;
    let target_h = image_size * tiles_h;
    let resized = image::imageops::resize(image, target_w, target_h, FilterType::CatmullRom);

    let mut out = Vec::with_capacity((tiles_w * tiles_h) as usize);
    for i in 0..(tiles_w * tiles_h) {
        let x = (i % tiles_w) * image_size;
        let y = (i / tiles_w) * image_size;
        let crop = image::imageops::crop_imm(&resized, x, y, image_size, image_size).to_image();
        out.push(crop);
    }

    (out, target_ratio)
}

fn image_to_tensor_nchw<B: burn::tensor::backend::Backend>(
    img: &RgbImage,
    device: &B::Device,
) -> burn::tensor::Tensor<B, 4> {
    let (w, h) = img.dimensions();
    let h_usize = h as usize;
    let w_usize = w as usize;
    let hw = h_usize * w_usize;

    // Normalize to [-1, 1] using mean=0.5/std=0.5.
    let mut data = vec![0f32; 3 * hw];
    for y in 0..h_usize {
        for x in 0..w_usize {
            let p = img.get_pixel(x as u32, y as u32).0;
            let idx = y * w_usize + x;
            let r = (p[0] as f32 / 255.0) * 2.0 - 1.0;
            let g = (p[1] as f32 / 255.0) * 2.0 - 1.0;
            let b = (p[2] as f32 / 255.0) * 2.0 - 1.0;
            data[idx] = r;
            data[hw + idx] = g;
            data[2 * hw + idx] = b;
        }
    }

    let tensor_data = burn::tensor::TensorData::new(data, [1, 3, h_usize, w_usize]);
    burn::tensor::Tensor::<B, 4>::from_data(tensor_data, device)
}

fn cmd_generate_ocr(opts: &GenerateOcrOpts) -> anyhow::Result<()> {
    match opts.backend {
        BackendKind::Vulkan => cmd_generate_ocr_vulkan(opts),
        BackendKind::Ndarray => cmd_generate_ocr_ndarray(opts),
    }
}

fn cmd_generate_ocr_vulkan(opts: &GenerateOcrOpts) -> anyhow::Result<()> {
    use burn::backend::Vulkan as B;
    use burn::tensor::DType;
    use burn::tensor::Int;
    use burn::tensor::Tensor;
    use model::deepseek_ocr2::DeepseekOcr2ForCausalLM;
    use model::deepseek_v2::{DeepseekV2Config, KvCache};
    use store_adapters::{ChainAdapter, SelectiveCastDTypeAdapter};

    // GPU device (wgpu + Vulkan).
    let device = burn::backend::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    // Build the model skeleton.
    let config = DeepseekV2Config {
        kv_cache_dtype: match opts.kv_cache {
            KvCacheDtype::F32 => DType::F32,
            KvCacheDtype::F16 => DType::F16,
        },
        ..Default::default()
    };
    let mut model = DeepseekOcr2ForCausalLM::<B>::new(&config, &device);

    // Load weights (PyTorch -> Burn).
    //
    // On Vulkan/WebGPU, some SAM ops are unstable in F16. Keep SAM in F32 while keeping the
    // rest of the model in F16 to save memory.
    let adapter = ChainAdapter::new(
        PyTorchToBurnAdapter,
        // DeepSeek-OCR-2 weights are BF16 on HF, but BF16 is still flaky on some Vulkan/WebGPU
        // drivers. Default to F16 for backend stability and keep the vision tower in F32.
        SelectiveCastDTypeAdapter::new(DType::F16)
            .with_prefix("model.sam_model", DType::F32)
            .with_prefix("model.qwen2_model", DType::F32)
            .with_prefix("model.projector", DType::F32)
            .with_prefix("model.view_seperator", DType::F32),
    );
    let mut store = SafetensorsStore::from_file(&opts.weights)
        .with_from_adapter(adapter)
        .skip_enum_variants(true)
        .validate(false);
    let apply = model
        .load_from(&mut store)
        .context("failed to load model weights")?;
    eprintln!(
        "loaded: applied={}, missing={}, skipped={}, errors={}",
        apply.applied.len(),
        apply.missing.len(),
        apply.skipped.len(),
        apply.errors.len()
    );
    drop(store);
    if opts.trim_memory {
        trim_memory_after_weights_load(&opts.weights);
    }

    // Tokenizer + prompt tokenization (one image placeholder).
    let tokenizer = Tokenizer::from_file(&opts.tokenizer)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to load tokenizer.json")?;

    // Load the input image once.
    let orig = image::open(&opts.image)
        .with_context(|| format!("failed to open image {:?}", opts.image))?
        .to_rgb8();

    // Apply rotation (manual or heuristic auto-rotate).
    let rotate_used = if opts.rotate != Rotate::R0 {
        opts.rotate
    } else if opts.auto_rotate {
        auto_rotate_choice(&orig)
    } else {
        Rotate::R0
    };
    if std::env::var("DEEPSEEK_DEBUG_OCR").is_ok() || std::env::var("DEEPSEEK_DEBUG_TOPK").is_ok() {
        eprintln!("debug: rotate_used={rotate_used:?}");
    }
    let orig = rotate_apply(orig, rotate_used);

    let (orig_w, orig_h) = orig.dimensions();

    let (patches, crop_ratio) =
        if opts.crop_mode && (orig_w > opts.crop_image_size || orig_h > opts.crop_image_size) {
            let (crops, ratio) = dynamic_preprocess(&orig, 2, 6, opts.crop_image_size);
            (Some(crops), ratio)
        } else {
            (None, (1, 1))
        };

    let (mut input_ids_vec, images_seq_mask) = tokenize_with_image(
        &tokenizer,
        DEFAULT_OCR_PROMPT,
        opts.image_token_id,
        opts.image_size,
        crop_ratio,
        opts.crop_image_size,
    )?;
    let prompt_len = input_ids_vec.len();

    // Base (global) view: letterbox-pad to square.
    // HF reference uses mean=0.5 => pad color = int(0.5 * 255) = 127.
    let img_base = pad_to_square_rgb(&orig, opts.image_size, 127)?;
    // Keep vision input in F32 (SAM weights are loaded as F32 for backend stability).
    let image_base = image_to_tensor_nchw::<B>(&img_base, &device);

    // Local crops (optional): stack into a single [P, 3, crop_image_size, crop_image_size] tensor.
    let patches = patches.map(|crops| {
        let mut tensors = Vec::with_capacity(crops.len());
        for crop in crops.iter() {
            tensors.push(image_to_tensor_nchw::<B>(crop, &device));
        }
        Tensor::cat(tensors, 0)
    });

    // Build input ids tensor.
    let data = burn::tensor::TensorData::new(input_ids_vec.clone(), [1, prompt_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, &device);

    // Replace placeholder embeddings with vision tokens for the first forward.
    let inputs_embeds = model
        .model
        .build_inputs_embeds_with_image(input_ids, image_base, patches, &images_seq_mask)
        .context("failed to build multimodal embeddings")?;
    if std::env::var("DEEPSEEK_DEBUG_OCR").is_ok() {
        let [b, seq, hidden] = inputs_embeds.dims();
        let data = inputs_embeds
            .clone()
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .context("failed to read inputs_embeds")?;
        let (nan, min, max) = nan_min_max(&data);
        let mut sum = 0f64;
        let mut cnt = 0usize;
        for &v in data.iter() {
            if v.is_nan() {
                continue;
            }
            sum += v as f64;
            cnt += 1;
        }
        let mean = if cnt == 0 {
            f32::NAN
        } else {
            (sum / cnt as f64) as f32
        };
        let fp: Vec<f32> = data.iter().copied().take(16).collect();
        eprintln!("debug: inputs_embeds nan={nan} min={min} max={max} mean={mean}");
        eprintln!("debug: inputs_embeds fingerprint={fp:?}");

        if b == 1 && hidden > 0 {
            let fp_pos = |pos: usize| -> Vec<f32> {
                let off = pos * hidden;
                data.iter().skip(off).copied().take(16).collect()
            };
            if seq >= 2 {
                eprintln!("debug: inputs_embeds[pos0]={:?}", fp_pos(0));
                eprintln!("debug: inputs_embeds[pos1]={:?}", fp_pos(1));
                eprintln!("debug: inputs_embeds[pos_last]={:?}", fp_pos(seq - 1));
                // For the default crop settings (2 tiles => 288 local tokens, 256 global tokens, 1 sep),
                // these positions are useful to compare against the HF reference dump.
                if seq > 289 {
                    eprintln!("debug: inputs_embeds[pos289]={:?}", fp_pos(289));
                }
                if seq > 545 {
                    eprintln!("debug: inputs_embeds[pos545]={:?}", fp_pos(545));
                }
            }
        }
    }

    let mut caches: Vec<Option<KvCache<B>>> = Vec::new();
    let mut logits = model.forward_embeds(inputs_embeds, &mut caches);

    if std::env::var("DEEPSEEK_DEBUG_TOPK").is_ok() {
        let [_, seq, vocab] = logits.dims();
        let last = logits
            .clone()
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .context("failed to read logits")?;
        let best = topk_f32(&last, 10);
        eprintln!(
            "debug: step0 top10 ids={:?}",
            best.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        );
        let mut toks = Vec::new();
        for (i, _) in best.iter() {
            let s = tokenizer
                .decode(&[*i as u32], false)
                .unwrap_or_else(|_| "<decode_err>".to_string());
            toks.push(s);
        }
        eprintln!("debug: step0 top10 tok={toks:?}");
        eprintln!(
            "debug: step0 top10 logit={:?}",
            best.iter().map(|(_, v)| *v).collect::<Vec<_>>()
        );
    }

    // Greedy decode loop.
    //
    // Performance note: transferring the whole vocab logits to CPU each step is very slow
    // (vocab is large). Prefer GPU argmax and only fall back to CPU when constraints (e.g.
    // no-repeat-ngram) force it.
    for step in 0..opts.max_new_tokens {
        let [_, seq, vocab] = logits.dims();
        let last_t = logits.clone().slice([0..1, (seq - 1)..seq, 0..vocab]);

        let banned_ids = no_repeat_ngram_banned_ids(&input_ids_vec, opts.no_repeat_ngram_size);

        // Fast path: GPU argmax.
        let mut next_id: i64 = {
            let idx = last_t.clone().argmax(2);
            let idx = idx
                .into_data()
                // On Vulkan/WGPU, Burn uses i32 for Int tensors.
                .to_vec::<i32>()
                .context("failed to read argmax id")?;
            idx[0] as i64
        };

        // If constraints are active and we hit a banned id, fall back to CPU scan.
        if !banned_ids.is_empty() && banned_ids.iter().any(|&b| b as i64 == next_id) {
            let last = last_t
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .context("failed to read logits")?;
            let mut mask = vec![false; vocab];
            for id in banned_ids {
                if id < vocab {
                    mask[id] = true;
                }
            }
            let (best_i, _best_v, nan) = argmax_f32(&last, Some(&mask));
            if nan > 0 {
                eprintln!("warning: argmax saw {nan} NaNs in logits");
            }
            next_id = best_i as i64;
        }

        input_ids_vec.push(next_id);
        if std::env::var("DEEPSEEK_DEBUG_TOKENS").is_ok() {
            let piece = tokenizer
                .decode(&[next_id as u32], false)
                .unwrap_or_else(|_| "<decode_err>".to_string());
            eprintln!("debug: step{step} next_id={next_id} tok={piece:?}");
        }
        if next_id == opts.eos_token_id {
            break;
        }

        let data = burn::tensor::TensorData::new(vec![next_id], [1, 1]);
        let ids = Tensor::<B, 2, Int>::from_data(data, &device);
        logits = model.forward(ids, &mut caches);
    }

    let gen_ids_u32: Vec<u32> = input_ids_vec[prompt_len..]
        .iter()
        .filter_map(|&v| u32::try_from(v).ok())
        .collect();
    let out = tokenizer
        .decode(&gen_ids_u32, false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("decode failed")?;
    // Match HF inference: strip the trailing EOS marker if present, then trim whitespace.
    let stop_str = "<｜end▁of▁sentence｜>";
    let out = out
        .strip_suffix(stop_str)
        .unwrap_or(&out)
        .trim()
        .to_string();

    println!("{out}");
    Ok(())
}

fn cmd_generate_ocr_ndarray(opts: &GenerateOcrOpts) -> anyhow::Result<()> {
    use burn::tensor::DType;
    use burn::tensor::Int;
    use burn::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use model::deepseek_ocr2::DeepseekOcr2ForCausalLM;
    use model::deepseek_v2::{DeepseekV2Config, KvCache};
    use store_adapters::{CastDTypeAdapter, ChainAdapter};

    type B = NdArray<f32, i32>;
    let device = NdArrayDevice::Cpu;

    anyhow::ensure!(
        opts.kv_cache == KvCacheDtype::F32,
        "ndarray backend only supports --kv-cache f32"
    );

    let config = DeepseekV2Config {
        kv_cache_dtype: DType::F32,
        ..Default::default()
    };
    let mut model = DeepseekOcr2ForCausalLM::<B>::new(&config, &device);

    // NdArray doesn't support BF16/F16 weights; cast to F32 while loading.
    let adapter = ChainAdapter::new(PyTorchToBurnAdapter, CastDTypeAdapter::to_f32());
    let mut store = SafetensorsStore::from_file(&opts.weights)
        .with_from_adapter(adapter)
        .skip_enum_variants(true)
        .validate(false);
    let apply = model
        .load_from(&mut store)
        .context("failed to load model weights")?;
    eprintln!(
        "loaded: applied={}, missing={}, skipped={}, errors={}",
        apply.applied.len(),
        apply.missing.len(),
        apply.skipped.len(),
        apply.errors.len()
    );
    drop(store);
    if opts.trim_memory {
        trim_memory_after_weights_load(&opts.weights);
    }

    let tokenizer = Tokenizer::from_file(&opts.tokenizer)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("failed to load tokenizer.json")?;

    let orig = image::open(&opts.image)
        .with_context(|| format!("failed to open image {:?}", opts.image))?
        .to_rgb8();

    let rotate_used = if opts.rotate != Rotate::R0 {
        opts.rotate
    } else if opts.auto_rotate {
        auto_rotate_choice(&orig)
    } else {
        Rotate::R0
    };
    if std::env::var("DEEPSEEK_DEBUG_OCR").is_ok() || std::env::var("DEEPSEEK_DEBUG_TOPK").is_ok() {
        eprintln!("debug: rotate_used={rotate_used:?}");
    }
    let orig = rotate_apply(orig, rotate_used);

    let (orig_w, orig_h) = orig.dimensions();

    let (patches, crop_ratio) =
        if opts.crop_mode && (orig_w > opts.crop_image_size || orig_h > opts.crop_image_size) {
            let (crops, ratio) = dynamic_preprocess(&orig, 2, 6, opts.crop_image_size);
            (Some(crops), ratio)
        } else {
            (None, (1, 1))
        };

    let (mut input_ids_vec, images_seq_mask) = tokenize_with_image(
        &tokenizer,
        DEFAULT_OCR_PROMPT,
        opts.image_token_id,
        opts.image_size,
        crop_ratio,
        opts.crop_image_size,
    )?;
    let prompt_len = input_ids_vec.len();

    // HF reference uses mean=0.5 => pad color = int(0.5 * 255) = 127.
    let img_base = pad_to_square_rgb(&orig, opts.image_size, 127)?;
    let image_base = image_to_tensor_nchw::<B>(&img_base, &device);

    let patches = patches.map(|crops| {
        let mut tensors = Vec::with_capacity(crops.len());
        for crop in crops.iter() {
            tensors.push(image_to_tensor_nchw::<B>(crop, &device));
        }
        Tensor::cat(tensors, 0)
    });

    let data = burn::tensor::TensorData::new(input_ids_vec.clone(), [1, prompt_len]);
    let input_ids = Tensor::<B, 2, Int>::from_data(data, &device);

    let inputs_embeds = model
        .model
        .build_inputs_embeds_with_image(input_ids, image_base, patches, &images_seq_mask)
        .context("failed to build multimodal embeddings")?;

    let mut caches: Vec<Option<KvCache<B>>> = Vec::new();
    let mut logits = model.forward_embeds(inputs_embeds, &mut caches);

    if std::env::var("DEEPSEEK_DEBUG_TOPK").is_ok() {
        let [_, seq, vocab] = logits.dims();
        let last = logits
            .clone()
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .context("failed to read logits")?;
        let best = topk_f32(&last, 10);
        eprintln!(
            "debug: step0 top10 ids={:?}",
            best.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        );
        let mut toks = Vec::new();
        for (i, _) in best.iter() {
            let s = tokenizer
                .decode(&[*i as u32], false)
                .unwrap_or_else(|_| "<decode_err>".to_string());
            toks.push(s);
        }
        eprintln!("debug: step0 top10 tok={toks:?}");
        eprintln!(
            "debug: step0 top10 logit={:?}",
            best.iter().map(|(_, v)| *v).collect::<Vec<_>>()
        );
    }

    for _ in 0..opts.max_new_tokens {
        let [_, seq, vocab] = logits.dims();
        let last = logits
            .clone()
            .slice([0..1, (seq - 1)..seq, 0..vocab])
            .cast(DType::F32);
        let last = last
            .into_data()
            .to_vec::<f32>()
            .context("failed to read logits")?;
        let banned = no_repeat_ngram_banned_ids(&input_ids_vec, opts.no_repeat_ngram_size);
        let banned = if banned.is_empty() {
            None
        } else {
            let mut mask = vec![false; vocab];
            for id in banned {
                if id < vocab {
                    mask[id] = true;
                }
            }
            Some(mask)
        };
        let (best_i, _best_v, nan) = argmax_f32(&last, banned.as_deref());
        if nan > 0 {
            eprintln!("warning: argmax saw {nan} NaNs in logits");
        }
        let next_id = best_i as i64;

        input_ids_vec.push(next_id);
        if next_id == opts.eos_token_id {
            break;
        }

        let data = burn::tensor::TensorData::new(vec![next_id], [1, 1]);
        let ids = Tensor::<B, 2, Int>::from_data(data, &device);
        logits = model.forward(ids, &mut caches);
    }

    let gen_ids_u32: Vec<u32> = input_ids_vec[prompt_len..]
        .iter()
        .filter_map(|&v| u32::try_from(v).ok())
        .collect();
    let out = tokenizer
        .decode(&gen_ids_u32, false)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("decode failed")?;
    let stop_str = "<｜end▁of▁sentence｜>";
    let out = out
        .strip_suffix(stop_str)
        .unwrap_or(&out)
        .trim()
        .to_string();

    println!("{out}");
    Ok(())
}
