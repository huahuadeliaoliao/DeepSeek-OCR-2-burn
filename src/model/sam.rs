//! SAM ViT-B image encoder used by DeepSeek-OCR-2 (DeepEncoderV2).
//!
//! This is a direct port of `deepencoderv2.py` (ImageEncoderViT) to Burn.
//! For now, it focuses on inference for the base 1024x1024 view (no pos/rel interpolation).

use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode};
use burn::tensor::{DType, Int, Tensor};

/// Create a resize grid for `grid_sample_2d` that matches PyTorch
/// `F.interpolate(..., align_corners=False)`.
///
/// With `align_corners=False`, the sampling points are at the centers of the
/// output pixels, producing normalized coordinates in `[-1 + 1/out, 1 - 1/out]`.
fn resize_grid_align_corners_false<B: Backend>(
    batch: usize,
    out_h: usize,
    out_w: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let oh = out_h as f32;
    let ow = out_w as f32;

    let mut grid: Vec<f32> = Vec::with_capacity(batch * out_h * out_w * 2);
    for _ in 0..batch {
        for y in 0..out_h {
            let yf = 2.0 * ((y as f32) + 0.5) / oh - 1.0;
            for x in 0..out_w {
                let xf = 2.0 * ((x as f32) + 0.5) / ow - 1.0;
                // grid_sample expects [x, y] in normalized coordinates.
                grid.push(xf);
                grid.push(yf);
            }
        }
    }

    Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(grid, [batch, out_h, out_w, 2]),
        device,
    )
}

fn resize_nchw_align_corners_false<B: Backend>(
    x: Tensor<B, 4>,
    out_h: usize,
    out_w: usize,
    mode: InterpolateMode,
) -> Tensor<B, 4> {
    let device = x.device();
    let [b, _c, _h, _w] = x.dims();

    let grid = resize_grid_align_corners_false::<B>(b, out_h, out_w, &device);
    let options = GridSampleOptions::new(mode)
        .with_padding_mode(GridSamplePaddingMode::Zeros)
        .with_align_corners(false);

    x.grid_sample_2d(grid, options)
}

fn resize_nchw_align_corners_true<B: Backend>(
    x: Tensor<B, 4>,
    out_h: usize,
    out_w: usize,
    mode: InterpolateMode,
) -> Tensor<B, 4> {
    // Burn's `tensor::module::interpolate` currently uses `align_corners=true` mapping.
    // We expose it as an optional path mainly for debugging / HF alignment experiments.
    burn::tensor::module::interpolate(
        x,
        [out_h, out_w],
        burn::tensor::ops::InterpolateOptions::new(mode),
    )
}

fn nchw_to_nhwc<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // [B, C, H, W] -> [B, H, W, C]
    x.swap_dims(1, 3).swap_dims(1, 2)
}

fn nhwc_to_nchw<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    // [B, H, W, C] -> [B, C, H, W]
    x.swap_dims(1, 3).swap_dims(2, 3)
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub proj: Conv2d<B>,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(in_chans: usize, embed_dim: usize, device: &B::Device) -> Self {
        // kernel_size=stride=16, padding=valid, bias=true
        let proj = Conv2dConfig::new([in_chans, embed_dim], [16, 16])
            .with_stride([16, 16])
            .with_padding(burn::nn::PaddingConfig2d::Valid)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self { proj }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // [B,3,H,W] -> [B,H',W',C]
        let x = self.proj.forward(x);
        nchw_to_nhwc(x)
    }
}

#[derive(Module, Debug)]
pub struct MlpBlock<B: Backend> {
    pub lin1: Linear<B>,
    pub lin2: Linear<B>,
}

impl<B: Backend> MlpBlock<B> {
    pub fn new(embedding_dim: usize, mlp_dim: usize, device: &B::Device) -> Self {
        let lin1 = LinearConfig::new(embedding_dim, mlp_dim)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let lin2 = LinearConfig::new(mlp_dim, embedding_dim)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self { lin1, lin2 }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.lin2.forward(gelu(self.lin1.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm2d<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub bias: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> LayerNorm2d<B> {
    pub fn new(num_channels: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Initializer::Ones.init([num_channels], device);
        let bias = Initializer::Zeros.init([num_channels], device);
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // x: [B, C, H, W] normalize across C.
        let dtype = x.dtype();
        let x_f32 = x.clone().cast(burn::tensor::DType::F32);
        let mean = x_f32.clone().mean_dim(1);
        let var = (x_f32.clone() - mean.clone()).square().mean_dim(1);
        let y = (x_f32 - mean) / (var + self.eps).sqrt();
        let y = y.cast(dtype);

        // Apply affine.
        let [c] = self.weight.shape().dims();
        let w = self.weight.val().reshape([1, c, 1, 1]);
        let b = self.bias.val().reshape([1, c, 1, 1]);
        y * w + b
    }
}

#[derive(Module, Debug)]
pub enum NeckLayer<B: Backend> {
    Conv(Conv2d<B>),
    Norm(LayerNorm2d<B>),
}

impl<B: Backend> NeckLayer<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Conv(conv) => conv.forward(x),
            Self::Norm(norm) => norm.forward(x),
        }
    }
}

fn window_partition<B: Backend>(
    x: Tensor<B, 4>, // [B, H, W, C]
    window_size: usize,
) -> (Tensor<B, 4>, (usize, usize)) {
    let [b, h, w, c] = x.dims();
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let x = if pad_h > 0 || pad_w > 0 {
        let x_nchw = nhwc_to_nchw(x);
        // Burn's `Tensor::pad` currently trips a Fusion DTypeMismatch on some backends when the
        // tensor is F16. Use an explicit zero-pad with `slice_assign` to keep the dtype stable.
        let x_nchw = pad_nchw_zeros(x_nchw, pad_h, pad_w);
        nchw_to_nhwc(x_nchw)
    } else {
        x
    };

    let hp = h + pad_h;
    let wp = w + pad_w;

    let h_div = hp / window_size;
    let w_div = wp / window_size;
    let windows = x
        .reshape([b, h_div, window_size, w_div, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_div * w_div, window_size, window_size, c]);

    (windows, (hp, wp))
}

fn pad_nchw_zeros<B: Backend>(x: Tensor<B, 4>, pad_h: usize, pad_w: usize) -> Tensor<B, 4> {
    let device = x.device();
    let dtype = x.dtype();
    let [b, c, h, w] = x.dims();

    let mut out = Tensor::<B, 4>::zeros([b, c, h + pad_h, w + pad_w], &device).cast(dtype);
    out = out.slice_assign([0..b, 0..c, 0..h, 0..w], x);
    out
}

fn window_unpartition<B: Backend>(
    windows: Tensor<B, 4>, // [B * n_windows, window, window, C]
    window_size: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Tensor<B, 4> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let [n, _, _, c] = windows.dims();

    let h_div = hp / window_size;
    let w_div = wp / window_size;
    let windows_per_img = h_div * w_div;
    let b = n / windows_per_img;

    let x = windows
        .reshape([b, h_div, w_div, window_size, window_size, c])
        .swap_dims(2, 3)
        .reshape([b, hp, wp, c]);

    if hp > h || wp > w {
        x.slice([0..b, 0..h, 0..w, 0..c])
    } else {
        x
    }
}

fn get_rel_pos<B: Backend>(
    q_size: usize,
    k_size: usize,
    rel_pos: Tensor<B, 2>, // [L, head_dim]
    device: &B::Device,
) -> Tensor<B, 3> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;
    let [l, head_dim] = rel_pos.dims();
    let rel_pos = if l != max_rel_dist {
        // Resize along the "length" dimension, matching PyTorch
        // `F.interpolate(..., mode="linear", align_corners=False)`.
        //
        // Burn's `interpolate` currently behaves like `align_corners=True`, so we use `grid_sample`
        // with `align_corners=false` instead.
        let rel = rel_pos
            .swap_dims(0, 1) // [C, L]
            .reshape::<4, _>([1, head_dim, l, 1]);
        let rel =
            resize_nchw_align_corners_false::<B>(rel, max_rel_dist, 1, InterpolateMode::Bilinear);
        rel.reshape([head_dim, max_rel_dist]).swap_dims(0, 1)
    } else {
        rel_pos
    };

    // For the current OCR2 configs we mostly have q_size == k_size.
    assert_eq!(
        q_size, k_size,
        "q/k rel_pos scaling not implemented (q={q_size}, k={k_size})"
    );
    let offset = (k_size - 1) as i64;
    let mut idx = Vec::with_capacity(q_size * k_size);
    for qi in 0..q_size {
        for ki in 0..k_size {
            idx.push(qi as i64 - ki as i64 + offset);
        }
    }

    let idx = Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::new(idx, [q_size * k_size]),
        device,
    );
    rel_pos.select(0, idx).reshape([q_size, k_size, head_dim])
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub qkv: Linear<B>,
    pub proj: Linear<B>,
    pub rel_pos_h: Param<Tensor<B, 2>>,
    pub rel_pos_w: Param<Tensor<B, 2>>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub input_size: [usize; 2],
}

impl<B: Backend> Attention<B> {
    pub fn new(dim: usize, num_heads: usize, input_size: [usize; 2], device: &B::Device) -> Self {
        let head_dim = dim / num_heads;
        let qkv = LinearConfig::new(dim, dim * 3)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let proj = LinearConfig::new(dim, dim)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);

        // rel_pos_*: [2*size - 1, head_dim]
        let rel_pos_h = Initializer::Zeros.init([2 * input_size[0] - 1, head_dim], device);
        let rel_pos_w = Initializer::Zeros.init([2 * input_size[1] - 1, head_dim], device);

        Self {
            qkv,
            proj,
            rel_pos_h,
            rel_pos_w,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            input_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let device = x.device();
        let [b, h, w, dim] = x.dims();
        let l = h * w;

        // qkv: [B, H, W, 3*dim] -> [B, L, 3, heads, head_dim]
        let qkv = self
            .qkv
            .forward(x)
            .reshape([b, l, 3, self.num_heads, self.head_dim]);

        // Slice q/k/v and permute to [B, heads, L, head_dim]
        let q = qkv
            .clone()
            .slice([0..b, 0..l, 0..1, 0..self.num_heads, 0..self.head_dim])
            .squeeze_dim(2)
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .slice([0..b, 0..l, 1..2, 0..self.num_heads, 0..self.head_dim])
            .squeeze_dim(2)
            .swap_dims(1, 2);
        let v = qkv
            .slice([0..b, 0..l, 2..3, 0..self.num_heads, 0..self.head_dim])
            .squeeze_dim(2)
            .swap_dims(1, 2);

        // Compute decomposed rel-pos bias.
        let q_flat = q.clone().reshape([b * self.num_heads, l, self.head_dim]); // [B*heads, L, head_dim]
        let rh = get_rel_pos(h, h, self.rel_pos_h.val(), &device); // [H, H, head_dim]
        let rw = get_rel_pos(w, w, self.rel_pos_w.val(), &device); // [W, W, head_dim]

        let r_q = q_flat.reshape([b * self.num_heads, h, w, self.head_dim]); // [B*heads, H, W, head_dim]
        let r_q5 = r_q.clone().unsqueeze_dim::<5>(3); // [B*heads, H, W, 1, head_dim]

        let rh5 = rh.unsqueeze_dim::<4>(0).unsqueeze_dim::<5>(2); // [1, H, 1, H, head_dim]
        let rel_h: Tensor<B, 4> = (r_q5.clone() * rh5).sum_dims_squeeze::<4, _>(&[4]); // [B*heads, H, W, H]

        let rw5 = rw.unsqueeze_dim::<4>(0).unsqueeze_dim::<5>(1); // [1, 1, W, W, head_dim]
        let rel_w: Tensor<B, 4> = (r_q5 * rw5).sum_dims_squeeze::<4, _>(&[4]); // [B*heads, H, W, W]

        let rel_h = rel_h.unsqueeze_dim::<5>(4); // [B*heads, H, W, H, 1]
        let rel_w = rel_w.unsqueeze_dim::<5>(3); // [B*heads, H, W, 1, W]
        let attn_bias: Tensor<B, 4> = (rel_h + rel_w)
            .reshape([b * self.num_heads, l, h * w])
            .reshape([b, self.num_heads, l, l]);

        // Scores: [B, heads, L, L]
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(self.scale)
            .add(attn_bias);

        // Upcast softmax to F32 for numerical stability and backend correctness.
        let scores_dtype = scores.dtype();
        let weights = softmax(scores.cast(DType::F32), 3).cast(scores_dtype);
        let ctx = weights.matmul(v); // [B, heads, L, head_dim]

        let ctx = ctx
            .reshape([b, self.num_heads, h, w, self.head_dim])
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([b, h, w, dim]);

        self.proj.forward(ctx)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attn: Attention<B>,
    pub norm2: LayerNorm<B>,
    pub mlp: MlpBlock<B>,
    pub window_size: usize,
}

impl<B: Backend> Block<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        window_size: usize,
        input_size: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let norm1 = LayerNormConfig::new(dim).with_epsilon(1e-6).init(device);
        let norm2 = LayerNormConfig::new(dim).with_epsilon(1e-6).init(device);
        let input_size_attn = if window_size == 0 {
            input_size
        } else {
            [window_size, window_size]
        };
        let attn = Attention::new(dim, num_heads, input_size_attn, device);
        let mlp = MlpBlock::new(dim, (dim as f64 * mlp_ratio) as usize, device);
        Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let shortcut = x.clone();
        let mut x = self.norm1.forward(x);

        if self.window_size > 0 {
            let [_, h, w, _] = x.dims();
            let (windows, pad_hw) = window_partition(x, self.window_size);
            let windows = self.attn.forward(windows);
            x = window_unpartition(windows, self.window_size, pad_hw, (h, w));
        } else {
            x = self.attn.forward(x);
        }

        let x = shortcut + x;
        let x_norm2 = self.norm2.forward(x.clone());
        x + self.mlp.forward(x_norm2)
    }
}

#[derive(Module, Debug)]
pub struct SamVitB<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub pos_embed: Param<Tensor<B, 4>>,
    pub blocks: Vec<Block<B>>,
    pub neck: Vec<NeckLayer<B>>,
    pub net_2: Conv2d<B>,
    pub net_3: Conv2d<B>,
    pub img_size: usize,
    pub patch_size: usize,
}

impl<B: Backend> SamVitB<B> {
    pub fn new(device: &B::Device) -> Self {
        // Fixed configuration for DeepSeek-OCR-2.
        let img_size = 1024usize;
        let patch_size = 16usize;
        let embed_dim = 768usize;
        let depth = 12usize;
        let num_heads = 12usize;
        let mlp_ratio = 4.0f64;
        let out_chans = 256usize;
        let window_size = 14usize;
        let global_attn_indexes = [2usize, 5, 8, 11];
        let input_size = [img_size / patch_size, img_size / patch_size]; // [64,64]

        let patch_embed = PatchEmbed::new(3, embed_dim, device);
        let pos_embed =
            Initializer::Zeros.init([1, input_size[0], input_size[1], embed_dim], device);

        let blocks = (0..depth)
            .map(|i| {
                let ws = if global_attn_indexes.contains(&i) {
                    0
                } else {
                    window_size
                };
                Block::new(embed_dim, num_heads, mlp_ratio, ws, input_size, device)
            })
            .collect();

        // neck: Conv(1x1, bias=false) -> LN2d -> Conv(3x3, bias=false) -> LN2d
        let neck_conv1 = Conv2dConfig::new([embed_dim, out_chans], [1, 1])
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let neck_norm1 = LayerNorm2d::new(out_chans, 1e-6, device);
        let neck_conv2 = Conv2dConfig::new([out_chans, out_chans], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let neck_norm2 = LayerNorm2d::new(out_chans, 1e-6, device);
        let neck = vec![
            NeckLayer::Conv(neck_conv1),
            NeckLayer::Norm(neck_norm1),
            NeckLayer::Conv(neck_conv2),
            NeckLayer::Norm(neck_norm2),
        ];

        let net_2 = Conv2dConfig::new([out_chans, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let net_3 = Conv2dConfig::new([512, 896], [3, 3])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        Self {
            patch_embed,
            pos_embed,
            blocks,
            neck,
            net_2,
            net_3,
            img_size,
            patch_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Patch embed -> NHWC tokens.
        let mut x = self.patch_embed.forward(x);

        // Absolute pos embed.
        let [_, h, w, _] = x.dims();
        let [_, ph, pw, c] = self.pos_embed.shape().dims();
        let pos = if (h, w) == (ph, pw) {
            self.pos_embed.val()
        } else {
            // Resize [1, ph, pw, c] -> [1, h, w, c].
            //
            // HF uses:
            //   F.interpolate(mode='bicubic', antialias=True, align_corners=False)
            //
            // Burn doesn't expose `antialias`, and bicubic isn't implemented for `grid_sample` on GPU,
            // so we default to bilinear `grid_sample` with `align_corners=false` (closest available).
            //
            // For debugging/alignment, you can opt into Burn's bicubic interpolate (align_corners=true)
            // by setting: `DEEPSEEK_SAM_POS_RESIZE=interp_bicubic`.
            let pos_nchw = nhwc_to_nchw(self.pos_embed.val()); // [1, c, ph, pw]
            let pos_nchw = match std::env::var("DEEPSEEK_SAM_POS_RESIZE").as_deref() {
                Ok("interp_bicubic") => {
                    resize_nchw_align_corners_true::<B>(pos_nchw, h, w, InterpolateMode::Bicubic)
                }
                Ok("interp_bilinear") => {
                    resize_nchw_align_corners_true::<B>(pos_nchw, h, w, InterpolateMode::Bilinear)
                }
                // Default.
                _ => {
                    resize_nchw_align_corners_false::<B>(pos_nchw, h, w, InterpolateMode::Bilinear)
                }
            };
            nchw_to_nhwc(pos_nchw).reshape([1, h, w, c])
        };
        x = x + pos;

        // Transformer blocks (NHWC).
        for blk in self.blocks.iter() {
            x = blk.forward(x);
        }

        // Neck and downsample (NCHW).
        let mut x = nhwc_to_nchw(x);
        for layer in self.neck.iter() {
            x = layer.forward(x);
        }
        let x2 = self.net_2.forward(x);
        self.net_3.forward(x2)
    }
}
