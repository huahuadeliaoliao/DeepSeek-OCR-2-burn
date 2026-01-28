# DeepSeek-OCR-2 Burn

Experimental, pure-Rust inference for the DeepSeek-OCR-2 vision-language OCR model using
[Burn](https://github.com/tracel-ai/burn).

This repository focuses on:

- Loading Hugging Face `safetensors` weights (PyTorch layout) into Burn.
- End-to-end OCR inference (vision -> LLM injection) in Rust.
- Running with either the **Vulkan (wgpu)** backend or the **NdArray (CPU)** backend.

Status: the OCR pipeline works, but the runtime and memory usage are not yet competitive with
mature Python stacks (e.g. PyTorch + FlashAttention / vLLM). Expect slower inference, especially on
integrated GPUs.

## Requirements

- Rust (stable, edition 2024).
- For `--backend vulkan`: a working Vulkan driver (wgpu uses Vulkan).
- For `--backend ndarray`: CPU only.

## Model Weights (Hugging Face)

Download the model files into `hf_deepseek_ocr2/` at the repository root.

### Option A: `huggingface-cli`

```bash
pip install -U huggingface-hub

huggingface-cli download deepseek-ai/DeepSeek-OCR-2 \
  --local-dir hf_deepseek_ocr2 \
  --local-dir-use-symlinks False \
  --include "model-*.safetensors" "tokenizer.json"
```

You should end up with:

- `hf_deepseek_ocr2/model-00001-of-000001.safetensors`
- `hf_deepseek_ocr2/tokenizer.json`

## Build

```bash
cargo build --release
```

## Usage

### OCR (image -> text)

The OCR prompt is currently built-in as:

```text
<image>
Free OCR.
```

Run OCR on an image:

```bash
./target/release/deepseek_ocr2_burn generate-ocr \
  --backend vulkan \
  --weights hf_deepseek_ocr2/model-00001-of-000001.safetensors \
  --tokenizer hf_deepseek_ocr2/tokenizer.json \
  --image path/to/image.jpg \
  --auto-rotate
```

Common flags:

- `--backend vulkan|ndarray` (default: `vulkan`)
- `--rotate 0|90|180|270` (clockwise) or `--auto-rotate`
- `--no-crop` disable dynamic tiling (by default large images are tiled into 768x768 crops plus a
  1024x1024 global view, matching the HF reference behavior)
- `--trim-memory` best-effort reduce *CPU-side* memory after loading weights (drops OS page cache
  for the weights file + `malloc_trim(0)` on glibc)
- `--kv-cache f32|f16` (default: `f32`) - `f16` is experimental; outputs may drift on some setups

### Text-only generation

```bash
./target/release/deepseek_ocr2_burn generate-text \
  --backend vulkan \
  --weights hf_deepseek_ocr2/model-00001-of-000001.safetensors \
  --tokenizer hf_deepseek_ocr2/tokenizer.json \
  --prompt "Hello"
```

## Backends

### Vulkan (wgpu)

- Uses Burn's WGPU Vulkan backend.
- Best option for GPU acceleration.
- On integrated GPUs the "GPU memory" is shared system RAM; system memory usage can look high.

### NdArray (CPU)

- Pure CPU backend (Burn NdArray).
- Requires casting weights to FP32 (NdArray does not support BF16/F16 weights), which increases
  memory usage.

## Memory Usage (Example)

Measured on Linux, Intel iGPU, 32GB RAM, image 1080x1920, `--max-new-tokens 32`, `--trim-memory`:

- `vulkan`: peak process RSS ~4.5 GiB
- `ndarray`: peak process RSS ~15.7 GiB

Notes:

- System memory reported by `free`/`top` can be higher than process RSS on Vulkan (especially on
  integrated GPUs) due to driver/unified-memory allocations and file page cache.
- The numbers above are an example; they vary with image resolution, generation length, driver,
  and hardware.

## Performance Notes

Burn is still evolving, and its GPU kernels and memory planning are not yet as optimized as mature
Python runtimes. This project prioritizes correctness and a pure-Rust pipeline, so you should
expect slower inference and higher memory usage in many configurations.

## License

See `LICENSE`.

