//! Minimal DeepSeek-V2 (decoder-only) implementation for inference in Burn.
//!
//! This is intentionally scoped to what DeepSeek-OCR-2 uses as its language backbone:
//! - RMSNorm
//! - RoPE MHA (LLaMA-style)
//! - SwiGLU MLP
//! - MoE MLP (top-k routing)
//!
//! Vision components (SAM + Qwen2 encoder) are not implemented here yet.

use burn::module::Param;
use burn::module::{Ignored, Initializer, Module};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, Int, Tensor};

fn dbg_stats<B: Backend, const D: usize>(name: &str, t: &Tensor<B, D>) {
    if std::env::var("DEEPSEEK_DEBUG_ATTN").is_err() {
        return;
    }
    let data = t.clone().cast(DType::F32).into_data().to_vec::<f32>();
    let Ok(data) = data else {
        eprintln!("debug: {name}: failed to read data");
        return;
    };
    let mut nan = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in data.iter() {
        if v.is_nan() {
            nan += 1;
            continue;
        }
        min = min.min(v);
        max = max.max(v);
    }
    eprintln!(
        "debug: {name}: nan={nan} min={min} max={max} shape={:?} dtype={:?}",
        t.dims(),
        t.dtype()
    );
}

/// LLaMA RoPE variant that splits the last dimension in half (not interleaved).
///
/// DeepSeek-OCR-2 sets `use_mla=false` and uses standard MHA layers (HF uses `LlamaAttention`),
/// so we follow `transformers.models.llama.modeling_llama.apply_rotary_pos_emb`.
fn apply_rope_half_split<B: Backend>(
    q: Tensor<B, 4>,        // [B, H, S, D]
    k: Tensor<B, 4>,        // [B, H, S, D]
    inv_freq: Tensor<B, 1>, // [D/2]
    start: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let device = q.device();
    let [bq, hq, seq, dim] = q.dims();
    let [bk, hk, seq_k, dim_k] = k.dims();
    assert_eq!(bq, bk);
    assert_eq!(hq, hk);
    assert_eq!(seq, seq_k);
    assert_eq!(dim, dim_k);
    assert_eq!(dim % 2, 0);
    let half = dim / 2;

    let pos = Tensor::<B, 1, Int>::arange(start as i64..(start + seq) as i64, &device).float(); // [S]
    let freqs = pos.unsqueeze_dim::<2>(1) * inv_freq.unsqueeze_dim::<2>(0); // [S, D/2]
    let emb = Tensor::cat(vec![freqs.clone(), freqs], 1); // [S, D]
    let cos = emb
        .clone()
        .cos()
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0); // [1, 1, S, D]
    let sin = emb.sin().unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0); // [1, 1, S, D]

    let q1 = q.clone().slice([0..bq, 0..hq, 0..seq, 0..half]);
    let q2 = q.clone().slice([0..bq, 0..hq, 0..seq, half..dim]);
    let q_rot = Tensor::cat(vec![-q2, q1], 3);
    let q = q * cos.clone() + q_rot * sin.clone();

    let k1 = k.clone().slice([0..bk, 0..hk, 0..seq, 0..half]);
    let k2 = k.clone().slice([0..bk, 0..hk, 0..seq, half..dim]);
    let k_rot = Tensor::cat(vec![-k2, k1], 3);
    let k = k * cos + k_rot * sin;

    (q, k)
}

#[derive(Debug, Clone)]
pub struct DeepseekV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[allow(dead_code)]
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[allow(dead_code)]
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    /// KV cache dtype used during autoregressive decoding.
    ///
    /// Default is `F32` for numerical stability / backend correctness.
    pub kv_cache_dtype: DType,
    // MoE
    pub first_k_dense_replace: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub moe_intermediate_size: usize,
    pub num_experts_per_tok: usize,
}

impl Default for DeepseekV2Config {
    fn default() -> Self {
        Self {
            vocab_size: 129_280,
            hidden_size: 1280,
            intermediate_size: 6848,
            max_position_embeddings: 8192,
            num_hidden_layers: 12,
            num_attention_heads: 10,
            num_key_value_heads: 10,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            kv_cache_dtype: DType::F32,
            first_k_dense_replace: 1,
            n_routed_experts: 64,
            n_shared_experts: 2,
            moe_intermediate_size: 896,
            num_experts_per_tok: 6,
        }
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(d_model: usize, epsilon: f64, device: &B::Device) -> Self {
        let weight = Initializer::Ones.init([d_model], device);
        Self { weight, epsilon }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Match HF `DeepseekV2RMSNorm`:
        // - compute variance/norm in F32
        // - cast back to input dtype
        // - apply learned weight in input dtype
        let input_dtype = x.dtype();
        let x_f32 = x.clone().cast(DType::F32);
        let rms = (x_f32.clone().square().mean_dim(D - 1) + self.epsilon).sqrt();
        let x_norm = (x_f32 / rms).cast(input_dtype);
        self.weight.val().unsqueeze() * x_norm
    }
}

#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        // DeepSeek-V2 uses bias=False in all MLP linears.
        let gate_proj = LinearConfig::new(d_model, d_ff)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let up_proj = LinearConfig::new(d_model, d_ff)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let down_proj = LinearConfig::new(d_ff, d_model)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // x: [n, d_model]
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    // We keep these in `Option` so we can temporarily `take()` them and let
    // `slice_assign` mutate in-place (avoids O(seq^2) copies on GPU backends).
    pub k: Option<Tensor<B, 4>>, // [batch, heads, cap, head_dim]
    pub v: Option<Tensor<B, 4>>, // [batch, heads, cap, head_dim]
    pub len: usize,              // number of valid timesteps currently stored
    pub cap: usize,              // allocated capacity along the sequence dimension
}

#[derive(Module, Debug)]
pub struct LlamaSelfAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    /// RoPE inverse frequencies for Llama-style half-split rotation.
    ///
    /// Shape: `[head_dim / 2]`.
    pub inv_freq: Tensor<B, 1>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub kv_cache_dtype: Ignored<DType>,
}

impl<B: Backend> LlamaSelfAttention<B> {
    pub fn new(config: &DeepseekV2Config, device: &B::Device) -> Self {
        // DeepSeek-V2 (in OCR2) uses standard MHA (no GQA) with bias=False.
        let q_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let k_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let v_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let o_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        let head_dim = config.hidden_size / config.num_attention_heads;
        // Match HF Llama rotary embedding:
        // inv_freq[i] = 1 / (theta^(2*i/head_dim)) for i in [0..head_dim/2).
        let inv_freq = Tensor::<B, 1, Int>::arange_step(0..head_dim as i64, 2, device)
            .float()
            .div_scalar(head_dim as f32)
            .mul_scalar(config.rope_theta.ln())
            .exp()
            .recip();

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            inv_freq,
            n_heads: config.num_attention_heads,
            head_dim,
            kv_cache_dtype: Ignored(config.kv_cache_dtype),
        }
    }

    /// Forward pass with an in-place KV cache (for autoregressive decoding).
    ///
    /// - `x`: `[batch, seq_new, hidden]`
    /// - `cache`: stores `[batch, heads, seq_total, head_dim]`
    pub fn forward(&self, x: Tensor<B, 3>, cache: &mut Option<KvCache<B>>) -> Tensor<B, 3> {
        let device = x.device();
        let cache_dtype = *self.kv_cache_dtype;
        // Keep the output dtype consistent with the model activation dtype (typically F16 on Vulkan).
        // We'll upcast the attention matmuls/softmax to F32 for backend correctness.
        let out_dtype = x.dtype();
        dbg_stats("attn.in_x", &x);
        let [batch, seq_new, _hidden] = x.dims();
        let past_len = cache.as_ref().map(|c| c.len).unwrap_or(0);
        let seq_total = past_len + seq_new;

        // Projections.
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);
        dbg_stats("attn.q_proj", &q);
        dbg_stats("attn.k_proj", &k);
        dbg_stats("attn.v_proj", &v);

        // [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq_new, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq_new, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq_new, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        dbg_stats("attn.q_reshaped", &q);
        dbg_stats("attn.k_reshaped", &k);
        dbg_stats("attn.v_reshaped", &v);

        // Apply RoPE to the *new* tokens only.
        //
        // NOTE: Keep attention math (RoPE + matmuls + softmax) in F32 for numerical stability and
        // Vulkan/WebGPU backend correctness. We'll cast back to the model dtype at the end.
        let q = q.cast(DType::F32);
        let k = k.cast(DType::F32);
        let v = v.cast(DType::F32);
        let (q, k) = apply_rope_half_split(q, k, self.inv_freq.clone(), past_len);
        dbg_stats("attn.q_rope", &q);
        dbg_stats("attn.k_rope", &k);

        // Store KV cache in the requested dtype (opt-in for memory savings).
        let k_store = if cache_dtype == DType::F32 {
            k.clone()
        } else {
            k.clone().cast(cache_dtype)
        };
        let v_store = if cache_dtype == DType::F32 {
            v.clone()
        } else {
            v.clone().cast(cache_dtype)
        };

        // Update cache (efficiently, without O(seq^2) concatenation).
        let (k_all, v_all) = match cache {
            Some(kv) => {
                // Grow capacity if needed.
                if kv.cap < seq_total {
                    let new_cap = (kv.cap * 2).max(seq_total);

                    let mut k_new = Tensor::<B, 4>::zeros(
                        [batch, self.n_heads, new_cap, self.head_dim],
                        &device,
                    )
                    .cast(cache_dtype);
                    let mut v_new = Tensor::<B, 4>::zeros(
                        [batch, self.n_heads, new_cap, self.head_dim],
                        &device,
                    )
                    .cast(cache_dtype);

                    // Copy existing cache into the new buffers.
                    let k_old = kv.k.as_ref().expect("kv.k missing").clone().slice([
                        0..batch,
                        0..self.n_heads,
                        0..past_len,
                        0..self.head_dim,
                    ]);
                    let v_old = kv.v.as_ref().expect("kv.v missing").clone().slice([
                        0..batch,
                        0..self.n_heads,
                        0..past_len,
                        0..self.head_dim,
                    ]);
                    k_new = k_new.slice_assign(
                        [0..batch, 0..self.n_heads, 0..past_len, 0..self.head_dim],
                        k_old,
                    );
                    v_new = v_new.slice_assign(
                        [0..batch, 0..self.n_heads, 0..past_len, 0..self.head_dim],
                        v_old,
                    );

                    kv.k = Some(k_new);
                    kv.v = Some(v_new);
                    kv.cap = new_cap;
                }

                // Write the new keys/values at [past_len..seq_total).
                let mut k_buf = kv.k.take().expect("kv.k missing");
                k_buf = k_buf.slice_assign(
                    [
                        0..batch,
                        0..self.n_heads,
                        past_len..seq_total,
                        0..self.head_dim,
                    ],
                    k_store.clone(),
                );
                kv.k = Some(k_buf);

                let mut v_buf = kv.v.take().expect("kv.v missing");
                v_buf = v_buf.slice_assign(
                    [
                        0..batch,
                        0..self.n_heads,
                        past_len..seq_total,
                        0..self.head_dim,
                    ],
                    v_store.clone(),
                );
                kv.v = Some(v_buf);

                kv.len = seq_total;

                // Read the valid prefix only.
                let k_all = kv.k.as_ref().expect("kv.k missing").clone().slice([
                    0..batch,
                    0..self.n_heads,
                    0..seq_total,
                    0..self.head_dim,
                ]);
                let v_all = kv.v.as_ref().expect("kv.v missing").clone().slice([
                    0..batch,
                    0..self.n_heads,
                    0..seq_total,
                    0..self.head_dim,
                ]);
                (k_all, v_all)
            }
            None => {
                // Initial cache: allocate a bit of headroom to avoid early reallocations.
                let cap = (seq_total * 2).max(1024);
                let mut k_buf =
                    Tensor::<B, 4>::zeros([batch, self.n_heads, cap, self.head_dim], &device)
                        .cast(cache_dtype);
                let mut v_buf =
                    Tensor::<B, 4>::zeros([batch, self.n_heads, cap, self.head_dim], &device)
                        .cast(cache_dtype);
                k_buf = k_buf.slice_assign(
                    [0..batch, 0..self.n_heads, 0..seq_total, 0..self.head_dim],
                    k_store.clone(),
                );
                v_buf = v_buf.slice_assign(
                    [0..batch, 0..self.n_heads, 0..seq_total, 0..self.head_dim],
                    v_store.clone(),
                );
                *cache = Some(KvCache {
                    k: Some(k_buf),
                    v: Some(v_buf),
                    len: seq_total,
                    cap,
                });
                // For the initial prompt pass, keep attention math in F32 (more stable) even when
                // the cache is stored in a lower precision.
                (k, v)
            }
        };

        // Always do attention matmul/softmax in F32 for backend correctness.
        //
        // NOTE: When the KV cache is stored in a lower precision (e.g. F16), we cast the cached
        // tensors back to F32 for the attention math. This keeps outputs stable, but doesn't
        // necessarily reduce peak memory on all backends (casts create temporary buffers).
        let k_all_f32 = if k_all.dtype() == DType::F32 {
            k_all
        } else {
            k_all.cast(DType::F32)
        };
        let v_all_f32 = if v_all.dtype() == DType::F32 {
            v_all
        } else {
            v_all.cast(DType::F32)
        };

        // Attention scores: [batch, heads, seq_new, seq_total]
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scores = q.matmul(k_all_f32.swap_dims(2, 3)).div_scalar(scale);
        dbg_stats("attn.scores", &scores);

        // Causal mask for the new tokens (shape [seq_new, seq_total]).
        let q_pos = Tensor::<B, 1, Int>::arange(past_len as i64..seq_total as i64, &device);
        let k_pos = Tensor::<B, 1, Int>::arange(0..seq_total as i64, &device);
        let q_pos: Tensor<B, 2, Int> = q_pos.unsqueeze_dim(1);
        let k_pos: Tensor<B, 2, Int> = k_pos.unsqueeze_dim(0);
        let mask_2d: Tensor<B, 2, Bool> = k_pos.greater(q_pos);
        let mut mask: Tensor<B, 4, Bool> = mask_2d.unsqueeze();
        mask = mask.repeat_dim(0, batch).repeat_dim(1, self.n_heads);

        // Mask out future positions.
        let scores = scores.mask_fill(mask, -1.0e4);
        let weights = softmax(scores, 3);
        dbg_stats("attn.weights", &weights);

        let context = weights.matmul(v_all_f32); // [batch, heads, seq_new, head_dim]
        dbg_stats("attn.context", &context);
        let context =
            context
                .swap_dims(1, 2)
                .reshape([batch, seq_new, self.n_heads * self.head_dim]);

        let out = self.o_proj.forward(context.cast(out_dtype));
        dbg_stats("attn.out", &out);
        out
    }
}

#[derive(Module, Debug)]
pub struct MoEMlp<B: Backend> {
    pub gate: Linear<B>,
    pub experts: Vec<SwiGluMlp<B>>,
    pub shared_experts: SwiGluMlp<B>,
    pub top_k: usize,
    pub n_experts: usize,
}

impl<B: Backend> MoEMlp<B> {
    pub fn new(config: &DeepseekV2Config, device: &B::Device) -> Self {
        let gate = LinearConfig::new(config.hidden_size, config.n_routed_experts)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        let experts = (0..config.n_routed_experts)
            .map(|_| SwiGluMlp::new(config.hidden_size, config.moe_intermediate_size, device))
            .collect();

        let shared_experts = SwiGluMlp::new(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            device,
        );

        Self {
            gate,
            experts,
            shared_experts,
            top_k: config.num_experts_per_tok,
            n_experts: config.n_routed_experts,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let dtype = x.dtype();
        let [batch, seq, hidden] = x.dims();
        let n_tokens = batch * seq;

        // Flatten tokens: [n_tokens, hidden]
        let x_flat = x.clone().reshape([n_tokens, hidden]);

        // Gate scores: [n_tokens, n_experts].
        //
        // NOTE: Some Vulkan/WebGPU drivers have correctness issues with integer tensor readbacks and/or
        // `sort_descending_with_indices` (used by `topk_with_indices`). To keep routing correct and
        // deterministic, we do the softmax + top-k selection on CPU (n_experts=64 is tiny), then
        // upload the results back to the device.
        // Match HF (modeling_deepseekv2.py): gate logits are always computed in float32,
        // regardless of the model's weight dtype.
        //
        // This matters because MoE routing is discontinuous: tiny numeric differences can flip
        // expert selection on some ambiguous inputs.
        let x_gate = x_flat.clone().cast(DType::F32);
        let w_gate = self.gate.weight.val().clone().cast(DType::F32);
        let logits = x_gate.matmul(w_gate); // [n_tokens, n_experts] (F32)
        let logits_vec = logits
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("failed to read MoE gate logits");

        let mut topk_idx_cpu: Vec<i64> = Vec::with_capacity(n_tokens * self.top_k);
        let mut topk_weight_cpu: Vec<f32> = Vec::with_capacity(n_tokens * self.top_k);

        for t in 0..n_tokens {
            let start = t * self.n_experts;
            let end = start + self.n_experts;
            let logits_t = &logits_vec[start..end];

            // Stable softmax in f32 (matches HF).
            let mut max = f32::NEG_INFINITY;
            for &v in logits_t.iter() {
                if v.is_nan() {
                    continue;
                }
                max = max.max(v);
            }
            let mut probs_t: Vec<f32> = Vec::with_capacity(self.n_experts);
            let mut sum = 0f32;
            for &v in logits_t.iter() {
                let e = (v - max).exp();
                probs_t.push(e);
                sum += e;
            }
            let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
            for p in probs_t.iter_mut() {
                *p *= inv;
            }

            // Top-k selection (descending).
            let mut best: Vec<(usize, f32)> = Vec::with_capacity(self.top_k);
            for (i, &p) in probs_t.iter().enumerate() {
                if p.is_nan() {
                    continue;
                }
                if best.len() < self.top_k {
                    best.push((i, p));
                    if best.len() == self.top_k {
                        best.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    continue;
                }
                if best.last().copied().is_some_and(|(_, min_p)| p <= min_p) {
                    continue;
                }
                let mut pos = 0usize;
                while pos < best.len()
                    && best[pos]
                        .1
                        .partial_cmp(&p)
                        .unwrap_or(std::cmp::Ordering::Less)
                        == std::cmp::Ordering::Greater
                {
                    pos += 1;
                }
                best.insert(pos, (i, p));
                best.truncate(self.top_k);
            }

            for (i, p) in best.into_iter() {
                topk_idx_cpu.push(i as i64);
                topk_weight_cpu.push(p);
            }
        }

        if std::env::var("DEEPSEEK_DEBUG_MOE").is_ok() {
            eprintln!("debug: moe topk_idx={topk_idx_cpu:?}");
            eprintln!("debug: moe topk_weight(f32)={topk_weight_cpu:?}");
        }

        let topk_weight: Tensor<B, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(topk_weight_cpu.clone(), [n_tokens, self.top_k]),
            &device,
        );

        // Expand each token `top_k` times.
        let x_rep: Tensor<B, 3> = x_flat.clone().unsqueeze_dim(1).repeat_dim(1, self.top_k);
        let x_rep: Tensor<B, 2> = x_rep.reshape([n_tokens * self.top_k, hidden]);

        // Sort assignments by expert id for contiguous slices.
        //
        // `argsort` is still shaky on some Vulkan/WebGPU drivers. Do the sort on CPU (this is tiny
        // compared to the MLP compute) and then re-index tensors on-device.
        let expert_idx_vec = topk_idx_cpu;
        let n_assign = expert_idx_vec.len();
        let mut order_vec: Vec<i64> = (0..n_assign as i64).collect();
        order_vec.sort_by_key(|&i| expert_idx_vec[i as usize]);

        // Invert the permutation so we can restore the original (token-major) order without scatter.
        let mut inv_vec: Vec<i64> = vec![0; n_assign];
        for (sorted_pos, &orig_pos) in order_vec.iter().enumerate() {
            inv_vec[orig_pos as usize] = sorted_pos as i64;
        }
        let order = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(order_vec.clone(), [n_assign]),
            &device,
        );
        let inv = Tensor::<B, 1, Int>::from_data(
            burn::tensor::TensorData::new(inv_vec, [n_assign]),
            &device,
        );
        let mut counts: Vec<usize> = vec![0; self.n_experts];
        for &e in expert_idx_vec.iter() {
            let e = e as usize;
            if e < counts.len() {
                counts[e] += 1;
            }
        }
        if std::env::var("DEEPSEEK_DEBUG_MOE").is_ok() {
            eprintln!("debug: moe counts={counts:?}");
        }

        let x_rep_sorted = x_rep.select(0, order);

        let mut offset: usize = 0;
        let mut out_parts: Vec<Tensor<B, 2>> = Vec::new();

        for (expert_id, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let start = offset;
            let end = offset + count;
            offset = end;

            // Slice this expert's assigned tokens.
            let tokens_e = x_rep_sorted.clone().slice(start..end); // [count, hidden]

            // Expert MLP.
            let expert_out = self.experts[expert_id].forward(tokens_e); // [count, hidden]
            out_parts.push(expert_out);
        }

        // Concatenate expert outputs in sorted assignment order.
        let out_sorted = if out_parts.is_empty() {
            // Shouldn't happen in normal inference, but keep a sane fallback.
            Tensor::<B, 2>::zeros([n_assign, hidden], &device).cast(dtype)
        } else {
            Tensor::cat(out_parts, 0)
        };

        // Restore original assignment order (token-major) and apply weights in F32 like HF.
        let out = out_sorted.select(0, inv); // [n_tokens * top_k, hidden]
        let out = out.reshape([n_tokens, self.top_k, hidden]).cast(DType::F32);
        let w = topk_weight
            .reshape([n_tokens, self.top_k])
            .unsqueeze_dim::<3>(2); // [n_tokens, top_k, 1]
        let routed = (out * w)
            .sum_dim(1) // [n_tokens, hidden]
            .cast(dtype)
            .reshape([batch, seq, hidden]);

        // Add shared experts (dense path).
        let shared = self
            .shared_experts
            .forward(x_flat)
            .reshape([batch, seq, hidden]);
        if std::env::var("DEEPSEEK_DEBUG_MOE").is_ok() {
            dbg_stats("moe.shared", &shared);
        }

        let out = routed + shared;
        if std::env::var("DEEPSEEK_DEBUG_MOE").is_ok() {
            dbg_stats("moe.out_total", &out);
        }
        out
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum MlpBlock<B: Backend> {
    Dense(SwiGluMlp<B>),
    Moe(MoEMlp<B>),
}

impl<B: Backend> MlpBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Dense(mlp) => {
                let [batch, seq, hidden] = x.dims();
                mlp.forward(x.reshape([batch * seq, hidden]))
                    .reshape([batch, seq, hidden])
            }
            Self::Moe(moe) => moe.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct DeepseekV2DecoderLayer<B: Backend> {
    pub self_attn: LlamaSelfAttention<B>,
    pub mlp: MlpBlock<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> DeepseekV2DecoderLayer<B> {
    pub fn new(config: &DeepseekV2Config, layer_idx: usize, device: &B::Device) -> Self {
        let self_attn = LlamaSelfAttention::new(config, device);
        let input_layernorm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);
        let post_attention_layernorm =
            RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        let mlp = if layer_idx < config.first_k_dense_replace {
            MlpBlock::Dense(SwiGluMlp::new(
                config.hidden_size,
                config.intermediate_size,
                device,
            ))
        } else {
            MlpBlock::Moe(MoEMlp::new(config, device))
        };

        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, cache: &mut Option<KvCache<B>>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x, cache);
        let x = residual + x;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

#[derive(Module, Debug)]
pub struct DeepseekV2Model<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DeepseekV2DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

impl<B: Backend> DeepseekV2Model<B> {
    pub fn new(config: &DeepseekV2Config, device: &B::Device) -> Self {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|i| DeepseekV2DecoderLayer::new(config, i, device))
            .collect();
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        Self {
            embed_tokens,
            layers,
            norm,
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        caches: &mut Vec<Option<KvCache<B>>>,
    ) -> Tensor<B, 3> {
        let x = self.embed_tokens.forward(input_ids);
        self.forward_embeds(x, caches)
    }

    pub fn forward_embeds(
        &self,
        mut inputs_embeds: Tensor<B, 3>,
        caches: &mut Vec<Option<KvCache<B>>>,
    ) -> Tensor<B, 3> {
        if caches.len() != self.layers.len() {
            caches.resize_with(self.layers.len(), || None);
        }

        // Decoder blocks.
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            inputs_embeds = layer.forward(inputs_embeds, cache);
        }

        // Final norm.
        self.norm.forward(inputs_embeds)
    }
}

#[derive(Module, Debug)]
pub struct DeepseekV2ForCausalLM<B: Backend> {
    pub model: DeepseekV2Model<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> DeepseekV2ForCausalLM<B> {
    pub fn new(config: &DeepseekV2Config, device: &B::Device) -> Self {
        let model = DeepseekV2Model::new(config, device);
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self { model, lm_head }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        caches: &mut Vec<Option<KvCache<B>>>,
    ) -> Tensor<B, 3> {
        let hidden = self.model.forward(input_ids, caches);
        self.logits_from_hidden(hidden)
    }

    #[allow(dead_code)]
    pub fn forward_embeds(
        &self,
        inputs_embeds: Tensor<B, 3>,
        caches: &mut Vec<Option<KvCache<B>>>,
    ) -> Tensor<B, 3> {
        let hidden = self.model.forward_embeds(inputs_embeds, caches);
        self.logits_from_hidden(hidden)
    }

    fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, hidden_size] = hidden.dims();
        let vocab = self.lm_head.weight.shape().dims::<2>()[1];
        self.lm_head
            .forward(hidden.reshape([batch * seq, hidden_size]))
            .reshape([batch, seq, vocab])
    }
}
