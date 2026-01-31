//! Qwen2 decoder-as-encoder used in DeepSeek-OCR-2 vision tower (DeepEncoderV2).
//!
//! This implements the minimal subset used at inference:
//! - RMSNorm
//! - RoPE attention with GQA (num_kv_heads != num_heads)
//! - SwiGLU MLP
//! - Prefix-LM attention mask (non-causal prefix + causal suffix)

use burn::module::{Initializer, Module};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, Int, Tensor};

use super::deepseek_v2::RmsNorm;

#[derive(Debug, Clone)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[allow(dead_code)]
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Self {
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            max_position_embeddings: 131_072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

fn prefix_lm_mask<B: Backend>(
    batch: usize,
    n_heads: usize,
    seq: usize,
    n_prefix: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    // Mask definition (True = masked / disallowed):
    // - prefix rows: can attend to prefix cols only
    // - suffix rows: can attend to all prefix cols and causal within suffix
    let pos = Tensor::<B, 1, Int>::arange(0..seq as i64, device);
    let row: Tensor<B, 2, Int> = pos.clone().unsqueeze_dim(1);
    let col: Tensor<B, 2, Int> = pos.unsqueeze_dim(0);

    let prefix_row = row.clone().lower_elem(n_prefix as i64); // [seq,1]
    let query_col = col.clone().greater_equal_elem(n_prefix as i64); // [1,seq]
    let disallow_prefix: Tensor<B, 2, Bool> = prefix_row.clone().bool_and(query_col.clone()); // [seq,seq]

    let query_row = row.clone().greater_equal_elem(n_prefix as i64); // [seq,1]
    let future: Tensor<B, 2, Bool> = col.greater(row); // [seq,seq]
    let disallow_query: Tensor<B, 2, Bool> = query_row.bool_and(query_col).bool_and(future); // [seq,seq]

    let disallow: Tensor<B, 2, Bool> = disallow_prefix.bool_or(disallow_query);
    disallow
        .unsqueeze::<4>() // [1,1,seq,seq]
        .repeat_dim(0, batch)
        .repeat_dim(1, n_heads)
}

/// Qwen2/Llama RoPE variant that splits the last dimension in half (not interleaved).
///
/// This matches `transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb`.
fn apply_rope_half_split<B: Backend>(
    q: Tensor<B, 4>,        // [B, H, S, D]
    k: Tensor<B, 4>,        // [B, H_kv, S, D]
    inv_freq: Tensor<B, 1>, // [D/2]
    start: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let device = q.device();
    let [bq, hq, seq, dim] = q.dims();
    let [bk, hk, seq_k, dim_k] = k.dims();
    assert_eq!(seq, seq_k);
    assert_eq!(dim, dim_k);
    assert_eq!(bq, bk);
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

#[derive(Module, Debug)]
pub struct Qwen2Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    /// RoPE inverse frequencies for Llama/Qwen-style half-split rotation.
    ///
    /// Shape: `[head_dim / 2]`.
    pub inv_freq: Tensor<B, 1>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Qwen2Attention<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let q_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let k_proj = LinearConfig::new(
            config.hidden_size,
            config.num_key_value_heads * (config.hidden_size / config.num_attention_heads),
        )
        .with_bias(true)
        .with_initializer(Initializer::Zeros)
        .init(device);
        let v_proj = LinearConfig::new(
            config.hidden_size,
            config.num_key_value_heads * (config.hidden_size / config.num_attention_heads),
        )
        .with_bias(true)
        .with_initializer(Initializer::Zeros)
        .init(device);
        let o_proj = LinearConfig::new(config.hidden_size, config.hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        let head_dim = config.hidden_size / config.num_attention_heads;
        // Match HF Qwen2 rotary embedding:
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
            n_kv_heads: config.num_key_value_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let out_dtype = x.dtype();

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // [B, seq, heads, head_dim] -> [B, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // RoPE (Qwen2/Llama style: half split).
        //
        // NOTE: Do RoPE + attention in F32 for numerical stability and backend correctness
        // (especially important when the vision tower weights are F16 on Vulkan/WebGPU).
        let (q, k) = apply_rope_half_split(
            q.clone().cast(DType::F32),
            k.clone().cast(DType::F32),
            self.inv_freq.clone(),
            0,
        );
        let v = v.cast(DType::F32);

        // Expand KV heads (GQA) => [B, heads, seq, head_dim]
        let group = self.n_heads / self.n_kv_heads;
        let k = k
            .unsqueeze_dim::<5>(2) // [B, kv, 1, seq, head_dim]
            .repeat_dim(2, group)
            .reshape([batch, self.n_heads, seq, self.head_dim]);
        let v = v.unsqueeze_dim::<5>(2).repeat_dim(2, group).reshape([
            batch,
            self.n_heads,
            seq,
            self.head_dim,
        ]);

        // Attention.
        let scale = (self.head_dim as f32).sqrt();
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar(scale)
            .mask_fill(mask, -1.0e4);
        let weights = softmax(scores, 3);
        let ctx = weights.matmul(v); // [B, heads, seq, head_dim]

        let ctx = ctx
            .swap_dims(1, 2)
            .reshape([batch, seq, self.n_heads * self.head_dim]);
        self.o_proj.forward(ctx.cast(out_dtype))
    }
}

#[derive(Module, Debug)]
pub struct Qwen2Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Qwen2Mlp<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let gate_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let up_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let down_proj = LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::Zeros)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct Qwen2DecoderLayer<B: Backend> {
    pub self_attn: Qwen2Attention<B>,
    pub mlp: Qwen2Mlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen2DecoderLayer<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let self_attn = Qwen2Attention::new(config, device);
        let mlp = Qwen2Mlp::new(config, device);
        let input_layernorm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);
        let post_attention_layernorm =
            RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x);
        let x = self.self_attn.forward(x, mask);
        let x = residual + x;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

#[derive(Module, Debug)]
pub struct Qwen2Model<B: Backend> {
    pub layers: Vec<Qwen2DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

impl<B: Backend> Qwen2Model<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| Qwen2DecoderLayer::new(config, device))
            .collect();
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);
        Self { layers, norm }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let mut x = x;
        for layer in self.layers.iter() {
            x = layer.forward(x, mask.clone());
        }
        self.norm.forward(x)
    }
}

// Match HF key path: `qwen2_model.model.model.*`
#[derive(Module, Debug)]
pub struct Qwen2Wrapper<B: Backend> {
    pub model: Qwen2Model<B>,
}

impl<B: Backend> Qwen2Wrapper<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        Self {
            model: Qwen2Model::new(config, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Qwen2Decoder2Encoder<B: Backend> {
    pub model: Qwen2Wrapper<B>,
    pub query_768: Embedding<B>,
    pub query_1024: Embedding<B>,
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl<B: Backend> Qwen2Decoder2Encoder<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let model = Qwen2Wrapper::new(config, device);

        let query_768 = EmbeddingConfig::new(144, config.hidden_size)
            .with_initializer(Initializer::Zeros)
            .init(device);
        let query_1024 = EmbeddingConfig::new(256, config.hidden_size)
            .with_initializer(Initializer::Zeros)
            .init(device);

        Self {
            model,
            query_768,
            query_1024,
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // x: [B, hidden, H, W] -> tokens [B, H*W, hidden]
        let [batch, hidden, h, w] = x.dims();
        assert_eq!(hidden, self.hidden_size);
        let n_query = h * w;

        let x = x.reshape([batch, hidden, n_query]).swap_dims(1, 2);

        let param_img = match n_query {
            144 => self.query_768.weight.val(),
            256 => self.query_1024.weight.val(),
            _ => panic!("unsupported n_query={n_query} (expected 144 or 256)"),
        };

        let batch_query = param_img.unsqueeze_dim(0).repeat_dim(0, batch); // [B, n_query, hidden]
        let x_combined = Tensor::cat(vec![x, batch_query], 1); // [B, 2*n_query, hidden]

        let device = x_combined.device();
        let mask = prefix_lm_mask::<B>(batch, self.num_heads, 2 * n_query, n_query, &device);
        let y = self.model.model.forward(x_combined, mask); // [B, 2*n_query, hidden]

        // Return the causal query part.
        y.slice([0..batch, n_query..(2 * n_query), 0..hidden])
    }
}
