//! DeepSeek-OCR-2 end-to-end model (vision + language) in Burn.
//!
//! Architecture (HF reference):
//! - `model.sam_model`: SAM ViT-B image encoder producing `[B, 896, 16, 16]`
//! - `model.qwen2_model`: Qwen2 decoder-as-encoder producing `[B, 256, 896]`
//! - `model.projector.layers`: Linear(896 -> 1280)
//! - `model.view_seperator`: learnable 1280-d vector appended to vision tokens
//! - Language backbone: DeepSeek-V2 (12 layers, MoE) + `lm_head`

use burn::module::{Initializer, Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::IndexingUpdateOp;
use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};

use super::deepseek_v2::{DeepseekV2Config, DeepseekV2DecoderLayer, KvCache, RmsNorm};
use super::qwen2::{Qwen2Config, Qwen2Decoder2Encoder};
use super::sam::SamVitB;

fn dbg_stats<B: Backend, const D: usize>(name: &str, t: &Tensor<B, D>) {
    if std::env::var("DEEPSEEK_DEBUG_VISION").is_err() {
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

#[derive(Module, Debug)]
pub struct MlpProjector<B: Backend> {
    pub layers: Linear<B>,
}

impl<B: Backend> MlpProjector<B> {
    pub fn new(input_dim: usize, n_embed: usize, device: &B::Device) -> Self {
        let layers = LinearConfig::new(input_dim, n_embed)
            .with_bias(true)
            .with_initializer(Initializer::Zeros)
            .init(device);
        Self { layers }
    }
}

#[derive(Module, Debug)]
pub struct DeepseekOcr2Model<B: Backend> {
    // Vision tower.
    pub sam_model: SamVitB<B>,
    pub qwen2_model: Qwen2Decoder2Encoder<B>,
    pub projector: MlpProjector<B>,
    pub view_seperator: Param<Tensor<B, 1>>,

    // Language backbone (DeepSeek-V2).
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<DeepseekV2DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

impl<B: Backend> DeepseekOcr2Model<B> {
    pub fn new(config: &DeepseekV2Config, device: &B::Device) -> Self {
        let sam_model = SamVitB::new(device);
        let qwen2_cfg = Qwen2Config::default();
        let qwen2_model = Qwen2Decoder2Encoder::new(&qwen2_cfg, device);
        let projector = MlpProjector::new(896, config.hidden_size, device);
        let view_seperator = Initializer::Zeros.init([config.hidden_size], device);

        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|i| DeepseekV2DecoderLayer::new(config, i, device))
            .collect();
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        Self {
            sam_model,
            qwen2_model,
            projector,
            view_seperator,
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

        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            inputs_embeds = layer.forward(inputs_embeds, cache);
        }

        self.norm.forward(inputs_embeds)
    }

    /// Compute multimodal `inputs_embeds` by replacing `<image>` placeholder tokens with vision
    /// embeddings produced by (SAM -> Qwen2 -> projector).
    ///
    /// Current scope:
    /// - batch size = 1
    pub fn build_inputs_embeds_with_image(
        &self,
        input_ids: Tensor<B, 2, Int>,
        image_base: Tensor<B, 4>,
        patches: Option<Tensor<B, 4>>,
        images_seq_mask: &[bool],
    ) -> anyhow::Result<Tensor<B, 3>> {
        let device = input_ids.device();
        let [batch, seq] = input_ids.dims();
        anyhow::ensure!(batch == 1, "only batch=1 is supported for now");
        anyhow::ensure!(
            images_seq_mask.len() == seq,
            "images_seq_mask length mismatch (mask={}, seq={seq})",
            images_seq_mask.len()
        );

        // Base token embeddings.
        let base = self.embed_tokens.forward(input_ids); // [1, seq, hidden]
        let [_, _, hidden] = base.dims();
        let embed_dtype = base.dtype();
        let base = base.squeeze_dim(0); // [seq, hidden]
        if std::env::var("DEEPSEEK_DEBUG_VISION").is_ok() {
            dbg_stats("w.projector.weight", &self.projector.layers.weight.val());
            if let Some(bias) = &self.projector.layers.bias {
                dbg_stats("w.projector.bias", &bias.val());
            }
        }

        // Global view: [n_query, hidden] where n_query depends on the SAM feature map size:
        // - 1024 base -> 16x16 -> 256
        // - 768  base -> 12x12 -> 144
        //
        // On Vulkan/WebGPU, Qwen2 vision ops are unstable in F16 for this model. Keep the whole
        // vision tower in F32, then cast to the language embedding dtype at the end.
        let global = {
            let feats = self.sam_model.forward(image_base); // [1, 896, 16, 16] (for 1024)
            dbg_stats("vision.global.sam", &feats);
            let feats = self.qwen2_model.forward(feats); // [1, 256, 896]
            dbg_stats("vision.global.qwen2", &feats);
            let feats = self.projector.layers.forward(feats); // [1, 256, hidden]
            dbg_stats("vision.global.proj", &feats);
            let [_, n_query, _] = feats.dims();
            feats.reshape([n_query, hidden])
        };

        // Optional local patches (each crop_image_size -> (crop/16)^2/16 tokens, typically 144 for 768).
        let vision = if let Some(patches) = patches {
            let [p, _, _, _] = patches.dims();
            let feats = self.sam_model.forward(patches); // [P, 896, 12, 12] (for 768)
            dbg_stats("vision.local.sam", &feats);
            let feats = self.qwen2_model.forward(feats); // [P, 144, 896]
            dbg_stats("vision.local.qwen2", &feats);
            let feats = self.projector.layers.forward(feats); // [P, 144, hidden]
            dbg_stats("vision.local.proj", &feats);
            let [_, n_query, _] = feats.dims();
            let feats = feats.reshape([p * n_query, hidden]);
            let sep = self.view_seperator.val().unsqueeze_dim(0);
            // Match the HF reference implementation injection order (yes, this differs from the
            // tokenizer's `<image>` token expansion order):
            // local(patches) -> global(base) -> view_seperator.
            Tensor::cat(vec![feats, global, sep], 0)
        } else {
            let sep = self.view_seperator.val().unsqueeze_dim(0);
            Tensor::cat(vec![global, sep], 0)
        };
        // Cast vision tokens to match the language embedding dtype for injection.
        let vision = vision.cast(embed_dtype);
        dbg_stats("vision.tokens", &vision);

        let [n_img, _] = vision.dims();

        // Collect target positions for image tokens.
        let mut img_pos: Vec<usize> = Vec::new();
        for (i, &is_img) in images_seq_mask.iter().enumerate() {
            if is_img {
                img_pos.push(i);
            }
        }
        anyhow::ensure!(
            img_pos.len() == n_img,
            "image token count mismatch (mask_true={}, vision_tokens={n_img})",
            img_pos.len()
        );

        // Fast path: the tokenizer expands `<image>` into a single contiguous block of placeholder
        // tokens, so we can use `slice_assign` instead of a huge scatter index tensor.
        //
        // This avoids:
        // - building an `[n_img, hidden]` integer index on CPU (hundreds of thousands of entries),
        // - uploading it to the GPU, and
        // - running a scatter kernel just to replace rows.
        if !img_pos.is_empty() && img_pos.windows(2).all(|w| w[1] == w[0].saturating_add(1)) {
            let start = img_pos[0];
            let end = start + n_img;
            anyhow::ensure!(
                end <= seq,
                "image token block out of bounds (start={start}, end={end}, seq={seq})"
            );
            let merged = base.slice_assign([start..end, 0..hidden], vision);
            dbg_stats("mm.merged", &merged);
            return Ok(merged.unsqueeze::<3>());
        }

        // Slow fallback: non-contiguous masks (shouldn't happen for our current prompt format).
        //
        // Burn currently only supports scatter add; we emulate assignment by zeroing first.
        let mut keep_mask: Vec<f32> = Vec::with_capacity(seq);
        for &is_img in images_seq_mask.iter() {
            keep_mask.push(if is_img { 0.0 } else { 1.0 });
        }
        let keep =
            Tensor::<B, 1>::from_data(burn::tensor::TensorData::new(keep_mask, [seq]), &device)
                .unsqueeze_dim::<2>(1) // [seq, 1]
                .cast(embed_dtype);
        let base = base * keep;

        let mut idx = Vec::with_capacity(n_img * hidden);
        for &p in img_pos.iter() {
            idx.extend(std::iter::repeat_n(p as i64, hidden));
        }
        let idx = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(idx, [n_img, hidden]),
            &device,
        );

        let merged = base.scatter(0, idx, vision, IndexingUpdateOp::Add);
        dbg_stats("mm.merged", &merged);
        Ok(merged.unsqueeze::<3>())
    }
}

#[derive(Module, Debug)]
pub struct DeepseekOcr2ForCausalLM<B: Backend> {
    pub model: DeepseekOcr2Model<B>,
    pub lm_head: Linear<B>,
}

impl<B: Backend> DeepseekOcr2ForCausalLM<B> {
    pub fn new(config: &DeepseekV2Config, device: &B::Device) -> Self {
        let model = DeepseekOcr2Model::new(config, device);
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

    /// Forward pass that only computes logits for the last token.
    ///
    /// This avoids the very expensive `[batch * seq, vocab]` projection on long prompts.
    #[allow(dead_code)]
    pub fn forward_last(
        &self,
        input_ids: Tensor<B, 2, Int>,
        caches: &mut Vec<Option<KvCache<B>>>,
    ) -> Tensor<B, 3> {
        let hidden = self.model.forward(input_ids, caches);
        self.logits_last_from_hidden(hidden)
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

    pub fn forward_embeds_last(
        &self,
        inputs_embeds: Tensor<B, 3>,
        caches: &mut Vec<Option<KvCache<B>>>,
    ) -> Tensor<B, 3> {
        let hidden = self.model.forward_embeds(inputs_embeds, caches);
        self.logits_last_from_hidden(hidden)
    }

    fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, hidden_size] = hidden.dims();
        let vocab = self.lm_head.weight.shape().dims::<2>()[1];
        self.lm_head
            .forward(hidden.reshape([batch * seq, hidden_size]))
            .reshape([batch, seq, vocab])
    }

    fn logits_last_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, hidden_size] = hidden.dims();
        debug_assert!(seq > 0);

        let vocab = self.lm_head.weight.shape().dims::<2>()[1];
        let last = hidden.slice([0..batch, (seq - 1)..seq, 0..hidden_size]); // [B, 1, H]
        self.lm_head
            .forward(last.reshape([batch, hidden_size]))
            .reshape([batch, 1, vocab])
    }
}
