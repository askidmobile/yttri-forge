//! Quantized Gemma 4 text decoder for GGUF.
//!
//! This is the GGUF/QTensor counterpart of `models::gemma4::text`.
//! It implements mixed sliding/global attention, shared KV layers, per-layer
//! embeddings, GELU gated MLPs, layer scaling, and final-logit softcapping.

use std::sync::Arc;

use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, QMatMul, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};

#[derive(Clone, Debug)]
struct QuantLinear(QMatMul);

impl QuantLinear {
    fn new(weight: QTensor) -> Result<Self> {
        Ok(Self(QMatMul::from_qtensor(weight)?))
    }

    fn from_arc(weight: Arc<QTensor>) -> Result<Self> {
        Ok(Self(QMatMul::from_arc(weight)?))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }

    fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        self.0.embedding(ids)
    }
}

fn v_norm(v: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = v.dtype();
    let v = v.to_dtype(DType::F32)?;
    let rms = (v.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    v.broadcast_div(&rms)?.to_dtype(dtype)
}

#[derive(Clone, Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
    head_dim: usize,
}

impl RotaryEmbedding {
    fn from_freqs(
        freqs_tensor: &Tensor,
        head_dim: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let rotary_dim = freqs_tensor.elem_count();
        let half = rotary_dim / 2;
        let inv_freq = freqs_tensor.flatten_all()?.narrow(0, 0, half)?.reshape((1, half))?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.matmul(&inv_freq.to_device(device)?)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
            head_dim,
        })
    }

    fn new_standard(
        head_dim: usize,
        rotary_dim: usize,
        theta: f64,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let half = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
            head_dim,
        })
    }

    fn apply(&self, xs: &Tensor, index_pos: usize) -> Result<Tensor> {
        let seq_len = xs.dim(2)?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        if self.rotary_dim == self.head_dim {
            candle_nn::rotary_emb::rope(&xs.contiguous()?, &cos, &sin)
        } else {
            // Partial RoPE: вращаем только первые rotary_dim измерений
            let xs_rot = xs.narrow(3, 0, self.rotary_dim)?.contiguous()?;
            let xs_pass = xs.narrow(3, self.rotary_dim, self.head_dim - self.rotary_dim)?;
            let xs_rotated = candle_nn::rotary_emb::rope(&xs_rot, &cos, &sin)?;
            Tensor::cat(&[&xs_rotated, &xs_pass], 3)
        }
    }
}

#[derive(Clone, Debug)]
enum KvCache {
    Full(candle_nn::kv_cache::KvCache),
    Sliding(candle_nn::kv_cache::RotatingKvCache),
}

impl KvCache {
    fn mask(&self, seq_len: usize, device: &Device, dtype: DType) -> Result<Option<Tensor>> {
        match self {
            Self::Full(cache) => {
                if seq_len == 1 {
                    Ok(None)
                } else {
                    Ok(Some(crate::utils::build_additive_causal_mask(
                        seq_len,
                        cache.current_seq_len(),
                        None,
                        device,
                        dtype,
                    )?))
                }
            }
            Self::Sliding(cache) => cache
                .attn_mask(seq_len, device)?
                .map(|mask| additive_mask(&mask, dtype))
                .transpose(),
        }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Full(cache) => cache.append(k, v),
            Self::Sliding(cache) => cache.append(k, v),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Full(cache) => cache.reset(),
            Self::Sliding(cache) => cache.reset(),
        }
    }
}

fn additive_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
    let zeros = Tensor::zeros(mask.shape(), dtype, mask.device())?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, mask.device())?
        .to_dtype(dtype)?
        .broadcast_as(mask.shape())?;
    mask.where_cond(&neg_inf, &zeros)
}

#[derive(Clone, Debug)]
struct SharedAttention {
    k: Tensor,
    v: Tensor,
    mask: Option<Tensor>,
}

#[derive(Clone, Debug)]
struct Attention {
    q_proj: QuantLinear,
    k_proj: Option<QuantLinear>,
    v_proj: Option<QuantLinear>,
    o_proj: QuantLinear,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
    cache: Option<KvCache>,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        index_pos: usize,
        shared: Option<&SharedAttention>,
    ) -> Result<(Tensor, Option<SharedAttention>)> {
        let (batch, seq_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let q = self.rotary.apply(&q, index_pos)?;

        let own_kv = match (&self.k_proj, &self.v_proj, &self.k_norm, &mut self.cache) {
            (Some(k_proj), Some(v_proj), Some(k_norm), Some(cache)) => {
                let k = k_proj
                    .forward(xs)?
                    .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v_proj
                    .forward(xs)?
                    .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?;
                let k = k_norm.forward(&k.contiguous()?)?;
                let k = self.rotary.apply(&k, index_pos)?;
                let v = v_norm(&v, 1e-6)?;
                let mask = cache.mask(seq_len, xs.device(), q.dtype())?;
                let (k, v) = cache.append(&k, &v)?;
                Some(SharedAttention { k, v, mask })
            }
            (None, None, None, None) => None,
            _ => candle::bail!("incomplete Gemma 4 KV projection/cache configuration"),
        };
        let kv = own_kv.as_ref().or(shared).ok_or_else(|| {
            candle::Error::Msg("Gemma 4 shared-KV layer has no source cache".into())
        })?;

        let repeats = self.num_heads / self.num_kv_heads;
        let k = crate::utils::repeat_kv(kv.k.clone(), repeats)?.contiguous()?;
        let v = crate::utils::repeat_kv(kv.v.clone(), repeats)?.contiguous()?;

        // Gemma 4 metadata defines attention scale as 1.0, not 1/sqrt(head_dim).
        let mut weights = q.matmul(&k.transpose(2, 3)?)?;
        if let Some(mask) = &kv.mask {
            weights = weights.broadcast_add(mask)?;
        }
        let weights = candle_nn::ops::softmax_last_dim(&weights)?;
        let output = weights.matmul(&v)?.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;
        Ok((self.o_proj.forward(&output)?, own_kv))
    }

    fn reset(&mut self) {
        if let Some(cache) = &mut self.cache {
            cache.reset();
        }
    }
}

#[derive(Clone, Debug)]
struct Mlp {
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?.gelu()?;
        self.down.forward(&(gate * self.up.forward(xs)?)?)
    }
}

#[derive(Clone, Debug)]
struct Layer {
    attention: Attention,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: Mlp,
    per_layer_inp_gate: QuantLinear,
    per_layer_proj: QuantLinear,
    per_layer_post_norm: RmsNorm,
    output_scale: Tensor,
    shared_source: Option<usize>,
}

impl Layer {
    fn forward(
        &mut self,
        xs: &Tensor,
        per_layer_input: &Tensor,
        index_pos: usize,
        shared: Option<&SharedAttention>,
    ) -> Result<(Tensor, Option<SharedAttention>)> {
        let residual = xs;
        let normalized = self.attention_norm.forward(xs)?;
        let (attention, own_kv) = self.attention.forward(&normalized, index_pos, shared)?;
        let attention = self.post_attention_norm.forward(&attention)?;
        let xs = (residual + attention)?;

        let residual = &xs;
        let ffn = self.ffn_norm.forward(&xs)?;
        let ffn = self.mlp.forward(&ffn)?;
        let ffn = self.post_ffn_norm.forward(&ffn)?;
        let mut xs = (residual + ffn)?;

        let residual = &xs;
        let gate = self.per_layer_inp_gate.forward(&xs)?.gelu()?;
        let per_layer = self.per_layer_proj.forward(&(gate * per_layer_input)?)?;
        let per_layer = self.per_layer_post_norm.forward(&per_layer)?;
        xs = (residual + per_layer)?;
        xs = xs.broadcast_mul(&self.output_scale)?;
        Ok((xs, own_kv))
    }
}

#[derive(Clone, Debug)]
pub struct ModelWeights {
    token_embedding: QuantLinear,
    per_layer_token_embedding: QuantLinear,
    per_layer_model_proj: QuantLinear,
    per_layer_proj_norm: RmsNorm,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QuantLinear,
    hidden_size: usize,
    per_layer_size: usize,
    final_logit_softcap: Option<f64>,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let arch = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(String::as_str)
            .unwrap_or("");
        if arch != "gemma4" {
            candle::bail!("quantized Gemma 4 loader expected architecture gemma4, got {arch:?}")
        }
        let get_u32 = |key: &str| -> Result<usize> {
            Ok(ct
                .metadata
                .get(key)
                .ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata {key}")))?
                .to_u32()? as usize)
        };
        let get_f32 = |key: &str| -> Result<f32> {
            ct.metadata
                .get(key)
                .ok_or_else(|| candle::Error::Msg(format!("missing GGUF metadata {key}")))?
                .to_f32()
        };

        let layer_count = get_u32("gemma4.block_count")?;
        let context_length = get_u32("gemma4.context_length")?;
        let hidden_size = get_u32("gemma4.embedding_length")?;
        let num_heads = get_u32("gemma4.attention.head_count")?;
        let num_kv_heads = get_u32("gemma4.attention.head_count_kv")?;
        let global_head_dim = get_u32("gemma4.attention.key_length")?;
        let sliding_head_dim = get_u32("gemma4.attention.key_length_swa")?;
        let sliding_window = get_u32("gemma4.attention.sliding_window")?;
        let shared_kv_layers = get_u32("gemma4.attention.shared_kv_layers")?;
        let per_layer_size = get_u32("gemma4.embedding_length_per_layer_input")?;
        let eps = get_f32("gemma4.attention.layer_norm_rms_epsilon")? as f64;
        let global_theta = get_f32("gemma4.rope.freq_base")? as f64;
        let sliding_theta = get_f32("gemma4.rope.freq_base_swa")? as f64;
        let final_logit_softcap = ct
            .metadata
            .get("gemma4.final_logit_softcapping")
            .and_then(|v| v.to_f32().ok())
            .map(|v| v as f64);
        let pattern = ct
            .metadata
            .get("gemma4.attention.sliding_window_pattern")
            .ok_or_else(|| candle::Error::Msg("missing Gemma 4 sliding-window pattern".into()))?
            .to_vec()?
            .iter()
            .map(gguf_file::Value::to_bool)
            .collect::<Result<Vec<_>>>()?;
        if pattern.len() != layer_count {
            candle::bail!(
                "Gemma 4 sliding-window pattern has {} entries, expected {layer_count}",
                pattern.len()
            )
        }
        let own_kv_layers = layer_count.checked_sub(shared_kv_layers).ok_or_else(|| {
            candle::Error::Msg("Gemma 4 shared_kv_layers exceeds block_count".into())
        })?;
        let source_for = |sliding: bool| -> Result<usize> {
            (0..own_kv_layers)
                .rev()
                .find(|&i| pattern[i] == sliding)
                .ok_or_else(|| {
                    candle::Error::Msg("Gemma 4 has no matching shared-KV source".into())
                })
        };
        let sliding_source = source_for(true)?;
        let global_source = source_for(false)?;

        let global_rotary = if let Ok(freqs_qt) = ct.tensor(reader, "rope_freqs.weight", device) {
            let freqs_t = freqs_qt.dequantize(device)?;
            RotaryEmbedding::from_freqs(&freqs_t, global_head_dim, context_length, device)?
        } else {
            // Fallback: 25% partial rotary for Gemma 4 full_attention
            RotaryEmbedding::new_standard(
                global_head_dim,
                global_head_dim / 4,
                global_theta,
                context_length,
                device,
            )?
        };
        let sliding_rotary = RotaryEmbedding::new_standard(
            sliding_head_dim,
            sliding_head_dim,
            sliding_theta,
            context_length,
            device,
        )?;

        let token_q = Arc::new(ct.tensor(reader, "token_embd.weight", device)?);
        let token_embedding = QuantLinear::from_arc(token_q.clone())?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(weight) => QuantLinear::new(weight)?,
            Err(_) => QuantLinear::from_arc(token_q)?,
        };
        let per_layer_token_embedding =
            QuantLinear::new(ct.tensor(reader, "per_layer_token_embd.weight", device)?)?;
        let per_layer_model_proj =
            QuantLinear::new(ct.tensor(reader, "per_layer_model_proj.weight", device)?)?;
        let per_layer_proj_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "per_layer_proj_norm.weight", device)?,
            eps,
        )?;
        let norm = RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, eps)?;

        let mut layers = Vec::with_capacity(layer_count);
        for index in 0..layer_count {
            let p = format!("blk.{index}");
            let sliding = pattern[index];
            let head_dim = if sliding {
                sliding_head_dim
            } else {
                global_head_dim
            };
            let owns_kv = index < own_kv_layers;
            let k_proj = owns_kv
                .then(|| ct.tensor(reader, &format!("{p}.attn_k.weight"), device))
                .transpose()?
                .map(QuantLinear::new)
                .transpose()?;
            let v_proj = owns_kv
                .then(|| ct.tensor(reader, &format!("{p}.attn_v.weight"), device))
                .transpose()?
                .map(QuantLinear::new)
                .transpose()?;
            let k_norm = owns_kv
                .then(|| ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device))
                .transpose()?
                .map(|weight| RmsNorm::from_qtensor(weight, eps))
                .transpose()?;
            let cache = owns_kv.then(|| {
                if sliding {
                    KvCache::Sliding(candle_nn::kv_cache::RotatingKvCache::new(2, sliding_window))
                } else {
                    // KvCache grows by this initial chunk; reserving full 131K here
                    // allocates several GiB before first-token generation.
                    KvCache::Full(candle_nn::kv_cache::KvCache::new(
                        2,
                        context_length.min(2048),
                    ))
                }
            });
            let shared_source = (!owns_kv).then_some(if sliding {
                sliding_source
            } else {
                global_source
            });

            layers.push(Layer {
                attention: Attention {
                    q_proj: QuantLinear::new(ct.tensor(
                        reader,
                        &format!("{p}.attn_q.weight"),
                        device,
                    )?)?,
                    k_proj,
                    v_proj,
                    o_proj: QuantLinear::new(ct.tensor(
                        reader,
                        &format!("{p}.attn_output.weight"),
                        device,
                    )?)?,
                    q_norm: RmsNorm::from_qtensor(
                        ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                        eps,
                    )?,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    rotary: if sliding {
                        sliding_rotary.clone()
                    } else {
                        global_rotary.clone()
                    },
                    cache,
                },
                attention_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                    eps,
                )?,
                post_attention_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                    eps,
                )?,
                ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                    eps,
                )?,
                post_ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                    eps,
                )?,
                mlp: Mlp {
                    gate: QuantLinear::new(ct.tensor(
                        reader,
                        &format!("{p}.ffn_gate.weight"),
                        device,
                    )?)?,
                    up: QuantLinear::new(ct.tensor(
                        reader,
                        &format!("{p}.ffn_up.weight"),
                        device,
                    )?)?,
                    down: QuantLinear::new(ct.tensor(
                        reader,
                        &format!("{p}.ffn_down.weight"),
                        device,
                    )?)?,
                },
                per_layer_inp_gate: QuantLinear::new(ct.tensor(
                    reader,
                    &format!("{p}.inp_gate.weight"),
                    device,
                )?)?,
                per_layer_proj: QuantLinear::new(ct.tensor(
                    reader,
                    &format!("{p}.proj.weight"),
                    device,
                )?)?,
                per_layer_post_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_norm.weight"), device)?,
                    eps,
                )?,
                output_scale: ct
                    .tensor(reader, &format!("{p}.layer_output_scale.weight"), device)?
                    .dequantize(device)?,
                shared_source,
            });
        }

        Ok(Self {
            token_embedding,
            per_layer_token_embedding,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            norm,
            output,
            hidden_size,
            per_layer_size,
            final_logit_softcap,
        })
    }

    pub fn forward(&mut self, ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (batch, seq_len) = ids.dims2()?;
        let mut xs = (self.token_embedding.embedding(ids)? * (self.hidden_size as f64).sqrt())?;

        let per_layer_tokens =
            (self.per_layer_token_embedding.embedding(ids)? * (self.per_layer_size as f64).sqrt())?;
        let projected =
            (self.per_layer_model_proj.forward(&xs)? / (self.hidden_size as f64).sqrt())?;
        let projected = self.per_layer_proj_norm.forward(&projected.reshape((
            batch,
            seq_len,
            self.layers.len(),
            self.per_layer_size,
        ))?)?;
        let per_layer = ((projected
            + per_layer_tokens.reshape((
                batch,
                seq_len,
                self.layers.len(),
                self.per_layer_size,
            ))?)?
            / 2f64.sqrt())?;

        let mut shared_kv: Vec<Option<SharedAttention>> = vec![None; self.layers.len()];
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let layer_input = per_layer.narrow(2, index, 1)?.squeeze(2)?;
            let shared = layer
                .shared_source
                .and_then(|source| shared_kv[source].as_ref());
            let (next, own_kv) = layer.forward(&xs, &layer_input, index_pos, shared)?;
            if own_kv.is_some() {
                shared_kv[index] = own_kv;
            }
            xs = next;
        }

        let xs = self.norm.forward(&xs)?;
        let xs = xs.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        let logits = self.output.forward(&xs)?;
        match self.final_logit_softcap {
            Some(cap) => Ok(((logits / cap)?.tanh()? * cap)?),
            None => Ok(logits),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.attention.reset();
        }
    }
}
