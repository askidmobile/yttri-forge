//! Qwen3.5 single-head MTP runtime over thin mixed-Q8 GGUF.

use candle_core::{
    quantized::{gguf_file, QMatMul},
    DType, Device, IndexOp, Module, Result, Tensor,
};
use candle_nn::RmsNorm;
use std::{fs::File, path::Path, sync::Arc};

use super::{
    model_profile::{ModelProfile, MtpProfile},
    model_weights::ModelWeights,
    multimodal::MROPE_DIMENSION_SOURCES,
};

// hidden/heads/kv_heads — из MtpProfile (4B: 2560/16/4, Qwen3.8-27B: 5120/24/4).
// HEAD_DIM/ROPE_DIM/ROPE_BASE — инварианты семейства qwen35, валидируются
// в MtpProfile::read_and_validate.
const HEAD_DIM: usize = 256;
const ROPE_DIM: usize = 64;
const ROPE_BASE: f64 = 10_000_000.0;

#[derive(Clone)]
struct MtpKv {
    k: Tensor, // [1, kv, KV_HEADS, HEAD_DIM], F16
    v: Tensor,
    len: usize,
}

#[derive(Clone, Default)]
struct MtpSlot {
    kv: Option<MtpKv>,
    pending_target_hidden: Option<Tensor>, // normalized target hidden [1,HIDDEN]
}

#[derive(Clone)]
struct MtpSlotCheckpoint {
    kv: Option<MtpKv>,
    pending_target_hidden: Option<Tensor>,
}

struct MtpTransaction {
    checkpoint: MtpSlotCheckpoint,
}

pub struct Qwen35Mtp {
    profile: MtpProfile,
    device: Device,
    hnorm: RmsNorm,
    enorm: RmsNorm,
    eh_proj: QMatMul,
    attn_norm: RmsNorm,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o: QMatMul,
    ffn_norm: RmsNorm,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    head_norm: RmsNorm,
    shared_head: QMatMul,
    slots: Vec<MtpSlot>,
    transactions: Vec<Option<MtpTransaction>>,
}

impl Qwen35Mtp {
    pub fn load(
        path: &Path,
        device: Device,
        slots: usize,
        text_profile: &ModelProfile,
        shared_head: QMatMul,
    ) -> Result<Self> {
        let file = File::open(path).map_err(candle_core::Error::wrap)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(candle_core::Error::wrap)?;
        let mmap = Arc::new(mmap);
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let content = gguf_file::Content::read(&mut cursor)?;
        let profile = MtpProfile::read_and_validate(&content, text_profile)?;
        let rms_norm_eps = profile.rms_norm_eps;
        let data: &[u8] = &mmap;
        let qmat = |name: &str| -> Result<QMatMul> {
            QMatMul::from_qtensor(content.tensor_from_slice(data, name, &device)?)
        };
        let norm = |name: &str| -> Result<RmsNorm> {
            let weight = content
                .tensor_from_slice(data, name, &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .to_device(&device)?;
            Ok(RmsNorm::new(weight, rms_norm_eps))
        };
        let prefix = format!("blk.{}", profile.mtp_block);
        let mut mtp = Self {
            profile,
            device: device.clone(),
            hnorm: norm(&format!("{prefix}.nextn.hnorm.weight"))?,
            enorm: norm(&format!("{prefix}.nextn.enorm.weight"))?,
            eh_proj: qmat(&format!("{prefix}.nextn.eh_proj.weight"))?,
            attn_norm: norm(&format!("{prefix}.attn_norm.weight"))?,
            q: qmat(&format!("{prefix}.attn_q.weight"))?,
            k: qmat(&format!("{prefix}.attn_k.weight"))?,
            v: qmat(&format!("{prefix}.attn_v.weight"))?,
            q_norm: norm(&format!("{prefix}.attn_q_norm.weight"))?,
            k_norm: norm(&format!("{prefix}.attn_k_norm.weight"))?,
            o: qmat(&format!("{prefix}.attn_output.weight"))?,
            ffn_norm: norm(&format!("{prefix}.post_attention_norm.weight"))?,
            gate: qmat(&format!("{prefix}.ffn_gate.weight"))?,
            up: qmat(&format!("{prefix}.ffn_up.weight"))?,
            down: qmat(&format!("{prefix}.ffn_down.weight"))?,
            head_norm: norm(&format!("{prefix}.nextn.shared_head_norm.weight"))?,
            shared_head,
            slots: vec![MtpSlot::default(); slots],
            transactions: (0..slots).map(|_| None).collect(),
        };
        // Keep ownership construction after loaders stopped borrowing device.
        mtp.device = device;
        Ok(mtp)
    }

    pub fn profile(&self) -> &MtpProfile {
        &self.profile
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.check_slot(slot)?;
        self.slots[slot] = MtpSlot::default();
        self.transactions[slot] = None;
        Ok(())
    }

    pub fn begin(&mut self, slot: usize) -> Result<()> {
        self.check_slot(slot)?;
        if self.transactions[slot].is_some() {
            candle_core::bail!("MTP transaction already active for slot {slot}");
        }
        let state = &self.slots[slot];
        self.transactions[slot] = Some(MtpTransaction {
            checkpoint: MtpSlotCheckpoint {
                kv: state.kv.clone(),
                pending_target_hidden: state.pending_target_hidden.clone(),
            },
        });
        Ok(())
    }

    pub fn rollback(&mut self, slot: usize) -> Result<()> {
        self.check_slot(slot)?;
        if let Some(transaction) = self.transactions[slot].take() {
            self.slots[slot].kv = transaction.checkpoint.kv;
            self.slots[slot].pending_target_hidden =
                transaction.checkpoint.pending_target_hidden;
        }
        Ok(())
    }

    /// Keep only rows target actually verified and replace draft carry with
    /// target normalized hidden from last consumed input.
    pub fn commit(&mut self, slot: usize, verified_target_hidden: &[Tensor]) -> Result<()> {
        self.check_slot(slot)?;
        let transaction = self.transactions[slot]
            .take()
            .ok_or_else(|| candle_core::Error::Msg("MTP transaction is not active".into()))?;
        if verified_target_hidden.is_empty() {
            self.slots[slot].kv = transaction.checkpoint.kv;
            self.slots[slot].pending_target_hidden =
                transaction.checkpoint.pending_target_hidden;
            return Ok(());
        }
        let base = transaction
            .checkpoint
            .kv
            .as_ref()
            .map(|cache| cache.len)
            .unwrap_or(0);
        let committed = base
            .checked_add(verified_target_hidden.len())
            .ok_or_else(|| candle_core::Error::Msg("MTP committed length overflow".into()))?;
        let cache = self.slots[slot]
            .kv
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("MTP draft produced no KV state".into()))?;
        if committed > cache.k.dim(1)? {
            candle_core::bail!("MTP committed length exceeds draft KV");
        }
        cache.len = committed;
        self.slots[slot].pending_target_hidden = Some(
            verified_target_hidden
                .last()
                .expect("checked non-empty")
                .clone(),
        );
        Ok(())
    }

    /// Catch target prompt state up. `embeddings` include post-Vision rows;
    /// `target_hidden` are normalized target rows for same chunk.
    pub fn catch_up(
        &mut self,
        slot: usize,
        embeddings: &Tensor,
        target_hidden: &Tensor,
        start_pos: usize,
        rope_positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<()> {
        self.check_slot(slot)?;
        let (_, seq, hidden) = embeddings.dims3()?;
        if target_hidden.dims() != [1, seq, self.profile.hidden_size] || hidden != self.profile.hidden_size {
            candle_core::bail!("MTP catch-up shape mismatch");
        }
        let previous = self.slots[slot]
            .pending_target_hidden
            .clone()
            .unwrap_or(Tensor::zeros((1, self.profile.hidden_size), DType::F32, &self.device)?);
        let shifted = if seq == 1 {
            previous.unsqueeze(1)?
        } else {
            Tensor::cat(
                &[
                    &previous.unsqueeze(1)?,
                    &target_hidden.narrow(1, 0, seq - 1)?,
                ],
                1,
            )?
        };
        let _ = self.forward_rows(slot, embeddings, &shifted, start_pos, rope_positions)?;
        self.slots[slot].pending_target_hidden =
            Some(target_hidden.i((.., seq - 1, ..))?);
        Ok(())
    }

    pub fn draft(
        &mut self,
        slot: usize,
        first_token: u32,
        start_pos: usize,
        width: usize,
        target: &ModelWeights,
    ) -> Result<Vec<u32>> {
        self.check_slot(slot)?;
        if self.transactions[slot].is_none() {
            candle_core::bail!("MTP draft requires active transaction");
        }
        // P0.5b: черновой контур полностью на GPU — id предыдущего шага
        // остаётся CUDA-тензором (H2D один раз в начале, D2H один раз в конце).
        // Раньше каждый шаг делал D2H-sync (argmax→u32) + H2D (from_vec),
        // дважды осушая pipeline на итерацию: ~3 мс × width на раунд.
        let mut token_t =
            Tensor::from_vec(vec![first_token], (1, 1), &self.device)?;
        let mut hidden = self.slots[slot]
            .pending_target_hidden
            .clone()
            .ok_or_else(|| candle_core::Error::Msg("MTP target hidden is not initialized".into()))?;
        let mut tokens_gpu: Vec<Tensor> = Vec::with_capacity(width);
        for offset in 0..width {
            let embedding = target.embed_tokens(&token_t, &self.device)?;
            let mtp_hidden = self.forward_rows(
                slot,
                &embedding,
                &hidden.unsqueeze(1)?,
                start_pos + offset,
                None,
            )?;
            let pre_head = mtp_hidden.i((.., 0, ..))?;
            let normalized = self.head_norm.forward(&pre_head)?;
            let logits = self.shared_head.forward(&normalized)?;
            // Аргмакс на устройстве; тай-брейк на равных максимумах может
            // отличаться от host-argmax — на драфт не влияет (верифицирует target).
            if logits.device().is_cpu() {
                let id = argmax(&logits.to_dtype(DType::F32)?.to_vec2()?[0]);
                tokens_gpu.push(Tensor::from_vec(vec![id as u32], (1, 1), &self.device)?);
            } else {
                tokens_gpu.push(
                    logits
                        .argmax(candle_core::D::Minus1)?
                        .to_dtype(DType::U32)?
                        .reshape((1, 1))?,
                );
            }
            hidden = pre_head;
            token_t = tokens_gpu.last().unwrap().clone();
        }
        // Единственный D2H на весь draft.
        let mut out = Vec::with_capacity(width);
        for tg in &tokens_gpu {
            out.push(tg.flatten_all()?.to_vec1::<u32>()?[0]);
        }
        Ok(out)
    }

    fn forward_rows(
        &mut self,
        slot: usize,
        embeddings: &Tensor,
        target_hidden: &Tensor,
        start_pos: usize,
        rope_positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<Tensor> {
        let (_, seq, hidden) = embeddings.dims3()?;
        if target_hidden.dims() != [1, seq, hidden] || hidden != self.profile.hidden_size {
            candle_core::bail!("MTP input shape mismatch");
        }
        let e = self.enorm.forward(embeddings)?;
        let h = self.hnorm.forward(target_hidden)?;
        let projected = self.eh_proj.forward(&Tensor::cat(&[&e, &h], 2)?)?;
        let residual = &projected;
        let mixed = self.attention(
            slot,
            &self.attn_norm.forward(&projected)?,
            start_pos,
            rope_positions,
        )?;
        let after_attention = (mixed + residual)?;
        let ffn = self.ffn_norm.forward(&after_attention)?;
        let activated = self.gate.forward(&ffn)?.silu_mul_direct(&self.up.forward(&ffn)?)?;
        self.down.forward(&activated)? + after_attention
    }

    fn attention(
        &mut self,
        slot: usize,
        xs: &Tensor,
        start_pos: usize,
        rope_positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<Tensor> {
        let (_, seq, _) = xs.dims3()?;
        let qg = self.q.forward(xs)?.reshape((1, seq, self.profile.head_count, HEAD_DIM * 2))?;
        let q = qg
            .narrow(3, 0, HEAD_DIM)?
            .transpose(1, 2)?
            .contiguous()?;
        let gate = qg
            .narrow(3, HEAD_DIM, HEAD_DIM)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((1, seq, self.profile.kv_head_count, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((1, seq, self.profile.kv_head_count, HEAD_DIM))?
            .transpose(1, 2)?;
        let q = self
            .q_norm
            .forward(&q.flatten(0, 2)?)?
            .reshape((1, self.profile.head_count, seq, HEAD_DIM))?;
        let k = self
            .k_norm
            .forward(&k.flatten(0, 2)?)?
            .reshape((1, self.profile.kv_head_count, seq, HEAD_DIM))?;
        let (cos, sin) = match rope_positions {
            Some(positions) => mrope_tables(positions, &self.device)?,
            None => rope_tables(start_pos, seq, &self.device)?,
        };
        let q = apply_partial_rope(&q, &cos, &sin)?;
        let k = apply_partial_rope(&k, &cos, &sin)?;
        let k_new = k.transpose(1, 2)?.to_dtype(DType::F16)?.contiguous()?;
        let v_new = v.transpose(1, 2)?.to_dtype(DType::F16)?.contiguous()?;

        let past = self.slots[slot].kv.as_ref().map(|cache| cache.len).unwrap_or(0);
        let (k_all, v_all) = match self.slots[slot].kv.as_ref() {
            Some(cache) if cache.len > 0 => (
                Tensor::cat(&[&cache.k.narrow(1, 0, cache.len)?, &k_new], 1)?,
                Tensor::cat(&[&cache.v.narrow(1, 0, cache.len)?, &v_new], 1)?,
            ),
            _ => (k_new, v_new),
        };
        let total = past + seq;
        self.slots[slot].kv = Some(MtpKv {
            k: k_all.clone(),
            v: v_all.clone(),
            len: total,
        });

        let k = k_all.to_dtype(DType::F32)?.transpose(1, 2)?;
        let v = v_all.to_dtype(DType::F32)?.transpose(1, 2)?;
        let repeats = self.profile.head_count / self.profile.kv_head_count;
        let k = k
            .unsqueeze(2)?
            .broadcast_as((1, self.profile.kv_head_count, repeats, total, HEAD_DIM))?
            .contiguous()?
            .reshape((1, self.profile.head_count, total, HEAD_DIM))?;
        let v = v
            .unsqueeze(2)?
            .broadcast_as((1, self.profile.kv_head_count, repeats, total, HEAD_DIM))?
            .contiguous()?
            .reshape((1, self.profile.head_count, total, HEAD_DIM))?;
        let q_f32 = q.to_dtype(DType::F32)?.contiguous()?;
        let scores = (q_f32
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            * (1.0 / (HEAD_DIM as f64).sqrt()))?;
        let mask = (0..seq)
            .flat_map(|row| {
                (0..total).map(move |column| {
                    if column <= past + row {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, seq, total), &self.device)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let mixed = probs.contiguous()?.matmul(&v.contiguous()?)?;
        let mixed = (mixed * candle_nn::ops::sigmoid(&gate)?)?
            .transpose(1, 2)?
            .reshape((1, seq, self.profile.head_count * HEAD_DIM))?;
        self.o.forward(&mixed)
    }

    fn check_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.slots.len() {
            candle_core::bail!("MTP slot {slot} is out of range");
        }
        Ok(())
    }
}

fn rope_tables(start: usize, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut cos = Vec::with_capacity(len * ROPE_DIM / 2);
    let mut sin = Vec::with_capacity(len * ROPE_DIM / 2);
    for position in start..start + len {
        for i in 0..ROPE_DIM / 2 {
            let frequency = 1.0 / ROPE_BASE.powf((2 * i) as f64 / ROPE_DIM as f64);
            let value = position as f64 * frequency;
            cos.push(value.cos() as f32);
            sin.push(value.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, ROPE_DIM / 2), device)?,
        Tensor::from_vec(sin, (len, ROPE_DIM / 2), device)?,
    ))
}

fn mrope_tables(positions: &[Vec<u32>; 3], device: &Device) -> Result<(Tensor, Tensor)> {
    let len = positions[0].len();
    if positions.iter().any(|axis| axis.len() != len) {
        candle_core::bail!("MTP MRoPE position lengths differ");
    }
    let frequencies = (0..ROPE_DIM)
        .step_by(2)
        .map(|index| 1.0 / ROPE_BASE.powf(index as f64 / ROPE_DIM as f64))
        .collect::<Vec<_>>();
    let mut cos = Vec::with_capacity(len * frequencies.len());
    let mut sin = Vec::with_capacity(len * frequencies.len());
    for token in 0..len {
        for (&axis, &frequency) in MROPE_DIMENSION_SOURCES.iter().zip(&frequencies) {
            let angle = positions[axis as usize][token] as f64 * frequency;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, ROPE_DIM / 2), device)?,
        Tensor::from_vec(sin, (len, ROPE_DIM / 2), device)?,
    ))
}

fn apply_partial_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, _, dim) = xs.dims4()?;
    let rotated = candle_nn::rotary_emb::rope(
        &xs.narrow(3, 0, ROPE_DIM)?.contiguous()?,
        cos,
        sin,
    )?;
    Tensor::cat(&[&rotated, &xs.narrow(3, ROPE_DIM, dim - ROPE_DIM)?], 3)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}
