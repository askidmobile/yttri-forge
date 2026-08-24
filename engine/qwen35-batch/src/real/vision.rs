//! Qwen3.5 Vision encoder over separate mixed-Q8 GGUF.

use candle_core::{
    quantized::{gguf_file, QMatMul},
    DType, Device, IndexOp, Module, Result, Tensor, D,
};
use candle_nn::LayerNorm;
use std::{path::Path, sync::Arc};

use super::{model_profile::VisionProfile, multimodal::GridThw};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionBackend {
    Cpu,
    Metal,
    Cuda,
}

impl VisionBackend {
    fn for_device(device: &Device) -> Result<Self> {
        if device.is_cuda() {
            #[cfg(feature = "cuda")]
            return Ok(Self::Cuda);
            #[cfg(not(feature = "cuda"))]
            candle_core::bail!("Qwen3.5 Vision CUDA device requires cuda feature");
        }
        if device.is_metal() {
            #[cfg(feature = "metal")]
            return Ok(Self::Metal);
            #[cfg(not(feature = "metal"))]
            candle_core::bail!("Qwen3.5 Vision Metal device requires metal feature");
        }
        if device.is_cpu() {
            return Ok(Self::Cpu);
        }
        candle_core::bail!("unsupported Qwen3.5 Vision backend")
    }
}

struct QLinear {
    weight: QMatMul,
    bias: Tensor,
}

impl QLinear {
    fn load(
        content: &gguf_file::Content,
        data: &[u8],
        weight: &str,
        bias: &str,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: QMatMul::from_qtensor(content.tensor_from_slice(data, weight, device)?)?,
            bias: content
                .tensor_from_slice(data, bias, &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .to_device(device)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.weight.forward(xs)?.broadcast_add(&self.bias)
    }
}

struct PatchEmbed {
    temporal: [QMatMul; 2],
    bias: Tensor,
    input_per_slice: usize,
}

impl PatchEmbed {
    fn load(content: &gguf_file::Content, data: &[u8], device: &Device) -> Result<Self> {
        let load = |name: &str| -> Result<QMatMul> {
            let weight = content
                .tensor_from_slice(data, name, device)?
                .reshape((1024, 3 * 16 * 16))?;
            QMatMul::from_qtensor(weight)
        };
        Ok(Self {
            temporal: [load("v.patch_embd.weight")?, load("v.patch_embd.weight.1")?],
            bias: content
                .tensor_from_slice(data, "v.patch_embd.bias", &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .to_device(device)?,
            input_per_slice: 3 * 16 * 16,
        })
    }

    fn forward(&self, patches: &Tensor) -> Result<Tensor> {
        let (_, width) = patches.dims2()?;
        if width != self.input_per_slice * 2 {
            candle_core::bail!(
                "Vision patch width {width}, expected {}",
                self.input_per_slice * 2
            );
        }
        // Processor row order is [channel, temporal, y, x]. Temporal slices
        // are strided by channel, not contiguous halves.
        let rows = patches.dim(0)?;
        let patch_area = 16 * 16;
        let packed = patches.reshape((rows, 3, 2, patch_area))?;
        let first_input = packed
            .i((.., .., 0, ..))?
            .reshape((rows, self.input_per_slice))?;
        let second_input = packed
            .i((.., .., 1, ..))?
            .reshape((rows, self.input_per_slice))?;
        let first = self.temporal[0].forward(&first_input)?;
        let second = self.temporal[1].forward(&second_input)?;
        (first + second)?.broadcast_add(&self.bias)
    }
}

struct VisionAttention {
    qkv: QLinear,
    proj: QLinear,
    heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn load(
        content: &gguf_file::Content,
        data: &[u8],
        layer: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("v.blk.{layer}");
        Ok(Self {
            qkv: QLinear::load(
                content,
                data,
                &format!("{prefix}.attn_qkv.weight"),
                &format!("{prefix}.attn_qkv.bias"),
                device,
            )?,
            proj: QLinear::load(
                content,
                data,
                &format!("{prefix}.attn_out.weight"),
                &format!("{prefix}.attn_out.bias"),
                device,
            )?,
            heads: 16,
            head_dim: 64,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        backend: VisionBackend,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((seq_len, 3, self.heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let q = qkv.i(0)?.squeeze(0)?;
        let k = qkv.i(1)?.squeeze(0)?;
        let v = qkv.i(2)?.squeeze(0)?;
        let (q, k) = apply_rotary(&q, &k, cos, sin)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let output = if backend == VisionBackend::Cuda {
            #[cfg(feature = "cuda")]
            {
                let lengths: Vec<u32> = cu_seqlens
                    .iter()
                    .map(|&value| u32::try_from(value).map_err(candle_core::Error::wrap))
                    .collect::<Result<_>>()?;
                let lengths = Tensor::from_vec(lengths, (cu_seqlens.len(),), xs.device())?;
                let max_len = cu_seqlens
                    .windows(2)
                    .map(|window| window[1] - window[0])
                    .max()
                    .unwrap_or(0);
                candle_flash_attn::flash_attn_varlen(
                    &q.to_dtype(DType::F16)?.contiguous()?,
                    &k.to_dtype(DType::F16)?.contiguous()?,
                    &v.to_dtype(DType::F16)?.contiguous()?,
                    &lengths,
                    &lengths,
                    max_len,
                    max_len,
                    scale as f32,
                    false,
                )?
                .to_dtype(xs.dtype())?
            }
            #[cfg(not(feature = "cuda"))]
            unreachable!()
        } else {
            let mut outputs = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
            for window in cu_seqlens.windows(2) {
                let len = window[1] - window[0];
                if len == 0 {
                    candle_core::bail!("Vision attention contains empty sequence");
                }
                let q = q
                    .narrow(0, window[0], len)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((self.heads, len, self.head_dim))?;
                let k = k
                    .narrow(0, window[0], len)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((self.heads, len, self.head_dim))?;
                let v = v
                    .narrow(0, window[0], len)?
                    .transpose(0, 1)?
                    .contiguous()?
                    .reshape((self.heads, len, self.head_dim))?;
                let weights = candle_nn::ops::softmax_last_dim(
                    &(q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?,
                )?;
                outputs.push(
                    weights
                        .matmul(&v)?
                        .transpose(0, 1)?
                        .contiguous()?
                        .reshape((len, self.heads * self.head_dim))?,
                );
            }
            Tensor::cat(&outputs, 0)?
        };
        self.proj
            .forward(&output.reshape((seq_len, self.heads * self.head_dim))?)
    }
}

struct VisionMlp {
    up: QLinear,
    down: QLinear,
}

impl VisionMlp {
    fn load(
        content: &gguf_file::Content,
        data: &[u8],
        layer: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("v.blk.{layer}");
        Ok(Self {
            up: QLinear::load(
                content,
                data,
                &format!("{prefix}.ffn_up.weight"),
                &format!("{prefix}.ffn_up.bias"),
                device,
            )?,
            down: QLinear::load(
                content,
                data,
                &format!("{prefix}.ffn_down.weight"),
                &format!("{prefix}.ffn_down.bias"),
                device,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down.forward(&self.up.forward(xs)?.gelu()?)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn load(
        content: &gguf_file::Content,
        data: &[u8],
        layer: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("v.blk.{layer}");
        Ok(Self {
            norm1: load_layer_norm(content, data, &format!("{prefix}.ln1"), device)?,
            norm2: load_layer_norm(content, data, &format!("{prefix}.ln2"), device)?,
            attention: VisionAttention::load(content, data, layer, device)?,
            mlp: VisionMlp::load(content, data, layer, device)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        backend: VisionBackend,
    ) -> Result<Tensor> {
        let xs = (xs
            + self
                .attention
                .forward(&self.norm1.forward(xs)?, cu_seqlens, cos, sin, backend)?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
    }
}

struct PatchMerger {
    norm: LayerNorm,
    fc1: QLinear,
    fc2: QLinear,
}

impl PatchMerger {
    fn load(content: &gguf_file::Content, data: &[u8], device: &Device) -> Result<Self> {
        Ok(Self {
            norm: load_layer_norm(content, data, "v.post_ln", device)?,
            fc1: QLinear::load(content, data, "mm.0.weight", "mm.0.bias", device)?,
            fc2: QLinear::load(content, data, "mm.2.weight", "mm.2.bias", device)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if !seq_len.is_multiple_of(4) {
            candle_core::bail!("Vision sequence length {seq_len} is not divisible by 4");
        }
        let xs = self.norm.forward(xs)?.reshape((seq_len / 4, 4096))?;
        self.fc2.forward(&self.fc1.forward(&xs)?.gelu_erf()?)
    }
}

pub struct Qwen35Vision {
    _mmap: Arc<memmap2::Mmap>,
    profile: VisionProfile,
    backend: VisionBackend,
    patch_embed: PatchEmbed,
    position_embed: Tensor,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    device: Device,
}

impl Qwen35Vision {
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        Self::load_with_policy(path, device, true)
    }

    pub fn load_reference(path: &Path, device: Device) -> Result<Self> {
        Self::load_with_policy(path, device, false)
    }

    fn load_with_policy(path: &Path, device: Device, production_q8: bool) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { memmap2::MmapOptions::new().map(&file)? });
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let content = gguf_file::Content::read(&mut cursor)?;
        let profile = if production_q8 {
            VisionProfile::read_and_validate(&content)?
        } else {
            VisionProfile::read_reference(&content)?
        };
        let backend = VisionBackend::for_device(&device)?;
        let data: &[u8] = &mmap;
        let patch_embed = PatchEmbed::load(&content, data, &device)?;
        let position_embed = content
            .tensor_from_slice(data, "v.position_embd.weight", &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_device(&device)?;
        let blocks = (0..profile.block_count)
            .map(|layer| VisionBlock::load(&content, data, layer, &device))
            .collect::<Result<_>>()?;
        let merger = PatchMerger::load(&content, data, &device)?;
        Ok(Self {
            _mmap: mmap,
            profile,
            backend,
            patch_embed,
            position_embed,
            blocks,
            merger,
            device,
        })
    }

    pub fn profile(&self) -> &VisionProfile {
        &self.profile
    }

    pub fn forward(&self, patches: &Tensor, grids: &[GridThw]) -> Result<Tensor> {
        validate_grids(patches, grids)?;
        let mut hidden = self
            .patch_embed
            .forward(&patches.to_device(&self.device)?)?;
        hidden = (hidden + interpolate_positions(&self.position_embed, grids, 2)?)?;
        let position_ids = vision_position_ids(grids, 2)?;
        let (cos, sin) = rotary_embeddings(
            &position_ids,
            self.blocks[0].attention.head_dim,
            &self.device,
        )?;
        let cu_seqlens = attention_seqlens(grids)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &cu_seqlens, &cos, &sin, self.backend)?;
        }
        self.merger.forward(&hidden)
    }
}

fn load_layer_norm(
    content: &gguf_file::Content,
    data: &[u8],
    prefix: &str,
    device: &Device,
) -> Result<LayerNorm> {
    let load = |suffix: &str| -> Result<Tensor> {
        content
            .tensor_from_slice(data, &format!("{prefix}.{suffix}"), &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .to_device(device)
    };
    Ok(LayerNorm::new(load("weight")?, load("bias")?, 1e-6))
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let width = xs.dim(D::Minus1)?;
    Tensor::cat(
        &[
            &xs.narrow(D::Minus1, width / 2, width - width / 2)?.neg()?,
            &xs.narrow(D::Minus1, 0, width / 2)?,
        ],
        D::Minus1,
    )
}

fn apply_rotary(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let cos = cos.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?;
    let sin = sin.unsqueeze(D::Minus2)?.to_dtype(DType::F32)?;
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    Ok((
        (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?.to_dtype(q_dtype)?,
        (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?.to_dtype(k_dtype)?,
    ))
}

fn validate_grids(patches: &Tensor, grids: &[GridThw]) -> Result<()> {
    if grids.is_empty() {
        candle_core::bail!("Vision grids cannot be empty");
    }
    let expected = grids.iter().try_fold(0usize, |total, grid| {
        let count = grid
            .t
            .checked_mul(grid.h)
            .and_then(|value| value.checked_mul(grid.w))
            .ok_or_else(|| candle_core::Error::Msg("Vision patch count overflow".into()))?;
        total
            .checked_add(count)
            .ok_or_else(|| candle_core::Error::Msg("Vision patch count overflow".into()))
    })?;
    let (actual, width) = patches.dims2()?;
    if actual != expected || width != 3 * 2 * 16 * 16 {
        candle_core::bail!(
            "Vision patches shape {:?}, expected [{expected}, {}]",
            patches.dims(),
            3 * 2 * 16 * 16
        );
    }
    for grid in grids {
        if !grid.h.is_multiple_of(2) || !grid.w.is_multiple_of(2) {
            candle_core::bail!("Vision grid must be divisible by merge size 2: {grid:?}");
        }
    }
    Ok(())
}

fn attention_seqlens(grids: &[GridThw]) -> Result<Vec<usize>> {
    let mut result = vec![0];
    let mut total = 0usize;
    for grid in grids {
        let area = grid
            .h
            .checked_mul(grid.w)
            .ok_or_else(|| candle_core::Error::Msg("Vision grid area overflow".into()))?;
        for _ in 0..grid.t {
            total = total
                .checked_add(area)
                .ok_or_else(|| candle_core::Error::Msg("Vision sequence overflow".into()))?;
            result.push(total);
        }
    }
    Ok(result)
}

fn vision_position_ids(grids: &[GridThw], merge: usize) -> Result<Vec<[usize; 2]>> {
    let mut positions = Vec::new();
    for grid in grids {
        let mut frame = Vec::with_capacity(grid.h * grid.w);
        for block_h in 0..grid.h / merge {
            for block_w in 0..grid.w / merge {
                for inner_h in 0..merge {
                    for inner_w in 0..merge {
                        frame.push([block_h * merge + inner_h, block_w * merge + inner_w]);
                    }
                }
            }
        }
        for _ in 0..grid.t {
            positions.extend_from_slice(&frame);
        }
    }
    Ok(positions)
}

fn rotary_embeddings(
    positions: &[[usize; 2]],
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let rotary_dim = head_dim / 2;
    let inv: Vec<f32> = (0..rotary_dim)
        .step_by(2)
        .map(|index| 1.0 / 10000f32.powf(index as f32 / rotary_dim as f32))
        .collect();
    let mut values = Vec::with_capacity(positions.len() * rotary_dim);
    for position in positions {
        for axis in position {
            values.extend(inv.iter().map(|value| *axis as f32 * value));
        }
    }
    let rotary = Tensor::from_vec(values, (positions.len(), rotary_dim), device)?;
    let embedding = Tensor::cat(&[&rotary, &rotary], 1)?;
    Ok((embedding.cos()?, embedding.sin()?))
}

fn interpolate_positions(
    position_embed: &Tensor,
    grids: &[GridThw],
    merge: usize,
) -> Result<Tensor> {
    let side = 48usize;
    let mut outputs = Vec::with_capacity(grids.len());
    for grid in grids {
        let positions = vision_position_ids(
            &[GridThw {
                t: 1,
                h: grid.h,
                w: grid.w,
            }],
            merge,
        )?;
        let mut indices = Vec::with_capacity(positions.len() * 4);
        let mut weights = Vec::with_capacity(positions.len() * 4);
        for [row, col] in positions {
            let (rows, row_weights) = linear_taps(row, grid.h, side);
            let (cols, col_weights) = linear_taps(col, grid.w, side);
            for row_axis in 0..2 {
                for col_axis in 0..2 {
                    indices.push((rows[row_axis] * side + cols[col_axis]) as u32);
                    weights.push(row_weights[row_axis] * col_weights[col_axis]);
                }
            }
        }
        let indices = Tensor::from_vec(indices, (positions_len(grid), 4), position_embed.device())?;
        let weights = Tensor::from_vec(
            weights,
            (positions_len(grid), 4, 1),
            position_embed.device(),
        )?;
        let frame = position_embed
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((positions_len(grid), 4, 1024))?;
        let frame = frame.broadcast_mul(&weights)?.sum(1)?;
        outputs.push(frame.repeat((grid.t, 1))?);
    }
    Tensor::cat(&outputs, 0)
}

fn positions_len(grid: &GridThw) -> usize {
    grid.h * grid.w
}

fn linear_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    let source = if size == 1 {
        0.0
    } else {
        index as f32 * (side - 1) as f32 / (size - 1) as f32
    };
    let floor = source.floor() as usize;
    let ceil = source.ceil().min((side - 1) as f32) as usize;
    let fraction = source - floor as f32;
    ([floor, ceil], [1.0 - fraction, fraction])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_patch_slices_keep_channel_order() -> Result<()> {
        let rows = 1;
        let area = 4;
        let values: Vec<f32> = (0..3)
            .flat_map(|channel| {
                (0..2).flat_map(move |temporal| {
                    (0..area).map(move |offset| (channel * 100 + temporal * 10 + offset) as f32)
                })
            })
            .collect();
        let packed = Tensor::from_vec(values, (rows, 3, 2, area), &Device::Cpu)?;
        assert_eq!(
            packed.i((.., .., 0, ..))?.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 1., 2., 3., 100., 101., 102., 103., 200., 201., 202., 203.]
        );
        assert_eq!(
            packed.i((.., .., 1, ..))?.flatten_all()?.to_vec1::<f32>()?,
            vec![10., 11., 12., 13., 110., 111., 112., 113., 210., 211., 212., 213.]
        );
        Ok(())
    }

    #[test]
    fn positions_follow_merge_block_order() -> Result<()> {
        assert_eq!(
            vision_position_ids(&[GridThw { t: 1, h: 2, w: 4 }], 2)?,
            vec![
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3]
            ]
        );
        assert_eq!(
            attention_seqlens(&[GridThw { t: 2, h: 2, w: 4 }])?,
            vec![0, 8, 16]
        );
        Ok(())
    }
}
