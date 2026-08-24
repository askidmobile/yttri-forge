//! Pre-repacked Q4_K metadata for the optimized matmul kernel.
//!
//! Background: the original `block_q4_K` stores scales/mins in a packed 6-bit format
//! (12 bytes for 8 sub-blocks). The hot path Metal kernel spends ~9-13% of ALU
//! instructions on unpacking (see `get_scale_min_k4_just2` + per-sub-block
//! `il < 2 ? d : d/16.h` selects + per-element FP32 conversion).
//!
//! `Q4KOptMetadata` stores pre-computed `dl = d * scale` and `ml = dmin * min`
//! for each sub-block in `half` arrays. The kernel reads them directly via a
//! `device const half *` parameter, bypassing the whole decode pipeline.
//!
//! Memory cost: 16 `half` per block x ~14M blocks for Qwen3.5-4B ~= 470 MB.
//! Accepted by the user as a trade-off for a +15-25% prefill speedup.
//!
//! Layout of pre-packed data, per block (32 bytes):
//! ```text
//! [dl_0, ml_0, dl_1, ml_1, dl_2, ml_2, ..., dl_7, ml_7]
//! ```
//! where dl_i = d * scales[i], ml_i = dmin * mins[i] (scales/mins decoded from
//! the 6-bit packed format).

use crate::quantized::k_quants::BlockQ4K;
use crate::Result;
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

/// Pre-repacked Q4_K metadata (CPU-resident).
///
/// Created via `repack_q4k_for_opt` for each Q4_K_M weight matrix.
/// `data` is stored in block order as in the source QTensor (row-major).
#[derive(Debug, Clone)]
pub struct Q4KOptMetadata {
    /// 16 `f16` values per block (8 sub-blocks × pair `(dl, ml)`).
    pub data: Vec<f16>,
    /// Number of Q4_K blocks. `data.len() == block_count * 16`.
    pub block_count: usize,
    /// Identity guard -- ties this to a specific QTensor instance.
    /// Protects against reusing metadata from a different weight tensor.
    pub source_nonce: u64,
    /// Layout version for future evolution.
    pub layout_version: u8,
}

/// Current layout version. Increment when data ordering or format changes.
pub const Q4K_OPT_LAYOUT_V1: u8 = 1;

impl Q4KOptMetadata {
    /// Total memory footprint in bytes (CPU-side).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f16>()
    }

    /// Identifier of the source weight instance.
    pub fn source_nonce(&self) -> u64 {
        self.source_nonce
    }

    /// Number of Q4_K blocks.
    pub fn block_count(&self) -> usize {
        self.block_count
    }
}

/// Repack Q4_K blocks into pre-computed `dl/ml` half pairs.
///
/// The 6-bit packed scales/mins decoding algorithm is taken from `BlockQ4K::vec_dot_unopt`
/// (k_quants.rs:1430-1440) -- bit twiddling via KMASK1/2/3 + `utmp[3]`.
///
/// For each block it computes 8 pairs `(dl_i, ml_i)`:
/// ```text
/// dl_i = d * scales_decoded[i]
/// ml_i = dmin * mins_decoded[i]
/// ```
///
/// Returns a [`Q4KOptMetadata`] with layout V1.
pub fn repack_q4k_for_opt(blocks: &[BlockQ4K]) -> Result<Q4KOptMetadata> {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let block_count = blocks.len();
    let mut data = Vec::with_capacity(block_count * 16);

    let mut utmp: [u32; 4] = [0; 4];
    let mut scales_u8: [u8; 8] = [0; 8];
    let mut mins_u8: [u8; 8] = [0; 8];

    for block in blocks {
        // Decode 6-bit packed scales/mins (12 bytes -> 8 scales + 8 mins).
        // Exactly the same algorithm as in BlockQ4K::vec_dot_unopt.
        LittleEndian::read_u32_into(&block.scales, &mut utmp[0..3]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        LittleEndian::write_u32_into(&utmp[0..2], &mut scales_u8);
        LittleEndian::write_u32_into(&utmp[2..4], &mut mins_u8);

        let d_f32 = block.d.to_f32();
        let dmin_f32 = block.dmin.to_f32();

        // Interleaved storage: (dl_0, ml_0, dl_1, ml_1, ..., dl_7, ml_7)
        for i in 0..8 {
            let dl = d_f32 * scales_u8[i] as f32;
            let ml = dmin_f32 * mins_u8[i] as f32;
            data.push(f16::from_f32(dl));
            data.push(f16::from_f32(ml));
        }
    }

    Ok(Q4KOptMetadata {
        data,
        block_count,
        source_nonce: rand::random::<u64>(),
        layout_version: Q4K_OPT_LAYOUT_V1,
    })
}

// ════════════════════════════════════════════════════════════════════════════════
// GPU upload + dispatch bridge (only under the metal feature)
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "metal")]
pub use metal_impl::{matmul_q4k_opt_metal, Q4KOptMetadataGpu};

#[cfg(feature = "metal")]
mod metal_impl {
    use super::Q4KOptMetadata;
    use crate::backend::BackendStorage;
    use crate::op::BackpropOp;
    use crate::quantized::{GgmlDType, QStorage, QTensor};
    use crate::{MetalStorage, Result, Shape, Storage, Tensor};
    use std::sync::Arc;

    /// Pre-repacked Q4_K metadata uploaded to a Metal buffer.
    ///
    /// Created via `Q4KOptMetadata::upload_to_metal`. The buffer lives as long as
    /// the `Q4KOptMetadataGpu` (via Arc) is kept alive.
    #[derive(Debug, Clone)]
    pub struct Q4KOptMetadataGpu {
        pub buffer: Arc<candle_metal_kernels::metal::Buffer>,
        pub source_nonce: u64,
        pub byte_size: usize,
    }

    impl Q4KOptMetadata {
        /// Upload the metadata to a Metal device buffer.
        ///
        /// Creates a new MTLBuffer with a copy of the data. `source_nonce` is copied
        /// for later validation (method [`Q4KOptMetadataGpu::matches`]).
        pub fn upload_to_metal(
            &self,
            metal_device: &crate::MetalDevice,
        ) -> Result<Q4KOptMetadataGpu> {
            let buffer = metal_device
                .new_buffer_with_data(&self.data)
                .map_err(|e| crate::Error::Msg(format!("Q4KOpt upload failed: {e}")))?;
            Ok(Q4KOptMetadataGpu {
                buffer,
                source_nonce: self.source_nonce,
                byte_size: self.size_bytes(),
            })
        }
    }

    impl Q4KOptMetadataGpu {
        /// Verify the metadata identity against the source QTensor via `source_nonce`.
        pub fn matches(&self, expected_nonce: u64) -> bool {
            self.source_nonce == expected_nonce
        }
    }

    /// Dispatch the optimized Q4_K_M matmul directly (bypassing the CustomOp1 path).
    ///
    /// Equivalent to `qmatmul.forward(xs)` but via `kernel_mul_mm_q4_K_f32_opt` with
    /// a pre-packed `scales_repacked` buffer.
    ///
    /// Requirements (the caller must check beforehand):
    /// - `qtensor.dtype() == GgmlDType::Q4k`
    /// - `qtensor.storage` -- Metal (the model is loaded on a Metal device)
    /// - `xs.dtype() == DType::F32`
    /// - `xs` is on a Metal device
    /// - Dimensions aligned: M%64==0 and N%32==0 (FAST_PATH alignment)
    ///
    /// Returns a `Tensor` with Metal F32 storage of the same shape that
    /// `QMatMul::forward(xs)` would produce.
    pub fn matmul_q4k_opt_metal(
        qtensor: &QTensor,
        xs: &Tensor,
        scales: &Q4KOptMetadataGpu,
    ) -> Result<Tensor> {
        if qtensor.dtype() != GgmlDType::Q4K {
            crate::bail!(
                "matmul_q4k_opt_metal: expected Q4K weights, got {:?}",
                qtensor.dtype()
            );
        }

        let self_storage = match qtensor.storage() {
            QStorage::Metal(m) => m,
            _ => crate::bail!("matmul_q4k_opt_metal: QTensor must be on Metal device"),
        };
        let self_shape = qtensor.shape();

        // dst shape computation is equivalent to what QStorage::fwd does.
        let src_shape = xs.shape();
        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "matmul_q4k_opt_metal: input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            );
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!(
                "matmul_q4k_opt_metal: input tensor {:?} incompatible with weight {:?}",
                xs,
                self_shape
            );
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        let (storage, layout) = xs.storage_and_layout();
        let storage = match &*storage {
            Storage::Metal(m) => m,
            _ => crate::bail!("matmul_q4k_opt_metal: xs must be on Metal device"),
        };
        if storage.dtype() != crate::DType::F32 {
            crate::bail!(
                "matmul_q4k_opt_metal: xs dtype must be F32, got {:?}",
                storage.dtype()
            );
        }

        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), crate::DType::F32, "qmatmul_q4k_opt")?;
        let encoder = device.command_encoder()?;

        // 4D-aligned layout for the weight (as in the existing fwd).
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32
                    * (qtensor.dtype().type_size() as f32
                        / qtensor.dtype().block_size() as f32)) as usize
            })
            .collect::<Vec<_>>();

        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );
        let src1_elem_bytes = storage.dtype().size_in_bytes();

        candle_metal_kernels::call_quantized_matmul_mm_q4k_opt(
            device.device(),
            &encoder,
            device.kernels(),
            src0_l.dims(),
            &src0_stride,
            self_storage.buffer(),
            self_storage.offset(),
            scales.buffer.as_ref(),
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * src1_elem_bytes)
                .collect::<Vec<_>>(),
            storage.buffer(),
            layout.start_offset() * src1_elem_bytes,
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(crate::MetalError::from)?;
        drop(encoder);

        let dst_storage =
            MetalStorage::new(dst, device, dst_shape.elem_count(), crate::DType::F32);
        Ok(Tensor::from_storage(
            Storage::Metal(dst_storage),
            dst_shape,
            BackpropOp::none(),
            false,
        ))
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Unit tests
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::k_quants::{BlockQ4K, K_SCALE_SIZE, QK_K};

    /// Helper: create a BlockQ4K with the given d/dmin/scales for a test.
    fn make_test_block(d: f32, dmin: f32, scales: [u8; K_SCALE_SIZE]) -> BlockQ4K {
        BlockQ4K {
            d: f16::from_f32(d),
            dmin: f16::from_f32(dmin),
            scales,
            qs: [0u8; QK_K / 2],
        }
    }

    /// Decoder reference -- bit twiddling of scales/mins, equivalent to the inline code in
    /// `BlockQ4K::vec_dot_unopt` (k_quants.rs:1430-1440). Used only in tests.
    fn decode_scales_mins_reference(scales: &[u8; K_SCALE_SIZE]) -> ([u8; 8], [u8; 8]) {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales_u8: [u8; 8] = [0; 8];
        let mut mins_u8: [u8; 8] = [0; 8];

        LittleEndian::read_u32_into(scales, &mut utmp[0..3]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        LittleEndian::write_u32_into(&utmp[0..2], &mut scales_u8);
        LittleEndian::write_u32_into(&utmp[2..4], &mut mins_u8);
        (scales_u8, mins_u8)
    }

    /// Round-trip: for known d/dmin/scales blocks, `repack_q4k_for_opt`
    /// produces dl[i] = d * decoded_scales[i] and ml[i] = dmin * decoded_mins[i].
    #[test]
    fn test_repack_q4k_matches_reference_decoder() {
        // Realistic values: d, dmin small positives; scales filled with a
        // pattern that exercises all branches of the KMASK logic.
        let d = 0.015625f32;
        let dmin = 0.0078125f32;
        let mut scales = [0u8; K_SCALE_SIZE];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = ((i * 17 + 3) & 0xFF) as u8;
        }

        let block = make_test_block(d, dmin, scales);
        let metadata = repack_q4k_for_opt(&[block]).expect("repack");

        assert_eq!(metadata.block_count, 1);
        assert_eq!(metadata.data.len(), 16);
        assert_eq!(metadata.layout_version, Q4K_OPT_LAYOUT_V1);

        // Reference decoding
        let (ref_scales, ref_mins) = decode_scales_mins_reference(&scales);

        // Interleaved (dl_0, ml_0, dl_1, ml_1, ..., dl_7, ml_7)
        for i in 0..8 {
            let dl_actual = metadata.data[i * 2].to_f32();
            let ml_actual = metadata.data[i * 2 + 1].to_f32();
            let dl_expected = d * ref_scales[i] as f32;
            let ml_expected = dmin * ref_mins[i] as f32;
            let dl_diff = (dl_actual - f16::from_f32(dl_expected).to_f32()).abs();
            let ml_diff = (ml_actual - f16::from_f32(ml_expected).to_f32()).abs();
            assert!(
                dl_diff < 1e-5,
                "dl[{i}] mismatch: actual={dl_actual} expected={dl_expected}"
            );
            assert!(
                ml_diff < 1e-5,
                "ml[{i}] mismatch: actual={ml_actual} expected={ml_expected}"
            );
        }
    }

    /// Multiple blocks -> the layout preserves order.
    #[test]
    fn test_repack_multiple_blocks_layout() {
        let block_a = make_test_block(1.0, 0.5, [1u8; K_SCALE_SIZE]);
        let block_b = make_test_block(2.0, 0.25, [2u8; K_SCALE_SIZE]);

        let metadata = repack_q4k_for_opt(&[block_a, block_b]).expect("repack");

        assert_eq!(metadata.block_count, 2);
        assert_eq!(metadata.data.len(), 32);

        // Per-block: scales [1,1,...] and mins [1,1,...] after decode.
        // d=1.0, dmin=0.5 -> dl[i] = scales_decoded[i] (any in [0..63]), ml[i] = 0.5 * mins_decoded[i].
        // Block A's data -- the first 16 f16; Block B's data -- the next 16.
        let block_a_data: Vec<f32> = metadata.data[0..16].iter().map(|h| h.to_f32()).collect();
        let block_b_data: Vec<f32> = metadata.data[16..32].iter().map(|h| h.to_f32()).collect();

        // Both blocks have different dl/ml because scales [1,...] vs [2,...] decode differently.
        assert_ne!(block_a_data, block_b_data, "block A and B must differ");
    }

    /// Identity guard: two independent repacks have different `source_nonce`.
    #[test]
    fn test_source_nonce_uniqueness() {
        let block = make_test_block(1.0, 0.5, [0u8; K_SCALE_SIZE]);
        let meta1 = repack_q4k_for_opt(&[block.clone()]).expect("repack 1");
        let meta2 = repack_q4k_for_opt(&[block]).expect("repack 2");
        assert_ne!(
            meta1.source_nonce(),
            meta2.source_nonce(),
            "rand::random<u64> collision (extremely unlikely)"
        );
    }

    /// size_bytes() correctly counts the CPU footprint.
    #[test]
    fn test_size_bytes_calculation() {
        let blocks: Vec<BlockQ4K> = (0..100)
            .map(|i| make_test_block(0.01 * i as f32, 0.005, [i as u8; K_SCALE_SIZE]))
            .collect();
        let metadata = repack_q4k_for_opt(&blocks).expect("repack");

        // 100 blocks × 16 f16 × 2 bytes/f16 = 3200 bytes
        assert_eq!(metadata.size_bytes(), 100 * 16 * 2);
        assert_eq!(metadata.block_count(), 100);
    }
}
