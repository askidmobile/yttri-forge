//! Bridge for `kernel_mul_mm_q4_K_f32_v4` (Full Half Pipeline).
//!
//! Final piece of the Q4_K_M kernel optimization journey: the mc accumulator converts
//! `simdgroup_float8x8` -> `simdgroup_half8x8` to bypass the F32 limiter (92.28%
//! in the V3 measurement). Lossy: fp16 accumulator numerical drift over the K dimension --
//! gate cosine >= 0.99 + semantic eq >= 8/10.
//!
//! Reuses `Q4KOptMetadataGpu` from `q4k_opt` -- the same metadata buffer for V2/V3/V4.

#[cfg(feature = "metal")]
pub use metal_impl::matmul_q4k_v4_metal;

#[cfg(feature = "metal")]
mod metal_impl {
    use crate::backend::BackendStorage;
    use crate::op::BackpropOp;
    use crate::quantized::q4k_opt::Q4KOptMetadataGpu;
    use crate::quantized::{GgmlDType, QStorage, QTensor};
    use crate::{MetalStorage, Result, Shape, Storage, Tensor};

    /// Dispatch V4 Q4_K_M matmul (full half pipeline).
    ///
    /// Same requirements as `matmul_q4k_opt_metal` / `matmul_q4k_v3_metal`:
    /// - `qtensor.dtype() == GgmlDType::Q4k`
    /// - `qtensor` on a Metal device
    /// - `xs.dtype() == DType::F32`
    /// - Dimensions aligned: `M%64==0 && N%32==0` (FAST_PATH)
    ///
    /// Difference from V3: the kernel accumulator (`mc`) is now `simdgroup_half8x8` instead of
    /// `simdgroup_float8x8`. Lossy numerical change -- `simdgroup_store` performs an
    /// auto-conversion half -> float dst.
    pub fn matmul_q4k_v4_metal(
        qtensor: &QTensor,
        xs: &Tensor,
        scales: &Q4KOptMetadataGpu,
    ) -> Result<Tensor> {
        if qtensor.dtype() != GgmlDType::Q4K {
            crate::bail!(
                "matmul_q4k_v4_metal: expected Q4K weights, got {:?}",
                qtensor.dtype()
            );
        }

        let self_storage = match qtensor.storage() {
            QStorage::Metal(m) => m,
            _ => crate::bail!("matmul_q4k_v4_metal: QTensor must be on Metal device"),
        };
        let self_shape = qtensor.shape();

        let src_shape = xs.shape();
        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "matmul_q4k_v4_metal: input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            );
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!(
                "matmul_q4k_v4_metal: input tensor {:?} incompatible with weight {:?}",
                xs,
                self_shape
            );
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);

        let (storage, layout) = xs.storage_and_layout();
        let storage = match &*storage {
            Storage::Metal(m) => m,
            _ => crate::bail!("matmul_q4k_v4_metal: xs must be on Metal device"),
        };
        if storage.dtype() != crate::DType::F32 {
            crate::bail!(
                "matmul_q4k_v4_metal: xs dtype must be F32, got {:?}",
                storage.dtype()
            );
        }

        let device = storage.device().clone();
        let dst = device.new_buffer(dst_shape.elem_count(), crate::DType::F32, "qmatmul_q4k_v4")?;
        let encoder = device.command_encoder()?;

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

        candle_metal_kernels::call_quantized_matmul_mm_q4k_v4(
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
