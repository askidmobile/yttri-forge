use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, Layout, MetalDevice, MetalStorage, Result, Shape, D};
use candle_metal_kernels::metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
    /// Byte offset from the start of the buffer to this tensor's data.
    /// For regular (per-tensor) buffers it is 0.
    /// For a shared NoCopy buffer (mmap zero-copy) it is the tensor's offset in the file.
    offset: usize,
    /// Data size of the tensor in bytes. None = the whole buffer (back-compat).
    tensor_size: Option<usize>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device
            .new_buffer_builder()
            .with_zeros(size)
            .with_label("qstorage_zeros")
            .build()?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
            offset: 0,
            tensor_size: None,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Byte offset from the start of the buffer to this tensor's data.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Native F16 dequantization on the GPU without an intermediate F32 buffer.
    /// 2x memory savings on embed_tokens, output projection and other large
    /// quantized weights. Uses the Metal kernel `kernel_dequantize_*_f16`
    /// (see candle-metal-kernels/src/metal_src/quantized.metal).
    ///
    /// For F16/F32/BF16 (non-quantized) tensors it does a regular F32 dequantize
    /// -> to_dtype(F16) fallback (the kernel supports only Q-types).
    pub fn dequantize_f16(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::backend::BackendStorage;
        use crate::MetalError;

        // Fallback for non-quantized types (F16/F32/BF16): the kernel does not support them,
        // but the semantics are the same -- just dequantize + cast to F16.
        if matches!(
            self.dtype,
            GgmlDType::F16 | GgmlDType::F32 | GgmlDType::BF16
        ) {
            let f32_storage = self.dequantize(elem_count)?;
            return f32_storage.to_dtype(
                &crate::Layout::contiguous(elem_count),
                DType::F16,
            );
        }

        let dst = self.device.new_buffer(elem_count, DType::F16, "dequantize_f16")?;
        let encoder = self.device.command_encoder()?;
        candle_metal_kernels::call_dequantize_q_to_half(
            self.device.device(),
            &encoder,
            self.device.kernels(),
            self.dtype.into(),
            &self.buffer,
            self.offset,
            &dst,
            elem_count,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(dst, self.device.clone(), elem_count, DType::F16))
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let copy_size = self.storage_size_in_bytes();
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(copy_size)
            .with_label("qstorage_dequantize_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, self.offset, &buffer, 0, copy_size);
        }
        self.device.flush_and_wait_current()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
            GgmlDType::IQ2XXS
            | GgmlDType::IQ2XS
            | GgmlDType::IQ3XXS
            | GgmlDType::IQ1S
            | GgmlDType::IQ4NL
            | GgmlDType::IQ3S
            | GgmlDType::IQ2S
            | GgmlDType::IQ4XS
            | GgmlDType::IQ1M => {
                crate::bail!(
                    "Metal dequantization is not implemented for {:?}",
                    self.dtype
                )
            }
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&out)
            .with_label("qstorage_dequantized")
            .build()?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantized")
            .build()?;
        self.buffer = buffer;
        self.offset = 0;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix")
            .build()?;
        self.buffer = buffer;
        self.offset = 0;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_imatrix_onto")
            .build()?;
        self.buffer = buffer;
        self.offset = 0;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self
            .device
            .new_buffer_builder()
            .with_data(&qcpu_storage.data()?)
            .with_label("qstorage_quantize_onto")
            .build()?;
        self.buffer = buffer;
        self.offset = 0;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.tensor_size.unwrap_or(self.buffer.length())
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &MetalStorage,
        ids_l: &Layout,
    ) -> Result<MetalStorage> {
        use crate::MetalError;

        if ids.dtype() != DType::U32 {
            crate::bail!("quantized embedding expects u32 ids, got {:?}", ids.dtype())
        }
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids_len = ids_l.shape().elem_count();
        let device = self.device.clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(ids_len * hidden, DType::F32)
            .with_label("qembedding")
            .build()?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_quantized_get_rows(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            hidden,
            hidden * self.dtype.type_size() / self.dtype.block_size(),
            ids_len,
            &self.buffer,
            ids.buffer(),
            ids_l.start_offset() * DType::U32.size_in_bytes(),
            &dst,
        )
        .map_err(MetalError::from)?;
        Ok(MetalStorage::new(
            dst,
            device.clone(),
            ids_len * hidden,
            DType::F32,
        ))
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        let input_is_f16 = matches!(storage.dtype(), DType::F16);
        for batch_id in 0..m {
            candle_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                input_is_f16,
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                self.offset,
                batch_id * n * DType::F32.size_in_bytes(),
                &dst,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        // Enable the mm-path for Q4K/Q6K (Q4_K_M is the main dtype of
        // Qwen3.5-2B + Q6K embedding) -- it gives ~8x prefill speedup (2048
        // vector matmul -> 1 matrix matmul). The corresponding kernels
        // `kernel_mul_mm_q{4,6}_K_f{32,16}` are defined in quantized.metal.
        //
        // The remaining K-quants (Q2/Q3/Q5/Q8) and IQ-quants still go through
        // fwd_mv: either there is no mm-kernel, or there are alignment issues. If
        // new dtypes appear in supported models -- extend this list.
        if matches!(
            self.dtype,
            GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q5K
                | GgmlDType::Q8K
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ2XS
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ1S
                | GgmlDType::IQ4NL
                | GgmlDType::IQ3S
                | GgmlDType::IQ2S
                | GgmlDType::IQ4XS
                | GgmlDType::IQ1M
        ) {
            return self.fwd_mv(self_shape, storage, layout);
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let dst = device
            .new_buffer_builder()
            .with_size_for(dst_shape.elem_count(), DType::F32)
            .with_label("qmatmul")
            .build()?;
        let encoder = device.command_encoder()?;

        if !matches!(storage.dtype(), DType::F32 | DType::F16) {
            crate::bail!(
                "qmatmul Metal mm path supports only F32/F16 input, got {:?}",
                storage.dtype()
            )
        }
        let input_is_f16 = matches!(storage.dtype(), DType::F16);

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        let src1_elem_bytes = storage.dtype().size_in_bytes();

        candle_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            self.dtype.into(),
            input_is_f16,
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            self.offset,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * src1_elem_bytes)
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * src1_elem_bytes,
            dst_shape.dims(),
            0,
            &dst,
        )
        .map_err(MetalError::from)?;

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), DType::F32);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let size = self.storage_size_in_bytes();
        let buffer = self
            .device
            .new_buffer_builder()
            .with_size(size)
            .with_label("qstorage_data_blit")
            .build()?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, self.offset, &buffer, 0, size);
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec::<u8>(&buffer, size))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device
        .new_buffer_builder()
        .with_data(data)
        .with_label("qstorage_load_quantized")
        .build()?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
        offset: 0,
        tensor_size: None,
    }))
}

pub fn load_quantized_bytes(
    device: &MetalDevice,
    data: &[u8],
    dtype: super::GgmlDType,
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    Ok(QStorage::Metal(QMetalStorage {
        dtype,
        device: device.clone(),
        buffer,
        offset: 0,
        tensor_size: None,
    }))
}

/// Creates a QStorage from a shared NoCopy Metal buffer (zero-copy mmap).
///
/// Instead of copying data into a separate Metal buffer, it references a slice
/// of the shared buffer via an offset. The buffer size is the whole file; the offset points
/// to the start of the specific tensor's data.
pub fn load_quantized_from_shared_buffer(
    device: &MetalDevice,
    shared_buffer: Arc<Buffer>,
    dtype: super::GgmlDType,
    offset: usize,
    tensor_size: usize,
) -> Result<QStorage> {
    Ok(QStorage::Metal(QMetalStorage {
        dtype,
        device: device.clone(),
        buffer: shared_buffer,
        offset,
        tensor_size: Some(tensor_size),
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

impl From<GgmlDType> for candle_metal_kernels::GgmlDType {
    fn from(value: GgmlDType) -> Self {
        match value {
            GgmlDType::Q4_0 => candle_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => candle_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => candle_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => candle_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => candle_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => candle_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => candle_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => candle_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => candle_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => candle_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => candle_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => candle_metal_kernels::GgmlDType::Q8K,
            GgmlDType::IQ2XXS => candle_metal_kernels::GgmlDType::IQ2XXS,
            GgmlDType::IQ2XS => candle_metal_kernels::GgmlDType::IQ2XS,
            GgmlDType::IQ3XXS => candle_metal_kernels::GgmlDType::IQ3XXS,
            GgmlDType::IQ1S => candle_metal_kernels::GgmlDType::IQ1S,
            GgmlDType::IQ4NL => candle_metal_kernels::GgmlDType::IQ4NL,
            GgmlDType::IQ3S => candle_metal_kernels::GgmlDType::IQ3S,
            GgmlDType::IQ2S => candle_metal_kernels::GgmlDType::IQ2S,
            GgmlDType::IQ4XS => candle_metal_kernels::GgmlDType::IQ4XS,
            GgmlDType::IQ1M => candle_metal_kernels::GgmlDType::IQ1M,
            GgmlDType::F16 => candle_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => candle_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => candle_metal_kernels::GgmlDType::BF16,
        }
    }
}
