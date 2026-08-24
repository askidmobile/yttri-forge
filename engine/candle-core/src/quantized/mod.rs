use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::{borrow::Cow, sync::OnceLock};

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod k_quants;
pub mod q4k_opt;
pub mod q4k_v3;
pub mod q4k_v4;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

fn as_t_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
    /// Lazily initialized storage for repacked quantized data. Currently raw bits, could be `QStorage` in the future.
    /// Not always used.
    #[allow(dead_code)]
    repacked_qs: OnceLock<Option<Vec<u8>>>,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        let data: &[u8] = &data;
        match device {
            Device::Cpu => Ok(Self::Cpu(dtype.from_data(Cow::Borrowed(data)))),
            Device::Metal(d) => match dtype {
                GgmlDType::F32 => metal::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => metal::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => metal::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => metal::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => metal::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => metal::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => metal::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => metal::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => metal::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => metal::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => metal::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => metal::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => metal::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => metal::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(data)),
                GgmlDType::IQ2XXS
                | GgmlDType::IQ2XS
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ1S
                | GgmlDType::IQ4NL
                | GgmlDType::IQ3S
                | GgmlDType::IQ2S
                | GgmlDType::IQ4XS
                | GgmlDType::IQ1M => metal::load_quantized_bytes(d, data.as_ref(), dtype),
            },
            Device::Cuda(d) => match dtype {
                GgmlDType::F32 => cuda::load_quantized(d, as_t_slice::<f32>(data)),
                GgmlDType::F16 => cuda::load_quantized(d, as_t_slice::<f16>(data)),
                GgmlDType::Q4_0 => cuda::load_quantized(d, as_t_slice::<BlockQ4_0>(data)),
                GgmlDType::Q4_1 => cuda::load_quantized(d, as_t_slice::<BlockQ4_1>(data)),
                GgmlDType::Q5_0 => cuda::load_quantized(d, as_t_slice::<BlockQ5_0>(data)),
                GgmlDType::Q5_1 => cuda::load_quantized(d, as_t_slice::<BlockQ5_1>(data)),
                GgmlDType::Q8_0 => cuda::load_quantized(d, as_t_slice::<BlockQ8_0>(data)),
                GgmlDType::Q8_1 => cuda::load_quantized(d, as_t_slice::<BlockQ8_1>(data)),
                GgmlDType::Q2K => cuda::load_quantized(d, as_t_slice::<BlockQ2K>(data)),
                GgmlDType::Q3K => cuda::load_quantized(d, as_t_slice::<BlockQ3K>(data)),
                GgmlDType::Q4K => cuda::load_quantized(d, as_t_slice::<BlockQ4K>(data)),
                GgmlDType::Q5K => cuda::load_quantized(d, as_t_slice::<BlockQ5K>(data)),
                GgmlDType::Q6K => cuda::load_quantized(d, as_t_slice::<BlockQ6K>(data)),
                GgmlDType::Q8K => cuda::load_quantized(d, as_t_slice::<BlockQ8K>(data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(data)),
                GgmlDType::IQ3XXS => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ3XXS)
                }
                GgmlDType::IQ2S => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ2S)
                }
                GgmlDType::IQ3S => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ3S)
                }
                GgmlDType::IQ2XS => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ2XS)
                }
                GgmlDType::IQ4XS => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ4XS)
                }
                GgmlDType::IQ2XXS => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ2XXS)
                }
                GgmlDType::IQ1M => {
                    cuda::load_quantized_bytes(d, data.as_ref(), GgmlDType::IQ1M)
                }
                GgmlDType::IQ1S | GgmlDType::IQ4NL => {
                    crate::bail!("CUDA is not implemented for {:?}", dtype)
                }
            },
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    /// Dequantize a row-slice [row_start, row_end) of a 2D weight [n, k] into
    /// a contiguous f32 buffer [(row_end - row_start) * k] on the compute
    /// device.
    ///
    /// CUDA-only: uses the per-block dequantize kernel slicing the device
    /// buffer at a byte offset — no full-weight f32 allocation. Non-CUDA
    /// backends are intentionally not supported here (they use the
    /// `QTensor::dequantize_rowslice` fallback which does full dequant +
    /// reshape + narrow).
    pub fn dequantize_rowslice(
        &self,
        row_start: usize,
        row_end: usize,
        k: usize,
    ) -> Result<Storage> {
        match self {
            QStorage::Cuda(storage) => {
                let cuda = storage.dequantize_rowslice(row_start, row_end, k)?;
                Ok(Storage::Cuda(cuda))
            }
            _ => crate::bail!(
                "dequantize_rowslice (storage-level) only supports CUDA storage; \
                 use QTensor::dequantize_rowslice for CPU/Metal fallback"
            ),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr_with_guard(stream),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    IQ1M,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            29 => Self::IQ1M,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ1S => 19,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            Self::IQ1M => 29,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
            Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => Box::new(RawQuantizedType::zeros(*self, elem_count)),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        let data: &[u8] = &data;
        match self {
            Self::F32 => Box::new(as_t_slice::<f32>(data).to_vec()),
            Self::F16 => Box::new(as_t_slice::<f16>(data).to_vec()),
            Self::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(data).to_vec()),
            Self::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(data).to_vec()),
            Self::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(data).to_vec()),
            Self::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(data).to_vec()),
            Self::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(data).to_vec()),
            Self::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(data).to_vec()),
            Self::Q2K => Box::new(as_t_slice::<BlockQ2K>(data).to_vec()),
            Self::Q3K => Box::new(as_t_slice::<BlockQ3K>(data).to_vec()),
            Self::Q4K => Box::new(as_t_slice::<BlockQ4K>(data).to_vec()),
            Self::Q5K => Box::new(as_t_slice::<BlockQ5K>(data).to_vec()),
            Self::Q6K => Box::new(as_t_slice::<BlockQ6K>(data).to_vec()),
            Self::Q8K => Box::new(as_t_slice::<BlockQ8K>(data).to_vec()),
            Self::BF16 => Box::new(as_t_slice::<bf16>(data).to_vec()),
            Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => Box::new(RawQuantizedType::from_data(*self, std::borrow::Cow::Borrowed(data))),
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
            Self::IQ2XXS => 2 + k_quants::QK_K / 8 * 2,
            Self::IQ2XS => 2 + k_quants::QK_K / 8 * 2 + k_quants::QK_K / 32,
            Self::IQ3XXS => 2 + 3 * (k_quants::QK_K / 8),
            Self::IQ1S => 2 + k_quants::QK_K / 8 + k_quants::QK_K / 16,
            Self::IQ4NL => 2 + 32 / 2,
            Self::IQ3S => 2 + 13 * (k_quants::QK_K / 32) + k_quants::QK_K / 64,
            Self::IQ2S => 2 + k_quants::QK_K / 4 + k_quants::QK_K / 16,
            Self::IQ4XS => 2 + 2 + k_quants::QK_K / 64 + k_quants::QK_K / 2,
            Self::IQ1M => k_quants::QK_K / 8 + k_quants::QK_K / 16 + k_quants::QK_K / 32,
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
            Self::IQ4NL => 32,
            Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => k_quants::QK_K,
        }
    }
}

pub(crate) struct RawQuantizedType {
    dtype: GgmlDType,
    data: Vec<u8>,
}

impl RawQuantizedType {
    fn zeros(dtype: GgmlDType, elem_count: usize) -> Self {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        Self {
            dtype,
            data: vec![0; size],
        }
    }

    pub(crate) fn from_data(dtype: GgmlDType, data: Cow<'_, [u8]>) -> Self {
        Self {
            dtype,
            data: data.into_owned(),
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
}

impl QuantizedType for RawQuantizedType {
    fn matmul_t(&self, _mkn: (usize, usize, usize), _lhs: &[f32], _dst: &mut [f32]) -> Result<()> {
        crate::bail!("CPU matmul is not implemented for {:?}", self.dtype)
    }

    fn matmul_t_f16(
        &self,
        _mkn: (usize, usize, usize),
        _lhs: &[f16],
        _dst: &mut [f16],
    ) -> Result<()> {
        crate::bail!("CPU f16 matmul is not implemented for {:?}", self.dtype)
    }

    fn embedding(&self, _ids: &[u32], _rows: usize, _hidden: usize) -> Result<CpuStorage> {
        crate::bail!("CPU embedding is not implemented for {:?}", self.dtype)
    }

    fn dequantize(&self, _elem_count: usize) -> Result<CpuStorage> {
        crate::bail!("CPU dequantization is not implemented for {:?}", self.dtype)
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.data.len()
    }

    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn block_size(&self) -> usize {
        self.dtype.block_size()
    }

    fn from_float(&mut self, _xs: &[f32]) {
        panic!("CPU quantization is not implemented for {:?}", self.dtype)
    }

    fn from_float_imatrix(&mut self, _xs: &[f32], _imatrix_weights: &[f32], _n_per_row: usize) {
        panic!(
            "CPU imatrix quantization is not implemented for {:?}",
            self.dtype
        )
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn dtype(&self) -> GgmlDType {
        self.dtype
    }
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn embedding(&self, ids: &[u32], rows: usize, hidden: usize) -> Result<CpuStorage> {
        if !hidden.is_multiple_of(T::BLCK_SIZE) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                T::BLCK_SIZE
            )
        }
        let row_blocks = hidden / T::BLCK_SIZE;
        if self.len() != rows * row_blocks {
            crate::bail!(
                "quantized tensor has {} blocks, expected {}",
                self.len(),
                rows * row_blocks
            )
        }
        let mut out = vec![0f32; ids.len() * hidden];
        for (out_row, &row_id) in ids.iter().enumerate() {
            let row = row_id as usize;
            if row >= rows {
                crate::bail!("embedding id {row} is out of range for {rows} rows")
            }
            let src = &self[row * row_blocks..(row + 1) * row_blocks];
            let dst = &mut out[out_row * hidden..(out_row + 1) * hidden];
            T::to_float(src, dst);
        }
        Ok(CpuStorage::F32(out))
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self {
            storage,
            shape,
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            repacked_qs: OnceLock::new(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Changes only tensor metadata while preserving quantized storage.
    ///
    /// New shape must keep element count and a block-aligned last dimension so
    /// existing quantized matmul kernels can consume storage without dequantizing.
    pub fn reshape<S: Into<Shape>>(mut self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.elem_count() != self.shape.elem_count() {
            crate::bail!(
                "cannot reshape quantized tensor {:?} to {:?}: element count differs",
                self.shape,
                shape
            )
        }
        check_shape(&shape, self.storage.block_size())?;
        self.shape = shape;
        Ok(self)
    }

    /// Internal accessor to QStorage (for intra-crate dispatch helpers like `q4k_opt::matmul_q4k_opt_metal`).
    #[allow(dead_code)]
    pub(crate) fn storage(&self) -> &QStorage {
        &self.storage
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    /// Fused indexed MoE matmul на CUDA (q8_1 quantization входа + per-expert
    /// kernel). `self` — packed веса [n_experts, n, k]; input [batch, topk, k]
    /// F32 CUDA; ids [batch, topk] U32. Возвращает [batch, topk, n].
    /// Поддержанные dtype: IQ2S/IQ2XS/IQ2XXS, IQ3S/IQ3XXS, IQ4XS, Q8_0, Q2K-Q6K.
    #[cfg(feature = "cuda")]
    pub fn indexed_moe_forward_cuda(&self, input: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let qs = match &self.storage {
            QStorage::Cuda(c) => c,
            _ => crate::bail!("indexed_moe_forward_cuda: weights not on CUDA"),
        };
        let (in_st, in_l) = input.storage_and_layout();
        let in_cuda = match &*in_st {
            crate::Storage::Cuda(c) => c,
            _ => crate::bail!("indexed_moe_forward_cuda: input not on CUDA"),
        };
        let (ids_st, ids_l) = ids.storage_and_layout();
        let ids_cuda = match &*ids_st {
            crate::Storage::Cuda(c) => c,
            _ => crate::bail!("indexed_moe_forward_cuda: ids not on CUDA"),
        };
        let (out, out_shape) = qs.indexed_moe_forward(
            &self.shape,
            in_cuda,
            &in_l,
            ids_cuda,
            &ids_l,
        )?;
        Ok(Tensor::from((crate::Storage::Cuda(out), out_shape)))
    }

    /// Dual indexed MoE (gate+up одним запуском, общий вход).
    /// Возвращает (gate_out, up_out) оба [batch, topk, n].
    #[cfg(feature = "cuda")]
    pub fn indexed_moe_forward_dual_cuda(
        &self,
        other: &QTensor,
        input: &Tensor,
        ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (qs, qs_other) = match (&self.storage, &other.storage) {
            (QStorage::Cuda(a), QStorage::Cuda(b)) => (a, b),
            _ => crate::bail!("indexed_moe_forward_dual_cuda: weights not on CUDA"),
        };
        if self.shape != other.shape {
            crate::bail!("dual moe: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        }
        let (in_st, in_l) = input.storage_and_layout();
        let in_cuda = match &*in_st {
            crate::Storage::Cuda(c) => c,
            _ => crate::bail!("dual moe: input not CUDA"),
        };
        let (ids_st, ids_l) = ids.storage_and_layout();
        let ids_cuda = match &*ids_st {
            crate::Storage::Cuda(c) => c,
            _ => crate::bail!("dual moe: ids not CUDA"),
        };
        let (out1, out2, shape) = qs.indexed_moe_forward_dual(
            qs_other,
            &self.shape,
            in_cuda,
            &in_l,
            ids_cuda,
            &ids_l,
        )?;
        Ok((
            Tensor::from((crate::Storage::Cuda(out1), shape.clone())),
            Tensor::from((crate::Storage::Cuda(out2), shape)),
        ))
    }

    /// Dequantize a row-slice `[row_start, row_end)` of a 2D view `[n, k]` into
    /// a contiguous f32 tensor `[(row_end - row_start), k]` on `device`.
    ///
    /// `k` is the number of columns (last dim of the shape). The tensor is
    /// treated as 2D `[n, k]` where `n = elem_count / k`.
    ///
    /// CUDA IQ-types: per-block dequantize kernel slices the device buffer at
    /// a byte offset — no full-weight f32 allocation (the OOM-avoiding path).
    /// CPU/Metal: falls back to full `dequantize` + reshape + narrow.
    pub fn dequantize_rowslice(
        &self,
        row_start: usize,
        row_end: usize,
        k: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let rows = row_end
            .checked_sub(row_start)
            .ok_or_else(|| crate::Error::Msg(format!("row_end < row_start ({row_end} < {row_start})")).bt())?;
        if k == 0 {
            crate::bail!("dequantize_rowslice: k must be non-zero");
        }
        let elem_count = self.shape.elem_count();
        let n = elem_count / k;
        if row_end > n {
            crate::bail!(
                "dequantize_rowslice: row_end {} > n_rows {} (elem_count={}, k={})",
                row_end,
                n,
                elem_count,
                k
            );
        }
        // Fast path: CUDA storage uses the per-block rowslice kernel directly.
        let out_shape = Shape::from((rows, k));
        if rows == 0 {
            // Empty slice: return a zero-element tensor with the right shape.
            let none = crate::op::BackpropOp::none();
            return crate::tensor::from_storage(
                Storage::Cpu(crate::CpuStorage::F32(Vec::new())),
                out_shape,
                none,
                false,
            )
            .to_device(device);
        }
        match &self.storage {
            QStorage::Cuda(storage) => {
                let cuda = storage.dequantize_rowslice(row_start, row_end, k)?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(cuda), out_shape, none, false)
                    .to_device(device)
            }
            // Fallback: full dequant + reshape to [n, k] + narrow(0, row_start, rows).
            // CPU/Metal reference tests use small experts so the full dequant
            // is fine and matches the reference numerically.
            _ => {
                let full = self.dequantize(device)?;
                let full = full.to_dtype(crate::DType::F32)?.reshape((n, k))?;
                full.narrow(0, row_start, rows)
            }
        }
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        // For Metal -- a native GPU-kernel writes half directly without an F32 intermediate,
        // halving peak memory when loading weights with CANDLE_DEQUANTIZE_ALL_F16.
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            QStorage::Metal(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Metal(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        let (rows, hidden) = self.shape.dims2()?;
        if !hidden.is_multiple_of(self.dtype().block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype().block_size()
            )
        }
        let mut out_shape = ids.dims().to_vec();
        out_shape.push(hidden);
        let device = self.device();
        let ids = ids
            .to_device(&device)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .contiguous()?;
        let storage = match &self.storage {
            QStorage::Cpu(storage) => {
                let ids = ids.to_vec1::<u32>()?;
                Storage::Cpu(storage.embedding(&ids, rows, hidden)?)
            }
            QStorage::Metal(storage) => match &*ids.storage() {
                Storage::Metal(ids_storage) => {
                    Storage::Metal(storage.embedding(rows, hidden, ids_storage, ids.layout())?)
                }
                _ => unreachable!("ids were moved to the QTensor device"),
            },
            QStorage::Cuda(storage) => match &*ids.storage() {
                Storage::Cuda(ids_storage) => {
                    Storage::Cuda(storage.embedding(rows, hidden, ids_storage, ids.layout())?)
                }
                _ => unreachable!("ids were moved to the QTensor device"),
            },
        };
        let none = crate::op::BackpropOp::none();
        Ok(crate::tensor::from_storage(storage, out_shape, none, false))
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        // Layout safety: the PTX path (`indexed_moe_forward_fused_q8_1_input`) reads
        // input/ids via `as_cuda_slice` without applying layout offset/strides,
        // so we require c-contiguous layout and zero start_offset.
        if !x.layout().is_contiguous() {
            crate::bail!("indexed_moe_forward requires contiguous input tensor");
        }
        if x.layout().start_offset() != 0 {
            crate::bail!("indexed_moe_forward requires zero-offset input tensor");
        }
        if !ids.layout().is_contiguous() {
            crate::bail!("indexed_moe_forward requires contiguous ids tensor");
        }
        if ids.layout().start_offset() != 0 {
            crate::bail!("indexed_moe_forward requires zero-offset ids tensor");
        }
        match &self.storage {
            QStorage::Cuda(s) => match (&*x.storage(), &*ids.storage()) {
                (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                _ => crate::bail!(
                    "indexed_moe_forward requires CUDA tensors for input and ids"
                ),
            },
            QStorage::Metal(_) => crate::bail!(
                "indexed_moe_forward is not implemented for the Metal backend"
            ),
            QStorage::Cpu(_) => crate::bail!(
                "indexed_moe_forward is not implemented for the CPU backend"
            ),
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.storage.device_ptr_with_guard(stream)
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        // F16/BF16 веса → TensorF16 (не F32!): иначе BF16-модель раздувается
        // в 2x VRAM (27B BF16: 53.8GB → 107GB F32 → OOM на A100 80GB,
        // поймано 2026-08-11). F32 остаётся F32.
        let t = match qtensor.dtype() {
            GgmlDType::F32 => Self::Tensor(qtensor.dequantize(&qtensor.device())?),
            GgmlDType::F16 | GgmlDType::BF16 => {
                Self::TensorF16(qtensor.dequantize_f16(&qtensor.device())?)
            }
            _ if DEQUANTIZE_ALL.with(|b| *b) => {
                Self::Tensor(qtensor.dequantize(&qtensor.device())?)
            }
            _ if DEQUANTIZE_ALL_F16.with(|b| *b) => {
                Self::TensorF16(qtensor.dequantize_f16(&qtensor.device())?)
            }
            _ => Self::QTensor(qtensor),
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            _ => crate::bail!(
                "indexed_moe_forward is only supported for QTensor-backed QMatMul, \
                 not for dequantized Tensor/TensorF16 variants"
            ),
        }
    }

    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.embedding(ids),
            Self::Tensor(w) | Self::TensorF16(w) => {
                let mut final_dims = ids.dims().to_vec();
                final_dims.push(w.dim(D::Minus1)?);
                let ids = ids.to_device(w.device())?.flatten_all()?;
                w.index_select(&ids, 0)?.reshape(final_dims)
            }
        }
    }
}

pub enum Q8_1Activation {
    #[cfg(feature = "cuda")]
    Cuda {
        slice: cuda::Q81ScratchRef,
        ncols: usize,
        b_size: usize,
    },
}

impl QTensor {
    #[cfg(feature = "cuda")]
    pub fn prequantize_q8_1(x: &Tensor) -> Result<Option<Q8_1Activation>> {
        let (b_size, k) = match x.dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => return Ok(None),
        };
        if b_size == 0 || b_size > 8 {
            return Ok(None);
        }
        let (storage, layout) = x.storage_and_layout();
        if !layout.is_contiguous() {
            return Ok(None);
        }
        if let Storage::Cuda(c) = &*storage {
            let slice = c.as_cuda_slice::<f32>()?;
            let view = slice.slice(layout.start_offset()..layout.start_offset() + b_size * k);
            let q8_1_buf = cuda::QCudaStorage::prequantize_q8_1(c.device(), &view, k, b_size)?;
            return Ok(Some(Q8_1Activation::Cuda {
                slice: q8_1_buf,
                ncols: k,
                b_size,
            }));
        }
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_prequant(&self, x: &Tensor, prequant: Option<&Q8_1Activation>) -> Result<Tensor> {
        if !cuda::FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(Q8_1Activation::Cuda { slice, ncols, b_size }) = prequant {
                let (x_b_size, x_k) = match x.dims() {
                    [b, m, k] => (b * m, *k),
                    [b, k] => (*b, *k),
                    _ => (0, 0),
                };
                if x_b_size == *b_size && x_k == *ncols {
                    if let QStorage::Cuda(c) = &self.storage {
                        {
                            let (n, k) = self.shape.dims2()?;
                            if *ncols == k {
                                let out = c.mul_mat_vec_with_prequant_q8_1(&self.shape, slice, *ncols, n, *b_size)?;
                                let mut out_shape = x.dims().to_vec();
                                out_shape.pop();
                                out_shape.push(n);
                                let none = crate::op::BackpropOp::none();
                                return Ok(crate::tensor::from_storage(Storage::Cuda(out), out_shape, none, false));
                            }
                        }
                    }
                }
            }
        }
        x.apply_op1_no_bwd(self)
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];

                // Try the 8-column BlockQ4Kx8 repacked path.
                #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
                if self_storage.dtype() == GgmlDType::Q4K && n.is_multiple_of(8) {
                    use zerocopy::{FromBytes, IntoBytes};

                    let total_blocks =
                        self_storage.storage_size_in_bytes() / std::mem::size_of::<BlockQ4K>();
                    let repacked = self.repacked_qs.get_or_init(|| {
                        let blocks = unsafe {
                            std::slice::from_raw_parts(
                                self_storage.as_ptr() as *const BlockQ4K,
                                total_blocks,
                            )
                        };
                        let packed = k_quants::pack_to_q4kx8(blocks, n);
                        Some(packed.as_bytes().to_vec())
                    });
                    if let Some(repacked_bytes) = repacked {
                        let block_x8: &[BlockQ4Kx8] =
                            <[BlockQ4Kx8]>::ref_from_bytes(repacked_bytes).map_err(|_| {
                                crate::Error::Msg(
                                    "repacked_qs alignment invariant violated".to_string(),
                                )
                            })?;

                        k_quants::matmul_q4k_x8(
                            (dst_shape.elem_count() / n, k, n),
                            slice,
                            block_x8,
                            &mut dst_storage,
                        )?;
                        return Ok((crate::CpuStorage::F32(dst_storage), dst_shape));
                    }
                }

                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape))
            }
            _ => crate::bail!("Expected f32/f16"),
        }
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl QMatMul {
    pub fn device(&self) -> Device {
        match self {
            Self::QTensor(t) => t.device(),
            Self::Tensor(w) | Self::TensorF16(w) => w.device().clone(),
        }
    }

    #[allow(unused_variables)]
    pub fn forward_with_prequant(&self, xs: &Tensor, prequant: Option<&Q8_1Activation>) -> Result<Tensor> {
        use crate::Module;
        let xs = if !xs.device().same_device(&self.device()) {
            xs.to_device(&self.device())?
        } else {
            xs.clone()
        };
        match self {
            #[cfg(feature = "cuda")]
            Self::QTensor(t) => t.forward_with_prequant(&xs, prequant),
            _ => self.forward(&xs),
        }
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if !xs.device().same_device(&self.device()) {
            xs.to_device(&self.device())?
        } else {
            xs.clone()
        };
        if std::env::var_os("QWEN36_TRACE_MMQ").is_some() {
            let variant = match self {
                Self::QTensor(_) => "QTensor",
                Self::Tensor(_) => "Tensor",
                Self::TensorF16(_) => "TensorF16",
            };
            eprintln!("[qmatmul-fwd] variant={} xs={:?}", variant, xs.shape());
        }
        match self {
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}
