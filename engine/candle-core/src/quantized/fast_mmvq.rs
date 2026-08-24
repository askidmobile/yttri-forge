//! CUDA fast path for GGUF matmul (fallback stub when libmoe.a is not compiled).

use super::cuda::QCudaStorage;
use crate::{CudaStorage, Result, Shape};

pub fn try_fwd(
    _qstorage: &QCudaStorage,
    _self_shape: &Shape,
    _rhs: &CudaStorage,
    _rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    // Falls back to dequantize_matmul_vec / dequantize_matmul in quantized/cuda.rs (PTX runtime)
    Ok(None)
}
