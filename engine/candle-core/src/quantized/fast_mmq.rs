//! CUDA fast path для плотного prefill: Tensor-Core MMA MMQ (llama.cpp mul_mat_q).
//! Активен при m > 8 (batch/seq). Для m <= 8 используется MMVQ (fast_mmvq/dequantize_matmul_vec).

use super::cuda::QCudaStorage;
use crate::{CudaStorage, Result, Shape};

pub fn try_fwd(
    qstorage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    qstorage.mul_mat_q_mma(self_shape, rhs, rhs_l)
}
