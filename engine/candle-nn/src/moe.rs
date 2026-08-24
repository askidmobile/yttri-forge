// Mixture-of-Experts (MoE) GEMM operations on CUDA via dynamic PTX runtime.
//
// Uses WMMA grouped kernels for prefill and vectorized MMVQ for decode.
// All kernels are dynamically loaded through cudarc without any link-time dependency on libmoe.a or cudart.

#[allow(unused_imports)]
use candle::quantized::{self, QTensor};
use candle::{Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle::cuda_backend::cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
    use candle::cuda_backend::WrapErr;
    use candle::quantized::GgmlDType;
    use candle::{CudaDevice, CudaStorage, DType, Storage};
    use half::{bf16, f16};

    const CEILDIV: fn(usize, usize) -> usize = |x, y| (x + y - 1) / y;

    fn calculate_expert_offsets(
        dev: &CudaDevice,
        expert_ids: &CudaSlice<u32>,
        size_m: usize,
        num_experts: usize,
    ) -> Result<CudaSlice<i32>> {
        let expert_counts = dev.alloc_zeros::<i32>(num_experts)?;
        let expert_offsets = dev.alloc_zeros::<i32>(num_experts + 1)?;

        // 1. Count tokens per expert
        let threads = 256;
        let blocks = CEILDIV(size_m, threads);
        let count_func = dev.get_or_load_func("count_tokens_per_expert_kernel", &candle::cuda_backend::kernels::MOE)?;
        let count_cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = count_func.builder();
        builder.arg(expert_ids);
        builder.arg(&expert_counts);
        let size_m_i = size_m as i32; builder.arg(&size_m_i);
        unsafe { builder.launch(count_cfg) }.w()?;

        // 2. Prefix sum to get expert offsets (supports up to 65536 experts via chunked scan)
        let scan_threads = (num_experts.next_power_of_two()).clamp(32, 1024);
        let smem_size = (scan_threads * std::mem::size_of::<i32>()) as u32;
        let scan_func = dev.get_or_load_func("expert_prefix_sum_kernel", &candle::cuda_backend::kernels::MOE)?;
        let scan_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (scan_threads as u32, 1, 1),
            shared_mem_bytes: smem_size,
        };
        let mut builder = scan_func.builder();
        builder.arg(&expert_counts);
        builder.arg(&expert_offsets);
        let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
        unsafe { builder.launch(scan_cfg) }.w()?;

        Ok(expert_offsets)
    }

    pub fn moe_gemm_cuda(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        fn cuda_fwd<T: candle::cuda_backend::CudaDType + DeviceRepr>(
            input: &Tensor,
            weights: &Tensor,
            topk_weights: &Option<Tensor>,
            sorted_token_ids: &Tensor,
            expert_ids: &Tensor,
            topk: usize,
            is_prefill: bool,
            is_bf16: bool,
        ) -> Result<Tensor> {
            let (mut size_m, size_k1) = input.dims2()?;
            if topk_weights.is_none() {
                size_m *= topk;
            }
            let (num_experts, size_n, size_k) = weights.dims3()?;
            if size_k != size_k1 {
                candle::bail!(
                    "input size_k ({size_k1}) and weight size_k ({size_k}) mismatch!"
                );
            }
            let dev = input.device().as_cuda_device()?;
                        
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let (weights_storage, _) = weights.storage_and_layout();
            let weights_slice = match &*weights_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("weight must be a cuda tensor"),
            };

            let (sorted_ids_storage, _) = sorted_token_ids.storage_and_layout();
            let sorted_ids_slice = match &*sorted_ids_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
            };

            let (expert_ids_storage, _) = expert_ids.storage_and_layout();
            let expert_ids_slice = match &*expert_ids_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("expert_ids must be a cuda tensor"),
            };

            let expert_offsets = calculate_expert_offsets(dev, expert_ids_slice, size_m, num_experts)?;

            let output_slice = unsafe { dev.alloc::<T>(size_m * size_n) }?;

            let kernel_name = match (is_bf16, is_prefill) {
                (false, true) => "moe_gemm_wmma_f16_prefill",
                (false, false) => "moe_gemm_wmma_f16_decode",
                (true, true) => "moe_gemm_wmma_bf16_prefill",
                (true, false) => "moe_gemm_wmma_bf16_decode",
            };

            let func = dev.get_or_load_func(kernel_name, &candle::cuda_backend::kernels::MOE)?;

            let grid_n = CEILDIV(size_n, 32);
            let grid = (num_experts as u32, grid_n as u32, 1);
            let block = (128, 1, 1);

            let a_sh_bytes = (32 * 16 * std::mem::size_of::<T>() + 15) & !15;
            let b_sh_bytes = (32 * 16 * std::mem::size_of::<T>() + 15) & !15;
            let c_sh_bytes = (32 * 32 * std::mem::size_of::<f32>() + 15) & !15;
            let smem_bytes = (a_sh_bytes + b_sh_bytes + c_sh_bytes) as u32;

            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem_bytes,
            };

            let topk_w_guard = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
            let topk_w_slice = match &topk_w_guard {
                Some((tw_storage, _)) => match &**tw_storage {
                    Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                    _ => candle::bail!("topk_weights must be a cuda tensor"),
                },
                None => None,
            };

            let mut builder = func.builder();
            builder.arg(input_slice);
            builder.arg(weights_slice);
            builder.arg(sorted_ids_slice);
            builder.arg(&expert_offsets);
            if let Some(tw) = topk_w_slice {
                builder.arg(tw);
            } else {
                builder.arg(&0u64);
            }
            builder.arg(&output_slice);
            let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
            let topk_i = topk as i32; builder.arg(&topk_i);
            let size_m_i = size_m as i32; builder.arg(&size_m_i);
            let size_n_i = size_n as i32; builder.arg(&size_n_i);
            let size_k_i = size_k as i32; builder.arg(&size_k_i);

            unsafe { builder.launch(cfg) }.w()?;

            let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());
            Ok(Tensor::from_storage(
                Storage::Cuda(output),
                (size_m, size_n),
                candle::op::BackpropOp::none(),
                false,
            ))
        }

        match input.dtype() {
            DType::F16 => cuda_fwd::<f16>(
                input, weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, false,
            ),
            DType::BF16 => cuda_fwd::<bf16>(
                input, weights, topk_weights, sorted_token_ids, expert_ids, topk, is_prefill, true,
            ),
            dtype => candle::bail!("moe_gemm only accepts f16/bf16 inputs, got {dtype:?}"),
        }
    }

    pub fn moe_gemm_gguf_cuda(
        input: &Tensor,
        weights: &QTensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
        dtype: DType,
    ) -> Result<Tensor> {
        let (mut size_m, size_k) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
        if size_k != size_k1 {
            candle::bail!(
                "input size_k ({size_k}) and weight size_k ({size_k1}) mismatch!"
            );
        }
        if size_k % 8 != 0 {
            candle::bail!("size_k must be divisible by 8, got {size_k}");
        }

        let dev = input.device().as_cuda_device()?;
                        let weight_ptr = weights.device_ptr()?;

        let (sorted_ids_storage, _) = sorted_token_ids.storage_and_layout();
        let sorted_ids_slice = match &*sorted_ids_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (expert_ids_storage, _) = expert_ids.storage_and_layout();
        let expert_ids_slice = match &*expert_ids_storage {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("expert_ids must be a cuda tensor"),
        };

        let topk_w_guard = topk_weights.as_ref().map(|tw| tw.storage_and_layout());
        let topk_w_slice = match &topk_w_guard {
            Some((tw_storage, _)) => match &**tw_storage {
                Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?),
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            },
            None => None,
        };

        let output_slice = unsafe { dev.alloc::<f32>(size_m * size_n) }?;

        let quant_name = match weights.dtype() {
            GgmlDType::Q8_0 => "q8_0",
            GgmlDType::Q4K => "q4_k",
            GgmlDType::Q2K => "q2_k",
            GgmlDType::Q3K => "q3_k",
            GgmlDType::Q5K => "q5_k",
            GgmlDType::Q6K => "q6_k",
            d => candle::bail!("moe_gemm_gguf does not support weight dtype {d:?}"),
        };

        if is_prefill {
            let expert_offsets = calculate_expert_offsets(dev, expert_ids_slice, size_m, num_experts)?;
            let input_act = input.to_dtype(dtype)?;
            let (act_storage, _) = input_act.storage_and_layout();

            let type_str = if dtype == DType::F16 { "f16" } else { "bf16" };
            let kernel_name = format!("moe_gemm_gguf_prefill_{type_str}_{quant_name}");
            let func = dev.get_or_load_func(&kernel_name, &candle::cuda_backend::kernels::MOE)?;

            let grid = (num_experts as u32, CEILDIV(size_n, 32) as u32, 1);
            let wrap_size = if matches!(weights.dtype(), GgmlDType::Q8_0 | GgmlDType::Q4K) { 32 } else { 64 };
            let block = (wrap_size as u32, 4, 1);

            let block_size_bytes = weights.dtype().type_size();
            let qk = weights.dtype().block_size();
            let a_sh_bytes = (32 * qk * 2 + 15) & !15;
            let b_sh_bytes = (32 * qk * 2 + 15) & !15;
            let b_quant_sh_bytes = (32 * block_size_bytes + 15) & !15;
            let c_sh_bytes = (32 * 32 * std::mem::size_of::<f32>() + 15) & !15;
            let smem_bytes = (a_sh_bytes + b_sh_bytes + b_quant_sh_bytes + c_sh_bytes) as u32;

            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem_bytes,
            };

            let weight_dev_ptr = weight_ptr as u64;

            let mut builder = func.builder();
            match &*act_storage {
                Storage::Cuda(c) => {
                    if dtype == DType::F16 {
                        builder.arg(c.as_cuda_slice::<f16>()?);
                    } else {
                        builder.arg(c.as_cuda_slice::<bf16>()?);
                    }
                }
                _ => candle::bail!("input must be a cuda tensor"),
            }
            builder.arg(&weight_dev_ptr);
            builder.arg(sorted_ids_slice);
            builder.arg(&expert_offsets);
            if let Some(tw) = topk_w_slice {
                builder.arg(tw);
            } else {
                builder.arg(&0u64);
            }
            builder.arg(&output_slice);
            let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
            let topk_i = topk as i32; builder.arg(&topk_i);
            let size_m_i = size_m as i32; builder.arg(&size_m_i);
            let size_n_i = size_n as i32; builder.arg(&size_n_i);
            let size_k_i = size_k as i32; builder.arg(&size_k_i);

            unsafe { builder.launch(cfg) }.w()?;
        } else {
            // Decode path: Quantize input to Q8_1
            let (input_storage, _) = input.storage_and_layout();
            let input_slice = match &*input_storage {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("input must be a cuda tensor"),
            };

            let matrix_row_padding = 512;
            let k_padded = (size_k + matrix_row_padding - 1) / matrix_row_padding * matrix_row_padding;
            let m_quant = if topk_weights.is_some() { size_m } else { size_m / topk };
            let q8_1_size = m_quant * (k_padded / 32 * std::mem::size_of::<candle::quantized::k_quants::BlockQ8_1>());
            let mut y_q8_1 = dev.alloc_zeros::<u8>(q8_1_size)?;

            let quant_func = dev.get_or_load_func("quantize_q8_1", &candle::cuda_backend::kernels::QUANTIZED)?;
            let num_blocks = (k_padded + 255) / 256;
            let quant_cfg = LaunchConfig {
                grid_dim: (num_blocks as u32, m_quant as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut quant_builder = quant_func.builder();
            quant_builder.arg(input_slice);
            quant_builder.arg(&mut y_q8_1);
            let size_k_i = size_k as i32; quant_builder.arg(&size_k_i);
            let k_padded_i = k_padded as i32; quant_builder.arg(&k_padded_i);
            unsafe { quant_builder.launch(quant_cfg) }.w()?;

            let kernel_name = format!("moe_gemm_gguf_{quant_name}");
            let func = dev.get_or_load_func(&kernel_name, &candle::cuda_backend::kernels::MOE)?;

            let n_warps = 4;
            let grid_dim = (CEILDIV(size_n, n_warps) as u32, size_m as u32, 1);
            let block_dim = (32, n_warps as u32, 1);
            let block_size_bytes = weights.dtype().type_size();
            let qk = weights.dtype().block_size();
            let shared_bytes = (size_k / qk * block_size_bytes * n_warps + 1024) as u32;

            let cfg = LaunchConfig {
                grid_dim,
                block_dim,
                shared_mem_bytes: shared_bytes,
            };

            let weight_dev_ptr = weight_ptr as u64;

            let mut builder = func.builder();
            builder.arg(&weight_dev_ptr);
            builder.arg(&y_q8_1);
            builder.arg(sorted_ids_slice);
            builder.arg(expert_ids_slice);
            if let Some(tw) = topk_w_slice {
                builder.arg(tw);
            } else {
                builder.arg(&0u64);
            }
            builder.arg(&output_slice);
            let num_experts_i = num_experts as i32; builder.arg(&num_experts_i);
            let topk_i = topk as i32; builder.arg(&topk_i);
            let size_m_i = size_m as i32; builder.arg(&size_m_i);
            let size_n_i = size_n as i32; builder.arg(&size_n_i);
            let size_k_i = size_k as i32; builder.arg(&size_k_i);
            let k_padded_i = k_padded as i32; builder.arg(&k_padded_i);

            unsafe { builder.launch(cfg) }.w()?;
        }

        let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());
        Ok(Tensor::from_storage(
            Storage::Cuda(output),
            (size_m, size_n),
            candle::op::BackpropOp::none(),
            false,
        ))
    }
}

#[allow(unused_variables)]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return cuda::moe_gemm_cuda(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                topk,
                is_prefill,
            );
        }
    }
    candle::bail!("moe_gemm (dense MoE) is only supported on CUDA")
}

#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
    dtype: candle::DType,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if input.device().is_cuda() {
            return cuda::moe_gemm_gguf_cuda(
                input,
                weights,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                topk,
                is_prefill,
                dtype,
            );
        }
    }
    candle::bail!("moe_gemm_gguf is only supported on CUDA")
}

// ─── Phase 3: PTX MoE backend (cudarc dynamic-loading, no libmoe.a) ─────────────
//
// Fused K-quant (Q2_K, Q4_K) expert GEMM via PTX kernels in moe_quantized.cu.
// IQ types bail — caller falls back to reference backend.
// CPU flattens+sorts route plan by expert (PD-010), uploads sorted arrays to
// GPU, launches moe_gate_up_kernel then moe_down_proj_kernel (or prefill).
// No cudaMallocAsync in hot path (PD-011).

#[cfg(all(feature = "cuda", feature = "moe-cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn moe_ptx_gguf(
    input: &Tensor,             // [n_tokens, n_embd] F32 on CUDA
    gate: &QTensor,             // [n_experts, n_ff, n_embd] packed quantized
    up: &QTensor,               // [n_experts, n_ff, n_embd] packed quantized
    down: &QTensor,             // [n_experts, n_embd, n_ff] packed quantized
    expert_ids: &[Vec<usize>],  // [n_tokens][topk] selected expert indices
    weights: &[Vec<f32>],      // [n_tokens][topk] normalized routing weights
    _n_experts: usize,
    n_ff: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::{
        DeviceRepr, LaunchConfig, PushKernelArg,
    };
    use candle::cuda_backend::{kernels, WrapErr};
    use candle::op::BackpropOp;
    use candle::quantized::GgmlDType;

    let dtype = gate.dtype();
    // Only Q2_K and Q4_K have fused PTX dispatch. IQ types and other K-quants
    // bail → caller (qwen35-batch) falls back to reference backend.
    // ponytail: add IQ grid tables to moe_quantized.cu to support IQ types here.
    // quant_type values match GgmlDType::to_u32() (Q2K=10, Q4K=12) and MOE_QTYPE_* in moe_dequant.cuh.
    let (quant_type, block_size, type_size) = match dtype {
        GgmlDType::Q2K => (10i32, dtype.block_size() as i32, dtype.type_size() as i32),
        GgmlDType::Q4K => (12i32, dtype.block_size() as i32, dtype.type_size() as i32),
        _ => {
            candle::bail!(
                "moe_ptx_gguf: dtype {:?} not supported by fused PTX kernels \
                 (only Q2K/Q4K). Use reference backend for this quant.",
                dtype
            )
        }
    };

    if up.dtype() != dtype || down.dtype() != dtype {
        candle::bail!(
            "moe_ptx_gguf: gate/up/down dtype mismatch: {:?} / {:?} / {:?}",
            dtype,
            up.dtype(),
            down.dtype()
        );
    }

    let (n_tokens, n_embd) = input.dims2()?;
    if expert_ids.len() != n_tokens || weights.len() != n_tokens {
        candle::bail!(
            "moe_ptx_gguf: route plan length {} != n_tokens {}",
            expert_ids.len(),
            n_tokens
        );
    }

    let dev = input.device().as_cuda_device()?;

    // Flatten + sort by expert on CPU (PD-010).
    let topk = expert_ids.first().map_or(0, |v| v.len());
    let mut triples: Vec<(i32, i32, f32)> = Vec::with_capacity(n_tokens * topk);
    for (t, (eks, ws)) in expert_ids.iter().zip(weights.iter()).enumerate() {
        if eks.len() != ws.len() {
            candle::bail!("moe_ptx_gguf: expert/weight length mismatch at token {}", t);
        }
        for (e, w) in eks.iter().zip(ws.iter()) {
            triples.push((*e as i32, t as i32, *w));
        }
    }
    triples.sort_by_key(|(e, _, _)| *e);

    let m_total = triples.len();
    if m_total == 0 {
        let zeros = Tensor::zeros((n_tokens, n_embd), candle::DType::F32, input.device())?;
        return Ok(zeros);
    }

    let sorted_expert_ids: Vec<i32> = triples.iter().map(|(e, _, _)| *e).collect();
    let sorted_token_ids: Vec<i32> = triples.iter().map(|(_, t, _)| *t).collect();
    let sorted_weights: Vec<f32> = triples.iter().map(|(_, _, w)| *w).collect();

    // Input as CudaSlice<f32>, sliced from layout start_offset.
    let (input_storage, input_layout) = input.storage_and_layout();
    let input_slice = match &*input_storage {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("moe_ptx_gguf: input must be a cuda tensor"),
    };
    let input_view = input_slice.slice(input_layout.start_offset()..);

    // Raw device pointers for packed weights (PD-011: pass as &u64).
    let gate_ptr = gate.device_ptr()? as u64;
    let up_ptr = up.device_ptr()? as u64;
    let down_ptr = down.device_ptr()? as u64;

    // Upload sorted route arrays.
    let d_expert_ids = dev.clone_htod(&sorted_expert_ids)?;
    let d_token_ids = dev.clone_htod(&sorted_token_ids)?;
    let d_weights = dev.clone_htod(&sorted_weights)?;

    // Workspace: intermediate [m_total, n_ff] + output [n_tokens, n_embd].
    let mut intermediate = unsafe { dev.alloc::<f32>(m_total * n_ff)? };
    let mut output = dev.alloc_zeros::<f32>(n_tokens * n_embd)?;

    let block_dim: u32 = 256;

    // ── Launch moe_gate_up_kernel ───────────────────────────────────────────
    // Grid: (ceil(n_ff/256), m_total), Block: 256.
    let grid_gate = ((n_ff as u32 + block_dim - 1) / block_dim, m_total as u32, 1);
    let cfg_gate = LaunchConfig {
        grid_dim: grid_gate,
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let gate_up_func = dev.get_or_load_func("moe_gate_up_kernel", &kernels::MOE_QUANTIZED)?;
    let mut b1 = gate_up_func.builder();
    b1.arg(&input_view);
    b1.arg(&gate_ptr);
    b1.arg(&up_ptr);
    b1.arg(&d_expert_ids);
    b1.arg(&d_token_ids);
    b1.arg(&mut intermediate);
    candle::builder_arg!(
        b1, n_embd as i32, n_ff as i32, m_total as i32, quant_type, block_size, type_size
    );
    // SAFETY: kernel args match moe_gate_up_kernel signature.
    unsafe { b1.launch(cfg_gate) }.w()?;

    // ── Launch moe_down_proj_kernel (decode or prefill variant) ─────────────
    // Grid: (ceil(n_embd/256), m_total), Block: 256.
    let down_name = if is_prefill {
        "moe_prefill_down_proj_kernel"
    } else {
        "moe_down_proj_kernel"
    };
    let grid_down = ((n_embd as u32 + block_dim - 1) / block_dim, m_total as u32, 1);
    let cfg_down = LaunchConfig {
        grid_dim: grid_down,
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let down_func = dev.get_or_load_func(down_name, &kernels::MOE_QUANTIZED)?;
    let mut b2 = down_func.builder();
    b2.arg(&intermediate);
    b2.arg(&down_ptr);
    b2.arg(&d_expert_ids);
    b2.arg(&d_weights);
    b2.arg(&mut output);
    b2.arg(&d_token_ids);
    candle::builder_arg!(
        b2, n_ff as i32, n_embd as i32, m_total as i32, quant_type, block_size, type_size
    );
    // SAFETY: kernel args match moe_down_proj_kernel signature.
    unsafe { b2.launch(cfg_down) }.w()?;

    let out_storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    let out_tensor = Tensor::from_storage(
        candle::Storage::Cuda(out_storage),
        (n_tokens, n_embd),
        BackpropOp::none(),
        false,
    );
    Ok(out_tensor)
}

#[cfg(not(all(feature = "cuda", feature = "moe-cuda")))]
#[allow(clippy::too_many_arguments)]
pub fn moe_ptx_gguf(
    _: &Tensor,
    _: &QTensor,
    _: &QTensor,
    _: &QTensor,
    _: &[Vec<usize>],
    _: &[Vec<f32>],
    _: usize,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle::bail!("moe_ptx_gguf requires cuda + moe-cuda features")
}
