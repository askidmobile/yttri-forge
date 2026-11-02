//! Qwen3.6 35B-A3B MoE block.
//!
//! Router (top-k), packed routed experts, sigmoid-gated shared expert.
//! Reference backend dequantizes selected experts — diagnostics/parity only.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{ops::softmax_last_dim, Linear};
use std::sync::Arc;

// Выгрузка экспертов (план 2026-09-04-moe-expert-offload, фаза 2).
#[cfg(feature = "cuda")]
use super::expert_store::{MoeRuntime, SharedLayerStore};
#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// Где лежат маршрутизируемые эксперты MoE (PD-002).
pub enum ExpertWeights {
    /// Упакованные VRAM-тензоры: Reference-бэкенд и резидентный Ptx-путь.
    Packed(PackedExperts),
    /// Хранилище слоя при выгрузке (pinned host + таблицы указателей).
    #[cfg(feature = "cuda")]
    Store(SharedLayerStore),
}

// ─── Forward mode ─────────────────────────────────────────────────────────────

/// Distinguishes chunked prefill from batched decode.
///
/// Mathematical semantics are identical; only workspace/scratch selection
/// differs in the PTX backend (Phase 3). Reference backend ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardMode {
    Prefill,
    DecodeBatch,
}

/// Редукция выхода MoE по topk. На CUDA — одно ядро moe_weighted_sum_kernel
/// вместо пары `broadcast_mul` + `sum` движка: на профиле декода Ornith эти
/// два ядра стоили bmul 0.48 мс/токен и fast_sum 0.16 мс/токен.
#[cfg(feature = "cuda")]
fn moe_reduce(
    down: Tensor,
    w_t: &Tensor,
    topk: usize,
    cuda_dev: &candle_core::CudaDevice,
) -> Result<Tensor> {
    // down ранга 3: [tokens, topk, n_out]
    let (_tokens, topk_in, n_out) = down.dims3()?;
    if topk_in != topk {
        candle_core::bail!("moe_reduce: topk {topk_in} != {topk}");
    }
    candle_core::quantized::moe_weighted_sum(cuda_dev, &down, w_t, n_out, topk)
}

/// Порог, с которого MoE уходит на MMQ с группировкой. Ниже — группировка
/// (count/scan/sort + D2H) съедает больше, чем экономит на тайлах: у блока
/// один блок-сортировки плюс синхронизация на слой.
fn moe_mmq_min_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MOE_MMQ_MIN_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(32)
    })
}

/// Порог числа экспертов, для которого MMQ-обёртки вообще собраны.
fn moe_mmq_shape_ok(packed: &PackedExperts) -> bool {
    packed.n_experts <= 1024
}

// ─── Backend selection ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeBackend {
    /// CPU/Metal reference: dequantizes selected experts. Diagnostics/parity only.
    Reference,
    /// CUDA PTX sparse GEMM.
    Ptx,
}

pub(crate) fn select_backend(
    requested: Option<&str>,
    is_cuda: bool,
    gate: GgmlDType,
    up: GgmlDType,
    down: GgmlDType,
) -> Result<MoeBackend> {
    match requested {
        Some("reference") => return Ok(MoeBackend::Reference),
        Some("ptx") => {}
        Some(value) => {
            candle_core::bail!("invalid MOE_BACKEND={value:?}; expected ptx or reference")
        }
        None if !is_cuda => return Ok(MoeBackend::Reference),
        None => {}
    }
    if !is_cuda {
        candle_core::bail!("MOE_BACKEND=ptx requires CUDA")
    }
    let dual_supported = gate == up
        && matches!(
            gate,
            GgmlDType::IQ2S
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ3S
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ4XS
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q4K
                | GgmlDType::Q6K
        );
    let down_supported = matches!(
        down,
        GgmlDType::IQ2S
            | GgmlDType::IQ2XS
            | GgmlDType::IQ2XXS
            | GgmlDType::IQ3S
            | GgmlDType::IQ3XXS
            | GgmlDType::IQ4XS
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    );
    if !dual_supported || !down_supported {
        candle_core::bail!(
            "PTX MoE does not support routed dtypes gate={gate:?} up={up:?} down={down:?}"
        )
    }
    Ok(MoeBackend::Ptx)
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    const GATE: GgmlDType = GgmlDType::IQ2XXS;
    const DOWN: GgmlDType = GgmlDType::IQ2S;

    #[test]
    fn cuda_defaults_to_ptx_for_supported_dtypes() {
        assert_eq!(
            select_backend(None, true, GATE, GATE, DOWN).unwrap(),
            MoeBackend::Ptx
        );
    }

    #[test]
    fn reference_override_and_non_cuda_default_work() {
        assert_eq!(
            select_backend(Some("reference"), true, GATE, GATE, DOWN).unwrap(),
            MoeBackend::Reference
        );
        assert_eq!(
            select_backend(None, false, GATE, GATE, DOWN).unwrap(),
            MoeBackend::Reference
        );
    }

    #[test]
    fn required_quant_matrix_is_supported() {
        for (gate, down) in [
            (GgmlDType::IQ2XXS, GgmlDType::IQ3XXS),
            (GgmlDType::IQ2S, GgmlDType::IQ4XS),
            (GgmlDType::IQ2XS, GgmlDType::IQ3XXS),
            (GgmlDType::IQ3XXS, GgmlDType::IQ4XS),
            (GgmlDType::Q4K, GgmlDType::Q5K),
            (GgmlDType::Q4K, GgmlDType::Q6K),
        ] {
            assert_eq!(
                select_backend(Some("ptx"), true, gate, gate, down).unwrap(),
                MoeBackend::Ptx
            );
        }
    }

    #[test]
    fn explicit_ptx_rejects_non_cuda_or_unsupported_dtypes() {
        assert!(select_backend(Some("ptx"), false, GATE, GATE, DOWN).is_err());
        assert!(select_backend(Some("ptx"), true, GATE, GgmlDType::IQ2S, DOWN).is_err());
        assert!(select_backend(Some("unknown"), true, GATE, GATE, DOWN).is_err());
    }
}

// ─── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Qwen35MoeConfig {
    pub hidden_size: usize,
    pub n_experts: usize,
    pub n_experts_per_tok: usize,
    pub routed_intermediate: usize,
    pub shared_intermediate: usize,
    pub norm_topk_prob: bool,
}

// ─── Route plan ───────────────────────────────────────────────────────────────

/// Result of router top-k selection.
#[derive(Debug, Clone)]
pub struct RoutePlan {
    /// Selected expert indices per token: `[n_tokens][k]`.
    pub experts: Vec<Vec<usize>>,
    /// Normalized routing weights per token: `[n_tokens][k]`.
    pub weights: Vec<Vec<f32>>,
    /// Router logits `[n_tokens, n_experts]` F32.
    pub logits: Tensor,
    /// Softmax probabilities `[n_tokens, n_experts]` F32.
    pub probs: Tensor,
}

// ─── Router ───────────────────────────────────────────────────────────────────

/// F32-dequantized router linear with llama.cpp-compatible top-k.
pub struct MoeRouter {
    linear: Linear,
    n_experts: usize,
    n_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeRouter {
    pub fn new(
        linear: Linear,
        n_experts: usize,
        n_experts_per_tok: usize,
        norm_topk_prob: bool,
    ) -> Self {
        Self {
            linear,
            n_experts,
            n_experts_per_tok,
            norm_topk_prob,
        }
    }

    /// Compute router logits, softmax, stable top-k, and weight normalization.
    ///
    /// Matches llama.cpp qwen3moe router transform:
    /// ```text
    /// logits   = mm(ffn_gate_inp, x)     → [n_tokens, n_experts]
    /// probs    = softmax(logits)
    /// selected = top_k(probs, k)         — stable, tie-break by lower expert index
    /// weights  = gather(probs, selected)
    /// if norm_topk_prob: weights /= sum(weights)
    /// ```
    ///
    /// `arg_sort_last_dim` is **unstable** and would break parity — we use
    /// manual iterative argmax on CPU (probs are small: n_tokens × n_experts).
    pub fn route_topk(&self, xs: &Tensor) -> Result<RoutePlan> {
        let (n_tokens, _) = xs.dims2()?;
        let logits = self.linear.forward(xs)?.to_dtype(DType::F32)?;
        let probs = softmax_last_dim(&logits)?;

        // Stable top-k on CPU.
        let probs_cpu = probs.to_device(&Device::Cpu)?;
        let probs_rows: Vec<Vec<f32>> = probs_cpu.to_vec2()?;

        let k = self.n_experts_per_tok.min(self.n_experts);
        let mut experts = Vec::with_capacity(n_tokens);
        let mut weights = Vec::with_capacity(n_tokens);

        for row in &probs_rows {
            let mut masked = vec![false; row.len()];
            let mut selected: Vec<(usize, f32)> = Vec::with_capacity(k);

            for _ in 0..k {
                // Strict `>` → first max wins → tie-break by lower expert index.
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &p) in row.iter().enumerate() {
                    if !masked[i] && p > best_val {
                        best_val = p;
                        best_idx = i;
                    }
                }
                masked[best_idx] = true;
                selected.push((best_idx, best_val));
            }

            if self.norm_topk_prob {
                let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
                if sum > 0.0 {
                    for (_, w) in &mut selected {
                        *w /= sum;
                    }
                }
            }

            experts.push(selected.iter().map(|(e, _)| *e).collect());
            weights.push(selected.iter().map(|(_, w)| *w).collect());
        }

        Ok(RoutePlan {
            experts,
            weights,
            logits,
            probs,
        })
    }
}

// ─── Packed experts ───────────────────────────────────────────────────────────

/// Packed routed expert weights (quantized, not dequantized at load time).
///
/// GGUF shapes (after dimension reversal on read):
/// - `gate`: `[n_experts, n_ff, n_embd]`
/// - `up`:   `[n_experts, n_ff, n_embd]`
/// - `down`: `[n_experts, n_embd, n_ff]`
pub struct PackedExperts {
    pub gate: Arc<QTensor>,
    pub up: Arc<QTensor>,
    pub down: Arc<QTensor>,
    pub n_experts: usize,
}

// ─── Shared expert ────────────────────────────────────────────────────────────

/// Sigmoid-gated shared SwiGLU expert.
///
/// llama.cpp qwen2moe pattern (qwen35moe shares it):
/// ```text
/// cur_gate = silu(mm(gate_inp, x)) / mm(gate_inp, x)   [= sigmoid, bit-exact]
/// cur_ffn  = down(silu(gate(x)) * up(x))
/// out      = cur_ffn * cur_gate
/// ```
pub struct SharedExpert {
    /// `ffn_gate_inp_shexp` → `[n_tokens, 1]` scalar gate.
    gate_inp: Linear,
    /// `ffn_gate_shexp`.
    gate: QMatMul,
    /// `ffn_up_shexp`.
    up: QMatMul,
    /// `ffn_down_shexp`.
    down: QMatMul,
}

impl SharedExpert {
    pub fn new(gate_inp: Linear, gate: QMatMul, up: QMatMul, down: QMatMul) -> Self {
        Self {
            gate_inp,
            gate,
            up,
            down,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_logits = self.gate_inp.forward(xs)?; // [n_tokens, 1]
        let gate_scalar = silu_div(&gate_logits)?; // [n_tokens, 1]
        let gp = self.gate.forward(xs)?; // [n_tokens, n_ff_shexp]
        let up = self.up.forward(xs)?; // [n_tokens, n_ff_shexp]
        let act = gp.silu_mul_direct(&up)?; // [n_tokens, n_ff_shexp]
        let ffn = self.down.forward(&act)?; // [n_tokens, n_embd]
        ffn.broadcast_mul(&gate_scalar)
    }
}

/// `silu(x) / x` — llama.cpp's exact gate computation (numerically = sigmoid).
///
/// At `x = 0` the limit is `0.5`; we return `0.5` to avoid `0/0` NaN.
/// Bit-exact with llama.cpp for all nonzero `x` (same op sequence: silu then div).
/// $f(x) = \text{silu}(x)/x = \text{sigmoid}(x)$ — математически тождественно для всех $x$,
/// включая $x=0$ ($\text{sigmoid}(0) = 0.5$). Выполняется на GPU без CPU round-trip.
fn silu_div(xs: &Tensor) -> Result<Tensor> {
    candle_nn::ops::sigmoid(xs)
}

// ─── MoE block ────────────────────────────────────────────────────────────────

/// Qwen3.6 MoE feed-forward block.
pub struct Qwen35MoeBlock {
    router: MoeRouter,
    pub(crate) routed: ExpertWeights,
    shared: SharedExpert,
    backend: MoeBackend,
    #[allow(dead_code)]
    cfg: Qwen35MoeConfig,
    /// Индекс слоя ствола (след маршрутизации, стейджинг префила).
    layer_idx: usize,
    /// Общий рантайм MoE модели (след + стейджинг) — только при выгрузке.
    #[cfg(feature = "cuda")]
    runtime: Option<Arc<MoeRuntime>>,
    /// Capture только чистого GPU-подграфа shared + routed experts. CPU-MoE
    /// остаётся снаружи и перекрывается с replay.
    #[cfg(feature = "cuda")]
    gpu_graph: std::sync::Mutex<Option<MoeGpuGraphState>>,
    #[cfg(feature = "cuda")]
    gpu_graph_failed: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "cuda")]
struct MoeGpuGraphState {
    exec: cudarc::driver::sys::CUgraphExec,
    cu_graph: cudarc::driver::sys::CUgraph,
    stream: Arc<cudarc::driver::CudaStream>,
    x: Tensor,
    ids: Tensor,
    weights: Tensor,
    shared: Tensor,
    routed: Tensor,
    gpu_experts: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for MoeGpuGraphState {}

#[cfg(feature = "cuda")]
impl MoeGpuGraphState {
    fn stage_and_launch(
        &mut self,
        xs: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let ids = ids.narrow(1, 0, self.gpu_experts)?.contiguous()?;
        let weights = weights.narrow(1, 0, self.gpu_experts)?.contiguous()?;
        self.x.slice_set(xs, 0, 0)?;
        self.ids.slice_set(&ids, 0, 0)?;
        self.weights.slice_set(&weights, 0, 0)?;
        let res = unsafe { cudarc::driver::sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuGraphLaunch (MoE GPU) failed: {res:?}");
        }
        Ok((self.shared.clone(), self.routed.clone()))
    }
}

#[cfg(feature = "cuda")]
impl Drop for MoeGpuGraphState {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            let _ = cudarc::driver::sys::cuGraphDestroy(self.cu_graph);
        }
    }
}

impl Qwen35MoeBlock {
    pub fn new(
        cfg: Qwen35MoeConfig,
        router: MoeRouter,
        routed: PackedExperts,
        shared: SharedExpert,
        backend: MoeBackend,
    ) -> Self {
        Self {
            router,
            routed: ExpertWeights::Packed(routed),
            shared,
            backend,
            cfg,
            layer_idx: usize::MAX,
            #[cfg(feature = "cuda")]
            runtime: None,
            #[cfg(feature = "cuda")]
            gpu_graph: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            gpu_graph_failed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Полный конструктор: `new()` делегирует сюда с упакованными весами.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_weights(
        cfg: Qwen35MoeConfig,
        router: MoeRouter,
        routed: ExpertWeights,
        shared: SharedExpert,
        backend: MoeBackend,
        layer_idx: usize,
        runtime: Option<Arc<MoeRuntime>>,
    ) -> Self {
        Self {
            router,
            routed,
            shared,
            backend,
            cfg,
            layer_idx,
            runtime,
            gpu_graph: std::sync::Mutex::new(None),
            gpu_graph_failed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[cfg(feature = "cuda")]
    fn gpu_graph_enabled() -> bool {
        std::env::var("MOE_GPU_GRAPH").as_deref() == Ok("1")
    }

    #[cfg(feature = "cuda")]
    fn forward_gpu_routed(
        &self,
        cuda_dev: &candle_core::CudaDevice,
        st: &super::expert_store::ExpertLayerStore,
        xs: &Tensor,
        ids_t: &Tensor,
        w_t: &Tensor,
        gpu_experts: usize,
    ) -> Result<(Tensor, Tensor)> {
        let shared = self.shared.forward(xs)?;
        let routed = if gpu_experts > 0 {
            let ids_gpu = ids_t.narrow(1, 0, gpu_experts)?.contiguous()?;
            let weights_gpu = w_t.narrow(1, 0, gpu_experts)?.contiguous()?;
            let x3 = xs.to_dtype(DType::F32)?.unsqueeze(1)?.contiguous()?;
            let (gate_shape, down_shape) = st.shapes();
            let (gate, up) = candle_core::quantized::indexed_moe_forward_dual_table(
                cuda_dev,
                st.gate.dtype,
                gate_shape,
                st.gate.table(),
                st.up.table(),
                &x3,
                &ids_gpu,
            )?;
            let act = gate.silu_mul_direct(&up)?.contiguous()?;
            let down = candle_core::quantized::indexed_moe_forward_table(
                cuda_dev,
                st.down.dtype,
                down_shape,
                st.down.table(),
                &act,
                &ids_gpu,
            )?;
            moe_reduce(down, &weights_gpu, gpu_experts, cuda_dev)?
        } else {
            Tensor::zeros((1usize, 1usize, xs.dims2()?.1), DType::F32, xs.device())?
        };
        Ok((shared, routed))
    }

    #[cfg(feature = "cuda")]
    fn capture_gpu_graph(
        &self,
        cuda_dev: &candle_core::CudaDevice,
        st: &super::expert_store::ExpertLayerStore,
        xs: &Tensor,
        ids_t: &Tensor,
        w_t: &Tensor,
        gpu_experts: usize,
    ) -> Result<MoeGpuGraphState> {
        let (_, hidden) = xs.dims2()?;
        let x = Tensor::zeros((1usize, hidden), DType::F32, xs.device())?;
        let ids = Tensor::zeros((1usize, gpu_experts), DType::U32, xs.device())?;
        let weights = Tensor::zeros((1usize, gpu_experts), DType::F32, xs.device())?;
        let shared_out = Tensor::zeros((1usize, hidden), DType::F32, xs.device())?;
        let routed_out = Tensor::zeros((1usize, hidden), DType::F32, xs.device())?;
        x.slice_set(xs, 0, 0)?;
        ids.slice_set(&ids_t.narrow(1, 0, gpu_experts)?.contiguous()?, 0, 0)?;
        weights.slice_set(&w_t.narrow(1, 0, gpu_experts)?.contiguous()?, 0, 0)?;

        let stream = cuda_dev.cuda_stream();
        let _htod_guard = cuda_dev.enable_cuda_graph_htod_cache();
        unsafe {
            cudarc::driver::result::stream::begin_capture(
                stream.cu_stream(),
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        }
        .map_err(candle_core::Error::wrap)?;
        let (shared, routed) = match self.forward_gpu_routed(
            cuda_dev, st, &x, &ids, &weights, gpu_experts,
        ) {
            Ok(value) => value,
            Err(error) => {
                let _ = unsafe {
                    cudarc::driver::result::stream::end_capture(stream.cu_stream())
                };
                return Err(error);
            }
        };
        if let Err(error) = shared_out.slice_set(&shared, 0, 0) {
            let _ = unsafe {
                cudarc::driver::result::stream::end_capture(stream.cu_stream())
            };
            return Err(error);
        }
        if let Err(error) = routed_out.slice_set(&routed, 0, 0) {
            let _ = unsafe {
                cudarc::driver::result::stream::end_capture(stream.cu_stream())
            };
            return Err(error);
        }
        drop(shared);
        drop(routed);
        let cu_graph = unsafe {
            cudarc::driver::result::stream::end_capture(stream.cu_stream())
        }
        .map_err(candle_core::Error::wrap)?;
        if cu_graph.is_null() {
            candle_core::bail!("MoE GPU graph capture returned null graph");
        }
        let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        let res = unsafe {
            cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0)
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS || exec.is_null() {
            unsafe {
                let _ = cudarc::driver::sys::cuGraphDestroy(cu_graph);
            }
            candle_core::bail!("MoE GPU graph instantiate failed: {res:?}");
        }
        Ok(MoeGpuGraphState {
            exec,
            cu_graph,
            stream,
            x,
            ids,
            weights,
            shared: shared_out,
            routed: routed_out,
            gpu_experts,
        })
    }

    #[cfg(feature = "cuda")]
    fn forward_ptx_cuda_async(
        &self,
        cuda_dev: &candle_core::CudaDevice,
        store: &SharedLayerStore,
        xs: &Tensor,
        ids_t: &Tensor,
        w_t: &Tensor,
        gpu_experts: usize,
        prepared: Option<(&'static AsyncCpuStage, u32)>,
    ) -> Result<Tensor> {
        let prof = moe_prof::on().then(std::time::Instant::now);
        if let Some(error) = async_cpu_error().lock().expect("async CPU error").take() {
            candle_core::bail!("CPU-MoE async: {error}");
        }
        let (_, n_embd) = xs.dims2()?;
        let k_topk = ids_t.elem_count();
        let mapped = moe_cpu_mapped_enabled() && prepared.is_some();
        let (stage, seq) = match prepared {
            Some((stage, seq)) => (stage, seq),
            None => (
                async_cpu_stage(cuda_dev, self.layer_idx, k_topk, n_embd)?,
                async_cpu_next_seq(),
            ),
        };
        let ids_dev = crate::real::paged_kv_cuda::tensor_cuda_ptr(ids_t)?;
        let weights_dev = crate::real::paged_kv_cuda::tensor_cuda_ptr(w_t)?;
        let xs_dev = crate::real::paged_kv_cuda::tensor_cuda_ptr(xs)?;
        let stream = cuda_dev.cuda_stream();

        let (ids_host, weights_host, xs_host, out_host, copy_done, export_flag) = if mapped {
            (
                stage.mapped_ids.as_mut_ptr(),
                stage.mapped_weights.as_mut_ptr(),
                stage.mapped_xs.as_mut_ptr(),
                stage.mapped_out.as_mut_ptr(),
                None,
                stage.export_flag.ptr as usize,
            )
        } else {
            let ids_host = stage.ids.as_mut_ptr();
            let weights_host = stage.weights.as_mut_ptr();
            let xs_host = stage.xs.as_mut_ptr();
            unsafe {
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    ids_host.cast(),
                    ids_dev,
                    k_topk * std::mem::size_of::<u32>(),
                    stream.cu_stream(),
                );
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    weights_host.cast(),
                    weights_dev,
                    k_topk * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                );
                cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    xs_host.cast(),
                    xs_dev,
                    n_embd * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                );
            }
            let done = stream.record_event(None).map_err(candle_core::Error::wrap)?;
            (ids_host, weights_host, xs_host, stage.out.as_mut_ptr(), Some(done), 0)
        };

        async_cpu_sender()
            .send(AsyncCpuJob {
                store: store.clone(),
                copy_done,
                export_flag,
                ids: ids_host as usize,
                weights: weights_host as usize,
                xs: xs_host as usize,
                out: out_host as usize,
                flag: stage.flag.ptr as usize,
                seq,
                k_topk,
                n_embd,
                gpu_experts,
            })
            .map_err(|_| candle_core::Error::Msg("CPU-MoE async worker unavailable".into()))?;

        let (shared, gpu_routed) = self
            .gpu_graph
            .lock()
            .expect("MoE GPU graph")
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("MoE GPU graph missing".into()))?
            .stage_and_launch(xs, ids_t, w_t)?;
        let res = unsafe {
            cudarc::driver::sys::cuStreamWaitValue32_v2(
                stream.cu_stream(),
                stage.flag.dev_ptr,
                seq,
                cudarc::driver::sys::CUstreamWaitValue_flags_enum::CU_STREAM_WAIT_VALUE_GEQ
                    as u32,
            )
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle_core::bail!("cuStreamWaitValue32 (CPU-MoE) failed: {res:?}");
        }
        // Без memset: и mapped-ядро, и H2D полностью перезаписывают буфер,
        // а `zeros` добавлял отдельный kernel launch на каждый из 41 слоёв.
        let routed = {
            let buf = unsafe { cuda_dev.alloc::<f32>(n_embd)? };
            let storage = candle_core::CudaStorage::wrap_cuda_slice(buf, cuda_dev.clone());
            Tensor::from((
                candle_core::Storage::Cuda(storage),
                (1usize, 1usize, n_embd),
            ))
        };
        if mapped {
            let dst = crate::real::paged_kv_cuda::tensor_cuda_ptr(&routed)?;
            let gpu = crate::real::paged_kv_cuda::tensor_cuda_ptr(&gpu_routed)?;
            let shr = crate::real::paged_kv_cuda::tensor_cuda_ptr(&shared)?;
            let cpu = stage.mapped_out.dev_ptr();
            let func = cuda_dev.get_or_load_func(
                "moe_merge_mapped",
                &candle_kernels::MOE_ROUTER,
            )?;
            let cfg = LaunchConfig {
                grid_dim: ((n_embd.div_ceil(256)) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i = n_embd as i32;
            let mut builder = func.builder();
            builder.arg(&dst);
            builder.arg(&gpu);
            builder.arg(&shr);
            builder.arg(&cpu);
            builder.arg(&n_i);
            unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
            if let Some(t0) = prof {
                moe_prof::record(t0.elapsed().as_nanos() as u64);
            }
            Ok(routed)
        } else {
            let dst = crate::real::paged_kv_cuda::tensor_cuda_ptr(&routed)?;
            let rc = unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    stage.out.as_ptr() as *const _,
                    n_embd * std::mem::size_of::<f32>(),
                    stream.cu_stream(),
                )
            };
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                candle_core::bail!("CPU-MoE async H2D failed: {rc:?}");
            }
            let out = routed.broadcast_add(&gpu_routed)?.broadcast_add(&shared)?;
            if let Some(t0) = prof {
                moe_prof::record(t0.elapsed().as_nanos() as u64);
            }
            Ok(out)
        }
    }

    /// Forward: route → routed SwiGLU + shared expert → combine.
    ///
    /// `xs` shape: `[batch, seq_len, n_embd]` → output same shape.
    pub fn forward(&self, xs: &Tensor, mode: ForwardMode) -> Result<Tensor> {
        let (batch, seq_len, n_embd) = xs.dims3()?;
        let xs_2d = xs.reshape(((), n_embd))?;

        #[cfg(feature = "cuda")]
        if matches!(self.backend, MoeBackend::Ptx) && xs.device().is_cuda() {
            let combined = self.forward_ptx_cuda(&xs_2d, mode)?;
            return combined.reshape((batch, seq_len, n_embd));
        }

        let route = self.router.route_topk(&xs_2d)?;
        let t0 = std::time::Instant::now();
        // Reference/legacy-путь работает только с упакованными весами;
        // хранилище выгрузки ходит через forward_ptx_cuda.
        let packed = match &self.routed {
            ExpertWeights::Packed(p) => p,
            #[cfg(feature = "cuda")]
            ExpertWeights::Store(_) => {
                candle_core::bail!("хранилище выгрузки не поддерживает reference-бэкенд (FR-009)")
            }
        };
        let routed = self.backend.routed_swiglu(&xs_2d, packed, &route, mode)?;
        let shared = self.shared.forward(&xs_2d)?;
        if crate::scheduler::trace_on() && matches!(self.backend, MoeBackend::Ptx) {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let tot = TOTAL.fetch_add(
                t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            ) + t0.elapsed().as_micros() as u64;
            if n % 40 == 0 {
                eprintln!(
                    "[moe] routed_swiglu avg {:.2}ms over {} calls",
                    tot as f64 / n as f64 / 1000.0,
                    n
                );
            }
        }
        let combined = routed.broadcast_add(&shared)?;

        combined.reshape((batch, seq_len, n_embd))
    }

    /// Полностью GPU-путь (CUDA): router → GPU softmax+topk kernel →
    /// dual indexed GEMM (gate+up) → SwiGLU → down → weighted sum.
    /// Декод — без единого D2H/H2D round-trip; ids шага копируются в след
    /// маршрутизации (PD-003). При выгрузке (FR-004) ядра читают промахи
    /// zero-copy из pinned-памяти через таблицы указателей; префил (FR-005)
    /// идёт через стейджинг: один D2H ids на слой на чанк, подъём union,
    /// таблицы на время слоя.
    #[cfg(feature = "cuda")]
    fn forward_ptx_cuda(&self, xs: &Tensor, mode: ForwardMode) -> Result<Tensor> {
        use candle_core::quantized::{indexed_moe_forward_dual_table, indexed_moe_forward_table};
        let k = self.router.n_experts_per_tok;
        let device = xs.device();
        let cuda_dev = device.as_cuda_device()?;
        let (n_tokens, n_embd) = xs.dims2()?;

        // Router на GPU.
        let logits = self
            .router
            .linear
            .forward(xs)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        // След пишется только на декоде (префил ходит своими temp-ids).
        let trace = match (&self.runtime, mode) {
            (Some(rt), ForwardMode::DecodeBatch) => Some((rt.clone(), self.layer_idx)),
            _ => None,
        };
        let mapped_router = moe_cpu_mapped_enabled()
            && moe_cpu_async_enabled()
            && std::env::var("ROUTER_WARP").as_deref() != Ok("0")
            && moe_cpu_enabled()
            && matches!(mode, ForwardMode::DecodeBatch)
            && n_tokens == 1
            && self.router.n_experts == 256
            && k == 8
            && matches!(&self.routed, ExpertWeights::Store(_))
            && self.gpu_graph.lock().expect("MoE GPU graph").is_some();
        let prepared = if mapped_router {
            Some((
                async_cpu_stage(cuda_dev, self.layer_idx, k, n_embd)?,
                async_cpu_next_seq(),
            ))
        } else {
            None
        };
        let trace_view = trace.as_ref().map(|(rt, layer)| (rt.as_ref(), *layer));
        let (ids_t, w_t) = if let Some((stage, seq)) = prepared {
            gpu_softmax_topk_mapped(
                cuda_dev,
                &logits,
                self.router.n_experts,
                k,
                self.router.norm_topk_prob,
                trace_view,
                xs,
                stage,
                seq,
            )?
        } else {
            gpu_softmax_topk(
                cuda_dev,
                &logits,
                self.router.n_experts,
                k,
                self.router.norm_topk_prob,
                trace_view,
            )?
        };

        let make_x3 = || -> Result<Tensor> {
            // Indexed kernels reuse each token row across top-k routes.
            Ok(xs.to_dtype(DType::F32)?.unsqueeze(1)?.contiguous()?)
        };

        // Взвешивание GPU-весами + редукция по topk.
        let weighted_sum = |down: Tensor, w_t: &Tensor| -> Result<Tensor> {
            let w = w_t.unsqueeze(candle_core::D::Minus1)?; // [tokens, topk, 1]
            down.broadcast_mul(&w)?.sum(candle_core::D::Minus2)
        };
        let add_shared = |routed: Tensor, xs: &Tensor| -> Result<Tensor> {
            let shared = self.shared.forward(xs)?;
            routed.broadcast_add(&shared)
        };

        match (&self.routed, mode) {
            (ExpertWeights::Store(store), ForwardMode::DecodeBatch)
                if moe_cpu_enabled() && n_tokens == 1 =>
            {
                if moe_cpu_async_enabled()
                    && self.gpu_graph.lock().expect("MoE GPU graph").is_some()
                {
                    return self.forward_ptx_cuda_async(
                        cuda_dev,
                        store,
                        xs,
                        &ids_t,
                        &w_t,
                        moe_cpu_gpu_experts().min(ids_t.elem_count().saturating_sub(1)),
                        prepared,
                    );
                }
                // CPU-путь: сырые Q4_K байты уже в host-памяти. Раньше здесь
                // было ТРИ блокирующих D2H на слой (ids, веса маршрута, xs) —
                // 120 синхронизаций на токен. Ставим все копирования в один
                // поток и ждём ОДНО событие, затем читаем host-буферы.
                let timing = moe_cpu_timing_enabled();
                let t_layer = std::time::Instant::now();
                let k_topk = ids_t.elem_count();
                let n_embd_host = xs.dims2()?.1;
                let gpu_experts = moe_cpu_gpu_experts().min(k_topk.saturating_sub(1));
                let mut stage = cpu_decode_stage(cuda_dev, k_topk, n_embd_host)?;
                let stage = stage.as_mut().expect("CPU decode stage");
                let ids_ptr_host = stage.ids.as_mut_ptr();
                let w_ptr_host = stage.weights.as_mut_ptr();
                let xs_ptr_host = stage.xs.as_mut_ptr();
                let stream = cuda_dev.cuda_stream();
                let ids_ptr = crate::real::paged_kv_cuda::tensor_cuda_ptr(&ids_t)?;
                let w_ptr = crate::real::paged_kv_cuda::tensor_cuda_ptr(&w_t)?;
                let xs_ptr = crate::real::paged_kv_cuda::tensor_cuda_ptr(xs)?;
                let ids_ptr = ids_ptr as cudarc::driver::sys::CUdeviceptr;
                let w_ptr = w_ptr as cudarc::driver::sys::CUdeviceptr;
                let xs_ptr = xs_ptr as cudarc::driver::sys::CUdeviceptr;
                unsafe {
                    cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                        ids_ptr_host as *mut _,
                        ids_ptr,
                        k_topk * std::mem::size_of::<u32>(),
                        stream.cu_stream(),
                    );
                    cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                        w_ptr_host as *mut _,
                        w_ptr,
                        k_topk * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    );
                    cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                        xs_ptr_host as *mut _,
                        xs_ptr,
                        n_embd_host * std::mem::size_of::<f32>(),
                        stream.cu_stream(),
                    );
                }
                let copy_done = stream
                    .record_event(None)
                    .map_err(candle_core::Error::wrap)?;
                // Держим единственный guard на слое, пока CPU-воркер читает те же
                // immutable host weights. Указатель жив до завершения join.
                let store_guard = store.lock().expect("expert layer store");
                let store_addr = (&*store_guard as *const super::expert_store::ExpertLayerStore) as usize;
                let ids_addr = ids_ptr_host as usize;
                let weights_addr = w_ptr_host as usize;
                let xs_addr = xs_ptr_host as usize;
                let graph_enabled = Self::gpu_graph_enabled() && gpu_experts > 0;
                let (gpu_job, cpu_job) = std::thread::scope(|scope| {
                    let cpu_thread = scope.spawn(move || -> Result<(Vec<f32>, std::time::Duration, std::time::Duration)> {
                        let t_wait = std::time::Instant::now();
                        copy_done
                            .synchronize()
                            .map_err(candle_core::Error::wrap)?;
                        let wait_elapsed = t_wait.elapsed();
                        let ids_host = unsafe {
                            std::slice::from_raw_parts(ids_addr as *const u32, k_topk)
                        };
                        let w_host = unsafe {
                            std::slice::from_raw_parts(weights_addr as *const f32, k_topk)
                        };
                        let xs_cpu = unsafe {
                            std::slice::from_raw_parts(xs_addr as *const f32, n_embd_host)
                        };
                        let st = unsafe {
                            &*(store_addr as *const super::expert_store::ExpertLayerStore)
                        };
                        let t_cpu = std::time::Instant::now();
                        let routed = rayon::scope(|_| {
                            moe_forward_cpu(
                                st,
                                &ids_host[gpu_experts..],
                                &w_host[gpu_experts..],
                                xs_cpu,
                                n_embd_host,
                            )
                        })?;
                        Ok((routed, wait_elapsed, t_cpu.elapsed()))
                    });
                    // CUDA-ветку обязательно ставим на вызывающем потоке; worker
                    // только ждёт D2H event и считает CPU experts.
                    let gpu_job = (|| -> Result<(Tensor, Tensor, std::time::Duration, Option<cudarc::driver::CudaEvent>, bool)> {
                        let st = unsafe {
                            &*(store_addr as *const super::expert_store::ExpertLayerStore)
                        };
                        let replayed = if graph_enabled {
                            self.gpu_graph
                                .lock()
                                .expect("MoE GPU graph")
                                .as_mut()
                                .map(|state| state.stage_and_launch(xs, &ids_t, &w_t))
                                .transpose()?
                        } else {
                            None
                        };
                        let (shared, gpu_routed, capture) = match replayed {
                            Some((shared, routed)) => (shared, routed, false),
                            None => {
                                let (shared, routed) = self.forward_gpu_routed(
                                    cuda_dev, st, xs, &ids_t, &w_t, gpu_experts,
                                )?;
                                let capture = graph_enabled
                                    && !self
                                        .gpu_graph_failed
                                        .load(std::sync::atomic::Ordering::Acquire);
                                (shared, routed, capture)
                            }
                        };
                        let gpu_done = if timing {
                            Some(stream.record_event(None).map_err(candle_core::Error::wrap)?)
                        } else {
                            None
                        };
                        Ok((shared, gpu_routed, t_layer.elapsed(), gpu_done, capture))
                    })();
                    let cpu_job = cpu_thread
                        .join()
                        .map_err(|_| candle_core::Error::Msg("CPU-MoE worker panicked".into()))
                        .and_then(|job| job);
                    (gpu_job, cpu_job)
                });
                let (shared, gpu_routed, enqueue_elapsed, gpu_done, capture) = gpu_job?;
                let (routed_cpu, wait_elapsed, cpu_elapsed) = cpu_job?;
                if capture {
                    stream.synchronize().map_err(candle_core::Error::wrap)?;
                    let st = &*store_guard;
                    match self.capture_gpu_graph(
                        cuda_dev, st, xs, &ids_t, &w_t, gpu_experts,
                    ) {
                        Ok(state) => {
                            *self.gpu_graph.lock().expect("MoE GPU graph") = Some(state);
                            eprintln!("[moe-graph] captured layer {}", self.layer_idx);
                        }
                        Err(error) => {
                            self.gpu_graph_failed
                                .store(true, std::sync::atomic::Ordering::Release);
                            eprintln!("[moe-graph] layer {} capture failed: {error}", self.layer_idx);
                        }
                    }
                }
                drop(store_guard);
                let d2h_elapsed = t_layer.elapsed();
                let out_host = stage.out.as_mut_ptr().map_err(candle_core::Error::wrap)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(routed_cpu.as_ptr(), out_host, n_embd_host);
                }
                let routed = Tensor::zeros((1usize, 1usize, n_embd_host), DType::F32, xs.device())?;
                let dst = crate::real::paged_kv_cuda::tensor_cuda_ptr(&routed)?;
                let out_src = stage.out.as_ptr().map_err(candle_core::Error::wrap)?;
                let rc = unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        dst,
                        out_src as *const _,
                        n_embd_host * std::mem::size_of::<f32>(),
                        cuda_dev.cuda_stream().cu_stream(),
                    )
                };
                if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    candle_core::bail!("CPU-MoE H2D: {rc:?}");
                }
                if timing {
                    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    static D2H: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    static CPU: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    static ENQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    static WAIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    static READY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let d2h_us = d2h_elapsed.as_micros() as u64;
                    let cpu_us = cpu_elapsed.as_micros() as u64;
                    let enq_us = enqueue_elapsed.as_micros() as u64;
                    let wait_us = wait_elapsed.as_micros() as u64;
                    let ready = gpu_done.as_ref().is_some_and(|event| event.is_complete());
                    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    D2H.fetch_add(d2h_us, std::sync::atomic::Ordering::Relaxed);
                    CPU.fetch_add(cpu_us, std::sync::atomic::Ordering::Relaxed);
                    ENQ.fetch_add(enq_us, std::sync::atomic::Ordering::Relaxed);
                    WAIT.fetch_add(wait_us, std::sync::atomic::Ordering::Relaxed);
                    READY.fetch_add(ready as u64, std::sync::atomic::Ordering::Relaxed);
                    if n % 41 == 0 {
                        let d = D2H.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 41.0;
                        let c = CPU.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 41.0;
                        let e = ENQ.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 41.0;
                        let w = WAIT.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 41.0;
                        let r = READY.swap(0, std::sync::atomic::Ordering::Relaxed);
                        eprintln!(
                            "[moe-cpu] d2h={d:.1}us enqueue={e:.1}us wait={w:.1}us cpu={c:.1}us gpu_ready={r}/41"
                        );
                    }
                }
                return Ok(routed.broadcast_add(&gpu_routed)?.broadcast_add(&shared)?);
            }
            (ExpertWeights::Store(store), ForwardMode::DecodeBatch) => {
                let x3 = make_x3()?;
                let store = store.lock().expect("expert layer store");
                let (g_shape, d_shape) = store.shapes();
                // gate+up одним dual GEMM; промахи ядро читает zero-copy.
                let (gate, up) = indexed_moe_forward_dual_table(
                    cuda_dev,
                    store.gate.dtype,
                    g_shape,
                    store.gate.table(),
                    store.up.table(),
                    &x3,
                    &ids_t,
                )?;
                let act = gate.silu_mul_direct(&up)?.contiguous()?;
                let down = indexed_moe_forward_table(
                    cuda_dev,
                    store.down.dtype,
                    d_shape,
                    store.down.table(),
                    &act,
                    &ids_t,
                )?; // [tokens, topk, n_embd]
                drop(store);
                add_shared(weighted_sum(down, &w_t)?, xs)
            }
            (ExpertWeights::Store(store), ForwardMode::Prefill) => {
                let x3 = make_x3()?;
                let runtime = self.runtime.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("moe runtime отсутствует при выгрузке".to_string())
                })?;
                let staging = runtime.staging().ok_or_else(|| {
                    candle_core::Error::Msg(
                        "стейджинг не выделен: init_paged_decode должен пройти раньше первого префила (PD-010)"
                            .to_string(),
                    )
                })?;
                // FR-005: один D2H ids на слой на чанк.
                let ids_host = ids_t
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<u32>()?;
                // Фаза 4: кэш (свободные слоты → стейджинг); таблицы
                // полностью перезаписываются подготовкой.
                let staging_experts = match runtime.cache.get() {
                    Some(cache) => {
                        cache.prefill_prepare_layer(
                            cuda_dev,
                            store,
                            self.layer_idx,
                            staging,
                            &ids_host,
                        )?
                    }
                    None => {
                        let mut store = store.lock().expect("expert layer store");
                        store.prefill_prepare_layer(cuda_dev, staging, &ids_host)?;
                        Vec::new()
                    }
                };
                let mut st = store.lock().expect("expert layer store");
                let (g_shape, d_shape) = st.shapes();
                let (gate, up) = indexed_moe_forward_dual_table(
                    cuda_dev,
                    st.gate.dtype,
                    g_shape,
                    st.gate.table(),
                    st.up.table(),
                    &x3,
                    &ids_t,
                )?;
                let act = gate.silu_mul_direct(&up)?.contiguous()?;
                let down = indexed_moe_forward_table(
                    cuda_dev,
                    st.down.dtype,
                    d_shape,
                    st.down.table(),
                    &act,
                    &ids_t,
                )?;
                // После ядер слоя: staging-эксперты → host (буфер стейджинга
                // перезапишет следующий слой); кэш-записи сохраняются.
                drop(st);
                match runtime.cache.get() {
                    Some(cache) => {
                        cache.prefill_release_staging(cuda_dev, &store, &staging_experts)?;
                    }
                    None => {
                        let mut st = store.lock().expect("expert layer store");
                        st.prefill_release_layer(cuda_dev)?;
                    }
                }
                add_shared(weighted_sum(down, &w_t)?, xs)
            }
            (ExpertWeights::Packed(packed), mode) => {
                // ── MMQ на тензорных ядрах с группировкой по экспертам ──
                // Только префил и только когда токенов достаточно: группировка
                // делает D2H за слой, а на декоде это ломает захват CUDA-графа
                // и при m=1 всё равно невыгодно (8 пар на 256 экспертов).
                // MMQ-путь: префил — всегда (там группировка снимает
                // перечитывание весов эксперта, 1074 -> 7459 t/s).
                //
                // Декод идёт на MMVQ, и это не недосмотр: замер Ornith на
                // RTX 4090, ctx 32768, decode — MMQ 89.9 t/s против MMVQ
                // 125.0 t/s. На одном токене у эксперта одна пара, MMQ читает
                // те же веса, но гонит их через shared memory и считает тайл
                // из mmq_x=32 колонок ради одной — чистые накладные расходы.
                // Ветка оставлена под MOE_MMQ_DECODE=1 (захват CUDA-графа при
                // ней проходит: 5861 узлов, res=CUDA_SUCCESS).
                let m_total = n_tokens * k;
                let mmq_ok = match mode {
                    ForwardMode::Prefill => n_tokens >= moe_mmq_min_tokens(),
                    ForwardMode::DecodeBatch => {
                        std::env::var("MOE_MMQ_DECODE").map(|v| v == "1").unwrap_or(false)
                            && m_total <= candle_core::quantized::moe_mmq_x_for(m_total)
                    }
                };
                if mmq_ok
                    && candle_core::quantized::moe_mmq_enabled()
                    && moe_mmq_shape_ok(packed)
                {
                    use candle_core::quantized::{moe_grouping, moe_weighted_combine};
                    let (ids_st, ids_lay) = ids_t.storage_and_layout();
                    let ids_v = match &*ids_st {
                        candle_core::Storage::Cuda(c) => {
                            c.as_cuda_slice::<u32>()?.slice(ids_lay.start_offset()..)
                        }
                        _ => candle_core::bail!("moe mmq: ids не на CUDA"),
                    };
                    let (w_st, w_lay) = w_t.storage_and_layout();
                    let w_v = match &*w_st {
                        candle_core::Storage::Cuda(c) => {
                            c.as_cuda_slice::<f32>()?.slice(w_lay.start_offset()..)
                        }
                        _ => candle_core::bail!("moe mmq: веса маршрута не на CUDA"),
                    };
                    let group = moe_grouping(
                        cuda_dev,
                        &ids_v,
                        &w_v,
                        n_tokens,
                        k,
                        packed.n_experts,
                        mode == ForwardMode::Prefill,
                    )?;
                    let x_f32 = xs.to_dtype(DType::F32)?.contiguous()?;
                    let gate = packed.gate.moe_mmq_project_cuda(&x_f32, &group, true)?;
                    let up = packed.up.moe_mmq_project_cuda(&x_f32, &group, true)?;
                    let act = gate.silu_mul_direct(&up)?.contiguous()?;
                    let down = packed.down.moe_mmq_project_cuda(&act, &group, false)?;
                    // dims2 = [m_total, n_out] — берём ИМЕННО ширину.
                    let (_, n_out) = down.dims2()?;
                    if std::env::var_os("MOE_MMQ_DEBUG").is_some() {
                        eprintln!(
                            "[moe-mmq] layer={} mode={:?} m_total={} ncols_max={} \
gate={:?} up={:?} down={:?} x={:?} n_out={}",
                            self.layer_idx,
                            mode,
                            group.m_total(),
                            group.ncols_max,
                            packed.gate.shape().dims(),
                            packed.up.shape().dims(),
                            packed.down.shape().dims(),
                            x_f32.dims(),
                            n_out,
                        );
                    }
                    let routed = moe_weighted_combine(cuda_dev, &group, &down, n_out)?;
                    return add_shared(routed, xs);
                }
                // Резидентный путь: те же ядра через ленивую таблицу упаковки.
                // MOE_GROUPED=1 — grouped-вариант ядер (вес эксперта читается
                // один раз на все его токены). У dual-ядра grouped-инстанса
                // нет, поэтому на нём gate и up идут двумя отдельными
                // запусками: вход квантуется дважды, но это [tokens, n_embd]
                // против [n_experts, n_ff, n_embd] чтений весов.
                // MOE_GROUPED читается на каждом MoE-слое на токен (41 раз). Значение
                // неизменно в процессе — кэшируем.
                let x3 = make_x3()?;
                let grouped = {
                    static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    *G.get_or_init(|| std::env::var("MOE_GROUPED").map(|v| v != "0").unwrap_or(false))
                };
                let (gate, up) = if grouped {
                    let g = packed.gate.indexed_moe_forward_cuda(&x3, &ids_t)?;
                    let u = packed.up.indexed_moe_forward_cuda(&x3, &ids_t)?;
                    (g, u)
                } else {
                    packed
                        .gate
                        .indexed_moe_forward_dual_cuda(&packed.up, &x3, &ids_t)?
                };
                let act = gate.silu_mul_direct(&up)?.contiguous()?;
                let down = packed.down.indexed_moe_forward_cuda(&act, &ids_t)?; // [tokens, topk, n_embd]
                add_shared(moe_reduce(down, &w_t, k, cuda_dev)?, xs)
            }
        }
    }

    /// Config accessor for diagnostics / adapter reporting.
    pub fn config(&self) -> &Qwen35MoeConfig {
        &self.cfg
    }
}

impl MoeBackend {
    fn routed_swiglu(
        &self,
        xs: &Tensor,
        experts: &PackedExperts,
        route: &RoutePlan,
        _mode: ForwardMode,
    ) -> Result<Tensor> {
        match self {
            Self::Reference => reference_routed_swiglu(xs, experts, route),
            Self::Ptx => ptx_routed_swiglu(xs, experts, route),
        }
    }
}

/// Fused MoE путь (CUDA): indexed_moe_forward ядра — одна группа запусков на
/// проекцию вместо per-expert dequantize_rowslice (~1000 launches/шаг).
/// Работает для routed IQ2/IQ3/IQ4_XS и K-quants/Q8_0 экспертов.
#[cfg(feature = "cuda")]
fn ptx_routed_swiglu(xs: &Tensor, experts: &PackedExperts, route: &RoutePlan) -> Result<Tensor> {
    // Legacy route-plan PTX path; production CUDA path validates routed dtypes at load.
    let supported = |qt: &Arc<QTensor>| {
        matches!(
            qt.dtype(),
            GgmlDType::IQ3S
                | GgmlDType::IQ2S
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ4XS
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        )
    };
    if !(supported(&experts.gate) && supported(&experts.up) && supported(&experts.down)) {
        return reference_routed_swiglu(xs, experts, route);
    }
    let (n_tokens, n_embd) = xs.dims2()?;
    let topk = route.experts.first().map(|e| e.len()).unwrap_or(0);
    if n_tokens == 0 || topk == 0 {
        return Tensor::zeros((n_tokens, n_embd), DType::F32, xs.device());
    }
    let device = xs.device();

    // ids [n_tokens, topk] u32 на device.
    let ids_flat: Vec<u32> = route
        .experts
        .iter()
        .flat_map(|e| e.iter().map(|&x| x as u32))
        .collect();
    let ids_t = Tensor::from_vec(ids_flat, (n_tokens, topk), device)?;

    // xs → [n_tokens, topk, n_embd] contiguous (q8_1 quantize внутри ядра).
    let x3 = xs
        .to_dtype(DType::F32)?
        .unsqueeze(1)?
        .broadcast_as((n_tokens, topk, n_embd))?
        .contiguous()?;

    // gate/up: [n_tokens, topk, n_ff]; SwiGLU; down: [n_tokens, topk, n_embd].
    let gate = experts.gate.indexed_moe_forward_cuda(&x3, &ids_t)?;
    let up = experts.up.indexed_moe_forward_cuda(&x3, &ids_t)?;
    let act = gate.silu_mul_direct(&up)?.contiguous()?;
    let down = experts.down.indexed_moe_forward_cuda(&act, &ids_t)?;

    // Взвешивание route weights и редукция по topk → [n_tokens, n_embd].
    let w_flat: Vec<f32> = route.weights.iter().flatten().copied().collect();
    let w = Tensor::from_vec(w_flat, (n_tokens, topk, 1), device)?;
    down.broadcast_mul(&w)?.sum(candle_core::D::Minus2)
}

#[cfg(not(feature = "cuda"))]
fn ptx_routed_swiglu(_xs: &Tensor, _experts: &PackedExperts, _route: &RoutePlan) -> Result<Tensor> {
    candle_core::bail!("PTX MoE backend requires cuda feature")
}

/// Reference routed expert SwiGLU: dequantize packed experts, batch per-expert.
///
/// Diagnostics/parity only — dequantizes the full packed tensor. Production
/// uses PTX sparse GEMM (Phase 3) which consumes packed storage directly.
///
/// Groups tokens by selected expert, batch-matmuls each expert's tokens,
/// weights the output, and scatter-adds (`index_add`) into the result.
fn reference_routed_swiglu(
    xs: &Tensor,
    experts: &PackedExperts,
    route: &RoutePlan,
) -> Result<Tensor> {
    let device = xs.device();
    let xs = xs.to_dtype(DType::F32)?;
    let (n_tokens, n_embd) = xs.dims2()?;
    let n_experts = experts.n_experts;

    // Per-expert dequantize on demand. The packed shapes are
    // gate/up: [n_experts, n_ff, n_embd], down: [n_experts, n_embd, n_ff].
    // For expert `e`, the rowslice [e, e+1) of the 2D view [n_experts, n_ff*n_embd]
    // gives a [1, n_ff*n_embd] f32 tensor; reshape to [n_ff, n_embd].
    // This avoids dequantizing all 256 experts to f32 at once (the OOM cause).
    // n_ff is recovered from the gate shape dims().
    let gate_dims = experts.gate.shape().dims();
    // Packed shape can be [n_experts, n_ff, n_embd] (3D) or a flat [n_experts*n_ff*n_embd].
    let n_ff = match gate_dims {
        [_, a, _b] => *a, // [n_experts, n_ff, n_embd]
        _ => candle_core::bail!(
            "reference_routed_swiglu: unexpected gate rank {:?}",
            gate_dims
        ),
    };
    // down shape is [n_experts, n_embd, n_ff] — n_ff is dims[2], not dims[1].
    let down_dims = experts.down.shape().dims();
    match down_dims {
        [_, _embd, ff] => {
            if *ff != n_ff {
                candle_core::bail!(
                    "reference_routed_swiglu: n_ff mismatch gate {} vs down {}",
                    n_ff,
                    ff
                );
            }
        }
        _ => candle_core::bail!(
            "reference_routed_swiglu: unexpected down rank {:?}",
            down_dims
        ),
    };
    let gate_k = n_ff * n_embd; // cols of 2D view [n_experts, n_ff*n_embd]

    // Group tokens by selected expert: expert → [(token_idx, weight)].
    let mut expert_tokens: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
    for (t, (eks, ws)) in route.experts.iter().zip(route.weights.iter()).enumerate() {
        for (e, w) in eks.iter().zip(ws.iter()) {
            expert_tokens[*e].push((t, *w));
        }
    }

    let mut out = Tensor::zeros((n_tokens, n_embd), DType::F32, device)?;
    for e in 0..n_experts {
        let tokens = &expert_tokens[e];
        if tokens.is_empty() {
            continue;
        }
        let token_ids: Vec<u32> = tokens.iter().map(|(t, _)| *t as u32).collect();
        let weights: Vec<f32> = tokens.iter().map(|(_, w)| *w).collect();

        let idx = Tensor::from_vec(token_ids.clone(), token_ids.len(), device)?;
        let x_subset = xs.index_select(&idx, 0)?; // [n_sel, n_embd]

        // Dequantize just expert e: rowslice [e, e+1) → [1, n_ff*n_embd] → squeeze [n_ff, n_embd].
        let gate_w = experts
            .gate
            .dequantize_rowslice(e, e + 1, gate_k, device)?
            .reshape((n_ff, n_embd))?;
        let up_w = experts
            .up
            .dequantize_rowslice(e, e + 1, gate_k, device)?
            .reshape((n_ff, n_embd))?;
        let down_w = experts
            .down
            .dequantize_rowslice(e, e + 1, gate_k, device)?
            .reshape((n_embd, n_ff))?;

        let gate_out = x_subset.matmul(&gate_w.t()?)?; // [n_sel, n_ff]
        let up_out = x_subset.matmul(&up_w.t()?)?; // [n_sel, n_ff]
        let act = gate_out.silu_mul_direct(&up_out)?; // [n_sel, n_ff]
        let expert_out = act.matmul(&down_w.t()?)?; // [n_sel, n_embd]

        let w_len = weights.len();
        let w = Tensor::from_vec(weights, (w_len, 1), device)?;
        let weighted = expert_out.broadcast_mul(&w)?; // [n_sel, n_embd]

        out = out.index_add(&idx, &weighted, 0)?;
    }
    Ok(out)
}

// ─── Construction helpers (used by model_weights.rs and tests) ────────────────

/// Create an F32 `QTensor` from an F32 `Tensor` on CPU.
///
/// For synthetic test experts and small reference fixtures. The source tensor
/// **must** be on `Device::Cpu` (required by `QTensor::quantize_onto`).
pub fn f32_qtensor(t: &Tensor) -> Result<Arc<QTensor>> {
    use candle_core::quantized::GgmlDType;
    let qt = QTensor::quantize_onto(t, GgmlDType::F32, &Device::Cpu)?;
    Ok(Arc::new(qt))
}
// ─── GPU softmax+topk dispatch (CUDA, moe_router.cu) ─────────────────────────

/// Stable softmax+topk на GPU: logits [n_tokens, n_experts] F32 →
/// (ids [n_tokens, k] u32, weights [n_tokens, k] f32) — без D2H.
/// `trace` — (рантайм MoE, индекс слоя): после ядра ids копируются в
/// постоянный буфер следа (PD-003) — d2d копия захватывается в граф.
#[cfg(feature = "cuda")]
fn gpu_softmax_topk(
    dev: &candle_core::CudaDevice,
    logits: &Tensor,
    n_experts: usize,
    topk: usize,
    norm_topk_prob: bool,
    trace: Option<(&super::expert_store::MoeRuntime, usize)>,
) -> Result<(Tensor, Tensor)> {
    gpu_softmax_topk_sel(dev, logits, n_experts, topk, norm_topk_prob, trace, None)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_softmax_topk_mapped(
    dev: &candle_core::CudaDevice,
    logits: &Tensor,
    n_experts: usize,
    topk: usize,
    norm_topk_prob: bool,
    trace: Option<(&super::expert_store::MoeRuntime, usize)>,
    xs: &Tensor,
    stage: &'static AsyncCpuStage,
    seq: u32,
) -> Result<(Tensor, Tensor)> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let (n_tokens, _) = logits.dims2()?;
    if n_tokens != 1 || n_experts > 256 || topk > 8 {
        candle_core::bail!("mapped router requires one token, <=256 experts and topk<=8");
    }
    let (l_st, l_lay) = logits.storage_and_layout();
    let logits_view = match &*l_st {
        candle_core::Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(l_lay.start_offset()..),
        _ => candle_core::bail!("mapped router: logits not CUDA"),
    };
    let ids = unsafe { dev.alloc::<u32>(topk)? };
    let mut weights = unsafe { dev.alloc::<f32>(topk)? };
    let ids_i32 = unsafe { ids.transmute::<i32>(topk) }
        .ok_or_else(|| candle_core::Error::Msg("mapped router ids transmute".into()))?;
    let n_experts_i = n_experts as i32;
    let topk_i = topk as i32;
    let norm_i = norm_topk_prob as i32;
    let n_embd_i = xs.dim(1)? as i32;
    let xs_ptr = crate::real::paged_kv_cuda::tensor_cuda_ptr(xs)?;
    let host_ids = stage.mapped_ids.dev_ptr();
    let host_weights = stage.mapped_weights.dev_ptr();
    let host_xs = stage.mapped_xs.dev_ptr();
    let export_flag = stage.export_flag.dev_ptr;
    let func = dev.get_or_load_func(
        "moe_softmax_topk_warp_export_kernel",
        &candle_kernels::MOE_ROUTER,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&logits_view);
    builder.arg(&ids_i32);
    builder.arg(&mut weights);
    builder.arg(&xs_ptr);
    builder.arg(&host_ids);
    builder.arg(&host_weights);
    builder.arg(&host_xs);
    builder.arg(&export_flag);
    builder.arg(&n_experts_i);
    builder.arg(&topk_i);
    builder.arg(&norm_i);
    builder.arg(&n_embd_i);
    builder.arg(&seq);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    drop(l_st);

    if let Some((runtime, layer)) = trace {
        runtime.trace.copy_in(dev, layer, &ids, 1)?;
    }
    let ids_storage = candle_core::CudaStorage::wrap_cuda_slice(ids, dev.clone());
    let ids_t = Tensor::from((candle_core::Storage::Cuda(ids_storage), (1usize, topk)));
    let w_storage = candle_core::CudaStorage::wrap_cuda_slice(weights, dev.clone());
    let w_t = Tensor::from((candle_core::Storage::Cuda(w_storage), (1usize, topk)));
    Ok((ids_t, w_t))
}

/// `force_warp`: None — по env ROUTER_WARP, Some(b) — принудительно.
/// Разделение нужно регресс-тесту: менять env в параллельных тестах нельзя.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_softmax_topk_sel(
    dev: &candle_core::CudaDevice,
    logits: &Tensor,
    n_experts: usize,
    topk: usize,
    norm_topk_prob: bool,
    trace: Option<(&super::expert_store::MoeRuntime, usize)>,
    force_warp: Option<bool>,
) -> Result<(Tensor, Tensor)> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};
    let (n_tokens, _) = logits.dims2()?;
    let (l_st, l_lay) = logits.storage_and_layout();
    let logits_view = match &*l_st {
        candle_core::Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(l_lay.start_offset()..),
        _ => candle_core::bail!("gpu_softmax_topk: logits not CUDA"),
    };

    let ids = unsafe { dev.alloc::<u32>(n_tokens * topk)? };
    let mut weights = unsafe { dev.alloc::<f32>(n_tokens * topk)? };

    // ROUTER_WARP=0 — откат на прежний блочный вариант.
    let warp_router = force_warp
        .unwrap_or_else(|| std::env::var("ROUTER_WARP").as_deref() != Ok("0"));
    let norm_flag: i32 = if norm_topk_prob { 1 } else { 0 };
    let n_experts_i = n_experts as i32;
    let topk_i = topk as i32;
    // kernel ждёт int32_t* для ids — transmute view u32→i32.
    let ids_i32 = unsafe { ids.transmute::<i32>(n_tokens * topk) }
        .ok_or_else(|| candle_core::Error::Msg("ids transmute".into()))?;
    if warp_router {
        // Одно warp на токен, 4 строки на блок: без shared и барьеров.
        let func =
            dev.get_or_load_func("moe_softmax_topk_warp_kernel", &candle_kernels::MOE_ROUTER)?;
        let rows_per_block = 4u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_tokens as u32).div_ceil(rows_per_block), 1, 1),
            block_dim: (32, rows_per_block, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&logits_view);
        b.arg(&ids_i32);
        b.arg(&mut weights);
        b.arg(&n_experts_i);
        b.arg(&topk_i);
        b.arg(&norm_flag);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    } else {
        let func = dev.get_or_load_func("moe_softmax_topk_kernel", &candle_kernels::MOE_ROUTER)?;
        let block = 256u32; // >= n_experts(256), power of 2
        let shared = (n_experts * 4 + n_experts) as u32; // probs f32 + masked u8
        let cfg = LaunchConfig {
            grid_dim: (n_tokens as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        };
        let mut b = func.builder();
        b.arg(&logits_view);
        b.arg(&ids_i32);
        b.arg(&mut weights);
        b.arg(&n_experts_i);
        b.arg(&topk_i);
        b.arg(&norm_flag);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }
    drop(l_st);

    // PD-003: ids шага → постоянный буфер следа (d2d, захватывается в граф).
    if let Some((runtime, layer)) = trace {
        runtime.trace.copy_in(dev, layer, &ids, n_tokens)?;
    }

    let ids_storage = candle_core::CudaStorage::wrap_cuda_slice(ids, dev.clone());
    let ids_t = Tensor::from((candle_core::Storage::Cuda(ids_storage), (n_tokens, topk)));
    let w_storage = candle_core::CudaStorage::wrap_cuda_slice(weights, dev.clone());
    let w_t = Tensor::from((candle_core::Storage::Cuda(w_storage), (n_tokens, topk)));
    Ok((ids_t, w_t))
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_router_tests {
    use super::*;

    #[test]
    fn gpu_router_matches_stable_cpu_top8() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let n_tokens = 4usize;
        let n_experts = 256usize;
        let topk = 8usize;
        let logits: Vec<f32> = (0..n_tokens * n_experts)
            .map(|i| {
                let token = i / n_experts;
                let expert = i % n_experts;
                (((expert * 73 + token * 29) % 997) as f32 - 498.0) / 37.0
            })
            .collect();
        let logits_t = Tensor::from_slice(&logits, (n_tokens, n_experts), &device)?;
        // Регресс-гард: warp-роутер и прежний блочный обязаны совпасть
        // побитово по ids и численно по весам (tie-break по меньшему индексу).
        let (ids_blk, w_blk) = gpu_softmax_topk_sel(
            device.as_cuda_device()?, &logits_t, n_experts, topk, true, None, Some(false))?;
        let (ids, weights) = gpu_softmax_topk_sel(
            device.as_cuda_device()?, &logits_t, n_experts, topk, true, None, Some(true))?;
        {
            let a = ids_blk.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
            let b = ids.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
            let wa = w_blk.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let wb = weights.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a, b, "warp vs block: ids расходятся");
            for (i, (x, y)) in wa.iter().zip(wb.iter()).enumerate() {
                assert!((x - y).abs() < 1e-6, "warp vs block: вес {i}: {x} vs {y}");
            }
        }
        let ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

        for token in 0..n_tokens {
            let row = &logits[token * n_experts..(token + 1) * n_experts];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
            let exp_sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|x| x / exp_sum).collect();
            let mut expected_ids: Vec<usize> = (0..n_experts).collect();
            expected_ids.sort_by(|a, b| {
                probs[*b]
                    .partial_cmp(&probs[*a])
                    .unwrap()
                    .then_with(|| a.cmp(b))
            });
            expected_ids.truncate(topk);
            let selected_sum: f32 = expected_ids.iter().map(|&e| probs[e]).sum();

            assert_eq!(
                ids[token],
                expected_ids.iter().map(|&e| e as u32).collect::<Vec<_>>()
            );
            for rank in 0..topk {
                let expected = probs[expected_ids[rank]] / selected_sum;
                assert!(
                    (weights[token][rank] - expected).abs() < 1e-6,
                    "token={token} rank={rank}: actual={} expected={expected}",
                    weights[token][rank]
                );
            }
            let sum: f32 = weights[token].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "token={token} sum={sum}");
        }
        Ok(())
    }
}

// ─── CPU-путь экспертов (MOE_CPU=1) ──────────────────────────────────────────
//
// Причина: декод 35B-A3B читает 540 МиБ весов экспертов на токен. Через
// PCIe 3.0 x16 (15.75 ГБ/с) это потолок ~29 т/с, а llama.cpp считает
// экспертов на CPU из RAM (полоса >100 ГБ/с) и берёт 45 т/с. Поэтому на
// выгруженном хранилище (`MOE_EXPERTS=ram`) считаем так же: сырые байты
// Q4_K уже лежат в host-памяти, GEMV идёт по ним напрямую с rayon по
// потокам, без единого H2D на экспертные матрицы.
#[cfg(feature = "cuda")]
pub fn moe_cpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MOE_CPU").as_deref() == Ok("1"))
}

#[cfg(feature = "cuda")]
fn moe_cpu_gpu_experts() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MOE_CPU_GPU_EXPERTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

#[cfg(feature = "cuda")]
fn moe_cpu_async_enabled() -> bool {
    std::env::var("MOE_CPU_ASYNC").as_deref() == Ok("1")
}

#[cfg(feature = "cuda")]
fn moe_cpu_mapped_enabled() -> bool {
    std::env::var("MOE_CPU_MAPPED").as_deref() == Ok("1")
}

#[cfg(feature = "cuda")]
fn moe_cpu_timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MOE_CPU_TIMING").is_some())
}

#[cfg(feature = "cuda")]
struct CpuDecodeStage {
    ids: CpuReadPinned<u32>,
    weights: CpuReadPinned<f32>,
    xs: CpuReadPinned<f32>,
    out: cudarc::driver::PinnedHostSlice<f32>,
}

/// Page-locked буфер без CU_MEMHOSTALLOC_WRITECOMBINED. cudarc alloc_pinned
/// всегда включает WC; для D2H -> CPU такое чтение заметно медленнее.
#[cfg(feature = "cuda")]
struct CpuReadPinned<T> {
    ptr: *mut T,
    len: usize,
    event: cudarc::driver::CudaEvent,
}

#[cfg(feature = "cuda")]
unsafe impl<T> Send for CpuReadPinned<T> {}
#[cfg(feature = "cuda")]
unsafe impl<T> Sync for CpuReadPinned<T> {}

#[cfg(feature = "cuda")]
impl<T> CpuReadPinned<T> {
    fn alloc(ctx: &Arc<cudarc::driver::CudaContext>, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| candle_core::Error::Msg("CPU pinned size overflow".into()))?;
        let raw = unsafe { cudarc::driver::result::malloc_host(bytes, 0) }
            .map_err(candle_core::Error::wrap)?;
        if raw.is_null() {
            candle_core::bail!("CPU pinned allocation returned null for {bytes} bytes");
        }
        let event = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(candle_core::Error::wrap)?;
        Ok(Self {
            ptr: raw as *mut T,
            len,
            event,
        })
    }

    fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(feature = "cuda")]
impl<T> Drop for CpuReadPinned<T> {
    fn drop(&mut self) {
        let _ = self.event.synchronize();
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr.cast());
        }
    }
}

#[cfg(feature = "cuda")]
struct MappedFlag {
    ptr: *mut u32,
    dev_ptr: cudarc::driver::sys::CUdeviceptr,
}

#[cfg(feature = "cuda")]
unsafe impl Send for MappedFlag {}
#[cfg(feature = "cuda")]
unsafe impl Sync for MappedFlag {}

#[cfg(feature = "cuda")]
impl MappedFlag {
    fn alloc(ctx: &Arc<cudarc::driver::CudaContext>) -> Result<Self> {
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(
                4,
                cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        }
        .map_err(candle_core::Error::wrap)? as *mut u32;
        if ptr.is_null() {
            candle_core::bail!("MoE async flag allocation returned null");
        }
        unsafe { *ptr = 0 };
        let mut dev_ptr = 0;
        let res = unsafe { cudarc::driver::sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, ptr.cast(), 0) };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = cudarc::driver::result::free_host(ptr.cast());
            }
            candle_core::bail!("MoE async flag mapping failed: {res:?}");
        }
        let _ = ctx;
        Ok(Self { ptr, dev_ptr })
    }

    fn signal(&self, value: u32) {
        unsafe {
            std::sync::atomic::AtomicU32::from_ptr(self.ptr)
                .store(value, std::sync::atomic::Ordering::Release);
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for MappedFlag {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::result::free_host(self.ptr.cast());
        }
    }
}

#[cfg(feature = "cuda")]
struct MappedHostBuffer<T> {
    ptr: *mut T,
    dev_ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
}

#[cfg(feature = "cuda")]
unsafe impl<T> Send for MappedHostBuffer<T> {}
#[cfg(feature = "cuda")]
unsafe impl<T> Sync for MappedHostBuffer<T> {}

#[cfg(feature = "cuda")]
impl<T> MappedHostBuffer<T> {
    fn alloc(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| candle_core::Error::Msg("mapped host size overflow".into()))?;
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(
                bytes,
                cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP,
            )
        }
        .map_err(candle_core::Error::wrap)? as *mut T;
        if ptr.is_null() {
            candle_core::bail!("mapped host allocation returned null for {bytes} bytes");
        }
        let mut dev_ptr = 0;
        let rc = unsafe {
            cudarc::driver::sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, ptr.cast(), 0)
        };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            unsafe { let _ = cudarc::driver::result::free_host(ptr.cast()); }
            candle_core::bail!("mapped host device pointer failed: {rc:?}");
        }
        Ok(Self { ptr, dev_ptr, len })
    }

    fn as_ptr(&self) -> *const T { self.ptr }
    fn as_mut_ptr(&self) -> *mut T { self.ptr }
    fn len(&self) -> usize { self.len }
    fn dev_ptr(&self) -> cudarc::driver::sys::CUdeviceptr { self.dev_ptr }
}

#[cfg(feature = "cuda")]
impl<T> Drop for MappedHostBuffer<T> {
    fn drop(&mut self) {
        unsafe { let _ = cudarc::driver::result::free_host(self.ptr.cast()); }
    }
}

#[cfg(feature = "cuda")]
struct AsyncCpuStage {
    ids: CpuReadPinned<u32>,
    weights: CpuReadPinned<f32>,
    xs: CpuReadPinned<f32>,
    out: CpuReadPinned<f32>,
    flag: MappedFlag,
    mapped_ids: MappedHostBuffer<u32>,
    mapped_weights: MappedHostBuffer<f32>,
    mapped_xs: MappedHostBuffer<f32>,
    mapped_out: MappedHostBuffer<f32>,
    export_flag: MappedFlag,
}

#[cfg(feature = "cuda")]
struct AsyncCpuJob {
    store: SharedLayerStore,
    copy_done: Option<cudarc::driver::CudaEvent>,
    export_flag: usize,
    ids: usize,
    weights: usize,
    xs: usize,
    out: usize,
    flag: usize,
    seq: u32,
    k_topk: usize,
    n_embd: usize,
    gpu_experts: usize,
}

/// Опциональный счётчик host-времени асинхронного MoE.
///
/// `YF_MOE_PROF=1` печатает сумму host-времени, потраченного в
/// `forward_ptx_cuda_async` (router, постановка graph, wait и merge) на
/// каждый полный токен из 41 слоя. Нужен, чтобы отличить стоимость
/// постановки работы от времени исполнения на GPU.
#[cfg(feature = "cuda")]
mod moe_prof {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub static NS: AtomicU64 = AtomicU64::new(0);
    pub static WCALLS: AtomicU64 = AtomicU64::new(0);
    pub static WNS: AtomicU64 = AtomicU64::new(0);

    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("YF_MOE_PROF").as_deref() == Ok("1"))
    }

    pub fn record_worker(elapsed_ns: u64) {
        WNS.fetch_add(elapsed_ns, Ordering::Relaxed);
        if WCALLS.fetch_add(1, Ordering::Relaxed) % 41 == 40 {
            let ns = WNS.swap(0, Ordering::Relaxed);
            eprintln!("[moe-prof] cpu_worker_per_token={:.3}ms", ns as f64 / 1e6);
        }
    }

    pub fn record(elapsed_ns: u64) {
        NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        if CALLS.fetch_add(1, Ordering::Relaxed) % 41 == 40 {
            let ns = NS.swap(0, Ordering::Relaxed);
            eprintln!("[moe-prof] host_per_token={:.3}ms", ns as f64 / 1e6);
        }
    }
}

#[cfg(feature = "cuda")]
fn async_cpu_error() -> &'static std::sync::Mutex<Option<String>> {
    static ERROR: std::sync::OnceLock<std::sync::Mutex<Option<String>>> =
        std::sync::OnceLock::new();
    ERROR.get_or_init(|| std::sync::Mutex::new(None))
}

/// Забрать отложенную ошибку CPU-MoE после drain устройства.
#[cfg(feature = "cuda")]
pub fn take_async_cpu_error() -> Option<String> {
    async_cpu_error()
        .lock()
        .expect("async CPU error")
        .take()
}

#[cfg(feature = "cuda")]
fn async_cpu_next_seq() -> u32 {
    static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

#[cfg(feature = "cuda")]
fn async_cpu_sender() -> &'static std::sync::mpsc::Sender<AsyncCpuJob> {
    static SENDER: std::sync::OnceLock<std::sync::mpsc::Sender<AsyncCpuJob>> =
        std::sync::OnceLock::new();
    SENDER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<AsyncCpuJob>();
        std::thread::Builder::new()
            .name("moe-cpu-async".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let base = std::time::Instant::now();
                    let prof_on = moe_prof::on();
                    let t_cpu_start = std::sync::atomic::AtomicU64::new(0);
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if let Some(copy_done) = job.copy_done.as_ref() {
                            copy_done.synchronize().map_err(candle_core::Error::wrap)?;
                        } else {
                            let flag = unsafe {
                                &*std::sync::atomic::AtomicU32::from_ptr(job.export_flag as *mut u32)
                            };
                            let deadline = std::time::Instant::now()
                                + std::time::Duration::from_secs(5);
                            while flag.load(std::sync::atomic::Ordering::Acquire) != job.seq {
                                if std::time::Instant::now() > deadline {
                                    candle_core::bail!("CPU-MoE: export flag timeout");
                                }
                                std::hint::spin_loop();
                            }
                        }
                        if prof_on {
                            t_cpu_start.store(base.elapsed().as_nanos() as u64,
                                              std::sync::atomic::Ordering::Relaxed);
                        }
                        let ids = unsafe {
                            std::slice::from_raw_parts(job.ids as *const u32, job.k_topk)
                        };
                        let weights = unsafe {
                            std::slice::from_raw_parts(job.weights as *const f32, job.k_topk)
                        };
                        let xs = unsafe {
                            std::slice::from_raw_parts(job.xs as *const f32, job.n_embd)
                        };
                        let store = job.store.lock().expect("expert layer store");
                        // Весь слой входит в существующий Rayon pool один раз:
                        // внутренние параллельные фазы переиспользуют workers.
                        rayon::scope(|_| {
                            moe_forward_cpu(
                                &store,
                                &ids[job.gpu_experts..],
                                &weights[job.gpu_experts..],
                                xs,
                                job.n_embd,
                            )
                        })
                    }));
                    if prof_on {
                        let s0 = t_cpu_start.load(std::sync::atomic::Ordering::Relaxed);
                        if s0 > 0 {
                            let now = base.elapsed().as_nanos() as u64;
                            moe_prof::record_worker(now.saturating_sub(s0));
                        }
                    }
                    match outcome {
                        Ok(Ok(result)) => unsafe {
                            std::ptr::copy_nonoverlapping(
                                result.as_ptr(),
                                job.out as *mut f32,
                                job.n_embd,
                            );
                        },
                        Ok(Err(error)) => {
                            *async_cpu_error().lock().expect("async CPU error") =
                                Some(error.to_string());
                            // Fail-closed: слой не должен прочитать прошлый out.
                            unsafe {
                                std::ptr::write_bytes(job.out as *mut u32, 0xff, job.n_embd);
                            }
                        }
                        Err(_) => {
                            *async_cpu_error().lock().expect("async CPU error") =
                                Some("CPU-MoE worker panicked".into());
                            unsafe {
                                std::ptr::write_bytes(job.out as *mut u32, 0xff, job.n_embd);
                            }
                        }
                    }
                    unsafe {
                        std::sync::atomic::AtomicU32::from_ptr(job.flag as *mut u32)
                            .store(job.seq, std::sync::atomic::Ordering::Release);
                    }
                }
            })
            .expect("spawn CPU-MoE async worker");
        tx
    })
}

#[cfg(feature = "cuda")]
fn async_cpu_stage(
    dev: &candle_core::CudaDevice,
    layer: usize,
    k: usize,
    n_embd: usize,
) -> Result<&'static AsyncCpuStage> {
    static STAGES: std::sync::OnceLock<
        std::sync::Mutex<Option<Vec<AsyncCpuStage>>>,
    > = std::sync::OnceLock::new();
    let mut guard = STAGES
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .expect("async CPU stages");
    if guard.is_none() {
        let ctx = dev.cuda_stream().context().clone();
        let mut stages = Vec::with_capacity(64);
        for _ in 0..64 {
            stages.push(AsyncCpuStage {
                ids: CpuReadPinned::alloc(&ctx, k)?,
                weights: CpuReadPinned::alloc(&ctx, k)?,
                xs: CpuReadPinned::alloc(&ctx, n_embd)?,
                out: CpuReadPinned::alloc(&ctx, n_embd)?,
                flag: MappedFlag::alloc(&ctx)?,
                mapped_ids: MappedHostBuffer::alloc(k)?,
                mapped_weights: MappedHostBuffer::alloc(k)?,
                mapped_xs: MappedHostBuffer::alloc(n_embd)?,
                mapped_out: MappedHostBuffer::alloc(n_embd)?,
                export_flag: MappedFlag::alloc(&ctx)?,
            });
        }
        *guard = Some(stages);
    }
    let stages = guard.as_ref().expect("async CPU stages initialized");
    let stage = stages
        .get(layer)
        .ok_or_else(|| candle_core::Error::Msg(format!("MoE async layer {layer} out of range")))?;
    if stage.ids.len() != k || stage.xs.len() != n_embd {
        candle_core::bail!("MoE async stage shape changed");
    }
    let ptr = stage as *const AsyncCpuStage;
    drop(guard);
    // STAGES never changes or drops during process lifetime.
    Ok(unsafe { &*ptr })
}

#[cfg(feature = "cuda")]
static CPU_DECODE_STAGE: std::sync::OnceLock<
    std::sync::Mutex<Option<CpuDecodeStage>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn cpu_decode_stage(
    dev: &candle_core::CudaDevice,
    k: usize,
    n_embd: usize,
) -> Result<std::sync::MutexGuard<'static, Option<CpuDecodeStage>>> {
    let slot = CPU_DECODE_STAGE.get_or_init(|| std::sync::Mutex::new(None));
    let mut stage = slot.lock().expect("CPU decode stage");
    let replace = stage
        .as_ref()
        .map(|s| s.ids.len() != k || s.xs.len() != n_embd)
        .unwrap_or(true);
    if replace {
        let ctx = dev.cuda_stream().context().clone();
        *stage = Some(CpuDecodeStage {
            ids: CpuReadPinned::alloc(&ctx, k)?,
            weights: CpuReadPinned::alloc(&ctx, k)?,
            xs: CpuReadPinned::alloc(&ctx, n_embd)?,
            out: unsafe { ctx.alloc_pinned::<f32>(n_embd) }
                .map_err(candle_core::Error::wrap)?,
        });
    }
    Ok(stage)
}

#[cfg(feature = "cuda")]
mod cpu_moe {
    use super::*;
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, BlockQ8K, QK_K};
    use candle_core::quantized::{GgmlType, GgmlDType};
    use rayon::prelude::*;

    /// GEMV по сырым байтам Q4_K / Q6_K. Строки идут параллельно по rayon.
    /// dtype выбирается по матрице: в GGUF down-проекция экспертов обычно
    /// Q6_K, а gate/up — Q4_K; читать их одним типом нельзя (мусор).
    pub fn gemv(
        w: &[u8],
        dtype: GgmlDType,
        x8: &[BlockQ8K],
        n_rows: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        let blocks_per_row = k / QK_K;
        match dtype {
            GgmlDType::Q4K => {
                let need = n_rows * blocks_per_row;
                let bs = std::mem::size_of::<BlockQ4K>();
                if w.len() < need * bs {
                    candle_core::bail!(
                        "cpu gemv q4k: {} байт мало для {n_rows}×{k} (нужно {})",
                        w.len(),
                        need * bs
                    );
                }
                let wb: &[BlockQ4K] =
                    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const BlockQ4K, need) };
                Ok((0..n_rows)
                    .into_par_iter()
                    .map(|r| BlockQ4K::vec_dot(k, &wb[r * blocks_per_row..(r + 1) * blocks_per_row], x8))
                    .collect())
            }
            GgmlDType::Q6K => {
                let need = n_rows * blocks_per_row;
                let bs = std::mem::size_of::<BlockQ6K>();
                if w.len() < need * bs {
                    candle_core::bail!(
                        "cpu gemv q6k: {} байт мало для {n_rows}×{k} (нужно {})",
                        w.len(),
                        need * bs
                    );
                }
                let wb: &[BlockQ6K] =
                    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const BlockQ6K, need) };
                Ok((0..n_rows)
                    .into_par_iter()
                    .map(|r| BlockQ6K::vec_dot(k, &wb[r * blocks_per_row..(r + 1) * blocks_per_row], x8))
                    .collect())
            }
            other => candle_core::bail!("cpu gemv: dtype {other:?} не поддержан"),
        }
    }

    /// Квантование активации f32 → Q8_K (та же схема, что ждёт vec_dot).
    pub fn quantize_activation(x: &[f32]) -> Result<Vec<BlockQ8K>> {
        if x.len() % QK_K != 0 {
            candle_core::bail!("quantize_activation: длина {} не кратна {QK_K}", x.len());
        }
        let mut out = vec![BlockQ8K::zeros(); x.len() / QK_K];
        BlockQ8K::from_float(x, &mut out);
        Ok(out)
    }
}

/// Полный CPU-проход MoE для одного слота, батчированный по экспертам.
///
/// Раньше на слой приходилось 24 rayon-вызова (8 экспертов × 3 матрицы), то
/// есть 960 fork/join на токен — накладные расходы съедали всё. Здесь два
/// rayon-прохода на слой: (1) gate+up для всех экспертов сразу, (2) down для
/// всех экспертов сразу. Замер масштабирования: 1 поток 1.81 т/с, 18 потоков
/// 7.87 т/с, 72 (HT) 3.37 т/с — HT только вредит.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_forward_cpu(
    st: &super::expert_store::ExpertLayerStore,
    ids: &[u32],
    weights: &[f32],
    xs_cpu: &[f32],
    n_embd: usize,
) -> Result<Vec<f32>> {
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, BlockQ8K, QK_K};
    use candle_core::quantized::{GgmlDType, GgmlType};
    use rayon::prelude::*;

    let (g_shape, _d_shape) = st.shapes();
    let n_ff = g_shape.1;
    let n_experts = st.gate.n_experts;

    let x8 = cpu_moe::quantize_activation(xs_cpu)?;

    let sel: Vec<(usize, f32)> = ids
        .iter()
        .zip(weights.iter())
        .map(|(&e, &w)| (e as usize, w))
        .filter(|(e, _)| *e < n_experts)
        .collect();
    if sel.is_empty() {
        return Ok(vec![0f32; n_embd]);
    }

    // Проход 1: gate+up для всех выбранных экспертов одним parallel_iter.
    let gate_bytes = st.gate.expert_bytes;
    let up_bytes = st.up.expert_bytes;
    let down_bytes = st.down.expert_bytes;
    let g_dtype = st.gate.dtype;
    let u_dtype = st.up.dtype;

    let gate_slices: Vec<&[u8]> = sel.iter().map(|(e, _)| st.gate.expert_host_slice(*e)).collect();
    let up_slices: Vec<&[u8]> = sel.iter().map(|(e, _)| st.up.expert_host_slice(*e)).collect();

    let _ = (gate_bytes, up_bytes);

    // Один плоский parallel по (эксперт × строка): без вложенного rayon.
    let n_jobs = sel.len() * n_ff;
    let mut acts = vec![0f32; n_jobs];
    let bprow_g = n_embd / QK_K;
    let bprow_u = n_embd / QK_K;
    if g_dtype != GgmlDType::Q4K || u_dtype != GgmlDType::Q4K {
        candle_core::bail!("cpu moe: ожидался Q4K для gate/up, got {g_dtype:?}/{u_dtype:?}");
    }
    let g_blocks: Vec<&[BlockQ4K]> = gate_slices
        .iter()
        .map(|sl| unsafe {
            std::slice::from_raw_parts(sl.as_ptr() as *const BlockQ4K, n_ff * bprow_g)
        })
        .collect();
    let u_blocks: Vec<&[BlockQ4K]> = up_slices
        .iter()
        .map(|sl| unsafe {
            std::slice::from_raw_parts(sl.as_ptr() as *const BlockQ4K, n_ff * bprow_u)
        })
        .collect();
    // Разбиение по СТРОКАМ, а не по экспертам: 8 экспертов давали максимум
    // 8 параллельных задач, то есть 5–6 занятых ядер из 18. Строк 8×512=4096
    // — rayon загружает все потоки.
    let rows_per_task = 32usize;
    // По 8 строк за раз: активация Q8_K загружается и распаковывается один
    // раз на 8 строк вместо восьми (то, что llama.cpp получает через repack
    // в block_q4_Kx8). 8·bprow блоков подряд — это 8 последовательных строк.
    let rows8 = 8usize;
    acts
        .par_chunks_mut(rows8)
        .enumerate()
        .for_each(|(t, a8)| {
            let base = t * rows8;
            let j = base / n_ff;
            let r = base % n_ff;
            let n = a8.len();
            let mut acc_g = [0f32; 8];
            let mut acc_u = [0f32; 8];
            let gs = &g_blocks[j][r * bprow_g..(r + n) * bprow_g];
            let us = &u_blocks[j][r * bprow_u..(r + n) * bprow_u];
            #[cfg(target_feature = "avx2")]
            if n == 8 {
                candle_core::quantized::avx::vec_dot_q4k_q8k_x8(n_embd, gs, &x8, &mut acc_g);
                candle_core::quantized::avx::vec_dot_q4k_q8k_x8(n_embd, us, &x8, &mut acc_u);
            } else {
                for off in 0..n {
                    acc_g[off] = BlockQ4K::vec_dot(
                        n_embd, &gs[off * bprow_g..(off + 1) * bprow_g], &x8);
                    acc_u[off] = BlockQ4K::vec_dot(
                        n_embd, &us[off * bprow_u..(off + 1) * bprow_u], &x8);
                }
            }
            #[cfg(not(target_feature = "avx2"))]
            {
                for off in 0..n {
                    acc_g[off] = BlockQ4K::vec_dot(
                        n_embd, &gs[off * bprow_g..(off + 1) * bprow_g], &x8);
                    acc_u[off] = BlockQ4K::vec_dot(
                        n_embd, &us[off * bprow_u..(off + 1) * bprow_u], &x8);
                }
            }
            for off in 0..n {
                let gv = acc_g[off];
                a8[off] = (gv / (1.0 + (-gv).exp())) * acc_u[off];
            }
        });

    // Проход 2: down для всех выбранных экспертов + взвешенная сумма.
    let down_slices: Vec<&[u8]> = sel.iter().map(|(e, _)| st.down.expert_host_slice(*e)).collect();
    let d_dtype = st.down.dtype;
    let _ = down_bytes;

    let mut down_all = vec![0f32; sel.len() * n_embd];
    if d_dtype != GgmlDType::Q4K && d_dtype != GgmlDType::Q6K {
        candle_core::bail!("cpu moe: down dtype {d_dtype:?} не поддержан");
    }
    let bprow_d = n_ff / QK_K;
    // down-матрицы: Q6_K или Q4_K — приводим к enum внутри задачи.
    let d8s: Vec<Vec<BlockQ8K>> = acts
        .chunks(n_ff)
        .map(|a| {
            let mut a8 = vec![BlockQ8K::zeros(); bprow_d];
            BlockQ8K::from_float(a, &mut a8);
            a8
        })
        .collect();
    let d_blocks_q4: Vec<Option<&[BlockQ4K]>> = down_slices
        .iter()
        .map(|sl| {
            (d_dtype == GgmlDType::Q4K).then(|| unsafe {
                std::slice::from_raw_parts(sl.as_ptr() as *const BlockQ4K, n_embd * bprow_d)
            })
        })
        .collect();
    let d_blocks_q6: Vec<Option<&[BlockQ6K]>> = down_slices
        .iter()
        .map(|sl| {
            (d_dtype == GgmlDType::Q6K).then(|| unsafe {
                std::slice::from_raw_parts(sl.as_ptr() as *const BlockQ6K, n_embd * bprow_d)
            })
        })
        .collect();
    down_all
        .par_chunks_mut(rows_per_task)
        .enumerate()
        .for_each(|(t, chunk)| {
            let base = t * rows_per_task;
            for off in 0..chunk.len() {
                let idx = base + off;
                let j = idx / n_embd;
                let r = idx % n_embd;
                let a8 = &d8s[j];
                chunk[off] = if let Some(dq4) = d_blocks_q4[j] {
                    let sl = &dq4[r * bprow_d..(r + 1) * bprow_d];
                    BlockQ4K::vec_dot(n_ff, sl, a8)
                } else if let Some(dq6) = d_blocks_q6[j] {
                    let sl = &dq6[r * bprow_d..(r + 1) * bprow_d];
                    BlockQ6K::vec_dot(n_ff, sl, a8)
                } else {
                    0.0
                };
            }
        });
    let mut acc = vec![0f32; n_embd];
    for ((_, w), d) in sel.iter().zip(down_all.chunks(n_embd)) {
        for (a, v) in acc.iter_mut().zip(d) {
            *a += *w * *v;
        }
    }
    Ok(acc)
}

/// GEMV по сырым байтам Q4_K / Q6_K. Строки идут параллельно.
#[cfg(feature = "cuda")]
pub fn cpu_gemv(
    w: &[u8],
    dtype: candle_core::quantized::GgmlDType,
    x8: &[candle_core::quantized::k_quants::BlockQ8K],
    n_rows: usize,
    k: usize,
) -> Result<Vec<f32>> {
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ6K, QK_K};
    use candle_core::quantized::{GgmlDType, GgmlType};

    let blocks_per_row = k / QK_K;
    match dtype {
        GgmlDType::Q4K => {
            let need = n_rows * blocks_per_row;
            let bs = std::mem::size_of::<BlockQ4K>();
            if w.len() < need * bs {
                candle_core::bail!("cpu_gemv q4k: {} байт мало для {n_rows}×{k}", w.len());
            }
            let wb: &[BlockQ4K] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const BlockQ4K, need) };
            Ok((0..n_rows)
                .map(|r| BlockQ4K::vec_dot(k, &wb[r * blocks_per_row..(r + 1) * blocks_per_row], x8))
                .collect())
        }
        GgmlDType::Q6K => {
            let need = n_rows * blocks_per_row;
            let bs = std::mem::size_of::<BlockQ6K>();
            if w.len() < need * bs {
                candle_core::bail!("cpu_gemv q6k: {} байт мало для {n_rows}×{k}", w.len());
            }
            let wb: &[BlockQ6K] =
                unsafe { std::slice::from_raw_parts(w.as_ptr() as *const BlockQ6K, need) };
            Ok((0..n_rows)
                .map(|r| BlockQ6K::vec_dot(k, &wb[r * blocks_per_row..(r + 1) * blocks_per_row], x8))
                .collect())
        }
        other => candle_core::bail!("cpu_gemv: dtype {other:?} не поддержан"),
    }
}
