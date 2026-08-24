// gated_delta_net_fused.rs — Rust bindings для batch-aware fused Metal kernels
//
// Phase 1 порта fused Gated DeltaNet kernel из llama.cpp (T-265).
//
// Заменяет цикл `for t in 0..seq_len { dispatch_delta_rule(...) }` на
// одну функцию `dispatch_gdn_fused()`, делающую 4 dispatch'а на слой
// (вместо нынешних 4×T dispatch'ей в `dispatch_delta_rule`).
//
// Это радикально снижает CPU dispatch overhead и устраняет CPU prefill
// через Accelerate в `forward_prefill`.
//
// ВАЖНО: переиспользует `DeltaNetMetalState` из `delta_rule_metal.rs` —
// не дублирует определение persistent буферов слоя. Старая реализация
// `dispatch_delta_rule` остаётся параллельно для accuracy-сравнения
// (Phase 2-5 спецификации T-265).

use candle_core::{DType, MetalDevice, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MtlDevice};
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::sync::Arc;

use super::delta_rule_metal::DeltaNetMetalState;

/// Metal shader source — включается при компиляции
const GDN_FUSED_METAL_SOURCE: &str = include_str!("gated_delta_net_fused.metal");

/// Параметры для fused kernels.
/// Layout ДОЛЖЕН точно совпадать со struct GdnFusedParams в .metal файле.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GdnFusedParams {
    pub n_tokens: u32,
    pub channels: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub q_scale: f32,
    pub rms_norm_eps: f32,
    pub n_v_heads: u32,
    pub conv_kernel: u32,
    /// T-282 fix: GQA expansion в DeltaNet l2_norm_batch kernel.
    /// `head_k = head_v % n_k_heads` — иначе для 4B (n_v=32, n_k=16)
    /// heads 16..31 читали K-data вместо Q.
    pub n_k_heads: u32,
}

/// Скомпилированные Metal pipelines для fused gated delta net.
/// Создаётся один раз при загрузке модели, переиспользуется для всех слоёв.
pub struct GdnFusedPipelines {
    pub conv1d_batch_prep: ComputePipeline,
    pub l2_norm_batch: ComputePipeline,
    pub recurrent_batch: ComputePipeline,
    pub norm_gate_batch: ComputePipeline,
}

/// Размерность simdgroup в Metal (фиксированная константа Apple GPU)
const SIMD_WIDTH: usize = 32;

/// Кол-во строк state на thread в recurrent kernel (NSG из .metal)
const NSG: usize = 4;

/// Размер value_dim per head — захардкожен для Qwen3.5
const HEAD_V_DIM: usize = 128;

/// Компилирует Metal шейдер и создаёт все 4 pipeline.
pub fn compile_gdn_fused_pipelines(metal_device: &MetalDevice) -> Result<GdnFusedPipelines> {
    let device: &MtlDevice = metal_device.metal_device();

    let library = device
        .new_library_with_source(GDN_FUSED_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;

    let fn_conv1d = library
        .get_function("gdn_conv1d_batch_prep", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_l2_norm = library
        .get_function("gdn_l2_norm_batch", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_recurrent = library
        .get_function("gdn_recurrent_batch", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_norm_gate = library
        .get_function("gdn_norm_gate_batch", None)
        .map_err(candle_core::Error::wrap)?;

    let conv1d_batch_prep = device
        .new_compute_pipeline_state_with_function(&fn_conv1d)
        .map_err(candle_core::Error::wrap)?;
    let l2_norm_batch = device
        .new_compute_pipeline_state_with_function(&fn_l2_norm)
        .map_err(candle_core::Error::wrap)?;
    let recurrent_batch = device
        .new_compute_pipeline_state_with_function(&fn_recurrent)
        .map_err(candle_core::Error::wrap)?;
    let norm_gate_batch = device
        .new_compute_pipeline_state_with_function(&fn_norm_gate)
        .map_err(candle_core::Error::wrap)?;

    Ok(GdnFusedPipelines {
        conv1d_batch_prep,
        l2_norm_batch,
        recurrent_batch,
        norm_gate_batch,
    })
}

/// Выполняет полный fused DeltaNet forward pass на Metal GPU за 4 dispatch'а
/// для всей последовательности из T токенов.
///
/// Заменяет цикл `for t in 0..seq_len { dispatch_delta_rule(...) }` —
/// вместо 4×T dispatch'ей делается 4 (по одному на kernel, token loop внутри
/// kernels 1 и 3).
///
/// Входы:
///   qkv_t   — [1, T, channels]    результат joint QKV проекции
///   z_t     — [1, T, value_dim]   результат z проекции (для gate)
///   beta_t  — [1, T, n_v_heads]   raw beta logits
///   alpha_t — [1, T, n_v_heads]   raw alpha logits
///
/// Выход: [1, T, value_dim] на GPU, готовый для финальной ssm_out проекции.
///
/// ВСЁ выполняется на GPU — ноль GPU↔CPU синхронизаций.
pub fn dispatch_gdn_fused(
    metal_device: &MetalDevice,
    pipelines: &GdnFusedPipelines,
    layer_state: &DeltaNetMetalState,
    qkv_t: &Tensor,   // [1, T, channels]
    z_t: &Tensor,     // [1, T, value_dim]
    beta_t: &Tensor,  // [1, T, n_v_heads]
    alpha_t: &Tensor, // [1, T, n_v_heads]
    params: &GdnFusedParams,
) -> Result<Tensor> {
    let n_tokens = params.n_tokens as usize;
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let value_dim = params.value_dim as usize;
    let hkd = HEAD_V_DIM; // = HEAD_K_DIM в Qwen3.5
    let hvd = HEAD_V_DIM;

    if n_tokens == 0 {
        candle_core::bail!("dispatch_gdn_fused: n_tokens == 0");
    }

    // F32 baseline (как llama.cpp ggml_gated_delta_net — все 6 входов F32).
    // Long-prompt тест показал: F16 в GDN batch buffers даёт cumulative drift
    // в recurrent state за seq=5K+ токенов — output деградирует в noise токены
    // несмотря на F32 ssm_state. Промежуточные F16 buffers (q/k/v/beta/gate)
    // вносят quantization error на каждом токене recurrent update.
    // Memory savings достигаются другими путями (compute_per_buffer=15, sync per layer).
    let qkv_t_f32 = ensure_f32(qkv_t)?;
    let z_t_f32 = ensure_f32(z_t)?;
    let beta_t_f32 = ensure_f32(beta_t)?;
    let alpha_t_f32 = ensure_f32(alpha_t)?;

    // Извлекаем raw Metal буферы
    let qkv_buf = get_metal_buffer(&qkv_t_f32)?;
    let z_buf = get_metal_buffer(&z_t_f32)?;
    let beta_buf = get_metal_buffer(&beta_t_f32)?;
    let alpha_buf = get_metal_buffer(&alpha_t_f32)?;

    // ═══════════════════════════════════════════════════════════════
    // Аллокация temp буферов под весь batch
    //
    // ВАЖНО: нельзя переиспользовать `DeltaNetTempBuffers` (одна копия на слой,
    // [channels] / [n_v_heads*hd] размеры), потому что нам нужны буферы
    // на ВЕСЬ batch [T, ...]. Аллоцируем здесь — для production можно вынести
    // в отдельный TempBuffers struct, переиспользуемый между вызовами.
    // ═══════════════════════════════════════════════════════════════
    // F32 batch buffers (как llama.cpp). На длинных контекстах (>1K токенов) F16
    // здесь критичен — kernel inputs читаются recurrent loop'ом, error накапливается.
    let qkv_conv_buf = metal_device.new_buffer(n_tokens * channels, DType::F32, "gdn_qkv_conv")?;
    let q_buf = metal_device.new_buffer(n_tokens * value_dim, DType::F32, "gdn_q")?;
    let k_buf = metal_device.new_buffer(n_tokens * value_dim, DType::F32, "gdn_k")?;
    let v_buf = metal_device.new_buffer(n_tokens * value_dim, DType::F32, "gdn_v")?;
    let beta_out_buf = metal_device.new_buffer(n_tokens * n_v, DType::F32, "gdn_beta")?;
    let gate_out_buf = metal_device.new_buffer(n_tokens * n_v, DType::F32, "gdn_gate")?;
    let raw_output_buf =
        metal_device.new_buffer(n_tokens * value_dim, DType::F32, "gdn_raw_out")?;
    let gated_output_buf =
        metal_device.new_buffer(n_tokens * value_dim, DType::F32, "gdn_gated_out")?;

    // ═══════════════════════════════════════════════════════════════
    // Dispatch 1: gdn_conv1d_batch_prep
    // Grid: (channels) threads, threadgroup: 256
    // Token loop ВНУТРИ kernel'а — sequential обновление conv_state.
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.conv1d_batch_prep);

        encoder.set_buffer(0, Some(&qkv_buf), 0);
        encoder.set_buffer(1, Some(&beta_buf), 0);
        encoder.set_buffer(2, Some(&alpha_buf), 0);
        encoder.set_buffer(3, Some(&layer_state.conv_weights), 0);
        encoder.set_buffer(4, Some(&layer_state.dt_bias), 0);
        encoder.set_buffer(5, Some(&layer_state.ssm_a), 0);
        encoder.set_buffer(6, Some(&layer_state.conv_state), 0);
        encoder.set_buffer(7, Some(&qkv_conv_buf), 0);
        encoder.set_buffer(8, Some(&beta_out_buf), 0);
        encoder.set_buffer(9, Some(&gate_out_buf), 0);
        encoder.set_bytes(10, params);

        encoder.use_resource(&*qkv_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*beta_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*alpha_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*layer_state.conv_weights, MTLResourceUsage::Read);
        encoder.use_resource(&*layer_state.dt_bias, MTLResourceUsage::Read);
        encoder.use_resource(&*layer_state.ssm_a, MTLResourceUsage::Read);
        encoder.use_resource(
            &*layer_state.conv_state,
            MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0),
        );
        encoder.use_resource(&*qkv_conv_buf, MTLResourceUsage::Write);
        encoder.use_resource(&*beta_out_buf, MTLResourceUsage::Write);
        encoder.use_resource(&*gate_out_buf, MTLResourceUsage::Write);

        let grid = MTLSize {
            width: channels,
            height: 1,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: 256.min(channels),
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(grid, tg_size);
    }

    // ═══════════════════════════════════════════════════════════════
    // Dispatch 2: gdn_l2_norm_batch
    // Grid: (n_v_heads, T) threadgroups, threadgroup: (1, head_k_dim=128)
    // Каждый (head, token) — независимый threadgroup.
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.l2_norm_batch);

        encoder.set_buffer(0, Some(&qkv_conv_buf), 0);
        encoder.set_buffer(1, Some(&q_buf), 0);
        encoder.set_buffer(2, Some(&k_buf), 0);
        encoder.set_buffer(3, Some(&v_buf), 0);
        encoder.set_bytes(4, params);

        encoder.use_resource(&*qkv_conv_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*q_buf, MTLResourceUsage::Write);
        encoder.use_resource(&*k_buf, MTLResourceUsage::Write);
        encoder.use_resource(&*v_buf, MTLResourceUsage::Write);

        let tg_count = MTLSize {
            width: n_v,
            height: n_tokens,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: 1,
            height: hkd,
            depth: 1,
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }

    // ═══════════════════════════════════════════════════════════════
    // Dispatch 3: gdn_recurrent_batch
    // Grid: (HEAD_V_DIM/NSG=32, n_v_heads) threadgroups
    // Threadgroup: (32, NSG=4) = 128 threads = 4 simdgroups
    // Token loop ВНУТРИ — state в регистрах через simd_sum.
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.recurrent_batch);

        encoder.set_buffer(0, Some(&q_buf), 0);
        encoder.set_buffer(1, Some(&k_buf), 0);
        encoder.set_buffer(2, Some(&v_buf), 0);
        encoder.set_buffer(3, Some(&beta_out_buf), 0);
        encoder.set_buffer(4, Some(&gate_out_buf), 0);
        encoder.set_buffer(5, Some(&layer_state.ssm_state), 0);
        encoder.set_buffer(6, Some(&raw_output_buf), 0);
        encoder.set_bytes(7, params);

        encoder.use_resource(&*q_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*k_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*v_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*beta_out_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*gate_out_buf, MTLResourceUsage::Read);
        encoder.use_resource(
            &*layer_state.ssm_state,
            MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0),
        );
        encoder.use_resource(&*raw_output_buf, MTLResourceUsage::Write);

        // col_block = HEAD_V_DIM / NSG = 32, head = n_v_heads
        let tg_count = MTLSize {
            width: hvd / NSG,
            height: n_v,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: SIMD_WIDTH, // 32
            height: NSG,       // 4
            depth: 1,
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }

    // ═══════════════════════════════════════════════════════════════
    // Dispatch 4: gdn_norm_gate_batch
    // Grid: (n_v_heads, T) threadgroups, threadgroup: (1, head_v_dim=128)
    // RMS norm per head + SiLU(z) gating, токены параллельно.
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.norm_gate_batch);

        encoder.set_buffer(0, Some(&raw_output_buf), 0);
        encoder.set_buffer(1, Some(&z_buf), 0);
        encoder.set_buffer(2, Some(&layer_state.norm_weight), 0);
        encoder.set_buffer(3, Some(&gated_output_buf), 0);
        encoder.set_bytes(4, params);

        encoder.use_resource(&*raw_output_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*z_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*layer_state.norm_weight, MTLResourceUsage::Read);
        encoder.use_resource(&*gated_output_buf, MTLResourceUsage::Write);

        let tg_count = MTLSize {
            width: n_v,
            height: n_tokens,
            depth: 1,
        };
        let tg_size = MTLSize {
            width: 1,
            height: hvd,
            depth: 1,
        };
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }

    // ═══════════════════════════════════════════════════════════════
    // Обернуть gated_output_buf в Candle Tensor (zero-copy)
    // Размер: [1, T, value_dim]
    // ═══════════════════════════════════════════════════════════════
    let total_elems = n_tokens * value_dim;
    let output_storage = MetalStorage::new(
        gated_output_buf.clone(),
        metal_device.clone(),
        total_elems,
        DType::F32,
    );
    let output_tensor = Tensor::from_storage(
        Storage::Metal(output_storage),
        (1, n_tokens, value_dim),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(output_tensor)
}

/// Cast тензор в F32 contiguous если он не F32.
fn ensure_f32(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() == DType::F32 {
        if tensor.is_contiguous() {
            Ok(tensor.clone())
        } else {
            tensor.contiguous()
        }
    } else {
        tensor.to_dtype(DType::F32)?.contiguous()
    }
}

// ═══════════════════════════════════════════════════════════════
// Вспомогательные функции
// ═══════════════════════════════════════════════════════════════

/// Извлекает raw Metal Buffer из Candle Tensor.
/// Tensor должен быть F32 (как llama.cpp ggml_gated_delta_net API).
fn get_metal_buffer(tensor: &Tensor) -> Result<Arc<Buffer>> {
    debug_assert_eq!(tensor.dtype(), DType::F32);
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => Ok(Arc::new(ms.buffer().clone())),
        _ => candle_core::bail!("gated_delta_net_fused: tensor не на Metal device"),
    }
}
