// delta_rule_metal.rs — Rust bindings для Metal delta_rule compute shaders
//
// Компилирует 4 Metal ядра из delta_rule.metal и предоставляет API для dispatch.
// Используется в DeltaNetLayer::forward() для выполнения рекуррентного шага
// целиком на GPU, устраняя GPU↔CPU синхронизацию.
//
// Архитектура:
// 1. delta_conv1d_prep — Conv1d + sigmoid(beta) + softplus(alpha)*A
// 2. delta_l2_norm_expand — L2 norm + head expansion 16→32 + Q scaling
// 3. delta_rule_kernel — Decay + S^T@k + delta + rank-1 update + S^T@q
// 4. delta_norm_gate_kernel — Group RMS norm + SiLU(z) gating

use candle_core::{DType, MetalDevice, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MtlDevice};
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::sync::Arc;

/// Metal shader source — включается при компиляции
const DELTA_RULE_METAL_SOURCE: &str = include_str!("delta_rule.metal");

/// Параметры DeltaNet слоя, передаваемые в Metal kernel через constant buffer.
/// Layout ДОЛЖЕН точно совпадать со struct DeltaParams в delta_rule.metal.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeltaParams {
    pub n_k_heads: u32,
    pub n_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub channels: u32,
    pub conv_kernel: u32,
    pub q_scale: f32,
    pub rms_norm_eps: f32,
    pub heads_per_kv: u32,
}

/// Скомпилированные Metal pipelines для delta_rule.
/// Создаётся один раз при загрузке модели, переиспользуется для всех слоёв.
pub struct DeltaRulePipelines {
    pub conv1d_prep: ComputePipeline,
    pub l2_norm_expand: ComputePipeline,
    pub delta_rule: ComputePipeline,
    pub norm_gate: ComputePipeline,
}

/// Persistent Metal буферы для одного DeltaNet слоя.
/// Живут всё время жизни модели, сбрасываются при clear().
pub struct DeltaNetMetalState {
    /// SSM state: [n_v_heads * head_v_dim * head_v_dim] = [32 * 128 * 128] = 524288 floats
    pub ssm_state: Arc<Buffer>,
    /// Conv1d state: [(conv_k - 1) * channels] = [3 * 8192] = 24576 floats
    pub conv_state: Arc<Buffer>,
    /// Conv1d kernel weights: [channels * conv_k] — persistent per layer
    pub conv_weights: Arc<Buffer>,
    /// dt_bias: [n_v_heads] — persistent per layer
    pub dt_bias: Arc<Buffer>,
    /// ssm_a: [n_v_heads] — persistent per layer (negative values for decay)
    pub ssm_a: Arc<Buffer>,
    /// RMS norm weight: [head_v_dim] — persistent per layer
    pub norm_weight: Arc<Buffer>,
}

/// Временные (temp) Metal буферы, переиспользуемые между слоями.
/// Аллоцируются один раз, используются как scratch space.
pub struct DeltaNetTempBuffers {
    /// qkv_conv: [channels] = [8192] floats
    pub qkv_conv: Arc<Buffer>,
    /// q: [n_v_heads * head_k_dim] = [4096] floats
    pub q: Arc<Buffer>,
    /// k: [n_v_heads * head_k_dim] = [4096] floats
    pub k: Arc<Buffer>,
    /// v: [n_v_heads * head_v_dim] = [4096] floats
    pub v: Arc<Buffer>,
    /// beta: [n_v_heads] = [32] floats
    pub beta: Arc<Buffer>,
    /// gate: [n_v_heads] = [32] floats
    pub gate: Arc<Buffer>,
    /// delta_output: [n_v_heads * head_v_dim] = [4096] floats — raw output
    pub delta_output: Arc<Buffer>,
    /// gated_output: [value_dim] = [4096] floats — final gated output
    pub gated_output: Arc<Buffer>,
}

/// Компилирует Metal шейдер и создаёт все 4 pipeline.
pub fn compile_delta_rule_pipelines(metal_device: &MetalDevice) -> Result<DeltaRulePipelines> {
    let device: &MtlDevice = metal_device.metal_device();

    // Компиляция Metal source code
    let library = device
        .new_library_with_source(DELTA_RULE_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;

    // Получение функций по именам
    let fn_conv1d_prep = library
        .get_function("delta_conv1d_prep", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_l2_norm = library
        .get_function("delta_l2_norm_expand", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_delta_rule = library
        .get_function("delta_rule_kernel", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_norm_gate = library
        .get_function("delta_norm_gate_kernel", None)
        .map_err(candle_core::Error::wrap)?;

    // Создание compute pipeline state для каждой функции
    let conv1d_prep = device
        .new_compute_pipeline_state_with_function(&fn_conv1d_prep)
        .map_err(candle_core::Error::wrap)?;
    let l2_norm_expand = device
        .new_compute_pipeline_state_with_function(&fn_l2_norm)
        .map_err(candle_core::Error::wrap)?;
    let delta_rule = device
        .new_compute_pipeline_state_with_function(&fn_delta_rule)
        .map_err(candle_core::Error::wrap)?;
    let norm_gate = device
        .new_compute_pipeline_state_with_function(&fn_norm_gate)
        .map_err(candle_core::Error::wrap)?;

    Ok(DeltaRulePipelines {
        conv1d_prep,
        l2_norm_expand,
        delta_rule,
        norm_gate,
    })
}

/// Создаёт persistent Metal буферы для одного DeltaNet слоя.
///
/// `conv_weights_data`, `dt_bias_data`, `ssm_a_data`, `norm_weight_data` —
/// данные весов, загруженные из GGUF.
pub fn create_layer_metal_state(
    metal_device: &MetalDevice,
    params: &DeltaParams,
    conv_weights_data: &[f32],
    dt_bias_data: &[f32],
    ssm_a_data: &[f32],
    norm_weight_data: &[f32],
) -> Result<DeltaNetMetalState> {
    let hd = params.head_v_dim as usize;
    let n_v = params.n_v_heads as usize;
    let channels = params.channels as usize;
    let conv_k = params.conv_kernel as usize;

    // SSM state — zero-initialized
    let ssm_state_size = n_v * hd * hd * std::mem::size_of::<f32>();
    let ssm_state = metal_device.allocate_zeros(ssm_state_size)?;

    // Conv state — zero-initialized
    let conv_state_size = (conv_k - 1) * channels * std::mem::size_of::<f32>();
    let conv_state = metal_device.allocate_zeros(conv_state_size)?;

    // Веса — загружаем данные
    let conv_weights = metal_device.new_buffer_with_data(conv_weights_data)?;
    let dt_bias = metal_device.new_buffer_with_data(dt_bias_data)?;
    let ssm_a = metal_device.new_buffer_with_data(ssm_a_data)?;
    let norm_weight = metal_device.new_buffer_with_data(norm_weight_data)?;

    Ok(DeltaNetMetalState {
        ssm_state,
        conv_state,
        conv_weights,
        dt_bias,
        ssm_a,
        norm_weight,
    })
}

/// Создаёт временные буферы, переиспользуемые между слоями.
pub fn create_temp_buffers(
    metal_device: &MetalDevice,
    params: &DeltaParams,
) -> Result<DeltaNetTempBuffers> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let hkd = params.head_k_dim as usize;
    let hvd = params.head_v_dim as usize;

    // F32 буферы для per-token decode path. Decode requires precision: long-prompt
    // tests showed что F16 в decode kernels накапливает ошибку (модель сразу EOS на
    // first decode token после большого prefill). Decode buffers маленькие (per-token
    // ~16-512 элементов) — F32 не critical для memory.
    // F16 остаётся только в GDN fused (prefill, batch buffers ~480 МБ × layers).
    Ok(DeltaNetTempBuffers {
        qkv_conv: metal_device.new_buffer(channels, DType::F32, "qkv_conv")?,
        q: metal_device.new_buffer(n_v * hkd, DType::F32, "q_buf")?,
        k: metal_device.new_buffer(n_v * hkd, DType::F32, "k_buf")?,
        v: metal_device.new_buffer(n_v * hvd, DType::F32, "v_buf")?,
        beta: metal_device.new_buffer(n_v, DType::F32, "beta_buf")?,
        gate: metal_device.new_buffer(n_v, DType::F32, "gate_buf")?,
        delta_output: metal_device.new_buffer(n_v * hvd, DType::F32, "delta_out")?,
        gated_output: metal_device.new_buffer(n_v * hvd, DType::F32, "gated_out")?,
    })
}

/// Выполняет полный delta_rule forward pass на Metal GPU.
///
/// Входы: 4 тензора-результата QMatMul проекций (уже на GPU).
/// Выход: тензор [1, 1, value_dim] на GPU, готовый для ssm_out QMatMul.
///
/// ВСЁ выполняется на GPU — ноль GPU↔CPU синхронизаций.
pub fn dispatch_delta_rule(
    metal_device: &MetalDevice,
    pipelines: &DeltaRulePipelines,
    layer_state: &DeltaNetMetalState,
    temp: &DeltaNetTempBuffers,
    params: &DeltaParams,
    // Входные тензоры (результаты QMatMul, на GPU)
    qkv_t: &Tensor,   // [1, 1, channels]
    z_t: &Tensor,     // [1, 1, value_dim]
    beta_t: &Tensor,  // [1, 1, n_v_heads]
    alpha_t: &Tensor, // [1, 1, n_v_heads]
) -> Result<Tensor> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let hkd = params.head_k_dim as usize;
    let hvd = params.head_v_dim as usize;
    let value_dim = params.value_dim as usize;

    // ПРИМЕЧАНИЕ: sync НЕ нужен в production — QMatMul и наши kernels
    // идут через один Candle Metal command pipeline (ordering гарантирован).
    // В тестах sync вызывается ПЕРЕД dispatch_delta_rule отдельно.

    // Decode path: F32 для precision (см. ensure_f32 — model выдаёт F32 от QMatMul).
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
    // Kernel 1: delta_conv1d_prep
    // Grid: (channels, 1, 1), Threadgroup: (256, 1, 1)
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.conv1d_prep);

        // Аргументы (порядок совпадает с .metal файлом)
        encoder.set_buffer(0, Some(&qkv_buf), 0); // qkv_raw
        encoder.set_buffer(1, Some(&beta_buf), 0); // beta_raw
        encoder.set_buffer(2, Some(&alpha_buf), 0); // alpha_raw
        encoder.set_buffer(3, Some(&layer_state.conv_weights), 0); // conv_weights
        encoder.set_buffer(4, Some(&layer_state.dt_bias), 0); // dt_bias
        encoder.set_buffer(5, Some(&layer_state.ssm_a), 0); // ssm_a
        encoder.set_buffer(6, Some(&layer_state.conv_state), 0); // conv_state (read-write)
        encoder.set_buffer(7, Some(&temp.qkv_conv), 0); // qkv_conv_out
        encoder.set_buffer(8, Some(&temp.beta), 0); // beta_out
        encoder.set_buffer(9, Some(&temp.gate), 0); // gate_out
        encoder.set_bytes(10, params); // DeltaParams

        // Resource usage hints
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
        encoder.use_resource(&*temp.qkv_conv, MTLResourceUsage::Write);
        encoder.use_resource(&*temp.beta, MTLResourceUsage::Write);
        encoder.use_resource(&*temp.gate, MTLResourceUsage::Write);

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
        // encoder dropped → end_encoding() вызывается автоматически
    }

    // ═══════════════════════════════════════════════════════════════
    // Kernel 2: delta_l2_norm_expand
    // Grid: (n_v_heads, head_k_dim, 1) = (32, 128, 1)
    // Threadgroup: (1, 128, 1) — одна v-голова = один threadgroup
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.l2_norm_expand);

        encoder.set_buffer(0, Some(&temp.qkv_conv), 0); // qkv_conv input
        encoder.set_buffer(1, Some(&temp.q), 0); // q_out
        encoder.set_buffer(2, Some(&temp.k), 0); // k_out
        encoder.set_buffer(3, Some(&temp.v), 0); // v_out
        encoder.set_bytes(4, params); // DeltaParams

        encoder.use_resource(&*temp.qkv_conv, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.q, MTLResourceUsage::Write);
        encoder.use_resource(&*temp.k, MTLResourceUsage::Write);
        encoder.use_resource(&*temp.v, MTLResourceUsage::Write);

        // dispatch_thread_groups: n_v_heads threadgroups, каждый (1, hkd) threads
        // Гарантирует точный размер threadgroup для shared memory reduction
        let tg_count = MTLSize {
            width: n_v,
            height: 1,
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
    // Kernel 3: delta_rule_kernel
    // n_v_heads threadgroups, каждый (1, head_v_dim) threads
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.delta_rule);

        encoder.set_buffer(0, Some(&temp.q), 0);
        encoder.set_buffer(1, Some(&temp.k), 0);
        encoder.set_buffer(2, Some(&temp.v), 0);
        encoder.set_buffer(3, Some(&temp.beta), 0);
        encoder.set_buffer(4, Some(&temp.gate), 0);
        encoder.set_buffer(5, Some(&layer_state.ssm_state), 0);
        encoder.set_buffer(6, Some(&temp.delta_output), 0);
        encoder.set_bytes(7, params);

        encoder.use_resource(&*temp.q, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.k, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.v, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.beta, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.gate, MTLResourceUsage::Read);
        encoder.use_resource(
            &*layer_state.ssm_state,
            MTLResourceUsage(MTLResourceUsage::Read.0 | MTLResourceUsage::Write.0),
        );
        encoder.use_resource(&*temp.delta_output, MTLResourceUsage::Write);

        let tg_count = MTLSize {
            width: n_v,
            height: 1,
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
    // Kernel 4: delta_norm_gate_kernel
    // n_v_heads threadgroups, каждый (1, head_v_dim) threads
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.norm_gate);

        encoder.set_buffer(0, Some(&temp.delta_output), 0);
        encoder.set_buffer(1, Some(&z_buf), 0);
        encoder.set_buffer(2, Some(&layer_state.norm_weight), 0);
        encoder.set_buffer(3, Some(&temp.gated_output), 0);
        encoder.set_bytes(4, params);

        encoder.use_resource(&*temp.delta_output, MTLResourceUsage::Read);
        encoder.use_resource(&*z_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*layer_state.norm_weight, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.gated_output, MTLResourceUsage::Write);

        let tg_count = MTLSize {
            width: n_v,
            height: 1,
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
    // Обернуть gated_output буфер в Candle Tensor (zero-copy)
    // ═══════════════════════════════════════════════════════════════
    let output_storage = MetalStorage::new(
        temp.gated_output.clone(),
        metal_device.clone(),
        value_dim,
        DType::F32,
    );
    let output_tensor = Tensor::from_storage(
        Storage::Metal(output_storage),
        (1, 1, value_dim),
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

/// Сброс состояния Metal буферов (для новой беседы).
pub fn clear_metal_state(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalState,
) -> Result<()> {
    // Ожидаем завершения всех GPU операций перед записью через CPU
    metal_device.wait_until_completed()?;

    // PoC: просто записываем нули через CPU (unified memory — прямой доступ)
    unsafe {
        let ssm_ptr = layer_state.ssm_state.contents() as *mut f32;
        let ssm_len = layer_state.ssm_state.length() / std::mem::size_of::<f32>();
        std::ptr::write_bytes(ssm_ptr, 0, ssm_len);

        let conv_ptr = layer_state.conv_state.contents() as *mut f32;
        let conv_len = layer_state.conv_state.length() / std::mem::size_of::<f32>();
        std::ptr::write_bytes(conv_ptr, 0, conv_len);
    }

    Ok(())
}

/// Захват snapshot GPU state в CPU Vec<f32> (T-274, Phase 2).
///
/// Выделяет два новых `Vec<f32>` под `ssm_state` и `conv_state` логического
/// размера (по `DeltaParams`) и копирует данные из GPU buffers через
/// `Buffer::contents()`.
///
/// **Важно**: используем `DeltaParams` для logical sizes, потому что
/// `Buffer::length()` возвращает физический размер (округлённый до page
/// boundary — например, запрос 98304 байт даёт 131072 байт buffer).
///
/// Требует sync GPU перед чтением — вызывает `wait_until_completed()` сам.
/// Caller дополнительный sync делать не обязан.
///
/// Возвращает `(ssm_state, conv_state)` в порядке полей `DeltaNetMetalState`.
pub fn snapshot_metal_state(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalState,
    params: &DeltaParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    // Ожидаем завершения всех GPU операций перед чтением через CPU
    metal_device.wait_until_completed()?;

    let n_v = params.n_v_heads as usize;
    let hd = params.head_v_dim as usize;
    let conv_k = params.conv_kernel as usize;
    let channels = params.channels as usize;

    let ssm_len = n_v * hd * hd;
    let conv_len = (conv_k - 1) * channels;

    // Sanity: физический buffer должен вмещать logical size.
    let ssm_phys = layer_state.ssm_state.length() / std::mem::size_of::<f32>();
    let conv_phys = layer_state.conv_state.length() / std::mem::size_of::<f32>();
    debug_assert!(
        ssm_phys >= ssm_len,
        "ssm physical {} < logical {}",
        ssm_phys,
        ssm_len
    );
    debug_assert!(
        conv_phys >= conv_len,
        "conv physical {} < logical {}",
        conv_phys,
        conv_len
    );

    let mut ssm = vec![0.0f32; ssm_len];
    let mut conv = vec![0.0f32; conv_len];

    // SAFETY: Metal storage shared (StorageModeShared) — unified memory.
    // После wait_until_completed GPU больше не пишет в эти буферы.
    unsafe {
        let ssm_ptr = layer_state.ssm_state.contents() as *const f32;
        std::ptr::copy_nonoverlapping(ssm_ptr, ssm.as_mut_ptr(), ssm_len);

        let conv_ptr = layer_state.conv_state.contents() as *const f32;
        std::ptr::copy_nonoverlapping(conv_ptr, conv.as_mut_ptr(), conv_len);
    }

    Ok((ssm, conv))
}

/// Восстановление GPU state из CPU Vec<f32> (T-274, Phase 2).
///
/// Перезаписывает `ssm_state` и `conv_state` GPU буферы данными из CPU slice'ов.
/// Размер проверяется против `DeltaParams` (logical size), а не `Buffer::length()`
/// (который возвращает physical, округлённый до page boundary).
///
/// Требует sync GPU перед записью (чтобы предыдущий forward завершился) — делается внутри.
///
/// Returns `Err` если размеры snapshot'а не совпадают с logical size.
pub fn restore_metal_state(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalState,
    params: &DeltaParams,
    ssm: &[f32],
    conv: &[f32],
) -> Result<()> {
    let n_v = params.n_v_heads as usize;
    let hd = params.head_v_dim as usize;
    let conv_k = params.conv_kernel as usize;
    let channels = params.channels as usize;

    let ssm_len = n_v * hd * hd;
    let conv_len = (conv_k - 1) * channels;

    if ssm.len() != ssm_len {
        candle_core::bail!(
            "restore_metal_state: ssm size mismatch: snap={} vs logical={}",
            ssm.len(),
            ssm_len
        );
    }
    if conv.len() != conv_len {
        candle_core::bail!(
            "restore_metal_state: conv size mismatch: snap={} vs logical={}",
            conv.len(),
            conv_len
        );
    }

    // Sync: дожидаемся всех pending GPU ops, чтобы не писать в buffer,
    // в который GPU всё ещё читает/пишет.
    metal_device.wait_until_completed()?;

    // SAFETY: Metal shared storage; после wait_until_completed CPU может писать.
    // Физический размер buffer'а >= logical (page-aligned), но пишем только logical.
    unsafe {
        let ssm_ptr = layer_state.ssm_state.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(ssm.as_ptr(), ssm_ptr, ssm_len);

        let conv_ptr = layer_state.conv_state.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(conv.as_ptr(), conv_ptr, conv_len);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// Вспомогательные функции
// ═══════════════════════════════════════════════════════════════

/// Извлекает raw Metal Buffer из Candle Tensor.
/// Tensor должен быть F32 (per-token decode kernels работают в F32 для precision).
fn get_metal_buffer(tensor: &Tensor) -> Result<Arc<Buffer>> {
    debug_assert_eq!(tensor.dtype(), DType::F32);
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => {
            // Клонируем Arc<Buffer> для использования после drop storage guard
            // Это безопасно — Arc просто увеличивает refcount
            Ok(Arc::new(ms.buffer().clone()))
        }
        _ => candle_core::bail!("delta_rule_metal: tensor не на Metal device"),
    }
}

// ═══════════════════════════════════════════════════════════════
// Тестовые утилиты (для S-4: accuracy test)
// ═══════════════════════════════════════════════════════════════

/// Создаёт Metal буфер из f32 slice (для тестирования).
pub fn buffer_from_f32(metal_device: &MetalDevice, data: &[f32]) -> Result<Arc<Buffer>> {
    metal_device.new_buffer_with_data(data)
}

/// Читает Metal буфер в Vec<f32> (для тестирования).
/// ВАЖНО: требует GPU→CPU sync перед вызовом!
pub fn buffer_to_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
    unsafe {
        let ptr = buffer.contents() as *const f32;
        std::slice::from_raw_parts(ptr, count).to_vec()
    }
}
