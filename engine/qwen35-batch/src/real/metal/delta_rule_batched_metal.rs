// delta_rule_batched_metal.rs — Rust bindings для batched DeltaNet decode Metal kernels
//
// Phase 1 true batched decode: 4 kernel из delta_rule_batched.metal с осью slot B.
// В отличие от delta_rule_metal.rs (single-token, time-multiplex), здесь один
// dispatch обрабатывает B одновременных слотов: state/scratch с leading B,
// grid добавляет B ось. Чтение весов (conv/dt_bias/ssm_a/norm) — одно на B слотов
// (shared across slots), что даёт bandwidth выигрыш — фундамент continuous batching.
//
// Архитектура (идентична delta_rule_metal.rs, + slot ось):
// 1. delta_conv1d_prep_batched — Conv1d + sigmoid(beta) + softplus(alpha)*A, per slot
// 2. delta_l2_norm_expand_batched — L2 norm + head expansion 16→32 + Q scaling, per (head,slot)
// 3. delta_rule_kernel_batched — Decay + S^T@k + delta + rank-1 update + S^T@q, per (head,slot)
// 4. delta_norm_gate_kernel_batched — Group RMS norm + SiLU(z) gating, per (head,slot)
//
// Layout буферов (leading B): см. delta_rule_batched.metal.

use candle_core::{DType, MetalDevice, MetalStorage, Result, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MtlDevice};
use objc2_metal::{MTLResourceUsage, MTLSize};
use std::sync::Arc;

/// Metal shader source — включается при компиляции.
const DELTA_RULE_BATCHED_METAL_SOURCE: &str = include_str!("delta_rule_batched.metal");

/// Параметры DeltaNet слоя для batched kernel. Layout ДОЛЖЕН совпадать со
/// struct DeltaParams в delta_rule_batched.metal.
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
    /// B — число одновременных слотов (batch). 1..MAX_BATCH.
    pub batch_size: u32,
}

/// Скомпилированные Metal pipelines для batched delta_rule.
/// Создаются один раз при загрузке модели, переиспользуются для всех слоёв.
pub struct DeltaRuleBatchedPipelines {
    pub conv1d_prep: ComputePipeline,
    pub l2_norm_expand: ComputePipeline,
    pub delta_rule: ComputePipeline,
    pub norm_gate: ComputePipeline,
}

/// Persistent batched Metal буферы для одного DeltaNet слоя.
/// Ведущие B: state изолирован per slot. Веса — shared across slots (без B).
/// Живут всё время жизни модели; сбрасываются при clear().
pub struct DeltaNetMetalStateBatched {
    /// SSM state: [B * n_v_heads * head_v_dim * head_v_dim] floats
    pub ssm_state: Arc<Buffer>,
    /// Conv1d state: [B * (conv_k - 1) * channels] floats
    pub conv_state: Arc<Buffer>,
    /// Conv1d kernel weights: [channels * conv_k] — shared across slots
    pub conv_weights: Arc<Buffer>,
    /// dt_bias: [n_v_heads] — shared across slots
    pub dt_bias: Arc<Buffer>,
    /// ssm_a: [n_v_heads] — shared across slots (negative decay)
    pub ssm_a: Arc<Buffer>,
    /// RMS norm weight: [head_v_dim] — shared across slots/heads
    pub norm_weight: Arc<Buffer>,
    /// B — capacity (slot count), фиксированная при аллокации.
    pub capacity_b: u32,
}

/// Временные (temp) batched буферы, переиспользуемые между слоями.
/// Все scratch с leading B: один dispatch покрывает B слотов.
pub struct DeltaNetTempBuffersBatched {
    /// qkv_conv: [B * channels] floats — conv1d output
    pub qkv_conv: Arc<Buffer>,
    /// q: [B * n_v_heads * head_k_dim] floats
    pub q: Arc<Buffer>,
    /// k: [B * n_v_heads * head_k_dim] floats
    pub k: Arc<Buffer>,
    /// v: [B * n_v_heads * head_v_dim] floats
    pub v: Arc<Buffer>,
    /// beta: [B * n_v_heads] floats
    pub beta: Arc<Buffer>,
    /// gate: [B * n_v_heads] floats
    pub gate: Arc<Buffer>,
    /// delta_output: [B * n_v_heads * head_v_dim] floats — raw recurrent output
    pub delta_output: Arc<Buffer>,
    /// gated_output: [B * value_dim] floats — final gated output
    pub gated_output: Arc<Buffer>,
    /// B — capacity (slot count), фиксированная при аллокации.
    pub capacity_b: u32,
}

/// Компилирует 4 batched Metal шейдера и создаёт pipelines.
pub fn compile_delta_rule_batched_pipelines(
    metal_device: &MetalDevice,
) -> Result<DeltaRuleBatchedPipelines> {
    let device: &MtlDevice = metal_device.metal_device();

    let library = device
        .new_library_with_source(DELTA_RULE_BATCHED_METAL_SOURCE, None)
        .map_err(candle_core::Error::wrap)?;

    let fn_conv1d_prep = library
        .get_function("delta_conv1d_prep_batched", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_l2_norm = library
        .get_function("delta_l2_norm_expand_batched", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_delta_rule = library
        .get_function("delta_rule_kernel_batched", None)
        .map_err(candle_core::Error::wrap)?;
    let fn_norm_gate = library
        .get_function("delta_norm_gate_kernel_batched", None)
        .map_err(candle_core::Error::wrap)?;

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

    Ok(DeltaRuleBatchedPipelines {
        conv1d_prep,
        l2_norm_expand,
        delta_rule,
        norm_gate,
    })
}

/// Создаёт persistent batched Metal буферы для одного DeltaNet слоя.
///
/// `capacity_b` — максимальное число одновременных слотов. State аллоцируется
/// под полный capacity; при batch_size < capacity_b просто используется часть.
///
/// `conv_weights_data`, `dt_bias_data`, `ssm_a_data`, `norm_weight_data` —
/// данные весов из GGUF (shared across slots, без B дублирования).
pub fn create_layer_metal_state_batched(
    metal_device: &MetalDevice,
    params: &DeltaParams,
    capacity_b: u32,
    conv_weights_data: &[f32],
    dt_bias_data: &[f32],
    ssm_a_data: &[f32],
    norm_weight_data: &[f32],
) -> Result<DeltaNetMetalStateBatched> {
    let hd = params.head_v_dim as usize;
    let n_v = params.n_v_heads as usize;
    let channels = params.channels as usize;
    let conv_k = params.conv_kernel as usize;
    let b = capacity_b as usize;

    // SSM state — zero-initialized, [B * n_v * hd * hd]
    let ssm_state_size = b * n_v * hd * hd * std::mem::size_of::<f32>();
    let ssm_state = metal_device.allocate_zeros(ssm_state_size)?;

    // Conv state — zero-initialized, [B * (conv_k - 1) * channels]
    let conv_state_size = b * (conv_k - 1) * channels * std::mem::size_of::<f32>();
    let conv_state = metal_device.allocate_zeros(conv_state_size)?;

    // Веса — shared across slots (без B)
    let conv_weights = metal_device.new_buffer_with_data(conv_weights_data)?;
    let dt_bias = metal_device.new_buffer_with_data(dt_bias_data)?;
    let ssm_a = metal_device.new_buffer_with_data(ssm_a_data)?;
    let norm_weight = metal_device.new_buffer_with_data(norm_weight_data)?;

    Ok(DeltaNetMetalStateBatched {
        ssm_state,
        conv_state,
        conv_weights,
        dt_bias,
        ssm_a,
        norm_weight,
        capacity_b,
    })
}

/// Создаёт batched временные буферы под capacity_b слотов.
/// Переиспользуются между слоями (одни tempures на модель или per-layer — на выбор caller'а).
pub fn create_temp_buffers_batched(
    metal_device: &MetalDevice,
    params: &DeltaParams,
    capacity_b: u32,
) -> Result<DeltaNetTempBuffersBatched> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let hkd = params.head_k_dim as usize;
    let hvd = params.head_v_dim as usize;
    let value_dim = params.value_dim as usize;
    let b = capacity_b as usize;

    // Decode path: F32 для precision (см. delta_rule_metal.rs — decode precision critical).
    Ok(DeltaNetTempBuffersBatched {
        qkv_conv: metal_device.new_buffer(b * channels, DType::F32, "b_qkv_conv")?,
        q: metal_device.new_buffer(b * n_v * hkd, DType::F32, "b_q")?,
        k: metal_device.new_buffer(b * n_v * hkd, DType::F32, "b_k")?,
        v: metal_device.new_buffer(b * n_v * hvd, DType::F32, "b_v")?,
        beta: metal_device.new_buffer(b * n_v, DType::F32, "b_beta")?,
        gate: metal_device.new_buffer(b * n_v, DType::F32, "b_gate")?,
        delta_output: metal_device.new_buffer(b * n_v * hvd, DType::F32, "b_delta_out")?,
        gated_output: metal_device.new_buffer(b * value_dim, DType::F32, "b_gated_out")?,
        capacity_b,
    })
}

/// Выполняет полный batched delta_rule forward pass на Metal GPU.
///
/// Входы: 4 тензора-результата batched QMatMul проекций (уже на GPU):
///   qkv_t   [B, 1, channels]
///   z_t     [B, 1, value_dim]
///   beta_t  [B, 1, n_v_heads]
///   alpha_t [B, 1, n_v_heads]
/// Выход: тензор [B, 1, value_dim] на GPU, готовый для batched ssm_out QMatMul.
///
/// `params.batch_size` — активное число слотов (<= state/temp capacity_b).
/// ВСЁ выполняется на GPU — ноль GPU↔CPU синхронизаций.
pub fn dispatch_delta_rule_batched(
    metal_device: &MetalDevice,
    pipelines: &DeltaRuleBatchedPipelines,
    layer_state: &DeltaNetMetalStateBatched,
    temp: &DeltaNetTempBuffersBatched,
    params: &DeltaParams,
    // Входные batched тензоры (результаты batched QMatMul, contiguous F32 на GPU)
    qkv_t: &Tensor,   // [B, 1, channels]
    z_t: &Tensor,     // [B, 1, value_dim]
    beta_t: &Tensor,  // [B, 1, n_v_heads]
    alpha_t: &Tensor, // [B, 1, n_v_heads]
    // Indirection: batch_idx → реальный slot_idx (persistent state индексация).
    // Длина B. Нужен т.к. активный batch может быть меньше capacity, и порядок
    // слотов в batch'е не обязан совпадать с их физическими region-индексами.
    slot_ids: &[u32],
) -> Result<Tensor> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let hkd = params.head_k_dim as usize;
    let hvd = params.head_v_dim as usize;
    let value_dim = params.value_dim as usize;
    let b = params.batch_size as usize;

    // Sanity: активный batch не превышает capacity аллоцированных буферов.
    debug_assert!(
        b as u32 <= layer_state.capacity_b,
        "batch_size {} > state capacity {}",
        b,
        layer_state.capacity_b
    );
    debug_assert!(
        b as u32 <= temp.capacity_b,
        "batch_size {} > temp capacity {}",
        b,
        temp.capacity_b
    );

    // Decode path: F32 contiguous для precision.
    let qkv_t_f32 = ensure_f32_contig(qkv_t)?;
    let z_t_f32 = ensure_f32_contig(z_t)?;
    let beta_t_f32 = ensure_f32_contig(beta_t)?;
    let alpha_t_f32 = ensure_f32_contig(alpha_t)?;

    // Проверяем форму входов (leading B).
    let (b_qkv, _s, _c) = qkv_t_f32.dims3()?;
    let (b_z, _s, _v) = z_t_f32.dims3()?;
    let (b_beta, _s, _h) = beta_t_f32.dims3()?;
    let (b_alpha, _s, _h) = alpha_t_f32.dims3()?;
    if b_qkv != b || b_z != b || b_beta != b || b_alpha != b {
        candle_core::bail!(
            "dispatch_delta_rule_batched: input batch mismatch: qkv={} z={} beta={} alpha={} vs params.batch_size={}",
            b_qkv, b_z, b_beta, b_alpha, b
        );
    }

    // Извлекаем raw Metal буферы (flat [B * ...], leading B — kernel индексирует по slot).
    let qkv_buf = get_metal_buffer(&qkv_t_f32)?;
    let z_buf = get_metal_buffer(&z_t_f32)?;
    let beta_buf = get_metal_buffer(&beta_t_f32)?;
    let alpha_buf = get_metal_buffer(&alpha_t_f32)?;

    // Indirection buffer: batch_idx → real slot_idx. Persistent state (ssm_state,
    // conv_state) индексируется через slot_ids[bidx], а temp/входы — по batch_idx.
    // Без этого при batch shrink (один slot finished) оставшийся slot попал бы в
    // чужой state-region → drift.
    assert_eq!{
        slot_ids.len(),
        b,
        "dispatch_delta_rule_batched: slot_ids.len {} != batch_size {}",
        slot_ids.len(),
        b,
    };
    let slot_ids_buf = metal_device.new_buffer_with_data(slot_ids)?;

    // ═══════════════════════════════════════════════════════════════
    // Kernel 1: delta_conv1d_prep_batched
    // Grid: (channels, B, 1), Threadgroup: (256, 1, 1) — один поток на (channel, slot)
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.conv1d_prep);

        encoder.set_buffer(0, Some(&qkv_buf), 0);               // qkv_raw
        encoder.set_buffer(1, Some(&beta_buf), 0);              // beta_raw
        encoder.set_buffer(2, Some(&alpha_buf), 0);             // alpha_raw
        encoder.set_buffer(3, Some(&layer_state.conv_weights), 0);
        encoder.set_buffer(4, Some(&layer_state.dt_bias), 0);
        encoder.set_buffer(5, Some(&layer_state.ssm_a), 0);
        encoder.set_buffer(6, Some(&layer_state.conv_state), 0); // read-write
        encoder.set_buffer(7, Some(&temp.qkv_conv), 0);          // output
        encoder.set_buffer(8, Some(&temp.beta), 0);              // output
        encoder.set_buffer(9, Some(&temp.gate), 0);              // output
        encoder.set_bytes(10, params);                           // DeltaParams
        encoder.set_buffer(11, Some(&slot_ids_buf), 0);            // slot_ids (indirection)

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
        encoder.use_resource(&*slot_ids_buf, MTLResourceUsage::Read);

        let grid = MTLSize {
            width: channels,
            height: b,
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
    // Kernel 2: delta_l2_norm_expand_batched
    // Grid: threadgroups (n_v_heads, B, 1), each threadgroup (1, head_k_dim, 1)
    // Один threadgroup = один (v_head, slot) — shared memory reduction.
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.l2_norm_expand);

        encoder.set_buffer(0, Some(&temp.qkv_conv), 0); // input
        encoder.set_buffer(1, Some(&temp.q), 0);        // output
        encoder.set_buffer(2, Some(&temp.k), 0);        // output
        encoder.set_buffer(3, Some(&temp.v), 0);        // output
        encoder.set_bytes(4, params);                    // DeltaParams

        encoder.use_resource(&*temp.qkv_conv, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.q, MTLResourceUsage::Write);
        encoder.use_resource(&*temp.k, MTLResourceUsage::Write);
        encoder.use_resource(&*temp.v, MTLResourceUsage::Write);

        let tg_count = MTLSize {
            width: n_v,
            height: b,
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
    // Kernel 3: delta_rule_kernel_batched
    // Grid: threadgroups (n_v_heads, B, 1), each threadgroup (1, head_v_dim, 1)
    // Один threadgroup = один (head, slot) —shared mem изоляция автоматична.
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.delta_rule);

        encoder.set_buffer(0, Some(&temp.q), 0);
        encoder.set_buffer(1, Some(&temp.k), 0);
        encoder.set_buffer(2, Some(&temp.v), 0);
        encoder.set_buffer(3, Some(&temp.beta), 0);
        encoder.set_buffer(4, Some(&temp.gate), 0);
        encoder.set_buffer(5, Some(&layer_state.ssm_state), 0); // read-write
        encoder.set_buffer(6, Some(&temp.delta_output), 0);     // output
        encoder.set_bytes(7, params);                           // DeltaParams
        encoder.set_buffer(8, Some(&slot_ids_buf), 0);           // slot_ids (indirection)

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
        encoder.use_resource(&*slot_ids_buf, MTLResourceUsage::Read);

        let tg_count = MTLSize {
            width: n_v,
            height: b,
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
    // Kernel 4: delta_norm_gate_kernel_batched
    // Grid: threadgroups (n_v_heads, B, 1), each threadgroup (1, head_v_dim, 1)
    // ═══════════════════════════════════════════════════════════════
    {
        let encoder = metal_device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipelines.norm_gate);

        encoder.set_buffer(0, Some(&temp.delta_output), 0); // input
        encoder.set_buffer(1, Some(&z_buf), 0);            // z (batched)
        encoder.set_buffer(2, Some(&layer_state.norm_weight), 0);
        encoder.set_buffer(3, Some(&temp.gated_output), 0); // output
        encoder.set_bytes(4, params);                        // DeltaParams

        encoder.use_resource(&*temp.delta_output, MTLResourceUsage::Read);
        encoder.use_resource(&*z_buf, MTLResourceUsage::Read);
        encoder.use_resource(&*layer_state.norm_weight, MTLResourceUsage::Read);
        encoder.use_resource(&*temp.gated_output, MTLResourceUsage::Write);

        let tg_count = MTLSize {
            width: n_v,
            height: b,
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
    // Обернуть batched gated_output буфер в Candle Tensor (zero-copy)
    // Shape: [B, 1, value_dim] — leading B, готово для batched ssm_out QMatMul.
    // ═══════════════════════════════════════════════════════════════
    let output_storage = MetalStorage::new(
        temp.gated_output.clone(),
        metal_device.clone(),
        b * value_dim,
        DType::F32,
    );
    let output_tensor = Tensor::from_storage(
        Storage::Metal(output_storage),
        (b, 1, value_dim),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok(output_tensor)
}

/// Cast tensor в F32 contiguous если он не F32 (decode precision critical).
fn ensure_f32_contig(tensor: &Tensor) -> Result<Tensor> {
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

/// Сброс state batched буферов (zero всей allocated capacity) для новой беседы.
pub fn clear_metal_state_batched(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalStateBatched,
) -> Result<()> {
    // Ожидаем завершения всех GPU операций перед записью через CPU.
    metal_device.wait_until_completed()?;

    // Unified memory — прямой доступ. Зануляем всю capacity (B * ...).
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

/// Записывает single-slot state в конкретный slot batched-буфера (htod в unified
/// memory). Phase 4 adapter: seed per-slot state из prefill snapshot в batched
/// decode buffers перед первым batched decode-шагом.
///
/// `ssm_init` — [n_v_heads * head_v_dim * head_v_dim] (один slot, без B).
/// `conv_init` — [(conv_kernel - 1) * channels] (один slot, без B).
/// `slot` — индекс слота [0, capacity_b).
///
/// Layout batched-буфера: leading B → slot-регион начинается по смещению
/// `slot * per_slot_len`. Записываем только per-slot данные (остальные слоты
/// не трогаем).
pub struct DeltaNetMetalSlotCheckpoint {
    ssm_state: Arc<Buffer>,
    conv_state: Arc<Buffer>,
}

/// GPU blit snapshot of one slot region. No CPU readback.
pub fn checkpoint_slot_metal_state(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalStateBatched,
    slot: usize,
) -> Result<DeltaNetMetalSlotCheckpoint> {
    let capacity = layer_state.capacity_b as usize;
    if slot >= capacity {
        candle_core::bail!("checkpoint slot {slot} >= capacity {capacity}");
    }
    let ssm_bytes = layer_state.ssm_state.length() / capacity;
    let conv_bytes = layer_state.conv_state.length() / capacity;
    let ssm_state = metal_device.allocate_zeros(ssm_bytes)?;
    let conv_state = metal_device.allocate_zeros(conv_bytes)?;
    let mut blit = metal_device.blit_command_encoder()?;
    blit.copy_from_buffer(
        &layer_state.ssm_state,
        slot * ssm_bytes,
        &ssm_state,
        0,
        ssm_bytes,
    );
    blit.copy_from_buffer(
        &layer_state.conv_state,
        slot * conv_bytes,
        &conv_state,
        0,
        conv_bytes,
    );
    Ok(DeltaNetMetalSlotCheckpoint {
        ssm_state,
        conv_state,
    })
}

pub fn restore_slot_metal_checkpoint(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalStateBatched,
    slot: usize,
    checkpoint: &DeltaNetMetalSlotCheckpoint,
) -> Result<()> {
    let capacity = layer_state.capacity_b as usize;
    if slot >= capacity {
        candle_core::bail!("restore slot {slot} >= capacity {capacity}");
    }
    let ssm_bytes = layer_state.ssm_state.length() / capacity;
    let conv_bytes = layer_state.conv_state.length() / capacity;
    if checkpoint.ssm_state.length() != ssm_bytes || checkpoint.conv_state.length() != conv_bytes {
        candle_core::bail!("Metal slot checkpoint shape mismatch");
    }
    let mut blit = metal_device.blit_command_encoder()?;
    blit.copy_from_buffer(
        &checkpoint.ssm_state,
        0,
        &layer_state.ssm_state,
        slot * ssm_bytes,
        ssm_bytes,
    );
    blit.copy_from_buffer(
        &checkpoint.conv_state,
        0,
        &layer_state.conv_state,
        slot * conv_bytes,
        conv_bytes,
    );
    Ok(())
}

pub fn restore_slot_metal_state(
    metal_device: &MetalDevice,
    layer_state: &DeltaNetMetalStateBatched,
    slot: usize,
    ssm_init: &[f32],
    conv_init: &[f32],
) -> Result<()> {
    metal_device.wait_until_completed()?;
    let b = layer_state.capacity_b as usize;
    debug_assert!(slot < b, "restore_slot: slot {slot} >= capacity_b {b}");
    let ssm_len = ssm_init.len();
    let conv_len = conv_init.len();

    unsafe {
        let ssm_ptr = layer_state.ssm_state.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(ssm_init.as_ptr(), ssm_ptr.add(slot * ssm_len), ssm_len);

        let conv_ptr = layer_state.conv_state.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(conv_init.as_ptr(), conv_ptr.add(slot * conv_len), conv_len);
    }
    Ok(())
}

/// Извлекает raw Metal Buffer из Candle Tensor.
/// Tensor должен быть F32 contiguous (batched decode kernels работают в F32).
fn get_metal_buffer(tensor: &Tensor) -> Result<Arc<Buffer>> {
    debug_assert_eq!(tensor.dtype(), DType::F32);
    debug_assert!(tensor.is_contiguous(), "delta_rule_batched: tensor не contiguous");
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(ms) => Ok(Arc::new(ms.buffer().clone())),
        _ => candle_core::bail!("delta_rule_batched_metal: tensor не на Metal device"),
    }
}

// ═══════════════════════════════════════════════════════════════
// Phase 1 unit-тесты: batched kernel parity vs single-token kernel.
//
// Доказательство корректности оси slot: один batched dispatch [B,1,...]
// даёт per-slot output, бит-точно равный B независимым single-token
// dispatch'ам с теми же per-slot входами и идентичным начальным state.
// Это валидирует, что slot-индексация и (head,slot) изоляция корректны —
// math per (head,slot) идентична per-head math из delta_rule.metal.
//
// Запуск (macOS, Metal):
//   cargo test -p qwen35-batch --features "real-model metal" \
//       --lib delta_rule_batched_metal::tests -- --nocapture
// ═══════════════════════════════════════════════════════════════

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::real::metal::delta_rule_metal::{
        self, DeltaNetMetalState as SingleState, DeltaNetTempBuffers as SingleTemp,
    };
    use candle_core::Device;

    /// Детерминированный LCG (без внешних rng deps) → f32 в [-1, 1).
    struct Lcg {
        s: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { s: seed.wrapping_add(0x9E3779B97F4A7C15) }
        }
        fn next_u32(&mut self) -> u32 {
            // PCG-style: LCG + xorshift high bits
            self.s = self
                .s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.s >> 33) as u32) ^ ((self.s >> 11) as u32)
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn fill(&mut self, out: &mut [f32]) {
            for x in out.iter_mut() {
                *x = self.next_f32();
            }
        }
    }

    /// Малые размерности (близко к Qwen3.5-4B, уменьшено для скорости):
    /// n_k=8, n_v=16 (GQA 2:1), hd=16 → key=128, value=256, channels=512.
    fn small_params_single() -> delta_rule_metal::DeltaParams {
        delta_rule_metal::DeltaParams {
            n_k_heads: 8,
            n_v_heads: 16,
            head_k_dim: 16,
            head_v_dim: 16,
            key_dim: 128,
            value_dim: 256,
            channels: 512,
            conv_kernel: 4,
            q_scale: 1.0 / 4.0, // 1/sqrt(16)
            rms_norm_eps: 1e-6,
            heads_per_kv: 2,
        }
    }

    fn small_params_batched(b: u32) -> DeltaParams {
        DeltaParams {
            n_k_heads: 8,
            n_v_heads: 16,
            head_k_dim: 16,
            head_v_dim: 16,
            key_dim: 128,
            value_dim: 256,
            channels: 512,
            conv_kernel: 4,
            q_scale: 1.0 / 4.0,
            rms_norm_eps: 1e-6,
            heads_per_kv: 2,
            batch_size: b,
        }
    }

    /// Генерирует shared веса + per-slot входы + (опц.) per-slot начальный state.
    struct Fixture {
        conv_weights: Vec<f32>, // [channels*conv_k]
        dt_bias: Vec<f32>,      // [n_v]
        ssm_a: Vec<f32>,        // [n_v] — отрицательные decay
        norm_weight: Vec<f32>,  // [hd]
        // per-slot входы
        qkv: Vec<Vec<f32>>,  // [B][channels]
        z: Vec<Vec<f32>>,    // [B][value_dim]
        beta: Vec<Vec<f32>>, // [B][n_v]
        alpha: Vec<Vec<f32>>, // [B][n_v]
        // начальный state per slot (одинаковый для всех slots — для parity)
        ssm_init: Vec<f32>, // [n_v*hd*hd]
        conv_init: Vec<f32>, // [(conv_k-1)*channels]
    }

    fn make_fixture(b: usize, seed: u64) -> Fixture {
        let n_v = 16usize;
        let hd = 16usize;
        let channels = 512usize;
        let conv_k = 4usize;
        let value_dim = 256usize;
        let mut rng = Lcg::new(seed);

        let conv_weights = {
            let mut v = vec![0.0f32; channels * conv_k];
            rng.fill(&mut v);
            v
        };
        // dt_bias маленький положительный; ssm_a отрицательный (decay gate).
        let dt_bias = (0..n_v).map(|_| rng.next_f32() * 0.2).collect::<Vec<_>>();
        let ssm_a = (0..n_v).map(|_| -(0.05 + (rng.next_f32().abs()) * 0.8)).collect::<Vec<_>>();
        let norm_weight = (0..hd).map(|_| 0.5 + rng.next_f32().abs()).collect::<Vec<_>>();

        let qkv = (0..b)
            .map(|_| {
                let mut v = vec![0.0f32; channels];
                rng.fill(&mut v);
                v
            })
            .collect();
        let z = (0..b)
            .map(|_| {
                let mut v = vec![0.0f32; value_dim];
                rng.fill(&mut v);
                v
            })
            .collect();
        let beta = (0..b)
            .map(|_| {
                let mut v = vec![0.0f32; n_v];
                rng.fill(&mut v);
                v
            })
            .collect();
        let alpha = (0..b)
            .map(|_| {
                let mut v = vec![0.0f32; n_v];
                rng.fill(&mut v);
                v
            })
            .collect();

        let ssm_init = {
            let mut v = vec![0.0f32; n_v * hd * hd];
            rng.fill(&mut v);
            v
        };
        let conv_init = {
            let mut v = vec![0.0f32; (conv_k - 1) * channels];
            rng.fill(&mut v);
            v
        };

        Fixture {
            conv_weights,
            dt_bias,
            ssm_a,
            norm_weight,
            qkv,
            z,
            beta,
            alpha,
            ssm_init,
            conv_init,
        }
    }

    /// Поднимает Metal device или skip-ает тест ( нет GPU / CI ).
    fn metal_device() -> Option<(Device, MetalDevice)> {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP batched parity: Metal device unavailable: {e}");
                return None;
            }
        };
        let metal_device = match &device {
            Device::Metal(m) => m.clone(),
            _ => unreachable!(),
        };
        Some((device, metal_device))
    }

    /// Запускает single-token kernel для одного slot'а с заданными входами и
    /// начальным state, возвращает output [value_dim].
    fn run_single(
        metal_device: &MetalDevice,
        single_pipelines: &delta_rule_metal::DeltaRulePipelines,
        single_state: &SingleState,
        single_temp: &SingleTemp,
        params: &delta_rule_metal::DeltaParams,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
        device: &Device,
    ) -> candle_core::Result<Vec<f32>> {
        let (b, _s, _c) = (1usize, 1usize, qkv.len());
        let qkv_t = Tensor::from_vec(qkv.to_vec(), (b, 1, qkv.len()), device)?;
        let z_t = Tensor::from_vec(z.to_vec(), (b, 1, z.len()), device)?;
        let beta_t = Tensor::from_vec(beta.to_vec(), (b, 1, beta.len()), device)?;
        let alpha_t = Tensor::from_vec(alpha.to_vec(), (b, 1, alpha.len()), device)?;

        let out = delta_rule_metal::dispatch_delta_rule(
            metal_device,
            single_pipelines,
            single_state,
            single_temp,
            params,
            &qkv_t,
            &z_t,
            &beta_t,
            &alpha_t,
        )?;
        // to_vec1 syncs GPU→CPU и копирует (zero-copy tensor алиасит temp.gated_output).
        out.flatten_all()?.to_vec1::<f32>()
    }

    /// Запускает batched kernel для B слотов, возвращает [B][value_dim].
    fn run_batched(
        metal_device: &MetalDevice,
        batched_pipelines: &DeltaRuleBatchedPipelines,
        batched_state: &DeltaNetMetalStateBatched,
        batched_temp: &DeltaNetTempBuffersBatched,
        params: &DeltaParams,
        fx: &Fixture,
        device: &Device,
    ) -> candle_core::Result<Vec<Vec<f32>>> {
        let b = params.batch_size as usize;
        let channels = params.channels as usize;
        let value_dim = params.value_dim as usize;
        let n_v = params.n_v_heads as usize;

        // Interleave per-slot inputs в [B, 1, ...] contiguous (row-major, leading B).
        let mut qkv_flat = Vec::with_capacity(b * channels);
        let mut z_flat = Vec::with_capacity(b * value_dim);
        let mut beta_flat = Vec::with_capacity(b * n_v);
        let mut alpha_flat = Vec::with_capacity(b * n_v);
        for s in 0..b {
            qkv_flat.extend_from_slice(&fx.qkv[s]);
            z_flat.extend_from_slice(&fx.z[s]);
            beta_flat.extend_from_slice(&fx.beta[s]);
            alpha_flat.extend_from_slice(&fx.alpha[s]);
        }
        let qkv_t = Tensor::from_vec(qkv_flat, (b, 1, channels), device)?;
        let z_t = Tensor::from_vec(z_flat, (b, 1, value_dim), device)?;
        let beta_t = Tensor::from_vec(beta_flat, (b, 1, n_v), device)?;
        let alpha_t = Tensor::from_vec(alpha_flat, (b, 1, n_v), device)?;

        let out = dispatch_delta_rule_batched(
            metal_device,
            batched_pipelines,
            batched_state,
            batched_temp,
            params,
            &qkv_t,
            &z_t,
            &beta_t,
            &alpha_t,
            &(0..b).map(|i| i as u32).collect::<Vec<_>>(),
        )?;
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        Ok(flat.chunks(value_dim).map(|c| c.to_vec()).collect())
    }

    /// Записывает batched state buffer: B копий per-slot init-данных [slot, ...].
    fn write_batched_state(
        metal_device: &MetalDevice,
        state: &DeltaNetMetalStateBatched,
        ssm_init: &[f32],
        conv_init: &[f32],
    ) -> candle_core::Result<()> {
        metal_device.wait_until_completed()?;
        let b = state.capacity_b as usize;
        unsafe {
            let ssm_ptr = state.ssm_state.contents() as *mut f32;
            let ssm_len = ssm_init.len();
            for s in 0..b {
                std::ptr::copy_nonoverlapping(
                    ssm_init.as_ptr(),
                    ssm_ptr.add(s * ssm_len),
                    ssm_len,
                );
            }
            let conv_ptr = state.conv_state.contents() as *mut f32;
            let conv_len = conv_init.len();
            for s in 0..b {
                std::ptr::copy_nonoverlapping(
                    conv_init.as_ptr(),
                    conv_ptr.add(s * conv_len),
                    conv_len,
                );
            }
        }
        Ok(())
    }

    /// Основной parity check для заданного B и seed: batched per-slot == single per-slot,
/// с идентичным НЕ-нулевым начальным state (ssa + conv) per slot.
    fn assert_parity(b: usize, seed: u64) {
        let (device, metal_device) = match metal_device() {
            Some(x) => x,
            None => return,
        };

        let sp = small_params_single();
        let bp = small_params_batched(b as u32);
        let fx = make_fixture(b, seed);

        // Pipelines: single + batched (компиляция MSL — валидирует синтаксис ядер).
        let single_pip = delta_rule_metal::compile_delta_rule_pipelines(&metal_device)
            .expect("compile single pipelines");
        let batched_pip = compile_delta_rule_batched_pipelines(&metal_device)
            .expect("compile batched pipelines");

        // --- Single path: B независимых dispatch'ей, каждый из идентичного init state ---
        let single_state = delta_rule_metal::create_layer_metal_state(
            &metal_device,
            &sp,
            &fx.conv_weights,
            &fx.dt_bias,
            &fx.ssm_a,
            &fx.norm_weight,
        )
        .expect("create single state");
        let single_temp = delta_rule_metal::create_temp_buffers(&metal_device, &sp)
            .expect("create single temp");

        let mut single_outputs = Vec::with_capacity(b);
        for s in 0..b {
            // Restore одинакового init state per slot (полная перезапись → изоляция).
            delta_rule_metal::restore_metal_state(
                &metal_device,
                &single_state,
                &sp,
                &fx.ssm_init,
                &fx.conv_init,
            )
            .expect("restore single state");
            let out = run_single(
                &metal_device,
                &single_pip,
                &single_state,
                &single_temp,
                &sp,
                &fx.qkv[s],
                &fx.z[s],
                &fx.beta[s],
                &fx.alpha[s],
                &device,
            )
            .expect("run single dispatch");
            single_outputs.push(out);
        }

        // --- Batched path: один dispatch [B,1,...] из того же init state ---
        let batched_state = create_layer_metal_state_batched(
            &metal_device,
            &bp,
            b as u32,
            &fx.conv_weights,
            &fx.dt_bias,
            &fx.ssm_a,
            &fx.norm_weight,
        )
        .expect("create batched state");
        let batched_temp = create_temp_buffers_batched(&metal_device, &bp, b as u32)
            .expect("create batched temp");
        write_batched_state(&metal_device, &batched_state, &fx.ssm_init, &fx.conv_init)
            .expect("write batched state");

        let batched_outputs = run_batched(
            &metal_device,
            &batched_pip,
            &batched_state,
            &batched_temp,
            &bp,
            &fx,
            &device,
        )
        .expect("run batched dispatch");

        // --- Bit-exact parity per slot ---
        assert_eq!(
            batched_outputs.len(),
            b,
            "batched returned {} rows vs B={}",
            batched_outputs.len(),
            b
        );
        let value_dim = sp.value_dim as usize;
        for s in 0..b {
            assert_eq!(
                batched_outputs[s].len(),
                value_dim,
                "slot {} output len mismatch",
                s
            );
            let mismatches: Vec<(usize, f32, f32)> = single_outputs[s]
                .iter()
                .zip(batched_outputs[s].iter())
                .enumerate()
                .filter(|(_, (a, bb))| a.to_bits() != bb.to_bits())
                .map(|(i, (a, bb))| (i, *a, *bb))
                .collect();
            if !mismatches.is_empty() {
                let max_diff = mismatches
                    .iter()
                    .map(|(_, a, bb)| (a - bb).abs())
                    .fold(0.0f32, f32::max);
                panic!(
                    "slot {} batched != single: {}/{} mismatches, max_abs_diff={:.3e}, first few: {:?}",
                    s,
                    mismatches.len(),
                    value_dim,
                    max_diff,
                    mismatches.iter().take(5).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn batched_vs_single_parity_b3() {
        assert_parity(3, 42);
    }

    #[test]
    fn batched_vs_single_parity_b4() {
        assert_parity(4, 1337);
    }

    /// B=1: sanity что добавление batch-оси не сломало single-stream path.
    #[test]
    fn batched_b1_matches_single() {
        assert_parity(1, 7);
    }
}
