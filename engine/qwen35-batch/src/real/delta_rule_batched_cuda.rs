// delta_rule_batched_cuda.rs — Rust bindings для batched DeltaNet decode CUDA kernels
//
// Phase 2: true batched decode с осью slot B (NVIDIA Win/Linux). Структурный
// аналог metal/delta_rule_batched_metal.rs. Сами ядра —
// candle-kernels/src/delta_rule_batched.cu (зарегистрированы как
// `candle_kernels::DELTA_RULE_BATCHED`, имена без mangling через extern "C").
//
// В отличие от delta_rule_cuda.rs (single-token, time-multiplex), один dispatch
// обрабатывает B одновременных слотов: state/scratch с leading B, grid добавляет
// B ось (blockIdx.y = slot). Чтение весов (conv/dt_bias/ssm_a/norm) — одно на
// B слотов (shared across slots), что даёт bandwidth выигрыш.
//
// ВСЁ на GPU — ноль host-sync на рекуррентном шаге. Decode в F32 (precision).
//
// ─────────────────────────────────────────────────────────────────────────────
// СБОРКА/ТЕСТ ТОЛЬКО на yttri-win (RTX 3060) с `--features cuda`. На macOS/Metal
// этот модуль cfg-выключен.
// ─────────────────────────────────────────────────────────────────────────────

use candle_core::{CudaDevice, CudaStorage, Device, Result, Storage, Tensor};
use cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};

/// Параметры DeltaNet слоя для batched kernel. Layout ДОЛЖЕН совпадать со
/// `struct DeltaParams` в delta_rule_batched.cu (#[repr(C)], 12 полей = 48 байтов).
/// Передаётся в ядро по значению через `builder.arg(&params)`.
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

// SAFETY: POD #[repr(C)] struct из u32/f32 — валидно как kernel-аргумент по значению.
unsafe impl cudarc::driver::DeviceRepr for DeltaParams {}

/// Persistent batched CUDA-буферы одного DeltaNet слоя. Ведущие B: state
/// изолирован per slot, живёт на GPU всё время жизни модели. Веса — shared
/// across slots (без B дублирования). Мутируются in-place device-side.
pub struct DeltaNetCudaStateBatched {
    /// SSM state: [B * n_v_heads * head_v_dim * head_v_dim] f32
    pub ssm_state: CudaSlice<f32>,
    /// Conv1d state: [B * (conv_k - 1) * channels] f32
    pub conv_state: CudaSlice<f32>,
    /// Conv1d веса: [channels * conv_k] — shared across slots
    pub conv_weights: CudaSlice<f32>,
    /// dt_bias: [n_v_heads] — shared across slots
    pub dt_bias: CudaSlice<f32>,
    /// ssm_a: [n_v_heads] — shared across slots (negative decay)
    pub ssm_a: CudaSlice<f32>,
    /// RMS norm веса: [head_v_dim] — shared across slots/heads
    pub norm_weight: CudaSlice<f32>,
    /// B — capacity (slot count), фиксированная при аллокации.
    pub capacity_b: u32,
}

/// Временные (scratch) batched буферы, переиспользуемые между слоями.
/// Все scratch с leading B: один dispatch покрывает B слотов.
pub struct DeltaNetCudaTempBatched {
    pub qkv_conv: CudaSlice<f32>,     // [B * channels]
    pub q: CudaSlice<f32>,             // [B * n_v_heads * head_k_dim]
    pub k: CudaSlice<f32>,             // [B * n_v_heads * head_k_dim]
    pub v: CudaSlice<f32>,             // [B * n_v_heads * head_v_dim]
    pub beta: CudaSlice<f32>,          // [B * n_v_heads]
    pub gate: CudaSlice<f32>,          // [B * n_v_heads]
    pub delta_output: CudaSlice<f32>,  // [B * n_v_heads * head_v_dim]
    pub gated_output: CudaSlice<f32>,  // [B * value_dim] — final gated output (reused)
    /// Кэш slot_ids на device (пересоздаётся только при смене состава батча).
    /// Раньше clone_htod делал H2D+alloc на КАЖДЫЙ блок на КАЖДЫЙ шаг —
    /// ~350µs × 30-48 блоков = доминирующий overhead decode-шага (A100 замер).
    pub slot_ids_cache: std::sync::Mutex<Option<(std::sync::Arc<CudaSlice<u32>>, Vec<u32>)>>,
    /// B — capacity (slot count), фиксированная при аллокации.
    pub capacity_b: u32,
}

/// Создаёт persistent batched буферы слоя. `capacity_b` — максимальное число
/// одновременных слотов; state аллоцируется под полный capacity, а активный
/// batch_size <= capacity просто использует часть. Веса из GGUF — htod один раз.
pub fn create_layer_cuda_state_batched(
    dev: &CudaDevice,
    params: &DeltaParams,
    capacity_b: u32,
    conv_weights_data: &[f32],
    dt_bias_data: &[f32],
    ssm_a_data: &[f32],
    norm_weight_data: &[f32],
) -> Result<DeltaNetCudaStateBatched> {
    let hd = params.head_v_dim as usize;
    let n_v = params.n_v_heads as usize;
    let channels = params.channels as usize;
    let conv_k = params.conv_kernel as usize;
    let b = capacity_b as usize;

    let ssm_state = dev.alloc_zeros::<f32>(b * n_v * hd * hd)?;
    let conv_state = dev.alloc_zeros::<f32>(b * (conv_k - 1) * channels)?;

    let conv_weights = dev.clone_htod(conv_weights_data)?;
    let dt_bias = dev.clone_htod(dt_bias_data)?;
    let ssm_a = dev.clone_htod(ssm_a_data)?;
    let norm_weight = dev.clone_htod(norm_weight_data)?;

    Ok(DeltaNetCudaStateBatched {
        ssm_state,
        conv_state,
        conv_weights,
        dt_bias,
        ssm_a,
        norm_weight,
        capacity_b,
    })
}

/// Создаёт batched scratch-буферы под capacity_b слотов (один комплект на модель,
/// переиспользуется слоями). F32 для precision (decode precision critical).
pub fn create_temp_buffers_batched(
    dev: &CudaDevice,
    params: &DeltaParams,
    capacity_b: u32,
) -> Result<DeltaNetCudaTempBatched> {
    let channels = params.channels as usize;
    let n_v = params.n_v_heads as usize;
    let hkd = params.head_k_dim as usize;
    let hvd = params.head_v_dim as usize;
    let value_dim = params.value_dim as usize;
    let b = capacity_b as usize;

    Ok(DeltaNetCudaTempBatched {
        qkv_conv: dev.alloc_zeros::<f32>(b * channels)?,
        q: dev.alloc_zeros::<f32>(b * n_v * hkd)?,
        k: dev.alloc_zeros::<f32>(b * n_v * hkd)?,
        v: dev.alloc_zeros::<f32>(b * n_v * hvd)?,
        beta: dev.alloc_zeros::<f32>(b * n_v)?,
        gate: dev.alloc_zeros::<f32>(b * n_v)?,
        delta_output: dev.alloc_zeros::<f32>(b * n_v * hvd)?,
        gated_output: dev.alloc_zeros::<f32>(b * value_dim)?,
        slot_ids_cache: std::sync::Mutex::new(None),
        capacity_b,
    })
}

/// Строит `CudaView<f32>` (с учётом offset) из storage-guard'а F32 CUDA-тензора.
/// Guard передаётся caller'ом и обязан жить дольше возвращённого view (borrow).
fn cuda_slice_view<'a>(storage: &'a Storage, offset: usize) -> Result<CudaView<'a, f32>> {
    match storage {
        Storage::Cuda(cs) => Ok(cs.as_cuda_slice::<f32>()?.slice(offset..)),
        _ => candle_core::bail!("delta_rule_batched_cuda: тензор не на CUDA device"),
    }
}

/// Выполняет полный batched delta_rule forward pass на CUDA GPU (4 ядра, ноль host-sync).
///
/// Входы — 4 batched результата QMatMul (уже на GPU, contiguous F32):
///   qkv_t  [B, 1, channels]
///   z_t    [B, 1, value_dim]
///   beta_t [B, 1, n_v_heads]
///   alpha_t[B, 1, n_v_heads]
/// Выход — [B, 1, value_dim] на GPU, готовый для batched ssm_out QMatMul.
///
/// `params.batch_size` — активное число слотов (<= state/temp capacity_b).
pub fn dispatch_delta_rule_batched(
    dev: &CudaDevice,
    state: &DeltaNetCudaStateBatched,
    temp: &DeltaNetCudaTempBatched,
    params: &DeltaParams,
    qkv_t: &Tensor,
    z_t: &Tensor,
    beta_t: &Tensor,
    alpha_t: &Tensor,
    // Indirection: batch_idx → реальный slot_idx (persistent state индексация).
    slot_ids: &[u32],
) -> Result<Tensor> {
    let channels = params.channels as u32;
    let n_v = params.n_v_heads as u32;
    let hkd = params.head_k_dim as u32;
    let hvd = params.head_v_dim as u32;
    let value_dim = params.value_dim as usize;
    let b = params.batch_size as usize;
    let b_u32 = params.batch_size;
    let block_ch = 256u32;

    // Sanity: активный batch не превышает capacity аллоцированных буферов.
    debug_assert!(
        b_u32 <= state.capacity_b,
        "batch_size {} > state capacity {}",
        b_u32,
        state.capacity_b
    );
    debug_assert!(
        b_u32 <= temp.capacity_b,
        "batch_size {} > temp capacity {}",
        b_u32,
        temp.capacity_b
    );

    // Decode path: F32 contiguous (precision). QMatMul выдаёт F32.
    let qkv_f = ensure_f32(qkv_t)?;
    let z_f = ensure_f32(z_t)?;
    let beta_f = ensure_f32(beta_t)?;
    let alpha_f = ensure_f32(alpha_t)?;

    // Проверка формы входов (leading B).
    let (b_qkv, _s, _c) = qkv_f.dims3()?;
    let (b_z, _s, _v) = z_f.dims3()?;
    let (b_beta, _s, _h) = beta_f.dims3()?;
    let (b_alpha, _s, _h) = alpha_f.dims3()?;
    if b_qkv != b || b_z != b || b_beta != b || b_alpha != b {
        candle_core::bail!(
            "dispatch_delta_rule_batched: input batch mismatch: qkv={} z={} beta={} alpha={} vs batch_size={}",
            b_qkv, b_z, b_beta, b_alpha, b
        );
    }

    // Storage-guard'ы держим живыми до конца всех launch'ей (view'ы их заимствуют).
    let (qkv_st, qkv_lay) = qkv_f.storage_and_layout();
    let (z_st, z_lay) = z_f.storage_and_layout();
    let (beta_st, beta_lay) = beta_f.storage_and_layout();
    let (alpha_st, alpha_lay) = alpha_f.storage_and_layout();
    let qkv_v = cuda_slice_view(&qkv_st, qkv_lay.start_offset())?;
    let z_v = cuda_slice_view(&z_st, z_lay.start_offset())?;
    let beta_v = cuda_slice_view(&beta_st, beta_lay.start_offset())?;
    let alpha_v = cuda_slice_view(&alpha_st, alpha_lay.start_offset())?;

    // Indirection buffer: batch_idx → real slot_idx. Persistent state (ssm_state,
    // conv_state) индексируется через slot_ids[bidx], temp/входы — по batch_idx.
    // Без этого при batch shrink оставшийся slot попал бы в чужой state-region.
    assert_eq!{
        slot_ids.len(),
        b,
        "dispatch_delta_rule_batched: slot_ids.len {} != batch_size {}",
        slot_ids.len(),
        b,
    };
    let slot_ids_arc = {
        let mut cache = temp.slot_ids_cache.lock().expect("slot_ids cache");
        let stale = match &*cache {
            Some((_, ids)) => ids.as_slice() != slot_ids,
            None => true,
        };
        if stale {
            *cache = Some((std::sync::Arc::new(dev.clone_htod(slot_ids)?), slot_ids.to_vec()));
        }
        cache.as_ref().unwrap().0.clone()
    };

    let p = *params;

    // ── Kernel 1: delta_conv1d_prep_batched ──
    // grid=(ceil(channels/256), B, 1), block=(256,1,1)
    {
        let func = dev.get_or_load_func(
            "delta_conv1d_prep_batched",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(block_ch), b_u32, 1),
            block_dim: (block_ch, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&qkv_v); // qkv_raw
        bb.arg(&beta_v); // beta_raw
        bb.arg(&alpha_v); // alpha_raw
        bb.arg(&state.conv_weights);
        bb.arg(&state.dt_bias);
        bb.arg(&state.ssm_a);
        bb.arg(&state.conv_state); // read-write persistent
        bb.arg(&temp.qkv_conv); // out
        bb.arg(&temp.beta); // out
        bb.arg(&temp.gate); // out
        bb.arg(&p);
        bb.arg(&*slot_ids_arc); // slot_ids (indirection)
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 2: delta_l2_norm_expand_batched ──
    // grid=(n_v, B, 1), block=(head_k_dim, 1, 1)
    {
        let func = dev.get_or_load_func(
            "delta_l2_norm_expand_batched",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, b_u32, 1),
            block_dim: (hkd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&temp.qkv_conv);
        bb.arg(&temp.q);
        bb.arg(&temp.k);
        bb.arg(&temp.v);
        bb.arg(&p);
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 3: delta_rule_kernel_batched ──
    // grid=(n_v, B, 1), block=(head_v_dim, 1, 1)
    {
        let func = dev.get_or_load_func(
            "delta_rule_kernel_batched",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, b_u32, 1),
            block_dim: (hvd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&temp.q);
        bb.arg(&temp.k);
        bb.arg(&temp.v);
        bb.arg(&temp.beta);
        bb.arg(&temp.gate);
        bb.arg(&state.ssm_state); // read-write persistent
        bb.arg(&temp.delta_output); // out
        bb.arg(&p);
        bb.arg(&*slot_ids_arc); // slot_ids (indirection)
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 4: delta_norm_gate_kernel_batched ──
    // grid=(n_v, B, 1), block=(head_v_dim,1,1). Пишет в temp.gated_output (reused).
    {
        let func = dev.get_or_load_func(
            "delta_norm_gate_kernel_batched",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, b_u32, 1),
            block_dim: (hvd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&temp.delta_output);
        bb.arg(&z_v);
        bb.arg(&state.norm_weight);
        bb.arg(&temp.gated_output); // out (reused scratch)
        bb.arg(&p);
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // Zero-copy: клонируем CudaSlice (cheap Arc), оборачиваем в Tensor [B,1,value_dim].
    // temp.gated_output остаётся владельцем underlying alloc; tensor держит clone.
    let gated_view = temp.gated_output.clone();
    let storage = CudaStorage::wrap_cuda_slice(gated_view, dev.clone());
    let output = Tensor::from_storage(
        Storage::Cuda(storage),
        (b, 1, value_dim),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(output)
}

/// Seq-вариант для MTP batched verify: K строк ОДНОГО слота одним набором из
/// 4 launch'ей (как обычный batched-шаг). Рекуррентные ядра (conv, delta) крутят
/// внутренний цикл по строкам — бит-эксактно K одиночным шагам; стейтлесс-ядра
/// (l2norm, gate) получают grid.y=K. Требует temp capacity_b >= K.
///
/// Входы: qkv_t [K,1,channels], z_t [K,1,value_dim], beta/alpha [K,1,n_v].
/// Выход: [K,1,value_dim] (zero-copy wrap temp.gated_output — потребить до
/// следующего DeltaNet-диспатча).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_delta_rule_batched_seq(
    dev: &CudaDevice,
    state: &DeltaNetCudaStateBatched,
    temp: &DeltaNetCudaTempBatched,
    params: &DeltaParams,
    qkv_t: &Tensor,
    z_t: &Tensor,
    beta_t: &Tensor,
    alpha_t: &Tensor,
    slot: u32,
) -> Result<Tensor> {
    let channels = params.channels as u32;
    let n_v = params.n_v_heads as u32;
    let hkd = params.head_k_dim as u32;
    let hvd = params.head_v_dim as u32;
    let value_dim = params.value_dim as usize;
    let block_ch = 256u32;

    let qkv_f = ensure_f32(qkv_t)?;
    let z_f = ensure_f32(z_t)?;
    let beta_f = ensure_f32(beta_t)?;
    let alpha_f = ensure_f32(alpha_t)?;

    let (k_rows, _s, _c) = qkv_f.dims3()?;
    let (k_z, _s, _v) = z_f.dims3()?;
    let (k_beta, _s, _h) = beta_f.dims3()?;
    let (k_alpha, _s, _h) = alpha_f.dims3()?;
    if k_z != k_rows || k_beta != k_rows || k_alpha != k_rows || k_rows == 0 {
        candle_core::bail!(
            "dispatch_delta_rule_batched_seq: row mismatch: qkv={k_rows} z={k_z} beta={k_beta} alpha={k_alpha}"
        );
    }
    if k_rows as u32 > temp.capacity_b {
        candle_core::bail!(
            "dispatch_delta_rule_batched_seq: {k_rows} rows > temp capacity {}",
            temp.capacity_b
        );
    }
    if slot >= state.capacity_b {
        candle_core::bail!(
            "dispatch_delta_rule_batched_seq: slot {slot} >= state capacity {}",
            state.capacity_b
        );
    }
    let seq_rows = k_rows as u32;

    let (qkv_st, qkv_lay) = qkv_f.storage_and_layout();
    let (z_st, z_lay) = z_f.storage_and_layout();
    let (beta_st, beta_lay) = beta_f.storage_and_layout();
    let (alpha_st, alpha_lay) = alpha_f.storage_and_layout();
    let qkv_v = cuda_slice_view(&qkv_st, qkv_lay.start_offset())?;
    let z_v = cuda_slice_view(&z_st, z_lay.start_offset())?;
    let beta_v = cuda_slice_view(&beta_st, beta_lay.start_offset())?;
    let alpha_v = cuda_slice_view(&alpha_st, alpha_lay.start_offset())?;

    let slot_ids = [slot];
    let slot_ids_arc = {
        let mut cache = temp.slot_ids_cache.lock().expect("slot_ids cache");
        let stale = match &*cache {
            Some((_, ids)) => ids.as_slice() != slot_ids,
            None => true,
        };
        if stale {
            *cache = Some((std::sync::Arc::new(dev.clone_htod(&slot_ids)?), slot_ids.to_vec()));
        }
        cache.as_ref().unwrap().0.clone()
    };

    let mut p = *params;
    p.batch_size = seq_rows;

    // ── Kernel 1 seq: conv1d prep, внутренний цикл по K строкам ──
    {
        let func = dev.get_or_load_func(
            "delta_conv1d_prep_batched_seq",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (channels.div_ceil(block_ch), 1, 1),
            block_dim: (block_ch, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&qkv_v);
        bb.arg(&beta_v);
        bb.arg(&alpha_v);
        bb.arg(&state.conv_weights);
        bb.arg(&state.dt_bias);
        bb.arg(&state.ssm_a);
        bb.arg(&state.conv_state);
        bb.arg(&temp.qkv_conv);
        bb.arg(&temp.beta);
        bb.arg(&temp.gate);
        bb.arg(&p);
        bb.arg(&*slot_ids_arc);
        bb.arg(&seq_rows);
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 2: l2 norm (стейтлесс — batch-ось = строки) ──
    {
        let func = dev.get_or_load_func(
            "delta_l2_norm_expand_batched",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, seq_rows, 1),
            block_dim: (hkd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&temp.qkv_conv);
        bb.arg(&temp.q);
        bb.arg(&temp.k);
        bb.arg(&temp.v);
        bb.arg(&p);
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 3 seq: delta rule, внутренний цикл по K строкам ──
    // Smem-вариант держит state головы [hd×hd] в динамическом shared memory:
    // global-трафик state = 2 прохода на форвард вместо 6×K. 128×128 → 64КБ
    // динамического smem — больше дефолтных 48КБ, нужен opt-in атрибут
    // (ставится один раз; при отказе — фолбэк на global-вариант).
    {
        let smem_bytes = (params.head_v_dim * params.head_v_dim * 4) as u32;
        static SMEM_SEQ_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_smem = *SMEM_SEQ_OK.get_or_init(|| {
            let ok = match dev.get_or_load_func(
                "delta_rule_kernel_batched_seq_smem",
                &candle_kernels::DELTA_RULE_BATCHED,
            ) {
                Ok(func) if smem_bytes <= 48 * 1024 => {
                    let _ = func;
                    true
                }
                Ok(func) => func
                    .set_attribute(
                        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem_bytes as i32,
                    )
                    .is_ok(),
                Err(_) => false,
            };
            if !ok {
                log::warn!(
                    "[delta-seq] smem-вариант недоступен ({} bytes) — global fallback",
                    smem_bytes
                );
            }
            ok
        });
        let (kernel_name, shared_mem) = if use_smem {
            ("delta_rule_kernel_batched_seq_smem", smem_bytes)
        } else {
            ("delta_rule_kernel_batched_seq", 0)
        };
        let func = dev.get_or_load_func(kernel_name, &candle_kernels::DELTA_RULE_BATCHED)?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, 1, 1),
            block_dim: (hvd, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        let mut bb = func.builder();
        bb.arg(&temp.q);
        bb.arg(&temp.k);
        bb.arg(&temp.v);
        bb.arg(&temp.beta);
        bb.arg(&temp.gate);
        bb.arg(&state.ssm_state);
        bb.arg(&temp.delta_output);
        bb.arg(&p);
        bb.arg(&*slot_ids_arc);
        bb.arg(&seq_rows);
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    // ── Kernel 4: norm+gate (стейтлесс — batch-ось = строки) ──
    {
        let func = dev.get_or_load_func(
            "delta_norm_gate_kernel_batched",
            &candle_kernels::DELTA_RULE_BATCHED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_v, seq_rows, 1),
            block_dim: (hvd, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bb = func.builder();
        bb.arg(&temp.delta_output);
        bb.arg(&z_v);
        bb.arg(&state.norm_weight);
        bb.arg(&temp.gated_output);
        bb.arg(&p);
        unsafe { bb.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    let gated_view = temp.gated_output.clone();
    let storage = CudaStorage::wrap_cuda_slice(gated_view, dev.clone());
    let output = Tensor::from_storage(
        Storage::Cuda(storage),
        (k_rows, 1, value_dim),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(output)
}

/// Cast в F32 contiguous (как single CUDA-путь).
fn ensure_f32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == candle_core::DType::F32 {
        if t.is_contiguous() {
            Ok(t.clone())
        } else {
            t.contiguous()
        }
    } else {
        t.to_dtype(candle_core::DType::F32)?.contiguous()
    }
}

/// Записывает batched state: B копий per-slot init-данных [slot, ...].
/// Полная перезапись ssm_state/conv_state (htod) для parity-тестов / restore.
pub fn write_batched_state(
    dev: &CudaDevice,
    state: &mut DeltaNetCudaStateBatched,
    ssm_init: &[f32], // [n_v*hd*hd] — per-slot init (копируется B раз)
    conv_init: &[f32], // [(conv_k-1)*channels] — per-slot init (копируется B раз)
) -> Result<()> {
    let b = state.capacity_b as usize;
    let ssm_len = ssm_init.len();
    let conv_len = conv_init.len();

    debug_assert!(
        state.ssm_state.len() >= b * ssm_len,
        "ssm state cap {} < B*per_slot {}",
        state.ssm_state.len(),
        b * ssm_len
    );
    debug_assert!(
        state.conv_state.len() >= b * conv_len,
        "conv state cap {} < B*per_slot {}",
        state.conv_state.len(),
        b * conv_len
    );

    // htod full capacity: interleave B copies of per-slot init.
    let mut ssm_full = Vec::with_capacity(b * ssm_len);
    for _ in 0..b {
        ssm_full.extend_from_slice(ssm_init);
    }
    let mut conv_full = Vec::with_capacity(b * conv_len);
    for _ in 0..b {
        conv_full.extend_from_slice(conv_init);
    }
    dev.memcpy_htod(&ssm_full, &mut state.ssm_state)?;
    dev.memcpy_htod(&conv_full, &mut state.conv_state)?;
    Ok(())
}

/// Snapshot batched state: dtoh-копия всей capacity (для parity-тестов / debug).
/// Возвращает (ssm [B*...], conv [B*...]).
pub fn snapshot_batched_state(
    dev: &CudaDevice,
    state: &DeltaNetCudaStateBatched,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let ssm = dev.clone_dtoh(&state.ssm_state)?;
    let conv = dev.clone_dtoh(&state.conv_state)?;
    Ok((ssm, conv))
}

/// Сброс state (новая беседа): зануляем оба persistent-буфера через htod нулей.
pub fn clear_cuda_state_batched(
    dev: &CudaDevice,
    state: &mut DeltaNetCudaStateBatched,
) -> Result<()> {
    let ssm_len = state.ssm_state.len();
    let conv_len = state.conv_state.len();
    dev.memcpy_htod(&vec![0.0f32; ssm_len], &mut state.ssm_state)?;
    dev.memcpy_htod(&vec![0.0f32; conv_len], &mut state.conv_state)?;
    Ok(())
}

/// Записывает single-slot state в конкретный slot batched-буфера (htod).
/// Phase 4 adapter: seed per-slot state из prefill snapshot в batched decode
/// buffers перед первым batched decode-шагом.
///
/// `ssm_init` — [n_v_heads * head_v_dim * head_v_dim] (один slot, без B).
/// `conv_init` — [(conv_kernel - 1) * channels] (один slot, без B).
/// `slot` — индекс слота [0, capacity_b).
///
/// Layout batched-буфера: leading B → slot-регион по смещению `slot * per_slot_len`.
/// Записываем только per-slot данные (остальные слоты не трогаем).
pub struct DeltaNetCudaSlotCheckpoint {
    ssm_state: CudaSlice<f32>,
    conv_state: CudaSlice<f32>,
}

/// D2D snapshot of one physical slot region. No host transfer.
pub fn checkpoint_slot_cuda_state(
    dev: &CudaDevice,
    state: &DeltaNetCudaStateBatched,
    slot: usize,
) -> Result<DeltaNetCudaSlotCheckpoint> {
    let capacity = state.capacity_b as usize;
    if slot >= capacity {
        candle_core::bail!("checkpoint slot {slot} >= capacity {capacity}");
    }
    let ssm_len = state.ssm_state.len() / capacity;
    let conv_len = state.conv_state.len() / capacity;
    let ssm_state = state.ssm_state.slice(slot * ssm_len..(slot + 1) * ssm_len);
    let conv_state = state.conv_state.slice(slot * conv_len..(slot + 1) * conv_len);
    let mut ssm_copy = dev.alloc_zeros::<f32>(ssm_len)?;
    let mut conv_copy = dev.alloc_zeros::<f32>(conv_len)?;
    dev.memcpy_dtod(&ssm_state, &mut ssm_copy)?;
    dev.memcpy_dtod(&conv_state, &mut conv_copy)?;
    Ok(DeltaNetCudaSlotCheckpoint {
        ssm_state: ssm_copy,
        conv_state: conv_copy,
    })
}

pub fn restore_slot_cuda_checkpoint(
    dev: &CudaDevice,
    state: &mut DeltaNetCudaStateBatched,
    slot: usize,
    checkpoint: &DeltaNetCudaSlotCheckpoint,
) -> Result<()> {
    let capacity = state.capacity_b as usize;
    if slot >= capacity {
        candle_core::bail!("restore slot {slot} >= capacity {capacity}");
    }
    let ssm_len = state.ssm_state.len() / capacity;
    let conv_len = state.conv_state.len() / capacity;
    if checkpoint.ssm_state.len() != ssm_len || checkpoint.conv_state.len() != conv_len {
        candle_core::bail!("CUDA slot checkpoint shape mismatch");
    }
    let mut ssm_dst = state.ssm_state.slice_mut(slot * ssm_len..(slot + 1) * ssm_len);
    let mut conv_dst = state.conv_state.slice_mut(slot * conv_len..(slot + 1) * conv_len);
    dev.memcpy_dtod(&checkpoint.ssm_state, &mut ssm_dst)?;
    dev.memcpy_dtod(&checkpoint.conv_state, &mut conv_dst)?;
    Ok(())
}

pub fn restore_slot_cuda_state(
    dev: &CudaDevice,
    state: &mut DeltaNetCudaStateBatched,
    slot: usize,
    ssm_init: &[f32],
    conv_init: &[f32],
) -> Result<()> {
    let b = state.capacity_b as usize;
    debug_assert!(slot < b, "restore_slot: slot {slot} >= capacity_b {b}");
    let ssm_len = ssm_init.len();
    let conv_len = conv_init.len();
    let ssm_off = slot * ssm_len;
    let conv_off = slot * conv_len;

    debug_assert!(
        ssm_off + ssm_len <= state.ssm_state.len(),
        "restore_slot ssm: off {} len {} > cap {}",
        ssm_off,
        ssm_len,
        state.ssm_state.len()
    );
    debug_assert!(
        conv_off + conv_len <= state.conv_state.len(),
        "restore_slot conv: off {} len {} > cap {}",
        conv_off,
        conv_len,
        state.conv_state.len()
    );

    {
        let mut ssm_view = state.ssm_state.slice_mut(ssm_off..ssm_off + ssm_len);
        dev.memcpy_htod(ssm_init, &mut ssm_view)?;
    }
    {
        let mut conv_view = state.conv_state.slice_mut(conv_off..conv_off + conv_len);
        dev.memcpy_htod(conv_init, &mut conv_view)?;
    }
    Ok(())
}

/// Достаёт `&CudaDevice` из CUDA-тензора.
pub fn cuda_device_of(t: &Tensor) -> Result<CudaDevice> {
    match t.device() {
        Device::Cuda(d) => Ok(d.clone()),
        _ => candle_core::bail!("delta_rule_batched_cuda: ожидался CUDA device"),
    }
}

// ═══════════════════════════════════════════════════════════════
// Phase 2 unit-тесты: batched kernel parity vs single-token kernel (CUDA).
//
// Доказательство корректности оси slot на NVIDIA: один batched dispatch [B,1,...]
// даёт per-slot output, бит-точно равный B независимым single-token dispatch'ам
// с теми же per-slot входами и идентичным начальным state. Math per (head,slot)
// идентична per-head math из delta_rule.cu.
//
// Запуск (yttri-win, RTX 3060, --features cuda):
//   cargo test -p qwen35-batch --features "real-model cuda" \
//       --lib delta_rule_batched_cuda::tests -- --nocapture --test-threads=1
// ═══════════════════════════════════════════════════════════════

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::real::delta_rule_cuda::{
        self, DeltaNetCudaState as SingleState, DeltaNetCudaTemp as SingleTemp,
    };

    /// Детерминированный LCG (без внешних rng deps) → f32 в [-1, 1).
    struct Lcg {
        s: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { s: seed.wrapping_add(0x9E3779B97F4A7C15) }
        }
        fn next_u32(&mut self) -> u32 {
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
    fn small_params_single() -> delta_rule_cuda::DeltaParams {
        delta_rule_cuda::DeltaParams {
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

    struct Fixture {
        conv_weights: Vec<f32>,
        dt_bias: Vec<f32>,
        ssm_a: Vec<f32>,
        norm_weight: Vec<f32>,
        qkv: Vec<Vec<f32>>,
        z: Vec<Vec<f32>>,
        beta: Vec<Vec<f32>>,
        alpha: Vec<Vec<f32>>,
        ssm_init: Vec<f32>,
        conv_init: Vec<f32>,
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
        let dt_bias = (0..n_v).map(|_| rng.next_f32() * 0.2).collect::<Vec<_>>();
        let ssm_a = (0..n_v).map(|_| -(0.05 + rng.next_f32().abs() * 0.8)).collect::<Vec<_>>();
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

    fn cuda_device() -> Option<CudaDevice> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d),
            Ok(_) => unreachable!(),
            Err(e) => {
                eprintln!("SKIP batched CUDA parity: CUDA device unavailable: {e}");
                None
            }
        }
    }

    fn run_single(
        dev: &CudaDevice,
        single_state: &mut SingleState,
        single_temp: &SingleTemp,
        params: &delta_rule_cuda::DeltaParams,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
        device: &Device,
    ) -> candle_core::Result<Vec<f32>> {
        let qkv_t = Tensor::from_vec(qkv.to_vec(), (1, 1, qkv.len()), device)?;
        let z_t = Tensor::from_vec(z.to_vec(), (1, 1, z.len()), device)?;
        let beta_t = Tensor::from_vec(beta.to_vec(), (1, 1, beta.len()), device)?;
        let alpha_t = Tensor::from_vec(alpha.to_vec(), (1, 1, alpha.len()), device)?;
        let out = delta_rule_cuda::dispatch_delta_rule(
            dev, single_state, single_temp, params, &qkv_t, &z_t, &beta_t, &alpha_t,
        )?;
        out.flatten_all()?.to_vec1::<f32>()
    }

    fn run_batched(
        dev: &CudaDevice,
        batched_state: &DeltaNetCudaStateBatched,
        batched_temp: &DeltaNetCudaTempBatched,
        params: &DeltaParams,
        fx: &Fixture,
        device: &Device,
    ) -> candle_core::Result<Vec<Vec<f32>>> {
        let b = params.batch_size as usize;
        let channels = params.channels as usize;
        let value_dim = params.value_dim as usize;
        let n_v = params.n_v_heads as usize;

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
            dev, batched_state, batched_temp, params, &qkv_t, &z_t, &beta_t, &alpha_t,
            &(0..b).map(|i| i as u32).collect::<Vec<_>>(),
        )?;
        let flat = out.flatten_all()?.to_vec1::<f32>()?;
        Ok(flat.chunks(value_dim).map(|c| c.to_vec()).collect())
    }

    fn assert_parity(b: usize, seed: u64) {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let device = Device::Cuda(dev.clone());

        let sp = small_params_single();
        let bp = small_params_batched(b as u32);
        let fx = make_fixture(b, seed);

        // --- Single path: B независимых dispatch'ей, каждый из идентичного init state ---
        let mut single_state = delta_rule_cuda::create_layer_cuda_state(
            &dev,
            &sp,
            &fx.conv_weights,
            &fx.dt_bias,
            &fx.ssm_a,
            &fx.norm_weight,
        )
        .expect("create single state");
        let single_temp = delta_rule_cuda::create_temp_buffers(&dev, &sp).expect("create single temp");

        let mut single_outputs = Vec::with_capacity(b);
        for s in 0..b {
            // Restore одинакового init state per slot (полная перезапись → изоляция).
            delta_rule_cuda::restore_cuda_state(&dev, &mut single_state, &fx.ssm_init, &fx.conv_init)
                .expect("restore single state");
            let out = run_single(
                &dev, &mut single_state, &single_temp, &sp, &fx.qkv[s], &fx.z[s], &fx.beta[s],
                &fx.alpha[s], &device,
            )
            .expect("run single dispatch");
            single_outputs.push(out);
        }

        // --- Batched path: один dispatch [B,1,...] из того же init state ---
        let mut batched_state = create_layer_cuda_state_batched(
            &dev,
            &bp,
            b as u32,
            &fx.conv_weights,
            &fx.dt_bias,
            &fx.ssm_a,
            &fx.norm_weight,
        )
        .expect("create batched state");
        let batched_temp = create_temp_buffers_batched(&dev, &bp, b as u32)
            .expect("create batched temp");
        write_batched_state(&dev, &mut batched_state, &fx.ssm_init, &fx.conv_init)
            .expect("write batched state");

        let batched_outputs = run_batched(
            &dev, &batched_state, &batched_temp, &bp, &fx, &device,
        )
        .expect("run batched dispatch");

        // --- Bit-exact parity per slot ---
        assert_eq!(batched_outputs.len(), b, "batched rows {} vs B={}", batched_outputs.len(), b);
        let value_dim = sp.value_dim as usize;
        for s in 0..b {
            assert_eq!(batched_outputs[s].len(), value_dim, "slot {} output len mismatch", s);
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
                    s, mismatches.len(), value_dim, max_diff, mismatches.iter().take(5).collect::<Vec<_>>()
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

    /// B=1: sanity что batch-ось не сломала single-stream path.
    #[test]
    fn batched_b1_matches_single() {
        assert_parity(1, 7);
    }
}
