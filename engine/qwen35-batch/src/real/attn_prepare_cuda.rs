//! Fused attention-prep для paged-декода (CUDA).
//!
//! Классический путь подготовки внимания делает ~15 мелких запусков на слой:
//! две `narrow().contiguous()` для Q и gate, две пары `transpose().contiguous()`,
//! q/k RMSNorm, два partial-RoPE (каждый — gather cos/sin, выборка ротируемой
//! части, 4 `broadcast_mul`, sub, add, cat) и каст K/V/Q в F16. На 8
//! attention-слоях это ~120 запусков на шаг, и по замеру 2026-09-16 эти слои
//! дают 207 ГБ/с против 295 ГБ/с у DeltaNet-слоёв при меньшем объёме весов —
//! то есть шаг держит не пропускная способность, а оверхед мелких ядер.
//!
//! Здесь всё это делает одно ядро `attn_prepare_decode` (см. flash_decode.cu),
//! а результат пишется в переиспользуемые буферы: адреса стабильны, поэтому
//! путь совместим с CUDA-графами и не аллоцирует на каждом шаге.
//!
//! Включается `YTTRI_ATTN_PREP_FUSED=1`; по умолчанию выключено, чтобы A/B на
//! стенде шёл против эталонной цепочки на одном и том же бинарнике.

use candle_core::{DType, Device, Result, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::real::paged_kv_cuda::tensor_cuda_ptr;

/// Слитое ядро включено по умолчанию; `YTTRI_ATTN_PREP_FUSED=0` выключает.
///
/// Замер Ornith-1.5-35B-A3B Q8_0, ctx 32768, декод: 134.3 и 137.1 t/s против
/// 128.1 и 128.0 на эталонной цепочке — то есть +5.5% и разрыв с llama.cpp
/// сокращается с 7.8% до 3.4%. Перекрёстный A/B (ON-OFF-OFF-ON) воспроизвёлся.
pub fn enabled() -> bool {
    // Кэш обязателен: вызов идёт из `forward_decode_batch_paged`, то есть
    // по разу на каждый attention-слой на токен (10 на Ornith-35B).
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("YTTRI_ATTN_PREP_FUSED")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Переиспользуемые буферы слитого прохода.
///
/// Раскладки совпадают с тем, что ждут потребители: `q` — [cap, n_head, hd] F16
/// (вход FA2), `gate` — [cap, n_head, hd] F32 (raw gate, sigmoid и умножение
/// делает хвост `forward_attn_decode_paged`), `k`/`v` — [cap, n_kv, hd] F16
/// (строки для `PagedKvPool::launch_append`).
#[derive(Debug, Clone)]
pub struct AttnPrepScratch {
    pub cap: usize,
    pub q: Tensor,
    pub gate: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

impl AttnPrepScratch {
    pub fn new(
        cap: usize,
        n_head: usize,
        n_kv: usize,
        hd: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            cap,
            q: Tensor::zeros((cap, n_head, hd), DType::F16, device)?,
            gate: Tensor::zeros((cap, n_head, hd), DType::F32, device)?,
            k: Tensor::zeros((cap, n_kv, hd), DType::F16, device)?,
            v: Tensor::zeros((cap, n_kv, hd), DType::F16, device)?,
        })
    }
}

/// Один запуск `attn_prepare_decode`: Q RMSNorm + partial RoPE + каст, K то же
/// самое, V — каст. Шейпы и математика повторяют эталонную цепочку из
/// `GatedAttentionLayer::forward_attn_decode_paged`.
///
/// Возвращает срезы `scratch` по фактический batch: (q, gate, k, v).
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    scratch: &AttnPrepScratch,
    qg: &Tensor,   // [b, n_head, 2*hd] F32
    k: &Tensor,    // [b, n_kv, hd] F32
    v: &Tensor,    // [b, n_kv, hd] F32
    qw: &Tensor,   // [hd] F32
    kw: &Tensor,   // [hd] F32
    cos: &Tensor,  // [max_pos, rope_dim/2] F32
    sin: &Tensor,  // [max_pos, rope_dim/2] F32
    pos: &Tensor,  // [b] U32
    b: usize,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    rope_dim: usize,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    if b > scratch.cap {
        candle_core::bail!(
            "attn_prepare: batch {b} превышает ёмкость буферов {}",
            scratch.cap
        );
    }
    let dev = qg.device().as_cuda_device()?;
    let q_out = scratch.q.narrow(0, 0, b)?;
    let gate_out = scratch.gate.narrow(0, 0, b)?;
    let k_out = scratch.k.narrow(0, 0, b)?;
    let v_out = scratch.v.narrow(0, 0, b)?;

    let qg_ptr = tensor_cuda_ptr(qg)?;
    let k_ptr = tensor_cuda_ptr(k)?;
    let v_ptr = tensor_cuda_ptr(v)?;
    let qw_ptr = tensor_cuda_ptr(qw)?;
    let kw_ptr = tensor_cuda_ptr(kw)?;
    let cos_ptr = tensor_cuda_ptr(cos)?;
    let sin_ptr = tensor_cuda_ptr(sin)?;
    let pos_ptr = tensor_cuda_ptr(pos)?;
    let q_out_ptr = tensor_cuda_ptr(&q_out)?;
    let gate_out_ptr = tensor_cuda_ptr(&gate_out)?;
    let k_out_ptr = tensor_cuda_ptr(&k_out)?;
    let v_out_ptr = tensor_cuda_ptr(&v_out)?;

    let n_head_i = n_head as i32;
    let n_kv_i = n_kv as i32;
    let hd_i = hd as i32;
    let rope_dim_i = rope_dim as i32;
    let rope_half_i = (rope_dim / 2) as i32;

    let func = dev.get_or_load_func("attn_prepare_decode", &candle_kernels::FLASH_DECODE)?;
    let cfg = LaunchConfig {
        grid_dim: ((n_head + 2 * n_kv) as u32, b as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&qg_ptr);
    builder.arg(&k_ptr);
    builder.arg(&v_ptr);
    builder.arg(&qw_ptr);
    builder.arg(&kw_ptr);
    builder.arg(&cos_ptr);
    builder.arg(&sin_ptr);
    builder.arg(&pos_ptr);
    builder.arg(&q_out_ptr);
    builder.arg(&gate_out_ptr);
    builder.arg(&k_out_ptr);
    builder.arg(&v_out_ptr);
    builder.arg(&n_head_i);
    builder.arg(&n_kv_i);
    builder.arg(&hd_i);
    builder.arg(&rope_dim_i);
    builder.arg(&rope_half_i);
    builder.arg(&eps);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    Ok((q_out, gate_out, k_out, v_out))
}
