//! Diagnostic split-K flash-decoding dispatch (CUDA). Long-KV parity still
//! differs from FA2, so production requires explicit `ENABLE_SPLITK_DECODE=1`.

use candle_core::{CudaDevice, Result, Storage, Tensor};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashDecodeParams {
    pub n_head: u32,
    pub n_kv: u32,
    pub hd: u32,
    pub kv_len: u32,
    pub splits: u32,
    pub scale: f32,
}

// SAFETY: POD repr(C) из u32/f32.
unsafe impl cudarc::driver::DeviceRepr for FlashDecodeParams {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PagedFlashDecodeParams {
    pub n_head: u32,
    pub n_kv: u32,
    pub hd: u32,
    pub splits: u32,
    pub scale: f32,
    pub page_size: u32,
    pub block_table_stride: u32,
}

// SAFETY: POD repr(C) из u32/f32.
unsafe impl cudarc::driver::DeviceRepr for PagedFlashDecodeParams {}

/// q: [1, n_head, 1, hd] (любой float dtype); k/v: [1, kv_len, n_kv, hd] F16
/// head-last contiguous. Возвращает [1, n_head, 1, hd] F16.
pub fn dispatch_flash_decode(
    dev: &CudaDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use candle_core::DType;
    let n_head = q.dim(1)?;
    let kv_len = k.dim(1)?;
    let n_kv = k.dim(2)?;
    let hd = k.dim(3)?;

    let q16 = q.to_dtype(DType::F16)?.reshape((n_head, hd))?.contiguous()?;
    let k16 = k.to_dtype(DType::F16)?.contiguous()?;
    let v16 = v.to_dtype(DType::F16)?.contiguous()?;

    let (q_st, q_lay) = q16.storage_and_layout();
    let (k_st, k_lay) = k16.storage_and_layout();
    let (v_st, v_lay) = v16.storage_and_layout();
    let q_v = view_f16(&q_st, q_lay.start_offset())?;
    let k_v = view_f16(&k_st, k_lay.start_offset())?;
    let v_v = view_f16(&v_st, v_lay.start_offset())?;

    // S: ~1024 токена на чанк, в рамках [8, 64].
    let splits = (kv_len as u32).div_ceil(1024).clamp(8, 64);
    let params = FlashDecodeParams {
        n_head: n_head as u32,
        n_kv: n_kv as u32,
        hd: hd as u32,
        kv_len: kv_len as u32,
        splits,
        scale,
    };

    let mut partials = unsafe {
        dev.alloc::<f32>(n_head * splits as usize * 4 * (2 + hd))?
    };
    let out = unsafe { dev.alloc::<half::f16>(n_head * hd)? };

    {
        let func = dev.get_or_load_func("flash_decode_partial", &candle_kernels::FLASH_DECODE)?;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, splits, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&q_v);
        b.arg(&k_v);
        b.arg(&v_v);
        b.arg(&mut partials);
        b.arg(&params);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }
    {
        let func = dev.get_or_load_func("flash_decode_combine", &candle_kernels::FLASH_DECODE)?;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&partials);
        b.arg(&out);
        b.arg(&params);
        unsafe { b.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    drop((q_st, k_st, v_st));
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        candle_core::Storage::Cuda(storage),
        (1usize, n_head, 1usize, hd),
    )))
}


/// Graph-safe split-K decode over the paged F16 KV pool.
///
/// q: `[B, n_head, hd]`; pools: `[capacity_tokens, n_kv, hd]` F16;
/// seqlens_k: `[B + 1]` device buffer; block_table: `[B, max_blocks]`.
pub fn dispatch_flash_decode_paged(
    dev: &CudaDevice,
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use super::paged_kv_cuda::{tensor_cuda_ptr, PAGE_SIZE};

    let (b_sz, n_head, hd) = q.dims3()?;
    let (pool_blocks, page_size, n_kv, pool_hd) = k_pool.dims4()?;
    let (table_rows, table_stride) = block_table.dims2()?;
    if q.dtype() != candle_core::DType::F16
        || k_pool.dtype() != candle_core::DType::F16
        || v_pool.dtype() != candle_core::DType::F16
    {
        candle_core::bail!("paged flash-decode requires F16 q/k/v");
    }
    if b_sz == 0 || table_rows < b_sz || seqlens_k.elem_count() < b_sz + 1 {
        candle_core::bail!("paged flash-decode batch/layout mismatch");
    }
    if pool_hd != hd || n_kv == 0 || n_head % n_kv != 0 {
        candle_core::bail!(
            "paged flash-decode dims: q={:?} k={:?}",
            q.dims(),
            k_pool.dims()
        );
    }
    if page_size != PAGE_SIZE {
        candle_core::bail!(
            "paged flash-decode page size {page_size}, expected {PAGE_SIZE}"
        );
    }

    let capacity_tokens = pool_blocks * page_size;
    let splits = (capacity_tokens / 4096).clamp(8, 64) as u32;
    let params = PagedFlashDecodeParams {
        n_head: n_head as u32,
        n_kv: n_kv as u32,
        hd: hd as u32,
        splits,
        scale,
        page_size: PAGE_SIZE as u32,
        block_table_stride: table_stride as u32,
    };

    let (q_st, q_lay) = q.storage_and_layout();
    let (k_st, k_lay) = k_pool.storage_and_layout();
    let (v_st, v_lay) = v_pool.storage_and_layout();
    let q_v = view_f16(&q_st, q_lay.start_offset())?;
    let k_v = view_f16(&k_st, k_lay.start_offset())?;
    let v_v = view_f16(&v_st, v_lay.start_offset())?;
    let seqlens_ptr = tensor_cuda_ptr(seqlens_k)?;
    let block_table_ptr = tensor_cuda_ptr(block_table)?;

    let partials_len = b_sz * n_head * splits as usize * (2 + hd);
    let mut partials = unsafe { dev.alloc::<f32>(partials_len)? };
    let mut out = unsafe { dev.alloc::<half::f16>(b_sz * n_head * hd)? };
    {
        let func = dev.get_or_load_func(
            "flash_decode_paged_partial",
            &candle_kernels::FLASH_DECODE,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, splits, b_sz as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&q_v);
        builder.arg(&k_v);
        builder.arg(&v_v);
        builder.arg(&seqlens_ptr);
        builder.arg(&block_table_ptr);
        builder.arg(&mut partials);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }
    {
        let func = dev.get_or_load_func(
            "flash_decode_paged_combine",
            &candle_kernels::FLASH_DECODE,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_head as u32, b_sz as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&partials);
        builder.arg(&mut out);
        builder.arg(&params);
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    }

    drop((q_st, k_st, v_st));
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        candle_core::Storage::Cuda(storage),
        (b_sz, n_head, hd),
    )))
}


// ─── Тайловый prefill для Turing (sm_75) ─────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlashPrefillSm75Params {
    pub n_head: u32,
    pub n_kv: u32,
    pub hd: u32,
    pub t: u32,
    pub b: u32,
    pub page_size: u32,
    pub block_table_stride: u32,
    pub causal: u32,
    pub scale: f32,
}

// SAFETY: POD repr(C) из u32/f32.
unsafe impl cudarc::driver::DeviceRepr for FlashPrefillSm75Params {}

/// `PREFILL_SM75=1` включает тайловый prefill-kernel на Turing.
/// Фаза 6: по умолчанию выключен; Фаза 9 (после паритета и бенча) — включит.
pub fn prefill_sm75_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| match std::env::var("PREFILL_SM75") {
        Ok(v) => v.trim() != "0",
        Err(_) => false,
    })
}

/// Чанки короче этого значения идут пер-строчным путём.
pub fn prefill_sm75_min_t() -> usize {
    static MIN_T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_T.get_or_init(|| {
        std::env::var("PREFILL_SM75_MIN_T")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(16)
    })
}

/// Тайловый paged prefill-attention (см. docs/design/2026-10-31-sm75-prefill-kernel.md).
///
/// q: `[B·T, n_head, hd]` F16; пулы: `[pool_blocks, 64, n_kv, hd]` F16;
/// block_table: `[B, max_blocks]` u32. Возвращает `[B·T, n_head, hd]` F16.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_flash_prefill_sm75(
    dev: &CudaDevice,
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    ctx: &super::paged_kv_cuda::PagedModelCtx,
    block_table: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use super::paged_kv_cuda::{tensor_cuda_ptr, PAGE_SIZE};

    let (rows, n_head, hd) = q.dims3()?;
    let (pool_blocks, page_size, n_kv, pool_hd) = k_pool.dims4()?;
    let (b, table_stride) = block_table.dims2()?;
    if q.dtype() != candle_core::DType::F16
        || k_pool.dtype() != candle_core::DType::F16
        || v_pool.dtype() != candle_core::DType::F16
    {
        candle_core::bail!("sm75 prefill требует F16 q/k/v");
    }
    if b == 0 || rows % b != 0 {
        candle_core::bail!("sm75 prefill: rows {rows} не делится на batch {b}");
    }
    if pool_hd != hd || n_kv == 0 || n_head % n_kv != 0 || hd != 256 {
        candle_core::bail!(
            "sm75 prefill dims: q={:?} k={:?} (нужен hd=256)",
            q.dims(),
            k_pool.dims()
        );
    }
    if page_size != PAGE_SIZE {
        candle_core::bail!("sm75 prefill: page_size {page_size}, ожидался {PAGE_SIZE}");
    }
    let t = rows / b;

    let (q_st, q_lay) = q.storage_and_layout();
    let (k_st, k_lay) = k_pool.storage_and_layout();
    let (v_st, v_lay) = v_pool.storage_and_layout();
    let q_v = view_f16(&q_st, q_lay.start_offset())?;
    let k_v = view_f16(&k_st, k_lay.start_offset())?;
    let v_v = view_f16(&v_st, v_lay.start_offset())?;
    let kv_len_ptr = unsafe {
        cudarc::driver::DevicePtr::device_ptr(&ctx.kv_len_dev, &dev.cuda_stream()).0
    };
    let slots_ptr = unsafe {
        cudarc::driver::DevicePtr::device_ptr(&ctx.slots_dev, &dev.cuda_stream()).0
    };
    let bt_ptr = tensor_cuda_ptr(block_table)?;

    let mut out = unsafe { dev.alloc::<half::f16>(rows * n_head * hd)? };
    let params = FlashPrefillSm75Params {
        n_head: n_head as u32,
        n_kv: n_kv as u32,
        hd: hd as u32,
        t: t as u32,
        b: b as u32,
        page_size: PAGE_SIZE as u32,
        block_table_stride: table_stride as u32,
        causal: 1,
        scale,
    };
    let func = dev.get_or_load_func("flash_prefill_sm75", &candle_kernels::FLASH_PREFILL_SM75)?;
    func.set_attribute(
        cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        65536,
    )
    .map_err(candle_core::Error::wrap)?;
    let cfg = LaunchConfig {
        grid_dim: (t.div_ceil(64) as u32, n_head as u32, b as u32),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 65536,
    };
    let mut builder = func.builder();
    builder.arg(&q_v);
    builder.arg(&k_v);
    builder.arg(&v_v);
    builder.arg(&kv_len_ptr);
    builder.arg(&slots_ptr);
    builder.arg(&bt_ptr);
    builder.arg(&mut out);
    builder.arg(&0u64);
    builder.arg(&params);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;

    drop((q_st, k_st, v_st));
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        candle_core::Storage::Cuda(storage),
        (rows, n_head, hd),
    )))
}


// ─── Диагностика PREFILL_NAN_TRACE ──────────────────────────────────────────

static NAN_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NAN_COUNTER: std::sync::Mutex<Option<cudarc::driver::CudaSlice<u32>>> =
    std::sync::Mutex::new(None);

pub fn nan_trace_enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("PREFILL_NAN_TRACE").map(|v| v.trim() != "0").unwrap_or(false))
}

fn nan_stride() -> u64 {
    static S: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("PREFILL_NAN_STRIDE")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&v: &u64| v > 0)
            .unwrap_or(10)
    })
}

/// Посчитать NaN/Inf в F16-тензоре и залогировать строкой
/// `[nancheck] call=N seq=T tag=... n=K`. Проверка идёт раз в `stride`
/// вызовов (по умолчанию 10 — раз на чанк, последний слой), счётчик-буфер
/// переиспользуется: на 130k первых версий с аллокацией на слой был OOM.
pub fn check_nonfinite_f16(dev: &CudaDevice, tag: &str, t: &Tensor, seq_len: usize) -> Result<()> {
    if !nan_trace_enabled() {
        return Ok(());
    }
    let call = NAN_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if call % nan_stride() != 0 {
        return Ok(());
    }
    if t.dtype() != candle_core::DType::F16 {
        eprintln!("[nancheck] call={call} seq={seq_len} tag={tag} skipped_non_f16");
        return Ok(());
    }
    let (st, lay) = t.storage_and_layout();
    let v = view_f16(&st, lay.start_offset())?;
    let n = t.elem_count() as u32;
    let mut guard = NAN_COUNTER.lock().expect("nan counter lock");
    if guard.is_none() {
        *guard = Some(dev.alloc_zeros::<u32>(1).map_err(candle_core::Error::wrap)?);
    }
    let counter = guard.as_mut().expect("nan counter just allocated");
    dev.cuda_stream()
        .memset_zeros(counter)
        .map_err(candle_core::Error::wrap)?;
    let func = dev.get_or_load_func(
        "yf_count_nonfinite_f16",
        &candle_kernels::FLASH_PREFILL_SM75,
    )?;
    let blocks = n.div_ceil(256).clamp(1, 2048);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&v);
    builder.arg(&n);
    builder.arg(&mut *counter);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    dev.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
    let host = dev.clone_dtoh(&*counter).map_err(candle_core::Error::wrap)?;
    drop(st);
    eprintln!("[nancheck] call={call} seq={seq_len} tag={tag} n={}", host[0]);
    Ok(())
}


static LOGIT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// Принудительная проверка (без страйда): используется для трассы блоков.
pub fn check_nonfinite_force(dev: &CudaDevice, tag: &str, t: &Tensor) -> Result<()> {
    if !nan_trace_enabled() {
        return Ok(());
    }
    match t.dtype() {
        candle_core::DType::F16 => {
            let (st, lay) = t.storage_and_layout();
            let v = view_f16(&st, lay.start_offset())?;
            let n = t.elem_count() as u32;
            count_launch(dev, tag, "f16", &v, n)?;
            drop(st);
            Ok(())
        }
        candle_core::DType::F32 => {
            let (st, lay) = t.storage_and_layout();
            let v = match &*st {
                candle_core::Storage::Cuda(cs) => {
                    cs.as_cuda_slice::<f32>()?.slice(lay.start_offset()..)
                }
                _ => candle_core::bail!("check_nonfinite_force: ожидался CUDA"),
            };
            let n = t.elem_count() as u32;
            count_launch(dev, tag, "f32", &v, n)?;
            drop(st);
            Ok(())
        }
        _ => Ok(()),
    }
}

fn count_launch<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + 'static>(
    dev: &CudaDevice,
    tag: &str,
    kind: &str,
    view: &cudarc::driver::CudaView<'_, T>,
    n: u32,
) -> Result<()> {
    let mut guard = NAN_COUNTER.lock().expect("nan counter lock");
    if guard.is_none() {
        *guard = Some(dev.alloc_zeros::<u32>(1).map_err(candle_core::Error::wrap)?);
    }
    let counter = guard.as_mut().expect("counter");
    dev.cuda_stream()
        .memset_zeros(counter)
        .map_err(candle_core::Error::wrap)?;
    let name = if kind == "f16" {
        "yf_count_nonfinite_f16"
    } else {
        "yf_count_nonfinite_f32"
    };
    let func = dev.get_or_load_func(name, &candle_kernels::FLASH_PREFILL_SM75)?;
    let blocks = n.div_ceil(256).clamp(1, 2048);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(view);
    builder.arg(&n);
    builder.arg(&mut *counter);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    dev.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    let host = dev.clone_dtoh(&*counter).map_err(candle_core::Error::wrap)?;
    eprintln!("[nancheck] force tag={tag} n={}", host[0]);
    Ok(())
}

/// Диагностика логитов (PREFILL_NAN_TRACE=1): сколько NaN/Inf в строке.
pub fn check_nonfinite_f32(dev: &CudaDevice, tag: &str, t: &Tensor) -> Result<()> {
    if !nan_trace_enabled() || t.dtype() != candle_core::DType::F32 {
        return Ok(());
    }
    let call = LOGIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let (st, lay) = t.storage_and_layout();
    let v = match &*st {
        candle_core::Storage::Cuda(cs) => cs.as_cuda_slice::<f32>()?.slice(lay.start_offset()..),
        _ => candle_core::bail!("check_nonfinite_f32: ожидался CUDA-тензор"),
    };
    let n = t.elem_count() as u32;
    let mut guard = NAN_COUNTER.lock().expect("nan counter lock");
    if guard.is_none() {
        *guard = Some(dev.alloc_zeros::<u32>(1).map_err(candle_core::Error::wrap)?);
    }
    let counter = guard.as_mut().expect("nan counter just allocated");
    dev.cuda_stream()
        .memset_zeros(counter)
        .map_err(candle_core::Error::wrap)?;
    let func = dev.get_or_load_func(
        "yf_count_nonfinite_f32",
        &candle_kernels::FLASH_PREFILL_SM75,
    )?;
    let blocks = n.div_ceil(256).clamp(1, 2048);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&v);
    builder.arg(&n);
    builder.arg(&mut *counter);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    dev.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    let host = dev.clone_dtoh(&*counter).map_err(candle_core::Error::wrap)?;
    drop(st);
    eprintln!("[nancheck] call={call} tag={tag} n={}", host[0]);
    Ok(())
}

fn view_f16<'a>(
    st: &'a candle_core::Storage,
    offset: usize,
) -> Result<cudarc::driver::CudaView<'a, half::f16>> {
    match st {
        Storage::Cuda(cs) => Ok(cs.as_cuda_slice::<half::f16>()?.slice(offset..)),
        _ => candle_core::bail!("flash_decode: expected CUDA f16 storage"),
    }
}

// Подавить unused warning для CudaSlice импорта (тип используется в сигнатурах).
#[allow(unused)]
type _T = CudaSlice<f32>;
