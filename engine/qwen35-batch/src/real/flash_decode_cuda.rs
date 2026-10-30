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
