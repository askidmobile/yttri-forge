use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, PushKernelArg, SyncOnDrop};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
    /// Кэш полной F16-деквантизации весов (IQ-типы). Один раз при первом
    /// matmul; дальше чистый HGEMM вместо dequant-всей-матрицы-на-шаг.
    /// Включается env QWEN36_DEQUANT_CACHE=1 (смысл: карты с большим VRAM,
    /// A100 80GB; на 12GB не влезает — там tiled fallback как было).
    dequant_cache: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<CudaStorage>>>>,
}

fn dequant_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN36_DEQUANT_CACHE").is_some())
}

pub(crate) static FORCE_DMMV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// QWEN36_FORCE_MMQ=1: MMQ kernels даже для m=1 (decode). Эксперимент.
fn force_mmq() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN36_FORCE_MMQ").is_some())
}

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

// Per-device Q8_1 scratch cache: 2 слота на размер, чтобы избежать self-deadlock
// когда вход слоя и выход attention_wo имеют одинаковую размерность (например 9B: 4096 == 4096).
static Q81_SCRATCH: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<
            crate::cuda_backend::DeviceId,
            std::collections::HashMap<usize, (CudaSlice<u8>, CudaSlice<u8>, std::sync::atomic::AtomicBool)>,
        >,
    >,
> = std::sync::OnceLock::new();

pub struct Q81ScratchSlice {
    dev_id: crate::cuda_backend::DeviceId,
    bytes: usize,
    slot_idx: usize,
}

impl Drop for Q81ScratchSlice {
    fn drop(&mut self) {
        if let Some(map) = Q81_SCRATCH.get() {
            if let Ok(guard) = map.lock() {
                if let Some(per_size) = guard.get(&self.dev_id) {
                    if let Some((_, _, in_use)) = per_size.get(&self.bytes) {
                        in_use.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        }
    }
}

pub struct Q81ScratchRef {
    ptr: *const CudaSlice<u8>,
    _token: Q81ScratchSlice,
}

unsafe impl Send for Q81ScratchRef {}
unsafe impl Sync for Q81ScratchRef {}

impl std::ops::Deref for Q81ScratchRef {
    type Target = CudaSlice<u8>;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl std::ops::DerefMut for Q81ScratchRef {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.ptr as *mut CudaSlice<u8>) }
    }
}

fn q8_1_scratch(dev: &CudaDevice, bytes: usize) -> Q81ScratchRef {
    let map = Q81_SCRATCH
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let bytes = bytes.max(1);
    let mut guard = map.lock().unwrap();
    let per_size = guard.entry(dev.id()).or_default();
    let entry = per_size.entry(bytes).or_insert_with(|| {
        let s0 = unsafe { dev.alloc::<u8>(bytes) }.expect("q8_1 scratch alloc 0");
        let s1 = unsafe { dev.alloc::<u8>(bytes) }.expect("q8_1 scratch alloc 1");
        (s0, s1, std::sync::atomic::AtomicBool::new(false))
    });

    let (ptr, slot_idx) = if !entry.2.swap(true, std::sync::atomic::Ordering::Relaxed) {
        (&entry.0 as *const CudaSlice<u8>, 0)
    } else {
        (&entry.1 as *const CudaSlice<u8>, 1)
    };

    Q81ScratchRef {
        ptr,
        _token: Q81ScratchSlice {
            dev_id: dev.id(),
            bytes,
            slot_idx,
        },
    }
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_GET_ROWS_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut cudarc::driver::CudaViewMut<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let mut dst_chunk = dst.slice_mut(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&mut dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        GgmlDType::IQ3XXS => ("dequantize_block_iq3_xxs_f32", true, 256, nb),
        GgmlDType::IQ2S => ("dequantize_block_iq2_s_f32", true, 256, nb),
        GgmlDType::IQ3S => ("dequantize_block_iq3_s_f32", true, 256, nb),
        GgmlDType::IQ2XS => ("dequantize_block_iq2_xs_f32", true, 256, nb),
        GgmlDType::IQ2XXS => ("dequantize_block_iq2_xxs_f32", true, 256, nb),
        GgmlDType::IQ1M => ("dequantize_block_iq1_m_f32", true, 256, nb),
        GgmlDType::IQ4XS => ("dequantize_block_iq4_xs_f32", true, 256, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Dequantize a row-slice [row_start, row_end) of an IQ-type weight [n, k] into
/// a contiguous f32 buffer [(row_end - row_start) * k].
///
/// IQ kernels index blocks by `blockIdx.x` and write `yy[i * QK_K + pos]`, so
/// slicing `data.inner` by byte offset shifts the base pointer — the kernel
/// sees the first block of the slice as block 0. This avoids dequantizing the
/// full [n, k] weight (n*k*4 bytes f32) which can exceed VRAM headroom.
fn dequantize_f32_rowslice(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    row_start: usize,
    row_end: usize,
    k: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let block_size = dtype.block_size();
    let type_size = dtype.type_size();
    let blocks_per_row = k / block_size;
    let chunk_rows = row_end - row_start;
    let nb = chunk_rows * blocks_per_row;
    let elem_count = chunk_rows * k;
    let byte_offset = row_start * blocks_per_row * type_size;
    let byte_len = nb * type_size;

    let (kernel_name, block_dim) = match dtype {
        GgmlDType::IQ3XXS => ("dequantize_block_iq3_xxs_f32", 256),
        GgmlDType::IQ2S => ("dequantize_block_iq2_s_f32", 256),
        GgmlDType::IQ3S => ("dequantize_block_iq3_s_f32", 256),
        GgmlDType::IQ2XS => ("dequantize_block_iq2_xs_f32", 256),
        GgmlDType::IQ2XXS => ("dequantize_block_iq2_xxs_f32", 256),
        GgmlDType::IQ1M => ("dequantize_block_iq1_m_f32", 256),
        GgmlDType::IQ4XS => ("dequantize_block_iq4_xs_f32", 256),
        // K-quants и Q8_0 — для MoE reference rowslice (Q8_0 эксперты).
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", 64),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", 64),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", 32),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", 64),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", 64),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", 32),
        _ => crate::bail!("unsupported dtype for rowslice dequant: {dtype:?}"),
    };
    let is_k = !matches!(dtype, GgmlDType::Q8_0);
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nb as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let src_view = data.inner.slice(byte_offset..byte_offset + byte_len);
    let mut builder = func.builder();
    builder.arg(&src_view);
    builder.arg(&dst);
    let nb32 = (elem_count / 32) as i32;
    if !is_k {
        // non-k ядра ждут nb32 (число 32-элементных блоков).
        builder.arg(&nb32);
    }
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        GgmlDType::IQ3XXS => ("dequantize_block_iq3_xxs_f16", true, 256, nb),
        GgmlDType::IQ2S => ("dequantize_block_iq2_s_f16", true, 256, nb),
        GgmlDType::IQ3S => ("dequantize_block_iq3_s_f16", true, 256, nb),
        GgmlDType::IQ2XS => ("dequantize_block_iq2_xs_f16", true, 256, nb),
        GgmlDType::IQ2XXS => ("dequantize_block_iq2_xxs_f16", true, 256, nb),
        GgmlDType::IQ1M => ("dequantize_block_iq1_m_f16", true, 256, nb),
        GgmlDType::IQ4XS => ("dequantize_block_iq4_xs_f16", true, 256, nb),
        // BF16 — не квант: простой cast bf16→f16 (dequantize_f16 вызывается
        // из QMatMul::from_arc для F16/BF16 весов; без этого полные BF16
        // GGUF падали "unsupported dtype", а через F32 — 2x VRAM/OOM).
        GgmlDType::BF16 => {
            let view = unsafe { data.inner.transmute::<half::bf16>(elem_count) }
                .ok_or_else(|| {
                    crate::Error::Msg("bf16 view: size mismatch".into()).bt()
                })?;
            let dst = unsafe { dev.alloc::<f16>(elem_count)? };
            let func = dev.get_or_load_func("cast_bf16_f16", &candle_kernels::CAST)?;
            let cfg = cudarc::driver::LaunchConfig::for_num_elems(elem_count as u32);
            let mut builder = func.builder();
            barg!(builder, elem_count);
            barg!(builder, 0usize); // num_dims = 0 (contiguous)
            let null_info: usize = 0;
            barg!(builder, null_info); // info = nullptr
            builder.arg(&view);
            builder.arg(&dst);
            unsafe { builder.launch(cfg) }.w()?;
            return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
        }
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn get_rows(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    hidden: usize,
    ids: &CudaView<u32>,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let (kernel_name, block_dim, block_num_y, can_stride_y) = match dtype {
        GgmlDType::F32 => (
            "get_rows_f32",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::F16 => (
            "get_rows_f16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::BF16 => (
            "get_rows_bf16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_0 => (
            "get_rows_q4_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_1 => (
            "get_rows_q4_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_0 => (
            "get_rows_q5_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_1 => (
            "get_rows_q5_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q8_0 => (
            "get_rows_q8_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q2K => ("get_rows_q2_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q3K => ("get_rows_q3_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q4K => ("get_rows_q4_K", 32, hidden / dtype.block_size(), false),
        GgmlDType::Q5K => ("get_rows_q5_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q6K => ("get_rows_q6_K", 64, hidden / dtype.block_size(), false),
        _ => crate::bail!("unsupported dtype for CUDA quantized embedding {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let ids_len = ids.len();
    let dst = unsafe { dev.alloc::<f32>(ids_len * hidden)? };
    if ids_len == 0 {
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    if !can_stride_y && block_num_y > u16::MAX as usize {
        crate::bail!("quantized embedding hidden size {hidden} exceeds CUDA grid y limit")
    }
    let grid_y = if can_stride_y {
        block_num_y.min(u16::MAX as usize)
    } else {
        block_num_y
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ids_len as u32, grid_y as u32, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_stride = hidden * dtype.type_size() / dtype.block_size();

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(ids);
    builder.arg(&dst);
    barg!(builder, hidden as i64, row_stride);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

// IQ-type fallback: dequantize weights to f32 on GPU, then matvec via cuBLAS.
// Used by dequantize_matmul_vec for IQ3XXS (no fused kernel exists).
// `rhs` is the [b, m, ncols] f32 activation storage.
fn dequantize_mul_mat_vec_via_cublas(
    data: &PaddedCudaSlice,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b: usize,
    m: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use crate::backend::BackendStorage;
    let storage = QCudaStorage {
        data: data.clone(),
        dtype,
        device: dev.clone(),
        dequant_cache: Default::default(),
    };
    // Dequantize [nrows, ncols] weights to f32
    let data_f32 = storage.dequantize(nrows * ncols)?;
    // cuBLAS: result[b, m, nrows] = rhs[b, m, ncols] @ data_f32[nrows, ncols]^T
    // Weights [nrows, ncols] row-major => transposed view [ncols, nrows] (swap strides).
    let weight_l =
        crate::Layout::new((ncols, nrows).into(), vec![1, ncols], 0)
            .broadcast_as((b, ncols, nrows))?;
    rhs.matmul(&data_f32, (b, m, nrows, ncols), rhs_l, &weight_l)
}

#[cfg(test)]
fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

impl QCudaStorage {
    /// Batch-vector matmul using pre-quantized Q8_1 activations.
    /// `y_q8_1` is the pre-quantized [b_size, k_padded] buffer.
    pub fn mul_mat_vec_with_prequant_q8_1(
        &self,
        self_shape: &crate::Shape,
        y_q8_1: &CudaSlice<u8>,
        ncols: usize,
        nrows: usize,
        b_size: usize,
    ) -> Result<CudaStorage> {
        let (n, k) = self_shape.dims2()?;
        if n != nrows || k != ncols {
            crate::bail!("shape mismatch in mul_mat_vec_with_prequant_q8_1: expected ({nrows}, {ncols}), got ({n}, {k})");
        }
        let data_elems = self.data.len / self.dtype.type_size() * self.dtype.block_size();
        if data_elems < ncols * nrows {
            crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
        }
        let dev = self.device();
        let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);

        let kernel_name = match self.dtype {
            GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
            GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
            GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
            GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
            GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
            GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
            GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
            GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
            GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
            GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
            GgmlDType::IQ2XXS => "mul_mat_vec_iq2_xxs_q8_1_cuda",
            GgmlDType::IQ1M => "mul_mat_vec_iq1_m_q8_1_cuda",
            GgmlDType::IQ2XS => "mul_mat_vec_iq2_xs_q8_1_cuda",
            GgmlDType::IQ2S => "mul_mat_vec_iq2_s_q8_1_cuda",
            GgmlDType::IQ3XXS => "mul_mat_vec_iq3_xxs_q8_1_cuda",
            GgmlDType::IQ3S => "mul_mat_vec_iq3_s_q8_1_cuda",
            GgmlDType::IQ4XS => "mul_mat_vec_iq4_xs_q8_1_cuda",
            _ => crate::bail!("unsupported dtype for quantized matmul {:?}", self.dtype),
        };
        let kernel_name = format!("{kernel_name}{b_size}");
        let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
        let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
        let (nblocks, nwarps) = match b_size {
            1 => (nrows as u32, 4),
            2..=4 => ((nrows as u32).div_ceil(2), 4),
            5..=8 => ((nrows as u32).div_ceil(2), 2),
            _ => crate::bail!("unexpected bsize {b_size}"),
        };
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (nblocks, 1, 1),
            block_dim: (WARP_SIZE as u32, nwarps, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&self.data.inner);
        builder.arg(y_q8_1);
        builder.arg(&dst);
        barg!(
            builder,
            /* ncols_x */ ncols as i32,
            /* nrows_x */ nrows as i32,
            /* nrows_y */ ncols_padded as i32,
            /* nrows_dst */ nrows as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
    }
}

impl QCudaStorage {
    /// Pre-quantize f32 activations to Q8_1 for reuse across multiple QMatMul operations.
    pub fn prequantize_q8_1(
        dev: &CudaDevice,
        y: &CudaView<f32>,
        ncols: usize,
        b_size: usize,
    ) -> Result<Q81ScratchRef> {
        let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut guard = q8_1_scratch(dev, y_size_in_bytes);
        {
            let mut view = guard.slice_mut(..);
            quantize_q8_1(y, &mut view, ncols, b_size, dev)?;
        }
        Ok(guard)
    }
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y (scratch: reused across matmuls, stream-ordered).
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1_guard = q8_1_scratch(dev, y_size_in_bytes);
    {
        let mut view = y_q8_1_guard.slice_mut(..);
        quantize_q8_1(y, &mut view, ncols, b_size, dev)?;
    }

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        GgmlDType::IQ2XXS => "mul_mat_vec_iq2_xxs_q8_1_cuda",
        GgmlDType::IQ2XS => "mul_mat_vec_iq2_xs_q8_1_cuda",
        GgmlDType::IQ2S => "mul_mat_vec_iq2_s_q8_1_cuda",
        GgmlDType::IQ3XXS => "mul_mat_vec_iq3_xxs_q8_1_cuda",
        GgmlDType::IQ3S => "mul_mat_vec_iq3_s_q8_1_cuda",
        GgmlDType::IQ4XS => "mul_mat_vec_iq4_xs_q8_1_cuda",
        GgmlDType::IQ1M => "mul_mat_vec_iq1_m_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&*y_q8_1_guard);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y (scratch: reused across matmuls, stream-ordered).
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1_guard = q8_1_scratch(dev, y_size_in_bytes);
    {
        let mut view = y_q8_1_guard.slice_mut(..);
        quantize_q8_1(y, &mut view, k, y_cols, dev)?;
    }

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &*y_q8_1_guard);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn contiguous_view<'a, T>(
    storage: &'a CudaSlice<T>,
    layout: &crate::Layout,
    name: &str,
) -> Result<CudaView<'a, T>> {
    let (start, end) = layout
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::Msg(format!("indexed moe: {name} not contiguous")).bt())?;
    if end > storage.len() {
        crate::bail!(
            "indexed moe: {name} layout range {start}..{end} exceeds storage length {}",
            storage.len()
        )
    }
    Ok(storage.slice(start..end))
}

#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_dispatch(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaView<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    let (n_experts, n, k) = w_shape.dims3()?;
    let (batch, input_dim1, input_k) = in_shape.dims3()?;
    let (ids_batch, topk) = idx_shape.dims2()?;
    if batch != ids_batch {
        crate::bail!("indexed moe batch mismatch: input={batch}, ids={ids_batch}")
    }
    if input_dim1 != 1 && input_dim1 != topk {
        crate::bail!("indexed moe input dim1 must be 1 or topk={topk}, got {input_dim1}")
    }
    if input_k != k {
        crate::bail!("indexed moe input width mismatch: weights={k}, input={input_k}")
    }
    if batch == 0 || topk == 0 || n_experts == 0 {
        crate::bail!(
            "indexed moe dimensions must be nonzero: experts={n_experts}, batch={batch}, topk={topk}"
        )
    }

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant_guard = q8_1_scratch(dev, y_size_in_bytes);
    {
        let mut view = input_quant_guard.slice_mut(..);
        quantize_q8_1(input, &mut view, k, total_rows, dev)?;
    }

    // output buffer
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

    let use_grouped = batch > 1
        && std::env::var_os("QWEN36_ENABLE_MOE_GROUPED").is_some()  // OFF by default — measured slower (2026-08-24)
        && matches!(
            w_dtype,
            GgmlDType::IQ2XXS
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2S
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ3S
                | GgmlDType::IQ4XS
        );
    let kernel_name = if use_grouped {
        match w_dtype {
            GgmlDType::IQ2XXS => "indexed_moe_forward_iq2_xxs_q8_1_grouped",
            GgmlDType::IQ2XS => "indexed_moe_forward_iq2_xs_q8_1_grouped",
            GgmlDType::IQ2S => "indexed_moe_forward_iq2_s_q8_1_grouped",
            GgmlDType::IQ3XXS => "indexed_moe_forward_iq3_xxs_q8_1_grouped",
            GgmlDType::IQ3S => "indexed_moe_forward_iq3_s_q8_1_grouped",
            GgmlDType::IQ4XS => "indexed_moe_forward_iq4_xs_q8_1_grouped",
            _ => unreachable!(),
        }
    } else {
        match w_dtype {
            GgmlDType::IQ2XXS => "indexed_moe_forward_iq2_xxs_q8_1",
            GgmlDType::IQ2XS => "indexed_moe_forward_iq2_xs_q8_1",
            GgmlDType::IQ2S => "indexed_moe_forward_iq2_s_q8_1",
            GgmlDType::IQ3XXS => "indexed_moe_forward_iq3_xxs_q8_1",
            GgmlDType::IQ3S => "indexed_moe_forward_iq3_s_q8_1",
            GgmlDType::IQ4XS => "indexed_moe_forward_iq4_xs_q8_1",
            GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
            GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
            GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
            GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
            GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
            GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
            _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
        }
    };
    let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
    // Grouped: grid=(n, n_experts). Non-grouped: grid=(n, batch, topk).
    let cfg = if use_grouped {
        let n_experts = w_shape.dims3()?.0;
        cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, n_experts as u32, 1),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, batch as u32, topk as u32),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        }
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&*input_quant_guard);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    let t0 = std::time::Instant::now();
    unsafe { builder.launch(cfg) }.w()?;
    let trace_mmq = std::env::var_os("QWEN36_TRACE_MMQ").is_some();
    if trace_mmq {
        let _ = dev.cuda_stream().synchronize();
        eprintln!(
            "[moe] {} dtype={:?} grid=({},{},{}) n={} k={} batch={} topk={} gpu={:.1}ms",
            if use_grouped { "GROUPED" } else { "BASIC" },
            w_dtype, n, batch, topk, n, k, batch, topk,
            t0.elapsed().as_secs_f64() * 1e3,
        );
    }

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    /// Dual indexed MoE: gate+up with shared input and two outputs.
    /// IQ types keep F32 input and use two measured-faster launches; K-quants share Q8_1 input.
    /// w1/w2 — packed [n_experts, n, k] одинакового dtype; input [batch, topk, k] f32;
    /// ids [batch, topk] u32. Выход: два [batch, topk, n] f32.
    pub fn indexed_moe_forward_dual(
        &self,
        other: &QCudaStorage,
        self_shape: &crate::Shape,
        input: &CudaStorage,
        input_l: &crate::Layout,
        ids: &CudaStorage,
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, CudaStorage, crate::Shape)> {
        let dtype = self.dtype();
        if dtype != other.dtype() {
            crate::bail!("dual moe: dtype mismatch {:?} vs {:?}", dtype, other.dtype());
        }
        let input_storage = input.as_cuda_slice::<f32>()?;
        let input_view = contiguous_view(input_storage, input_l, "input")?;
        let ids_storage = ids.as_cuda_slice::<u32>()?;
        let ids_view = contiguous_view(ids_storage, ids_l, "ids")?;
        let kernel_name = match dtype {
            GgmlDType::IQ2XXS => "indexed_moe_forward_dual_iq2_xxs_q8_1",
            GgmlDType::IQ2XS => "indexed_moe_forward_dual_iq2_xs_q8_1",
            GgmlDType::IQ2S => "indexed_moe_forward_dual_iq2_s_q8_1",
            GgmlDType::IQ3XXS => "indexed_moe_forward_dual_iq3_xxs_q8_1",
            GgmlDType::IQ3S => "indexed_moe_forward_dual_iq3_s_q8_1",
            GgmlDType::IQ4XS => "indexed_moe_forward_dual_iq4_xs_q8_1",
            GgmlDType::Q8_0 => "indexed_moe_forward_dual_q8_0_q8_1",
            GgmlDType::Q2K => "indexed_moe_forward_dual_q2k_q8_1",
            GgmlDType::Q4K => "indexed_moe_forward_dual_q4k_q8_1",
            GgmlDType::Q6K => "indexed_moe_forward_dual_q6k_q8_1",
            _ => crate::bail!("unsupported dtype for dual indexed moe {dtype:?}"),
        };
        let (n, k) = (self_shape.dims3()?.1, self_shape.dims3()?.2);
        let batch = input_l.shape().dims()[0];
        let topk = ids_l.shape().dims()[1];
        let input_dim1 = input_l.shape().dims()[1];

        // q8_1 quantize входа (один раз на обе проекции).
        let dev = &self.device;
        let total_rows = batch * input_dim1;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            k_padded * total_rows * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut input_quant_guard = q8_1_scratch(dev, y_size_in_bytes);
        {
            let mut view = input_quant_guard.slice_mut(..);
            quantize_q8_1(&input_view, &mut view, k, total_rows, dev)?;
        }

        let outsize = batch * topk * n;
        let out1 = unsafe { dev.alloc::<f32>(outsize)? };
        let out2 = unsafe { dev.alloc::<f32>(outsize)? };

        let func = dev.get_or_load_func(kernel_name, &candle_kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, batch as u32, topk as u32),
            block_dim: (WARP_SIZE as u32, 4, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(&self.data.inner);
        b.arg(&other.data.inner);
        b.arg(&*input_quant_guard);
        b.arg(&ids_view);
        b.arg(&out1);
        b.arg(&out2);
        barg!(b, n as i32);
        barg!(b, k as i32);
        barg!(b, batch as i32);
        barg!(b, topk as i32);
        barg!(b, k_padded as i32);
        barg!(b, input_dim1 as i32);
        let t0 = std::time::Instant::now();
        unsafe { b.launch(cfg) }.w()?;
        let trace_mmq = std::env::var_os("QWEN36_TRACE_MMQ").is_some();
        if trace_mmq {
            let _ = dev.cuda_stream().synchronize();
            eprintln!(
                "[moe2] dtype={:?} grid=({},{},{}) n={} k={} batch={} topk={} gpu={:.1}ms",
                dtype, n, batch, topk, n, k, batch, topk,
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }

        let shape: crate::Shape = (batch, topk, n).into();
        Ok((
            CudaStorage::wrap_cuda_slice(out1, dev.clone()),
            CudaStorage::wrap_cuda_slice(out2, dev.clone()),
            shape,
        ))
    }
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // Shape validation (was an assert that crashed the process).
        let in_dims = input_l.shape().dims();
        if in_dims.len() != 3 {
            crate::bail!(
                "indexed_moe_forward expects input rank 3 [batch, topk_or_1, k], got {:?}",
                in_dims
            );
        }
        let id_dims = ids_l.shape().dims();
        if id_dims.len() != 2 {
            crate::bail!(
                "indexed_moe_forward expects ids rank 2 [batch, topk], got {:?}",
                id_dims
            );
        }
        if in_dims[0] != id_dims[0] {
            crate::bail!(
                "indexed_moe_forward batch mismatch: input batch {} vs ids batch {}",
                in_dims[0],
                id_dims[0]
            );
        }
        if matches!(
            self.dtype(),
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
        ) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let input_view = contiguous_view(input_storage, input_l, "input")?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            let ids_view = contiguous_view(ids_storage, ids_l, "ids")?;
            indexed_moe_forward_dispatch(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                &input_view,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_view,
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
            dequant_cache: Default::default(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Деквантизация всей матрицы с кэшем (IQ-типы, QWEN36_DEQUANT_CACHE=1):
    /// один раз dequant в F32, дальше cuBLAS SGEMM по кэшу.
    fn cached_dequant_f32(&self, elem_count: usize) -> Result<std::sync::Arc<CudaStorage>> {
        let mut g = self.dequant_cache.lock().unwrap();
        if g.is_none() {
            if std::env::var("QWEN36_TRACE")
                .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false)
            {
                eprintln!(
                    "[cuda] dequant cache fill: dtype={:?} elems={} (~{:.0}MiB f32)",
                    self.dtype,
                    elem_count,
                    elem_count as f64 * 4.0 / 1048576.0
                );
            }
            let w = QCudaStorage {
                data: self.data.clone(),
                dtype: self.dtype,
                device: self.device.clone(),
                dequant_cache: Default::default(),
            }
            .dequantize(elem_count)?;
            *g = Some(std::sync::Arc::new(w));
        }
        Ok(g.as_ref().unwrap().clone())
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ2S
                | GgmlDType::IQ3S
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ1M
                | GgmlDType::IQ4XS
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
            // IQ-types added in candle-fork, CUDA dequant
            // not yet implemented -- fallback via bail.
            _ => crate::bail!("unsupported dtype for cuda dequantize: {:?}", self.dtype),
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&*data, &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &CudaStorage,
        ids_l: &crate::Layout,
    ) -> Result<CudaStorage> {
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids = ids.as_cuda_slice::<u32>()?;
        let ids = match ids_l.contiguous_offsets() {
            Some((o1, o2)) => ids.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-embedding",
            }
            .bt())?,
        };
        get_rows(&self.data, self.dtype, hidden, &ids, self.device())
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // Optimized MMVQ and MMQ paths (support most paths: BF16/F16/F32, batch 1-8, all quant types, reuses per-device workspace).
        if !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(result) = super::fast_mmvq::try_fwd(self, self_shape, storage, layout)? {
                return Ok(result);
            }
            if std::env::var_os("QWEN36_TRACE_MMQ").is_some() {
                let m = match layout.shape().dims() {
                    [b, m, _] => b * m,
                    [m, _] => *m,
                    _ => 0,
                };
                let (n, k) = self_shape.dims2()?;
                eprintln!("[mmq-entry] dtype={:?} m={} n={} k={}", self.dtype, m, n, k);
            }
            if let Some(result) = super::fast_mmq::try_fwd(self, self_shape, storage, layout)? {
                return Ok(result);
            }
            if std::env::var_os("QWEN36_TRACE_MMQ").is_some() {
                let (n, k) = self_shape.dims2()?;
                let m = match layout.shape().dims() {
                    [b, m, _] => b * m,
                    [m, _] => *m,
                    _ => 0,
                };
                eprintln!(
                    "[mmq-fallback] dtype={:?} m={} n={} k={} — MMQ returned None, using dequantize_matmul",
                    self.dtype, m, n, k,
                );
            }
        }
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            // 0, не 1: vec-ветка под FORCE_DMMV уходит в
            // dequantize_mul_mat_vec_via_cublas, который деквантует ВЕСЬ вес в
            // f32 без тайлинга — голова 27B [248320x5120] это 5.09 GB и OOM на
            // 12GB. Tiled dequantize_matmul корректен и для m=1 (chunked
            // dequant + cuBLAS gemv), диагностический путь остаётся честным
            // dequant-референсом.
            0
        } else if force_mmq() {
            // Эксперимент (шаг 2 perf-плана): MMQ вместо dmmv даже для m=1.
            0
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    /// Tensor-Core MMA MMQ (llama.cpp mul_mat_q) для плотного prefill (m>8).
    /// Возвращает None, если dtype/форма не поддержаны — caller фолбэчится.
    pub fn mul_mat_q_mma(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<Option<(CudaStorage, crate::Shape)>> {
        // dtype → (тэг ядра, mma tile_x_k, quant-обёртка)
        let (tag, tile_x_k, quant_kernel): (&str, usize, &str) = match self.dtype {
            GgmlDType::Q2K => ("q2_k", 100, "candle_mmq_quant_d2s6"),
            GgmlDType::Q3K => ("q3_k", 84, "candle_mmq_quant_d4"),
            GgmlDType::Q4K => ("q4_k", 76, "candle_mmq_quant_ds4"),
            GgmlDType::Q5K => ("q5_k", 76, "candle_mmq_quant_ds4"),
            GgmlDType::Q6K => ("q6_k", 76, "candle_mmq_quant_d4"),
            GgmlDType::Q4_0 => ("q4_0", 76, "candle_mmq_quant_ds4"),
            GgmlDType::Q8_0 => ("q8_0", 76, "candle_mmq_quant_d4"),
            // IQ: tile_x_k по mmq_get_mma_tile_x_k (iq2_xs/iq2_s -> Q3_K=84,
            // прочие -> Q8_0=76), layout активаций у всех -> D4.
            GgmlDType::IQ2XXS => ("iq2_xxs", 76, "candle_mmq_quant_d4"),
            GgmlDType::IQ2XS => ("iq2_xs", 84, "candle_mmq_quant_d4"),
            GgmlDType::IQ2S => ("iq2_s", 84, "candle_mmq_quant_d4"),
            GgmlDType::IQ3XXS => ("iq3_xxs", 76, "candle_mmq_quant_d4"),
            GgmlDType::IQ3S => ("iq3_s", 76, "candle_mmq_quant_d4"),
            GgmlDType::IQ4XS => ("iq4_xs", 76, "candle_mmq_quant_d4"),
            _ => return Ok(None),
        };
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            [b, m, k2] => (*b, *m, *k2),
            [m, k2] => (1, *m, *k2),
            _ => return Ok(None),
        };
        if k2 != k {
            return Ok(None);
        }
        let m_total = b * m;
        const MMQ_Y: usize = 128;
        const MMQ_NWARPS: usize = 8;
        const BLOCK_Q8_1_MMQ: usize = 144;
        // need_check=false требует nrows_x % MMQ_Y == 0; decode (m<=8) идёт через MMVQ.
        if n % MMQ_Y != 0 || m_total < 9 {
            return Ok(None);
        }
        let dev = &self.device;
        let rhs = storage.as_cuda_slice::<f32>()?;
        let rhs = match layout.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => return Ok(None),
        };

        // mmq_x: наибольший из {128,64,32} с shared-mem <= optin лимита.
        let pad_to = |a: usize, al: usize| ((a + al - 1) / al) * al;
        let nbs_for = |mmq_x: usize| {
            let nbs_ids = mmq_x * 4;
            let nbs_x = MMQ_Y * tile_x_k * 4;
            let nbs_y = pad_to(mmq_x * BLOCK_Q8_1_MMQ, MMQ_NWARPS * WARP_SIZE * 4);
            nbs_ids + nbs_x + nbs_y
        };
        let smpbo = 100 * 1024; // optin shared-mem лимит (sm_86 ~99KB, берём с запасом)
        let mmq_x = [128usize, 64, 32]
            .into_iter()
            .find(|&x| nbs_for(x) <= smpbo)
            .unwrap_or(32);
        let nbs = nbs_for(mmq_x);

        // 1) Квантизация активаций в q8_1_mmq.
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let blocks_per_row = k_padded / 128; // block_q8_1_mmq держит 4*QK8_1 = 128 значений
        let y_bytes = m_total * blocks_per_row * BLOCK_Q8_1_MMQ;
        let y_mmq = unsafe { dev.alloc::<u8>(y_bytes)? };
        let qfunc = dev.get_or_load_func(quant_kernel, &candle_kernels::CANDLE_MMQ_DENSE)?;
        let qcfg = cudarc::driver::LaunchConfig {
            grid_dim: (m_total as u32, ceil_div(k_padded, 512) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut qb = qfunc.builder();
            qb.arg(&rhs);
            qb.arg(&y_mmq);
            barg!(
                qb,
                /* ne00 */ k as i64,
                /* s01  */ k as i64,
                /* ne0  */ k_padded as i64,
                /* ne1  */ m_total as i32
            );
            unsafe { qb.launch(qcfg) }.w()?;
        }

        // 2) MMQ Tensor-Core GEMM: dst = weight[n,k] @ act[k,m] → [m,n].
        // stride_row_x в блоках кванта (block_size = qk: 256 для K-quants, 32 для legacy).
        let stride_row_x = k / self.dtype.block_size();
        let kernel_name = format!("candle_mmq_{tag}_x{mmq_x}");
        let func = dev.get_or_load_func(&kernel_name, &candle_kernels::CANDLE_MMQ_DENSE)?;
        if nbs > 48 * 1024 {
            func.set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                nbs as i32,
            )
            .map_err(|e| crate::Error::Msg(format!("mmq set_attribute: {e:?}")))?;
        }
        let dst = unsafe { dev.alloc::<f32>(n * m_total)? };
        // stream-k ветка ядра: grid.x = число тайлов (nty*ntx), grid.y/z=1.
        // Каждый блок ведёт один output-тайл по всему k (fixup не нужен).
        let ntx = ceil_div(m_total, mmq_x);
        let nty = ceil_div(n, MMQ_Y);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((nty * ntx) as u32, 1, 1),
            block_dim: (WARP_SIZE as u32, MMQ_NWARPS as u32, 1),
            shared_mem_bytes: nbs as u32,
        };
        {
            let mut mb = func.builder();
            mb.arg(&self.data.inner);
            mb.arg(&y_mmq);
            mb.arg(&dst);
            barg!(
                mb,
                /* ncols_x      */ k as i32,
                /* nrows_x      */ n as i32,
                /* ncols_dst    */ m_total as i32,
                /* stride_row_x */ stride_row_x as i32,
                /* ncols_y      */ m_total as i32,
                /* stride_col_dst */ n as i32,
                /* ncols_max    */ m_total as i32
            );
            unsafe { mb.launch(cfg) }.w()?;
        }

        if std::env::var_os("QWEN36_TRACE_MMQ").is_some() {
            eprintln!(
                "[mmq] dtype={:?} m={} n={} k={} mmq_x={} nbs={} tiles={}x{}",
                self.dtype, m_total, n, k, mmq_x, nbs, nty, ntx,
            );
        }
        let t_mmq = std::time::Instant::now();
        let trace_mmq = std::env::var_os("QWEN36_TRACE_MMQ").is_some();
        if trace_mmq {
            let _ = dev.cuda_stream().synchronize();
        }
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        if trace_mmq {
            eprintln!("[mmq] gpu={:.1}ms", t_mmq.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Some((
            CudaStorage::wrap_cuda_slice(dst, dev.clone()),
            out_shape.into(),
        )))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }

    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(*const u8, SyncOnDrop<'a>)> {
        let (ptr, guard) = self.data.inner.device_ptr(stream);
        Ok((ptr as *const u8, guard))
    }

    /// Dequantize a row-slice [row_start, row_end) of a 2D weight [n, k] into
    /// a contiguous f32 buffer [(row_end - row_start) * k].
    ///
    /// Uses the per-block IQ kernel by slicing the device buffer at a byte
    /// offset — no full-weight f32 allocation. Only IQ-types are supported
    /// (other dtypes fall back to `dequantize` + reshape).
    pub fn dequantize_rowslice(
        &self,
        row_start: usize,
        row_end: usize,
        k: usize,
    ) -> Result<CudaStorage> {
        dequantize_f32_rowslice(&self.data, self.dtype, row_start, row_end, k, self.device())
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs_slice = rhs.as_cuda_slice::<f32>()?;
        let rhs_slice = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs_slice.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b, m, k) = match rhs_l.shape().dims() {
            [b, m, k] => (*b, *m, *k),
            [b, k] => (*b, 1, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }
        let b_size = b * m;

        // IQ-types have no fused vec/matmul kernels — use dequantize + cuBLAS.
        let iq_type = matches!(
            self.dtype,
            GgmlDType::IQ3XXS
                | GgmlDType::IQ2S
                | GgmlDType::IQ3S
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ1M
                | GgmlDType::IQ4XS
        );
        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            if iq_type && dequant_cache_enabled() {
                use crate::backend::BackendStorage;
                let w = self.cached_dequant_f32(nrows * ncols)?;
                let weight_l = crate::Layout::new((ncols, nrows).into(), vec![1, ncols], 0)
                    .broadcast_as((b, ncols, nrows))?;
                rhs.matmul(w.as_ref(), (b, m, nrows, ncols), rhs_l, &weight_l)?
            } else {
                dequantize_mul_mat_vec_via_cublas(
                    &self.data,
                    rhs,
                    rhs_l,
                    self.dtype,
                    ncols,
                    nrows,
                    b,
                    m,
                    self.device(),
                )?
            }
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs_slice,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let is_iq = matches!(
            self.dtype,
            GgmlDType::IQ3XXS
                | GgmlDType::IQ2S
                | GgmlDType::IQ3S
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ1M
                | GgmlDType::IQ4XS
        );
        // Кэшированная полная деквантизация (A100 80GB): один matmul по кэшу
        // вместо tiled dequant каждый вызов.
        if is_iq && dequant_cache_enabled() && !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let w = self.cached_dequant_f32(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0)
                .broadcast_as((b, k, n))?;
            let out = storage.matmul(w.as_ref(), (b, m, n, k), layout, &rhs_l)?;
            let mut out_shape = layout.shape().dims().to_vec();
            out_shape.pop();
            out_shape.push(n);
            return Ok((out, out_shape.into()));
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed)
            || is_iq
            {
            // Tiled dequantize matmul: dequantize weight in row-chunks to avoid
            // allocating the full f32 weight (n*k*4 bytes) which can exceed VRAM
            // headroom during prefill and trigger CUDA unified-memory paging.
            // Each chunk: dequant [chunk_n, k] f32, cuBLAS matmul with activations
            // [b, m, k], scatter result into pre-allocated output [b, m, n].
            // Chunks write disjoint row ranges — no accumulation needed.
            let chunk_n = std::cmp::min(n, 512);
            let mut out_storage = CudaStorage::wrap_cuda_slice(
                unsafe { self.device.alloc::<f32>(b * m * n)? },
                self.device.clone(),
            );
            let mut row_start = 0;
            while row_start < n {
                let row_end = std::cmp::min(row_start + chunk_n, n);
                let chunk_rows = row_end - row_start;
                let data_f32 = dequantize_f32_rowslice(
                    &self.data,
                    self.dtype,
                    row_start,
                    row_end,
                    k,
                    self.device(),
                )?;
                let rhs_l = crate::Layout::new((k, chunk_rows).into(), vec![1, k], 0)
                    .broadcast_as((b, k, chunk_rows))?;
                let chunk_out =
                    storage.matmul(&data_f32, (b, m, chunk_rows, k), layout, &rhs_l)?;
                chunk_out.copy2d(
                    &mut out_storage,
                    /* d1 */ b * m,
                    /* d2 */ chunk_rows,
                    /* src_s */ chunk_rows,
                    /* dst_s */ n,
                    /* src_o */ 0,
                    /* dst_o */ row_start,
                )?;
                row_start = row_end;
            }
            out_storage
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
        dequant_cache: Default::default(),
    }))
}

/// Load raw quantized bytes (IQ-types) onto CUDA device.
/// Analog of metal::load_quantized_bytes — stores raw bytes in a PaddedCudaSlice,
/// keeps dtype metadata for kernel dispatch.
pub fn load_quantized_bytes(
    device: &CudaDevice,
    data: &[u8],
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
        dequant_cache: Default::default(),
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1.slice_mut(..), el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
