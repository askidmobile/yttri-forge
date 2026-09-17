//! fa_decode_probe — пейдженный split-KV декод: token-major против head-major
//! раскладки KV-пула.
//!
//! Вопрос: сколько стоит то, что на фиксированную голову читается 512 Б
//! подряд с шагом 2 КБ (раскладка [page, token, head, dim]) против сплошного
//! чтения по голове ([page, head, token, dim]). Ядро FA2 принимает шаги
//! строками, поэтому обе раскладки проверяются БЕЗ пересборки flash-attn.
//!
//! Запуск: fa_decode_probe [kv_len] [iters]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const PAGE: usize = 64;

#[allow(clippy::too_many_arguments)]
fn run_paged(
    dev: &candle_core::CudaDevice,
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    out: &Tensor,
    lse: &Tensor,
    block_table: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    b: usize,
    h: usize,
    h_k: usize,
    d: usize,
    rows: usize,
    window: usize,
    page: usize,
    kv_row_stride: u32,
    kv_head_stride: u32,
    scale: f32,
) -> Result<()> {
    use qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr;
    let stream = dev.cuda_stream();
    unsafe {
        candle_flash_attn::ffi::run_mha(
            tensor_cuda_ptr(q)? as *const std::ffi::c_void,
            tensor_cuda_ptr(k_pool)? as *const std::ffi::c_void,
            tensor_cuda_ptr(v_pool)? as *const std::ffi::c_void,
            tensor_cuda_ptr(out)? as *const std::ffi::c_void,
            tensor_cuda_ptr(lse)? as *const std::ffi::c_void,
            std::ptr::null(),
            tensor_cuda_ptr(seqlens_q)? as *const i32,
            tensor_cuda_ptr(seqlens_k)? as *const i32,
            0,
            (page * h_k * d) as u32,
            (page * h_k * d) as u32,
            0,
            0,
            (h * d) as u32,
            kv_row_stride,
            kv_row_stride,
            (h * d) as u32,
            d as u32,
            kv_head_stride,
            kv_head_stride,
            d as u32,
            b as u32,
            h as u32,
            h_k as u32,
            d as u32,
            ((d + 31) / 32 * 32) as u32,
            scale,
            rows as u32,
            window as u32,
            ((rows + 127) / 128 * 128) as u32,
            ((window + 127) / 128 * 128) as u32,
            q.dim(0)? as u32,
            0,
            1, // is_causal (window_right = 0)
            1, // unpadded_lse
            window as i32,
            0,
            0.0,
            tensor_cuda_ptr(block_table)? as *const i32,
            block_table.dim(1)? as u32,
            page as i32,
            std::ptr::null(),
            0,
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            0,
            0,
            0,
            0, // kv_is_q8
            4, // rows_per_position = ngroups (свёртка GQA как в продовом декоде)
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kv_len: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(30232);
    let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
    let (h, h_k, d, rows) = (4usize, 4usize, 256usize, 4usize);
    let dev = Device::new_cuda(0)?;
    let cuda = dev.as_cuda_device()?.clone();
    let scale = (1.0 / (d as f64).sqrt()) as f32;
    let n_pages = kv_len.div_ceil(PAGE);
    let elems = n_pages * PAGE * h_k * d;

    println!(
        "fa_decode_probe: kv_len={kv_len} pages={n_pages} h={h} h_k={h_k} d={d} rows={rows} iters={iters}"
    );

    // A: [page, token, head, dim] — продовая: kv_row = h_k*hd, kv_head = hd
    // B: [page, head, token, dim] — head-major: kv_row = hd, kv_head = PAGE*hd
    let k_a = Tensor::randn(0f32, 1f32, (elems,), &dev)?.to_dtype(DType::F16)?;
    let v_a = Tensor::randn(0f32, 1f32, (elems,), &dev)?.to_dtype(DType::F16)?;
    let k_b = Tensor::randn(0f32, 1f32, (elems,), &dev)?.to_dtype(DType::F16)?;
    let v_b = Tensor::randn(0f32, 1f32, (elems,), &dev)?.to_dtype(DType::F16)?;

    let table: Vec<i32> = (0..n_pages as i32).collect();
    let block_table = Tensor::from_vec(table, (1, n_pages), &dev)?;
    let q = Tensor::randn(0f32, 1f32, (rows, h, d), &dev)?.to_dtype(DType::F16)?;
    let out = Tensor::zeros((rows, h, d), DType::F16, &dev)?;
    let lse = Tensor::zeros((h, rows), DType::F32, &dev)?;
    let sq = Tensor::from_vec(vec![0i32, rows as i32], (2,), &dev)?;
    let sk = Tensor::from_vec(vec![0i32, kv_len as i32], (2,), &dev)?;

    let mut bench = |name: &str,
                     k: &Tensor,
                     v: &Tensor,
                     kv_row: u32,
                     kv_head: u32|
     -> Result<()> {
        let mut best = f64::MAX;
        for it in 0..iters + 2 {
            cuda.cuda_stream().synchronize()?;
            let t0 = Instant::now();
            run_paged(
                &cuda, &q, k, v, &out, &lse, &block_table, &sq, &sk, 1, h, h_k, d, rows, kv_len,
                PAGE, kv_row, kv_head, scale,
            )?;
            cuda.cuda_stream().synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if it >= 2 {
                best = best.min(ms);
            }
        }
        let bytes = 2.0 * (kv_len as f64) * (h_k as f64) * (d as f64) * 2.0; // K+V, f16
        println!(
            "{name}: {best:.3} ms  {:.1} GB/s",
            bytes / (best * 1e-3) / 1e9
        );
        Ok(())
    };

    bench("token-major [page,token,head,dim] ", &k_a, &v_a, (h_k * d) as u32, d as u32)?;
    bench("head-major  [page,head,token,dim] ", &k_b, &v_b, d as u32, (PAGE * d) as u32)?;
    Ok(())
}
