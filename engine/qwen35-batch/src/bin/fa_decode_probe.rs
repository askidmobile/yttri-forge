//! fa_decode_probe — декодное постраничное внимание: цена int8-пула (де-квант)
//! и GQA-свёртки.
//!
//! Два вопроса:
//!  * сколько стоит int8-пул против F16 при той же длине KV. Де-квант K/V
//!    делается в smem каждым CTA, а CTA — на голову запроса: без свёртки одну
//!    kv-голову читают `h/h_k` CTA (4 на этой модели), то есть де-квант
//!    повторяется 4 раза;
//!  * сколько даёт свёртка (`rows_per_position`): строки всех групп одной
//!    kv-головы обрабатывает один CTA, KV читается один раз.
//!
//! Запуск: fa_decode_probe [kv_len] [iters] [splits]
//!   splits = 0 — автоэвристика ядра, иначе QWEN36_FA_SPLITS.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const PAGE: usize = 64;
const H: usize = 16;
const H_K: usize = 4;
const D: usize = 256;

#[allow(clippy::too_many_arguments)]
fn run_paged(
    dev: &candle_core::CudaDevice,
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    scales: Option<(&Tensor, &Tensor)>,
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
    scale: f32,
    rows_per_position: usize,
    window_right: i32,
) -> Result<()> {
    use qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr;
    let (k_scale_ptr, v_scale_ptr, scale_batch_stride, scale_row_stride, kv_is_q8) = match scales {
        Some((ks, vs)) => (
            tensor_cuda_ptr(ks)? as *const std::ffi::c_void,
            tensor_cuda_ptr(vs)? as *const std::ffi::c_void,
            (page * h_k) as u32,
            h_k as u32,
            1,
        ),
        None => (
            std::ptr::null(),
            std::ptr::null(),
            0u32,
            0u32,
            0,
        ),
    };
    let stream = dev.cuda_stream();
    let kv_row_stride = (h_k * d) as u32;
    let kv_head_stride = d as u32;
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
            if window_right == 0 { 1 } else { 0 }, // is_causal
            1, // unpadded_lse
            if window_right == 0 { window as i32 } else { -1 },
            window_right,
            0.0,
            tensor_cuda_ptr(block_table)? as *const i32,
            block_table.dim(1)? as u32,
            page as i32,
            std::ptr::null(),
            0,
            0,
            k_scale_ptr,
            v_scale_ptr,
            scale_batch_stride,
            scale_row_stride,
            scale_batch_stride,
            scale_row_stride,
            kv_is_q8,
            rows_per_position as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

/// Квантует f16-пул [n_tokens, h_k, d] в int8 + масштабы [n_tokens, h_k].
fn quantize_q8(flat: &Tensor, n_tokens: usize) -> Result<(Tensor, Tensor)> {
    let dev = flat.device();
    let vals = flat.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut quants = vec![0u8; vals.len()];
    let mut scales = vec![0f32; n_tokens * H_K];
    let rows: Vec<(usize, usize, usize)> = (0..n_tokens * H_K)
        .map(|i| {
            let base = i * D;
            (i, base, base + D)
        })
        .collect();
    use rayon::prelude::*;
    let out: Vec<(Vec<u8>, f32)> = rows
        .par_iter()
        .map(|&(row, lo, hi)| {
            let mut mx = 0f32;
            for v in &vals[lo..hi] {
                let a = v.abs();
                if a > mx {
                    mx = a;
                }
            }
            let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            let mut q = vec![0u8; D];
            for (i, v) in vals[lo..hi].iter().enumerate() {
                let r = (v / s).round().clamp(-127.0, 127.0);
                q[i] = (r as i8) as u8;
            }
            let _ = row;
            (q, s)
        })
        .collect();
    for (i, (q, s)) in out.iter().enumerate() {
        quants[i * D..(i + 1) * D].copy_from_slice(q);
        scales[i] = *s;
    }
    let dev = dev.as_cuda_device()?.clone();
    let q_t = Tensor::from_vec(quants, vals.len(), &Device::Cuda(dev.clone()))?;
    let s_t = Tensor::from_vec(scales, n_tokens * H_K, &Device::Cuda(dev))?
        .to_dtype(DType::F16)?;
    Ok((q_t, s_t))
}

struct Bench {
    name: &'static str,
    kv_len: usize,
    elem_bytes: f64,
    ms: f64,
    out: Tensor,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let kv_len: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(30232);
    let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
    let splits: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(0);
    // 1 = продовый режим: масок нет (window_left/right = -1), поэтому в ядре
    // включается GQA-свёртка (обмен q_row/q_head) — именно так идёт декод.
    let prod_mode: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0);
    let window_right: i32 = if prod_mode == 1 { -1 } else { 0 };
    if splits > 0 {
        std::env::set_var("QWEN36_FA_SPLITS", splits.to_string());
    }
    let dev = Device::new_cuda(0)?;
    let cuda = dev.as_cuda_device()?.clone();
    let scale = (1.0 / (D as f64).sqrt()) as f32;
    let n_pages = kv_len.div_ceil(PAGE);
    let n_tokens = n_pages * PAGE;
    let ngroups = H / H_K;

    println!(
        "fa_decode_probe: kv_len={kv_len} pages={n_pages} h={H} h_k={H_K} d={D} \
         ngroups={ngroups} iters={iters} splits(pin)={splits} prod_mode={prod_mode}"
    );

    // Пул: [n_tokens, h_k, d] F16 (страницы подряд).
    let k_a = Tensor::randn(0f32, 1f32, (n_tokens * H_K * D,), &dev)?.to_dtype(DType::F16)?;
    let v_a = Tensor::randn(0f32, 1f32, (n_tokens * H_K * D,), &dev)?.to_dtype(DType::F16)?;

    // int8-пул: те же значения после квантования (сравнение честное: тот же вход).
    let (k_q, k_s) = quantize_q8(&k_a, n_tokens)?;
    let (v_q, v_s) = quantize_q8(&v_a, n_tokens)?;

    let table: Vec<i32> = (0..n_pages as i32).collect();
    let block_table = Tensor::from_vec(table, (1, n_pages), &dev)?;

    // q: [H, D] на одну позицию. Свёртка: [ngroups, h_k, D],
    // строка r = группа, ось головы = kv-голова (раскладка как в verify_onepass).
    let q_base = Tensor::randn(0f32, 1f32, (H, D), &dev)?.to_dtype(DType::F16)?;
    let q_fold = q_base
        .reshape((1usize, H_K, ngroups, D))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((ngroups, H_K, D))?;

    let mut results: Vec<Bench> = Vec::new();
    for &(q8, fold) in &[(false, false), (true, false), (false, true), (true, true)] {
        let (h, rows, rpp, q_t) = if fold {
            (H_K, ngroups, ngroups, q_fold.clone())
        } else {
            (H, 1usize, 0usize, q_base.reshape((1usize, H, D))?)
        };
        let out = Tensor::zeros((rows, h, D), DType::F16, &dev)?;
        let lse = Tensor::zeros((h, rows), DType::F32, &dev)?;
        let sq = Tensor::from_vec(vec![0i32, rows as i32], (2,), &dev)?;
        let sk = Tensor::from_vec(vec![0i32, kv_len as i32], (2,), &dev)?;
        let (k_pool, v_pool, scales) = if q8 {
            (&k_q, &v_q, Some((&k_s, &v_s)))
        } else {
            (&k_a, &v_a, None)
        };
        let mut best = f64::MAX;
        let total = iters + 3;
        for it in 0..total {
            cuda.cuda_stream().synchronize()?;
            let t0 = Instant::now();
            run_paged(
                &cuda, &q_t, k_pool, v_pool, scales, &out, &lse, &block_table, &sq, &sk, 1, h,
                H_K, D, rows, kv_len, PAGE, scale, rpp, window_right,
            )?;
            cuda.cuda_stream().synchronize()?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if it >= 3 {
                best = best.min(ms);
            }
        }
        let name = match (q8, fold) {
            (false, false) => "f16 no-fold",
            (true, false) => "q8  no-fold",
            (false, true) => "f16 fold   ",
            (true, true) => "q8  fold   ",
            _ => "?",
        };
        let elem_bytes = if q8 { 1.0 } else { 2.0 };
        results.push(Bench {
            name,
            kv_len,
            elem_bytes,
            ms: best,
            out: out.clone(),
        });
        println!(
            "{name}: {best:8.4} ms   KV {:.0} MB/вызов ({:.1} GB/s)",
            (kv_len * H_K * D) as f64 * elem_bytes * 2.0 / 1e6,
            (kv_len * H_K * D) as f64 * elem_bytes * 2.0 / (best * 1e-3) / 1e9
        );
    }

    // Сверка свёртки против построчного прохода (та же точность, иная раскладка).
    let un_fold = |t: &Tensor| -> Result<Tensor> {
        Ok(t.reshape((ngroups, H_K, D))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((1, H, D))?
            .to_dtype(DType::F32)?)
    };
    // Сверка типов пула: int8-пул — квантованная копия тех же данных, значит
    // выход должен совпадать с f16-путём с точностью до ошибки квантования.
    // Расхождение в разы = ядро читает байты как half (или наоборот).
    let out_f16 = results[0].out.to_dtype(DType::F32)?.reshape((1, H, D))?;
    let out_q8 = results[1].out.to_dtype(DType::F32)?.reshape((1, H, D))?;
    let d_q = (&out_q8 - &out_f16)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let mag = out_f16.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    println!("cross-check q8 vs f16 (no-fold): max|Δ|={d_q:.5} (|out|max={mag:.4}, rel={:.2e})", d_q / mag.max(1e-9));

    for (i, j) in [(0usize, 2usize), (1, 3)] {
        let a = results[j].out.clone();
        let b = results[i].out.to_dtype(DType::F32)?;
        let a = un_fold(&a)?;
        let b = b.reshape((1, H, D))?;
        let diff = (&a - &b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mag = a.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!(
            "diff {} vs {}: max|Δ|={:.5} (|out|max={:.3})",
            results[j].name.trim(),
            results[i].name.trim(),
            diff,
            mag
        );
    }
    Ok(())
}
