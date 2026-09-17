//! fa_prefill_probe — плотный против пейдженного префильного flash-attention.
//!
//! Замер 2026-09-16 показал, что продовый префил идёт пейдженной веткой FA2
//! (`block_table != nullptr`): kBlockN=32, KV читается через таблицу страниц.
//! Профиль `[pfp-attn]` дал 320–430 мс на слой на чанке 16k — ~29 % времени
//! префила при 26 % занятости SM. Этот пробник отвечает на вопрос, сколько
//! именно стоит пейдженная ветка по сравнению с плотной на тех же формах.
//!
//! Запуск: fa_prefill_probe [seqlen] [iters]

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

const PAGE: usize = 64;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let t: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(16384);
    let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(5);
    let (h, h_k, d) = (16usize, 4usize, 256usize);
    let dev = Device::new_cuda(0)?;
    let cuda = dev.as_cuda_device()?.clone();
    let scale = (1.0 / (d as f64).sqrt()) as f32;

    println!("fa_prefill_probe: seqlen={t} h={h} h_k={h_k} d={d} iters={iters}");

    // Плотные Q/K/V: q [1, h, T, d], k/v [1, h_k, T, d] F16.
    // ВАЖНО: FA2 (и серверный fa_probe.rs) ждут [b, seq, heads, d]. Раньше здесь
    // были [1, h, t, d] и k/v [1, h_k, t, d] — ядро читало это как b=1, seq=h,
    // heads=t и «мерило» крошечную задачу (5 мс вместо 100).
    let q = Tensor::randn(0f32, 1f32, (1, h, t, d), &dev)?.to_dtype(DType::F16)?;
    let q_fa = q.transpose(1, 2)?.contiguous()?;
    let k_fa = Tensor::randn(0f32, 1f32, (1, t, h_k, d), &dev)?.to_dtype(DType::F16)?;
    let v_fa = Tensor::randn(0f32, 1f32, (1, t, h_k, d), &dev)?.to_dtype(DType::F16)?;

    // Пейдженный пул: [num_pages, PAGE, h_k, d] F16 + таблица страниц.
    let n_pages = t.div_ceil(PAGE);
    assert_eq!(t % PAGE, 0, "seqlen должен быть кратен странице пула");
    let k_pool = k_fa.reshape((n_pages, PAGE, h_k, d))?;
    let v_pool = v_fa.reshape((n_pages, PAGE, h_k, d))?;
    let table: Vec<i32> = (0..n_pages as i32).collect();
    let block_table = Tensor::from_vec(table, (1, n_pages), &dev)?;

    let seq_q = Tensor::from_vec(vec![0i32, t as i32], (2,), &dev)?;
    let seq_k = Tensor::from_vec(vec![0i32, t as i32], (2,), &dev)?;
    let lse = Tensor::zeros((h, t), DType::F32, &dev)?;
    let out_paged = Tensor::zeros((t, h, d), DType::F16, &dev)?;

    // --- плотная ветка: штатный API candle-flash-attn ---
    let mut dense_ms = f64::MAX;
    for it in 0..iters + 1 {
        let t0 = Instant::now();
        let out = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true)?;
        cuda.cuda_stream().synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&out);
        if it > 0 {
            dense_ms = dense_ms.min(ms);
        }
    }

    // --- пейдженная ветка: тот же run_mha, но с block_table ---
    let q_v = q.reshape((t, h, d))?;
    let q_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&q_v)?;
    let k_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&k_pool)?;
    let v_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&v_pool)?;
    let o_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&out_paged)?;
    let lse_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&lse)?;
    let bt_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&block_table)?;
    let sq_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&seq_q)?;
    let sk_ptr = qwen35_batch::real::paged_kv_cuda::tensor_cuda_ptr(&seq_k)?;

    let run_paged = || -> Result<()> {
        unsafe {
            candle_flash_attn::ffi::run_mha(
                q_ptr as *const std::ffi::c_void,
                k_ptr as *const std::ffi::c_void,
                v_ptr as *const std::ffi::c_void,
                o_ptr as *const std::ffi::c_void,
                lse_ptr as *const std::ffi::c_void,
                std::ptr::null(),
                sq_ptr as *const i32,
                sk_ptr as *const i32,
                0,
                (PAGE * h_k * d) as u32,
                (PAGE * h_k * d) as u32,
                0,
                0,
                (h * d) as u32,
                (h_k * d) as u32,
                (h_k * d) as u32,
                (h * d) as u32,
                d as u32,
                d as u32,
                d as u32,
                d as u32,
                1,
                h as u32,
                h_k as u32,
                d as u32,
                ((d + 31) / 32 * 32) as u32,
                scale,
                t as u32,
                t as u32,
                ((t + 127) / 128 * 128) as u32,
                ((t + 127) / 128 * 128) as u32,
                t as u32,
                0,
                1,
                1,
                -1,
                0,
                0.0,
                bt_ptr as *const i32,
                0,
                PAGE as i32,
                std::ptr::null(),
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                0,
                std::ptr::null(),
                std::ptr::null(),
                cuda.cuda_stream().cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    };

    let mut paged_ms = f64::MAX;
    for it in 0..iters + 1 {
        cuda.cuda_stream().synchronize()?;
        let t0 = Instant::now();
        run_paged()?;
        cuda.cuda_stream().synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if it > 0 {
            paged_ms = paged_ms.min(ms);
        }
    }

    // Проверка, что ветки вообще считают одно и то же (средняя |разница|).
    let dense_out = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true)?
        .reshape((t, h, d))?
        .to_dtype(DType::F32)?;
    let paged_out = out_paged.to_dtype(DType::F32)?;
    let diff = (&dense_out - &paged_out)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    println!("mean|dense-paged| = {diff:.5}");

    // ── Численная сверка с референсом в f32 (matmul + softmax) на малом T ──
    // Нужна потому, что плотная и пейдженная ветки выше считают одним и тем же
    // ядром, и их сравнение не ловит ошибку раскладки. Здесь референс считается
    // обычными операциями candle в f32.
    let t_ref: usize = std::env::var("FA_REF_T").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    if t_ref >= 16 {
        let qr = Tensor::randn(0f32, 1f32, (1, h, t_ref, d), &dev)?.to_dtype(DType::F16)?;
        let kr = Tensor::randn(0f32, 1f32, (1, t_ref, h_k, d), &dev)?.to_dtype(DType::F16)?;
        let vr = Tensor::randn(0f32, 1f32, (1, t_ref, h_k, d), &dev)?.to_dtype(DType::F16)?;
        let qr_fa = qr.transpose(1, 2)?.contiguous()?;
        let out_k = candle_flash_attn::flash_attn(&qr_fa, &kr, &vr, scale, true)?.to_dtype(DType::F32)?;

        let reps = h / h_k;
        let k_f = kr.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // [1,h_k,t,d]
        let v_f = vr.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let k_exp = k_f
            .unsqueeze(2)?
            .broadcast_as((1, h_k, reps, t_ref, d))?
            .contiguous()?
            .reshape((1, h, t_ref, d))?;
        let v_exp = v_f
            .unsqueeze(2)?
            .broadcast_as((1, h_k, reps, t_ref, d))?
            .contiguous()?
            .reshape((1, h, t_ref, d))?;
        let qf = qr.to_dtype(DType::F32)?;
        let scores = (qf.matmul(&k_exp.transpose(2, 3)?.contiguous()?)? * scale as f64)?;
        let mask = Tensor::tril2(t_ref, DType::F32, &dev)?;
        let mask_add = ((mask - 1.0)? * 1e9)?;
        let scores = scores.broadcast_add(&mask_add.unsqueeze(0)?.unsqueeze(0)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ref_out = attn.matmul(&v_exp)?.transpose(1, 2)?.contiguous()?; // [1,t,h,d]
        let diff = (out_k - ref_out.clone())?.abs()?;
        let mean = diff.mean_all()?.to_scalar::<f32>()?;
        let max = diff.max_all()?.to_scalar::<f32>()?;
        let ref_abs = ref_out.abs()?.mean_all()?.to_scalar::<f32>()?;
        println!(
            "точность (T={t_ref}, F16-ядро против f32-референса): mean|Δ|={mean:.5} max|Δ|={max:.4} при mean|ref|={ref_abs:.3}"
        );
    }

    println!("dense  : {dense_ms:.1} ms");
    println!("paged  : {paged_ms:.1} ms");
    println!("paged/dense = {:.2}x", paged_ms / dense_ms);
    Ok(())
}
