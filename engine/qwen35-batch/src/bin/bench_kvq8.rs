//! bench_kvq8 — точность int8 paged KV.
//!
//! Прогоняет случайные K/V через квантующую запись в пул и распаковку обратно,
//! считает ошибку. Ожидание для симметричного int8 с масштабом на 256 значений
//! — порядка 0.2-0.5% по среднему модулю; десятки процентов означают ошибку в
//! адресации или в редукции максимума.
//!
//! Запуск: bench_kvq8 [tokens]

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use qwen35_batch::real::paged_kv_cuda::{tensor_cuda_ptr, PAGE_SIZE};

fn ptr(t: &Tensor) -> Result<u64> {
    Ok(tensor_cuda_ptr(t)?)
}

fn main() -> Result<()> {
    let tokens: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let (n_kv, hd) = (4usize, 256usize);
    let dev = Device::new_cuda(0)?;
    let cuda = dev.as_cuda_device()?.clone();
    let blocks = tokens.div_ceil(PAGE_SIZE);

    // Пул: int8 хранится в U8-тензоре (тот же байт, знак трактует ядро).
    let pool_elems = blocks * PAGE_SIZE * n_kv * hd;
    let k_pool = Tensor::zeros(pool_elems, DType::U8, &dev)?;
    let v_pool = Tensor::zeros(pool_elems, DType::U8, &dev)?;
    let k_scale = Tensor::zeros(blocks * PAGE_SIZE * n_kv, DType::F16, &dev)?;
    let v_scale = Tensor::zeros(blocks * PAGE_SIZE * n_kv, DType::F16, &dev)?;

    // Исходные строки: [1, T, n_kv, hd]. Нормальное распределение — как у
    // реальных K/V после нормировки.
    let k_src = Tensor::randn(0f32, 1f32, (1, tokens, n_kv, hd), &dev)?.to_dtype(DType::F16)?;
    let v_src = Tensor::randn(0f32, 1f32, (1, tokens, n_kv, hd), &dev)?.to_dtype(DType::F16)?;

    let bt: Vec<u32> = (0..blocks as u32).collect();
    let block_table = Tensor::from_vec(bt, blocks, &dev)?;
    let slots = Tensor::from_vec(vec![0u32], 1, &dev)?;
    let kv_len = Tensor::from_vec(vec![0u32], 1, &dev)?;

    let f = cuda.get_or_load_func(
        "kv_append_paged_q8_multi",
        &candle_core::cuda_backend::kernels::QUANTIZED,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (n_kv as u32, 1, tokens as u32),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let (b_i, t_i, nkv_i, hd_i, ps_i, mb_i, win_i) = (
        1i32,
        tokens as i32,
        n_kv as i32,
        hd as i32,
        PAGE_SIZE as i32,
        blocks as i32,
        tokens as i32,
    );
    let mut bld = f.builder();
    let (kp, vp, ksc, vsc) = (ptr(&k_pool)?, ptr(&v_pool)?, ptr(&k_scale)?, ptr(&v_scale)?);
    let (ks, vs) = (ptr(&k_src)?, ptr(&v_src)?);
    let (btp, sp, klp) = (ptr(&block_table)?, ptr(&slots)?, ptr(&kv_len)?);
    bld.arg(&kp); bld.arg(&vp); bld.arg(&ksc); bld.arg(&vsc);
    bld.arg(&ks); bld.arg(&vs);
    bld.arg(&btp); bld.arg(&sp); bld.arg(&klp);
    bld.arg(&b_i); bld.arg(&t_i); bld.arg(&nkv_i); bld.arg(&hd_i);
    bld.arg(&ps_i); bld.arg(&mb_i); bld.arg(&win_i);
    unsafe { bld.launch(cfg) }.map_err(candle_core::Error::wrap)?;

    // Распаковка обратно
    let out = Tensor::zeros(tokens * n_kv * hd, DType::F16, &dev)?;
    let g = cuda.get_or_load_func(
        "kv_pool_dequant_q8",
        &candle_core::cuda_backend::kernels::QUANTIZED,
    )?;
    let cfg2 = LaunchConfig {
        grid_dim: (tokens as u32, n_kv as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b2 = g.builder();
    let (op, kp2, ksc2, btp2) = (ptr(&out)?, ptr(&k_pool)?, ptr(&k_scale)?, ptr(&block_table)?);
    let n_i = tokens as i32;
    b2.arg(&kp2); b2.arg(&ksc2); b2.arg(&op); b2.arg(&btp2);
    b2.arg(&n_i); b2.arg(&nkv_i); b2.arg(&hd_i); b2.arg(&ps_i);
    unsafe { b2.launch(cfg2) }.map_err(candle_core::Error::wrap)?;
    dev.synchronize()?;

    let src: Vec<f32> = k_src.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let got: Vec<f32> = out.to_dtype(DType::F32)?.to_vec1()?;
    if src.len() != got.len() {
        bail!("длины не совпали: {} против {}", src.len(), got.len());
    }
    let mut num = 0f64;
    let mut den = 0f64;
    let mut worst = 0f64;
    let mut nonfinite = 0usize;
    for (a, b) in got.iter().zip(src.iter()) {
        if !a.is_finite() {
            nonfinite += 1;
            continue;
        }
        let d = (*a as f64 - *b as f64).abs();
        num += d;
        den += (*b as f64).abs();
        worst = worst.max(d);
    }
    if nonfinite > 0 {
        bail!("нефинитных значений после распаковки: {nonfinite}");
    }
    println!(
        "токенов {tokens}, элементов {}: относительная ошибка {:.3}%, худшее отклонение {:.5}",
        src.len(),
        100.0 * num / den.max(1e-12),
        worst
    );
    let rel = 100.0 * num / den.max(1e-12);
    if rel > 2.0 {
        bail!("ошибка {rel:.2}% слишком велика для int8 — ищи адресацию или редукцию");
    }
    Ok(())
}
