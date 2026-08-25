//! bench_delta — стенд под рекуррентное ядро DeltaNet (P3 префилла).
//!
//! По профилю nsys это 25% GPU-времени префилла и самое неэффективное место
//! стека: ~0.5 TFLOPS против 12.7 пиковых FP32. Стенд запускает ядро напрямую,
//! без сервера и без модели, чтобы варианты ядра проверялись за секунды.
//!
//! Запуск: bench_delta [T] [iters]. Печатает время, эффективные FLOPS и
//! max|diff| между v1 и v2 (математика обязана совпадать).

use anyhow::{anyhow, Result};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{CudaDevice, Device};
use qwen35_batch::real::delta_rule_cuda::DeltaParams;

const N_V: u32 = 32;
const HKD: u32 = 128;
const HVD: u32 = 128;

fn params() -> DeltaParams {
    DeltaParams {
        n_k_heads: 16,
        n_v_heads: N_V,
        head_k_dim: HKD,
        head_v_dim: HVD,
        key_dim: 16 * HKD,
        value_dim: N_V * HVD,
        channels: 16 * HKD * 2 + N_V * HVD,
        conv_kernel: 4,
        q_scale: 1.0,
        rms_norm_eps: 1e-6,
        heads_per_kv: N_V / 16,
    }
}

/// Детерминированный «шум» — одинаковый вход для всех вариантов ядра.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed as f32 * 0.001 + 0.1;
    (0..n)
        .map(|_| {
            x = (x * 7.13 + 0.37).fract() - 0.5;
            x * 0.5
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run(dev: &CudaDevice, kernel: &str, t: usize, iters: usize) -> Result<(f64, Vec<f32>)> {
    let p = params();
    let (n_v, hkd, hvd) = (N_V as usize, HKD as usize, HVD as usize);
    let q = dev.clone_htod(&fill(t * n_v * hkd, 1))?;
    let k = dev.clone_htod(&fill(t * n_v * hkd, 2))?;
    let v = dev.clone_htod(&fill(t * n_v * hvd, 3))?;
    let beta = dev.clone_htod(&fill(t * n_v, 4))?;
    let gate = dev.clone_htod(&fill(t * n_v, 5))?;
    let out = dev.alloc_zeros::<f32>(t * n_v * hvd)?;
    let state0 = fill(n_v * hvd * hvd, 6);

    let cfg = if kernel.ends_with("_v2") {
        LaunchConfig {
            grid_dim: (n_v as u32, (hvd / 32) as u32, 1),
            block_dim: (32, 32, 1),
            shared_mem_bytes: (2 * hkd * 4) as u32,
        }
    } else {
        LaunchConfig {
            grid_dim: (n_v as u32, (hvd / 2) as u32, 1),
            block_dim: (32, 2, 1),
            shared_mem_bytes: 0,
        }
    };
    let func = dev.get_or_load_func(kernel, &candle_kernels::DELTA_RULE)?;
    let t_u32 = t as u32;

    let mut launch = |state: &mut _| -> Result<()> {
        let mut b = func.builder();
        b.arg(&q);
        b.arg(&k);
        b.arg(&v);
        b.arg(&beta);
        b.arg(&gate);
        b.arg(&*state);
        b.arg(&out);
        b.arg(&p);
        b.arg(&t_u32);
        unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch {kernel}: {e:?}"))?;
        Ok(())
    };

    // Прогрев + эталонный выход (состояние каждый раз одно и то же).
    let mut state = dev.clone_htod(&state0)?;
    launch(&mut state)?;
    dev.synchronize()?;
    let reference = dev.clone_dtoh(&out)?;

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let mut state = dev.clone_htod(&state0)?;
        launch(&mut state)?;
    }
    dev.synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    Ok((ms, reference))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let t: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(512);
    let iters: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(20);
    let Device::Cuda(dev) = Device::new_cuda(0)? else {
        return Err(anyhow!("нужен CUDA-девайс"));
    };
    // FLOPs на запуск: на токен и голову — Sᵀk, обновление S и Sᵀq по hkd*hvd FMA.
    let flops = 3.0 * 2.0 * t as f64 * N_V as f64 * (HKD as f64) * (HVD as f64);
    println!("bench_delta: T={t} n_v={N_V} hkd={HKD} hvd={HVD}, {iters} итераций\n");

    let mut base: Option<Vec<f32>> = None;
    for kernel in ["delta_rule_prefill", "delta_rule_prefill_v2"] {
        match run(&dev, kernel, t, iters) {
            Ok((ms, outv)) => {
                let diff = base.as_ref().map(|b: &Vec<f32>| {
                    b.iter()
                        .zip(outv.iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max)
                });
                println!(
                    "  {kernel:<26} {ms:8.3} мс  {:6.2} TFLOPS{}",
                    flops / (ms * 1e-3) / 1e12,
                    match diff {
                        Some(d) => format!("   max|diff| к v1 = {d:.3e}"),
                        None => String::new(),
                    }
                );
                if base.is_none() {
                    base = Some(outv);
                }
            }
            Err(e) => println!("  {kernel:<26} ОШИБКА: {e}"),
        }
    }
    println!("\nПик RTX 3060: FP32 ~12.7 TFLOPS.");
    Ok(())
}
