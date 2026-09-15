//! bench_proj — потолок проекций префилла на реальных формах Qwen3.5-4B.
//!
//! Меряет один matmul [M,K] × [N,K]ᵀ для путей, которые у нас есть, и считает
//! эффективные TOPS/TFLOPS. Заодно сравнивает 4 раздельные проекции DeltaNet
//! (qkv/z/b/a) с одной слитой — это то, что даёт наш формат и чего не даёт GGUF.
//!
//! Запуск: bench_proj [M] (умолчание 512 — размер чанка префилла).

use anyhow::Result;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};

const K: usize = 2560; // hidden_size
const ITERS: usize = 20;

fn timed(dev: &Device, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    dev.synchronize()?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.synchronize()?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

/// Отчёт: время и эффективная производительность (2*M*N*K флопов на matmul).
fn report(name: &str, m: usize, n: usize, ms: f64) {
    let flops = 2.0 * m as f64 * n as f64 * K as f64;
    println!(
        "  {name:<28} {ms:7.3} мс   {:6.1} T(FL)OPS",
        flops / (ms * 1e-3) / 1e12
    );
}

fn qmatmul(w_cpu: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<QMatMul> {
    Ok(QMatMul::from_qtensor(QTensor::quantize_onto(
        w_cpu, dtype, dev,
    )?)?)
}

fn main() -> Result<()> {
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    // Вторым аргументом — какую форму мерить. Замеры в одном процессе
    // отравляют друг друга (что меряется первым — быстрее в разы), поэтому
    // каждую форму гоняем отдельным запуском.
    let only = std::env::args().nth(2).unwrap_or_else(|| "all".to_string());
    let want = |name: &str| only == "all" || only == name;
    let dev = Device::new_cuda(0)?;
    println!("bench_proj: M={m} K={K} shape={only}, {ITERS} итераций\n");

    // Веса собираем на CPU: quantize_onto кладёт квант сразу в VRAM.
    let mk_w = |n: usize| -> Result<Tensor> {
        Ok(Tensor::randn(0f32, 0.02f32, (n, K), &Device::Cpu)?)
    };
    let x = Tensor::randn(0f32, 1f32, (1usize, m, K), &dev)?;

    // FFN (по профилю — самая крупная статья): gate/up [9216, 2560] и
    // down [2560, 9216]. K у down другой, поэтому у него свой вход.
    for (name, n, k) in [("ffn_gate", 9216usize, K), ("ffn_down", 2560, 9216)] {
        if !want(name) {
            continue;
        }
        println!("{name} [{n}, {k}]:");
        let w_cpu = Tensor::randn(0f32, 0.02f32, (n, k), &Device::Cpu)?;
        let xk = Tensor::randn(0f32, 1f32, (1usize, m, k), &dev)?;
        for (label, dtype) in [
            ("Q4_K MMQ (GGUF)", GgmlDType::Q4K),
            ("Q8_0 MMQ (сайдкар)", GgmlDType::Q8_0),
        ] {
            let q = QMatMul::from_qtensor(QTensor::quantize_onto(&w_cpu, dtype, &dev)?)?;
            let ms = timed(&dev, ITERS, || {
                q.forward(&xk)?;
                Ok(())
            })?;
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            println!("  {label:<24} {ms:7.3} мс   {:6.1} T(FL)OPS", flops / (ms * 1e-3) / 1e12);
        }
        println!();
    }

    // Проекции DeltaNet одного слоя: qkv, z, b, a + ssm_out.
    for (name, n) in [("qkv", 8192usize), ("z", 4096), ("b", 32), ("a", 32)] {
        if !want(name) {
            continue;
        }
        println!("{name} [{n}, {K}]:");
        let w_cpu = mk_w(n)?;
        for (label, dtype) in [("Q4_K MMQ (GGUF)", GgmlDType::Q4K), ("Q8_0 MMQ", GgmlDType::Q8_0)] {
            let q = qmatmul(&w_cpu, dtype, &dev)?;
            let ms = timed(&dev, ITERS, || {
                q.forward(&x)?;
                Ok(())
            })?;
            report(label, m, n, ms);
        }
        let w_f16 = w_cpu.to_device(&dev)?.to_dtype(DType::F16)?;
        let qm = QMatMul::TensorF16(w_f16);
        for (label, fast) in [("F16 GEMM (acc F32)", false), ("F16 GEMM (acc F16)", true)] {
            candle_core::cuda_backend::set_gemm_reduced_precision_f16(fast);
            let ms = timed(&dev, ITERS, || {
                qm.forward(&x)?;
                Ok(())
            })?;
            report(label, m, n, ms);
        }
        candle_core::cuda_backend::set_gemm_reduced_precision_f16(false);
        println!();
    }

    // Слитая проекция: qkv+z+b+a одним тензором — экономия запусков и один
    // проход по активациям вместо четырёх.
    let fused_n = 8192 + 4096 + 32 + 32;
    if !want("fused") && !want("fusedpad") && !want("fusedonly") && !want("split4") {
        return Ok(());
    }
    // Изолированные режимы: в процессе измеряется ТОЛЬКО одна конфигурация.
    // Комментарий ниже предупреждает, что замеры отравляют друг друга — и это
    // подтвердилось: одна и та же форма давала 0.429 мс внутри прогона all и
    // 0.044 мс отдельным процессом. Поэтому сравнение слитой и раздельных
    // проекций надо делать разными запусками.
    if want("fusedonly") || want("split4") {
        let iso_w = mk_w(fused_n)?;
        for (label, dtype) in [("Q4_K MMQ", GgmlDType::Q4K), ("Q8_0 MMQ", GgmlDType::Q8_0)] {
            if want("fusedonly") {
                let fused = qmatmul(&iso_w, dtype, &dev)?;
                let ms = timed(&dev, ITERS, || {
                    fused.forward(&x)?;
                    Ok(())
                })?;
                println!("ISO fused  {label:<10} M={m} {ms:7.3} мс");
            } else {
                let parts: Vec<QMatMul> = [8192usize, 4096, 32, 32]
                    .iter()
                    .map(|&n| qmatmul(&mk_w(n).unwrap(), dtype, &dev))
                    .collect::<Result<_>>()?;
                let ms = timed(&dev, ITERS, || {
                    for p in &parts {
                        p.forward(&x)?;
                    }
                    Ok(())
                })?;
                println!("ISO split4 {label:<10} M={m} {ms:7.3} мс");
            }
        }
        return Ok(());
    }
    // Сравнивать можно только пары, снятые подряд: в одном процессе замеры
    // отравляют друг друга (раздельные на одной и той же форме давали 11.7,
    // 10.1 и 7.5 мс в трёх блоках подряд). Поэтому каждая конфигурация —
    // отдельный запуск: `bench_proj 512 fused` и `bench_proj 512 fusedpad`.
    if want("fused") {
    println!("слитая qkv+z+b+a [{fused_n}, {K}] против четырёх раздельных:");
    let w_cpu = mk_w(fused_n)?;
    for (label, dtype) in [("Q4_K MMQ", GgmlDType::Q4K), ("Q8_0 MMQ", GgmlDType::Q8_0)] {
        let fused = qmatmul(&w_cpu, dtype, &dev)?;
        let ms_fused = timed(&dev, ITERS, || {
            fused.forward(&x)?;
            Ok(())
        })?;
        let parts: Vec<QMatMul> = [8192usize, 4096, 32, 32]
            .iter()
            .map(|&n| qmatmul(&mk_w(n).unwrap(), dtype, &dev))
            .collect::<Result<_>>()?;
        let ms_split = timed(&dev, ITERS, || {
            for p in &parts {
                p.forward(&x)?;
            }
            Ok(())
        })?;
        println!(
            "  {label:<12} слитая {ms_fused:7.3} мс | раздельные {ms_split:7.3} мс | выигрыш {:+.1}%",
            100.0 * (ms_split - ms_fused) / ms_split
        );
    }

    }
    if !want("fusedpad") {
        return Ok(());
    }
    // 12352 не кратно 128, и матмуль уходит с MMA-пути (гейт n % 128 == 0).
    // Дополняем до 12416 = 97*128: лишние строки выбрасываются при разрезании
    // выхода, зато вся группа считается тензорными ядрами.
    let padded_n = fused_n.div_ceil(128) * 128;
    println!("\nслитая с дополнением до кратности 128 [{padded_n}, {K}]:");
    let w_pad = mk_w(padded_n)?;
    for (label, dtype) in [("Q4_K MMQ", GgmlDType::Q4K), ("Q8_0 MMQ", GgmlDType::Q8_0)] {
        let q = qmatmul(&w_pad, dtype, &dev)?;
        let ms = timed(&dev, ITERS, || {
            q.forward(&x)?;
            Ok(())
        })?;
        let parts: Vec<QMatMul> = [8192usize, 4096, 32, 32]
            .iter()
            .map(|&n| qmatmul(&mk_w(n).unwrap(), dtype, &dev))
            .collect::<Result<_>>()?;
        let ms_split = timed(&dev, ITERS, || {
            for p in &parts {
                p.forward(&x)?;
            }
            Ok(())
        })?;
        println!(
            "  {label:<12} слитая {ms:7.3} мс | раздельные {ms_split:7.3} мс | выигрыш {:+.1}%",
            100.0 * (ms_split - ms) / ms_split
        );
    }

    println!("\nПик RTX 3060 (GA106): F16 c F32-акк ~25.6 TFLOPS, int8 ~51.2 TOPS.");
    Ok(())
}
