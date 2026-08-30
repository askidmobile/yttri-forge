//! Микробенчмарк повторного использования блока весов в CUDA MMVQ.
//!
//! Запуск:
//! `cargo run --release --features cuda --example mmvq_bench -- <dtype> <nrows> <ncols> [batch]`

#![cfg(feature = "cuda")]

use anyhow::{bail, Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use std::time::Instant;

const DEFAULT_NROWS: usize = 248_320;
const DEFAULT_NCOLS: usize = 4_096;
const WARMUP_RUNS: usize = 20;
const MEASURED_RUNS: usize = 200;

fn measured_runs() -> Result<usize> {
    match std::env::var("MMVQ_BENCH_RUNS") {
        Ok(value) => value
            .parse::<usize>()
            .context("MMVQ_BENCH_RUNS должен быть положительным целым")
            .and_then(|runs| {
                if runs > 0 {
                    Ok(runs)
                } else {
                    bail!("MMVQ_BENCH_RUNS должен быть больше нуля")
                }
            }),
        Err(_) => Ok(MEASURED_RUNS),
    }
}

fn parse_dtype(value: Option<&str>) -> Result<GgmlDType> {
    match value.unwrap_or("q4_k") {
        "q2_k" => Ok(GgmlDType::Q2K),
        "q4_k" => Ok(GgmlDType::Q4K),
        "q6_k" => Ok(GgmlDType::Q6K),
        "q8_0" => Ok(GgmlDType::Q8_0),
        value => bail!("неизвестный тип кванта {value:?}; ожидается q2_k, q4_k, q6_k или q8_0"),
    }
}

fn parse_positive(value: Option<&str>, default: usize, name: &str) -> Result<usize> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{name} должен быть положительным целым"))?;
    if parsed == 0 {
        bail!("{name} должен быть больше нуля")
    }
    Ok(parsed)
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let dtype = parse_dtype(args.first().map(String::as_str))?;
    let nrows = parse_positive(args.get(1).map(String::as_str), DEFAULT_NROWS, "nrows")?;
    let ncols = parse_positive(args.get(2).map(String::as_str), DEFAULT_NCOLS, "ncols")?;
    let requested_batch = args
        .get(3)
        .map(|value| parse_positive(Some(value), 0, "batch"))
        .transpose()?;
    if requested_batch.is_some_and(|batch| !(1..=8).contains(&batch)) {
        bail!("batch должен находиться в диапазоне 1..=8")
    }
    if ncols % 256 != 0 {
        bail!("ncols должен быть кратен 256 для K-квантов")
    }
    let measured_runs = measured_runs()?;
    let device = Device::new_cuda(0)?;
    let dev = device.as_cuda_device()?;

    println!("Создание случайных весов [{nrows}, {ncols}] и квантование в {dtype:?}...");
    let weights = Tensor::randn(0f32, 1f32, (nrows, ncols), &device)?;
    let weights = QTensor::quantize(&weights, dtype)?;
    let matmul = QMatMul::from_qtensor(weights)?;
    println!("Замер: {WARMUP_RUNS} прогревочных, {measured_runs} измеряемых запусков");

    let batches = requested_batch.map_or_else(|| (1..=4).collect(), |batch| vec![batch]);
    let mut previous_ms = None;
    let mut inputs = Vec::with_capacity(batches.len());
    for &batch in &batches {
        let input = Tensor::randn(0f32, 1f32, (batch, ncols), &device)?;
        inputs.push((batch, input.clone()));

        for _ in 0..WARMUP_RUNS {
            std::hint::black_box(matmul.forward(&input)?);
        }
        dev.cuda_stream()
            .synchronize()
            .context("не удалось синхронизировать CUDA-стрим перед замером")?;
        let started = Instant::now();
        for _ in 0..measured_runs {
            std::hint::black_box(matmul.forward(&input)?);
        }
        dev.cuda_stream()
            .synchronize()
            .context("не удалось синхронизировать CUDA-стрим после замера")?;

        let ms = started.elapsed().as_secs_f64() * 1_000.0 / measured_runs as f64;
        match previous_ms {
            Some(previous) => println!(
                "пакет {batch}: {ms:.3} мс/запуск, лишняя строка: {:+.3} мс",
                ms - previous
            ),
            None => println!("пакет {batch}: {ms:.3} мс/запуск"),
        }
        previous_ms = Some(ms);
    }

    if matches!(dtype, GgmlDType::Q2K | GgmlDType::Q4K | GgmlDType::Q6K) {
        let original_setting = std::env::var_os("QWEN36_MMVQ_HOISTED");
        for (batch, input) in &inputs {
            std::env::set_var("QWEN36_MMVQ_HOISTED", "0");
            let baseline = matmul.forward(input)?.flatten_all()?.to_vec1::<f32>()?;
            std::env::set_var("QWEN36_MMVQ_HOISTED", "all");
            let hoisted = matmul.forward(input)?.flatten_all()?.to_vec1::<f32>()?;

            let bitwise_equal = baseline
                .iter()
                .zip(&hoisted)
                .all(|(left, right)| left.to_bits() == right.to_bits());
            let max_abs_diff = baseline
                .iter()
                .zip(&hoisted)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            println!(
                "сверка {dtype:?} пакет {}: побитово={}, max_abs_diff={max_abs_diff:e}",
                batch, bitwise_equal
            );
        }
        match original_setting {
            Some(value) => std::env::set_var("QWEN36_MMVQ_HOISTED", value),
            None => std::env::remove_var("QWEN36_MMVQ_HOISTED"),
        }
    }

    Ok(())
}
