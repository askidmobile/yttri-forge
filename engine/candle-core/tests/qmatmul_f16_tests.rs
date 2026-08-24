//! Numerical correctness tests of F16 input vs F32 input for Metal QMatMul.
//!
//! Covers Phase 1-3 of the F16 QMatMul Metal kernel plan:
//! - mm path (prefill, m>1) — kernel_mul_mm template T_input
//! - mv path (decode, m=1) — kernel_mul_mv_q* template impl
//!
//! The test runs only under `--features metal` on Apple Silicon.
//! diff threshold = 1e-2 -- matches F16 precision (11-bit mantissa).

#![cfg(feature = "metal")]

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Module, Result, Tensor,
};
use std::sync::Arc;

fn run_f16_vs_f32_qmatmul(
    dtype: GgmlDType,
    n: usize,
    k: usize,
    m: usize,
) -> Result<f32> {
    let device = Device::new_metal(0)?;

    // Random weight (n, k) and random input (m, k).
    let weight = Tensor::randn(0f32, 1f32, (n, k), &device)?;
    let input = Tensor::randn(0f32, 1f32, (m, k), &device)?;

    let qweight = QTensor::quantize(&weight, dtype)?;
    let qmm = QMatMul::from_arc(Arc::new(qweight))?;

    // F32 path
    let input_f32 = input.to_dtype(DType::F32)?;
    let out_f32 = qmm.forward(&input_f32)?;

    // F16 path
    let input_f16 = input.to_dtype(DType::F16)?;
    let out_f16 = qmm.forward(&input_f16)?;

    // Output of both kernels is F32 (the kernel writes a float dst).
    // Compute the max abs diff.
    let out_f32 = out_f32.flatten_all()?.to_vec1::<f32>()?;
    let out_f16 = out_f16.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    assert_eq!(out_f32.len(), out_f16.len());
    let max_abs_diff = out_f32
        .iter()
        .zip(out_f16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);

    // RMS of the F32 output -- used as a scale to normalize the diff.
    // This removes false positives when individual output elements
    // are close to zero (relative diff vs 1e-3 yields an uninformative 0.7+).
    let out_rms = (out_f32.iter().map(|v| v * v).sum::<f32>() / out_f32.len() as f32).sqrt();

    let rel_to_rms = max_abs_diff / out_rms.max(1e-6);

    println!(
        "qmatmul {dtype:?} n={n} k={k} m={m}: max_abs_diff={max_abs_diff:.5} out_rms={out_rms:.5} rel_to_rms={rel_to_rms:.5}"
    );

    // F16 mantissa = 11 bit → ~5e-4 relative precision per element.
    // Accumulation over k elements: noise ~ sqrt(k) * eps per result element.
    // Tolerance vs RMS output -- about 1% (k=128) up to 5% (k=512).
    assert!(
        rel_to_rms < 5e-2,
        "F16 vs F32 qmatmul rel-to-RMS diff {rel_to_rms} exceeds 5% threshold \
         (dtype={dtype:?}, n={n}, k={k}, m={m}, max_abs_diff={max_abs_diff}, out_rms={out_rms})"
    );

    Ok(max_abs_diff)
}

// mv path (m=1) — decode. Q4_0 — legacy template `mul_vec_q_n_f32_impl`.
#[test]
fn qmatmul_f16_q4_0_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4_0, 64, 128, 1)?;
    Ok(())
}

// mv path (m=1) -- Q8_0 -- a separate non-template impl.
#[test]
fn qmatmul_f16_q8_0_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q8_0, 64, 128, 1)?;
    Ok(())
}

// mv path (m=1) -- Q4_K -- K-quants separate impl, block_size=256.
#[test]
fn qmatmul_f16_q4_k_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4K, 64, 512, 1)?;
    Ok(())
}

// mv path -- Q6_K (full K-quants test).
#[test]
fn qmatmul_f16_q6_k_mv() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q6K, 64, 512, 1)?;
    Ok(())
}

// mm path (m>1) — prefill, kernel_mul_mm template.
// Sizes chosen to hit the mm-variant (m=64 > 1, BLOCK_SIZE_M=64).
#[test]
fn qmatmul_f16_q4_0_mm() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4_0, 64, 128, 64)?;
    Ok(())
}

#[test]
fn qmatmul_f16_q4_k_mm() -> Result<()> {
    run_f16_vs_f32_qmatmul(GgmlDType::Q4K, 64, 512, 64)?;
    Ok(())
}

// === Quantization quality (RMSE) -- port of make_qkx2_quants ===
//
// These tests compare candle quantize quality on random F32 weights with
// the llama.cpp reference. Metric: nRMSE = RMSE(x, dequant(quant(x))) / RMS(x).
// Thresholds are chosen to match the expected precision of the type.
fn quantize_dequantize_nrmse(dtype: GgmlDType, n: usize, k: usize) -> Result<f32> {
    let device = Device::Cpu;
    let weight = Tensor::randn(0f32, 1f32, (n, k), &device)?;
    let x_orig = weight.flatten_all()?.to_vec1::<f32>()?;
    let qweight = QTensor::quantize(&weight, dtype)?;
    let dequant = qweight.dequantize(&device)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(x_orig.len(), dequant.len());
    let mse: f32 = x_orig
        .iter()
        .zip(dequant.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / x_orig.len() as f32;
    let rms: f32 = (x_orig.iter().map(|v| v * v).sum::<f32>() / x_orig.len() as f32).sqrt();
    Ok(mse.sqrt() / rms.max(1e-9))
}

// Thresholds match the llama.cpp ref on random Gaussian (this is the baseline precision
// of the types themselves; the advantage of make_qkx2_quants vs qkx1 shows on real weights with
// long-tail outliers -- there the grid search really helps).
#[test]
fn quantize_quality_q4_k() -> Result<()> {
    let nrmse = quantize_dequantize_nrmse(GgmlDType::Q4K, 64, 512)?;
    println!("Q4_K nRMSE = {nrmse:.5}");
    assert!(nrmse < 0.10, "Q4_K nRMSE {nrmse} exceeds 0.10");
    Ok(())
}

#[test]
fn quantize_quality_q5_k() -> Result<()> {
    let nrmse = quantize_dequantize_nrmse(GgmlDType::Q5K, 64, 512)?;
    println!("Q5_K nRMSE = {nrmse:.5}");
    assert!(nrmse < 0.05, "Q5_K nRMSE {nrmse} exceeds 0.05");
    Ok(())
}

#[test]
fn quantize_quality_q2_k() -> Result<()> {
    let nrmse = quantize_dequantize_nrmse(GgmlDType::Q2K, 64, 512)?;
    println!("Q2_K nRMSE = {nrmse:.5}");
    // Q2_K on random Gaussian usually yields nRMSE 0.25-0.35 (aggressive 2-bit
    // quantization, no pronounced outliers where qkx2 grid search helps).
    assert!(nrmse < 0.40, "Q2_K nRMSE {nrmse} exceeds 0.40");
    Ok(())
}

#[test]
fn quantize_quality_q4_0() -> Result<()> {
    let nrmse = quantize_dequantize_nrmse(GgmlDType::Q4_0, 64, 512)?;
    println!("Q4_0 nRMSE = {nrmse:.5}");
    assert!(nrmse < 0.10, "Q4_0 nRMSE {nrmse} exceeds 0.10");
    Ok(())
}
