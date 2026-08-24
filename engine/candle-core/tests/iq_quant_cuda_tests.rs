//! Isolated CUDA tests for IQ quant types (IQ2XXS, IQ3XXS, IQ2S, IQ3S, IQ2XS, IQ4XS).
//!
//! These types cannot use `QTensor::quantize` (CPU `from_float` panics for
//! `RawQuantizedType`). Instead we construct QTensors from raw bytes via
//! `QStorage::from_data` on a CUDA device — the same path used by the GGUF
//! loader — and verify that `QMatMul::forward` dispatches to `cuda_fwd`
//! (not `cpu_fwd`, which bails with "CPU matmul is not implemented for IQ3XXS").
//!
//! Each block is 256 elements (QK_K). We use d=1.0 (f16) and zeroed qs, which
//! produces a deterministic dequantized value per element. The matmul is
//! compared against a CPU reference computed from the same dequantized
//! weights (via `QTensor::dequantize` on CPU construction).
//!
//! Run on yttri-win: `cargo test --features cuda --package candle-core --test iq_quant_cuda_tests`

#![cfg(feature = "cuda")]

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use half::f16;
use std::sync::Arc;

const QK_K: usize = 256;

/// Build a zeroed block of raw bytes for the given IQ dtype, with d set to 1.0 (f16).
///
/// The block layout follows the CUDA struct definitions in quantized.cu:
/// - IQ3XXS: half d + 3*QK_K/8 bytes qs
/// - IQ2S:   half d + QK_K/4 qs + QK_K/32 qh + QK_K/32 scales
/// - IQ3S:   half d + QK_K/4 qs + QK_K/32 qh + QK_K/8 signs + QK_K/64 scales
/// - IQ2XS:  half d + QK_K/8 * uint16 qs + QK_K/32 scales
/// - IQ4XS:  half d + uint16 scales_h + QK_K/64 scales_l + QK_K/2 qs
fn make_zero_block_bytes(dtype: GgmlDType) -> Vec<u8> {
    let type_size = dtype.type_size();
    let block_size = dtype.block_size();
    assert_eq!(block_size, QK_K, "IQ types have block_size == QK_K == 256");
    let mut bytes = vec![0u8; type_size];
    // Set d (first 2 bytes) to f16(1.0).
    let d_f16: f16 = f16::from_f32(1.0);
    let d_bytes = d_f16.to_le_bytes();
    bytes[0] = d_bytes[0];
    bytes[1] = d_bytes[1];
    bytes
}

/// Build a QTensor on the given device from zeroed blocks (d=1.0).
///
/// `n_rows` x `n_cols` where n_cols must be a multiple of QK_K.
fn make_iq_qtensor(
    n_rows: usize,
    n_cols: usize,
    dtype: GgmlDType,
    device: &Device,
) -> Result<QTensor> {
    assert!(
        n_cols % QK_K == 0,
        "n_cols must be multiple of QK_K, got {n_cols}"
    );
    let blocks_per_row = n_cols / QK_K;
    let total_blocks = n_rows * blocks_per_row;
    let type_size = dtype.type_size();
    let total_bytes = total_blocks * type_size;
    // Build raw data: each block is zeroed with d=1.0.
    let block_template = make_zero_block_bytes(dtype);
    let mut raw = Vec::with_capacity(total_bytes);
    for _ in 0..total_blocks {
        raw.extend_from_slice(&block_template);
    }
    let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), device, dtype)?;
    QTensor::new(storage, (n_rows, n_cols))
}

/// Dequantize the IQ QTensor to a CPU f32 Tensor for reference computation.
///
/// On CUDA, `QTensor::dequantize` calls `QCudaStorage::dequantize` which runs
/// the CUDA dequantize kernel and returns a CudaStorage, then `.to_device(Cpu)`
/// copies it to CPU. This also validates the dequantize kernel itself.
fn dequantize_to_cpu_f32(qt: &QTensor) -> Result<Tensor> {
    let dev = qt.device();
    let t = qt.dequantize(&dev)?;
    t.to_device(&Device::Cpu)?.to_dtype(DType::F32)
}

/// Run QMatMul::forward on a CUDA device with a CUDA input tensor, and compare
/// against a CPU reference matmul using the dequantized weights.
///
/// This catches the "CPU matmul is not implemented for IQ3XXS" bug: if the
/// dispatch erroneously routes to `cpu_fwd`, the test fails with that error
/// instead of producing a result.
fn test_iq_matmul(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Dimensions: m=2, k=256 (1 block), n=4.
    // Weight is [n_rows=n, n_cols=k] (transposed in QMatMul: n first, then k).
    let m = 2;
    let k = QK_K; // single block per row
    let n = 4;

    // Build IQ weight tensor on CUDA: shape (n, k) — n rows, k cols.
    // Wrap in Arc so we can both feed QMatMul and dequantize the same weights.
    let qt = Arc::new(make_iq_qtensor(n, k, dtype, &device)?);
    let qmatmul = QMatMul::from_arc(qt.clone())?;

    // Input: (m, k) f32 on CUDA.
    let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let lhs = Tensor::from_slice(&lhs_data, (m, k), &device)?.to_dtype(DType::F32)?;

    // Run quantized matmul on CUDA. If dispatch is broken, this bails with
    // "CPU matmul is not implemented for IQ3XXS".
    let res = qmatmul.forward(&lhs)?;
    assert!(
        res.device().same_device(&device),
        "result should be on CUDA for {dtype:?}, got {:?}",
        res.device()
    );
    assert_eq!(res.dtype(), DType::F32, "result dtype for {dtype:?}");
    assert_eq!(res.shape().dims(), [m, n], "result shape for {dtype:?}");

    // Reference: dequantize weights to f32 on CPU, matmul.
    let w_cpu = dequantize_to_cpu_f32(&qt)?;
    let lhs_cpu = Tensor::from_slice(&lhs_data, (m, k), &Device::Cpu)?.to_dtype(DType::F32)?;
    // w_cpu shape is (n, k), need (k, n) for lhs @ w_t.
    let ref_mm = lhs_cpu.matmul(&w_cpu.t()?)?;

    // Compare.
    let res_cpu = res.to_device(&Device::Cpu)?;
    let diff = (&res_cpu - &ref_mm)?.abs()?.max_all()?.to_scalar::<f32>()?;
    // Relative tolerance: q8_1 activation quantization (8-bit, per-32 scale)
    // gives ~0.1-0.4% error regardless of the absolute magnitude of the
    // reference (zeroed blocks with d=1.0 dequantize to ~1.0 for IQ2/IQ3 and
    // ~4064 for IQ4XS, so absolute diffs differ by ~4000x across dtypes while
    // the relative error is the same).
    let ref_abs = ref_mm.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff.is_finite(), "non-finite diff {diff} for {dtype:?}");
    assert!(
        diff <= 2e-3 * ref_abs.max(1.0),
        "relative diff {diff} vs ref {ref_abs} too large for {dtype:?}"
    );

    Ok(())
}

#[test]
fn iq2xxs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ2XXS)
}

#[test]
fn iq3xxs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ3XXS)
}

#[test]
fn iq2s_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ2S)
}

#[test]
fn iq3s_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ3S)
}

#[test]
fn iq2xs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ2XS)
}

#[test]
fn iq4xs_cuda_matmul() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ4XS)
}

#[test]
fn iq1m_matmul_dispatches_on_cuda() -> Result<()> {
    test_iq_matmul(GgmlDType::IQ1M)
}

/// Verify that the dequantize kernel produces finite values for each IQ type.
/// This isolates dequantize correctness from the matmul dispatch.
///
/// Note: zeroed quant bytes do NOT dequantize to 0.0 — grid lookup tables have
/// nonzero entries at index 0, so the dequantized value is deterministic but
/// nonzero (e.g. 1.0 for most IQ types). We only assert finiteness here; the
/// matmul test cross-checks CUDA matmul against the CUDA-dequantized reference.
fn test_iq_dequantize_finite(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let n_rows = 1;
    let n_cols = QK_K;
    let qt = make_iq_qtensor(n_rows, n_cols, dtype, &device)?;
    let w = dequantize_to_cpu_f32(&qt)?;
    let vals = w.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(vals.len(), n_rows * n_cols, "len for {dtype:?}");
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite dequant value at {i} for {dtype:?}: {v}"
        );
    }
    // Sanity: not all values are identical NaN/sentinel — at least one finite
    // value exists (already checked above) and the min/max are finite.
    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min.is_finite() && max.is_finite(),
        "min/max not finite for {dtype:?}: {min} {max}"
    );
    Ok(())
}

#[test]
fn iq2xxs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ2XXS)
}

#[test]
fn iq3xxs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ3XXS)
}

#[test]
fn iq2s_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ2S)
}

#[test]
fn iq3s_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ3S)
}

#[test]
fn iq2xs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ2XS)
}

#[test]
fn iq4xs_cuda_dequantize() -> Result<()> {
    test_iq_dequantize_finite(GgmlDType::IQ4XS)
}

/// Multi-block test: k = 2*QK_K (two blocks per row), exercises the
/// dequantize kernel grid with >1 block.
///
/// Compares CUDA matmul against a CPU reference matmul using the
/// CUDA-dequantized weights (same approach as the single-block test). Zeroed
/// weights dequantize to nonzero values, so we do not expect a zero result.
fn test_iq_matmul_multiblock(dtype: GgmlDType) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let m = 2;
    let k = 2 * QK_K;
    let n = 4;

    let qt = Arc::new(make_iq_qtensor(n, k, dtype, &device)?);
    let qmatmul = QMatMul::from_arc(qt.clone())?;

    let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) / (m * k) as f32).collect();
    let lhs = Tensor::from_slice(&lhs_data, (m, k), &device)?.to_dtype(DType::F32)?;

    let res = qmatmul.forward(&lhs)?;
    assert_eq!(res.shape().dims(), [m, n], "multiblock shape for {dtype:?}");
    assert!(
        res.device().same_device(&device),
        "multiblock device for {dtype:?}, got {:?}",
        res.device()
    );

    // Reference: dequantize weights (CUDA) → CPU f32, matmul on CPU.
    let w_cpu = dequantize_to_cpu_f32(&qt)?;
    let lhs_cpu = Tensor::from_slice(&lhs_data, (m, k), &Device::Cpu)?.to_dtype(DType::F32)?;
    let ref_mm = lhs_cpu.matmul(&w_cpu.t()?)?;

    let res_cpu = res.to_device(&Device::Cpu)?;
    let diff = (&res_cpu - &ref_mm)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let ref_abs = ref_mm.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(
        diff.is_finite(),
        "non-finite multiblock diff {diff} for {dtype:?}"
    );
    assert!(
        diff <= 2e-3 * ref_abs.max(1.0),
        "relative multiblock diff {diff} vs ref {ref_abs} too large for {dtype:?}"
    );
    Ok(())
}

#[test]
fn iq2xxs_cuda_matmul_multiblock() -> Result<()> {
    test_iq_matmul_multiblock(GgmlDType::IQ2XXS)
}

#[test]
fn iq3xxs_cuda_matmul_multiblock() -> Result<()> {
    test_iq_matmul_multiblock(GgmlDType::IQ3XXS)
}

#[test]
fn iq4xs_cuda_matmul_multiblock() -> Result<()> {
    test_iq_matmul_multiblock(GgmlDType::IQ4XS)
}

fn make_iq_experts(
    dtype: GgmlDType,
    n_experts: usize,
    n: usize,
    k: usize,
    seed: u32,
    device: &Device,
) -> Result<QTensor> {
    assert_eq!(k % QK_K, 0);
    assert!(matches!(
        dtype,
        GgmlDType::IQ2S
            | GgmlDType::IQ2XS
            | GgmlDType::IQ2XXS
            | GgmlDType::IQ3S
            | GgmlDType::IQ3XXS
            | GgmlDType::IQ4XS
    ));
    let blocks = n_experts * n * k / QK_K;
    let mut raw = vec![0u8; blocks * dtype.type_size()];
    let mut state = seed;
    for block in raw.chunks_exact_mut(dtype.type_size()) {
        let d = f16::from_f32(0.125).to_le_bytes();
        block[..2].copy_from_slice(&d);
        for byte in &mut block[2..] {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *byte = (state >> 24) as u8;
        }
    }
    let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), device, dtype)?;
    QTensor::new(storage, (n_experts, n, k))
}

fn exact_q8_input(batch: usize, topk: usize, k: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(batch * topk * k);
    for task in 0..batch * topk {
        for pos in 0..k {
            let in_block = pos % 32;
            let q = if in_block == 0 {
                127
            } else {
                ((pos * 17 + task * 29) % 255) as i32 - 127
            };
            values.push(q as f32);
        }
    }
    values
}

fn assert_indexed_matches_dequantized(
    weights: &QTensor,
    input_data: &[f32],
    ids_data: &[u32],
    batch: usize,
    topk: usize,
    n: usize,
    k: usize,
    output: &Tensor,
) -> Result<()> {
    let weights_cpu = weights
        .dequantize(&weights.device())?
        .to_device(&Device::Cpu)?
        .to_vec3::<f32>()?;
    let output_cpu = output.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
    for b in 0..batch {
        for t in 0..topk {
            let task = b * topk + t;
            let expert = ids_data[task] as usize;
            let input = &input_data[task * k..(task + 1) * k];
            for row in 0..n {
                let expected: f32 = weights_cpu[expert][row]
                    .iter()
                    .zip(input)
                    .map(|(w, x)| w * x)
                    .sum();
                // Масштаб ошибки q8_1-квантования активаций - Σ|w·x|, НЕ
                // |результат|: при сокращении в дот-продукте |expected| может
                // быть на порядки меньше суммы модулей, и «относительная к
                // результату» ошибка взрывается (batch=33: diff 0.56 на
                // |expected|=10.8 при Σ|w·x| в сотни - штатное q8-округление).
                // Настоящая поломка ядра (знак/скейл/индексация) ошибается на
                // масштабе Σ|w·x| целиком - гейт 2e-3·Σ её ловит с запасом
                // в сотни раз. Прежний 1e-4·|expected| был выставлен для
                // f32-пути (77c4b711) и не обновлялся при переходе на q8_1
                // (33550814) - сьют красный с того коммита.
                let abs_scale: f32 = weights_cpu[expert][row]
                    .iter()
                    .zip(input)
                    .map(|(w, x)| (w * x).abs())
                    .sum();
                let actual = output_cpu[b][t][row];
                let tolerance = 0.02 + abs_scale * 2e-3;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "indexed {:?} mismatch b={b} topk={t} row={row}: actual={actual} expected={expected} tolerance={tolerance} abs_scale={abs_scale}",
                    weights.dtype()
                );
            }
        }
    }
    Ok(())
}

fn test_iq_indexed_moe(dtype: GgmlDType, batch: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (n_experts, n, k, topk) = (5, 7, 2 * QK_K, 8);
    let weights = make_iq_experts(dtype, n_experts, n, k, 7, &device)?;
    let input_data = exact_q8_input(batch, topk, k);
    let input = Tensor::from_slice(&input_data, (batch, topk, k), &device)?;
    let ids_data: Vec<u32> = (0..batch * topk)
        .map(|task| ((task * 3 + 1) % n_experts) as u32)
        .collect();
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;

    let output = weights.indexed_moe_forward_cuda(&input, &ids)?;
    assert_eq!(output.shape().dims(), [batch, topk, n]);
    assert_indexed_matches_dequantized(&weights, &input_data, &ids_data, batch, topk, n, k, &output)
}

#[test]
fn grouped_iq2xxs_all_routes_one_expert_matches_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (n_experts, n, k, batch, topk) = (5, 7, 2 * QK_K, 33, 8);
    let weights = make_iq_experts(GgmlDType::IQ2XXS, n_experts, n, k, 43, &device)?;
    let input_data = exact_q8_input(batch, topk, k);
    let input = Tensor::from_slice(&input_data, (batch, topk, k), &device)?;
    let ids_data = vec![2u32; batch * topk];
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;
    let output = weights.indexed_moe_forward_cuda(&input, &ids)?;
    assert_indexed_matches_dequantized(&weights, &input_data, &ids_data, batch, topk, n, k, &output)
}

#[test]
fn iq2s_indexed_moe_matches_dequantized_reference() -> Result<()> {
    test_iq_indexed_moe(GgmlDType::IQ2S, 4)
}

#[test]
fn iq2xxs_indexed_moe_matches_dequantized_reference() -> Result<()> {
    for batch in [1, 4, 5, 33] {
        test_iq_indexed_moe(GgmlDType::IQ2XXS, batch)?;
    }
    Ok(())
}

fn test_iq_indexed_moe_shared_input(dtype: GgmlDType, batch: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (n_experts, n, k, topk) = (5, 7, 2 * QK_K, 8);
    let weights = make_iq_experts(dtype, n_experts, n, k, 31, &device)?;
    let full_input = exact_q8_input(batch, topk, k);
    let input_data: Vec<f32> = (0..batch)
        .flat_map(|b| full_input[b * topk * k..b * topk * k + k].iter().copied())
        .collect();
    let input = Tensor::from_slice(&input_data, (batch, 1, k), &device)?;
    let ids_data: Vec<u32> = (0..batch * topk)
        .map(|task| ((task * 3 + 1) % n_experts) as u32)
        .collect();
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;
    let output = weights.indexed_moe_forward_cuda(&input, &ids)?;
    let expanded: Vec<f32> = (0..batch)
        .flat_map(|b| {
            let row = input_data[b * k..(b + 1) * k].to_vec();
            (0..topk).flat_map(move |_| row.clone())
        })
        .collect();
    assert_indexed_matches_dequantized(&weights, &expanded, &ids_data, batch, topk, n, k, &output)
}

#[test]
fn matrix_indexed_moe_shared_input_matches_dequantized_reference() -> Result<()> {
    for dtype in [
        GgmlDType::IQ2XS,
        GgmlDType::IQ2XXS,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ4XS,
    ] {
        test_iq_indexed_moe_shared_input(dtype, 5)?;
    }
    Ok(())
}

#[test]
fn iq3s_indexed_moe_matches_dequantized_reference() -> Result<()> {
    test_iq_indexed_moe(GgmlDType::IQ3S, 4)
}

#[test]
fn remaining_matrix_indexed_moe_matches_dequantized_reference() -> Result<()> {
    for dtype in [GgmlDType::IQ2XS, GgmlDType::IQ3XXS, GgmlDType::IQ4XS] {
        test_iq_indexed_moe(dtype, 5)?;
    }
    Ok(())
}

fn test_iq_indexed_moe_dual(dtype: GgmlDType, batch: usize, shared_input: bool) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (n_experts, n, k, topk) = (3, 5, 2 * QK_K, 8);
    let gate = make_iq_experts(dtype, n_experts, n, k, 11, &device)?;
    let up = make_iq_experts(dtype, n_experts, n, k, 19, &device)?;
    let full_input = exact_q8_input(batch, topk, k);
    let input_data: Vec<f32> = if shared_input {
        (0..batch)
            .flat_map(|b| full_input[b * topk * k..b * topk * k + k].iter().copied())
            .collect()
    } else {
        full_input
    };
    let input_dim1 = if shared_input { 1 } else { topk };
    let input = Tensor::from_slice(&input_data, (batch, input_dim1, k), &device)?;
    let ids_data: Vec<u32> = (0..batch * topk)
        .map(|task| ((task * 2 + 1) % n_experts) as u32)
        .collect();
    let ids = Tensor::from_slice(&ids_data, (batch, topk), &device)?;

    let gate_single = gate.indexed_moe_forward_cuda(&input, &ids)?;
    let up_single = up.indexed_moe_forward_cuda(&input, &ids)?;
    let (gate_dual, up_dual) = gate.indexed_moe_forward_dual_cuda(&up, &input, &ids)?;
    let gate_diff = (&gate_single - &gate_dual)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let up_diff = (&up_single - &up_dual)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert_eq!(gate_diff, 0.0, "gate dual differs from single");
    assert_eq!(up_diff, 0.0, "up dual differs from single");
    Ok(())
}

#[test]
fn iq2s_indexed_moe_dual_matches_single() -> Result<()> {
    test_iq_indexed_moe_dual(GgmlDType::IQ2S, 2, false)
}

#[test]
fn iq2xxs_indexed_moe_dual_matches_single() -> Result<()> {
    for batch in [1, 4, 5] {
        test_iq_indexed_moe_dual(GgmlDType::IQ2XXS, batch, false)?;
        test_iq_indexed_moe_dual(GgmlDType::IQ2XXS, batch, true)?;
    }
    Ok(())
}

#[test]
fn remaining_matrix_indexed_moe_dual_matches_single() -> Result<()> {
    for dtype in [GgmlDType::IQ2XS, GgmlDType::IQ3XXS, GgmlDType::IQ4XS] {
        test_iq_indexed_moe_dual(dtype, 5, true)?;
    }
    Ok(())
}

#[test]
fn iq_dual_indexed_moe_honors_view_offsets() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (n_experts, n, k, batch, topk) = (3, 5, 2 * QK_K, 5, 8);
    let gate = make_iq_experts(GgmlDType::IQ2XXS, n_experts, n, k, 11, &device)?;
    let up = make_iq_experts(GgmlDType::IQ2XXS, n_experts, n, k, 19, &device)?;

    let full_input_data = exact_q8_input(batch + 2, topk, k);
    let input_data = full_input_data[topk * k..(batch + 1) * topk * k].to_vec();
    let input =
        Tensor::from_slice(&full_input_data, (batch + 2, topk, k), &device)?.narrow(0, 1, batch)?;

    let full_ids_data: Vec<u32> = (0..(batch + 2) * topk)
        .map(|task| ((task * 2 + 1) % n_experts) as u32)
        .collect();
    let ids_data = full_ids_data[topk..(batch + 1) * topk].to_vec();
    let ids =
        Tensor::from_slice(&full_ids_data, (batch + 2, topk), &device)?.narrow(0, 1, batch)?;

    let (gate_out, up_out) = gate.indexed_moe_forward_dual_cuda(&up, &input, &ids)?;
    assert_indexed_matches_dequantized(
        &gate,
        &input_data,
        &ids_data,
        batch,
        topk,
        n,
        k,
        &gate_out,
    )?;
    assert_indexed_matches_dequantized(&up, &input_data, &ids_data, batch, topk, n, k, &up_out)
}

#[test]
fn iq_indexed_moe_rejects_noncontiguous_input() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (n_experts, n, k, batch, topk) = (3, 5, 2 * QK_K, 2, 8);
    let weights = make_iq_experts(GgmlDType::IQ2XXS, n_experts, n, k, 11, &device)?;
    let input_data = exact_q8_input(batch, topk, k);
    let input = Tensor::from_slice(&input_data, (batch, k, topk), &device)?.transpose(1, 2)?;
    let ids = Tensor::zeros((batch, topk), DType::U32, &device)?;
    let error = match weights.indexed_moe_forward_cuda(&input, &ids) {
        Ok(_) => panic!("non-contiguous input unexpectedly accepted"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("input not contiguous"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// Глобальная блокировка для CUDA-graph тестов: захват на одном stream не
/// потокобезопасен при параллельных тестах (cargo test default threads).
static GRAPH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Замок с игнором poison: паника одного graph-теста не должна каскадом
/// валить остальные PoisonError'ом (падал бы не виновник, а соседи по алфавиту).
fn graph_lock() -> std::sync::MutexGuard<'static, ()> {
    GRAPH_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Minimal CUDA graph capture/replay sanity: умножение через graph, replay дважды.
/// Изолирует механику cudarc begin/end_capture + launch от модели.
/// Контракт продакшна (adapter.rs, fd68dd73): аллокации внутри захвата
/// сбалансированы — промежуточный тензор дропается ДО end_capture, выход идёт
/// D2D-нодой во внешний (pre-alloc) буфер. Несбалансированный alloc-нод делает
/// граф одноразовым: второй cuGraphLaunch → CUDA_ERROR_INVALID_VALUE.
#[test]
fn cuda_graph_minimal_capture_replay() -> Result<()> {
    let _guard = graph_lock();
    use candle_core::cuda_backend::cudarc;
    let device = Device::new_cuda(0)?;
    let cuda_dev = device.as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();

    let x = Tensor::ones((16,), DType::F32, &device)?;
    let out = Tensor::zeros((16,), DType::F32, &device)?; // внешний буфер (до захвата)
    // Prime тех же ядер вне захвата; out = 1.0, чтобы отличить от результата графа.
    {
        let y = (x.clone() * 2.0)?;
        drop(y);
        out.slice_set(&x, 0, 0)?;
    }
    stream
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("sync0: {e}")))?;

    // Capture: y = x * 2 (alloc в graph pool) → D2D в out → drop(y) (free-нода).
    use cudarc::driver::{result as cres, sys as csys};
    unsafe { cres::stream::begin_capture(stream.cu_stream(), csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) }
        .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
    {
        let y = (x.clone() * 2.0)?;
        out.slice_set(&y, 0, 0)?;
        drop(y);
    }
    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
        .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;
    assert!(!cu_graph.is_null(), "end_capture returned null graph");
    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
    unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
    assert!(!exec.is_null(), "instantiate failed");
    // WDDM probe: upload фиксирует graph pool перед первым launch.
    if std::env::var("QWEN36_TEST_GRAPH_UPLOAD").as_deref() == Ok("1") {
        unsafe { csys::cuGraphUpload(exec, stream.cu_stream()) };
    }
    let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "launch1: {res:?}");
    stream
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("sync1: {e}")))?;
    // Второй launch того же exec — модельный сценарий делает N replay подряд.
    let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "launch2: {res:?}");
    stream
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("sync2: {e}")))?;
    let y0 = out.to_vec1::<f32>()?;
    assert!(y0.iter().all(|&v| v == 2.0), "graph replay wrong: {:?}", &y0[..4]);
    unsafe {
        csys::cuGraphExecDestroy(exec);
        csys::cuGraphDestroy(cu_graph);
    }
    Ok(())
}

/// Граф с alloc+free узлами (промежуточный тензор дропнут внутри захвата).
#[test]
fn cuda_graph_alloc_free_balanced_double_launch() -> Result<()> {
    let _guard = graph_lock();
    use candle_core::cuda_backend::cudarc;
    use cudarc::driver::{result as cres, sys as csys};
    let device = Device::new_cuda(0)?;
    let cuda_dev = device.as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();

    let x = Tensor::ones((16,), DType::F32, &device)?;
    // Prime.
    let _ = (x.clone() * 2.0)?;

    unsafe { cres::stream::begin_capture(stream.cu_stream(), csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) }
        .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
    {
        // alloc + free внутри захвата (y дропается до end_capture).
        let y = (x.clone() * 2.0)?;
        drop(y);
    }
    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
        .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;
    assert!(!cu_graph.is_null());
    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
    unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
    assert!(!exec.is_null(), "instantiate failed");
    for i in 0..3 {
        let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
        assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "launch{i}: {res:?}");
        stream.synchronize().map_err(|e| candle_core::Error::Msg(format!("sync{i}: {e}")))?;
    }
    unsafe {
        csys::cuGraphExecDestroy(exec);
        csys::cuGraphDestroy(cu_graph);
    }
    Ok(())
}

/// Граф БЕЗ alloc-узлов: запись в пред-аллоцированный буфер (WDDM проверка).
#[test]
fn cuda_graph_no_alloc_nodes_double_launch() -> Result<()> {
    let _guard = graph_lock();
    use candle_core::cuda_backend::cudarc;
    use cudarc::driver::{result as cres, sys as csys};
    let device = Device::new_cuda(0)?;
    let cuda_dev = device.as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();

    let x = Tensor::ones((16,), DType::F32, &device)?;
    let out = Tensor::zeros((16,), DType::F32, &device)?; // внешний буфер (pre-alloc)
    // Prime: copy2d kernel load.
    out.slice_set(&x, 0, 0)?;
    stream.synchronize().map_err(|e| candle_core::Error::Msg(format!("sync0: {e}")))?;

    unsafe { cres::stream::begin_capture(stream.cu_stream(), csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) }
        .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
    // Внутри захвата: только device-side копия в pre-alloc буфер, НОЛЬ аллокаций.
    out.slice_set(&x, 0, 0)?;
    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
        .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;
    assert!(!cu_graph.is_null());
    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
    unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
    assert!(!exec.is_null(), "instantiate failed");
    for i in 0..3 {
        let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
        assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "launch{i}: {res:?}");
        stream.synchronize().map_err(|e| candle_core::Error::Msg(format!("sync{i}: {e}")))?;
    }
    let y0 = out.to_vec1::<f32>()?;
    assert!(y0.iter().all(|&v| v == 1.0), "no-alloc graph wrong: {:?}", &y0[..4]);
    unsafe {
        csys::cuGraphExecDestroy(exec);
        csys::cuGraphDestroy(cu_graph);
    }
    Ok(())
}

/// Кастомные ядра с raw u64-pointer аргументами (cumsum/increment стиль) в графе.
#[test]
fn cuda_graph_raw_ptr_kernel_capture() -> Result<()> {
    let _guard = graph_lock();
    use candle_core::cuda_backend::cudarc;
    use cudarc::driver::{result as cres, sys as csys, LaunchConfig, PushKernelArg};
    let device = Device::new_cuda(0)?;
    let cuda_dev = device.as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();

    let kv_len = cuda_dev.alloc_zeros::<u32>(4)?;
    let slots = cuda_dev.alloc_zeros::<u32>(4)?;
    let out_t = Tensor::zeros(5, DType::U32, &device)?;

    let func = cuda_dev.get_or_load_func(
        "cumsum_seqlens_from_kvlen",
        &candle_core::cuda_backend::kernels::QUANTIZED,
    )?;
    // Prime eager.
    let out_ptr = {
        let (st, layout) = out_t.storage_and_layout();
        let cuda = match &*st {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("not cuda"),
        };
        let slice0 = cuda.as_cuda_slice::<u32>()?;
        let stream = slice0.stream();
        let slice = slice0.slice(layout.start_offset()..);
        let (p, _g) = cudarc::driver::DevicePtr::device_ptr(&slice, stream);
        p
    };
    {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = func.builder();
        b.arg(&kv_len);
        b.arg(&slots);
        b.arg(&out_ptr);
        b.arg(&4i32);
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
    }
    stream.synchronize().map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;

    unsafe {
        cres::stream::begin_capture(
            stream.cu_stream(),
            csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
    {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let mut b = func.builder();
        b.arg(&kv_len);
        b.arg(&slots);
        b.arg(&out_ptr);
        b.arg(&4i32);
        unsafe { b.launch(cfg) }.map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
    }
    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
        .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;
    assert!(!cu_graph.is_null());
    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
    let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "instantiate: {res:?}");
    let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "launch: {res:?}");
    stream.synchronize().map_err(|e| candle_core::Error::Msg(format!("sync2: {e}")))?;
    unsafe {
        csys::cuGraphExecDestroy(exec);
        csys::cuGraphDestroy(cu_graph);
    }
    Ok(())
}

/// Полный L=0 набор ядер модели в графе: embedding + rmsnorm + q8_1 + matvec + copy2d.
#[test]
fn cuda_graph_model_l0_stack_capture() -> Result<()> {
    let _guard = graph_lock();
    use candle_core::cuda_backend::cudarc;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use cudarc::driver::{result as cres, sys as csys};
    let device = Device::new_cuda(0)?;
    let cuda_dev = device.as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();

    // Embedding table Q6_K [vocab=512, hidden=256] + ids.
    let blocks = 512 * 256 / 256;
    let raw = vec![0u8; blocks * GgmlDType::Q6K.type_size()];
    let emb_qt = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Borrowed(&raw),
            &device,
            GgmlDType::Q6K,
        )?,
        (512, 256),
    )?;
    let emb = QMatMul::from_qtensor(emb_qt)?;
    // Output Q6_K [256, 512].
    let raw2 = vec![0u8; blocks * GgmlDType::Q6K.type_size()];
    let out_qt = QTensor::new(
        candle_core::quantized::QStorage::from_data(
            std::borrow::Cow::Borrowed(&raw2),
            &device,
            GgmlDType::Q6K,
        )?,
        (512, 256),
    )?;
    let out_mm = QMatMul::from_qtensor(out_qt)?;
    let ids_t = Tensor::from_vec(vec![7u32], (1, 1usize), &device)?;
    // Prime.
    {
        let e = emb.embedding(&ids_t)?.reshape((1, 1usize, 256))?;
        let h = (e.squeeze(1)? * 1.0)?;
        let l = out_mm.forward(&h)?;
        let _ = l.flatten_all()?.to_vec1::<f32>()?;
    }
    let logits_out = Tensor::zeros((1, 512), DType::F32, &device)?;

    unsafe {
        cres::stream::begin_capture(
            stream.cu_stream(),
            csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
    let e = emb.embedding(&ids_t)?.reshape((1, 1usize, 256))?;
    let h = (e.squeeze(1)? * 1.0)?;
    let l = out_mm.forward(&h)?;
    logits_out.slice_set(&l, 0, 0)?;
    drop(l);
    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
        .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;
    assert!(!cu_graph.is_null());
    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
    let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "instantiate: {res:?}");
    let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "launch: {res:?}");
    stream
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
    let v = logits_out.flatten_all()?.to_vec1::<f32>()?;
    assert!(v.iter().all(|x| x.is_finite()));
    unsafe {
        csys::cuGraphExecDestroy(exec);
        csys::cuGraphDestroy(cu_graph);
    }
    Ok(())
}

/// То же, но с PTX quantized ядром + cuBLAS matmul внутри захвата.
#[test]
fn cuda_graph_quantized_and_cublas_capture() -> Result<()> {
    let _guard = graph_lock();
    use candle_core::cuda_backend::cudarc;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    let device = Device::new_cuda(0)?;
    let cuda_dev = device.as_cuda_device()?;
    let stream = cuda_dev.cuda_stream();

    let k = 256usize;
    let n = 128usize;
    // Q4_K через raw-байты (QTensor::quantize CPU-only для части типов, а to_device — отдельно).
    let blocks = n * k / 256;
    let raw = vec![0u8; blocks * GgmlDType::Q4K.type_size()];
    let qs = candle_core::quantized::QStorage::from_data(
        std::borrow::Cow::Borrowed(&raw),
        &device,
        GgmlDType::Q4K,
    )?;
    let qt = QTensor::new(qs, (n, k))?;
    let mm = QMatMul::from_qtensor(qt)?;
    let x = Tensor::randn(0f32, 1.0, (1, 1, k), &device)?;

    // Prime всех ядер вне захвата.
    let _ = mm.forward(&x)?;
    let _ = x.matmul(&x.transpose(1, 2)?)?;

    use cudarc::driver::{result as cres, sys as csys};
    unsafe {
        cres::stream::begin_capture(
            stream.cu_stream(),
            csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        )
    }
    .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;
    let y = mm.forward(&x)?;
    let z = x.matmul(&x.transpose(1, 2)?)?;
    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
        .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;
    assert!(!cu_graph.is_null());
    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
    let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
    assert_eq!(res, csys::CUresult::CUDA_SUCCESS, "instantiate: {res:?}");
    assert!(!exec.is_null());
    let res = unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) };
    assert_eq!(
        res,
        csys::CUresult::CUDA_SUCCESS,
        "cuGraphLaunch failed: {res:?}"
    );
    stream
        .synchronize()
        .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
    let y0 = y.flatten_all()?.to_vec1::<f32>()?;
    assert!(y0.iter().all(|v| v.is_finite()));
    let z0 = z.flatten_all()?.to_vec1::<f32>()?;
    assert!(z0.iter().all(|v| v.is_finite()));
    unsafe {
        csys::cuGraphExecDestroy(exec);
        csys::cuGraphDestroy(cu_graph);
    }
    Ok(())
}

/// Decode MMVQ perf micro-benchmark: measures pure kernel bandwidth for
/// dense-model hot shapes (27B Q2_K_XL): QMatMul [n,k] @ [1,1,k] f32, B=1.
///
/// Run: cargo test --features cuda --package candle-core --test iq_quant_cuda_tests \
///   -- --ignored --exact mmvq_perf_dense --nocapture --test-threads=1
#[test]
#[ignore]
fn mmvq_perf_dense() -> Result<()> {
    use candle_core::quantized::k_quants::*;
    use candle_core::quantized::GgmlType;
    use std::time::Instant;

    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(c) => c.clone(),
        _ => unreachable!(),
    };

    // QWEN36_PERF_FILL_GB=N — занять N ГБ VRAM перед созданием весов
    // (эмуляция размещения весов 27B после 10GB аллокаций).
    let _fill: Vec<cudarc::driver::CudaSlice<u8>> =
        if let Ok(v) = std::env::var("QWEN36_PERF_FILL_GB") {
            let gb: usize = v.parse().unwrap_or(0);
            let mut keep = Vec::new();
            for _ in 0..gb {
                let chunk = unsafe { cuda.alloc::<u8>(1024 * 1024 * 1024) }?;
                keep.push(chunk);
            }
            keep
        } else {
            Vec::new()
        };

    // 27B Q2_K_XL hot shapes: (n_rows, k)
    let shapes: &[(usize, usize)] = &[
        (5120, 5120),    // delta qkv-ish / attn
        (17408, 5120),   // ffn up/gate (27B ffn=17408)
        (5120, 17408),   // ffn down (27B)
    ];

    fn bench_quant<T: GgmlType + Send + Sync + 'static>(
        cuda: &candle_core::CudaDevice,
        device: &Device,
        dtype: GgmlDType,
        n_rows: usize,
        k: usize,
        iters: usize,
    ) -> Result<()> {
        let row_bytes = k / T::BLCK_SIZE * std::mem::size_of::<T>();
        bench_raw(cuda, device, dtype, n_rows, k, row_bytes, iters)
    }

    fn bench_raw(
        cuda: &candle_core::CudaDevice,
        device: &Device,
        dtype: GgmlDType,
        n_rows: usize,
        k: usize,
        row_bytes: usize,
        iters: usize,
    ) -> Result<()> {
        let total = n_rows * row_bytes;
        // QWEN36_PERF_RANDOM=1: реалистичные байты весов. У IQ vec_dot
        // grid-lookup идёт по байту веса: константный филл даёт один индекс
        // на весь варп (broadcast из L1) и завышает скорость ядра в разы
        // против реальных весов (в модели те же ядра шли в ~3x медленнее).
        let raw: Vec<u8> = if std::env::var("QWEN36_PERF_RANDOM").as_deref() == Ok("1") {
            (0..total)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect()
        } else {
            vec![0x5Au8; total]
        };
        let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), device, dtype)?;
        let qt = Arc::new(QTensor::new(storage, (n_rows, k))?);
        let qmatmul = QMatMul::from_arc(qt)?;
        let y = Tensor::zeros((1, 1, k), DType::F32, device)?;
        // warmup (JIT + caches)
        for _ in 0..5 {
            let _ = qmatmul.forward(&y)?;
        }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = qmatmul.forward(&y)?;
        }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let el = t0.elapsed().as_secs_f64() / iters as f64;
        let gbs = total as f64 / el / 1e9;
        println!(
            "MMVQ {dtype:?} [{n_rows}x{k}] B=1: {:.3}ms {:.0} GB/s",
            el * 1000.0,
            gbs
        );
        Ok(())
    }

    fn bench_iq(
        cuda: &candle_core::CudaDevice,
        device: &Device,
        dtype: GgmlDType,
        n_rows: usize,
        k: usize,
        iters: usize,
    ) -> Result<()> {
        let row_bytes = k / dtype.block_size() * dtype.type_size();
        bench_raw(cuda, device, dtype, n_rows, k, row_bytes, iters)
    }

    for &(n, k) in shapes {
        bench_quant::<BlockQ2K>(&cuda, &device, GgmlDType::Q2K, n, k, 50)?;
        bench_quant::<BlockQ3K>(&cuda, &device, GgmlDType::Q3K, n, k, 50)?;
        bench_quant::<BlockQ4K>(&cuda, &device, GgmlDType::Q4K, n, k, 50)?;
        bench_quant::<BlockQ5K>(&cuda, &device, GgmlDType::Q5K, n, k, 50)?;
        bench_iq(&cuda, &device, GgmlDType::IQ2XXS, n, k, 50)?;
        bench_iq(&cuda, &device, GgmlDType::IQ2XS, n, k, 50)?;
        bench_iq(&cuda, &device, GgmlDType::IQ2S, n, k, 50)?;
        bench_iq(&cuda, &device, GgmlDType::IQ3XXS, n, k, 50)?;
        bench_iq(&cuda, &device, GgmlDType::IQ3S, n, k, 50)?;
        bench_iq(&cuda, &device, GgmlDType::IQ4XS, n, k, 50)?;
    }
    // Sustained: та же форма повторно после всех dtype (drift/thermal check).
    eprintln!("--- sustained re-run ---");
    bench_quant::<BlockQ2K>(&cuda, &device, GgmlDType::Q2K, 17408, 5120, 500)?;
    bench_quant::<BlockQ2K>(&cuda, &device, GgmlDType::Q2K, 5120, 17408, 500)?;

    // Graph-replay FFN-цепочки (w1+w3+silu+w2) ×48 — как в модели 27B.
    // Если replay медленнее eager ×N — проблема в graph-контексте MMVQ.
    {
        use cudarc::driver::result as cres;
        use cudarc::driver::sys as csys;
        let n = 17408usize;
        let k = 5120usize;
        let row_bytes = k / 256 * std::mem::size_of::<BlockQ2K>();
        let mk = |rows: usize| -> Result<Arc<QTensor>> {
            let raw = vec![0x5Au8; rows * row_bytes];
            let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), &device, GgmlDType::Q2K)?;
            Ok(Arc::new(QTensor::new(storage, (rows, k))?))
        };
        // 48 пар (w1: [n,k], w3: [n,k], w2: [k,n]) — 48 слоёв как в 27B.
        let mut layers = Vec::new();
        for _ in 0..48 {
            let w1 = QMatMul::from_arc(mk(n)?)?;
            let w3 = QMatMul::from_arc(mk(n)?)?;
            let w2raw = vec![0x5Au8; k * (n / 256 * std::mem::size_of::<BlockQ2K>())];
            let w2s = QStorage::from_data(std::borrow::Cow::Borrowed(&w2raw), &device, GgmlDType::Q2K)?;
            let w2 = QMatMul::from_arc(Arc::new(QTensor::new(w2s, (k, n))?))?;
            layers.push((w1, w3, w2));
        }
        let x = Tensor::zeros((1, 1, k), DType::F32, &device)?;
        let run_chain = |x: &Tensor| -> Result<Tensor> {
            let mut h = x.clone();
            for (w1, w3, w2) in &layers {
                let prequant = candle_core::quantized::QTensor::prequantize_q8_1(&h).ok().flatten();
                let a = w1.forward_with_prequant(&h, prequant.as_ref())?;
                let b = w3.forward_with_prequant(&h, prequant.as_ref())?;
                let s = a.silu_mul_direct(&b)?;
                h = w2.forward(&s)?;
            }
            Ok(h)
        };
        // eager
        for _ in 0..3 { let _ = run_chain(&x)?; }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let t0 = Instant::now();
        for _ in 0..5 { let _ = run_chain(&x)?; }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
        println!("FFN-chain x48 EAGER: {eager_ms:.2}ms/step");
        // graph capture + replay
        let stream = cuda.cuda_stream();
        unsafe { cres::stream::begin_capture(stream.cu_stream(), csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) }
            .map_err(candle_core::Error::wrap)?;
        let out = run_chain(&x);
        let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
            .map_err(candle_core::Error::wrap)?;
        out?;
        let mut exec: csys::CUgraphExec = std::ptr::null_mut();
        let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
        assert_eq!(res, csys::CUresult::CUDA_SUCCESS);
        // warm
        for _ in 0..3 { unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) }; }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let t0 = Instant::now();
        for _ in 0..20 { unsafe { csys::cuGraphLaunch(exec, stream.cu_stream()) }; }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let graph_ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
        println!("FFN-chain x48 GRAPH-REPLAY: {graph_ms:.2}ms/step");
    }
    Ok(())
}

/// Числовая сверка MMQ Tensor-Core prefill против dequant-эталона.
/// Оба используют ОДНИ и те же квантованные веса → изолирует корректность MMQ-ядра.
#[test]
#[ignore]
fn mmq_mma_matches_reference() -> Result<()> {
    use candle_core::quantized::{QMatMul, QTensor};
    let device = Device::new_cuda(0)?;
    let (n, k, m) = (512usize, 768usize, 64usize); // n % 128 == 0, m_total=64 > 8 → MMQ

    for &dtype in &[
        GgmlDType::Q4K,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q4_0,
        GgmlDType::Q8_0,
    ] {
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.037).sin()).collect();
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let w_t = Tensor::from_vec(w, (n, k), &device)?;
        let x_t = Tensor::from_vec(x, (1, m, k), &device)?;

        let qt = std::sync::Arc::new(QTensor::quantize(&w_t, dtype)?);
        let qmm = QMatMul::from_arc(qt.clone())?;
        let got = qmm.forward(&x_t)?; // m=64 → MMQ Tensor-Core путь

        // Эталон: dequant тех же весов + f32 matmul (2D: [m,k] @ [k,n] → [m,n]).
        let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
        let x_2d = x_t.reshape((m, k))?;
        let want = x_2d.matmul(&w_deq.t()?.contiguous()?)?.reshape((1, m, n))?;

        let diff = (&got - &want)?.abs()?;
        let max = diff.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        println!("MMQ {dtype:?}: max_abs_diff={max:.6} scale={scale:.4}");
        // MMQ квантует активации в q8_1, как и MMVQ, но аккумулирует иначе
        // (int32 с f16-масштабами, раскладка D4/DS4/D2S6) — отсюда допуск
        // шире, чем 2e-3 у MMVQ. Наблюдаемое отклонение 0.5-0.9% на всех семи
        // типах; порог 1.5% даёт ~2x запаса и при этом ловит поломку ядра.
        // Прежние 8% пропускали бы почти любую регрессию.
        assert!(
            max < 0.015 * scale,
            "MMQ {dtype:?} mismatch: max_abs_diff={max} vs scale={scale} (tolerance 1.5%)"
        );
    }
    Ok(())
}

/// MMQ при m > mmq_x (несколько тайлов по X): юнит-тесты выше гоняли m=64
/// (один тайл) и пропустили бы поломку tile-раскладки stream-k. A/B на модели
/// (512-префилл 27B) дал nRMSE 0.53 против dequant-референса — этот тест
/// бисектит: K-quant и IQ отдельно, m=512 (4 тайла x128) и m=200 (не кратно).
#[test]
fn mmq_mma_wide_batch_matches_reference() -> Result<()> {
    use candle_core::quantized::{QMatMul, QTensor};
    let device = Device::new_cuda(0)?;
    let (n, k) = (512usize, 768usize);

    // Полная матрица: все MMQ-dtype на широком батче. Модельный A/B дал
    // nRMSE 0.53 при зелёных Q4K/IQ2XXS - виновник в другом dtype или форме.
    for &m in &[512usize, 200] {
        for &dtype in &[
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8_0,
        ] {
            let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.037).sin()).collect();
            let w_t = Tensor::from_vec(w, (n, k), &device)?;
            let qt = std::sync::Arc::new(QTensor::quantize(&w_t, dtype)?);
            let qmm = QMatMul::from_arc(qt.clone())?;
            let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
            let x_t = Tensor::from_vec(x, (1, m, k), &device)?;
            let got = qmm.forward(&x_t)?;
            let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
            let want = x_t
                .reshape((m, k))?
                .matmul(&w_deq.t()?.contiguous()?)?
                .reshape((1, m, n))?;
            let max = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
            println!("MMQ-WIDE {dtype:?} m={m}: max_abs_diff={max:.6} scale={scale:.4}");
            assert!(
                max < 0.015 * scale,
                "MMQ-WIDE {dtype:?} m={m}: max_abs_diff={max} vs scale={scale}"
            );
        }
        for &dtype in &[
            GgmlDType::IQ2XS,
            GgmlDType::IQ2S,
            GgmlDType::IQ3XXS,
            GgmlDType::IQ3S,
            GgmlDType::IQ4XS,
        ] {
            let block_bytes = dtype.type_size();
            let row_blocks = k / dtype.block_size();
            let total = n * row_blocks * block_bytes;
            let mut raw: Vec<u8> = (0..total)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            for b in 0..(n * row_blocks) {
                raw[b * block_bytes] = 0x00;
                raw[b * block_bytes + 1] = 0x3C;
            }
            let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), &device, dtype)?;
            let qt = std::sync::Arc::new(QTensor::new(storage, (n, k))?);
            let qmm = QMatMul::from_arc(qt.clone())?;
            let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.011).sin()).collect();
            let x_t = Tensor::from_vec(x, (1, m, k), &device)?;
            let got = qmm.forward(&x_t)?;
            let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
            let want = x_t
                .reshape((m, k))?
                .matmul(&w_deq.t()?.contiguous()?)?
                .reshape((1, m, n))?;
            let max = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
            println!("MMQ-WIDE {dtype:?} m={m}: max_abs_diff={max:.6} scale={scale:.4}");
            assert!(
                max < 0.015 * scale,
                "MMQ-WIDE {dtype:?} m={m}: max_abs_diff={max} vs scale={scale}"
            );
        }
        // Реальные формы 27B: FFN up/gate и down, iq2_xxs.
        for &(nn, kk) in &[(17408usize, 5120usize), (5120, 17408)] {
            let dtype = GgmlDType::IQ2XXS;
            let block_bytes = dtype.type_size();
            let row_blocks = kk / dtype.block_size();
            let total = nn * row_blocks * block_bytes;
            let mut raw: Vec<u8> = (0..total)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            for b in 0..(nn * row_blocks) {
                raw[b * block_bytes] = 0x00;
                raw[b * block_bytes + 1] = 0x3C;
            }
            let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), &device, dtype)?;
            let qt = std::sync::Arc::new(QTensor::new(storage, (nn, kk))?);
            let qmm = QMatMul::from_arc(qt.clone())?;
            let x: Vec<f32> = (0..m * kk).map(|i| ((i as f32) * 0.007).cos()).collect();
            let x_t = Tensor::from_vec(x, (1, m, kk), &device)?;
            let got = qmm.forward(&x_t)?;
            let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
            let want = x_t
                .reshape((m, kk))?
                .matmul(&w_deq.t()?.contiguous()?)?
                .reshape((1, m, nn))?;
            let max = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
            println!("MMQ-WIDE IQ2XXS [{nn}x{kk}] m={m}: max_abs_diff={max:.6} scale={scale:.4}");
            assert!(
                max < 0.015 * scale,
                "MMQ-WIDE IQ2XXS [{nn}x{kk}] m={m}: max_abs_diff={max} vs scale={scale}"
            );
        }
        // Q4K: квантуем из float.
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.037).sin()).collect();
        let w_t = Tensor::from_vec(w, (n, k), &device)?;
        let qt = std::sync::Arc::new(QTensor::quantize(&w_t, GgmlDType::Q4K)?);
        let qmm = QMatMul::from_arc(qt.clone())?;
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let x_t = Tensor::from_vec(x, (1, m, k), &device)?;
        let got = qmm.forward(&x_t)?;
        let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
        let want = x_t
            .reshape((m, k))?
            .matmul(&w_deq.t()?.contiguous()?)?
            .reshape((1, m, n))?;
        let max = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        println!("MMQ-WIDE Q4K m={m}: max_abs_diff={max:.6} scale={scale:.4}");
        assert!(
            max < 0.015 * scale,
            "MMQ-WIDE Q4K m={m}: max_abs_diff={max} vs scale={scale}"
        );

        // IQ: raw-случайные веса, d=1.0.
        let dtype = GgmlDType::IQ2XXS;
        let block_bytes = dtype.type_size();
        let row_blocks = k / dtype.block_size();
        let total = n * row_blocks * block_bytes;
        let mut raw: Vec<u8> = (0..total)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        for b in 0..(n * row_blocks) {
            raw[b * block_bytes] = 0x00;
            raw[b * block_bytes + 1] = 0x3C;
        }
        let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), &device, dtype)?;
        let qt = std::sync::Arc::new(QTensor::new(storage, (n, k))?);
        let qmm = QMatMul::from_arc(qt.clone())?;
        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.011).sin()).collect();
        let x_t = Tensor::from_vec(x, (1, m, k), &device)?;
        let got = qmm.forward(&x_t)?;
        let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
        let want = x_t
            .reshape((m, k))?
            .matmul(&w_deq.t()?.contiguous()?)?
            .reshape((1, m, n))?;
        let max = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        println!("MMQ-WIDE IQ2XXS m={m}: max_abs_diff={max:.6} scale={scale:.4}");
        assert!(
            max < 0.015 * scale,
            "MMQ-WIDE IQ2XXS m={m}: max_abs_diff={max} vs scale={scale}"
        );
    }
    Ok(())
}

/// MMQ Tensor-Core для IQ-квантов против dequant-эталона.
/// Веса — случайные raw-байты (реалистичные grid-индексы; см. урок
/// constant-memory-divergent-lookup: константная заливка маскирует
/// дата-зависимые пути), f16-масштаб d каждого блока санирован в 1.0
/// (случайные байты дают NaN/Inf в d). Кванты и внутренние int-скейлы
/// остаются случайными.
#[test]
fn mmq_mma_iq_matches_reference() -> Result<()> {
    use candle_core::quantized::QMatMul;
    let device = Device::new_cuda(0)?;
    let (n, k, m) = (512usize, 768usize, 64usize); // n % 128 == 0, k % 256 == 0, m > 8 -> MMQ

    for &dtype in &[
        GgmlDType::IQ2XXS,
        GgmlDType::IQ2XS,
        GgmlDType::IQ2S,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ3S,
        GgmlDType::IQ4XS,
    ] {
        let block_bytes = dtype.type_size();
        let row_blocks = k / dtype.block_size();
        let total = n * row_blocks * block_bytes;
        let mut raw: Vec<u8> = (0..total)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
            .collect();
        // d (half, offset 0 каждого блока) := 1.0 (0x3C00 LE)
        for b in 0..(n * row_blocks) {
            raw[b * block_bytes] = 0x00;
            raw[b * block_bytes + 1] = 0x3C;
        }
        let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), &device, dtype)?;
        let qt = std::sync::Arc::new(QTensor::new(storage, (n, k))?);
        let qmm = QMatMul::from_arc(qt.clone())?;

        let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let x_t = Tensor::from_vec(x, (1, m, k), &device)?;
        let got = qmm.forward(&x_t)?; // m=64 -> MMQ путь

        let w_deq = qt.dequantize(&device)?.to_dtype(DType::F32)?;
        let x_2d = x_t.reshape((m, k))?;
        let want = x_2d.matmul(&w_deq.t()?.contiguous()?)?.reshape((1, m, n))?;

        let diff = (&got - &want)?.abs()?;
        let max = diff.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?.max(1e-6);
        println!("MMQ-IQ {dtype:?}: max_abs_diff={max:.6} scale={scale:.4}");
        // Тот же класс, что K-quant MMQ (1.5%): IQ-тайлы деквантуются в q8
        // без потерь (grid-значения — малые int), доп. ошибки нет.
        assert!(
            max < 0.015 * scale,
            "MMQ-IQ {dtype:?} mismatch: max_abs_diff={max} vs scale={scale} (tolerance 1.5%)"
        );
    }
    Ok(())
}

/// Холодный MMVQ decode-бенч: 16 разных весов по ~70 MB (4.5 GB total > L2 3MB)
/// прогоняются round-robin → каждый вызов читает не из L2, как реальная модель.
/// Проверка гипотезы: разрыв mmvq в модели vs микробенч = L2-cache locality.
#[test]
#[ignore]
fn mmvq_perf_cold() -> Result<()> {
    use std::time::Instant;
    let device = Device::new_cuda(0)?;
    let cuda = match &device {
        Device::Cuda(c) => c.clone(),
        _ => unreachable!(),
    };

    // 16 разных весов [17408, 5120] Q4_K (~70 MB каждый, 1.1 GB total > 3MB L2)
    let n = 17408usize;
    let k = 5120usize;
    let row_bytes = k / 256 * std::mem::size_of::<candle_core::quantized::k_quants::BlockQ4K>();
    let mut weights = Vec::new();
    for i in 0..16 {
        let raw = vec![(i as u8).wrapping_mul(37).wrapping_add(0x5A); n * row_bytes];
        let storage = QStorage::from_data(std::borrow::Cow::Borrowed(&raw), &device, GgmlDType::Q4K)?;
        let qt = std::sync::Arc::new(QTensor::new(storage, (n, k))?);
        weights.push(QMatMul::from_arc(qt)?);
    }
    let y = Tensor::zeros((1, 1, k), DType::F32, &device)?;

    // Warmup
    for w in &weights { let _ = w.forward(&y)?; }
    cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;

    // Round-robin: каждый вызов — другой вес (L2-cold как в реальной модели)
    let t0 = Instant::now();
    let iters = 200;
    for it in 0..iters {
        let _ = weights[it % weights.len()].forward(&y)?;
    }
    cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
    let el = t0.elapsed().as_secs_f64() / iters as f64;
    let bytes = n * row_bytes;
    println!("MMVQ-COLD Q4K [{n}x{k}] x16 round-robin: {:.3}ms {:.0} GB/s",
        el * 1000.0, bytes as f64 / el / 1e9);

    // Hot (тот же вес) для сравнения
    let t0 = Instant::now();
    for _ in 0..iters { let _ = weights[0].forward(&y)?; }
    cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
    let el_h = t0.elapsed().as_secs_f64() / iters as f64;
    println!("MMVQ-HOT  Q4K [{n}x{k}] x1  hot:       {:.3}ms {:.0} GB/s",
        el_h * 1000.0, bytes as f64 / el_h / 1e9);
    Ok(())
}
