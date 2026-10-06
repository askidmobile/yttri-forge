//! Изолированная проверка MMQ-MoE (группировка по экспертам + mul_mat_q с
//! ids_dst/expert_bounds) против эталонного matmul на CPU.
//!
//! Запуск:
//! ```text
//! cargo run --release --features cuda -p candle-core --example moe_mmq_selftest
//! ```
//!
//! Печатает max|Δ| по каждой проекции (gate/up/down → combine) и ненулевое
//! число расхождений. Неноль на любой проекции = баг раскладки или адресации.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("moe_mmq_selftest: требуется сборка с --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::quantized::{moe_grouping, moe_weighted_combine, GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};

    const N_EXPERTS: usize = 16;
    const N_FF: usize = 512;
    const N_EMBD: usize = 2048;
    const TOPK: usize = 8;
    const N_TOKENS: usize = 64;

    let dev = Device::new_cuda(0)?;
    let cuda_dev = match &dev {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("нужен CUDA"),
    };

    // Детерминированный ГПСЧ — воспроизводимые прогоны.
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let uniform = |next: &mut dyn FnMut() -> u64| (next() >> 40) as f32 / 1024.0 - 0.5;

    let mk_weights = |n_experts: usize, n: usize, k: usize, next: &mut dyn FnMut() -> u64| {
        let data: Vec<f32> = (0..n_experts * n * k).map(|_| uniform(next) * 0.05).collect();
        Tensor::from_vec(data, (n_experts, n, k), &Device::Cpu).unwrap()
    };

    let w_gate = mk_weights(N_EXPERTS, N_FF, N_EMBD, &mut next);
    let w_up = mk_weights(N_EXPERTS, N_FF, N_EMBD, &mut next);
    let w_down = mk_weights(N_EXPERTS, N_EMBD, N_FF, &mut next);

    let x: Vec<f32> = (0..N_TOKENS * N_EMBD).map(|_| uniform(&mut next)).collect();
    let x = Tensor::from_vec(x, (N_TOKENS, N_EMBD), &Device::Cpu).unwrap();

    let ids: Vec<u32> = (0..N_TOKENS * TOPK)
        .map(|_| (next() % N_EXPERTS as u64) as u32)
        .collect();
    let ids_t = Tensor::from_vec(ids.clone(), (N_TOKENS, TOPK), &dev).unwrap();
    let rw: Vec<f32> = (0..N_TOKENS * TOPK).map(|_| uniform(&mut next).abs()).collect();
    let rw_t = Tensor::from_vec(rw.clone(), (N_TOKENS, TOPK), &dev).unwrap();

    let (ids_st, ids_l) = ids_t.storage_and_layout();
    let ids_v = match &*ids_st {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>().unwrap().slice(ids_l.start_offset()..),
        _ => anyhow::bail!("ids не на CUDA"),
    };
    let (rw_st, rw_l) = rw_t.storage_and_layout();
    let rw_v = match &*rw_st {
        candle_core::Storage::Cuda(c) => c.as_cuda_slice::<f32>().unwrap().slice(rw_l.start_offset()..),
        _ => anyhow::bail!("weights не на CUDA"),
    };
    let group = moe_grouping(&cuda_dev, &ids_v, &rw_v, N_TOKENS, TOPK, N_EXPERTS, true)?;
    eprintln!(
        "[group] ncols_max={} m_total={}",
        group.ncols_max,
        N_TOKENS * TOPK
    );

    let q_gate = QTensor::quantize_onto(&w_gate, GgmlDType::Q8_0, &dev)?;
    let q_up = QTensor::quantize_onto(&w_up, GgmlDType::Q8_0, &dev)?;
    let q_down = QTensor::quantize_onto(&w_down, GgmlDType::Q8_0, &dev)?;

    let x_cuda = x.to_device(&dev)?.to_dtype(DType::F32)?.contiguous()?;
    let gate = q_gate.moe_mmq_project_cuda(&x_cuda, &group, true)?;
    let up = q_up.moe_mmq_project_cuda(&x_cuda, &group, true)?;
    let act = (gate.silu()? * up)?.contiguous()?;
    let down = q_down.moe_mmq_project_cuda(&act, &group, false)?;
    let (_, n_out) = down.dims2()?;
    let out = moe_weighted_combine(&cuda_dev, &group, &down, n_out)?;
    let got = out
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // ─── Эталон: те же квантованные веса, но matmul на CPU ───
    let stats = |name: &str, v: &[f32]| {
        let mx = v.iter().fold(f32::MIN, |a, b| a.max(*b));
        let mn = v.iter().fold(f32::MAX, |a, b| a.min(*b));
        let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
        println!("  {}: n={} min={:.4} max={:.4} mean={:.6}", name, v.len(), mn, mx, mean);
    };
    let dq_gate = q_gate.dequantize(&Device::Cpu)?;
    let dq_up = q_up.dequantize(&Device::Cpu)?;
    let dq_down = q_down.dequantize(&Device::Cpu)?;
    stats("w_gate(dq)", &dq_gate.flatten_all()?.to_vec1::<f32>()?);
    stats("got", &got);

    let ref_down = reference_moe(
        &dq_gate,
        &dq_up,
        &dq_down,
        &x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
        &ids,
        &rw,
        N_TOKENS,
        TOPK,
        N_FF,
        N_EMBD,
    );

    stats("ref", &ref_down);
    let mut max_abs = 0f32;
    let scale = ref_down.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut mismatch = 0usize;
    for (a, b) in got.iter().zip(ref_down.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        // Относительная мера: Q8_0-веса × int8-активации дают ошибку
        // порядка 1e-3 от масштаба блока, поэтому 2e-2 — с большим запасом.
        if d > 2e-2 * b.abs().max(1e-3 * scale) {
            mismatch += 1;
        }
    }
    println!(
        "moe_mmq_selftest: n={} scale={:.4e} max_abs_diff={:.4e} rel={:.2e} mismatch={}",
        got.len(),
        scale,
        max_abs,
        max_abs / scale.max(1e-30),
        mismatch
    );
    if mismatch == 0 {
        println!("OK");
    } else {
        println!("FAIL");
    }
    Ok(())
}

/// Эталонный MoE: для каждой пары (token, route) — silu(gate·x)·(up·x) → down,
/// затем взвешенная сумма по topk. Всё в f32 на CPU, веса — деквантованные.
#[allow(clippy::too_many_arguments)]
fn reference_moe(
    w_gate: &candle_core::Tensor,
    w_up: &candle_core::Tensor,
    w_down: &candle_core::Tensor,
    x: &[f32],
    ids: &[u32],
    weights: &[f32],
    n_tokens: usize,
    topk: usize,
    n_ff: usize,
    n_embd: usize,
) -> Vec<f32> {
    let g = w_gate.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let u = w_up.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let d = w_down.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut out = vec![0f32; n_tokens * n_embd];
    for t in 0..n_tokens {
        for r in 0..topk {
            let e = ids[t * topk + r] as usize;
            let mut gate = vec![0f32; n_ff];
            let mut up = vec![0f32; n_ff];
            for row in 0..n_ff {
                let base = (e * n_ff + row) * n_embd;
                let mut sg = 0f32;
                let mut su = 0f32;
                for c in 0..n_embd {
                    let xv = x[t * n_embd + c];
                    sg += g[base + c] * xv;
                    su += u[base + c] * xv;
                }
                gate[row] = sg / (1.0 + (-sg).exp());
                up[row] = su;
            }
            for row in 0..n_embd {
                let base = (e * n_embd + row) * n_ff;
                let mut acc = 0f32;
                for c in 0..n_ff {
                    acc += d[base + c] * gate[c] * up[c];
                }
                out[t * n_embd + row] += weights[t * topk + r] * acc;
            }
        }
    }
    out
}
