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
fn run(
    dev: &CudaDevice,
    kernel: &str,
    warps: u32,
    t: usize,
    iters: usize,
) -> Result<(f64, Vec<f32>)> {
    let p = params();
    let (n_v, hkd, hvd) = (N_V as usize, HKD as usize, HVD as usize);
    let q = dev.clone_htod(&fill(t * n_v * hkd, 1))?;
    let k = dev.clone_htod(&fill(t * n_v * hkd, 2))?;
    let v = dev.clone_htod(&fill(t * n_v * hvd, 3))?;
    let beta = dev.clone_htod(&fill(t * n_v, 4))?;
    let gate = dev.clone_htod(&fill(t * n_v, 5))?;
    let out = dev.alloc_zeros::<f32>(t * n_v * hvd)?;
    let state0 = fill(n_v * hvd * hvd, 6);

    // Ядро параметризовано по blockDim.y: warp ведёт одну колонку состояния,
    // блок — `warps` колонок. У v2 k/q грузятся в shared один раз на блок,
    // поэтому чем больше warps, тем меньше повторных чтений — но тем хуже
    // занятость SM. Развёртка ищет баланс.
    let cfg = LaunchConfig {
        grid_dim: (n_v as u32, (hvd / warps as usize) as u32, 1),
        block_dim: (32, warps, 1),
        shared_mem_bytes: if kernel.ends_with("_v2") {
            (2 * hkd * 4) as u32
        } else {
            0
        },
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
    dev.cuda_stream().synchronize()?;
    let reference = dev.clone_dtoh(&out)?;

    // H2D состояния (2 МБ) держим ВНЕ замера: внутри цикла оно добавляло
    // ~0.5 мс фиксированной платы и искажало масштабирование по T.
    let mut state = dev.clone_htod(&state0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch(&mut state)?;
    }
    dev.cuda_stream().synchronize()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    Ok((ms, reference))
}


// ────────────────────────────────────────────────────────────────────────────
// Эталоны на CPU: последовательный (как в v1-ядре) и chunked.
//
// Обозначения на голову: состояние S [hkd × hvd] (строки — key-размерность,
// столбцы — value). На токене t:
//     kv  = Sᵀ k_t                       S ← g_t S + k_t δᵀ
//     δ   = β_t (v_t − g_t kv)           o_t = S_tᵀ q_t   (S уже обновлён)
// где g_t = exp(gate_t).
//
// Chunked-форма. Пусть c_t = Σ_{i≤t} gate_i (лог-кумулята внутри блока),
// тогда G_t = exp(c_t) и S_t = G_t [S_0 + Σ_{i≤t} (k_i/G_i) δ_iᵀ]. Отсюда
//     δ_t = β_t ( v_t − S_0ᵀ k̂_t − Σ_{i<t} (k̃_iᵀ k̂_t) δ_i ),
// то есть треугольная система (I + diag(β) A) Δ = B (V − K̂ S_0), где
//     A[t][i] = exp(c_t − c_i) (k_i·k_t),  i < t.
// Выход и новое состояние:
//     o_t = G_t S_0ᵀ q_t + Σ_{i≤t} exp(c_t − c_i)(k_i·q_t) δ_i
//     S_C = exp(c_C) S_0 + Σ_i exp(c_C − c_i) k_i δ_iᵀ
//
// ВАЖНО про численность: k̃_i = k_i/G_i отдельно не считаем — при затухающем
// гейте 1/G_i растёт экспоненциально и переполняет f32. Все множители входят
// только как exp(c_t − c_i) при t ≥ i, то есть ≤ 1.
// ────────────────────────────────────────────────────────────────────────────

struct Dims {
    t: usize,
    n_v: usize,
    hkd: usize,
    hvd: usize,
}

fn ref_sequential(
    q: &[f32], k: &[f32], v: &[f32], beta: &[f32], gate: &[f32],
    s0: &[f32], d: &Dims, head: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (hkd, hvd, n_v) = (d.hkd, d.hvd, d.n_v);
    let mut s: Vec<f32> = s0[head * hkd * hvd..(head + 1) * hkd * hvd].to_vec();
    let mut out = vec![0f32; d.t * hvd];
    for t in 0..d.t {
        let g = (gate[t * n_v + head]).exp();
        let b = beta[t * n_v + head];
        let kb = (t * n_v + head) * hkd;
        let vb = (t * n_v + head) * hvd;
        let mut delta = vec![0f32; hvd];
        for col in 0..hvd {
            let mut kv = 0f32;
            for row in 0..hkd {
                kv += s[row * hvd + col] * k[kb + row];
            }
            delta[col] = (v[vb + col] - g * kv) * b;
        }
        for row in 0..hkd {
            let kr = k[kb + row];
            let qr = q[kb + row];
            for col in 0..hvd {
                let sv = g * s[row * hvd + col] + kr * delta[col];
                s[row * hvd + col] = sv;
                out[t * hvd + col] += sv * qr;
            }
        }
    }
    (out, s)
}

fn ref_chunked(
    q: &[f32], k: &[f32], v: &[f32], beta: &[f32], gate: &[f32],
    s0: &[f32], d: &Dims, head: usize, chunk: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (hkd, hvd, n_v) = (d.hkd, d.hvd, d.n_v);
    let mut s: Vec<f32> = s0[head * hkd * hvd..(head + 1) * hkd * hvd].to_vec();
    let mut out = vec![0f32; d.t * hvd];

    let mut t0 = 0usize;
    while t0 < d.t {
        let c = chunk.min(d.t - t0);
        // Лог-кумулята гейта внутри блока (inclusive).
        let mut clog = vec![0f32; c];
        let mut acc = 0f32;
        for i in 0..c {
            acc += gate[(t0 + i) * n_v + head];
            clog[i] = acc;
        }
        let krow = |i: usize| -> &[f32] {
            let b = ((t0 + i) * n_v + head) * hkd;
            &k[b..b + hkd]
        };
        let qrow = |i: usize| -> &[f32] {
            let b = ((t0 + i) * n_v + head) * hkd;
            &q[b..b + hkd]
        };

        // Δ: прямая подстановка по строкам блока.
        let mut delta = vec![0f32; c * hvd];
        for t in 0..c {
            let b_t = beta[(t0 + t) * n_v + head];
            let vb = ((t0 + t) * n_v + head) * hvd;
            let kt = krow(t);
            // W = β (V − G_t S_0ᵀ k_t)
            let gt = clog[t].exp();
            let mut w = vec![0f32; hvd];
            for col in 0..hvd {
                let mut s0k = 0f32;
                for row in 0..hkd {
                    s0k += s[row * hvd + col] * kt[row];
                }
                w[col] = b_t * (v[vb + col] - gt * s0k);
            }
            // − β Σ_{i<t} exp(c_t − c_i)(k_i·k_t) δ_i
            for i in 0..t {
                let ki = krow(i);
                let mut dot = 0f32;
                for r in 0..hkd {
                    dot += ki[r] * kt[r];
                }
                let coef = b_t * (clog[t] - clog[i]).exp() * dot;
                for col in 0..hvd {
                    w[col] -= coef * delta[i * hvd + col];
                }
            }
            delta[t * hvd..(t + 1) * hvd].copy_from_slice(&w);
        }

        // Выход блока.
        for t in 0..c {
            let qt = qrow(t);
            let gt = clog[t].exp();
            let ob = (t0 + t) * hvd;
            for col in 0..hvd {
                let mut acc = 0f32;
                for row in 0..hkd {
                    acc += s[row * hvd + col] * qt[row];
                }
                out[ob + col] = gt * acc;
            }
            for i in 0..=t {
                let ki = krow(i);
                let mut dot = 0f32;
                for r in 0..hkd {
                    dot += ki[r] * qt[r];
                }
                let coef = (clog[t] - clog[i]).exp() * dot;
                for col in 0..hvd {
                    out[ob + col] += coef * delta[i * hvd + col];
                }
            }
        }

        // Новое состояние блока.
        let gc = clog[c - 1];
        for row in 0..hkd {
            for col in 0..hvd {
                s[row * hvd + col] *= gc.exp();
            }
        }
        for i in 0..c {
            let ki = krow(i);
            let w = (gc - clog[i]).exp();
            for row in 0..hkd {
                let kw = w * ki[row];
                if kw == 0.0 {
                    continue;
                }
                for col in 0..hvd {
                    s[row * hvd + col] += kw * delta[i * hvd + col];
                }
            }
        }
        t0 += c;
    }
    (out, s)
}

/// Относительное расхождение. ВАЖНО: `f32::max(0, NaN)` возвращает 0, поэтому
/// сравнение двух испорченных векторов молча даёт «идеальный ноль» — сначала
/// проверяем финитность, иначе метрика врёт.
fn max_rel(a: &[f32], b: &[f32]) -> String {
    let bad_a = a.iter().filter(|x| !x.is_finite()).count();
    let bad_b = b.iter().filter(|x| !x.is_finite()).count();
    if bad_a > 0 || bad_b > 0 {
        return format!("НЕ-ФИНИТНО ({bad_a}/{bad_b} из {})", a.len());
    }
    if a.len() != b.len() {
        return format!("РАЗНАЯ ДЛИНА {} vs {}", a.len(), b.len());
    }
    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    let d = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale;
    format!("{d:.2e}")
}

/// Сверка математики: ядро против последовательного CPU-эталона и chunked.
fn check(dev: &CudaDevice, t: usize, chunk: usize) -> Result<()> {
    let (n_v, hkd, hvd) = (N_V as usize, HKD as usize, HVD as usize);
    let d = Dims { t, n_v, hkd, hvd };
    // Вход должен повторять реальный путь, иначе рекуррентность расходится и
    // сравнивать нечего: в модели k и q L2-нормированы (delta_l2_norm_prefill),
    // beta ∈ (0,1), гейт отрицательный (затухание).
    let l2 = |mut x: Vec<f32>, dim: usize| -> Vec<f32> {
        for row in x.chunks_mut(dim) {
            let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            for v in row.iter_mut() {
                *v /= n;
            }
        }
        x
    };
    let q = l2(fill(t * n_v * hkd, 1), hkd);
    let k = l2(fill(t * n_v * hkd, 2), hkd);
    let v = fill(t * n_v * hvd, 3);
    let beta = fill(t * n_v, 4)
        .iter()
        .map(|x| 1.0 / (1.0 + (-x * 4.0).exp()))
        .collect::<Vec<_>>();
    let gate = fill(t * n_v, 5).iter().map(|x| -(x.abs()) - 0.01).collect::<Vec<_>>();
    let s0 = fill(n_v * hvd * hvd, 6).iter().map(|x| x * 0.1).collect::<Vec<_>>();

    // GPU: текущее ядро.
    let p = params();
    let (dq, dk, dv) = (dev.clone_htod(&q)?, dev.clone_htod(&k)?, dev.clone_htod(&v)?);
    let (db, dg) = (dev.clone_htod(&beta)?, dev.clone_htod(&gate)?);
    let out = dev.alloc_zeros::<f32>(t * n_v * hvd)?;
    let mut state = dev.clone_htod(&s0)?;
    let func = dev.get_or_load_func("delta_rule_prefill", &candle_kernels::DELTA_RULE)?;
    let cfg = LaunchConfig {
        grid_dim: (N_V, (hvd / 2) as u32, 1),
        block_dim: (32, 2, 1),
        shared_mem_bytes: 0,
    };
    let t_u32 = t as u32;
    {
        let mut b = func.builder();
        b.arg(&dq);
        b.arg(&dk);
        b.arg(&dv);
        b.arg(&db);
        b.arg(&dg);
        b.arg(&state);
        b.arg(&out);
        b.arg(&p);
        b.arg(&t_u32);
        unsafe { b.launch(cfg) }.map_err(|e| anyhow!("launch: {e:?}"))?;
    }
    dev.cuda_stream().synchronize()?;
    let gpu_out = dev.clone_dtoh(&out)?;
    let gpu_state = dev.clone_dtoh(&state)?;

    let head = 0usize;
    let (seq_out, seq_state) = ref_sequential(&q, &k, &v, &beta, &gate, &s0, &d, head);
    let (chk_out, chk_state) = ref_chunked(&q, &k, &v, &beta, &gate, &s0, &d, head, chunk);

    // Срез головы 0 из GPU-выхода: [T, n_v, hvd] → [T, hvd].
    let mut gpu_head = vec![0f32; t * hvd];
    for tt in 0..t {
        let b = (tt * n_v + head) * hvd;
        gpu_head[tt * hvd..(tt + 1) * hvd].copy_from_slice(&gpu_out[b..b + hvd]);
    }
    let gpu_head_state = &gpu_state[head * hkd * hvd..(head + 1) * hkd * hvd];

    println!("T={t} chunk={chunk} (голова {head})");
    println!("  ядро vs последовательный CPU: out {}, state {}",
        max_rel(&gpu_head, &seq_out), max_rel(gpu_head_state, &seq_state));
    println!("  chunked vs последовательный:  out {}, state {}",
        max_rel(&seq_out, &chk_out), max_rel(&seq_state, &chk_state));
    println!("  масштаб: |out|max={:.3} |state|max={:.3}",
        seq_out.iter().fold(0f32, |m, x| m.max(x.abs())),
        seq_state.iter().fold(0f32, |m, x| m.max(x.abs())));
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args.next().unwrap_or_default();
    let check_mode = first == "check";
    let t: usize = if check_mode {
        args.next().and_then(|v| v.parse().ok()).unwrap_or(128)
    } else {
        first.parse().unwrap_or(512)
    };
    let iters: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(20);
    let Device::Cuda(dev) = Device::new_cuda(0)? else {
        return Err(anyhow!("нужен CUDA-девайс"));
    };
    if check_mode {
        for chunk in [16usize, 32, 64] {
            check(&dev, t, chunk)?;
        }
        return Ok(());
    }
    // FLOPs на запуск: на токен и голову — Sᵀk, обновление S и Sᵀq по hkd*hvd FMA.
    let flops = 3.0 * 2.0 * t as f64 * N_V as f64 * (HKD as f64) * (HVD as f64);
    println!("bench_delta: T={t} n_v={N_V} hkd={HKD} hvd={HVD}, {iters} итераций\n");

    let mut base: Option<Vec<f32>> = None;
    let variants: Vec<(&str, u32)> = vec![
        ("delta_rule_prefill", 2),
        ("delta_rule_prefill_v2", 8),
        // Диагностика: одна warp-редукция на токен вместо двух (математика
        // неверна, меряем только цену редукций).
        ("delta_rule_prefill_probe1", 2),
        // Диагностика: без редукций вообще.
        ("delta_rule_prefill_probe0", 2),
    ];
    for (kernel, warps) in variants {
        let kernel_label = format!("{kernel} warps={warps}");
        let kernel = kernel_label.split(' ').next().unwrap().to_string();
        match run(&dev, &kernel, warps, t, iters) {
            Ok((ms, outv)) => {
                let diff = base.as_ref().map(|b: &Vec<f32>| {
                    b.iter()
                        .zip(outv.iter())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max)
                });
                println!(
                    "  {kernel_label:<34} {ms:8.3} мс  {:6.2} TFLOPS{}",
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
            Err(e) => println!("  {kernel_label:<34} ОШИБКА: {e}"),
        }
    }
    println!("\nПик RTX 3060: FP32 ~12.7 TFLOPS.");
    Ok(())
}
