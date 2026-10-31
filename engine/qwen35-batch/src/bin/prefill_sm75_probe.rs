//! prefill_sm75_probe — пробник тайлового prefill-attention для Turing.
//!
//! План: docs/design/2026-10-31-sm75-prefill-kernel.md (§7.1, §8).
//! Фаза 1 — режим `layout`: проверяет ldmatrix-раскладки (A 16×8 через x2,
//! K→B через x1, V→B через x1.trans), свизл smem и `mma.m16n8k8` против
//! CPU-эталона; печатает атрибуты устройства (smem opt-in, число SM).
//!
//! Запуск: prefill_sm75_probe [layout]

use anyhow::{bail, Result};
use candle_core::Device;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use half::f16;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(|s| s.as_str()).unwrap_or("layout");
    match mode {
        "layout" => layout(),
        "fixed" => fixed(),
        "causal" => causal(),
        other => bail!("неизвестный режим {other:?}; поддержаны: layout, fixed, causal"),
    }
}

fn layout() -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let smem_optin = ctx.attribute(
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    )?;
    let nsm = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let cc_major =
        ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let cc_minor =
        ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    println!(
        "device: cc={cc_major}.{cc_minor} sm={nsm} smem_optin={smem_optin} B"
    );
    if cc_major < 7 || (cc_major == 7 && cc_minor < 5) {
        bail!("нужен sm_75+; устройство cc={cc_major}.{cc_minor}");
    }
    if smem_optin < 65536 {
        bail!("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN={smem_optin} < 65536 — план требует варианта C/D");
    }

    // Детерминированные входы; CPU считает по тем же F16-значениям.
    let a_h: Vec<f16> = (0..128)
        .map(|i| f16::from_f32((((i as i32 * 7) % 19) - 9) as f32 / 9.0))
        .collect();
    let k_h: Vec<f16> = (0..64)
        .map(|i| f16::from_f32((((i as i32 * 5) % 13) - 6) as f32 / 6.0))
        .collect();
    let v_h: Vec<f16> = (0..64)
        .map(|i| f16::from_f32((((i as i32 * 3) % 11) - 5) as f32 / 5.0))
        .collect();

    // CPU-эталон: qk[m][n] = Σ_k A[m][k]·K[n][k]; pv[m][d] = Σ_j A[m][j]·V[j][d]
    let af: Vec<f32> = a_h.iter().map(|x| x.to_f32()).collect();
    let kf: Vec<f32> = k_h.iter().map(|x| x.to_f32()).collect();
    let vf: Vec<f32> = v_h.iter().map(|x| x.to_f32()).collect();
    let mut qk_ref = vec![0f32; 128];
    let mut pv_ref = vec![0f32; 128];
    for m in 0..16 {
        for n in 0..8 {
            let mut acc = 0f32;
            for kk in 0..8 {
                acc += af[m * 8 + kk] * kf[n * 8 + kk];
            }
            qk_ref[m * 8 + n] = acc;
        }
        for d in 0..8 {
            let mut acc = 0f32;
            for j in 0..8 {
                acc += af[m * 8 + j] * vf[j * 8 + d];
            }
            pv_ref[m * 8 + d] = acc;
        }
    }

    let a_dev: CudaSlice<f16> = cuda.clone_htod(&a_h)?;
    let k_dev: CudaSlice<f16> = cuda.clone_htod(&k_h)?;
    let v_dev: CudaSlice<f16> = cuda.clone_htod(&v_h)?;
    let mut d_qk = unsafe { cuda.alloc::<f32>(128)? };
    let mut d_pv = unsafe { cuda.alloc::<f32>(128)? };

    let func = cuda.get_or_load_func(
        "flash_prefill_sm75_layout_probe",
        &candle_kernels::FLASH_PREFILL_SM75,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = func.builder();
    b.arg(&a_dev);
    b.arg(&k_dev);
    b.arg(&v_dev);
    b.arg(&mut d_qk);
    b.arg(&mut d_pv);
    unsafe { b.launch(cfg) }?;
    cuda.cuda_stream().synchronize()?;

    let qk = cuda.clone_dtoh(&d_qk)?;
    let pv = cuda.clone_dtoh(&d_pv)?;

    let mut max_qk = 0f32;
    let mut max_pv = 0f32;
    for i in 0..128 {
        if !qk[i].is_finite() || !pv[i].is_finite() {
            bail!("NaN/Inf в выходе ядра на индексе {i}: qk={} pv={}", qk[i], pv[i]);
        }
        max_qk = max_qk.max((qk[i] - qk_ref[i]).abs());
        max_pv = max_pv.max((pv[i] - pv_ref[i]).abs());
    }
    println!("layout: max|dQK|={max_qk:.3e} max|dPV|={max_pv:.3e}");
    const TOL: f32 = 2e-3;
    if max_qk > TOL || max_pv > TOL {
        bail!("layout mismatch: порог {TOL:.1e}; dQK={max_qk:.3e} dPV={max_pv:.3e}");
    }
    println!("layout OK");
    Ok(())
}


#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct FlashPrefillSm75Params {
    n_head: u32,
    n_kv: u32,
    hd: u32,
    t: u32,
    b: u32,
    page_size: u32,
    block_table_stride: u32,
    causal: u32,
    scale: f32,
}
unsafe impl cudarc::driver::DeviceRepr for FlashPrefillSm75Params {}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*seed >> 33) as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
}

/// Один прогон ядра против CPU-эталона. Возвращает (mean|Δ|, max|Δ|).
#[allow(clippy::too_many_arguments)]
fn run_case(
    t: usize,
    kv0: usize,
    causal: bool,
    nh: usize,
    nkv: usize,
    ref_heads: &[usize],
    seed_tag: u64,
) -> Result<(f64, f32)> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum;

    const HD: usize = 256;
    const PAGE: usize = 64;

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let scale = 1.0f32 / (HD as f32).sqrt();

    let pages = (kv0 + t).div_ceil(PAGE);
    let mut seed = 0x5eed_1234u64 ^ seed_tag.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let qf: Vec<f16> = (0..t * nh * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    let kf: Vec<f16> = (0..pages * PAGE * nkv * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    let vf: Vec<f16> = (0..pages * PAGE * nkv * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();

    // CPU-эталон (F32) только для выбранных голов — параллельно по головам.
    let q32: Vec<f32> = qf.iter().map(|x| x.to_f32()).collect();
    let k32: Vec<f32> = kf.iter().map(|x| x.to_f32()).collect();
    let v32: Vec<f32> = vf.iter().map(|x| x.to_f32()).collect();
    let group = nh / nkv;
    // Эталон по каждой голове отдельно, параллельно; наружу — (h, [t][HD]).
    let refs: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &h in ref_heads {
            let q32 = q32.as_slice();
            let k32 = k32.as_slice();
            let v32 = v32.as_slice();
            handles.push((
                h,
                scope.spawn(move || -> Vec<f32> {
                    let mut out = vec![0f32; t * HD];
                    let kvh = h / group;
                    for row in 0..t {
                        let pos = kv0 + row;
                        let q_base = (row * nh + h) * HD;
                        let n_vis = if causal { pos + 1 } else { pages * PAGE };
                        let mut logits = vec![0f32; n_vis];
                        for tok in 0..n_vis {
                            let k_base = (tok * nkv + kvh) * HD;
                            let mut acc = 0f32;
                            for d in 0..HD {
                                acc += q32[q_base + d] * k32[k_base + d];
                            }
                            logits[tok] = acc * scale;
                        }
                        let mmax = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut lsum = 0f32;
                        for x in logits.iter_mut() {
                            *x = (*x - mmax).exp();
                            lsum += *x;
                        }
                        let inv = 1.0 / lsum;
                        for d in 0..HD {
                            let mut acc = 0f32;
                            for tok in 0..n_vis {
                                let v_base = (tok * nkv + kvh) * HD;
                                acc += logits[tok] * v32[v_base + d];
                            }
                            out[row * HD + d] = acc * inv;
                        }
                    }
                    out
                }),
            ));
        }
        handles
            .into_iter()
            .map(|(h, j)| (h, j.join().expect("cpu reference thread")))
            .collect()
    });

    let q_dev = cuda.clone_htod(&qf)?;
    let k_dev = cuda.clone_htod(&kf)?;
    let v_dev = cuda.clone_htod(&vf)?;
    let kv_len_dev = cuda.clone_htod(&[kv0 as u32])?;
    let slots_dev = cuda.clone_htod(&[0u32])?;
    let table: Vec<u32> = (0..pages as u32).collect();
    let table_dev = cuda.clone_htod(&table)?;
    let mut out_dev = unsafe { cuda.alloc::<f16>(t * nh * HD)? };

    let params = FlashPrefillSm75Params {
        n_head: nh as u32,
        n_kv: nkv as u32,
        hd: HD as u32,
        t: t as u32,
        b: 1,
        page_size: PAGE as u32,
        block_table_stride: pages as u32,
        causal: causal as u32,
        scale,
    };

    let func = cuda.get_or_load_func("flash_prefill_sm75", &candle_kernels::FLASH_PREFILL_SM75)?;
    func.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        65536,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (t.div_ceil(64) as u32, nh as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 65536,
    };
    let mut bld = func.builder();
    bld.arg(&q_dev);
    bld.arg(&k_dev);
    bld.arg(&v_dev);
    bld.arg(&kv_len_dev);
    bld.arg(&slots_dev);
    bld.arg(&table_dev);
    bld.arg(&mut out_dev);
    bld.arg(&0u64); // lse = nullptr
    bld.arg(&params);
    unsafe { bld.launch(cfg) }?;
    cuda.cuda_stream().synchronize()?;

    let got = cuda.clone_dtoh(&out_dev)?;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut count = 0usize;
    for (h, ref_h) in &refs {
        for row in 0..t {
            let base = (row * nh + *h) * HD;
            for d in 0..HD {
                let x = got[base + d].to_f32();
                if !x.is_finite() {
                    bail!("не-конечное значение t={t} kv0={kv0} h={h} row={row} d={d}: {x}");
                }
                let diff = (x - ref_h[row * HD + d]).abs();
                max_abs = max_abs.max(diff);
                sum_abs += diff as f64;
                count += 1;
            }
        }
    }
    Ok((sum_abs / count as f64, max_abs))
}

/// Фаза 2: фиксированная форма — T=64, kv0=960, 16 страниц, без маски.
fn fixed() -> Result<()> {
    let (mean, max) = run_case(64, 960, false, 16, 2, &(0..16).collect::<Vec<_>>(), 0)?;
    println!("fixed: mean|d|={mean:.3e} max|d|={max:.3e} nan=0");
    if mean > 5e-4 || max > 5e-3 {
        bail!("fixed mismatch: mean {mean:.3e} max {max:.3e}");
    }
    println!("fixed OK");
    Ok(())
}

/// Фаза 3: причинность и границы kv0. Сравниваем головы 0 и 8
/// (kvh = 0 и 1), чтобы покрыть GQA и не гонять CPU-эталон по всем головам.
fn causal() -> Result<()> {
    let ts = [1usize, 16, 63, 64, 65, 127, 128];
    let kv0s = [0usize, 1, 63, 64, 65, 4113, 8191];
    let mut worst_mean = 0f64;
    let mut worst_max = 0f32;
    for &t in &ts {
        for &kv0 in &kv0s {
            let (mean, max) = run_case(t, kv0, true, 16, 2, &[0, 8], (t * 10_000 + kv0) as u64)?;
            worst_mean = worst_mean.max(mean);
            worst_max = worst_max.max(max);
            println!("causal: t={t:3} kv0={kv0:5} mean|d|={mean:.3e} max|d|={max:.3e}");
            if mean > 5e-4 || max > 5e-3 {
                bail!("causal mismatch at t={t} kv0={kv0}: mean {mean:.3e} max {max:.3e}");
            }
        }
    }
    println!("causal OK: worst mean={worst_mean:.3e} worst max={worst_max:.3e}");
    Ok(())
}
