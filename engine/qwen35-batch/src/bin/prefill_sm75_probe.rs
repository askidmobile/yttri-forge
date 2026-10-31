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
        other => bail!("неизвестный режим {other:?}; поддержаны: layout, fixed"),
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

/// Фаза 2: фиксированная форма — T=64, kv0=960, 16 страниц, без маски.
fn fixed() -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum;

    const T: usize = 64;
    const NH: usize = 16;
    const NKV: usize = 2;
    const HD: usize = 256;
    const KV0: usize = 960;
    const PAGES: usize = 16;
    const PAGE: usize = 64;

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let scale = 1.0f32 / (HD as f32).sqrt();

    let mut seed = 0x5eed_1234u64;
    let qf: Vec<f16> = (0..T * NH * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    let kf: Vec<f16> = (0..PAGES * PAGE * NKV * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    let vf: Vec<f16> = (0..PAGES * PAGE * NKV * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();

    println!("fixed: T={T} kv0={KV0} pages={PAGES} nh={NH} nkv={NKV} hd={HD}");

    // CPU-эталон: каждая строка видит все 1024 ключа (causal=0).
    let q32: Vec<f32> = qf.iter().map(|x| x.to_f32()).collect();
    let k32: Vec<f32> = kf.iter().map(|x| x.to_f32()).collect();
    let v32: Vec<f32> = vf.iter().map(|x| x.to_f32()).collect();
    let total_kv = PAGES * PAGE;
    let mut reference = vec![0f32; T * NH * HD];
    let group = NH / NKV;
    for row in 0..T {
        for h in 0..NH {
            let kvh = h / group;
            let q_base = (row * NH + h) * HD;
            let mut logits = vec![0f32; total_kv];
            for tok in 0..total_kv {
                let k_base = (tok * NKV + kvh) * HD;
                let mut acc = 0f32;
                for d in 0..HD {
                    acc += q32[q_base + d] * k32[k_base + d];
                }
                logits[tok] = acc * scale;
            }
            let mmax = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut l = 0f32;
            for x in logits.iter_mut() {
                *x = (*x - mmax).exp();
                l += *x;
            }
            let inv = 1.0 / l;
            for d in 0..HD {
                let mut acc = 0f32;
                for tok in 0..total_kv {
                    let v_base = (tok * NKV + kvh) * HD;
                    acc += logits[tok] * v32[v_base + d];
                }
                reference[q_base + d] = acc * inv;
            }
        }
    }

    let q_dev = cuda.clone_htod(&qf)?;
    let k_dev = cuda.clone_htod(&kf)?;
    let v_dev = cuda.clone_htod(&vf)?;
    let kv_len_dev = cuda.clone_htod(&[KV0 as u32])?;
    let slots_dev = cuda.clone_htod(&[0u32])?;
    let table: Vec<u32> = (0..PAGES as u32).collect();
    let table_dev = cuda.clone_htod(&table)?;
    let mut out_dev = unsafe { cuda.alloc::<f16>(T * NH * HD)? };

    let params = FlashPrefillSm75Params {
        n_head: NH as u32,
        n_kv: NKV as u32,
        hd: HD as u32,
        t: T as u32,
        b: 1,
        page_size: PAGE as u32,
        block_table_stride: PAGES as u32,
        causal: 0,
        scale,
    };

    let func = cuda.get_or_load_func("flash_prefill_sm75", &candle_kernels::FLASH_PREFILL_SM75)?;
    func.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        65536,
    )?;
    let cfg = LaunchConfig {
        grid_dim: ((T.div_ceil(64)) as u32, NH as u32, 1),
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
    let mut nan = 0usize;
    for i in 0..got.len() {
        let x = got[i].to_f32();
        if !x.is_finite() {
            nan += 1;
            continue;
        }
        let d = (x - reference[i]).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / got.len() as f64;
    println!("fixed: mean|d|={mean_abs:.3e} max|d|={max_abs:.3e} nan={nan}");
    if nan > 0 {
        bail!("fixed: {nan} не-конечных значений");
    }
    if mean_abs > 5e-4 || max_abs > 5e-3 {
        bail!("fixed mismatch: mean {mean_abs:.3e} > 5e-4 или max {max_abs:.3e} > 5e-3");
    }
    println!("fixed OK");
    Ok(())
}
