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
        other => bail!("неизвестный режим {other:?}; Фаза 1 поддерживает только layout"),
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
