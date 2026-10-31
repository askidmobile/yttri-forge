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
        "paged" => paged(),
        "parity" => parity(),
        other => bail!("неизвестный режим {other:?}; поддержаны: layout, fixed, causal, paged, parity"),
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

/// Параметры одного прогона ядра против CPU-эталона.
struct Case<'a> {
    t: usize,
    kv0: usize,
    causal: bool,
    nh: usize,
    nkv: usize,
    ref_heads: &'a [usize],
    seed_tag: u64,
    shuffle: bool,
    poison: bool,
    inplace: bool,
    want_lse: bool,
    batch: usize,
}

const HD: usize = 256;
const PAGE: usize = 64;

struct CaseOut {
    /// [batch·t, nh, hd] F32.
    got: Vec<f32>,
    /// (h, out[t][hd], lse[t]) для сверки.
    refs: Vec<(usize, Vec<f32>, Vec<f32>)>,
    /// [nh, batch·t] F32 с устройства (если want_lse).
    lse: Option<Vec<f32>>,
}

fn run_case(c: &Case<'_>) -> Result<CaseOut> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute_enum;

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?.clone();
    let scale = 1.0f32 / (HD as f32).sqrt();
    let pages = (c.kv0 + c.t).div_ceil(PAGE);

    let (qf, kf, vf, table) =
        gen_vectors(c.t, c.kv0, c.nh, c.nkv, c.seed_tag, c.poison, c.shuffle);

    // CPU-эталон по выбранным головам, параллельно.
    let q32: Vec<f32> = qf.iter().map(|x| x.to_f32()).collect();
    let k32: Vec<f32> = kf.iter().map(|x| x.to_f32()).collect();
    let v32: Vec<f32> = vf.iter().map(|x| x.to_f32()).collect();
    let group = c.nh / c.nkv;
    let refs: Vec<(usize, Vec<f32>, Vec<f32>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &h in c.ref_heads {
            let q32 = q32.as_slice();
            let k32 = k32.as_slice();
            let v32 = v32.as_slice();
            let table = table.as_slice();
            let (t, kv0, causal, nkv, nh) = (c.t, c.kv0, c.causal, c.nkv, c.nh);
            handles.push((
                h,
                scope.spawn(move || -> (Vec<f32>, Vec<f32>) {
                    let mut out = vec![0f32; t * HD];
                    let mut lse = vec![0f32; t];
                    let kvh = h / group;
                    for row in 0..t {
                        let pos = kv0 + row;
                        let q_base = (row * nh + h) * HD;
                        let n_vis = if causal { pos + 1 } else { pages * PAGE };
                        let phys = |tok: usize| -> usize {
                            (table[tok / PAGE] as usize * PAGE + tok % PAGE) * nkv * HD
                        };
                        let mut logits = vec![0f32; n_vis];
                        for tok in 0..n_vis {
                            let k_base = phys(tok) + kvh * HD;
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
                        lse[row] = mmax + lsum.ln();
                        for d in 0..HD {
                            let mut acc = 0f32;
                            for tok in 0..n_vis {
                                let v_base = phys(tok) + kvh * HD;
                                acc += logits[tok] * v32[v_base + d];
                            }
                            out[row * HD + d] = acc * inv;
                        }
                    }
                    (out, lse)
                }),
            ));
        }
        handles
            .into_iter()
            .map(|(h, j)| {
                let (out, lse) = j.join().expect("cpu reference thread");
                (h, out, lse)
            })
            .collect()
    });

    let q_dev = cuda.clone_htod(&qf)?;
    let k_dev = cuda.clone_htod(&kf)?;
    let v_dev = cuda.clone_htod(&vf)?;
    let kv_len_host: Vec<u32> = vec![c.kv0 as u32; c.batch];
    let kv_len_dev = cuda.clone_htod(&kv_len_host)?;
    let slots_host: Vec<u32> = (0..c.batch as u32).collect();
    let slots_dev = cuda.clone_htod(&slots_host)?;
    let mut table_host: Vec<u32> = Vec::with_capacity(c.batch * pages);
    for _ in 0..c.batch {
        table_host.extend_from_slice(&table);
    }
    let table_dev = cuda.clone_htod(&table_host)?;
    let mut out_dev = unsafe { cuda.alloc::<f16>(c.batch * c.t * c.nh * HD)? };
    let mut lse_dev = if c.want_lse {
        Some(unsafe { cuda.alloc::<f32>(c.nh * c.batch * c.t)? })
    } else {
        None
    };

    let params = FlashPrefillSm75Params {
        n_head: c.nh as u32,
        n_kv: c.nkv as u32,
        hd: HD as u32,
        t: c.t as u32,
        b: c.batch as u32,
        page_size: PAGE as u32,
        block_table_stride: pages as u32,
        causal: c.causal as u32,
        scale,
    };

    let func = cuda.get_or_load_func("flash_prefill_sm75", &candle_kernels::FLASH_PREFILL_SM75)?;
    func.set_attribute(
        CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        65536,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (c.t.div_ceil(64) as u32, c.nh as u32, c.batch as u32),
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
    if c.inplace {
        bld.arg(&q_dev);
    } else {
        bld.arg(&mut out_dev);
    }
    match lse_dev.as_mut() {
        Some(l) => {
            bld.arg(l);
        }
        None => {
            bld.arg(&0u64);
        }
    }
    bld.arg(&params);
    unsafe { bld.launch(cfg) }?;
    cuda.cuda_stream().synchronize()?;

    let raw: Vec<f16> = if c.inplace {
        cuda.clone_dtoh(&q_dev)?
    } else {
        cuda.clone_dtoh(&out_dev)?
    };
    let got: Vec<f32> = raw.iter().map(|x| x.to_f32()).collect();
    let lse = match lse_dev.as_ref() {
        Some(l) => Some(cuda.clone_dtoh(l)?.to_vec()),
        None => None,
    };
    Ok(CaseOut { got, refs, lse })
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Сверка CaseOut с CPU-эталоном (и LSE, если запрошена).
fn check(c: &Case<'_>, o: &CaseOut) -> Result<(f64, f32)> {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut count = 0usize;
    for (h, ref_h, _) in &o.refs {
        for b in 0..c.batch {
            for row in 0..c.t {
                let base = ((b * c.t + row) * c.nh + *h) * HD;
                for d in 0..HD {
                    let x = o.got[base + d];
                    if !x.is_finite() {
                        bail!(
                            "не-конечное значение t={} kv0={} b={b} h={h} row={row} d={d}: {x}",
                            c.t,
                            c.kv0
                        );
                    }
                    let diff = (x - ref_h[row * HD + d]).abs();
                    max_abs = max_abs.max(diff);
                    sum_abs += diff as f64;
                    count += 1;
                }
            }
        }
    }
    if let Some(lse_host) = o.lse.as_deref() {
        for (h, _, lse_ref) in &o.refs {
            for b in 0..c.batch {
                for row in 0..c.t {
                    let idx = h * (c.batch * c.t) + b * c.t + row;
                    let diff = (lse_host[idx] - lse_ref[row]).abs();
                    if diff > 5e-2 {
                        bail!(
                            "lse mismatch h={h} b={b} row={row}: got {} ref {}",
                            lse_host[idx],
                            lse_ref[row]
                        );
                    }
                }
            }
        }
    }
    Ok((sum_abs / count as f64, max_abs))
}

/// Фаза 2: фиксированная форма — T=64, kv0=960, 16 страниц, без маски.
fn fixed() -> Result<()> {
    let heads: Vec<usize> = (0..16).collect();
    let c = Case {
        t: 64, kv0: 960, causal: false, nh: 16, nkv: 2,
        ref_heads: &heads, seed_tag: 0, shuffle: false, poison: false,
        inplace: false, want_lse: false, batch: 1,
    };
    let o = run_case(&c)?;
    let (mean, max) = check(&c, &o)?;
    println!("fixed: mean|d|={mean:.3e} max|d|={max:.3e} nan=0");
    if mean > 5e-4 || max > 5e-3 {
        bail!("fixed mismatch: mean {mean:.3e} max {max:.3e}");
    }
    println!("fixed OK");
    Ok(())
}

/// Фаза 3: причинность и границы kv0.
fn causal() -> Result<()> {
    let ts = [1usize, 16, 63, 64, 65, 127, 128];
    let kv0s = [0usize, 1, 63, 64, 65, 4113, 8191];
    let heads = [0usize, 8];
    let mut worst_mean = 0f64;
    let mut worst_max = 0f32;
    for &t in &ts {
        for &kv0 in &kv0s {
            let c = Case {
                t, kv0, causal: true, nh: 16, nkv: 2, ref_heads: &heads,
                seed_tag: (t * 10_000 + kv0) as u64, shuffle: false,
                poison: false, inplace: false, want_lse: false, batch: 1,
            };
            let o = run_case(&c)?;
            let (mean, max) = check(&c, &o)?;
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

/// Фаза 4: пагинация (перестановка), NaN-хвост, геометрии, b=2, lse, in-place.
fn paged() -> Result<()> {
    let mut step = |name: &str, c: &Case<'_>| -> Result<()> {
        let o = run_case(c)?;
        let (mean, max) = check(c, &o)?;
        println!("paged/{name}: mean|d|={mean:.3e} max|d|={max:.3e}");
        if mean > 5e-4 || max > 5e-3 {
            bail!("{name} mismatch: mean {mean:.3e} max {max:.3e}");
        }
        Ok(())
    };

    let h2 = [0usize, 8];
    step("shuffle+poison", &Case {
        t: 64, kv0: 65, causal: true, nh: 16, nkv: 2, ref_heads: &h2,
        seed_tag: 11, shuffle: true, poison: true, inplace: false,
        want_lse: false, batch: 1,
    })?;
    let h4 = [0usize, 4, 8, 12];
    step("geom16/4", &Case {
        t: 64, kv0: 65, causal: true, nh: 16, nkv: 4, ref_heads: &h4,
        seed_tag: 12, shuffle: true, poison: false, inplace: false,
        want_lse: false, batch: 1,
    })?;
    step("geom24/4", &Case {
        t: 65, kv0: 70, causal: true, nh: 24, nkv: 4, ref_heads: &h4,
        seed_tag: 13, shuffle: true, poison: false, inplace: false,
        want_lse: false, batch: 1,
    })?;
    step("batch2", &Case {
        t: 32, kv0: 100, causal: true, nh: 16, nkv: 2, ref_heads: &h2,
        seed_tag: 14, shuffle: true, poison: false, inplace: false,
        want_lse: false, batch: 2,
    })?;
    step("inplace", &Case {
        t: 64, kv0: 65, causal: true, nh: 16, nkv: 2, ref_heads: &h2,
        seed_tag: 15, shuffle: true, poison: false, inplace: true,
        want_lse: false, batch: 1,
    })?;

    // LSE отдельным шагом (check сверяет o.lse с CPU-эталоном).
    let c = Case {
        t: 64, kv0: 65, causal: true, nh: 16, nkv: 2, ref_heads: &h2,
        seed_tag: 16, shuffle: false, poison: false, inplace: false,
        want_lse: true, batch: 1,
    };
    let o = run_case(&c)?;
    let (mean, max) = check(&c, &o)?;
    println!("paged/lse: mean|d|={mean:.3e} max|d|={max:.3e} lse OK");
    println!("paged OK");
    Ok(())
}

/// Детерминированные входы кейса: q [t·nh·hd], пул k/v [pages·64·nkv·hd],
/// таблица страниц (опционально перемешанная). NaN-хвост — по флагу poison.
fn gen_vectors(
    t: usize,
    kv0: usize,
    nh: usize,
    nkv: usize,
    seed_tag: u64,
    poison: bool,
    shuffle: bool,
) -> (Vec<f16>, Vec<f16>, Vec<f16>, Vec<u32>) {
    let pages = (kv0 + t).div_ceil(PAGE);
    let mut seed = 0x5eed_1234u64 ^ seed_tag.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let qf: Vec<f16> = (0..t * nh * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    let mut kf: Vec<f16> = (0..pages * PAGE * nkv * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    let mut vf: Vec<f16> = (0..pages * PAGE * nkv * HD)
        .map(|_| f16::from_f32(0.35 * lcg(&mut seed)))
        .collect();
    if poison {
        for tok in (kv0 + t)..(pages * PAGE) {
            for kvh in 0..nkv {
                for d in 0..HD {
                    let idx = (tok * nkv + kvh) * HD + d;
                    kf[idx] = f16::NAN;
                    vf[idx] = f16::NAN;
                }
            }
        }
    }
    let mut table: Vec<u32> = (0..pages as u32).collect();
    if shuffle {
        // table[i] = физическая страница, где лежит логический тайл i.
        let mut st = seed_tag ^ 0xdead_beef;
        for i in (1..pages).rev() {
            let j = (next_u64(&mut st) % (i as u64 + 1)) as usize;
            table.swap(i, j);
        }
        // Содержимое обязано переехать вместе с таблицей: физическая
        // страница table[i] должна хранить логический тайл i.
        let page_elems = PAGE * nkv * HD;
        let mut kf_phys = vec![f16::from_f32(0.0); kf.len()];
        let mut vf_phys = vec![f16::from_f32(0.0); vf.len()];
        for (logical, &physical) in table.iter().enumerate() {
            let src = logical * page_elems;
            let dst = physical as usize * page_elems;
            kf_phys[dst..dst + page_elems].copy_from_slice(&kf[src..src + page_elems]);
            vf_phys[dst..dst + page_elems].copy_from_slice(&vf[src..src + page_elems]);
        }
        kf = kf_phys;
        vf = vf_phys;
    }
    (qf, kf, vf, table)
}

/// Фаза 4b: паритет с пер-строчным decode-путём (эталон дизайна §7).
fn parity() -> Result<()> {
    use candle_core::{DType, Tensor};
    use qwen35_batch::real::flash_decode_cuda::dispatch_flash_decode_paged;

    let nh = 16usize;
    let nkv = 2usize;
    let heads = [0usize];
    let scale = 1.0f32 / (HD as f32).sqrt();
    let cases = [(65usize, 0usize), (128, 4113), (505, 0)];
    for (idx, &(t, kv0)) in cases.iter().enumerate() {
        let tag = 1000 + idx as u64;
        let (qf, kf, vf, table) = gen_vectors(t, kv0, nh, nkv, tag, false, false);
        let pages = (kv0 + t).div_ceil(PAGE);
        let case = Case {
            t, kv0, causal: true, nh, nkv, ref_heads: &heads, seed_tag: tag,
            shuffle: false, poison: false, inplace: false, want_lse: false, batch: 1,
        };
        let o = run_case(&case)?;
        let ours = o.got.clone();

        let dev = Device::new_cuda(0)?;
        let q_t = Tensor::from_vec(qf, (t, nh, HD), &dev)?.to_dtype(DType::F16)?;
        let k_t = Tensor::from_vec(kf, (pages, PAGE, nkv, HD), &dev)?.to_dtype(DType::F16)?;
        let v_t = Tensor::from_vec(vf, (pages, PAGE, nkv, HD), &dev)?.to_dtype(DType::F16)?;
        let mut seqlens = Vec::with_capacity(t + 1);
        let mut acc = 0u32;
        seqlens.push(0u32);
        for i in 0..t {
            acc += (kv0 + i + 1) as u32;
            seqlens.push(acc);
        }
        let sl = Tensor::from_vec(seqlens, (t + 1,), &dev)?;
        let bt = Tensor::from_vec(table, (1, pages), &dev)?
            .broadcast_as((t, pages))?
            .contiguous()?;
        let decoded = dispatch_flash_decode_paged(
            dev.as_cuda_device()?,
            &q_t,
            &k_t,
            &v_t,
            &sl,
            &bt,
            scale,
        )?;
        let dec = decoded.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let mut max_abs = 0f32;
        let mut sum_abs = 0f64;
        for i in 0..ours.len() {
            let d = (ours[i] - dec[i]).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
        }
        let mean = sum_abs / ours.len() as f64;
        // Контроль: decode-путь против CPU-эталона (голова 0).
        let (h0, ref_out, _) = &o.refs[0];
        let mut d_cpu = 0f32;
        for row in 0..t {
            for d in 0..HD {
                let dec_v = dec[(row * nh + *h0) * HD + d];
                d_cpu = d_cpu.max((dec_v - ref_out[row * HD + d]).abs());
            }
        }
        let mean_dec: f64 =
            (0..t * HD).map(|i| dec[((i / HD) * nh + *h0) * HD + i % HD].abs() as f64).sum::<f64>()
                / (t * HD) as f64;
        let mean_ref: f64 =
            ref_out.iter().map(|x| x.abs() as f64).sum::<f64>() / ref_out.len() as f64;
        println!(
            "parity: t={t:3} kv0={kv0:4} mean|d|={mean:.3e} max|d|={max_abs:.3e} dec_vs_cpu_h0={d_cpu:.3e} |dec|={mean_dec:.3e} |cpu|={mean_ref:.3e}"
        );
        if mean > 5e-4 || max_abs > 5e-3 {
            eprintln!("WARN parity mismatch t={t} kv0={kv0}: mean {mean:.3e} max {max_abs:.3e}");
        }
    }
    println!("parity done");
    Ok(())
}
