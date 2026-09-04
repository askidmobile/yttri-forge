//! Ворота 0 плана docs/plans/2026-09-04-moe-expert-offload.md:
//! эффективная полоса чтения весов экспертов боевыми ядрами
//! `indexed_moe_forward[_dual]_<dtype>_q8_1` из pinned RAM, отображённой в
//! адресное пространство устройства (zero-copy), против VRAM-копии тех же
//! байтов.
//!
//! Что проверяется (§6/§13 спеки 2026-09-04-moe-expert-offload):
//! 1. Потолок отображаемой host-памяти под WDDM: весь объём экспертов модели
//!    (≈8.5 ГБ для 35B-A3B IQ2_XXS) аллоцируется как pinned DEVICEMAP, по
//!    буферу на (слой, матрица) — PD-006.
//! 2. Отображение в контексте cudarc: `cuMemHostGetDevicePointer_v2` для
//!    каждого буфера; совпадение с host-указателем при UVA — в отчёт.
//! 3. Полоса по фактическому времени боевого ядра (T=1, k=8, batch=1), не
//!    memcpy: тот же запуск на VRAM-копии и на host-mapped буфере. Вход
//!    квантуется в Q8_1 штатным ядром `quantize_q8_1`, как в
//!    `indexed_moe_forward_dispatch`.
//! 4. Флаги аллокации: проход без WRITECOMBINED и с ним — выбор по замеру.
//!
//! Запуск на yttri-win:
//! ```text
//! cargo run --release --features cuda -p candle-core --example moe_zero_copy_bench ^
//!     -- D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf ^
//!     [--iters 8] [--quick] [--no-wc]
//! ```
//! Вердикт: полоса host-mapped ≥ 8 ГБ/с — ворота пройдены; иначе пересмотр
//! D-005 (стейджинг с синком на слой) до начала остальной работы.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("moe_zero_copy_bench: требуется сборка с --features cuda (yttri-win)");
}

#[cfg(feature = "cuda")]
mod bench {
    use anyhow::{bail, Context, Result};
    use candle_core::cuda_backend::{cudarc, kernels, CudaDevice, WrapErr};
    use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
    use candle_core::quantized::{gguf_file, GgmlDType};
    use candle_core::Device;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    type CuDevPtr = cudarc::driver::sys::CUdeviceptr;
    type CuStream = cudarc::driver::CudaStream;
    type CuSliceU8 = cudarc::driver::CudaSlice<u8>;

    const WARP: u32 = 32; // WARP_SIZE боевых ядер
    const TOPK: usize = 8; // активные эксперты Qwen3-MoE
    const READ_CHUNK: usize = 8 * 1024 * 1024;
    // ABI-константы диспетчера (quantized/cuda.rs: CUDA_QUANTIZE_BLOCK_SIZE,
    // MATRIX_ROW_PADDING).
    const QUANTIZE_BLOCK: usize = 256;
    const ROW_PADDING: usize = 512;

    fn ceil_div(p: usize, q: usize) -> usize {
        p.div_ceil(q)
    }

    fn pad(p: usize, q: usize) -> usize {
        (p + q - 1) / q * q
    }

    /// Pinned host-буфер, отображённый в адресное пространство устройства.
    /// Освобождается в Drop через `cuMemFreeHost`.
    struct HostMapped {
        ptr: *mut u8,
        dev_ptr: CuDevPtr,
        bytes: usize,
        /// dev_ptr == host ptr (UVA). Для корректности не требуется, идёт в отчёт.
        uva_same: bool,
    }

    impl HostMapped {
        /// flags: CU_MEMHOSTALLOC_DEVICEMAP [+ CU_MEMHOSTALLOC_WRITECOMBINED]
        unsafe fn alloc(bytes: usize, flags: u32) -> Result<Self> {
            let raw = cudarc::driver::result::malloc_host(bytes, flags).map_err(|e| {
                anyhow::anyhow!("malloc_host({bytes} байт, flags={flags:#x}) не удался: {e:?}")
            })?;
            if raw.is_null() {
                bail!("malloc_host({bytes} байт, flags={flags:#x}) вернул null");
            }
            let ptr = raw as *mut u8;
            let mut dev_ptr: CuDevPtr = 0;
            let rc =
                unsafe { cudarc::driver::sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, raw, 0) };
            if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                bail!("cuMemHostGetDevicePointer_v2: {rc:?}");
            }
            Ok(Self {
                ptr,
                dev_ptr,
                bytes,
                uva_same: dev_ptr as usize == ptr as usize,
            })
        }

        /// Запись байтов из файла в pinned-буфер.
        fn fill_from(&mut self, file: &mut std::fs::File, start: u64) -> Result<()> {
            file.seek(SeekFrom::Start(start))?;
            let buf = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) };
            let mut done = 0usize;
            while done < self.bytes {
                let n = self.bytes - done;
                let chunk = n.min(READ_CHUNK);
                file.read_exact(&mut buf[done..done + chunk])?;
                done += chunk;
            }
            Ok(())
        }
    }

    impl Drop for HostMapped {
        fn drop(&mut self) {
            unsafe {
                let _ = cudarc::driver::result::free_host(self.ptr as *mut _);
            }
        }
    }

    struct MatSpec {
        name: String,
        dtype: GgmlDType,
        /// Форма упаковки [n_experts, n, k]: n строк по row_bytes на эксперта.
        n: usize,
        k: usize,
        n_experts: usize,
        expert_bytes: usize,
        file_start: u64,
        /// Размер тензора в файле (byte_range возвращает (offset, размер)).
        file_bytes: usize,
        host: Option<HostMapped>,
    }

    struct LayerSpec {
        idx: usize,
        gate: MatSpec,
        up: MatSpec,
        down: MatSpec,
    }

    /// Имена боевых ядер — как в `indexed_moe_forward_dispatch`.
    fn q81_kernel_name(dtype: GgmlDType) -> Result<String> {
        Ok(match dtype {
            GgmlDType::IQ2XXS => "indexed_moe_forward_iq2_xxs_q8_1",
            GgmlDType::IQ2XS => "indexed_moe_forward_iq2_xs_q8_1",
            GgmlDType::IQ2S => "indexed_moe_forward_iq2_s_q8_1",
            GgmlDType::IQ3XXS => "indexed_moe_forward_iq3_xxs_q8_1",
            GgmlDType::IQ3S => "indexed_moe_forward_iq3_s_q8_1",
            GgmlDType::IQ4XS => "indexed_moe_forward_iq4_xs_q8_1",
            GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
            GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
            GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
            GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
            GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
            GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
            other => bail!("indexed_moe_forward не поддерживает {other:?}"),
        }
        .to_string())
    }

    fn dual_kernel_name(dtype: GgmlDType) -> Result<String> {
        Ok(match dtype {
            GgmlDType::IQ2XXS => "indexed_moe_forward_dual_iq2_xxs_q8_1",
            GgmlDType::IQ2XS => "indexed_moe_forward_dual_iq2_xs_q8_1",
            GgmlDType::IQ2S => "indexed_moe_forward_dual_iq2_s_q8_1",
            GgmlDType::IQ3XXS => "indexed_moe_forward_dual_iq3_xxs_q8_1",
            GgmlDType::IQ3S => "indexed_moe_forward_dual_iq3_s_q8_1",
            GgmlDType::IQ4XS => "indexed_moe_forward_dual_iq4_xs_q8_1",
            GgmlDType::Q2K => "indexed_moe_forward_dual_q2k_q8_1",
            GgmlDType::Q4K => "indexed_moe_forward_dual_q4k_q8_1",
            GgmlDType::Q6K => "indexed_moe_forward_dual_q6k_q8_1",
            GgmlDType::Q8_0 => "indexed_moe_forward_dual_q8_0_q8_1",
            other => bail!("indexed_moe_forward_dual не поддерживает {other:?}"),
        }
        .to_string())
    }

    fn mib(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0)
    }

    fn gbps(bytes: usize, secs: f64) -> f64 {
        if secs <= 0.0 {
            return 0.0;
        }
        bytes as f64 / secs / 1e9
    }

    struct Args {
        model: String,
        iters: usize,
        quick: bool,
        with_wc: bool,
        flush: bool,
    }

    fn parse_args() -> Result<Args> {
        let mut it = std::env::args().skip(1);
        let mut model = None;
        let mut iters = 8usize;
        let mut quick = false;
        let mut with_wc = true;
        let mut flush = true;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--iters" => {
                    iters = it
                        .next()
                        .context("--iters: нужно число")?
                        .parse()
                        .context("--iters: не число")?;
                }
                "--quick" => quick = true,
                "--no-wc" => with_wc = false,
                "--no-flush" => flush = false,
                other if model.is_none() => model = Some(other.to_string()),
                other => bail!("неизвестный аргумент {other}"),
            }
        }
        let model = model.context(
            "укажите путь к GGUF: moe_zero_copy_bench <model.gguf> [--iters N] [--quick]",
        )?;
        Ok(Args {
            model,
            iters,
            quick,
            with_wc,
            flush,
        })
    }

    /// Слой из имени тензора `blk.<N>.ffn_gate_exps[.weight]`.
    fn layer_of(name: &str, base: &str) -> Option<usize> {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let rest = stem.strip_suffix(base)?;
        let rest = rest.strip_prefix("blk.")?;
        rest.trim_end_matches('.').parse().ok()
    }

    fn collect_layers(content: &gguf_file::Content) -> Result<Vec<LayerSpec>> {
        use std::collections::BTreeMap;
        let mut gates: BTreeMap<usize, MatSpec> = BTreeMap::new();
        let mut ups: BTreeMap<usize, MatSpec> = BTreeMap::new();
        let mut downs: BTreeMap<usize, MatSpec> = BTreeMap::new();
        for (name, info) in content.tensor_infos.iter() {
            let (base, which) = if name.contains("ffn_gate_exps") {
                ("ffn_gate_exps", 0)
            } else if name.contains("ffn_up_exps") {
                ("ffn_up_exps", 1)
            } else if name.contains("ffn_down_exps") {
                ("ffn_down_exps", 2)
            } else {
                continue;
            };
            let layer = match layer_of(name, base) {
                Some(l) => l,
                None => continue, // nextn и прочие префиксы не трогаем
            };
            let (n_experts, n, k) = info.shape.dims3().with_context(|| {
                format!("{name}: ожидается форма [n_experts, n, k], получено {:?}", info.shape)
            })?;
            let (start, size) = content
                .tensor_byte_range(name)
                .with_context(|| format!("{name}: диапазон байтов"))?;
            let spec = MatSpec {
                name: name.clone(),
                dtype: info.ggml_dtype,
                n,
                k,
                n_experts,
                expert_bytes: n * (k / info.ggml_dtype.block_size() * info.ggml_dtype.type_size()),
                file_start: start as u64,
                file_bytes: size,
                host: None,
            };
            let slot = match which {
                0 => &mut gates,
                1 => &mut ups,
                _ => &mut downs,
            };
            slot.insert(layer, spec);
        }
        if gates.is_empty() {
            bail!("в файле нет тензоров ffn_gate_exps — это не MoE-модель?")
        }
        let mut layers = Vec::new();
        for (idx, gate) in gates {
            let up = ups
                .remove(&idx)
                .with_context(|| format!("blk.{idx}: нет ffn_up_exps"))?;
            let down = downs
                .remove(&idx)
                .with_context(|| format!("blk.{idx}: нет ffn_down_exps"))?;
            layers.push(LayerSpec { idx, gate, up, down });
        }
        Ok(layers)
    }

    /// Раскиданные id экспертов, чтобы не читать один эксперт 8 раз из L2.
    fn spread_ids(n_experts: usize) -> Vec<u32> {
        (0..TOPK)
            .map(|i| ((i * n_experts / TOPK) as u32).min(n_experts as u32 - 1))
            .collect()
    }

    /// Q8_1-квантование входа штатным ядром quantize_q8_1 — как в
    /// `indexed_moe_forward_dispatch`. input: rows×k f32; возвращает rows×k_padded
    /// байтов Q8_1.
    fn quantize_q8_1(
        dev: &CudaDevice,
        stream: &CuStream,
        input: &cudarc::driver::CudaSlice<f32>,
        rows: usize,
        k: usize,
    ) -> Result<CuSliceU8> {
        let k_padded = pad(k, ROW_PADDING);
        let q8_1_type_size = GgmlDType::Q8_1.type_size();
        let q8_1_block = GgmlDType::Q8_1.block_size();
        let row_bytes = k_padded / q8_1_block * q8_1_type_size;
        let mut dst = unsafe { dev.alloc::<u8>(rows * row_bytes)? };
        let func = dev.get_or_load_func("quantize_q8_1", &kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (ceil_div(k_padded, QUANTIZE_BLOCK) as u32, rows as u32, 1),
            block_dim: (QUANTIZE_BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (k_i, kpad_i) = (k as i32, k_padded as i32);
        let mut b = func.builder();
        b.arg(input).arg(&mut dst);
        b.arg(&k_i).arg(&kpad_i);
        unsafe { b.launch(cfg) }.w()?;
        stream.synchronize().w()?;
        Ok(dst)
    }

    struct TimedSide {
        secs: f64,
    }

    /// Замер боевого dual-ядра (gate+up одним запуском): VRAM и host-mapped.
    /// Тайминг по событиям вокруг ядра; между итерациями — продувка L2 (memset
    /// буфера больше L2), чтобы повторные запуски не читали из кэша.
    #[allow(clippy::too_many_arguments)]
    fn bench_dual(
        dev: &CudaDevice,
        stream: &std::sync::Arc<CuStream>,
        mut flush: Option<&mut CuSliceU8>,
        dtype: GgmlDType,
        w1: &CuSliceU8,
        w2: &CuSliceU8,
        mapped1: Option<CuDevPtr>,
        mapped2: Option<CuDevPtr>,
        q8_1: &CuSliceU8,
        ids: &cudarc::driver::CudaSlice<u32>,
        out1: &mut cudarc::driver::CudaSlice<f32>,
        out2: &mut cudarc::driver::CudaSlice<f32>,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Result<(TimedSide, Option<TimedSide>)> {
        let kernel = dual_kernel_name(dtype)?;
        let func = dev.get_or_load_func(&kernel, &kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, TOPK as u32),
            block_dim: (WARP, 4, 1),
            shared_mem_bytes: 0,
        };
        let (n_i, k_i, batch_i, topk_i, kpad_i, dim1_i) =
            (n as i32, k as i32, 1i32, TOPK as i32, pad(k, ROW_PADDING) as i32, TOPK as i32);

        let vram = {
            let mut b = func.builder();
            b.arg(w1).arg(w2).arg(q8_1).arg(ids).arg(&mut *out1).arg(&mut *out2);
            b.arg(&n_i).arg(&k_i).arg(&batch_i).arg(&topk_i).arg(&kpad_i).arg(&dim1_i);
            unsafe { b.launch(cfg) }.w()?;
            stream.synchronize().w()?;
            let mut ms = 0f32;
            for _ in 0..iters {
                if let Some(f) = flush.as_mut() {
                    stream.memset_zeros(&mut **f).w()?;
                }
                let ev0 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                unsafe { b.launch(cfg) }.w()?;
                let ev1 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                stream.synchronize().w()?;
                ms += ev0.elapsed_ms(&ev1)?;
            }
            ms as f64 / 1000.0 / iters as f64
        };

        let mapped = match (mapped1, mapped2) {
            (Some(d1), Some(d2)) => {
                let mut b = func.builder();
                b.arg(&d1).arg(&d2).arg(q8_1).arg(ids).arg(&mut *out1).arg(&mut *out2);
                b.arg(&n_i).arg(&k_i).arg(&batch_i).arg(&topk_i).arg(&kpad_i).arg(&dim1_i);
                unsafe { b.launch(cfg) }.w()?;
                stream.synchronize().w()?;
                let mut ms = 0f32;
                for _ in 0..iters {
                    if let Some(f) = flush.as_mut() {
                        stream.memset_zeros(&mut **f).w()?;
                    }
                    let ev0 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    unsafe { b.launch(cfg) }.w()?;
                    let ev1 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    stream.synchronize().w()?;
                    ms += ev0.elapsed_ms(&ev1)?;
                }
                Some(ms as f64 / 1000.0 / iters as f64)
            }
            _ => None,
        };

        Ok((
            TimedSide { secs: vram },
            mapped.map(|secs| TimedSide { secs }),
        ))
    }

    /// Замер боевого plain-ядра (down): VRAM и host-mapped.
    #[allow(clippy::too_many_arguments)]
    fn bench_plain(
        dev: &CudaDevice,
        stream: &std::sync::Arc<CuStream>,
        mut flush: Option<&mut CuSliceU8>,
        dtype: GgmlDType,
        w: &CuSliceU8,
        mapped: Option<CuDevPtr>,
        q8_1: &CuSliceU8,
        ids: &cudarc::driver::CudaSlice<u32>,
        out: &mut cudarc::driver::CudaSlice<f32>,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Result<(TimedSide, Option<TimedSide>)> {
        let kernel = q81_kernel_name(dtype)?;
        let func = dev.get_or_load_func(&kernel, &kernels::QUANTIZED)?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (n as u32, 1, TOPK as u32),
            block_dim: (WARP, 4, 1),
            shared_mem_bytes: 0,
        };
        let (n_i, k_i, batch_i, topk_i, kpad_i, dim1_i) =
            (n as i32, k as i32, 1i32, TOPK as i32, pad(k, ROW_PADDING) as i32, 1i32);

        let vram = {
            let mut b = func.builder();
            b.arg(w).arg(q8_1).arg(ids).arg(&mut *out);
            b.arg(&n_i).arg(&k_i).arg(&batch_i).arg(&topk_i).arg(&kpad_i).arg(&dim1_i);
            unsafe { b.launch(cfg) }.w()?;
            stream.synchronize().w()?;
            let mut ms = 0f32;
            for _ in 0..iters {
                if let Some(f) = flush.as_mut() {
                    stream.memset_zeros(&mut **f).w()?;
                }
                let ev0 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                unsafe { b.launch(cfg) }.w()?;
                let ev1 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                stream.synchronize().w()?;
                ms += ev0.elapsed_ms(&ev1)?;
            }
            ms as f64 / 1000.0 / iters as f64
        };

        let mapped = match mapped {
            Some(dp) => {
                let mut b = func.builder();
                b.arg(&dp).arg(q8_1).arg(ids).arg(&mut *out);
                b.arg(&n_i).arg(&k_i).arg(&batch_i).arg(&topk_i).arg(&kpad_i).arg(&dim1_i);
                unsafe { b.launch(cfg) }.w()?;
                stream.synchronize().w()?;
                let mut ms = 0f32;
                for _ in 0..iters {
                    if let Some(f) = flush.as_mut() {
                        stream.memset_zeros(&mut **f).w()?;
                    }
                    let ev0 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    unsafe { b.launch(cfg) }.w()?;
                    let ev1 = stream.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
                    stream.synchronize().w()?;
                    ms += ev0.elapsed_ms(&ev1)?;
                }
                Some(ms as f64 / 1000.0 / iters as f64)
            }
            None => None,
        };

        Ok((
            TimedSide { secs: vram },
            mapped.map(|secs| TimedSide { secs }),
        ))
    }

    /// Один проход: аллокация всех pinned-буферов с данными флагами, замер
    /// всех слоёв, освобождение. Возвращает суммарные байты/секунды по сторонам.
    fn run_pass(
        args: &Args,
        file: &mut std::fs::File,
        dev: &CudaDevice,
        stream: &std::sync::Arc<CuStream>,
        layers: &mut [LayerSpec],
        flags: u32,
        label: &str,
    ) -> Result<(usize, f64, usize, f64)> {
        let total: usize = layers
            .iter()
            .map(|l| l.gate.file_bytes + l.up.file_bytes + l.down.file_bytes)
            .sum();
        println!(
            "[gate0] аллокация pinned: {:.0} МиБ, {} буферов, flags={label} ({flags:#x})",
            mib(total),
            layers.len() * 3,
        );
        let t0 = Instant::now();
        for layer in layers.iter_mut() {
            for m in [&mut layer.gate, &mut layer.up, &mut layer.down] {
                let mut host = unsafe { HostMapped::alloc(m.file_bytes, flags)? };
                host.fill_from(file, m.file_start)?;
                m.host = Some(host);
            }
        }
        let alloc_secs = t0.elapsed().as_secs_f64();
        let mut uva_same = 0usize;
        let mut total_check = 0usize;
        for layer in layers.iter() {
            for m in [&layer.gate, &layer.up, &layer.down] {
                let h = m.host.as_ref().expect("pinned буфер");
                total_check += 1;
                if h.uva_same {
                    uva_same += 1;
                }
                if h.dev_ptr == 0 {
                    bail!("{}: dev_ptr == 0 — отображение не работает", m.name);
                }
            }
        }
        println!(
            "[gate0] pinned готов за {alloc_secs:.1} с; cuMemHostGetDevicePointer: OK, UVA dev==host: {}/{}",
            uva_same, total_check,
        );

        let mut bytes_v = 0usize;
        let mut secs_v = 0f64;
        let mut bytes_m = 0usize;
        let mut secs_m = 0f64;

        // Продувка L2 между итерациями: буфер заведомо больше L2 (2-3 МиБ у 3060).
        let mut flush_buf = if args.flush {
            Some(dev.alloc_zeros::<u8>(256 * 1024 * 1024)?)
        } else {
            None
        };

        for layer in layers.iter_mut() {
            if args.quick && layer.idx >= 4 {
                break;
            }
            let dt = layer.gate.dtype;
            // VRAM-копии матриц слоя.
            let mut load = |m: &MatSpec| -> Result<CuSliceU8> {
                let mut v = vec![0u8; m.file_bytes];
                file.seek(SeekFrom::Start(m.file_start))?;
                file.read_exact(&mut v)?;
                let mut wslice = unsafe { dev.alloc::<u8>(m.file_bytes)? };
                dev.memcpy_htod(&v, &mut wslice)?;
                Ok(wslice)
            };
            let gate_w = load(&layer.gate)?;
            let up_w = load(&layer.up)?;
            let down_w = load(&layer.down)?;

            // Входы и выходы боевых вызовов.
            let ids_v = spread_ids(layer.gate.n_experts);
            let mut ids = unsafe { dev.alloc::<u32>(TOPK)? };
            dev.memcpy_htod(&ids_v, &mut ids)?;

            // dual: вход [batch=1, topk=8, k_gate] — каждая строка своего эксперта.
            let x_gu = vec![0.001f32; TOPK * layer.gate.k];
            let mut x_gu_dev = unsafe { dev.alloc::<f32>(TOPK * layer.gate.k)? };
            dev.memcpy_htod(&x_gu, &mut x_gu_dev)?;
            let q_gu = quantize_q8_1(dev, stream, &x_gu_dev, TOPK, layer.gate.k)?;
            let mut out1 = dev.alloc_zeros::<f32>(TOPK * layer.gate.n)?;
            let mut out2 = dev.alloc_zeros::<f32>(TOPK * layer.gate.n)?;

            // plain: вход [batch=1, dim1=1, k_down] — один вектор активаций FFN.
            let x_d = vec![0.001f32; layer.down.k];
            let mut x_d_dev = unsafe { dev.alloc::<f32>(layer.down.k)? };
            dev.memcpy_htod(&x_d, &mut x_d_dev)?;
            let q_d = quantize_q8_1(dev, stream, &x_d_dev, 1, layer.down.k)?;
            let mut out_d = dev.alloc_zeros::<f32>(TOPK * layer.down.n)?;

            let (tv_dual, tm_dual) = bench_dual(
                dev, stream, flush_buf.as_mut(), dt, &gate_w, &up_w,
                layer.gate.host.as_ref().map(|h| h.dev_ptr),
                layer.up.host.as_ref().map(|h| h.dev_ptr),
                &q_gu, &ids, &mut out1, &mut out2,
                layer.gate.n, layer.gate.k, args.iters,
            )?;
            let (tv_down, tm_down) = bench_plain(
                dev, stream, flush_buf.as_mut(), dt, &down_w,
                layer.down.host.as_ref().map(|h| h.dev_ptr),
                &q_d, &ids, &mut out_d,
                layer.down.n, layer.down.k, args.iters,
            )?;

            let traffic_dual = TOPK * (layer.gate.expert_bytes + layer.up.expert_bytes);
            let traffic_down = TOPK * layer.down.expert_bytes;
            let layer_bytes = traffic_dual + traffic_down;
            let secs_v_layer = tv_dual.secs + tv_down.secs;
            let secs_m_layer = match (tm_dual.as_ref(), tm_down.as_ref()) {
                (Some(m1), Some(m2)) => Some(m1.secs + m2.secs),
                _ => None,
            };
            let gm_line = match secs_m_layer {
                Some(s) => format!("{:.2} ГБ/с", gbps(layer_bytes, s)),
                None => "-".into(),
            };
            println!(
                "[gate0]   blk.{}: VRAM {:.2} ГБ/с | mapped {}",
                layer.idx,
                gbps(layer_bytes, secs_v_layer),
                gm_line
            );
            bytes_v += layer_bytes;
            secs_v += secs_v_layer;
            if let Some(s) = secs_m_layer {
                bytes_m += layer_bytes;
                secs_m += s;
            }
        }

        // Освобождение pinned перед следующим проходом.
        for layer in layers.iter_mut() {
            layer.gate.host = None;
            layer.up.host = None;
            layer.down.host = None;
        }
        Ok((bytes_v, secs_v, bytes_m, secs_m))
    }

    pub fn run() -> Result<()> {
        let args = parse_args()?;
        let mut file = std::fs::File::open(&args.model)
            .with_context(|| format!("не открыть {}", args.model))?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut layers = collect_layers(&content)?;
        let total: usize = layers
            .iter()
            .map(|l| l.gate.file_bytes + l.up.file_bytes + l.down.file_bytes)
            .sum();
        let g = &layers[0].gate;
        println!(
            "[gate0] модель: {} | слоёв MoE: {} | экспертов/слой: {} | topk: {} | dtype: {:?}",
            args.model,
            layers.len(),
            g.n_experts,
            TOPK,
            g.dtype,
        );
        println!(
            "[gate0] гейт/ап: n={} k={} ({:.2} КиБ/эксперт), даун: n={} k={} | всего экспертов: {:.0} МиБ",
            g.n,
            g.k,
            g.expert_bytes as f64 / 1024.0,
            layers[0].down.n,
            layers[0].down.k,
            mib(total),
        );
        let device = Device::new_cuda(0).context("CUDA устройство 0")?;
        let dev = device.as_cuda_device()?.clone();
        let stream = dev.cuda_stream();

        let mut passes: Vec<(u32, &str)> =
            vec![(cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP, "DEVICEMAP")];
        if args.with_wc {
            passes.push((
                cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP
                    | cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED,
                "DEVICEMAP|WRITECOMBINED",
            ));
        }

        let mut report = Vec::new();
        for (flags, label) in passes {
            let (bv, sv, bm, sm) =
                run_pass(&args, &mut file, &dev, &stream, &mut layers, flags, label)?;
            let (gv, gm) = (gbps(bv, sv), gbps(bm, sm));
            println!(
                "[gate0] === итог {label}: VRAM {gv:.2} ГБ/с | mapped {gm:.2} ГБ/с (ratio {:.2})",
                if sv > 0.0 { gm / gv } else { 0.0 }
            );
            report.push((label, gv, gm));
        }

        let best = report
            .iter()
            .copied()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .expect("есть проходы");
        println!(
            "[gate0] ВЕРДИКТ: лучшая полоса mapped {:.2} ГБ/с ({}) при VRAM {:.2} ГБ/с",
            best.2, best.0, best.1
        );
        if best.2 >= 8.0 {
            println!("[gate0] GATE 0: PASS (порог 8 ГБ/с). Флаги: {}.", best.0);
        } else {
            println!(
                "[gate0] GATE 0: FAIL ({:.2} < 8 ГБ/с) — пересмотреть D-005 (стейджинг с синком на слой).",
                best.2
            );
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(e) = bench::run() {
        eprintln!("[gate0] ОШИБКА: {e:#}");
        std::process::exit(1);
    }
}
