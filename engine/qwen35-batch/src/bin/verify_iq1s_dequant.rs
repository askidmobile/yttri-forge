//! `qwen36_verify_iq1s_dequant` — сверка GPU dequant IQ-тензоров против
//! CPU-референса (пока только IQ1S) И проверка GPU-vs-GPU стабильности
//! декванта между прогонами в одном процессе (детекция гонки ядер).
//!
//! Запуск: qwen36_verify_iq1s_dequant <gguf-path>

use std::path::Path;

use candle_core::quantized::GgmlDType;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("usage: qwen36_verify_iq1s_dequant <gguf-path>");
    }
    let path = Path::new(&args[1]);

    let mut file = std::fs::File::open(path)?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

    let mut names: Vec<&String> = ct
        .tensor_infos
        .iter()
        .filter(|(_, info)| info.ggml_dtype == GgmlDType::IQ1S)
        .map(|(n, _)| n)
        .collect();
    names.sort();
    let mut checked = 0usize;
    let mut bad = 0usize;
    for name in &names {
        let info = &ct.tensor_infos[*name];
        let (start, size) = info.byte_range(ct.tensor_data_offset)?;
        let raw = &mmap[start..start + size];
        let elem_count = info.shape.elem_count();
        let cpu_ref = candle_core::quantized::iq1s::dequantize_iq1_s(raw, elem_count);

        // Три GPU-прогона на тензор: если хоть один даёт другой результат —
        // ядро декванта недетерминировано (гонка).
        let mut gpu_runs: Vec<Vec<f32>> = Vec::with_capacity(3);
        for _ in 0..3 {
            let qt_cuda = candle_core::quantized::ggml_file::qtensor_from_ggml(
                GgmlDType::IQ1S,
                raw,
                vec![elem_count],
                &candle_core::Device::new_cuda(0)?,
            )?;
            let gpu = qt_cuda
                .dequantize(&candle_core::Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            gpu_runs.push(gpu);
        }
        let mut max_diff = 0f32;
        let mut first_bad = 0usize;
        for (i, b) in gpu_runs[0].iter().enumerate() {
            let a = &cpu_ref[i];
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
                first_bad = i;
            }
        }
        let mut run_mismatch = 0usize;
        for r in &gpu_runs[1..] {
            for (i, b) in r.iter().enumerate() {
                if b.to_bits() != gpu_runs[0][i].to_bits() {
                    run_mismatch += 1;
                }
            }
        }
        let status = if max_diff == 0.0 && run_mismatch == 0 {
            "OK"
        } else {
            "DIFF"
        };
        println!(
            "{name}: {status} vs_cpu={max_diff:.3e} gpu_run_mismatch={run_mismatch} first={first_bad}"
        );
        if max_diff > 0.0 || run_mismatch > 0 {
            bad += 1;
        }
        checked += 1;
    }
    println!("итог: проверено {checked}, расхождений {bad}");
    Ok(())
}