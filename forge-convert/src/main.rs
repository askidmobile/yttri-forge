//! forge-convert — конвертер BASE safetensors → .ytf16 (F16-сайдкар).
//!
//! Usage:
//!   forge-convert --f16-heavy <model.safetensors...> [--gguf <path>] [-o dir] [--list]
//!
//! Мультифайловые шарды передаются списком (порядок не важен — резолв по индексу).

pub mod container;
pub mod mask;

use clap::Parser;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "forge-convert", version)]
struct Args {
    /// safetensors файлы модели (BASE). Шарды — через пробел.
    #[arg(required_unless_present = "list")]
    inputs: Vec<PathBuf>,

    /// Режим heavy: 9 групп проекций DeltaNet+Attention
    #[arg(long)]
    f16_heavy: bool,

    /// GGUF рядом с которым положить .ytf16 (для имени и sha256)
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// Выходная директория (default: рядом с первым входом)
    #[arg(short, long)]
    out: Option<PathBuf>,

    /// Только показать найденные heavy-тензоры
    #[arg(long)]
    list: bool,
}

fn main() {
    let args = Args::parse();
    if !args.f16_heavy {
        eprintln!("error: сейчас поддерживается только --f16-heavy");
        std::process::exit(2);
    }
    if let Err(e) = run(&args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn open_safetensors(paths: &[PathBuf]) -> Result<Vec<(String, memmap2::Mmap)>, String> {
    let mut out = Vec::new();
    for p in paths {
        let file = File::open(p).map_err(|e| format!("open {}: {e}", p.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| format!("mmap {}: {e}", p.display()))?;
        let _ = safetensors::SafeTensors::read_metadata(&mmap)
            .map_err(|e| format!("{}: not safetensors: {e}", p.display()))?;
        out.push((p.display().to_string(), mmap));
    }
    Ok(out)
}

struct Found {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
    shard: usize,
}

fn find_tensor_info(
    shards: &[(String, memmap2::Mmap)],
    name: &str,
) -> Option<Found> {
    for (si, (_, mmap)) in shards.iter().enumerate() {
        if let Ok((header_len, meta)) = safetensors::SafeTensors::read_metadata(mmap) {
            if let Some(info) = meta.tensors().get(name).copied() {
                // data_offsets отсчитываются от НАЧАЛА СЕКЦИИ ДАННЫХ, а она идёт
                // после 8 байт длины заголовка и самого JSON. Без этой базы все
                // тензоры читаются со сдвигом на размер заголовка (веса-мусор:
                // распределение похоже, значения чужие).
                let base = header_len + 8;
                return Some(Found {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    start: base + info.data_offsets.0,
                    end: base + info.data_offsets.1,
                    shard: si,
                });
            }
        }
    }
    None
}

fn run(args: &Args) -> Result<(), String> {
    let shards = open_safetensors(&args.inputs)?;

    // Индекс всех тензоров
    let mut index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (_, mmap) in &shards {
        let (_off, meta) = safetensors::SafeTensors::read_metadata(mmap)
            .map_err(|e| format!("safetensors meta: {e}"))?;
        for (name, info) in meta.tensors() {
            index.insert(name.to_string(), vec![info.shape.len()]);
        }
    }
    println!("safetensors index: {} tensors; sample: {:?}", index.len(), index.keys().take(3).collect::<Vec<_>>());

    // Классифицируем слои
    let mut layers_delta: Vec<u32> = Vec::new();
    let mut layers_attn: Vec<u32> = Vec::new();
    for name in index.keys() {
        if let Some((idx, kind, _)) = mask::classify(name) {
            match kind {
                mask::LayerKind::DeltaNet => {
                    if !layers_delta.contains(&idx) {
                        layers_delta.push(idx);
                    }
                }
                mask::LayerKind::Attention => {
                    if !layers_attn.contains(&idx) {
                        layers_attn.push(idx);
                    }
                }
            }
        }
    }
    layers_delta.sort();
    layers_attn.sort();

    // Планируем список (st_name, layer_idx, gguf_name)
    struct Plan {
        st_name: String,
        layer: u32,
        gguf: String,
    }
    let mut plan: Vec<Plan> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for &i in &layers_delta {
        for suffix in [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.out_proj.weight",
        ] {
            let st_name = format!("model.language_model.layers.{i}.{suffix}");
            if let Some(gguf_t) = mask::resolve(mask::LayerKind::DeltaNet, suffix) {
                if index.contains_key(&st_name) {
                    plan.push(Plan {
                        st_name,
                        layer: i,
                        gguf: mask::gguf_name(i, gguf_t),
                    });
                } else {
                    missing.push(st_name);
                }
            }
        }
    }
    for &i in &layers_attn {
        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ] {
            let st_name = format!("model.language_model.layers.{i}.{suffix}");
            if let Some(gguf_t) = mask::resolve(mask::LayerKind::Attention, suffix) {
                if index.contains_key(&st_name) {
                    plan.push(Plan {
                        st_name,
                        layer: i,
                        gguf: mask::gguf_name(i, gguf_t),
                    });
                } else {
                    missing.push(st_name);
                }
            }
        }
    }

    println!(
        "heavy plan: {} delta layers ×5 + {} attn layers ×4 = {} tensors (missing {})",
        layers_delta.len(),
        layers_attn.len(),
        plan.len(),
        missing.len()
    );

    if args.list {
        for p in &plan {
            let shape = find_tensor_info(&shards, &p.st_name)
                .map(|f| format!("{:?}", f.shape))
                .unwrap_or_else(|| "?".into());
            println!("  {} → {} {shape}", p.st_name, p.gguf);
        }
        return Ok(());
    }

    // Выходной путь
    let gguf_path = args.gguf.clone().unwrap_or_else(|| {
        // Дефолт: ищем *.gguf рядом с первым входом
        let dir = args.inputs[0].parent().unwrap_or(Path::new("."));
        let mut found = None;
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                if e.path().extension().map(|x| x == "gguf").unwrap_or(false) {
                    found = Some(e.path());
                    break;
                }
            }
        }
        found.unwrap_or_else(|| PathBuf::from("model.gguf"))
    });

    let gguf_sha = {
        use sha2::{Digest, Sha256};
        let mut f = File::open(&gguf_path)
            .map_err(|e| format!("open gguf {}: {e}", gguf_path.display()))?;
        let mut h = Sha256::new();
        std::io::copy(&mut f, &mut h).map_err(|e| format!("hash gguf: {e}"))?;
        format!("{:x}", h.finalize())
    };

    let out_path = gguf_path.with_extension("ytf16");
    let out_file =
        File::create(&out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?;


    let pre = container::ManifestPre {
        gguf_sha256: gguf_sha.clone(),
        mask: "heavy".into(),
    };
    let mut w = container::ContainerWriter::create(out_file, pre).map_err(|e| format!("container create: {e}"))?;

    // Стриминг: для каждого планового тензора — чтение шарда, cast→F16
    let t0 = std::time::Instant::now();
    let mut total_bytes = 0u64;
    for p in &plan {
        let found = find_tensor_info(&shards, &p.st_name)
            .ok_or_else(|| format!("tensor vanished: {}", p.st_name))?;
        w.note_dtype(format!("{:?}", found.dtype).as_str());
        let shape: Vec<usize> = found.shape.clone();
        let n_elem: usize = shape.iter().product();
        let data: &[u8] = &shards[found.shard].1[found.start..found.end];

        let f16_bytes = match found.dtype {
            safetensors::Dtype::BF16 => {
                // BF16 → F32 (расширение) → clamp → F16
                assert_eq!(data.len(), n_elem * 2);
                let mut clamped = 0u64;
                let mut f16le: Vec<u8> = Vec::with_capacity(n_elem * 2);
                for chunk in data.chunks_exact(2) {
                    let bf = u16::from_le_bytes([chunk[0], chunk[1]]);
                    // BF16→F32: байты в старшую половину
                    let f32v = f32::from_bits((bf as u32) << 16);
                    let cv = f32v.clamp(-65504.0, 65504.0);
                    if cv != f32v {
                        clamped += 1;
                    }
                    let hv = half::f16::from_f32(cv);
                    f16le.extend_from_slice(&hv.to_le_bytes());
                }
                w.note_clamped(clamped);
                f16le
            }
            safetensors::Dtype::F16 => data.to_vec(),
            safetensors::Dtype::F32 => {
                assert_eq!(data.len(), n_elem * 4);
                let mut f16le: Vec<u8> = Vec::with_capacity(n_elem * 2);
                for chunk in data.chunks_exact(4) {
                    let fv = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let hv = half::f16::from_f32(fv.clamp(-65504.0, 65504.0));
                    f16le.extend_from_slice(&hv.to_le_bytes());
                }
                f16le
            }
            other => {
                return Err(format!(
                    "{}: dtype {other:?} не поддерживается (нужен BASE BF16/F16/F32)",
                    p.st_name
                ))
            }
        };
        total_bytes += f16_bytes.len() as u64;
        w.add_tensor(&p.gguf, &shape, f16_bytes);
    }

    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "wrote {} : {} tensors, {:.2} GB, {:.1}s ({:.0} MB/s), gguf_sha256={}",
        out_path.display(),
        plan.len(),
        total_bytes as f64 / 1e9,
        secs,
        total_bytes as f64 / 1e6 / secs.max(1e-9),
        &gguf_sha[..12.min(gguf_sha.len())]
    );
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Смещения тензора должны отсчитываться от секции данных, а не от начала
    /// файла: иначе конвертер пишет в сайдкар чужие байты (баг 2026-08-25).
    #[test]
    fn tensor_offsets_are_relative_to_data_section() {
        let values = [1.0f32, -2.0, 3.5, 4.25];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![2, 2],
            &data,
        )
        .expect("view");
        let bytes = safetensors::serialize([("w", view)], &None).expect("serialize");
        let path = std::env::temp_dir().join("forge-convert-offsets-test.safetensors");
        std::fs::write(&path, &bytes).expect("write");

        let shards = open_safetensors(&[path.clone()]).expect("open");
        let found = find_tensor_info(&shards, "w").expect("tensor found");
        assert_eq!(&shards[found.shard].1[found.start..found.end], data.as_slice());

        std::fs::remove_file(&path).ok();
    }
}
