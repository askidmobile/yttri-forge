//! forge-convert — конвертер BASE safetensors → .ytf16 (F16-сайдкар).
//!
//! Usage:
//!   forge-convert --f16-heavy <model.safetensors...> [--gguf <path>] [-o dir] [--list]
//!
//! Мультифайловые шарды передаются списком (порядок не важен — резолв по индексу).

pub mod container;
pub mod mask;
pub mod pack;

use clap::Parser;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "forge-convert", version)]
struct Args {
    /// safetensors файлы модели (BASE). Шарды — через пробел.
    #[arg(required_unless_present_any = ["list", "verify_ytf"])]
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

    /// Эмулировать FP8 (E4M3, поблочный масштаб) поверх F16-хранения:
    /// проверка качества без нативных FP8-ядер (их нет на Ampere).
    #[arg(long)]
    fp8_emulate: bool,

    /// Размер блока масштабирования для --fp8-emulate
    #[arg(long, default_value_t = 128)]
    fp8_block: usize,

    /// Эмулировать Q8_0 (int8, блок 32, масштаб F16) — сравнение с FP8
    #[arg(long)]
    q8_emulate: bool,

    /// Включить в сайдкар FFN (ffn_gate/up/down каждого слоя). По профилю это
    /// 31.7% времени префилла; цена — размер сайдкара примерно ×2.
    #[arg(long)]
    f16_ffn: bool,

    /// Собрать САМОСТОЯТЕЛЬНЫЙ контейнер .ytf (v2) вместо сайдкара: вся
    /// языковая модель, типы тензоров и метаданные с эталонного GGUF,
    /// токенизатор внутри. Рантайму GGUF после этого не нужен.
    #[arg(long)]
    pack: bool,

    /// tokenizer.json для встраивания (умолчание — рядом с первым входом)
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Сверить готовый .ytf с эталонным GGUF потензорно (значения, не только
    /// имена и формы) и напечатать худшие расхождения.
    #[arg(long)]
    verify_ytf: Option<PathBuf>,

    /// Дополнительно записать слитую проекцию qkv+z+b+a одним тензором
    /// (дополненным до кратности 128). Движок использует её, если найдёт.
    #[arg(long)]
    fuse_in_proj: bool,
}

fn main() {
    let args = Args::parse();
    if let Some(ytf) = args.verify_ytf.clone() {
        let Some(gguf) = args.gguf.clone() else {
            eprintln!("error: --verify-ytf требует --gguf <эталон>");
            std::process::exit(2);
        };
        if let Err(e) = pack::verify(&ytf, &gguf, 15) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if !args.f16_heavy && !args.pack {
        eprintln!("error: нужен --f16-heavy (сайдкар) или --pack (самостоятельный контейнер)");
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

/// Округление до E4M3 (1-4-3, max 448) и обратно в f32.
/// Субнормали прижаты к 2^-9; NaN/inf → 0 (в весах их не бывает).
pub fn e4m3_round(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return 0.0;
    }
    let a = x.abs().min(448.0);
    if a < 2f32.powi(-9) {
        return 0.0;
    }
    // Экспонента с полом на минимальной нормали E4M3 (2^-6): ниже — субнормали
    // с фиксированным шагом.
    let e = a.log2().floor().max(-6.0);
    let step = (e - 3.0).exp2(); // 3 бита мантиссы
    let q = (a / step).round() * step;
    q.min(448.0).copysign(x)
}

/// Поблочное FP8-квантование (E4M3) F16-буфера: блок = `block` подряд идущих
/// значений, масштаб = amax/448. Возвращает (новые байты, сумма |Δ|, сумма |w|).
pub fn fp8_emulate_f16(bytes: &[u8], block: usize) -> (Vec<u8>, f64, f64) {
    let n = bytes.len() / 2;
    let mut vals: Vec<f32> = Vec::with_capacity(n);
    for c in bytes.chunks_exact(2) {
        vals.push(half::f16::from_le_bytes([c[0], c[1]]).to_f32());
    }
    let (mut sum_err, mut sum_abs) = (0f64, 0f64);
    let mut out = Vec::with_capacity(bytes.len());
    for chunk in vals.chunks(block) {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 448.0 } else { 1.0 };
        for &v in chunk {
            let q = e4m3_round(v / scale) * scale;
            sum_err += (v - q).abs() as f64;
            sum_abs += v.abs() as f64;
            out.extend_from_slice(&half::f16::from_f32(q).to_le_bytes());
        }
    }
    (out, sum_err, sum_abs)
}

/// Q8_0-эмуляция: блок 32 значения, масштаб amax/127, целые уровни.
/// Считает ту же метрику, что и fp8_emulate_f16 — для сравнения форматов.
pub fn q8_0_emulate_f16(bytes: &[u8], block: usize) -> (Vec<u8>, f64, f64) {
    let n = bytes.len() / 2;
    let mut vals: Vec<f32> = Vec::with_capacity(n);
    for c in bytes.chunks_exact(2) {
        vals.push(half::f16::from_le_bytes([c[0], c[1]]).to_f32());
    }
    let (mut sum_err, mut sum_abs) = (0f64, 0f64);
    let mut out = Vec::with_capacity(bytes.len());
    for chunk in vals.chunks(block) {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        // Масштаб хранится в F16 — учитываем и это округление.
        let scale = half::f16::from_f32(if amax > 0.0 { amax / 127.0 } else { 1.0 }).to_f32();
        for &v in chunk {
            let q = (v / scale).round().clamp(-127.0, 127.0) * scale;
            sum_err += (v - q).abs() as f64;
            sum_abs += v.abs() as f64;
            out.extend_from_slice(&half::f16::from_f32(q).to_le_bytes());
        }
    }
    (out, sum_err, sum_abs)
}

/// Раскладка DeltaNet из config.json модели.
#[derive(Debug, Clone, Copy)]
pub struct DeltaLayout {
    pub n_k: usize,
    pub n_v: usize,
    pub hk: usize,
    pub hv: usize,
}

impl DeltaLayout {
    /// Сколько v-голов приходится на одну k-голову.
    pub fn n_per_k(&self) -> usize {
        self.n_v / self.n_k.max(1)
    }
}

/// Прочитать раскладку из config.json рядом с safetensors.
fn read_delta_layout(dir: &Path) -> Option<DeltaLayout> {
    let raw = std::fs::read(dir.join("config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let t = cfg.get("text_config").unwrap_or(&cfg);
    let get = |k: &str| t.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    Some(DeltaLayout {
        n_k: get("linear_num_key_heads")?,
        n_v: get("linear_num_value_heads")?,
        hk: get("linear_key_head_dim")?,
        hv: get("linear_value_head_dim")?,
    })
}

/// HF пакует v/z/a/b по группам k-голов: индекс головы = g*n_per_k + j.
/// GGUF (llama.cpp) ждёт j-major: индекс = j*n_k + g. Переставляем блоки.
pub fn deinterleave_blocks(src: &[u8], n_k: usize, n_per_k: usize, block_bytes: usize) -> Vec<u8> {
    let heads = n_k * n_per_k;
    assert_eq!(src.len(), heads * block_bytes, "deinterleave: размер не бьётся");
    let mut out = Vec::with_capacity(src.len());
    for j in 0..n_per_k {
        for g in 0..n_k {
            let hf = g * n_per_k + j;
            out.extend_from_slice(&src[hf * block_bytes..(hf + 1) * block_bytes]);
        }
    }
    out
}

/// То же по столбцам: out_proj принимает v-пространство, порядок голов в его
/// входной размерности обязан совпадать с порядком v.
pub fn deinterleave_cols(
    src: &[u8],
    rows: usize,
    n_k: usize,
    n_per_k: usize,
    col_block_bytes: usize,
) -> Vec<u8> {
    let heads = n_k * n_per_k;
    let row_bytes = heads * col_block_bytes;
    assert_eq!(src.len(), rows * row_bytes, "deinterleave_cols: размер не бьётся");
    let mut out = Vec::with_capacity(src.len());
    for r in 0..rows {
        let base = r * row_bytes;
        for j in 0..n_per_k {
            for g in 0..n_k {
                let hf = g * n_per_k + j;
                let off = base + hf * col_block_bytes;
                out.extend_from_slice(&src[off..off + col_block_bytes]);
            }
        }
    }
    out
}

/// Привести F16-байты HF-тензора к раскладке GGUF (только DeltaNet-проекции).
fn repack_delta(gguf: &str, data: Vec<u8>, shape: &[usize], lay: Option<DeltaLayout>) -> Vec<u8> {
    let Some(l) = lay else { return data };
    let n_per_k = l.n_per_k();
    if n_per_k <= 1 || shape.len() != 2 {
        return data;
    }
    let (rows, cols) = (shape[0], shape[1]);
    let row_bytes = cols * 2;
    if gguf.ends_with("attn_qkv.weight") {
        // [q(n_k*hk) | k(n_k*hk) | v(n_v*hv)] — переставляем только v-хвост.
        let qk_rows = 2 * l.n_k * l.hk;
        if rows != qk_rows + l.n_v * l.hv {
            return data;
        }
        let split = qk_rows * row_bytes;
        let mut out = data[..split].to_vec();
        out.extend_from_slice(&deinterleave_blocks(
            &data[split..],
            l.n_k,
            n_per_k,
            l.hv * row_bytes,
        ));
        out
    } else if (gguf.ends_with("attn_z.weight") || gguf.ends_with("attn_gate.weight"))
        && rows == l.n_v * l.hv
    {
        deinterleave_blocks(&data, l.n_k, n_per_k, l.hv * row_bytes)
    } else if (gguf.ends_with("attn_a.weight")
        || gguf.ends_with("attn_b.weight")
        || gguf.ends_with("ssm_alpha.weight")
        || gguf.ends_with("ssm_beta.weight"))
        && rows == l.n_v
    {
        deinterleave_blocks(&data, l.n_k, n_per_k, row_bytes)
    } else if (gguf.ends_with("attn_out.weight") || gguf.ends_with("ssm_out.weight"))
        && cols == l.n_v * l.hv
    {
        deinterleave_cols(&data, rows, l.n_k, n_per_k, l.hv * 2)
    } else {
        data
    }
}

/// Прочитать тензор safetensors в f32 (BF16/F16/F32 на входе).
fn read_tensor_f32(
    shards: &[(String, memmap2::Mmap)],
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let found = find_tensor_info(shards, name).ok_or_else(|| format!("нет тензора {name}"))?;
    let raw: &[u8] = &shards[found.shard].1[found.start..found.end];
    let n: usize = found.shape.iter().product();
    let values = match found.dtype {
        safetensors::Dtype::BF16 => {
            if raw.len() != n * 2 {
                return Err(format!("{name}: BF16 длина {} != {}", raw.len(), n * 2));
            }
            raw.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect()
        }
        safetensors::Dtype::F16 => {
            if raw.len() != n * 2 {
                return Err(format!("{name}: F16 длина {} != {}", raw.len(), n * 2));
            }
            raw.chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }
        safetensors::Dtype::F32 => {
            if raw.len() != n * 4 {
                return Err(format!("{name}: F32 длина {} != {}", raw.len(), n * 4));
            }
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        other => return Err(format!("{name}: dtype {other:?} не поддерживается")),
    };
    Ok((values, found.shape))
}

fn run_pack(args: &Args) -> Result<(), String> {
    let shards = open_safetensors(&args.inputs)?;
    let dir = args.inputs[0].parent().unwrap_or(Path::new("."));

    // Полный список имён из всех шардов
    let mut hf_names: Vec<String> = Vec::new();
    for (_, mmap) in &shards {
        let (_off, meta) = safetensors::SafeTensors::read_metadata(mmap)
            .map_err(|e| format!("safetensors meta: {e}"))?;
        for (name, _) in meta.tensors() {
            hf_names.push(name.to_string());
        }
    }
    hf_names.sort();
    hf_names.dedup();

    let ref_gguf = args
        .gguf
        .clone()
        .ok_or_else(|| "--pack требует --gguf <эталон> (типы тензоров и метаданные)".to_string())?;
    let tokenizer = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| dir.join("tokenizer.json"));
    if !tokenizer.exists() {
        return Err(format!("нет tokenizer.json: {}", tokenizer.display()));
    }
    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| dir.to_path_buf())
        .join(
            ref_gguf
                .file_stem()
                .map(|s| format!("{}.ytf", s.to_string_lossy()))
                .unwrap_or_else(|| "model.ytf".into()),
        );

    let delta_layout = read_delta_layout(dir);
    match delta_layout {
        Some(l) => println!(
            "раскладка DeltaNet: n_k={} n_v={} head_k={} head_v={} (v-голов на k-голову: {})",
            l.n_k, l.n_v, l.hk, l.hv, l.n_per_k()
        ),
        None => println!("раскладка DeltaNet: config.json не найден — перепаковка НЕ выполняется"),
    }

    let t0 = std::time::Instant::now();
    let stats = pack::pack(
        &shards,
        &hf_names,
        &ref_gguf,
        &tokenizer,
        &out_path,
        delta_layout,
        args.fuse_in_proj,
        &read_tensor_f32,
        &repack_delta,
    )?;
    println!(
        "упаковано {} тензоров, {:.2} ГиБ, за {:.1} с → {}",
        stats.tensors,
        stats.bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        t0.elapsed().as_secs_f64(),
        out_path.display()
    );
    for (dt, n) in &stats.by_dtype {
        println!("  {dt}: {n}");
    }
    // Пропущенное — это видео-башня и MTP; их в v2 пока нет. Печатаем счётчик,
    // чтобы молчаливая потеря языкового тензора была видна.
    let lm_skipped: Vec<&String> = stats
        .skipped
        .iter()
        .filter(|n| n.starts_with("model.language_model."))
        .collect();
    println!(
        "пропущено {} тензоров (видео/MTP), из них языковых: {}",
        stats.skipped.len(),
        lm_skipped.len()
    );
    if !lm_skipped.is_empty() {
        for n in lm_skipped.iter().take(10) {
            println!("  ВНИМАНИЕ пропущен языковой тензор: {n}");
        }
        return Err("в контейнер не попали языковые тензоры".into());
    }
    Ok(())
}

fn run(args: &Args) -> Result<(), String> {
    if args.pack {
        return run_pack(args);
    }
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
    let mut layers_ffn: Vec<u32> = Vec::new();
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
                mask::LayerKind::Ffn => {
                    if !layers_ffn.contains(&idx) {
                        layers_ffn.push(idx);
                    }
                }
            }
        }
    }
    layers_delta.sort();
    layers_attn.sort();
    layers_ffn.sort();
    if !args.f16_ffn {
        layers_ffn.clear();
    }

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

    for &i in &layers_ffn {
        for suffix in [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            let st_name = format!("model.language_model.layers.{i}.{suffix}");
            if let Some(gguf_t) = mask::resolve(mask::LayerKind::Ffn, suffix) {
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
        "heavy plan: {} delta layers ×5 + {} attn layers ×4 + {} ffn layers ×3 = {} tensors (missing {})",
        layers_delta.len(),
        layers_attn.len(),
        layers_ffn.len(),
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
        mask: if args.f16_ffn { "heavy+ffn".into() } else { "heavy".into() },
    };
    let mut w = container::ContainerWriter::create(out_file, pre).map_err(|e| format!("container create: {e}"))?;

    // Раскладка DeltaNet: без неё v/z/a/b/out уедут в HF-порядке голов.
    let delta_layout = args.inputs[0]
        .parent()
        .and_then(read_delta_layout);
    match delta_layout {
        Some(l) => println!(
            "delta layout: n_k={} n_v={} head_k={} head_v={} (v-голов на k-голову: {})",
            l.n_k, l.n_v, l.hk, l.hv, l.n_per_k()
        ),
        None => println!("delta layout: config.json не найден — перепаковка v/z/a/b/out НЕ выполняется"),
    }

    // Стриминг: для каждого планового тензора — чтение шарда, cast→F16
    let t0 = std::time::Instant::now();
    let mut total_bytes = 0u64;
    let (mut fp8_err, mut fp8_abs) = (0f64, 0f64);
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
        let f16_bytes = repack_delta(&p.gguf, f16_bytes, &shape, delta_layout);
        let f16_bytes = if args.fp8_emulate {
            let (q, err, abs) = fp8_emulate_f16(&f16_bytes, args.fp8_block);
            fp8_err += err;
            fp8_abs += abs;
            q
        } else if args.q8_emulate {
            let (q, err, abs) = q8_0_emulate_f16(&f16_bytes, 32);
            fp8_err += err;
            fp8_abs += abs;
            q
        } else {
            f16_bytes
        };
        total_bytes += f16_bytes.len() as u64;
        w.add_tensor(&p.gguf, &shape, f16_bytes);
    }

    if args.q8_emulate {
        println!(
            "q8_0 emulate (int8, блок 32): относительная ошибка весов {:.3}%",
            100.0 * fp8_err / fp8_abs.max(1e-9)
        );
    }
    if args.fp8_emulate {
        println!(
            "fp8 emulate (E4M3, блок {}): относительная ошибка весов {:.3}%",
            args.fp8_block,
            100.0 * fp8_err / fp8_abs.max(1e-9)
        );
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

    /// HF-порядок голов (g*n_per_k + j) → GGUF-порядок (j*n_k + g).
    #[test]
    fn deinterleave_maps_hf_head_order_to_gguf() {
        // 2 k-головы × 2 под-головы, блок = 1 байт: HF [g0j0, g0j1, g1j0, g1j1]
        let src = [10u8, 11, 20, 21];
        let out = deinterleave_blocks(&src, 2, 2, 1);
        // GGUF ждёт [j0: g0, g1 | j1: g0, g1]
        assert_eq!(out, vec![10, 20, 11, 21]);
    }

    #[test]
    fn deinterleave_cols_permutes_within_each_row() {
        // 2 строки × 4 головы по 1 байту
        let src = [10u8, 11, 20, 21, 30, 31, 40, 41];
        let out = deinterleave_cols(&src, 2, 2, 2, 1);
        assert_eq!(out, vec![10, 20, 11, 21, 30, 40, 31, 41]);
    }

    /// E4M3 держит 3 бита мантиссы: относительная ошибка ≤ ~6%, знак и
    /// порядок сохраняются, ноль остаётся нулём.
    #[test]
    fn e4m3_round_keeps_scale_and_sign() {
        for x in [0.5f32, -0.5, 1.0, 3.14159, -0.0625, 100.0, 447.0] {
            let q = e4m3_round(x);
            assert_eq!(q.signum(), x.signum(), "знак {x}");
            let rel = (q - x).abs() / x.abs();
            assert!(rel <= 0.07, "x={x} q={q} rel={rel}");
        }
        assert_eq!(e4m3_round(0.0), 0.0);
        assert_eq!(e4m3_round(1e6), 448.0, "клампится в максимум E4M3");
        // Ниже минимальной субнормали (2^-9) E4M3 обнуляет — ровно поэтому
        // веса масштабируются поблочно, а не квантуются «как есть».
        assert_eq!(e4m3_round(0.001234), 0.0);
    }

    /// Q8_0 с блоком 32 точнее E4M3 с блоком 128 на типичном распределении
    /// весов: линейная сетка на узком блоке бьёт экспоненциальную на широком.
    #[test]
    fn q8_beats_fp8_on_weight_like_data() {
        let mut vals = Vec::new();
        let mut x = 0.123f32;
        for _ in 0..4096 {
            x = (x * 7.13 + 0.37).fract() - 0.5; // детерминированный «шум» ±0.5
            vals.push(half::f16::from_f32(x * 0.08));
        }
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (_, q8_err, abs) = q8_0_emulate_f16(&bytes, 32);
        let (_, fp8_err, _) = fp8_emulate_f16(&bytes, 128);
        assert!(
            q8_err < fp8_err,
            "q8={} fp8={} (abs={abs})",
            q8_err / abs,
            fp8_err / abs
        );
    }

    /// Поблочный масштаб спасает мелкие значения рядом с крупным выбросом.
    #[test]
    fn fp8_block_scaling_survives_outlier() {
        let mut vals = vec![half::f16::from_f32(0.01); 127];
        vals.push(half::f16::from_f32(400.0));
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (out, err, abs) = fp8_emulate_f16(&bytes, 128);
        assert_eq!(out.len(), bytes.len());
        assert!(err / abs < 0.5, "ошибка блока {}", err / abs);
        let first = half::f16::from_le_bytes([out[0], out[1]]).to_f32();
        assert!(first > 0.0, "мелкое значение не должно обнуляться: {first}");
    }

    /// Перепаковка qkv трогает только v-хвост: q и k остаются на месте.
    #[test]
    fn repack_qkv_keeps_q_and_k() {
        let lay = DeltaLayout { n_k: 2, n_v: 4, hk: 1, hv: 1 };
        // cols=1 (row_bytes=2): q(2 строки) k(2) v(4)
        let data: Vec<u8> = (0u8..16).collect();
        let out = repack_delta("blk.0.attn_qkv.weight", data.clone(), &[8, 1], Some(lay));
        assert_eq!(&out[..8], &data[..8], "q/k не должны двигаться");
        // v-строки HF [v0,v1,v2,v3] → GGUF [v0,v2,v1,v3]
        assert_eq!(&out[8..], &[8u8, 9, 12, 13, 10, 11, 14, 15]);
    }
}
