//! `qwen36_inspect` — GGUF inspector CLI (Phase 0).
//!
//! Reads a GGUF file via `candle_core::quantized::gguf_file` and emits a JSON
//! manifest containing: GGUF magic/version, metadata key-value pairs, tensor
//! names with rank, logical shape, dtype, byte offset, byte length, and a
//! stable architecture fingerprint. No weight data is copied or hashed — the
//! fingerprint is derived from architecture metadata + tensor manifest + file
//! size, making it reproducible without reading tensor payloads.
//!
//! Usage:
//!   qwen36_inspect <gguf-path> [--output <manifest.json>]
//!   qwen36_inspect --help
//!
//! The manifest is the checked-in fixture for Phase 1 validation tests
//! (`qwen35-batch/tests/fixtures/qwen35moe_profile.json`).

use anyhow::{anyhow, Context, Result};
use candle_core::quantized::gguf_file::{Content, Value, VersionedMagic};
use candle_core::quantized::GgmlDType;
use serde_json::{json, Map, Value as JsonValue};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

const TOKEN_IDS: [(&str, u32); 7] = [
    ("<|endoftext|>", 248044),
    ("<|im_start|>", 248045),
    ("<|im_end|>", 248046),
    ("<|vision_start|>", 248053),
    ("<|vision_end|>", 248054),
    ("<|image_pad|>", 248056),
    ("<|video_pad|>", 248057),
];

fn dtype_name(d: GgmlDType) -> &'static str {
    match d {
        GgmlDType::F32 => "F32",
        GgmlDType::F16 => "F16",
        GgmlDType::BF16 => "BF16",
        GgmlDType::Q4_0 => "Q4_0",
        GgmlDType::Q4_1 => "Q4_1",
        GgmlDType::Q5_0 => "Q5_0",
        GgmlDType::Q5_1 => "Q5_1",
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q8_1 => "Q8_1",
        GgmlDType::Q2K => "Q2_K",
        GgmlDType::Q3K => "Q3_K",
        GgmlDType::Q4K => "Q4_K",
        GgmlDType::Q5K => "Q5_K",
        GgmlDType::Q6K => "Q6_K",
        GgmlDType::Q8K => "Q8_K",
        GgmlDType::IQ2XXS => "IQ2_XXS",
        GgmlDType::IQ2XS => "IQ2_XS",
        GgmlDType::IQ3XXS => "IQ3_XXS",
        GgmlDType::IQ1S => "IQ1_S",
        GgmlDType::IQ4NL => "IQ4_NL",
        GgmlDType::IQ3S => "IQ3_S",
        GgmlDType::IQ2S => "IQ2_S",
        GgmlDType::IQ4XS => "IQ4_XS",
        GgmlDType::IQ1M => "IQ1_M",
    }
}

fn component_kind(metadata: &BTreeMap<String, Value>, tensor_names: &[String]) -> &'static str {
    if tensor_names.iter().any(|name| name.starts_with("v."))
        || metadata.contains_key("clip.projector_type")
    {
        "vision"
    } else if tensor_names.iter().any(|name| name.starts_with("blk.32."))
        || metadata.keys().any(|name| name.contains("nextn"))
    {
        "mtp"
    } else {
        "text"
    }
}

fn token_string(value: &Value) -> Option<&str> {
    match value {
        Value::String(token) => Some(token),
        _ => None,
    }
}

fn tokenizer_ids(metadata: &BTreeMap<String, Value>) -> JsonValue {
    let token_array = metadata.get("tokenizer.ggml.tokens");
    let tokens = match token_array {
        Some(Value::Array(tokens)) => Some(tokens),
        _ => None,
    };
    let mut values = Map::new();
    for (token, expected) in TOKEN_IDS {
        let actual = tokens.and_then(|items| {
            items
                .iter()
                .position(|value| token_string(value) == Some(token))
                .and_then(|index| u32::try_from(index).ok())
        });
        values.insert(
            token.to_string(),
            json!({
                "actual": actual,
                "expected": expected,
                "matches": actual == Some(expected),
            }),
        );
    }
    values.insert(
        "chat_eos_metadata".to_string(),
        json!({
            "actual": metadata.get("tokenizer.ggml.eos_token_id").and_then(|value| value.to_u32().ok()),
            "expected": 248046,
        }),
    );
    values.insert(
        "pad_metadata".to_string(),
        json!({
            "actual": metadata.get("tokenizer.ggml.padding_token_id").and_then(|value| value.to_u32().ok()),
            "expected": 248044,
        }),
    );
    JsonValue::Object(values)
}

fn magic_name(m: &VersionedMagic) -> &'static str {
    match m {
        VersionedMagic::GgufV1 => "GGUFv1",
        VersionedMagic::GgufV2 => "GGUFv2",
        VersionedMagic::GgufV3 => "GGUFv3",
    }
}

/// Convert a GGUF `Value` into a JSON value for the manifest.
fn value_to_json(v: &Value) -> JsonValue {
    match v {
        Value::U8(x) => json!(x),
        Value::I8(x) => json!(x),
        Value::U16(x) => json!(x),
        Value::I16(x) => json!(x),
        Value::U32(x) => json!(x),
        Value::I32(x) => json!(x),
        Value::U64(x) => json!(x),
        Value::I64(x) => json!(x),
        Value::F32(x) => json!(x),
        Value::F64(x) => json!(x),
        Value::Bool(x) => json!(x),
        Value::String(x) => json!(x),
        // Arrays can be large (e.g. tokenizer.ggml.tokens with 150k entries).
        // Record length + element type instead of embedding the full array.
        Value::Array(elems) => {
            let elem_type = elems.first().map(|e| e.value_type());
            json!({
                "type": "array",
                "length": elems.len(),
                "element_type": format!("{elem_type:?}"),
            })
        }
    }
}

/// Build a stable, reproducible fingerprint string from architecture-relevant
/// metadata + tensor manifest + file size. No tensor payload bytes are read.
fn build_fingerprint(
    metadata: &BTreeMap<String, Value>,
    tensor_entries: &[(String, GgmlDType, Vec<usize>, u64, usize)],
    file_size: u64,
) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    // Architecture-identifying metadata keys (sorted for determinism).
    let arch_keys = [
        "general.architecture",
        "general.file_type",
        "general.name",
        "general.quantization_version",
        "tokenizer.ggml.model",
        "qwen35moe.block_count",
        "qwen35moe.context_length",
        "qwen35moe.embedding_length",
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
        "qwen35moe.feed_forward_length",
        "qwen35moe.shared_expert_count",
        "qwen35moe.attention.head_count",
        "qwen35moe.attention.head_count_kv",
        "qwen35moe.attention.key_length",
        "qwen35moe.attention.value_length",
        "qwen35.block_count",
        "qwen35.context_length",
        "qwen35.embedding_length",
        "qwen35.feed_forward_length",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.value_length",
    ];
    for key in arch_keys {
        if let Some(val) = metadata.get(key) {
            key.hash(&mut hasher);
            format!("{val:?}").hash(&mut hasher);
        }
    }

    // Tensor manifest: name, dtype, shape, offset, byte_length.
    for (name, dtype, shape, offset, byte_len) in tensor_entries {
        name.hash(&mut hasher);
        dtype_name(*dtype).hash(&mut hasher);
        shape.hash(&mut hasher);
        offset.hash(&mut hasher);
        byte_len.hash(&mut hasher);
    }

    file_size.hash(&mut hasher);
    format!("qwen36-inspect-fingerprint-{:016x}", hasher.finish())
}

fn parse_args() -> Result<(PathBuf, Option<PathBuf>)> {
    let mut args = std::env::args().skip(1);
    let mut gguf_path: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("qwen36_inspect — GGUF manifest inspector (Phase 0)\n");
                println!("Usage:");
                println!("  qwen36_inspect <gguf-path> [--output <manifest.json>]");
                println!("\nThe manifest is printed to stdout if --output is not given.");
                std::process::exit(0);
            }
            "--output" | "-o" => {
                output =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        anyhow!("--output requires a path argument")
                    })?));
            }
            other if other.starts_with('-') => {
                return Err(anyhow!("unknown flag: {other}"));
            }
            _ => {
                if gguf_path.is_none() {
                    gguf_path = Some(PathBuf::from(arg));
                } else {
                    return Err(anyhow!("unexpected extra argument: {arg}"));
                }
            }
        }
    }

    let gguf_path = gguf_path.ok_or_else(|| {
        anyhow!("missing GGUF path — usage: qwen36_inspect <gguf-path> [--output <manifest.json>]")
    })?;
    Ok((gguf_path, output))
}

fn run() -> Result<()> {
    let (gguf_path, output_path) = parse_args()?;

    let file_size = std::fs::metadata(&gguf_path)
        .with_context(|| format!("stat GGUF: {}", gguf_path.display()))?
        .len();

    let file =
        File::open(&gguf_path).with_context(|| format!("open GGUF: {}", gguf_path.display()))?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .with_context(|| format!("mmap GGUF: {}", gguf_path.display()))?;

    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let ct = Content::read(&mut cursor).context("read GGUF header")?;

    // Sorted metadata for deterministic output.
    let metadata_sorted: BTreeMap<String, Value> = ct.metadata.into_iter().collect();
    let metadata_json: Map<String, JsonValue> = metadata_sorted
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();

    // Sorted tensor entries with byte ranges.
    let mut tensor_entries: Vec<(String, GgmlDType, Vec<usize>, u64, usize)> = Vec::new();
    for (name, info) in ct.tensor_infos.iter() {
        let (start, byte_len) = info
            .byte_range(ct.tensor_data_offset)
            .with_context(|| format!("byte_range for tensor {name}"))?;
        tensor_entries.push((
            name.clone(),
            info.ggml_dtype,
            info.shape.dims().to_vec(),
            start as u64,
            byte_len,
        ));
    }
    // Sort by name for deterministic manifest.
    tensor_entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Detect non-overlapping byte ranges (sanity, no weight data read).
    let mut overlap_warnings: Vec<String> = Vec::new();
    let mut sorted_by_offset = tensor_entries.clone();
    sorted_by_offset.sort_by_key(|e| e.3);
    for window in sorted_by_offset.windows(2) {
        let (name_a, _, _, start_a, len_a) = &window[0];
        let (name_b, _, _, start_b, _) = &window[1];
        let end_a = start_a + *len_a as u64;
        if *start_b < end_a {
            overlap_warnings.push(format!(
                "tensor '{name_a}' [{start_a}..{end_a}) overlaps '{name_b}' starting at {start_b}"
            ));
        }
    }

    // Collect distinct dtypes present in the file.
    let quant_set: Vec<&'static str> = {
        let mut dtypes: Vec<&'static str> = tensor_entries
            .iter()
            .map(|(_, d, _, _, _)| dtype_name(*d))
            .collect();
        dtypes.sort();
        dtypes.dedup();
        dtypes
    };

    let tensor_names: Vec<String> = tensor_entries.iter().map(|entry| entry.0.clone()).collect();
    let component = component_kind(&metadata_sorted, &tensor_names);
    let tokens = tokenizer_ids(&metadata_sorted);
    let fingerprint = build_fingerprint(&metadata_sorted, &tensor_entries, file_size);

    // Tensor count + total tensor bytes.
    let tensor_count = tensor_entries.len();
    let total_tensor_bytes: u64 = tensor_entries.iter().map(|e| e.4 as u64).sum();

    let tensors_json: Vec<JsonValue> = tensor_entries
        .iter()
        .map(|(name, dtype, shape, offset, byte_len)| {
            json!({
                "name": name,
                "dtype": dtype_name(*dtype),
                "rank": shape.len(),
                "shape": shape,
                "offset": offset,
                "byte_length": byte_len,
            })
        })
        .collect();

    let manifest = json!({
        "schema_version": "qwen36-inspect-v2",
        "component": component,
        "gguf_magic": magic_name(&ct.magic),
        "tensor_data_offset": ct.tensor_data_offset,
        "file_size_bytes": file_size,
        "fingerprint": fingerprint,
        "tensor_count": tensor_count,
        "total_tensor_bytes": total_tensor_bytes,
        "quant_set": quant_set,
        "overlap_warnings": overlap_warnings,
        "tokenizer_ids": tokens,
        "metadata": metadata_json,
        "tensors": tensors_json,
    });

    let manifest_str = serde_json::to_string_pretty(&manifest)?;
    match output_path {
        Some(path) => {
            let mut f = File::create(&path)
                .with_context(|| format!("create output: {}", path.display()))?;
            f.write_all(manifest_str.as_bytes())
                .with_context(|| format!("write output: {}", path.display()))?;
            eprintln!(
                "qwen36_inspect: manifest written to {} ({} tensors, fingerprint={})",
                path.display(),
                tensor_count,
                fingerprint
            );
        }
        None => {
            println!("{manifest_str}");
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen36_inspect: error: {e:#}");
        std::process::exit(1);
    }
}
