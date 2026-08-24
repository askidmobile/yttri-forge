use anyhow::{Context, Result};
use candle_core::Device;
use qwen35_batch::model::{DecodeBatch, DecodeItem, MultimodalPrefill, PrefillChunk};
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::BatchModel;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

fn argmax(logits: &[f32]) -> (u32, f32) {
    let mut top = [(0u32, f32::NEG_INFINITY); 2];
    for (index, &value) in logits.iter().enumerate() {
        if value > top[0].1 {
            top[1] = top[0];
            top[0] = (index as u32, value);
        } else if value > top[1].1 {
            top[1] = (index as u32, value);
        }
    }
    (top[0].0, top[0].1 - top[1].1)
}

fn usize_value(value: &Value, name: &str) -> Result<usize> {
    usize::try_from(value.as_u64().with_context(|| format!("{name} must be u64"))?)
        .with_context(|| format!("{name} does not fit usize"))
}

fn usize_vec(value: &Value, name: &str) -> Result<Vec<usize>> {
    value
        .as_array()
        .with_context(|| format!("{name} must be an array"))?
        .iter()
        .map(|value| usize_value(value, name))
        .collect()
}

fn u32_vec(value: &Value, name: &str) -> Result<Vec<u32>> {
    value
        .as_array()
        .with_context(|| format!("{name} must be an array"))?
        .iter()
        .map(|value| {
            u32::try_from(value.as_u64().with_context(|| format!("{name} must contain u64"))?)
                .with_context(|| format!("{name} value does not fit u32"))
        })
        .collect()
}

fn f32_vec(value: &Value, name: &str) -> Result<Vec<f32>> {
    value
        .as_array()
        .with_context(|| format!("{name} must be an array"))?
        .iter()
        .map(|value| {
            let value = value
                .as_f64()
                .with_context(|| format!("{name} must contain numbers"))?;
            if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
                anyhow::bail!("{name} contains invalid f32");
            }
            Ok(value as f32)
        })
        .collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let text = PathBuf::from(args.next().context(
        "usage: qwen35_multimodal_logits TEXT.gguf VISION.gguf PROCESSOR.json [steps]",
    )?);
    let vision = PathBuf::from(args.next().context("missing Vision GGUF")?);
    let processor = PathBuf::from(args.next().context("missing processor probe JSON")?);
    let steps = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()
        .context("steps must be an integer")?
        .unwrap_or(4);
    if args.next().is_some() || steps == 0 {
        anyhow::bail!(
            "usage: qwen35_multimodal_logits TEXT.gguf VISION.gguf PROCESSOR.json [steps]"
        );
    }

    let document: Value = serde_json::from_slice(&std::fs::read(&processor)?)?;
    let media = &document["media"];
    let prompt = &document["prompt"];
    let patch_shape = usize_vec(&media["patch_shape"], "media.patch_shape")?;
    if patch_shape.len() != 2 {
        anyhow::bail!("media.patch_shape must contain two values");
    }
    let grid = usize_vec(&media["grid_thw"], "media.grid_thw")?;
    if grid.len() != 3 {
        anyhow::bail!("media.grid_thw must contain three values");
    }
    let token_ids = u32_vec(&prompt["input_ids"], "prompt.input_ids")?;
    let mm_token_types = prompt["mm_token_type_ids"]
        .as_array()
        .context("prompt.mm_token_type_ids must be an array")?
        .iter()
        .map(|value| {
            u8::try_from(value.as_u64().context("mm token type must be u64")?)
                .context("mm token type does not fit u8")
        })
        .collect::<Result<Vec<_>>>()?;
    let rope = prompt["rope_positions"]
        .as_array()
        .context("prompt.rope_positions must be an array")?;
    if rope.len() != 3 {
        anyhow::bail!("prompt.rope_positions must contain three axes");
    }
    let rope_positions = [
        u32_vec(&rope[0], "rope axis 0")?,
        u32_vec(&rope[1], "rope axis 1")?,
        u32_vec(&rope[2], "rope axis 2")?,
    ];
    let decode_rope_delta = prompt["decode_rope_delta"]
        .as_i64()
        .context("prompt.decode_rope_delta must be i64")?;
    let patch_values = f32_vec(&media["patch_values"], "media.patch_values")?;

    let device = Device::new_cuda(0).context("open CUDA device 0")?;
    let mut adapter = Qwen35BatchAdapter::load(&text, device, 1)?;
    if std::env::var("QWEN35_VISION_REFERENCE").as_deref() == Ok("1") {
        adapter.load_vision_reference(&vision)?;
    } else {
        adapter.load_vision(&vision)?;
    }
    adapter.install_multimodal(
        0,
        MultimodalPrefill {
            token_ids: token_ids.clone(),
            media_grids: vec![[grid[0], grid[1], grid[2]]],
            patch_values,
            patch_rows: patch_shape[0],
            patch_width: patch_shape[1],
            mm_token_types,
            rope_positions,
            decode_rope_delta,
        },
    )?;

    let started = Instant::now();
    let mut logits = Vec::new();
    let chunk_count = token_ids.len().div_ceil(512);
    for (chunk_index, chunk) in token_ids.chunks(512).enumerate() {
        logits = adapter.prefill_chunk(&PrefillChunk {
            slot_idx: 0,
            reset_first: chunk_index == 0,
            tokens: chunk.to_vec(),
            start_pos: chunk_index * 512,
            is_final: chunk_index + 1 == chunk_count,
        })?;
    }
    let prefill = started.elapsed();
    let mut tokens = Vec::with_capacity(steps);
    for step in 0..steps {
        let (token, margin) = argmax(&logits);
        println!("{}", json!({"type":"logits","step":step,"argmax":token,"margin":margin,"values":logits}));
        tokens.push(token);
        if step + 1 == steps {
            break;
        }
        logits = adapter
            .decode_batch(&DecodeBatch {
                items: vec![DecodeItem {
                    slot_idx: 0,
                    token,
                    pos: token_ids.len() + step,
                }],
            })?
            .pop()
            .context("decode returned no logits")?;
    }
    println!(
        "{}",
        json!({
            "type":"tokens",
            "ids":tokens,
            "prompt_tokens":token_ids.len(),
            "decode_rope_delta":decode_rope_delta,
            "prefill_ms":prefill.as_secs_f64()*1000.0
        })
    );
    adapter.reset_slot(0)?;
    adapter.unload_vision();
    Ok(())
}
