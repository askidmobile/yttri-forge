use anyhow::{bail, Context, Result};
use candle_core::Device;
use qwen35_batch::model::{DecodeBatch, DecodeItem, PrefillChunk};
use qwen35_batch::real::{tokenizer, Qwen35BatchAdapter};
use qwen35_batch::BatchModel;
use serde_json::json;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best = index as u32;
            best_value = value;
        }
    }
    best
}

fn record(step: usize, logits: &[f32], include_values: bool) -> serde_json::Value {
    let mut indices: Vec<usize> = (0..logits.len())
        .filter(|&index| logits[index].is_finite())
        .collect();
    let top_len = 10.min(indices.len());
    if top_len > 0 {
        indices.select_nth_unstable_by(top_len - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        indices.truncate(top_len);
    }
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut finite = 0usize;
    let mut checksum = 0xcbf29ce484222325u64;
    for &value in logits {
        checksum ^= value.to_bits() as u64;
        checksum = checksum.wrapping_mul(0x100000001b3);
        if value.is_finite() {
            finite += 1;
            let value = value as f64;
            sum += value;
            sum_sq += value * value;
        }
    }
    let top: Vec<_> = indices
        .iter()
        .map(|&index| json!({"id": index, "logit": logits[index]}))
        .collect();
    let margin = if indices.len() >= 2 {
        logits[indices[0]] - logits[indices[1]]
    } else {
        f32::NAN
    };
    let mut record = json!({
        "type": "logits",
        "step": step,
        "argmax": argmax(logits),
        "margin": margin,
        "finite": finite,
        "sum": sum,
        "sum_sq": sum_sq,
        "checksum": format!("{checksum:016x}"),
        "top": top,
    });
    if include_values {
        record["values"] = json!(logits);
    }
    record
}

fn read_forced_tokens(path: &Path) -> Result<Vec<u32>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open forced-token JSONL: {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.starts_with('{') {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line)?;
        if record["type"] == "tokens" {
            return record["ids"]
                .as_array()
                .context("forced-token record missing ids")?
                .iter()
                .map(|id| {
                    u32::try_from(id.as_u64().context("forced token is not an integer")?)
                        .context("forced token is not u32")
                })
                .collect();
        }
    }
    bail!(
        "forced-token JSONL has no tokens record: {}",
        path.display()
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model = PathBuf::from(
        args.next()
            .context("usage: qwen35moe_logits <model.gguf> [steps]")?,
    );
    let steps = args
        .next()
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()
        .context("steps must be an integer")?
        .unwrap_or(8);
    let fixture = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "capital".into());
    let forced_path = args.next().map(PathBuf::from);
    if args.next().is_some() || steps == 0 || !matches!(fixture.as_str(), "capital" | "rust") {
        bail!("usage: qwen35moe_logits <model.gguf> [steps] [capital|rust] [forced.jsonl]")
    }
    let forced_tokens = forced_path.as_deref().map(read_forced_tokens).transpose()?;
    if let Some(tokens) = &forced_tokens {
        if tokens.len() < steps {
            bail!(
                "forced-token file has {} tokens, need {steps}",
                tokens.len()
            )
        }
    }

    // QWEN36_FORCE_DMMV=1: dequantize+cuBLAS вместо MMQ/MMVQ — референс для
    // A/B новых fused-путей (env-хука в candle-core нет, только setter).
    if std::env::var("QWEN36_FORCE_DMMV").as_deref() == Ok("1") {
        candle_core::quantized::cuda::set_force_dmmv(true);
        eprintln!("[logits] FORCE_DMMV=1: dequantize+cuBLAS path");
    }
    let device = Device::new_cuda(0).context("open CUDA device 0")?;
    let tokenizer = tokenizer::load_from_gguf_path(&model)?;
    let messages = match fixture.as_str() {
        "capital" => vec![
            tokenizer::ChatMsg {
                role: "system",
                content: "You are a helpful assistant. Answer concisely.",
            },
            tokenizer::ChatMsg {
                role: "user",
                content: "What is the capital of France? Answer with one word.",
            },
        ],
        "rust" => vec![tokenizer::ChatMsg {
            role: "user",
            content: "Write a numbered list of twenty concise Rust memory-safety facts. Do not stop before item twenty.",
        }],
        _ => unreachable!(),
    };
    let prompt = tokenizer::build_chatml_text(&messages);
    let mut prompt_tokens = tokenizer::encode_no_think(&tokenizer, &prompt)?;
    if let Ok(value) = std::env::var("QWEN36_LOGITS_PROMPT_TOKENS") {
        let target = value
            .parse::<usize>()
            .context("QWEN36_LOGITS_PROMPT_TOKENS must be a positive integer")?;
        if target == 0 {
            bail!("QWEN36_LOGITS_PROMPT_TOKENS must be a positive integer")
        }
        let fixture = prompt_tokens.clone();
        prompt_tokens = fixture.iter().copied().cycle().take(target).collect();
    }
    let vocab_size = tokenizer.get_vocab_size(true);
    if let Some(token) = forced_tokens
        .as_ref()
        .and_then(|tokens| tokens.iter().find(|&&token| token as usize >= vocab_size))
    {
        bail!("forced token {token} is outside vocab size {vocab_size}")
    }
    let backend = std::env::var("QWEN36_MOE_BACKEND").unwrap_or_else(|_| "auto".into());
    println!(
        "{}",
        json!({
            "type": "run",
            "schema": "qwen35moe-logits-v1",
            "backend": backend,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "fixture": fixture,
            "forced_tokens": forced_path,
            "steps": steps,
        })
    );

    let load_started = Instant::now();
    let mut adapter = Qwen35BatchAdapter::load(&model, device, 1)?;
    let model_load = load_started.elapsed();
    let prefill_started = Instant::now();
    let mut logits = adapter.prefill_chunk(&PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: prompt_tokens.clone(),
        start_pos: 0,
        is_final: true,
    })?;
    let prefill = prefill_started.elapsed();

    let mut full_steps = BTreeSet::new();
    if let Ok(value) = std::env::var("QWEN36_LOGITS_FULL_STEP") {
        full_steps.insert(
            value
                .parse::<usize>()
                .context("QWEN36_LOGITS_FULL_STEP must be an integer")?,
        );
    }
    if let Ok(value) = std::env::var("QWEN36_LOGITS_FULL_STEPS") {
        for step in value.split(',') {
            if step.is_empty() {
                bail!("QWEN36_LOGITS_FULL_STEPS contains an empty step")
            }
            full_steps.insert(
                step.parse::<usize>()
                    .context("QWEN36_LOGITS_FULL_STEPS must be comma-separated integers")?,
            );
        }
    }
    if let Some(step) = full_steps.iter().find(|&&step| step >= steps) {
        bail!("full logits step {step} is outside requested {steps} steps")
    }
    let include_all_values = std::env::var_os("QWEN36_LOGITS_FULL").is_some();
    let mut predicted = Vec::with_capacity(steps);
    let mut fed = Vec::with_capacity(steps);
    let mut decode_time = Duration::ZERO;
    let mut decode_calls = 0usize;
    for step in 0..steps {
        let prediction = argmax(&logits);
        println!(
            "{}",
            record(
                step,
                &logits,
                include_all_values || full_steps.contains(&step)
            )
        );
        let token = forced_tokens
            .as_ref()
            .map(|tokens| tokens[step])
            .unwrap_or(prediction);
        predicted.push(prediction);
        fed.push(token);
        // QWEN36_LOGITS_IGNORE_EOS=1 (только с forced.jsonl): не останавливаться
        // на EOS. Greedy-цепочка референса упирается в EOS за несколько сотен
        // шагов, а длинный teacher-forced гейт (8K) должен прогнать KV/state
        // пути на всю глубину - референс тоже форсит сквозь EOS.
        let ignore_eos = forced_tokens.is_some()
            && std::env::var("QWEN36_LOGITS_IGNORE_EOS").as_deref() == Ok("1");
        if (token == adapter.eos() && !ignore_eos) || step + 1 == steps {
            break;
        }
        let decode_started = Instant::now();
        logits = adapter
            .decode_batch(&DecodeBatch {
                items: vec![DecodeItem {
                    slot_idx: 0,
                    token,
                    pos: prompt_tokens.len() + step,
                }],
            })?
            .pop()
            .context("decode returned no logits")?;
        decode_time += decode_started.elapsed();
        decode_calls += 1;
    }
    println!(
        "{}",
        json!({
            "type": "tokens",
            "ids": predicted,
            "fed_ids": fed,
            "text": tokenizer::decode_text(&tokenizer, &predicted)?,
        })
    );
    println!(
        "{}",
        json!({
            "type": "performance",
            "model_load_ms": model_load.as_secs_f64() * 1000.0,
            "prefill_ms": prefill.as_secs_f64() * 1000.0,
            "prompt_tokens": prompt_tokens.len(),
            "prefill_tokens_per_s": prompt_tokens.len() as f64 / prefill.as_secs_f64(),
            "decode_ms": decode_time.as_secs_f64() * 1000.0,
            "decode_calls": decode_calls,
            "decode_tokens_per_s": if decode_calls == 0 {
                0.0
            } else {
                decode_calls as f64 / decode_time.as_secs_f64()
            },
        })
    );
    Ok(())
}
