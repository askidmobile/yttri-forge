//! perplexity — качество кванта одним числом.
//!
//! Перплексия = exp(средний -log p(следующий токен)). Меньше — лучше. Нужна,
//! чтобы выбирать рецепты квантов не на глаз: разница между Q4_K и Q5_K по
//! скорости видна сразу, а по качеству — только так.
//!
//! Схема: префилл первого токена, дальше декод по одному, накапливая
//! -log p(t_i | t_0..t_{i-1}). Это O(n) шагов декода вместо O(n²) префиллов.
//! Логиты префилла и декода отдают только последнюю позицию, поэтому оконного
//! варианта (как у llama.cpp) здесь нет — зато процедура одинакова для всех
//! сравниваемых конфигураций, а именно это и требуется.
//!
//! Запуск: perplexity <model.gguf|model.ytf> <text.txt> [n_tokens]

use anyhow::{bail, Context, Result};
use candle_core::Device;
use qwen35_batch::model::{DecodeBatch, DecodeItem, PrefillChunk};
use qwen35_batch::real::{tokenizer, Qwen35BatchAdapter};
use qwen35_batch::BatchModel;
use std::path::PathBuf;
use std::time::Instant;

/// -log softmax(logits)[target] с вычитанием максимума: без него exp даёт
/// переполнение на реальных логитах (они доходят до ~30).
fn neg_log_prob(logits: &[f32], target: u32) -> f32 {
    let idx = target as usize;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &v in logits {
        sum += ((v - max) as f64).exp();
    }
    (sum.ln() as f32) - (logits[idx] - max)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(args.next().context("нужен путь к модели")?);
    let text_path = PathBuf::from(args.next().context("нужен путь к тексту")?);
    let limit: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    let text = std::fs::read_to_string(&text_path)
        .with_context(|| format!("чтение {}", text_path.display()))?;
    let tok = tokenizer::load_from_gguf_path(&model)?;
    let encoded = tok
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("токенизация: {e}"))?;
    let mut tokens: Vec<u32> = encoded.get_ids().to_vec();
    if tokens.len() < 2 {
        bail!("в тексте меньше двух токенов");
    }
    tokens.truncate(limit);

    let device = Device::new_cuda(0).or_else(|_| Ok::<_, anyhow::Error>(Device::Cpu))?;
    let t0 = Instant::now();
    let mut adapter = Qwen35BatchAdapter::load(&model, device, 1)?;
    let load = t0.elapsed();

    // Префилл первого токена: его логиты предсказывают второй.
    let mut logits = adapter.prefill_chunk(&PrefillChunk {
        slot_idx: 0,
        reset_first: true,
        tokens: vec![tokens[0]],
        start_pos: 0,
        is_final: true,
    })?;

    let t0 = Instant::now();
    let mut nll_sum = 0.0f64;
    let mut counted = 0usize;
    for i in 1..tokens.len() {
        let nll = neg_log_prob(&logits, tokens[i]);
        if !nll.is_finite() {
            bail!("нефинитный -log p на позиции {i}: модель сломана");
        }
        nll_sum += nll as f64;
        counted += 1;
        if i + 1 == tokens.len() {
            break;
        }
        logits = adapter
            .decode_batch(&DecodeBatch {
                items: vec![DecodeItem {
                    slot_idx: 0,
                    token: tokens[i],
                    pos: i,
                }],
            })?
            .pop()
            .context("decode не вернул логиты")?;
    }
    let elapsed = t0.elapsed();

    let mean_nll = nll_sum / counted as f64;
    println!(
        "model={}\ntokens={counted} ppl={:.4} mean_nll={:.4}\nload={:.1}s eval={:.1}s ({:.1} tok/s)",
        model.display(),
        mean_nll.exp(),
        mean_nll,
        load.as_secs_f64(),
        elapsed.as_secs_f64(),
        counted as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}
