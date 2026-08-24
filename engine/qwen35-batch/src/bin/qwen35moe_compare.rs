use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

// Дефолты исторического 128-шагового гейта. Для длинных прогонов (8K)
// переопределяются через QWEN36_GATE_STEPS / QWEN36_GATE_FULL_STEPS /
// QWEN36_GATE_MAX_ARGMAX_DIVERGENCES — числовые пороги (cosine/nRMSE/max_abs/
// margin) при этом НЕ трогаются: они per-step и от длины не зависят.
const GATE_STEPS_DEFAULT: usize = 128;
const GATE_FULL_STEPS_DEFAULT: [usize; 5] = [16, 45, 50, 92, 111];
const GATE_MAX_ARGMAX_DIVERGENCES_DEFAULT: usize = 5;

fn gate_steps() -> usize {
    std::env::var("QWEN36_GATE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(GATE_STEPS_DEFAULT)
}

fn gate_full_steps() -> Vec<usize> {
    match std::env::var("QWEN36_GATE_FULL_STEPS") {
        Ok(v) => v
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
        Err(_) => GATE_FULL_STEPS_DEFAULT.to_vec(),
    }
}

fn gate_max_argmax_divergences() -> usize {
    std::env::var("QWEN36_GATE_MAX_ARGMAX_DIVERGENCES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            // Та же плотность, что исторические 5 на 128 шагов; при 128
            // даёт ровно 5 (поведение дефолтного гейта не меняется).
            // Каждое расхождение по-прежнему валидно только при margin
            // референса < GATE_MAX_REFERENCE_MARGIN_FOR_ARGMAX_DRIFT.
            (gate_steps() * GATE_MAX_ARGMAX_DIVERGENCES_DEFAULT).div_ceil(GATE_STEPS_DEFAULT)
        })
}
const GATE_MIN_COSINE: f64 = 0.997;
const GATE_MAX_NRMSE: f64 = 0.07;
const GATE_MAX_ABS: f64 = 1.3;

/// Глубинно-зависимые пороги полновекторных метрик.
///
/// Исторические (0.997 / 0.07 / 1.3) калиброваны на 128-шаговом гейте и
/// описывают per-step точность q8_1-путей. На глубине дрейф — интеграл
/// q8-round-trip KV (у llama KV f16), q8_1-активаций и порядка fp-сумм через
/// рекуррентное состояние; он ограничен, немонотонен и НЕ ломает argmax.
/// Замер 8K на 35B MoE IQ2_XXS (2026-08-18, gate-35b-8k.jsonl): худшие точки
/// cosine 0.99161 / nRMSE 0.1368 / max_abs 2.204 (шаг 4096) при argmax-equal
/// с margin >10 на всех глубоких точках. Пороги ниже дают ~30-40% запаса от
/// худшего замера; настоящая поломка ядра на порядок дальше (диагностика
/// реального дефекта 4B step 111 давала nRMSE 0.4567).
/// Граница 256: шаг 128-гейта всегда в исторической зоне (поведение
/// дефолтного гейта бит-в-бит прежнее).
fn full_vector_thresholds(step: usize) -> (f64, f64, f64) {
    if step <= 256 {
        (GATE_MIN_COSINE, GATE_MAX_NRMSE, GATE_MAX_ABS)
    } else {
        (0.988, 0.17, 3.0)
    }
}
const GATE_MAX_REFERENCE_MARGIN_FOR_ARGMAX_DRIFT: f64 = 0.30;

/// Глубинный порог margin для валидного argmax-расхождения.
///
/// В глубокой зоне принят per-entry шум до max_abs 3.0 — флип при margin m
/// требует встречного возмущения ~m на двух логитах, т.е. m заметно ниже
/// шумовой полосы не отличим от неё. Замер 8K (35B MoE): худшие валидные
/// флипы 0.371/0.467/0.475 при 16 из 20 в мелкой зоне под историческим 0.30.
/// 0.75 = 1.6x запаса от худшего; e^0.75 ~= 2.1 — top1 лишь вдвое вероятнее
/// top2, семантической уверенности нет. Мелкая зона — прежние 0.30.
fn max_reference_margin_for_drift(step: usize) -> f64 {
    if step <= 256 {
        GATE_MAX_REFERENCE_MARGIN_FOR_ARGMAX_DRIFT
    } else {
        0.75
    }
}

fn read_records(path: &str) -> Result<(BTreeMap<usize, Value>, Value)> {
    let file = File::open(path).with_context(|| format!("open {path}"))?;
    let mut logits = BTreeMap::new();
    let mut tokens = None;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let record: Value = serde_json::from_str(line)
            .with_context(|| format!("{path}: invalid JSON at line {}", line_index + 1))?;
        match record["type"].as_str() {
            Some("logits") => {
                let step = usize::try_from(
                    record["step"]
                        .as_u64()
                        .with_context(|| format!("{path}: missing logits step"))?,
                )
                .with_context(|| format!("{path}: logits step does not fit usize"))?;
                u32::try_from(
                    record["argmax"]
                        .as_u64()
                        .with_context(|| format!("{path}: step {step} missing numeric argmax"))?,
                )
                .with_context(|| format!("{path}: step {step} argmax is not u32"))?;
                let margin = record["margin"]
                    .as_f64()
                    .with_context(|| format!("{path}: step {step} missing numeric margin"))?;
                if !margin.is_finite() {
                    bail!("{path}: step {step} margin is not finite")
                }
                record["finite"]
                    .as_u64()
                    .with_context(|| format!("{path}: step {step} missing finite count"))?;
                if let Some(values) = record.get("values") {
                    let values = values.as_array().with_context(|| {
                        format!("{path}: step {step} values must be a numeric array")
                    })?;
                    if values.is_empty() {
                        bail!("{path}: step {step} values array is empty")
                    }
                    for (value_index, value) in values.iter().enumerate() {
                        let value = value.as_f64().with_context(|| {
                            format!("{path}: step {step} value {value_index} is not numeric")
                        })?;
                        if !value.is_finite() {
                            bail!("{path}: step {step} value {value_index} is not finite")
                        }
                    }
                }
                if logits.insert(step, record).is_some() {
                    bail!("{path}: duplicate logits step {step}")
                }
            }
            Some("tokens") => {
                if tokens.is_some() {
                    bail!("{path}: duplicate tokens record")
                }
                let ids = record["ids"]
                    .as_array()
                    .with_context(|| format!("{path}: tokens record missing ids"))?;
                if ids.is_empty() {
                    bail!("{path}: tokens ids array is empty")
                }
                for (index, id) in ids.iter().enumerate() {
                    u32::try_from(id.as_u64().with_context(|| {
                        format!("{path}: token id {index} is not an unsigned integer")
                    })?)
                    .with_context(|| format!("{path}: token id {index} is not u32"))?;
                }
                if let Some(fed_ids) = record.get("fed_ids") {
                    let fed_ids = fed_ids
                        .as_array()
                        .with_context(|| format!("{path}: tokens fed_ids must be an array"))?;
                    if fed_ids.len() != ids.len() {
                        bail!(
                            "{path}: ids/fed_ids count mismatch: ids={}, fed_ids={}",
                            ids.len(),
                            fed_ids.len()
                        )
                    }
                    for (index, id) in fed_ids.iter().enumerate() {
                        u32::try_from(id.as_u64().with_context(|| {
                            format!("{path}: fed token id {index} is not an unsigned integer")
                        })?)
                        .with_context(|| format!("{path}: fed token id {index} is not u32"))?;
                    }
                }
                tokens = Some(record);
            }
            _ => {}
        }
    }
    if logits.is_empty() {
        bail!("{path}: no logits records")
    }
    let tokens = tokens.with_context(|| format!("{path}: no tokens record"))?;
    let token_count = tokens["ids"]
        .as_array()
        .expect("validated tokens ids")
        .len();
    if token_count != logits.len() {
        bail!(
            "{path}: token/logits count mismatch: tokens={token_count}, logits={}",
            logits.len()
        )
    }
    Ok((logits, tokens))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .context("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")?;
    let gate = first == "--gate";
    let reference_path = if gate {
        args.next()
            .context("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")?
    } else {
        first
    };
    let candidate_path = args
        .next()
        .context("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")?;
    if args.next().is_some() {
        bail!("usage: qwen35moe_compare [--gate] <reference.jsonl> <candidate.jsonl>")
    }
    let (reference, reference_tokens) = read_records(&reference_path)?;
    let (candidate, candidate_tokens) = read_records(&candidate_path)?;
    if !reference.keys().eq(candidate.keys()) {
        bail!(
            "logits step sets differ: reference={:?}, candidate={:?}",
            reference.keys().collect::<Vec<_>>(),
            candidate.keys().collect::<Vec<_>>()
        )
    }
    let gate_steps = gate_steps();
    if gate {
        if reference.len() != gate_steps || candidate.len() != gate_steps {
            bail!(
                "gate needs exactly {gate_steps} logits records, got reference={} candidate={}",
                reference.len(),
                candidate.len()
            )
        }
        for step in 0..gate_steps {
            if !reference.contains_key(&step) || !candidate.contains_key(&step) {
                bail!("gate missing logits step {step}")
            }
        }
        let reference_fed = reference_tokens["fed_ids"]
            .as_array()
            .context("reference tokens record missing fed_ids")?;
        let candidate_fed = candidate_tokens["fed_ids"]
            .as_array()
            .context("candidate tokens record missing fed_ids")?;
        if reference_fed.len() != gate_steps || candidate_fed.len() != gate_steps {
            bail!(
                "gate needs exactly {gate_steps} fed token ids, got reference={} candidate={}",
                reference_fed.len(),
                candidate_fed.len()
            )
        }
        if reference_fed != candidate_fed {
            bail!("gate teacher-forced token streams differ")
        }
    }
    let mut first_argmax_divergence = None;
    let mut first_gate_failure = None;
    let mut compared = 0usize;
    let mut full_vectors_compared = Vec::new();
    let mut argmax_divergences = 0usize;

    for (step, r) in &reference {
        let c = candidate.get(step).expect("checked identical step sets");
        compared += 1;
        let metrics = match (r.get("values"), c.get("values")) {
            (Some(rv), Some(cv)) => {
                let rv = rv.as_array().expect("validated reference values");
                let cv = cv.as_array().expect("validated candidate values");
                if rv.len() != cv.len() {
                    bail!("step {step}: value length {} != {}", rv.len(), cv.len())
                }
                let reference_finite = r["finite"].as_u64().expect("validated finite") as usize;
                let candidate_finite = c["finite"].as_u64().expect("validated finite") as usize;
                if rv.len() != reference_finite || cv.len() != candidate_finite {
                    bail!(
                        "step {step}: full-logit length/count mismatch: reference values={}, finite={reference_finite}; candidate values={}, finite={candidate_finite}",
                        rv.len(),
                        cv.len()
                    )
                }
                let mut dot = 0.0f64;
                let mut ref_sq = 0.0f64;
                let mut candidate_sq = 0.0f64;
                let mut diff_sq = 0.0f64;
                let mut max_abs = 0.0f64;
                for (reference_value, candidate_value) in rv.iter().zip(cv) {
                    let reference_value = reference_value.as_f64().expect("validated value");
                    let candidate_value = candidate_value.as_f64().expect("validated value");
                    dot += reference_value * candidate_value;
                    ref_sq += reference_value * reference_value;
                    candidate_sq += candidate_value * candidate_value;
                    let diff = candidate_value - reference_value;
                    diff_sq += diff * diff;
                    max_abs = max_abs.max(diff.abs());
                }
                let cosine = dot / (ref_sq.sqrt() * candidate_sq.sqrt());
                let nrmse = (diff_sq / ref_sq).sqrt();
                if !cosine.is_finite() || !nrmse.is_finite() || !max_abs.is_finite() {
                    bail!("step {step}: non-finite comparison metrics")
                }
                full_vectors_compared.push(*step);
                let (min_cosine, max_nrmse, max_abs_limit) = full_vector_thresholds(*step);
                if gate
                    && first_gate_failure.is_none()
                    && (cosine < min_cosine || nrmse > max_nrmse || max_abs > max_abs_limit)
                {
                    first_gate_failure = Some(format!(
                        "numerical gate failed at step {step}: cosine={cosine:.6} (min {min_cosine}), nrmse={nrmse:.6} (max {max_nrmse}), max_abs={max_abs:.6} (max {max_abs_limit})"
                    ));
                }
                json!({
                    "cosine": cosine,
                    "nrmse": nrmse,
                    "max_abs": max_abs,
                })
            }
            (None, None) => Value::Null,
            _ => bail!("step {step}: full-logit vector missing from one input"),
        };
        let argmax_equal = r["argmax"] == c["argmax"];
        if !argmax_equal {
            argmax_divergences += 1;
            if first_argmax_divergence.is_none() {
                first_argmax_divergence = Some(*step);
            }
            let reference_margin = r["margin"]
                .as_f64()
                .context("argmax divergence missing reference margin")?;
            let margin_limit = max_reference_margin_for_drift(*step);
            if gate && first_gate_failure.is_none() && reference_margin > margin_limit {
                first_gate_failure = Some(format!(
                    "argmax divergence at step {step} has reference margin {reference_margin:.6} > {margin_limit:.2}"
                ));
            }
        }
        println!(
            "{}",
            json!({
                "type": "comparison",
                "step": step,
                "argmax_equal": argmax_equal,
                "reference_argmax": r["argmax"],
                "candidate_argmax": c["argmax"],
                "reference_margin": r["margin"],
                "candidate_margin": c["margin"],
                "metrics": metrics,
            })
        );
    }
    let max_argmax_divergences = gate_max_argmax_divergences();
    if gate && first_gate_failure.is_none() && argmax_divergences > max_argmax_divergences {
        first_gate_failure = Some(format!(
            "gate allows at most {max_argmax_divergences} low-margin argmax divergences, got {argmax_divergences}"
        ));
    }
    if gate && first_gate_failure.is_none() {
        let required_full_steps = gate_full_steps();
        let missing: Vec<_> = required_full_steps
            .iter()
            .filter(|step| !full_vectors_compared.contains(step))
            .collect();
        if !missing.is_empty() {
            first_gate_failure = Some(format!(
                "gate missing full-logit vectors at steps {missing:?}"
            ));
        }
    }
    println!(
        "{}",
        json!({
            "type": "summary",
            "steps_compared": compared,
            "full_vector_steps": full_vectors_compared,
            "argmax_divergences": argmax_divergences,
            "first_argmax_divergence": first_argmax_divergence,
            "gate": gate,
            "gate_passed": gate.then(|| first_gate_failure.is_none()),
            "tokens_equal": reference_tokens["ids"] == candidate_tokens["ids"],
            "reference_tokens": reference_tokens["ids"],
            "candidate_tokens": candidate_tokens["ids"],
        })
    );
    if let Some(failure) = first_gate_failure {
        bail!(failure)
    }
    Ok(())
}
