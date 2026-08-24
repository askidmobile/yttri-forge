use anyhow::{Context, Result};
use candle_core::Device;
use qwen35_batch::model::BatchModel;
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::BatchScheduler;
use qwen35_batch::slot::SlotStatus;
use serde_json::json;
use std::path::Path;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args.next().context("usage: qwen35_mtp_probe TEXT.gguf MTP.gguf PROMPT_IDS [MAX_NEW] [SLOTS]")?;
    // "-" вместо пути = baseline без MTP (для A/B скорости тем же бинарём).
    let mtp = args.next().context("missing MTP.gguf or -")?;
    let prompt = args.next().context("missing comma-separated prompt IDs")?;
    let max_new = args.next().as_deref().unwrap_or("8").parse::<usize>()?;
    let slots = args.next().as_deref().unwrap_or("1").parse::<usize>()?;
    let prompt = prompt
        .split(',')
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let device = Device::new_cuda(0)?;
    let mut adapter = Qwen35BatchAdapter::load(Path::new(&text), device, slots)?;
    if mtp != "-" {
        adapter.load_mtp(Path::new(&mtp))?;
    }
    let eos = adapter.eos();
    let vocab = adapter.vocab_size();
    let mut scheduler = BatchScheduler::new(adapter, slots, eos, vocab);
    for _ in 0..slots {
        scheduler.submit(prompt.clone(), max_new);
    }
    let mut outputs = vec![Vec::new(); slots];
    // Prefill отдельно от decode-тайминга: первый step делает prefill слотов.
    for _ in 0..slots {
        scheduler.step()?;
    }
    let started = std::time::Instant::now();
    loop {
        let finished = scheduler
            .slots_mut()
            .iter()
            .filter(|slot| slot.status == SlotStatus::Finished)
            .map(|slot| slot.idx)
            .collect::<Vec<_>>();
        for slot in finished {
            outputs[slot] = scheduler.slots_mut()[slot].generated_tokens().to_vec();
            scheduler.slots_mut()[slot].reset();
            scheduler.model_mut().reset_slot(slot)?;
        }
        if scheduler.step()? == qwen35_batch::scheduler::StepOutcome::Idle
            && scheduler.slots_mut().iter().all(|slot| slot.status == SlotStatus::Idle)
        {
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let generated: usize = outputs.iter().map(Vec::len).sum();
    let metrics = (0..slots)
        .map(|slot| scheduler.speculative_metrics(slot).cloned())
        .collect::<Vec<_>>();
    println!("{}", json!({
        "schema_version": "qwen35-mtp-probe-v1",
        "decode_seconds": elapsed,
        "decode_tokens": generated,
        "decode_tok_per_s": generated as f64 / elapsed.max(1e-9),
        "outputs": outputs,
        "metrics": metrics.iter().map(|metric| metric.as_ref().map(|value| json!({
            "enabled": value.enabled,
            "used": value.used,
            "drafted": value.drafted,
            "accepted": value.accepted,
            "fallback": value.fallback.map(|category| category.as_str()),
        }))).collect::<Vec<_>>()
    }));
    Ok(())
}
