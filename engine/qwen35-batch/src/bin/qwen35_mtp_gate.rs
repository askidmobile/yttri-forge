use anyhow::{Context, Result};
use candle_core::Device;
use qwen35_batch::model::BatchModel;
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::BatchScheduler;
use qwen35_batch::slot::SlotStatus;
use serde_json::json;
use std::path::Path;

fn run_probe(text: &str, mtp: Option<&str>, prompt: Vec<u32>, max_new: usize, slots: usize) -> Result<Vec<Vec<u32>>> {
    let device = Device::new_cuda(0)?;
    let mut adapter = Qwen35BatchAdapter::load(Path::new(text), device, slots)?;
    if let Some(path) = mtp {
        adapter.load_mtp(Path::new(path))?;
    }
    let eos = adapter.eos();
    let vocab = adapter.vocab_size();
    let mut scheduler = BatchScheduler::new(adapter, slots, eos, vocab);
    for _ in 0..slots {
        scheduler.submit(prompt.clone(), max_new);
    }
    let mut outputs = vec![Vec::new(); slots];
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
    Ok(outputs)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let text = args.next().context("missing TEXT.gguf")?;
    let mtp = args.next().context("missing MTP.gguf")?;
    // Опциональный список слотов: на 27B/12GB четыре слота не помещаются.
    let slots_list: Vec<usize> = args
        .next()
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 4]);
    let prompts: Vec<(&str, Vec<u32>)> = vec![
        ("en", vec![9707, 11, 18727, 70345]),
        ("ru", vec![18727, 70345, 169753, 32868]),
    ];
    let mut results = Vec::new();
    for (label, prompt) in prompts {
        for &slots in &slots_list {
            let baseline = run_probe(&text, None, prompt.clone(), 16, slots)?;
            let mtp_out = run_probe(&text, Some(&mtp), prompt.clone(), 16, slots)?;
            let bit_exact = baseline == mtp_out;
            println!("[gate] lang={} slots={} bit_exact={}", label, slots, bit_exact);
            if !bit_exact {
                println!("  baseline: {:?}", baseline);
                println!("  mtp:      {:?}", mtp_out);
                anyhow::bail!("bit-exact mismatch for lang={} slots={}", label, slots);
            }
            results.push(json!({
                "lang": label,
                "slots": slots,
                "bit_exact": bit_exact,
                "tokens": baseline[0],
            }));
        }
    }
    println!("{}", json!({
        "schema_version": "qwen35-mtp-phase7-gate-v1",
        "status": "pass",
        "results": results
    }));
    Ok(())
}
