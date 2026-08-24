use anyhow::Result;
use candle_core::Device;
use qwen35_batch::model::BatchModel;
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::BatchScheduler;
use qwen35_batch::slot::SlotStatus;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    // Путь к GGUF — argv[1], иначе дефолт yttri-win.
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "D:\\Models\\yttri\\qwen3.5-4b\\Qwen3.5-4B-Q4_K_M.gguf".to_string());
    let text = text.as_str();
    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    let device = Device::new_metal(0)?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let device = Device::Cpu;

    // Для моделей >8 GB (27B) на 12GB GPU используем 1 слот, чтобы уложиться в 11.26 GB физической VRAM
    let is_large_model = text.contains("27B") || text.contains("35B");
    let num_slots = if is_large_model { 1 } else { 4 };
    let mut adapter = Qwen35BatchAdapter::load(Path::new(text), device.clone(), num_slots)?;

    #[cfg(feature = "cuda")]
    if let Device::Cuda(c) = &device {
        let (used, reserved) = candle_core::cuda_backend::mem_pool::default_mempool_usage(c)
            .unwrap_or((0, 0));
        eprintln!("[mem] after load: pool used={}MiB reserved={}MiB", used/1024/1024, reserved/1024/1024);
    }


    // Warmup
    {
        let eos = adapter.eos();
        let vocab = adapter.vocab_size();
        let mut scheduler = BatchScheduler::new(&mut adapter, 1, eos, vocab);
        scheduler.submit(vec![9707; 16], 8);
        while scheduler.step()? != qwen35_batch::scheduler::StepOutcome::Idle {}
        scheduler.model_mut().reset_slot(0)?;
    }

    // Prefill 512 (B=1)
    for p_len in [512] {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }
        let prompt = vec![9707u32; p_len];
        let chunk_size = 512usize;
        let t0 = Instant::now();
        let mut pos = 0usize;
        while pos < p_len {
            let next_pos = (pos + chunk_size).min(p_len);
            let chunk_tokens = prompt[pos..next_pos].to_vec();
            let is_final = next_pos == p_len;
            let _ = adapter.prefill_chunk(&qwen35_batch::model::PrefillChunk {
                slot_idx: 0,
                reset_first: pos == 0,
                tokens: chunk_tokens,
                start_pos: pos,
                is_final,
            })?;
            pos = next_pos;
        }
        device.synchronize()?;
        let el = t0.elapsed().as_secs_f64();
        let tps = p_len as f64 / el;
        println!("CANDLE pp{} B=1: {:.2} tok/s ({:.3}s)", p_len, tps, el);
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let (used, reserved) = candle_core::cuda_backend::mem_pool::default_mempool_usage(c)
                .unwrap_or((0, 0));
            eprintln!("[mem] after prefill: pool used={}MiB reserved={}MiB", used/1024/1024, reserved/1024/1024);
        }
        adapter.reset_slot(0)?;
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }
    }

    // Decode B=1 (tg128)
    {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }
        let eos = adapter.eos();
        let vocab = adapter.vocab_size();
        let mut scheduler = BatchScheduler::new(&mut adapter, 1, eos, vocab);
        scheduler.submit(vec![9707; 32], 128);
        scheduler.step()?; // prefill
        let t0 = Instant::now();
        let mut gen = 0;
        loop {
            match scheduler.step()? {
                qwen35_batch::scheduler::StepOutcome::DidDecode(b) => gen += b,
                qwen35_batch::scheduler::StepOutcome::Idle => break,
                _ => {}
            }
            if scheduler.slots_mut()[0].status == SlotStatus::Finished {
                break;
            }
        }
        device.synchronize()?;
        let el = t0.elapsed().as_secs_f64();
        let tps = gen as f64 / el;
        println!("CANDLE tg128 B=1: {:.2} tok/s ({:.3}s, {} tok)", tps, el, gen);
        scheduler.model_mut().reset_slot(0)?;
    }

    // Decode B=4 (tg128 x 4) — только для моделей <= 9B (на 27B 4 слота превышают 12GB VRAM)
    if !is_large_model {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }
        let eos = adapter.eos();
        let vocab = adapter.vocab_size();
        let mut scheduler = BatchScheduler::new(&mut adapter, 4, eos, vocab);
        for _ in 0..4 {
            scheduler.submit(vec![9707; 32], 128);
        }
        // prefill all 4 slots
        for _ in 0..4 {
            scheduler.step()?;
        }
        let t0 = Instant::now();
        let mut gen = 0;
        loop {
            match scheduler.step()? {
                qwen35_batch::scheduler::StepOutcome::DidDecode(b) => gen += b,
                qwen35_batch::scheduler::StepOutcome::Idle => break,
                _ => {}
            }
            if scheduler.slots_mut().iter().all(|s| s.status == SlotStatus::Finished) {
                break;
            }
        }
        device.synchronize()?;
        let el = t0.elapsed().as_secs_f64();
        let tps = gen as f64 / el;
        println!("CANDLE tg128 B=4: {:.2} aggregate tok/s ({:.2} tok/s per-slot, {:.3}s, {} total tok)", tps, tps / 4.0, el, gen);
    }

    Ok(())
}
