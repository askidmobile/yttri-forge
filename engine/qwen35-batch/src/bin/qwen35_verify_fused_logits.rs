use anyhow::{Context, Result};
use candle_core::Device;
use qwen35_batch::model::{BatchModel, PrefillChunk};
use qwen35_batch::real::Qwen35BatchAdapter;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("usage: qwen35_verify_fused_logits MODEL OUT [LENS]")?;
    let output = args.next().context("missing output path")?;
    let lengths = args
        .next()
        .unwrap_or_else(|| "64,32000".to_string())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let device = Device::new_cuda(0)?;
    let mut adapter = Qwen35BatchAdapter::load(Path::new(&model), device, 1)?;
    adapter.load_mtp(Path::new(&model))?;
    let mut writer = BufWriter::new(File::create(output)?);

    for prompt_len in lengths {
        let prompt = (0..prompt_len)
            .map(|i| 10 + (i % 1000) as u32)
            .collect::<Vec<_>>();
        for (chunk_index, chunk) in prompt.chunks(1024).enumerate() {
            let start_pos = chunk_index * 1024;
            adapter.prefill_chunk(&PrefillChunk {
                slot_idx: 0,
                reset_first: chunk_index == 0,
                tokens: chunk.to_vec(),
                start_pos,
                is_final: start_pos + chunk.len() == prompt_len,
            })?;
        }
        adapter.speculative_begin(0)?;
        let rows = adapter.speculative_verify(0, &[42, 43], prompt_len)?;
        adapter.speculative_rollback(0)?;
        for row in rows {
            for value in row {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        eprintln!("[verify-parity] prompt={prompt_len} done");
    }
    writer.flush()?;
    Ok(())
}
