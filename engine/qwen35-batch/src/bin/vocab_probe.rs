//! vocab_probe — диагностика токенов GGUF vocab.
//! Usage: vocab_probe <gguf> [id...]
//! Печатает: всего токенов, сколько содержат U+FFFD, hex+utf8 для заданных id.

use anyhow::Result;
use candle_core::quantized::gguf_file::{Content, Value};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: vocab_probe <gguf> [id...]");
    let ids: Vec<usize> = args.filter_map(|a| a.parse().ok()).collect();

    let mut file = std::fs::File::open(&path)?;
    let ct = Content::read(&mut file)?;
    let tokens = match ct.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(arr)) => arr,
        _ => anyhow::bail!("no tokenizer.ggml.tokens"),
    };
    println!("total tokens: {}", tokens.len());

    let mut fffd_count = 0usize;
    let mut fffd_examples: Vec<(usize, String)> = Vec::new();
    for (i, v) in tokens.iter().enumerate() {
        if let Value::String(s) = v {
            if s.contains('\u{FFFD}') {
                fffd_count += 1;
                if fffd_examples.len() < 8 {
                    fffd_examples.push((i, s.clone()));
                }
            }
        }
    }
    println!("tokens containing U+FFFD: {fffd_count}");
    for (i, s) in &fffd_examples {
        println!("  fffd tok {i}: {s:?} bytes={:?}", s.as_bytes());
    }

    for &id in &ids {
        match tokens.get(id) {
            Some(Value::String(s)) => {
                println!("tok {id}: {s:?} bytes={:?}", s.as_bytes())
            }
            Some(other) => println!("tok {id}: non-string {other:?}"),
            None => println!("tok {id}: OUT OF RANGE"),
        }
    }

    // End-to-end: decode заданной последовательности через production decode_text.
    if !ids.is_empty() {
        let tok = qwen35_batch::real::tokenizer::load_from_gguf_path(std::path::Path::new(&path))?;
        let ids32: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        let text = qwen35_batch::real::tokenizer::decode_text(&tok, &ids32)?;
        println!("decode_text({ids32:?}) = {text:?}");
    }
    Ok(())
}
