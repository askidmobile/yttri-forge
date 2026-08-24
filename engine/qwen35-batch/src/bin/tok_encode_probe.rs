use qwen35_batch::real::tokenizer;
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("gguf path");
    let tok = tokenizer::load_from_gguf_path(std::path::Path::new(&path))?;
    for t in ["<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|im_start|>assistant\n<think>\n"] {
        let enc = tok.encode(t, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("{:?} -> {:?}", t, enc.get_ids());
    }
    Ok(())
}
