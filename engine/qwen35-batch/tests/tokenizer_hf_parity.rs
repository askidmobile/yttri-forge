//! Токенайзер из GGUF против эталонного HF `tokenizer.json` на родной разметке
//! инструментов Qwen3.5 (блок `# Tools`, `<tool_call><function=…><parameter=…>`,
//! `<tool_response>`, пустой и открытый `<think>`).
//!
//! Сайдкар Yttri (`yforge --sidecar`) режет промпт этим токенайзером сам, поэтому
//! id обязаны совпасть с эталоном до последнего. GGUF — `QWEN35_GGUF`,
//! эталон — `QWEN35_TOKENIZER_JSON`; чего-то нет — SKIP.
#![cfg(feature = "real-model")]

use qwen35_batch::real::tokenizer::load_from_gguf_path;

#[test]
fn gguf_tokenizer_matches_hf_on_native_tool_markup() {
    let (Ok(gguf), Ok(reference)) = (
        std::env::var("QWEN35_GGUF"),
        std::env::var("QWEN35_TOKENIZER_JSON"),
    ) else {
        eprintln!("SKIP: QWEN35_GGUF / QWEN35_TOKENIZER_JSON не заданы");
        return;
    };
    let forge = load_from_gguf_path(std::path::Path::new(&gguf)).unwrap();
    let hf = tokenizers::Tokenizer::from_file(&reference).unwrap();

    let prompt = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n\
        <tools>\n{\"type\": \"function\", \"function\": {\"name\": \"search_notes\", \"description\": \
        \"Search the user's notes\", \"parameters\": {\"properties\": {\"query\": {\"type\": \
        \"string\"}}, \"required\": [\"query\"], \"type\": \"object\"}}}\n</tools>\n\n\
        Ты помощник Yttri.<|im_end|>\n<|im_start|>user\nНайди аренду на Тверской<|im_end|>\n\
        <|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=search_notes>\n\
        <parameter=query>\nаренда Тверская\n</parameter>\n</function>\n</tool_call><|im_end|>\n\
        <|im_start|>user\n<tool_response>\n{\"total\": 3}\n</tool_response><|im_end|>\n\
        <|im_start|>assistant\n";
    let generated = "<think>\nНужен итог.\n</think>\n\nНашёл 3 заметки. 🏢";
    for text in [prompt, generated] {
        let got = forge.encode(text, false).unwrap();
        let want = hf.encode(text, false).unwrap();
        if got.get_ids() != want.get_ids() {
            let at = got
                .get_ids()
                .iter()
                .zip(want.get_ids())
                .take_while(|(a, b)| a == b)
                .count();
            let around = |e: &tokenizers::Encoding| {
                e.get_tokens()[at.saturating_sub(3)..(at + 4).min(e.len())].join("|")
            };
            panic!(
                "GGUF и HF режут по-разному с токена {at}: gguf [{}], hf [{}]",
                around(&got),
                around(&want)
            );
        }
    }
}
