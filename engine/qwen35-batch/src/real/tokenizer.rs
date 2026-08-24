//! Tokenizer extracted from GGUF + Qwen3.5 ChatML prompt builder.
//!
//! Копия логики `build_tokenizer_from_ggml` / `escape_json_string` из
//! `Yttri/frontend/src-tauri/src/modules/ai/local_llm/engine.rs:1756-1971`,
//! без Tauri-зависимостей. Используется quality-gate тестом, чтобы кодировать
//! реальные текстовые запросы и декодировать сгенерированные токены.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use candle_core::quantized::gguf_file;
use tokenizers::Tokenizer;

use super::multimodal::{self, IMAGE_PAD_TOKEN_ID, VIDEO_PAD_TOKEN_ID};

/// Загрузить tokenizer напрямую из GGUF-файла (читается только header + metadata
/// через mmap; веса не трогаются). Удобно для тестов, не зависящих от адаптера.
pub fn load_from_gguf_path(path: &std::path::Path) -> Result<Tokenizer> {
    use std::io::Cursor;
    let file = std::fs::File::open(path).map_err(|e| anyhow!("open GGUF: {e}"))?;
    let mmap =
        unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| anyhow!("mmap GGUF: {e}"))?;
    let mut c = Cursor::new(mmap.as_ref());
    let ct = gguf_file::Content::read(&mut c).map_err(|e| anyhow!("read GGUF: {e}"))?;
    load_from_gguf(&ct.metadata)
}

/// ID токена, открывающего thinking-блок в Qwen3.5-4B vocab (glyph U+2192-like).
/// Регистрируется как string из vocab по этому ID, литеральный glyph не пишем.
pub const THINK_OPEN_TOKEN_ID: u32 = 248068;
/// ID токена, закрывающего thinking-блок.
pub const THINK_CLOSE_TOKEN_ID: u32 = 248069;

/// Tool-call токены Qwen3.5: регистрируем по подстроке из vocab (type/name tags).
/// Содержимое берётся из GGUF vocab по этим ID, литеральные glyphы не пишем.
const QWEN3_TOOL_TOKEN_IDS: &[u32] = &[248070, 248071, 248072, 248073];

/// Загрузить tokenizer из GGUF metadata.
///
/// Приоритет: `tokenizer.huggingface.json` (готовый tokenizer.json в metadata),
/// иначе сборка из `tokenizer.ggml.*` ключей (BPE byte-level).
pub fn load_from_gguf(metadata: &HashMap<String, gguf_file::Value>) -> Result<Tokenizer> {
    if let Some(hf_json) = metadata.get("tokenizer.huggingface.json") {
        let json_str = hf_json
            .to_string()
            .map_err(|e| anyhow!("tokenizer.huggingface.json: {e}"))?;
        return Tokenizer::from_bytes(json_str.as_bytes())
            .map_err(|e| anyhow!("Failed to parse tokenizer.huggingface.json: {e}"));
    }
    build_from_ggml_keys(metadata)
}

fn build_from_ggml_keys(metadata: &HashMap<String, gguf_file::Value>) -> Result<Tokenizer> {
    let tokens_val = metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| anyhow!("tokenizer.ggml.tokens not found"))?;
    let tokens_arr = tokens_val
        .to_vec()
        .map_err(|e| anyhow!("tokenizer.ggml.tokens is not an array: {e}"))?;
    let mut vocab: Vec<String> = Vec::with_capacity(tokens_arr.len());
    for val in tokens_arr {
        let s = val
            .to_string()
            .map_err(|e| anyhow!("token is not a string: {e}"))?;
        vocab.push(s.clone());
    }

    let merges: Vec<String> = if let Some(merges_val) = metadata.get("tokenizer.ggml.merges") {
        let merges_arr = merges_val
            .to_vec()
            .map_err(|e| anyhow!("tokenizer.ggml.merges is not an array: {e}"))?;
        let mut result = Vec::with_capacity(merges_arr.len());
        for val in merges_arr {
            let s = val
                .to_string()
                .map_err(|e| anyhow!("merge is not a string: {e}"))?;
            result.push(s.clone());
        }
        result
    } else {
        vec![]
    };

    let token_types: Vec<u32> = if let Some(tt_val) = metadata.get("tokenizer.ggml.token_type") {
        let tt_arr = tt_val
            .to_vec()
            .map_err(|e| anyhow!("tokenizer.ggml.token_type is not an array: {e}"))?;
        // GGUF хранит token_type как I32; to_u32() принимает только U32 и
        // молча давал 1 для всех → <think>/</think> теряли special-статус и
        // токенизировались литералами (<,think,>) — модель не видела открытый
        // think-блок и обрывала ответ EOS'ом без контента (урок 2026-08-19).
        tt_arr
            .iter()
            .map(|v| match v {
                gguf_file::Value::I32(x) => *x as u32,
                gguf_file::Value::U32(x) => *x,
                _ => 1,
            })
            .collect()
    } else {
        vec![1u32; vocab.len()]
    };

    // Набор ID tool-токенов (special:false — видны при decode, нужны для
    // парсинга tool calls в production; quality-gate их не использует, но
    // регистрируем для совместимости с tokenizer из production).
    let tool_id_set: std::collections::HashSet<u32> =
        QWEN3_TOOL_TOKEN_IDS.iter().copied().collect();

    let mut vocab_json = String::from("{");
    for (id, token) in vocab.iter().enumerate() {
        if id > 0 {
            vocab_json.push(',');
        }
        let escaped = escape_json_string(token);
        vocab_json.push_str(&format!("\"{}\":{}", escaped, id));
    }
    vocab_json.push('}');

    let mut merges_json = String::from("[");
    for (i, merge) in merges.iter().enumerate() {
        if i > 0 {
            merges_json.push(',');
        }
        let escaped = escape_json_string(merge);
        merges_json.push_str(&format!("\"{}\"", escaped));
    }
    merges_json.push(']');

    let mut added_tokens_json = String::from("[");
    let mut added_count = 0;
    for (id, token) in vocab.iter().enumerate() {
        let tt = token_types.get(id).copied().unwrap_or(1);
        let is_special_by_type = tt == 3 || tt == 4;
        let is_special_by_name = token.starts_with("<|") && token.ends_with("|>");
        let is_tool_token = tool_id_set.contains(&(id as u32));
        if is_special_by_type || is_special_by_name || is_tool_token {
            if added_count > 0 {
                added_tokens_json.push(',');
            }
            let escaped = escape_json_string(token);
            let special_flag = if is_tool_token { "false" } else { "true" };
            added_tokens_json.push_str(&format!(
                "{{\"id\":{},\"content\":\"{}\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":{}}}",
                id, escaped, special_flag
            ));
            added_count += 1;
        }
    }
    added_tokens_json.push(']');

    let tokenizer_model = metadata
        .get("tokenizer.ggml.model")
        .and_then(|value| value.to_string().ok())
        .map(String::as_str)
        .unwrap_or("gpt2");
    let (pre_tokenizer, decoder) = if tokenizer_model == "gemma4" {
        // Gemma 4 uses SPM-style BPE: raw UTF-8, spaces become U+2581,
        // then merges run over the complete text (no GPT-2 byte mapping).
        let metaspace =
            r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":false}"#;
        (metaspace, metaspace)
    } else {
        let byte_level =
            r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"#;
        (byte_level, byte_level)
    };
    let tokenizer_json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":{},"normalizer":null,"pre_tokenizer":{},"post_processor":null,"decoder":{},"model":{{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":true,"ignore_merges":false,"vocab":{},"merges":{}}}}}"#,
        added_tokens_json, pre_tokenizer, decoder, vocab_json, merges_json,
    );

    Tokenizer::from_bytes(tokenizer_json.as_bytes())
        .map_err(|e| anyhow!("Failed to build tokenizer from GGML: {e}"))
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < '\u{0020}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Текстовое сообщение в роли (`system`/`user`/`assistant`).
pub struct ChatMsg<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Ordered mixed-content block. Media bytes are already processed; this type
/// only renders official placeholder spans without regrouping client content.
pub enum ChatContent<'a> {
    Text(&'a str),
    Image {
        visual_tokens: usize,
    },
    Video {
        frame_tokens: usize,
        timestamps: &'a [f64],
    },
}

pub struct MultimodalChatMsg<'a> {
    pub role: &'a str,
    pub content: &'a [ChatContent<'a>],
}

#[derive(Debug, Eq, PartialEq)]
pub struct EncodedMultimodalPrompt {
    pub ids: Vec<u32>,
    /// 0=text, 1=image, 2=video, matching pinned Transformers.
    pub mm_token_types: Vec<u8>,
}

/// Построить ChatML-текст до открывающего assistant-turn (без no-think блока).
pub fn build_chatml_text(messages: &[ChatMsg<'_>]) -> String {
    let mut prompt = String::with_capacity(2048);
    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(msg.role);
        prompt.push('\n');
        prompt.push_str(msg.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// ChatML с секцией инструментов (официальный Qwen tool template) и ролью
/// `tool` → user с обёрткой <tool_response>. Без tools побайтово равен
/// `build_chatml_text` — регрессий у text-only клиентов нет.
pub fn build_chatml_text_with_tools(
    messages: &[ChatMsg<'_>],
    tools: Option<&serde_json::Value>,
) -> String {
    let tools = tools.and_then(|t| t.as_array()).filter(|a| !a.is_empty());
    let Some(tools) = tools else {
        return build_chatml_text(messages);
    };
    let mut tool_defs = String::new();
    for t in tools {
        // OpenAI-формат: {"type":"function","function":{...}} — в <tools>
        // кладётся сам объект function.
        let f = t.get("function").unwrap_or(t);
        tool_defs.push_str(&serde_json::to_string(f).unwrap_or_default());
        tool_defs.push('\n');
    }
    let tools_section = format!(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tool_defs}</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"
    );
    let mut prompt = String::with_capacity(2048 + tool_defs.len());
    let mut system_done = false;
    for msg in messages {
        prompt.push_str("<|im_start|>");
        if msg.role == "tool" {
            prompt.push_str("user\n<tool_response>\n");
            prompt.push_str(msg.content);
            prompt.push_str("\n</tool_response>");
        } else if msg.role == "system" && !system_done {
            system_done = true;
            prompt.push_str("system\n");
            prompt.push_str(msg.content);
            prompt.push_str(&tools_section);
        } else {
            prompt.push_str(msg.role);
            prompt.push('\n');
            prompt.push_str(msg.content);
        }
        prompt.push_str("<|im_end|>\n");
    }
    if !system_done {
        // Нет system — вставляем tools-секцию первым сообщением.
        let injected = format!(
            "<|im_start|>system{}<|im_end|>\n",
            tools_section.trim_start()
        );
        prompt = injected + &prompt;
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Official ordered mixed-content ChatML. Text-only callers retain
/// `build_chatml_text`, so existing scalar prompt bytes stay unchanged.
pub fn build_chatml_multimodal(messages: &[MultimodalChatMsg<'_>]) -> Result<String> {
    if messages.is_empty() {
        return Err(anyhow!("no messages provided"));
    }
    let mut prompt = String::with_capacity(4096);
    for (index, message) in messages.iter().enumerate() {
        if message.role == "system" && index != 0 {
            return Err(anyhow!("system message must be first"));
        }
        if !matches!(message.role, "system" | "user" | "assistant") {
            return Err(anyhow!("unsupported chat role: {}", message.role));
        }
        let mut content = String::new();
        for part in message.content {
            match part {
                ChatContent::Text(text) => content.push_str(text),
                ChatContent::Image { visual_tokens } => {
                    if message.role == "system" {
                        return Err(anyhow!("system message cannot contain media"));
                    }
                    content.push_str(&multimodal::image_marker(*visual_tokens)?);
                }
                ChatContent::Video {
                    frame_tokens,
                    timestamps,
                } => {
                    if message.role == "system" {
                        return Err(anyhow!("system message cannot contain media"));
                    }
                    content.push_str(&multimodal::video_marker(*frame_tokens, timestamps)?);
                }
            }
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role);
        prompt.push('\n');
        prompt.push_str(content.trim());
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

/// Закодировать prompt + добавить no-think suffix (пустой thinking-блок).
///
/// Формат суффикса совпадает с production `enable_thinking=False`:
/// `THINK_OPEN \n\n THINK_CLOSE \n\n`. Токены newlines берутся из tokenizer.
/// Возвращает готовый Vec<u32> для подачи в scheduler.
pub fn encode_no_think(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("encode prompt: {e}"))?;
    let mut ids: Vec<u32> = enc.get_ids().to_vec();
    append_no_think(tokenizer, &mut ids)?;
    Ok(ids)
}

fn append_no_think(tokenizer: &Tokenizer, ids: &mut Vec<u32>) -> Result<()> {
    let nl_enc = tokenizer
        .encode("\n\n", false)
        .map_err(|e| anyhow!("encode newlines: {e}"))?;
    let nl_ids = nl_enc.get_ids();
    ids.push(THINK_OPEN_TOKEN_ID);
    ids.extend_from_slice(nl_ids);
    ids.push(THINK_CLOSE_TOKEN_ID);
    ids.extend_from_slice(nl_ids);
    Ok(())
}

pub fn encode_multimodal_no_think(
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<EncodedMultimodalPrompt> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("encode multimodal prompt: {e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    append_no_think(tokenizer, &mut ids)?;
    let mm_token_types = ids
        .iter()
        .map(|id| match *id {
            IMAGE_PAD_TOKEN_ID => 1,
            VIDEO_PAD_TOKEN_ID => 2,
            _ => 0,
        })
        .collect();
    Ok(EncodedMultimodalPrompt {
        ids,
        mm_token_types,
    })
}

/// Декодировать сгенерированные токены в текст, пропуская специальные токены.
/// Byte-mapped char → исходный байт (GPT-2 bytes_to_unicode, inverse).
/// Печатные диапазоны мапятся в себя; остальные байты — в U+0100+n по порядку.
fn char_to_byte(ch: char) -> Option<u8> {
    let c = ch as u32;
    let printable = (0x21..=0x7Eu32).contains(&c)
        || (0xA1..=0xACu32).contains(&c)
        || (0xAE..=0xFFu32).contains(&c);
    if printable {
        return Some(c as u8);
    }
    if (0x100..0x200).contains(&c) {
        let idx = (c - 0x100) as usize;
        // n-й непечатный байт в порядке 0..=255.
        let mut n = 0usize;
        for b in 0..=255u32 {
            let p = (0x21..=0x7Eu32).contains(&b)
                || (0xA1..=0xACu32).contains(&b)
                || (0xAE..=0xFFu32).contains(&b);
            if !p {
                if n == idx {
                    return Some(b as u8);
                }
                n += 1;
            }
        }
    }
    None
}

/// Декод токенов в текст. Ручной: собираем байты из byte-mapped строк ВСЕХ
/// токенов и делаем from_utf8_lossy один раз на всю последовательность.
/// Стандартный ByteLevel decoder декодит каждый токен отдельно — emoji/CJK,
/// разрезанные на несколько токенов (частичные UTF-8 последовательности),
/// превращались в U+FFFD. Спецтокены <|...|> пропускаются (skip_special).
pub fn decode_text(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        let Some(tok) = tokenizer.id_to_token(id) else {
            continue;
        };
        if tok.starts_with("<|") && tok.ends_with("|>") {
            continue;
        }
        for ch in tok.chars() {
            match char_to_byte(ch) {
                Some(b) => bytes.push(b),
                // Символ вне byte-mapping (прямая запись, напр. emoji в vocab) —
                // кодируем UTF-8 как есть.
                None => {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Удалить потенциальные хвостовые пустые строки из сгенерированного текста.
/// Специальные токены (включая thinking-блоки) скрыты при decode(skip_special=true),
/// поэтому для no-think ответа результат — чистый ответ модели.
pub fn strip_thinking(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    // Emoji-токен в GGUF vocab: decode должен вернуть emoji, не U+FFFD.
    #[test]
    fn emoji_token_decodes() {
        let mut md = std::collections::HashMap::new();
        md.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(vec![
                Value::String("a".to_string()),
                Value::String("🐱".to_string()),
            ]),
        );
        let tok = build_from_ggml_keys(&md).unwrap();
        let text = decode_text(&tok, &[1]).unwrap();
        assert_eq!(text, "🐱");
    }

    // Byte-mapped форма (как в реальном GGUF): пробел=Ġ(U+0120),
    // 🐱 = F0 9F 90 B1 → ð(U+00F0) Ł(U+0141) Ĳ(U+0132) ±(U+00B1),
    // разрезан на два токена — decode обязан склеить байты до UTF-8.
    #[test]
    fn byte_mapped_split_emoji_decodes() {
        let mut md = std::collections::HashMap::new();
        md.insert(
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(vec![
                Value::String("a".to_string()),
                Value::String("\u{120}\u{F0}\u{141}".to_string()), // " " + F0 9F
                Value::String("\u{132}\u{B1}".to_string()),        // 90 B1
                Value::String("<|im_end|>".to_string()),
            ]),
        );
        let tok = build_from_ggml_keys(&md).unwrap();
        let text = decode_text(&tok, &[1, 2]).unwrap();
        assert_eq!(text, " 🐱");
        // Спецтокен пропускается.
        let text = decode_text(&tok, &[1, 2, 3]).unwrap();
        assert_eq!(text, " 🐱");
    }

    #[test]
    fn text_chatml_path_stays_byte_identical() {
        let messages = [
            ChatMsg {
                role: "system",
                content: "system",
            },
            ChatMsg {
                role: "user",
                content: "Привет",
            },
        ];
        assert_eq!(
            build_chatml_text(&messages),
            "<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nПривет<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
