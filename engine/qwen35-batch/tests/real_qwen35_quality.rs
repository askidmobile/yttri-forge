//! Семантический quality-gate для Qwen3.5-4B: реальные ChatML-промпты через
//! настоящий токенизатор из GGUF, сравнение B=1 / B=2 / B=4, проверка качества
//! ответов (не только token parity).
//!
//! Гейт проверяет ДВА независимых свойства:
//! 1. **Greedy parity** — ответы B=2/B=4 покомпонентно равны B=1 (time-multiplexed
//!    snapshot/restore изолирует state ⇒ бит-точное равенство).
//! 2. **Семантическое качество** — декодированные ответы осмысленны: непусты,
//!    не являются эхом промпта, не обрываются на раннем EOS / не повторяются,
//!    содержат ожидаемые ключевые слова там, где они детерминированы.
//!
//! Запуск (macOS/Metal):
//! ```sh
//! YTTRI_MODEL_DIR=/path/to/qwen3.5-4b \
//! cargo test -p qwen35-batch --features real-model \
//!     --test real_qwen35_quality --release -- --ignored --nocapture
//! ```
//! На Windows/Linux CUDA добавь `--features real-model,cuda` и оберни `cargo`
//! через `C:\scripts\with_msvc.ps1`.

#![cfg(feature = "real-model")]

use std::path::PathBuf;

use qwen35_batch::real::tokenizer::{self, ChatMsg};
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::BatchScheduler;
use tokenizers::Tokenizer;

fn model_dir() -> PathBuf {
    std::env::var("YTTRI_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let win_path = PathBuf::from("D:\\Models\\lmstudio-community\\Qwen3.5-4B-GGUF");
            if win_path.exists() {
                win_path
            } else {
                PathBuf::from(
                    "/Volumes/Askid Dev/Projects/Yttri/frontend/src-tauri/resources/models/qwen3.5-4b",
                )
            }
        })
}

fn gguf_path() -> PathBuf {
    model_dir().join("Qwen3.5-4B-Q4_K_M.gguf")
}

fn accelerator_device() -> candle_core::Device {
    // Force CPU path (тест batched decode fallback без Metal/CUDA batched ctx).
    if std::env::var("YTTRI_FORCE_CPU")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!("[quality] YTTRI_FORCE_CPU=1 → Device::Cpu");
        return candle_core::Device::Cpu;
    }
    #[cfg(feature = "cuda")]
    {
        return candle_core::Device::new_cuda(0).expect("CUDA device");
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    {
        qwen35_batch::real::metal_utils::configure_metal_env();
        let device = candle_core::Device::new_metal(0).expect("Metal device");
        qwen35_batch::real::metal_utils::metal_probe(&device).expect("Metal probe");
        return device;
    }
    #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
    panic!("real Qwen3.5 quality gate requires feature `cuda` on this platform");
}

/// Один тестовый запрос.
struct Case {
    name: &'static str,
    messages: Vec<ChatMsg<'static>>,
    /// Для каждой группы ответ обязан содержать хотя бы один вариант (lowercase).
    expect_groups: &'static [&'static [&'static str]],
}

/// Набор реальных запросов: RU/EN, перевод, извлечение, рассуждение, длинный контекст.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "ru_factual_dialogue",
            messages: vec![
                ChatMsg { role: "system", content: "Ты полезный ассистент на русском." },
                ChatMsg { role: "user", content: "Какой металл жидкий при комнатной температуре?" },
            ],
            // ртуть / mercury / hg — любой из вариантов принимается
            expect_groups: &[&["ртут", "mercury", "hg"]],
        },
        Case {
            name: "en_translation_to_ru",
            messages: vec![
                ChatMsg { role: "system", content: "Translate the user's English sentence to Russian." },
                ChatMsg { role: "user", content: "The quick brown fox jumps over the lazy dog." },
            ],
            expect_groups: &[&["лис", "fox"]],
        },
        Case {
            name: "extraction",
            messages: vec![
                ChatMsg { role: "user", content: "Extract the person's name and the date from this text. Reply with only two lines: Name: <name>\nDate: <date>.\n\nOn 5 March 2024, Anna Petrova submitted the quarterly finance report to the board." },
            ],
            expect_groups: &[
                &["anna", "анна"],
                &["petrova", "петров"],
                &["2024"],
            ],
        },
        Case {
            name: "reasoning_summary",
            messages: vec![
                ChatMsg { role: "user", content: "Summarize in one Russian sentence: photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and glucose." },
            ],
            expect_groups: &[
                &["растен", "plants"],
                &["фотосинт", "photosynthesis"],
            ],
        },
    ]
}

/// Закодировать запрос (ChatML + no-think suffix) в токены.
fn encode_case(tokenizer: &Tokenizer, c: &Case) -> Vec<u32> {
    let text = tokenizer::build_chatml_text(&c.messages);
    tokenizer::encode_no_think(tokenizer, &text)
        .unwrap_or_else(|e| panic!("encode case {}: {e}", c.name))
}

/// Сырые generated IDs и их декодированный текст.
#[derive(Debug, PartialEq, Eq)]
struct Generated {
    ids: Vec<u32>,
    text: String,
}

fn decode_generated(tok: &Tokenizer, ids: Vec<u32>) -> Generated {
    let text = tokenizer::decode_text(tok, &ids).expect("decode generated tokens");
    Generated {
        ids,
        text: tokenizer::strip_thinking(&text),
    }
}

/// Максимальная длина повторяющейся униграммы (простой прокси на зацикливание).
fn max_repeated_unigram(text: &str) -> usize {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut runs = 1usize;
    let mut best = 1usize;
    for w in words.windows(2) {
        if w[0].eq_ignore_ascii_case(w[1]) {
            runs += 1;
            best = best.max(runs);
        } else {
            runs = 1;
        }
    }
    best
}

#[test]
#[ignore = "требует GGUF Qwen3.5-4B на диске + GPU; долго грузит ~2.7 ГБ"]
fn real_qwen35_quality_gate() {
    let _ = env_logger::try_init();
    let gguf = gguf_path();
    assert!(gguf.exists(), "GGUF не найден: {:?}", gguf);

    let tokenizer = tokenizer::load_from_gguf_path(&gguf).expect("load tokenizer from GGUF");
    eprintln!("[quality] tokenizer loaded");

    let device = accelerator_device();

    let cases = cases();
    let prompts: Vec<Vec<u32>> = cases.iter().map(|c| encode_case(&tokenizer, c)).collect();
    const MAX_NEW: usize = 72;

    // Ровно одна загрузка весов на каждый режим B=1/B=2/B=4.
    // B=1 adapter одновременно служит источником eos/vocab.
    let adapter_b1 = Qwen35BatchAdapter::load(&gguf, device.clone(), 1).expect("load adapter B=1");
    let eos = adapter_b1.eos();
    let vocab = adapter_b1.vocab_size();
    eprintln!("[quality] adapter params: eos={eos}, vocab={vocab}");

    // --- B=1 reference: один adapter, запросы последовательно ---
    let b1: Vec<Generated> = {
        let mut out = Vec::with_capacity(cases.len());
        let mut sched = BatchScheduler::new(adapter_b1, 1, eos, vocab);
        for prompt in &prompts {
            let generated = sched
                .run_with_collection(vec![prompt.clone()], MAX_NEW)
                .expect("run B=1");
            out.push(decode_generated(
                &tokenizer,
                generated.into_iter().next().unwrap(),
            ));
        }
        out
    };

    // --- B=2: по два промпта ---
    let b2: Vec<Generated> = {
        let adapter = Qwen35BatchAdapter::load(&gguf, device.clone(), 2).expect("load adapter B=2");
        let mut out = Vec::with_capacity(cases.len());
        let mut sched = BatchScheduler::new(adapter, 2, eos, vocab);
        let idxs = (0..cases.len()).collect::<Vec<_>>();
        for chunk in idxs.chunks(2) {
            let ps: Vec<Vec<u32>> = chunk.iter().map(|&i| prompts[i].clone()).collect();
            let outs = sched.run_with_collection(ps, MAX_NEW).expect("run B=2");
            for generated in outs {
                out.push(decode_generated(&tokenizer, generated));
            }
        }
        // Обработка чанков и outputs сохраняет исходный порядок запросов.
        out
    };

    // --- B=4: все промпты сразу ---
    let b4: Vec<Generated> = {
        let adapter = Qwen35BatchAdapter::load(&gguf, device.clone(), 4).expect("load adapter B=4");
        let mut sched = BatchScheduler::new(adapter, 4, eos, vocab);
        let outs = sched
            .run_with_collection(prompts.clone(), MAX_NEW)
            .expect("run B=4");
        outs.into_iter()
            .map(|ids| decode_generated(&tokenizer, ids))
            .collect()
    };

    // ═════════════ Вердикт 1: greedy parity ═════════════
    for i in 0..cases.len() {
        assert_eq!(
            b2[i], b1[i],
            "[parity] case {} ({}): B=2 != B=1",
            i, cases[i].name
        );
        assert_eq!(
            b4[i], b1[i],
            "[parity] case {} ({}): B=4 != B=1",
            i, cases[i].name
        );
    }
    eprintln!(
        "[parity] B=1 == B=2 == B=4: текстовых совпадений OK ({} случаев)",
        cases.len()
    );

    // ═════════════ Вердикт 2: семантическое качество ═════════════
    let mut any_failed = false;
    for (i, c) in cases.iter().enumerate() {
        let generated = &b1[i];
        let text = &generated.text;
        eprintln!("\n[quality] case {} — {}", i, c.name);
        eprintln!(
            "  prompt[last user]: {:.140}…",
            c.messages.last().map(|m| m.content).unwrap_or("")
        );
        eprintln!("  answer: {text:?}");

        // Валидный token stream: не EOS первым токеном и ни одного ID вне vocab.
        if generated.ids.first() == Some(&eos) {
            eprintln!("  ✗ FAIL: EOS первым сгенерированным токеном");
            any_failed = true;
            continue;
        }
        if generated.ids.iter().any(|&id| id as usize >= vocab) {
            eprintln!("  ✗ FAIL: token ID вне vocab");
            any_failed = true;
            continue;
        }
        // Непустой декодированный текст.
        if text.trim().is_empty() {
            eprintln!("  ✗ FAIL: пустой ответ (вероятно ранний EOS / NaN)");
            any_failed = true;
            continue;
        }
        // не эхо: последний user-контент не должен дословно повториться
        if let Some(last_user) = c.messages.iter().rev().find(|m| m.role == "user") {
            if text.contains(last_user.content) {
                eprintln!("  ✗ FAIL: ответ содержит дословный user-промпт (эхо)");
                any_failed = true;
                continue;
            }
        }
        // не зацикливание
        if max_repeated_unigram(text) > 8 {
            eprintln!("  ✗ FAIL: повтор одной униграммы > 8 (вероятное зацикливание)");
            any_failed = true;
            continue;
        }
        // Каждая ожидаемая смысловая группа должна иметь хотя бы одно совпадение.
        let lower = text.to_lowercase();
        let missing = c
            .expect_groups
            .iter()
            .find(|group| !group.iter().any(|kw| lower.contains(kw)));
        if let Some(group) = missing {
            eprintln!("  ✗ FAIL: не найдено ни одного варианта из {group:?}");
            any_failed = true;
            continue;
        }
        eprintln!("  ✓ OK");
    }

    assert!(
        !any_failed,
        "quality-gate: есть провальные случаи (см. вывод выше)"
    );
    eprintln!("\n[quality] ВСЕ случаи прошли both parity и семантическую проверку");
}
