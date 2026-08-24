//! Интеграционные тесты continuous-batching scheduler'а и per-slot state
//! на детерминированной mock-модели (без GGUF / Metal — быстро и воспроизводимо).
//!
//! Что проверяем:
//! 1. Parity: batched (N слотов) == sequential baseline per prompt.
//! 2. Slot recycling: запросов больше, чем слотов — outputs не смешиваются.
//! 3. Изоляция per-slot state при конкурентной обработке.
//! 4. Scheduler stats (batch sizes, max_concurrent) корректны.

use qwen35_batch::model::MockRecurrentModel;
use qwen35_batch::scheduler::BatchScheduler;

const EOS: u32 = u32::MAX; // не достигается в тестах → завершение по max_new

fn var_len_prompts(n: usize) -> Vec<Vec<u32>> {
    (0..n)
        .map(|i| {
            let len = 2 + (i * 7) % 9;
            (0..len)
                .map(|j| (i * 31 + j * 17 + 3) as u32 % 977)
                .collect()
        })
        .collect()
}

#[test]
fn parity_batched_vs_sequential_varlen() {
    let vocab = 2048;
    let prompts = var_len_prompts(8);
    let max_new = 16;

    let mut sched = BatchScheduler::new(MockRecurrentModel::new(4, vocab), 4, EOS, vocab);
    let batched = sched.run_with_collection(prompts.clone(), max_new).unwrap();

    let seq = BatchScheduler::sequential_reference(
        || MockRecurrentModel::new(1, vocab),
        prompts,
        max_new,
        EOS,
        vocab,
    )
    .unwrap();

    assert_eq!(
        batched, seq,
        "batched != sequential — per-slot state протекает"
    );
}

#[test]
fn max_concurrent_reaches_num_slots() {
    let vocab = 512;
    // 4 одинаково-длинных prompt'а, max_new большой → все 4 декодируют одновременно.
    let prompts = vec![
        vec![1u32, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
        vec![10, 11, 12],
    ];
    let mut sched = BatchScheduler::new(MockRecurrentModel::new(4, vocab), 4, EOS, vocab);
    let out = sched.run_with_collection(prompts, 24).unwrap();
    assert_eq!(out.len(), 4);
    assert_eq!(
        sched.stats().max_concurrent_decode,
        4,
        "слоты не работали параллельно"
    );
}

#[test]
fn recycling_preserves_output_mapping() {
    let vocab = 1024;
    let prompts = var_len_prompts(6); // 6 запросов на 2 слота
    let mut sched = BatchScheduler::new(MockRecurrentModel::new(2, vocab), 2, EOS, vocab);
    let batched = sched.run_with_collection(prompts.clone(), 10).unwrap();
    let seq = BatchScheduler::sequential_reference(
        || MockRecurrentModel::new(1, vocab),
        prompts,
        10,
        EOS,
        vocab,
    )
    .unwrap();
    assert_eq!(batched.len(), 6);
    assert_eq!(batched, seq, "outputs смешались при recycling слотов");
}

#[test]
fn eos_terminates_early_when_reached() {
    // Подберём prompt так, чтобы EOS (маленькое значение) встретилось рано.
    let vocab = 64;
    let eos = 7u32;
    let prompts = vec![vec![1u32, 2, 3], vec![9u32, 8, 7]];
    let mut sched = BatchScheduler::new(MockRecurrentModel::new(2, vocab), 2, eos, vocab);
    let out = sched.run_with_collection(prompts, 50).unwrap();
    assert_eq!(out.len(), 2);
    for (i, o) in out.iter().enumerate() {
        if o.len() < 50 {
            assert_eq!(
                *o.last().unwrap(),
                eos,
                "prompt {}: ранняя остановка не на EOS",
                i
            );
        }
    }
}
