//! Абстракция модели над batched decode loop.
//!
//! Scheduler не знает про Candle/Metal — он работает через [`BatchModel`].
//! Это позволяет (а) тестировать scheduler на детерминированном mock'е,
//! (б) подключить реальный адаптер Qwen3.5 (feature `real-model`) без правок
//! scheduler'а.
//!
//! ## Контракт batched decode
//! На каждом decode-шаге scheduler передаёт список активных слотов и их
//! текущие токены ([`DecodeBatch`]). Реализация:
//! 1. собирает input-эмбеддинги всех активных слотов в один тензор [B, hidden];
//! 2. выполняет QMatMul-проекции **одним батчем** (амортизирует чтение Q4-весов —
//!    доминирующая cost на bandwidth-bound Apple Silicon; для Q4K fast-path
//!    m%32==0 — паддинг B до 32 см. в реальном адаптере);
//! 3. для recurrent delta_rule + attention — per-slot step по per-slot state
//!    (state не имеет batch-оси в Candle-пути; это потолок без переписывания
//!    Metal-ядер, см. research doc);
//! 4. возвращает logits per slot для сэмплинга.
//!
//! ## Per-slot state
//! Scheduler передаёт stable `slot.idx`. Реализация держит N наборов
//! GDN-state (conv+ssm fp32) + KV-cache, адресуемых по `idx`. Сброс state
//! делается при admit (реализация получает [`PrefillChunk::reset_first`]=true
//! для первого чанка каждого запроса).

use anyhow::Result;
use std::any::Any;

/// Один активный decode-слот на шаге.
#[derive(Debug, Clone, Copy)]
pub struct DecodeItem {
    /// Stable индекс слота (адрес per-slot state в модели).
    pub slot_idx: usize,
    /// Текущий input-token (последний сгенерированный / последний токен prompt).
    pub token: u32,
    /// Абсолютная позиция для RoPE + KV-cache offset.
    pub pos: usize,
}

/// Пакет активных decode-слотов на одном шаге.
#[derive(Debug, Clone)]
pub struct DecodeBatch {
    pub items: Vec<DecodeItem>,
}

impl DecodeBatch {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
}

/// Один чанк prefill для одного слота.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// Stable индекс слота.
    pub slot_idx: usize,
    /// Сбросить per-slot state перед этим чанком (true для первого чанка запроса).
    pub reset_first: bool,
    /// Токены чанка.
    pub tokens: Vec<u32>,
    /// Стартовая абсолютная позиция чанка (для RoPE/KV).
    pub start_pos: usize,
    /// Last chunk of request prompt. MTP catch-up and batched seeding commit only
    /// here; intermediate chunks remain single-slot snapshots.
    pub is_final: bool,
}

/// Opaque multimodal prefill payload. Core scheduler keeps text behavior and
/// implementations without media support reject install explicitly.
#[derive(Debug, Clone)]
pub struct MultimodalPrefill {
    pub token_ids: Vec<u32>,
    /// Vision encoder grids in packed-patch order, regardless of media kind.
    pub media_grids: Vec<[usize; 3]>,
    pub patch_values: Vec<f32>,
    pub patch_rows: usize,
    pub patch_width: usize,
    pub mm_token_types: Vec<u8>,
    pub rope_positions: [Vec<u32>; 3],
    pub decode_rope_delta: i64,
}

/// Сэмплер: логиты → токен.
pub type SamplerCheckpoint = Box<dyn Any + Send>;

pub trait Sampler: Send {
    fn sample(&mut self, logits: &[f32]) -> u32;

    /// Snapshot only state belonging to `slot_idx`. Stateless samplers keep the
    /// default unit checkpoint. Speculative commit retains current state;
    /// rollback restores this checkpoint before baseline decode.
    fn checkpoint(&self, _slot_idx: usize) -> SamplerCheckpoint {
        Box::new(())
    }

    fn restore(&mut self, _slot_idx: usize, _checkpoint: SamplerCheckpoint) -> Result<()> {
        Ok(())
    }

    /// Per-slot вызов (qwen36-server: per-request sampling params + penalties).
    /// `generated` — токены, сгенерированные этим слотом ДО текущего шага
    /// (для presence/repetition penalties). Default — игнорировать slot+generated
    /// (поведение не меняется для GreedySampler и parity-тестов).
    fn sample_indexed(&mut self, _slot_idx: usize, _generated: &[u32], logits: &[f32]) -> u32 {
        self.sample(logits)
    }
}

/// Greedy argmax. Достаточен для parity (batched vs single-stream):
/// одинаковые логиты → одинаковый argmax → одинаковые токены.
#[derive(Default, Clone)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let mut best = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i as u32;
            }
        }
        best
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeFallback {
    Begin,
    Draft,
    Commit,
    Cancelled,
}

impl SpeculativeFallback {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Begin => "begin",
            Self::Draft => "draft",
            Self::Commit => "commit",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculativeMetrics {
    pub enabled: bool,
    pub used: bool,
    pub drafted: usize,
    pub accepted: usize,
    pub fallback: Option<SpeculativeFallback>,
}

/// Абстракция модели над batched prefill + batched decode.
pub trait BatchModel {
    /// Размер словаря.
    fn vocab_size(&self) -> usize;

    fn install_multimodal(&mut self, _slot: usize, _payload: MultimodalPrefill) -> Result<()> {
        anyhow::bail!("model does not support multimodal prefill")
    }

    /// Prefill одного чанка для одного слота. Обновляет per-slot state.
    /// Возвращает logits **последнего** токена чанка → scheduler сэмплирует
    /// первый сгенерированный токен (избегаем повторной обработки последнего
    /// токена prompt'а в рекуррентном state).
    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>>;

    /// Batched decode step: один токен на каждый активный слот, одним батчем.
    /// Возвращает логиты per-item (в порядке `batch.items`).
    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>>;

    /// Whether validated speculative state exists for this slot. Default keeps
    /// all existing model implementations on ordinary decode.
    fn speculative_available(&self, _slot: usize) -> bool {
        false
    }

    /// Checkpoint target and draft state at committed boundary.
    fn speculative_begin(&mut self, _slot: usize) -> Result<()> {
        anyhow::bail!("model does not support speculative transactions")
    }

    /// Advance draft model only. Returned IDs are deterministic draft argmax;
    /// target sampler remains sole authority for committed IDs.
    fn speculative_draft(
        &mut self,
        _slot: usize,
        _token: u32,
        _cache_pos: usize,
        _rope_pos: usize,
        _max_tokens: usize,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("model does not support speculative transactions")
    }

    /// Batched verify: consume `inputs` (K токенов, первый — current_token слота,
    /// далее принятые драфт-токены) на позициях pos..pos+K ОДНИМ multi-token
    /// forward'ом и вернуть логиты КАЖДОЙ позиции. Target state уходит вперёд на
    /// K позиций; caller обязан затем вызвать `speculative_accept` (или rollback).
    fn speculative_verify(
        &mut self,
        _slot: usize,
        _inputs: &[u32],
        _pos: usize,
    ) -> Result<Vec<Vec<f32>>> {
        anyhow::bail!("model does not support speculative transactions")
    }

    /// Выровнять target state после verify: оставить ровно `consumed` первых
    /// inputs (при consumed < K — откат к checkpoint'у транзакции и re-run
    /// принятого префикса одним чанком) и удержать hidden-строки 0..consumed
    /// для MTP commit.
    fn speculative_accept(&mut self, _slot: usize, _consumed: usize) -> Result<()> {
        anyhow::bail!("model does not support speculative transactions")
    }

    /// Align draft state to target-verified rows and release checkpoint.
    fn speculative_commit(&mut self, _slot: usize) -> Result<()> {
        anyhow::bail!("model does not support speculative transactions")
    }

    /// Restore target and draft state. Must be idempotent after begin failure.
    fn speculative_rollback(&mut self, _slot: usize) -> Result<()> {
        Ok(())
    }

    /// Сбросить per-slot state слота `idx` (новый запрос).
    fn reset_slot(&mut self, idx: usize) -> Result<()>;
}

impl<T: BatchModel + ?Sized> BatchModel for &mut T {
    fn vocab_size(&self) -> usize {
        (**self).vocab_size()
    }
    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>> {
        (**self).prefill_chunk(chunk)
    }
    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>> {
        (**self).decode_batch(batch)
    }
    fn reset_slot(&mut self, idx: usize) -> Result<()> {
        (**self).reset_slot(idx)
    }
    fn speculative_available(&self, slot: usize) -> bool {
        (**self).speculative_available(slot)
    }
    fn speculative_begin(&mut self, slot: usize) -> Result<()> {
        (**self).speculative_begin(slot)
    }
    fn speculative_draft(&mut self, slot: usize, token: u32, cp: usize, rp: usize, max_tok: usize) -> Result<Vec<u32>> {
        (**self).speculative_draft(slot, token, cp, rp, max_tok)
    }
    fn speculative_verify(&mut self, slot: usize, inputs: &[u32], pos: usize) -> Result<Vec<Vec<f32>>> {
        (**self).speculative_verify(slot, inputs, pos)
    }
    fn speculative_accept(&mut self, slot: usize, consumed: usize) -> Result<()> {
        (**self).speculative_accept(slot, consumed)
    }
    fn speculative_commit(&mut self, slot: usize) -> Result<()> {
        (**self).speculative_commit(slot)
    }
    fn speculative_rollback(&mut self, slot: usize) -> Result<()> {
        (**self).speculative_rollback(slot)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Mock-модель для unit-тестов scheduler'а.
//
// Критично для parity: per-slot state сбрасывается в 0 при admit и эволюционирует
// ТОЛЬКО от токенов самого слота (как реальная GDN state: reset → нулевой conv/ssm,
// далее детерминированно от входных токенов). Благодаря этому output слота —
// чистая функция его prompt'а, не зависящая от индекса слота и присутствия других
// слотов. Это и проверяет parity batched-vs-sequential: 4 конкурентных prompt'а
// дают те же последовательности, что и поодиночке, значит state не протекает
// между слотами.
// ────────────────────────────────────────────────────────────────────────────

/// Детерминированная pseudo-recurrent mock-модель для тестов scheduler'а.
pub struct MockRecurrentModel {
    vocab: usize,
    /// per-slot running state (имитация GDN ssm-state): u64, reset→0.
    states: Vec<u64>,
}

impl MockRecurrentModel {
    pub fn new(num_slots: usize, vocab: usize) -> Self {
        Self {
            vocab,
            states: vec![0u64; num_slots],
        }
    }

    fn logits_for(&self, token: u32, state: u64) -> Vec<f32> {
        let next = next_token(token, state, self.vocab as u64);
        let mut logits = vec![f32::NEG_INFINITY; self.vocab];
        logits[next as usize] = 1.0;
        logits
    }
}

/// Обновление рекуррентного state новым токеном (детерминированное).
#[inline]
pub fn mix(state: u64, token: u32) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(token as u64)
        .wrapping_add(1442695040888963407)
}

/// Предсказание следующего токена из (текущий токен, state). Детерминированное.
#[inline]
pub fn next_token(token: u32, state: u64, vocab: u64) -> u32 {
    let x = (token as u64)
        .wrapping_mul(2862933555777941757)
        .wrapping_add(state)
        .wrapping_add(3037000493);
    (x % vocab.max(1)) as u32
}

impl BatchModel for MockRecurrentModel {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>> {
        if chunk.reset_first {
            self.reset_slot(chunk.slot_idx)?;
        }
        let mut last = 0u32;
        for &tk in &chunk.tokens {
            self.states[chunk.slot_idx] = mix(self.states[chunk.slot_idx], tk);
            last = tk;
        }
        let state = self.states[chunk.slot_idx];
        Ok(self.logits_for(last, state))
    }

    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>> {
        // Имитация batched decode: каждый слот — независимый stateful шаг.
        // Реальная модель сделала бы batched matmul здесь; mock итерирует.
        let mut out = Vec::with_capacity(batch.items.len());
        for it in &batch.items {
            // consume token → update state → predict.
            self.states[it.slot_idx] = mix(self.states[it.slot_idx], it.token);
            let state = self.states[it.slot_idx];
            out.push(self.logits_for(it.token, state));
        }
        Ok(out)
    }

    fn reset_slot(&mut self, idx: usize) -> Result<()> {
        self.states[idx] = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_rejects_multimodal_install() {
        let mut model = MockRecurrentModel::new(1, 32);
        let payload = MultimodalPrefill {
            token_ids: vec![1],
            media_grids: vec![],
            patch_values: vec![],
            patch_rows: 0,
            patch_width: 0,
            mm_token_types: vec![0],
            rope_positions: [vec![0], vec![0], vec![0]],
            decode_rope_delta: 0,
        };
        assert_eq!(
            model
                .install_multimodal(0, payload)
                .unwrap_err()
                .to_string(),
            "model does not support multimodal prefill"
        );
    }

    #[test]
    fn mock_output_is_pure_function_of_prompt() {
        // Один и тот же prompt через два РАЗНЫХ слота → одинаковая последовательность.
        let prompt = vec![10u32, 20, 30];
        let run = |slot: usize| -> Vec<u32> {
            let mut m = MockRecurrentModel::new(4, 500);
            let chunk = PrefillChunk {
                slot_idx: slot,
                reset_first: true,
                tokens: prompt.clone(),
                start_pos: 0,
                is_final: true,
            };
            let l0 = m.prefill_chunk(&chunk).unwrap();
            let mut seq = vec![GreedySampler.sample(&l0)];
            for _ in 0..3 {
                let cur = *seq.last().unwrap();
                let l = m
                    .decode_batch(&DecodeBatch {
                        items: vec![DecodeItem {
                            slot_idx: slot,
                            token: cur,
                            pos: 0,
                        }],
                    })
                    .unwrap();
                seq.push(GreedySampler.sample(&l[0]));
            }
            seq
        };
        assert_eq!(
            run(0),
            run(3),
            "output зависит от индекса слота — state протекает"
        );
    }

    #[test]
    fn mock_concurrent_slots_isolated() {
        // Два слота с разными prompt'ами одновременно: state не протекает.
        let mut m = MockRecurrentModel::new(2, 700);
        let c0 = PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: vec![5, 6, 7],
            start_pos: 0,
            is_final: true,
        };
        let c1 = PrefillChunk {
            slot_idx: 1,
            reset_first: true,
            tokens: vec![50, 60],
            start_pos: 0,
            is_final: true,
        };
        let l0 = m.prefill_chunk(&c0).unwrap();
        let l1 = m.prefill_chunk(&c1).unwrap();
        // Разные prompt'ы → почти наверняка разные state → разные next.
        assert_ne!(GreedySampler.sample(&l0), GreedySampler.sample(&l1));
        // Состояния независимы: state[0] != state[1].
        assert_ne!(m.states[0], m.states[1]);
    }
}
