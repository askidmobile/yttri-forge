//! Per-slot state management — состояние одного из N параллельных слотов.
//!
//! В реальной Qwen3.5 это: (а) рекуррентное состояние 24 DeltaNet-слоёв
//! (conv_buf + ssm_state, fp32 — f16 дрейфует на 5K+ токенах по данным Yttri),
//! (б) KV-cache 8 attention-слоёв, (в) позиция index_pos для RoPE/KV.
//! Веса модели — shared, per-state — отдельные.
//!
//! Здесь описан lifecycle-стейт-машинный каркас (IDLE→PREFILL→DECODE→DONE),
//! не зависящий от конкретной реализации модели (см. [`crate::model::BatchModel`]).

/// Lifecycle одного слота генерации.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotStatus {
    /// Свободен, готов принять новый запрос.
    Idle,
    /// В режиме prefill: токены prompt'а подаются последовательными чанками.
    Prefilling,
    /// В режиме decode: генерация новых токенов батчем с другими слотами.
    Decoding,
    /// Завершён (EOS или max_new достигнут) — готов к перезапуску в Idle.
    Finished,
}

/// Границы токенов промпта для одного чанка prefill.
///
/// Continuous batching: prefill admits выполняются чанками, чтобы decode-слоты
/// не голодали (LLM-level batching). В минимальной версии один чанк = весь
/// prompt (prefill последовательный по дизайну прототипа).
#[derive(Debug, Clone, Copy)]
pub struct PrefillChunk {
    /// Стартовый индекс в prompt-токенах слота.
    pub start: usize,
    /// Сколько токенов покрыть этим чанком (включительно от start).
    pub len: usize,
}

/// Запрос, поданный в слот.
#[derive(Debug, Clone)]
pub struct SlotRequest {
    /// Токены промпта (tokenized).
    pub prompt: Vec<u32>,
    /// Сколько новых токенов сгенерировать максимум.
    pub max_new: usize,
    /// EOS token id; достижение завершает слот.
    pub eos: u32,
}

/// Один слот генерации с собственным state.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Стабильный индекс слота 0..N (для адресации per-slot state в модели).
    pub idx: usize,
    pub status: SlotStatus,
    /// Активный запрос (None в Idle/Finished).
    pub request: Option<SlotRequest>,
    /// Сколько токенов prompt'а уже обработано prefill'ом.
    pub prefill_done: usize,
    /// Сгенерированные decode-токены (после преfill'а).
    pub generated: Vec<u32>,
    /// Текущая абсолютная позиция в последовательности (= prefill_done + generated.len()).
    /// Нужна модели для RoPE index_pos и append'а KV-cache.
    /// Возрастает на 1 за decode-шаг; во время prefill — растёт на размер чанка.
    /// pub(crate): scheduler выставляет при primed-admit (prefix cache).
    pub(crate) index_pos: usize,
}

impl Slot {
    pub fn new(idx: usize) -> Self {
        Self {
            idx,
            status: SlotStatus::Idle,
            request: None,
            prefill_done: 0,
            generated: Vec::new(),
            index_pos: 0,
        }
    }

    /// Сброс слота в Idle (при admit нового запроса или после завершения).
    pub fn reset(&mut self) {
        self.status = SlotStatus::Idle;
        self.request = None;
        self.prefill_done = 0;
        self.generated.clear();
        self.index_pos = 0;
    }

    /// Принять новый запрос и перейти в Prefilling.
    pub fn admit(&mut self, req: SlotRequest) {
        debug_assert_eq!(self.status, SlotStatus::Idle, "admit в занятый слот");
        self.reset();
        self.request = Some(req);
        self.status = SlotStatus::Prefilling;
    }

    /// Токены prompt'а, ещё не обработанные prefill'ом.
    pub fn prefill_remaining_slice(&self) -> &[u32] {
        match &self.request {
            Some(r) => &r.prompt[self.prefill_done..],
            None => &[],
        }
    }

    /// Сколько токенов prefill осталось обработать.
    pub fn prefill_remaining(&self) -> usize {
        self.prefill_remaining_slice().len()
    }

    /// Отметить, что prefill обработал `n` токенов (после чанка).
    pub fn advance_prefill(&mut self, n: usize) {
        self.prefill_done += n;
        // index_pos во время prefill соответствует растущему контексту.
        // Реальная модель сама знает позиции; здесь лишь bookkeeping.
        self.index_pos = self.prefill_done;
        if let Some(r) = &self.request {
            if self.prefill_done >= r.prompt.len() {
                // Prefill завершён — переходим в decode.
                self.status = SlotStatus::Decoding;
            }
        }
    }

    /// Последний токен контекста (для подачи в decode-шаг как input).
    ///
    /// В decode фазе это последний сгенерированный токен; на первом decode-шаге
    /// (преход Prefilling→Decoding) это последний токен prompt'а.
    pub fn current_token(&self) -> u32 {
        if let Some(g) = self.generated.last() {
            return *g;
        }
        match &self.request {
            Some(r) => *r.prompt.last().expect("пустой prompt"),
            None => 0,
        }
    }

    /// Абсолютная позиция текущего generated input-token для decode.
    /// Первый token сэмплируется из prefill logits, но ещё не записан в KV.
    pub fn next_pos(&self) -> usize {
        self.index_pos
            .saturating_sub(usize::from(!self.generated.is_empty()))
    }

    pub fn remaining_new_tokens(&self) -> usize {
        self.request
            .as_ref()
            .map(|request| request.max_new.saturating_sub(self.generated.len()))
            .unwrap_or(0)
    }

    /// Добавить сгенерированный токен; возвращает `true` если слот завершён.
    pub fn push_token(&mut self, tok: u32) -> bool {
        self.push_verified(&[tok]) == 1 && self.status == SlotStatus::Finished
    }

    /// Commit verified prefix token-by-token. EOS/max_new are checked after
    /// every token; uncommitted suffix never changes cursor.
    pub fn push_verified(&mut self, tokens: &[u32]) -> usize {
        let mut committed = 0;
        for &tok in tokens {
            if self.status != SlotStatus::Decoding || self.remaining_new_tokens() == 0 {
                break;
            }
            self.generated.push(tok);
            self.index_pos += 1;
            committed += 1;
            let done = match &self.request {
                Some(request) => tok == request.eos || self.generated.len() >= request.max_new,
                None => true,
            };
            if done {
                self.status = SlotStatus::Finished;
                break;
            }
        }
        committed
    }

    /// Готов к decode (prefill сделан, ещё не завершён)?
    pub fn is_decoding(&self) -> bool {
        self.status == SlotStatus::Decoding
    }

    /// Готов к prefill-чанку прямо сейчас?
    pub fn is_prefilling(&self) -> bool {
        self.status == SlotStatus::Prefilling
    }

    /// Финальная последовательность токенов (prompt + generated), для parity-check.
    pub fn full_sequence(&self) -> Vec<u32> {
        let mut v = match &self.request {
            Some(r) => r.prompt.clone(),
            None => Vec::new(),
        };
        v.extend_from_slice(&self.generated);
        v
    }

    /// Сгенерированное (без prompt) — для проверки parity по длине/токенам.
    pub fn generated_tokens(&self) -> &[u32] {
        &self.generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_lifecycle() {
        let mut s = Slot::new(0);
        assert_eq!(s.status, SlotStatus::Idle);
        s.admit(SlotRequest {
            prompt: vec![1, 2, 3],
            max_new: 2,
            eos: 99,
        });
        assert_eq!(s.status, SlotStatus::Prefilling);
        assert_eq!(s.prefill_remaining(), 3);
        // Чанк prefill = 2 токена.
        s.advance_prefill(2);
        assert_eq!(s.status, SlotStatus::Prefilling);
        // Финальный чанк.
        s.advance_prefill(1);
        assert_eq!(s.status, SlotStatus::Decoding);
        assert_eq!(s.current_token(), 3);
        assert_eq!(s.next_pos(), 3);
        // Decode шаг 1.
        assert!(!s.push_token(7));
        assert_eq!(s.current_token(), 7);
        assert_eq!(s.next_pos(), 3);
        s.push_token(8);
        assert_eq!(s.next_pos(), 4);
        // Decode шаг 2 (max_new) уже сэмплирован → finished.
        assert_eq!(s.status, SlotStatus::Finished);
        assert_eq!(s.full_sequence(), vec![1, 2, 3, 7, 8]);
    }

    #[test]
    fn verified_prefix_stops_at_eos_and_limit() {
        let mut eos = Slot::new(0);
        eos.admit(SlotRequest {
            prompt: vec![1],
            max_new: 4,
            eos: 9,
        });
        eos.advance_prefill(1);
        assert_eq!(eos.push_verified(&[2, 9, 3]), 2);
        assert_eq!(eos.generated_tokens(), &[2, 9]);
        assert_eq!(eos.next_pos(), 2);

        let mut limited = Slot::new(1);
        limited.admit(SlotRequest {
            prompt: vec![1],
            max_new: 2,
            eos: 99,
        });
        limited.advance_prefill(1);
        assert_eq!(limited.push_verified(&[2, 3, 4]), 2);
        assert_eq!(limited.generated_tokens(), &[2, 3]);
    }

    #[test]
    fn slot_eos_terminates() {
        let mut s = Slot::new(1);
        s.admit(SlotRequest {
            prompt: vec![5],
            max_new: 100,
            eos: 42,
        });
        s.advance_prefill(1);
        assert!(s.is_decoding());
        s.push_token(10);
        assert!(!s.is_finished_via_eos());
        assert!(s.push_token(42)); // EOS
        assert_eq!(s.status, SlotStatus::Finished);
    }

    impl Slot {
        fn is_finished_via_eos(&self) -> bool {
            self.status == SlotStatus::Finished
        }
    }
}
