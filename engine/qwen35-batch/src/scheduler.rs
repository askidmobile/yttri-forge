//! Continuous batching scheduler — оркестрирует N слотов над [`BatchModel`].
//!
//! ## Дизайн (см. research doc / D-12 Yttri wiki)
//!
//! - Слоты [`Slot`] проходят IDLE→PREFILL→DECODE→(FINISHED→IDLE).
//! - Каждый выз [`BatchScheduler::step`] делает:
//!   1. **Prefill phase**: выбирает ОДИН Prefilling-слот и кормит его одним
//!      чанком. Когда чанк завершает prompt, логиты последнего токена
//!      сэмплируются → **первый сгенерированный токен** (как в production:
//!      избегаем повторной обработки последнего токена prompt'а в рекуррентном
//!      state — критично для GDN, где повторный forward последнего токена
//!      испортил бы conv/SSM state).
//!   2. **Decode phase**: собирает ВСЕ Decoding-слоты в один [`DecodeBatch`] и
//!      вызывает `model.decode_batch` — батчевые QMatMul амортизируют чтение весов.
//!   3. Сэмплирует токены, обновляет state; завершённые слоты помечаются Finished.
//!      Сбор outputs + reset в Idle делает caller ([`BatchScheduler::run_with_collection`]).
//!
//! ## Паритет
//! [`BatchScheduler::sequential_reference`] прогоняет те же запросы по одному
//! через single-slot scheduler — baseline. Greedy детерминирован ⇒ одинаковые
//! логиты ⇒ одинаковые токены ⇒ batched == sequential.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::Result;

use crate::model::{
    BatchModel, DecodeBatch, DecodeItem, GreedySampler, PrefillChunk, Sampler,
    SpeculativeFallback, SpeculativeMetrics,
};
use crate::slot::{Slot, SlotRequest, SlotStatus};

/// Статистика одного прогона scheduler'а.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Число decode-шагов (вызовов decode_batch).
    pub decode_steps: usize,
    /// Суммарное число токенов, эмитнутых decode-шагами + первых-из-prefill.
    pub total_decode_tokens: usize,
    /// Пиковое число одновременно декодируемых слотов.
    pub max_concurrent_decode: usize,
    /// Число prefill-чанков.
    pub prefill_chunks: usize,
    /// Полное wall-clock время прогона (нс).
    pub wall_ns: u128,
    /// Время prefill-фазы (нс).
    pub prefill_ns: u128,
    /// Время decode-фазы (нс).
    pub decode_ns: u128,
}

impl SchedulerStats {
    /// Агрегатные tok/s по decode-фазе.
    pub fn decode_aggregate_tps(&self) -> f64 {
        let s = self.decode_ns as f64 / 1e9;
        if s <= 0.0 {
            0.0
        } else {
            self.total_decode_tokens as f64 / s
        }
    }
}

/// Размер чанка prefill. usize::MAX = весь prompt одним forward (прототип).
/// Ограничение обязательно для длинных промптов: цельный prefill на N токенов
/// создаёт attention scores N×N×F32×heads (5.6K → ~2 GB transient → CUDA OOM
/// на 12 GB карте). 512 → scores 512×kv_len, десятки MB.
/// Interleaving с decode других слотов безопасен: prefill копит single-slot
/// state, decode работает по batched buffers — пересечений нет.
pub const PREFILL_CHUNK: usize = 512;

/// Размер чанка prefill с учётом env QWEN36_PREFILL_CHUNK (0/большое = целиком).
#[inline]
pub fn prefill_chunk_size() -> usize {
    static SZ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SZ.get_or_init(|| {
        std::env::var("QWEN36_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(PREFILL_CHUNK)
    })
}

/// Ширина спекулятивного драфта K (env QWEN36_MTP_WIDTH, default 3, максимум 8 —
/// потолок mmvq b_size и verify temp-буферов). Тюнится по фактической
/// принимаемости: при m == K re-run принятого префикса не нужен (state уже
/// консистентен), поэтому K чуть НИЖЕ типичной длины всплеска принимаемости
/// выигрывает у большего K. Замер 27B IQ2_XXS (128 ток): K=3 → 26.6 tok/s,
/// K=2 → 22.8, K=4 → 15.1 при baseline 18.6 — всплески упираются в ~3.
#[inline]
pub fn speculative_width() -> usize {
    static W: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("QWEN36_MTP_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(3)
            .min(8)
    })
}

/// P0.5c adaptive width (default ON, откат QWEN36_MTP_ADAPTIVE=0):
/// ширина драфта следует за фактической принимаемостью — см. спекулятивный
/// раунд. Убирает плату draft+rollback ~65 мс за раунды, где модель не
/// угадывает (m<=1), сохраняя длинные всплески при полном принятии.
fn mtp_adaptive_on() -> bool {
    static ADAPTIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ADAPTIVE.get_or_init(|| {
        std::env::var("QWEN36_MTP_ADAPTIVE")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Диагностический trace шагов (QWEN36_TRACE=1) — для расследования зависаний
/// dispatch loop в qwen36-server. Дёшево: одна проверка env на шаг.
fn trace_value_on(value: &str) -> bool {
    matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

#[inline]
pub fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("QWEN36_TRACE").map(|v| trace_value_on(&v)).unwrap_or(false))
}

/// Фазовый тайминг MTP-раунда (env QWEN36_MTP_TIMING=1) — per-round eprintln.
#[inline]
fn mtp_timing_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("QWEN36_MTP_TIMING").map(|v| trace_value_on(&v)).unwrap_or(false)
    })
}

#[cfg(test)]
mod trace_tests {
    use super::trace_value_on;

    #[test]
    fn trace_accepts_only_truthy_values() {
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert!(trace_value_on(value), "{value}");
        }
        for value in ["", "0", "false", "no", "off"] {
            assert!(!trace_value_on(value), "{value}");
        }
    }
}

pub struct BatchScheduler<M: BatchModel> {
    model: M,
    slots: Vec<Slot>,
    queue: VecDeque<SlotRequest>,
    sampler: Box<dyn Sampler>,
    speculative: Vec<SpeculativeMetrics>,
    /// P0.5c adaptive width: последний результат спекуляции per-slot
    /// (m принятых, K драфтованных) + счётчик пропусков для периодического
    /// probe-раунда после серии неудач.
    slot_last_m: Vec<usize>,
    slot_last_k: Vec<usize>,
    skip_count: Vec<usize>,
    stats: SchedulerStats,
    eos: u32,
}

/// Результат одного шага scheduler'а.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// Нечего делать (нет активных слотов/очереди).
    Idle,
    /// Выполнен prefill-чанк (возможно с emit'ом первого сгенерированного токена).
    DidPrefill { first_token_emitted: bool },
    /// Выполнен batched decode-шаг (B = batch size).
    DidDecode(usize),
}

impl<M: BatchModel> BatchScheduler<M> {
    pub fn new(model: M, num_slots: usize, eos: u32, _vocab: usize) -> Self {
        Self {
            model,
            slots: (0..num_slots).map(Slot::new).collect(),
            queue: VecDeque::new(),
            sampler: Box::new(GreedySampler),
            speculative: vec![SpeculativeMetrics::default(); num_slots],
            slot_last_m: vec![1; num_slots],
            slot_last_k: vec![1; num_slots],
            skip_count: vec![0; num_slots],
            stats: SchedulerStats::default(),
            eos,
        }
    }

    /// Подать запрос; возвращает индекс admit'нутого слота, либо None если ставился в очередь.
    pub fn submit(&mut self, prompt: Vec<u32>, max_new: usize) -> Option<usize> {
        let req = SlotRequest {
            prompt,
            max_new,
            eos: self.eos,
        };
        if let Some(idx) = self.idle_slot() {
            self.slots[idx].admit(req);
            self.speculative[idx] = SpeculativeMetrics::default();
            return Some(idx);
        }
        self.queue.push_back(req);
        None
    }

    /// Prefix-cache admit: snapshot уже покрывает `primed_prefix_len` токенов
    /// prompt'а. Слот стартует в Prefilling с prefill_done = primed_prefix_len —
    /// остаётся ОДИН токен (последний), который прогоняется через модель после
    /// restore snapshot'а (adapter.prefill_chunk, reset_first=false).
    /// Caller обязан внедрить snapshot через `model_mut().inject_slot_snapshot`
    /// сразу после возврата индекса слота.
    pub fn submit_primed(
        &mut self,
        prompt: Vec<u32>,
        max_new: usize,
        primed_prefix_len: usize,
    ) -> Option<usize> {
        debug_assert!(primed_prefix_len + 1 == prompt.len());
        let req = SlotRequest {
            prompt,
            max_new,
            eos: self.eos,
        };
        if let Some(idx) = self.idle_slot() {
            self.slots[idx].admit(req);
            self.speculative[idx] = SpeculativeMetrics::default();
            self.slots[idx].prefill_done = primed_prefix_len;
            self.slots[idx].index_pos = primed_prefix_len;
            return Some(idx);
        }
        // В очередь primed не поддерживаем (snapshot injection привязан к слоту);
        // caller должен обрабатывать None как cache-miss → обычный submit.
        None
    }

    fn idle_slot(&self) -> Option<usize> {
        self.slots
            .iter()
            .find(|s| s.status == SlotStatus::Idle)
            .map(|s| s.idx)
    }

    /// Один шаг scheduler'а. НЕ сбрасывает Finished-слоты (сбор+reset делает caller).
    pub fn step(&mut self) -> Result<StepOutcome> {
        self.step_with(&mut |_, _| false)
    }

    /// Шаг с внешним досрочным завершением слота (qwen36-server: stop-строки,
    /// отмена клиента). `should_stop(slot_idx, generated_after_step)` вызывается
    /// ПОСЛЕ model+push_token для каждого эмитнутого токена; true → слот
    /// переводится в Finished (сгенерированный токен сохраняется в generated —
    /// caller решает, эмитить ли его текст). По умолчанию (`step`) — как раньше.
    /// Снапшот статистики для внешней инструментации (P0.5b).
    pub fn stats_snapshot(&self) -> SchedulerStats {
        self.stats.clone()
    }

    pub fn step_with(
        &mut self,
        should_stop: &mut dyn FnMut(usize, &[u32]) -> bool,
    ) -> Result<StepOutcome> {
        self.admit_from_queue();

        // 1. Prefill phase: один чанк для одного Prefilling-слота.
        if let Some((sidx, chunk_size)) = self.next_prefill_chunk() {
            let reset = self.slots[sidx].prefill_done == 0;
            let tokens = self.slots[sidx].prefill_remaining_slice()[..chunk_size].to_vec();
            let start_pos = self.slots[sidx].prefill_done;
            let chunk = PrefillChunk {
                slot_idx: sidx,
                reset_first: reset,
                tokens,
                start_pos,
                is_final: chunk_size == self.slots[sidx].prefill_remaining(),
            };
            let t0 = Instant::now();
            if trace_on() {
                eprintln!("[step] prefill begin slot={sidx} tokens={chunk_size} start={start_pos}");
            }
            let logits = self.model.prefill_chunk(&chunk)?;
            if trace_on() {
                eprintln!("[step] prefill end slot={sidx} ({:?})", t0.elapsed());
            }
            self.stats.prefill_ns += t0.elapsed().as_nanos();
            self.stats.prefill_chunks += 1;
            self.slots[sidx].advance_prefill(chunk_size);
            // Если prompt исчерпан — сэмплируем первый сгенерированный токен
            // из логитов последнего токена prefill (избегаем повторной обработки
            // последнего токена prompt'а в рекуррентном state).
            let mut first_emitted = false;
            if self.slots[sidx].is_decoding() {
                let gen = self.slots[sidx].generated_tokens().to_vec();
                let tok = self.sampler.sample_indexed(sidx, &gen, &logits);
                self.stats.total_decode_tokens += 1;
                self.slots[sidx].push_token(tok);
                if should_stop(sidx, &self.slots[sidx].generated) {
                    self.slots[sidx].status = SlotStatus::Finished;
                }
                first_emitted = true;
            }
            return Ok(StepOutcome::DidPrefill {
                first_token_emitted: first_emitted,
            });
        }

        // 2. Decode phase. Eligible slots run bounded correctness-first MTP
        // transactions; unavailable/failed slots retain one ordinary batched step.
        let decoding: Vec<usize> = self
            .slots
            .iter()
            .filter(|slot| slot.is_decoding())
            .map(|slot| slot.idx)
            .collect();
        if decoding.is_empty() {
            return Ok(StepOutcome::Idle);
        }
        let active = decoding.len();
        self.stats.max_concurrent_decode = self.stats.max_concurrent_decode.max(active);
        let t0 = Instant::now();
        let mut fallback = Vec::new();
        for slot in decoding {
            if should_stop(slot, self.slots[slot].generated_tokens()) {
                self.slots[slot].status = SlotStatus::Finished;
                self.speculative[slot].fallback = Some(SpeculativeFallback::Cancelled);
                continue;
            }
            if !self.model.speculative_available(slot)
                || self.slots[slot].remaining_new_tokens() < 2
            {
                fallback.push(slot);
                continue;
            }
            if let Some(category) = self.speculative_slot(slot, should_stop)? {
                self.speculative[slot].fallback = Some(category);
                if category == SpeculativeFallback::Cancelled {
                    self.slots[slot].status = SlotStatus::Finished;
                } else {
                    fallback.push(slot);
                }
            }
        }
        if !fallback.is_empty() {
            self.baseline_decode_slots(&fallback, should_stop)?;
        }
        self.stats.decode_ns += t0.elapsed().as_nanos();
        self.stats.decode_steps += 1;
        Ok(StepOutcome::DidDecode(active))
    }

    fn speculative_slot(
        &mut self,
        slot: usize,
        should_stop: &mut dyn FnMut(usize, &[u32]) -> bool,
    ) -> Result<Option<SpeculativeFallback>> {
        // Фазовый тайминг раунда (env QWEN36_MTP_TIMING=1) — host-bound decode
        // требует знать, куда уходит время, прежде чем что-то оптимизировать.
        let timing = mtp_timing_on();
        let t_begin = Instant::now();
        self.speculative[slot].enabled = true;
        let sampler_checkpoint = self.sampler.checkpoint(slot);
        if self.model.speculative_begin(slot).is_err() {
            self.model.speculative_rollback(slot)?;
            self.sampler.restore(slot, sampler_checkpoint)?;
            return Ok(Some(SpeculativeFallback::Begin));
        }
        let t_draft = Instant::now();

        // P0.5c adaptive width (QWEN36_MTP_ADAPTIVE=0 для отката): ширина
        // следующего раунда следует за фактической принимаемостью. Полный
        // успех (m>=K) наращивает K; провал (m<=1) срезает к 1 → раунд
        // пропускается (дешёвый baseline decode вместо draft+rollback ~65мс).
        let mut width = self.slots[slot].remaining_new_tokens().min(speculative_width());
        if mtp_adaptive_on() {
            let (last_m, last_k) = (self.slot_last_m[slot], self.slot_last_k[slot]);
            if last_m >= last_k {
                self.slot_last_k[slot] = (last_k + 1).min(speculative_width());
            } else if last_m <= 1 && last_k > 1 {
                self.slot_last_k[slot] = last_k - 1;
            }
            let k_next = self.slot_last_k[slot];
            if k_next <= 1 {
                // Probe каждые 4 шага: не залипнуть в baseline навсегда.
                self.skip_count[slot] += 1;
                if self.skip_count[slot] % 4 != 3 {
                    return Ok(Some(SpeculativeFallback::Draft));
                }
                self.slot_last_k[slot] = 2;
            }
            width = width.min(k_next);
        }

        let draft = match self.model.speculative_draft(
            slot,
            self.slots[slot].current_token(),
            self.slots[slot].next_pos(),
            self.slots[slot].next_pos(),
            width,
        ) {
            Ok(draft) if !draft.is_empty() => draft,
            _ => {
                self.model.speculative_rollback(slot)?;
                self.sampler.restore(slot, sampler_checkpoint)?;
                return Ok(Some(SpeculativeFallback::Draft));
            }
        };
        self.speculative[slot].drafted += draft.len();

        // Batched verify: ОДИН multi-token target-forward на все K позиций.
        // Inputs: current_token + драфт без последнего (draft[K-1] только
        // сравнивается, как input он не подавался и в последовательной схеме).
        let pos = self.slots[slot].next_pos();
        let mut inputs = Vec::with_capacity(draft.len());
        inputs.push(self.slots[slot].current_token());
        inputs.extend_from_slice(&draft[..draft.len() - 1]);
        let t_verify = Instant::now();
        let rows = match self.model.speculative_verify(slot, &inputs, pos) {
            Ok(rows) if rows.len() == inputs.len() => rows,
            _ => {
                self.model.speculative_rollback(slot)?;
                self.sampler.restore(slot, sampler_checkpoint)?;
                return Ok(Some(SpeculativeFallback::Commit));
            }
        };
        let t_sample = Instant::now();
        let mut verified = Vec::with_capacity(draft.len());
        let mut accepted = 0usize;
        for (logits, &draft_id) in rows.iter().zip(&draft) {
            let mut history = self.slots[slot].generated_tokens().to_vec();
            history.extend_from_slice(&verified);
            let target = self.sampler.sample_indexed(slot, &history, logits);
            verified.push(target);
            if target != draft_id {
                break;
            }
            accepted += 1;
            if target == self.eos {
                break;
            }
        }
        let t_accept = Instant::now();
        // Выровнять target state: verify съел все K inputs, принято verified.len().
        if self.model.speculative_accept(slot, verified.len()).is_err() {
            self.model.speculative_rollback(slot)?;
            self.sampler.restore(slot, sampler_checkpoint)?;
            return Ok(Some(SpeculativeFallback::Commit));
        }
        let t_commit = Instant::now();

        if self.model.speculative_commit(slot).is_err() {
            self.model.speculative_rollback(slot)?;
            self.sampler.restore(slot, sampler_checkpoint)?;
            return Ok(Some(SpeculativeFallback::Commit));
        }
        // P0.5c: фиксируем результат раунда для adaptive width.
        self.slot_last_m[slot] = verified.len();
        self.slot_last_k[slot] = draft.len();
        if timing {
            eprintln!(
                "[mtp] slot={slot} K={} m={} begin={:.1}ms draft={:.1}ms verify={:.1}ms sample={:.1}ms accept={:.1}ms commit={:.1}ms",
                draft.len(),
                verified.len(),
                (t_draft - t_begin).as_secs_f64() * 1e3,
                (t_verify - t_draft).as_secs_f64() * 1e3,
                (t_sample - t_verify).as_secs_f64() * 1e3,
                (t_accept - t_sample).as_secs_f64() * 1e3,
                (t_commit - t_accept).as_secs_f64() * 1e3,
                t_commit.elapsed().as_secs_f64() * 1e3,
            );
        }
        self.speculative[slot].used = true;
        self.speculative[slot].accepted += accepted;
        for token in verified {
            if self.slots[slot].push_verified(&[token]) == 0 {
                break;
            }
            self.stats.total_decode_tokens += 1;
            if should_stop(slot, self.slots[slot].generated_tokens()) {
                self.slots[slot].status = SlotStatus::Finished;
                break;
            }
        }
        Ok(None)
    }

    fn baseline_decode_slots(
        &mut self,
        slots: &[usize],
        should_stop: &mut dyn FnMut(usize, &[u32]) -> bool,
    ) -> Result<()> {
        let items = slots
            .iter()
            .filter(|&&slot| self.slots[slot].is_decoding())
            .map(|&slot| DecodeItem {
                slot_idx: slot,
                token: self.slots[slot].current_token(),
                pos: self.slots[slot].next_pos(),
            })
            .collect::<Vec<_>>();
        if items.is_empty() {
            return Ok(());
        }
        let batch = DecodeBatch { items };
        if trace_on() {
            let poss: Vec<usize> = batch.items.iter().map(|item| item.pos).collect();
            eprintln!("[step] decode begin B={} pos={poss:?}", batch.len());
        }
        let logits = self.model.decode_batch(&batch)?;
        if logits.len() != batch.len() {
            anyhow::bail!(
                "decode returned {} rows for batch {}",
                logits.len(),
                batch.len()
            );
        }
        for (item, logits) in batch.items.iter().zip(&logits) {
            let generated = self.slots[item.slot_idx].generated_tokens().to_vec();
            let token = self
                .sampler
                .sample_indexed(item.slot_idx, &generated, logits);
            self.stats.total_decode_tokens += 1;
            self.slots[item.slot_idx].push_token(token);
            if should_stop(item.slot_idx, self.slots[item.slot_idx].generated_tokens()) {
                self.slots[item.slot_idx].status = SlotStatus::Finished;
            }
        }
        Ok(())
    }

    fn admit_from_queue(&mut self) {
        while self.queue.front().is_some() {
            let Some(idx) = self.idle_slot() else { break };
            let req = self.queue.pop_front().unwrap();
            self.slots[idx].admit(req);
            self.speculative[idx] = SpeculativeMetrics::default();
        }
    }

    fn next_prefill_chunk(&self) -> Option<(usize, usize)> {
        for s in &self.slots {
            if s.is_prefilling() {
                let n = s.prefill_remaining_slice().len().min(prefill_chunk_size());
                return Some((s.idx, n));
            }
        }
        None
    }

    fn all_idle(&self) -> bool {
        self.slots.iter().all(|s| s.status == SlotStatus::Idle)
    }

    /// Полный прогон: принимает prompts, прогоняет batched, возвращает сгенерированные
    /// токены per prompt в порядке submit. Собирает outputs из Finished-слотов ДО
    /// сброса (generated валиден только в Finished).
    pub fn run_with_collection(
        &mut self,
        prompts: Vec<Vec<u32>>,
        max_new: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let n = prompts.len();
        for p in prompts {
            self.submit(p, max_new);
        }
        let mut slot_order: Vec<Option<usize>> = vec![None; self.slots.len()];
        let mut next_order = 0usize;
        let mut outputs: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        let t0 = Instant::now();
        loop {
            // admit assigns order; Idle slots без order ждут admit.
            for s in &self.slots {
                if s.status != SlotStatus::Idle && slot_order[s.idx].is_none() {
                    slot_order[s.idx] = Some(next_order);
                    next_order += 1;
                }
            }
            // Собрать outputs из Finished (ДО reset) — generated ещё жив.
            let finished: Vec<(usize, usize)> = self
                .slots
                .iter()
                .filter(|s| s.status == SlotStatus::Finished)
                .filter_map(|s| slot_order[s.idx].map(|o| (s.idx, o)))
                .collect();
            for (sidx, order) in &finished {
                outputs[*order] = self.slots[*sidx].generated_tokens().to_vec();
                self.slots[*sidx].reset();
                self.model.reset_slot(*sidx)?;
                slot_order[*sidx] = None; // освободить order для пере-admit'а.
            }
            match self.step()? {
                StepOutcome::Idle if self.all_idle() && self.queue.is_empty() => break,
                _ => {}
            }
        }
        self.stats.wall_ns = t0.elapsed().as_nanos();
        Ok(outputs)
    }

    /// Полный прогон с per-request `max_new`: каждый промпт имеет собственный
    /// лимит генерации (в отличие от `run_with_collection`, где один max_new на
    /// все). Используется для регрессионных тестов batch shrink: один слот
    /// короче остальных → раннее завершение → batch сжимается.
    ///
    /// `prompts[i].1` = max_new для i-го запроса. Возвращает сгенерированные
    /// токены per prompt в порядке submit.
    pub fn run_with_per_request_max(
        &mut self,
        prompts: Vec<(Vec<u32>, usize)>,
    ) -> Result<Vec<Vec<u32>>> {
        let n = prompts.len();
        for (p, m) in prompts {
            self.submit(p, m);
        }
        let mut slot_order: Vec<Option<usize>> = vec![None; self.slots.len()];
        let mut next_order = 0usize;
        let mut outputs: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        loop {
            for s in &self.slots {
                if s.status != SlotStatus::Idle && slot_order[s.idx].is_none() {
                    slot_order[s.idx] = Some(next_order);
                    next_order += 1;
                }
            }
            let finished: Vec<(usize, usize)> = self
                .slots
                .iter()
                .filter(|s| s.status == SlotStatus::Finished)
                .filter_map(|s| slot_order[s.idx].map(|o| (s.idx, o)))
                .collect();
            for (sidx, order) in &finished {
                outputs[*order] = self.slots[*sidx].generated_tokens().to_vec();
                self.slots[*sidx].reset();
                self.model.reset_slot(*sidx)?;
                slot_order[*sidx] = None;
            }
            match self.step()? {
                StepOutcome::Idle if self.all_idle() && self.queue.is_empty() => break,
                _ => {}
            }
        }
        Ok(outputs)
    }

    /// Baseline: каждый prompt по одному через single-slot scheduler.
    /// Паритет: batched (N slots) == sequential per prompt.
    pub fn sequential_reference(
        model_factory: impl Fn() -> M,
        prompts: Vec<Vec<u32>>,
        max_new: usize,
        eos: u32,
        vocab: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let mut outputs = Vec::with_capacity(prompts.len());
        for p in prompts {
            let mut s = BatchScheduler::new(model_factory(), 1, eos, vocab);
            let mut o = s.run_with_collection(vec![p], max_new)?;
            outputs.append(&mut o);
        }
        Ok(outputs)
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    pub fn speculative_metrics(&self, slot: usize) -> Option<&SpeculativeMetrics> {
        self.speculative.get(slot)
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Установить сэмплер (qwen36-server: per-request sampling params через
    /// `Sampler::sample_indexed`). Default — GreedySampler.
    pub fn set_sampler(&mut self, sampler: Box<dyn Sampler>) {
        self.sampler = sampler;
    }

    /// Доступ к слотам для caller-driven сбора Finished (generated валиден
    /// только до `Slot::reset()`) — continuous-batching loop вне scheduler'а.
    pub fn slots_mut(&mut self) -> &mut [Slot] {
        &mut self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GreedySampler, MockRecurrentModel, PrefillChunk};

    fn prompts() -> Vec<Vec<u32>> {
        vec![
            vec![7, 11, 13, 17],
            vec![3, 5],
            vec![29, 31, 37, 41, 43, 47],
            vec![2],
        ]
    }

    #[test]
    fn batched_equals_sequential_parity() {
        let n_slots = 4;
        let max_new = 12;
        let eos = u32::MAX; // не достигается → гасим по max_new
        let vocab = 4096;
        let p = prompts();
        let mut sched =
            BatchScheduler::new(MockRecurrentModel::new(n_slots, vocab), n_slots, eos, vocab);
        let batched = sched
            .run_with_collection(p.clone(), max_new)
            .expect("batched");
        let seq = BatchScheduler::sequential_reference(
            || MockRecurrentModel::new(1, vocab),
            p,
            max_new,
            eos,
            vocab,
        )
        .expect("seq");
        assert_eq!(batched.len(), seq.len());
        for (i, (b, s)) in batched.iter().zip(seq.iter()).enumerate() {
            assert_eq!(
                b, s,
                "parity fail на prompt {}: batched {:?} vs seq {:?}",
                i, b, s
            );
        }
    }

    #[test]
    fn batched_more_requests_than_slots_recycles() {
        let n_slots = 2;
        let vocab = 1024;
        let max_new = 3;
        let eos = u32::MAX;
        let p: Vec<Vec<u32>> = (0..5)
            .map(|i| vec![i as u32 * 7 + 2, i as u32 + 1])
            .collect();
        let mut sched =
            BatchScheduler::new(MockRecurrentModel::new(n_slots, vocab), n_slots, eos, vocab);
        let batched = sched.run_with_collection(p.clone(), max_new).expect("run");
        let seq = BatchScheduler::sequential_reference(
            || MockRecurrentModel::new(1, vocab),
            p,
            max_new,
            eos,
            vocab,
        )
        .expect("seq");
        assert_eq!(batched, seq);
        assert_eq!(batched.len(), 5);
        for o in &batched {
            assert_eq!(o.len(), max_new);
        }
    }

    #[test]
    fn stats_recorded() {
        let vocab = 512;
        let p = prompts();
        let mut sched = BatchScheduler::new(MockRecurrentModel::new(4, vocab), 4, u32::MAX, vocab);
        let _ = sched.run_with_collection(p, 8).unwrap();
        let st = sched.stats();
        assert!(st.decode_steps > 0);
        assert!(st.total_decode_tokens > 0);
        assert!(st.prefill_chunks > 0);
        assert!(st.max_concurrent_decode >= 1);
    }

    #[test]
    fn batched_shrink_parity_early_finish() {
        // Регрессия slot-indirection: один слот короче остальных → batch
        // сжимается (B=2→1) после раннего завершения. Оставшийся слот должен
        // остаться bit-exact vs sequential (state не протекает в чужой slot).
        // До фикса slot indirection это падало: оставшийся slot читал бы
        // чужой batch_idx=0 state вместо своего slot_idx state.
        let n_slots = 2;
        let vocab = 2048;
        let eos = u32::MAX;
        let prompts: Vec<(Vec<u32>, usize)> = vec![
            (vec![7, 11, 13], 10), // длинный
            (vec![3, 5], 2),       // короткий → раннее завершение → shrink
        ];

        let mut sched =
            BatchScheduler::new(MockRecurrentModel::new(n_slots, vocab), n_slots, eos, vocab);
        let batched = sched
            .run_with_per_request_max(prompts.clone())
            .expect("batched");

        // Sequential reference: каждый prompt по одному в свой single-slot scheduler.
        let mut seq = Vec::with_capacity(prompts.len());
        for (p, m) in prompts {
            let mut s = BatchScheduler::new(MockRecurrentModel::new(1, vocab), 1, eos, vocab);
            let mut o = s.run_with_per_request_max(vec![(p, m)]).expect("seq");
            seq.append(&mut o);
        }

        assert_eq!(batched.len(), seq.len());
        for (i, (b, s)) in batched.iter().zip(seq.iter()).enumerate() {
            assert_eq!(
                b, s,
                "shrink parity fail на prompt {}: batched {:?} vs seq {:?}",
                i, b, s
            );
        }
        assert_eq!(batched[0].len(), 10, "длинный слот не досчитал");
        assert_eq!(batched[1].len(), 2, "короткий слот не досчитал");
    }

    #[test]
    fn first_token_comes_from_prefill_logits() {
        // Один слот: prefill полного prompt'а → первый сгенерированный токен
        // сэмплируется ИЗ логитов последнего токена, возвращённых prefill_chunk.
        // Это критично для реальной модели: мы НЕ переобрабатываем последний
        // токен prompt'а в рекуррентном state (повторный mix/forward испортил бы
        // conv/SSM state GDN). Поэтому ожидание строим той же моделью — greedy-
        // sample логитов prefill_chunk — а не хардкожим формулу (устаревший
        // slot-seed дизайн давал seed=1 и неверное 501).
        let vocab = 1024;
        let prompt = vec![42u32, 7, 19];

        // Reference: первый токен = argmax логитов, возвращённых prefill_chunk.
        let mut ref_model = MockRecurrentModel::new(1, vocab);
        let chunk = PrefillChunk {
            slot_idx: 0,
            reset_first: true,
            tokens: prompt.clone(),
            start_pos: 0,
            is_final: true,
        };
        let ref_logits = ref_model.prefill_chunk(&chunk).unwrap();
        let expected = GreedySampler.sample(&ref_logits);

        // Scheduler: один слот, max_new=1 → ровно первый сгенерированный токен.
        let mut sched = BatchScheduler::new(MockRecurrentModel::new(1, vocab), 1, u32::MAX, vocab);
        let out = sched.run_with_collection(vec![prompt.clone()], 1).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0], expected, "первый токен не из prefill-логитов");
    }
}
