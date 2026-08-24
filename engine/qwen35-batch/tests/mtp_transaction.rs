use anyhow::{anyhow, Result};
use qwen35_batch::model::{
    mix, next_token, BatchModel, DecodeBatch, PrefillChunk, Sampler, SamplerCheckpoint,
    SpeculativeFallback, SpeculativeMetrics,
};
use qwen35_batch::scheduler::BatchScheduler;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    Begin,
    Draft,
    Commit,
}

struct TransactionModel {
    vocab: usize,
    states: Vec<u64>,
    checkpoint: Vec<Option<u64>>,
    verify_inputs: Vec<Option<Vec<u32>>>,
    failure: Failure,
    reject_at: Option<usize>,
}

impl TransactionModel {
    fn new(slots: usize, vocab: usize) -> Self {
        Self {
            vocab,
            states: vec![0; slots],
            checkpoint: vec![None; slots],
            verify_inputs: vec![None; slots],
            failure: Failure::None,
            reject_at: None,
        }
    }

    fn logits(&self, token: u32, state: u64) -> Vec<f32> {
        let mut logits = vec![f32::NEG_INFINITY; self.vocab];
        logits[next_token(token, state, self.vocab as u64) as usize] = 1.0;
        logits
    }
}

impl BatchModel for TransactionModel {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>> {
        if chunk.reset_first {
            self.states[chunk.slot_idx] = 0;
        }
        let mut last = 0;
        for &token in &chunk.tokens {
            self.states[chunk.slot_idx] = mix(self.states[chunk.slot_idx], token);
            last = token;
        }
        Ok(self.logits(last, self.states[chunk.slot_idx]))
    }

    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(batch.len());
        for item in &batch.items {
            self.states[item.slot_idx] = mix(self.states[item.slot_idx], item.token);
            out.push(self.logits(item.token, self.states[item.slot_idx]));
        }
        Ok(out)
    }

    fn speculative_available(&self, slot: usize) -> bool {
        slot % 2 == 0
    }

    fn speculative_begin(&mut self, slot: usize) -> Result<()> {
        if self.failure == Failure::Begin {
            return Err(anyhow!("injected begin"));
        }
        self.checkpoint[slot] = Some(self.states[slot]);
        Ok(())
    }

    fn speculative_draft(
        &mut self,
        slot: usize,
        mut token: u32,
        _cache_pos: usize,
        _rope_pos: usize,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        if self.failure == Failure::Draft {
            return Err(anyhow!("injected draft"));
        }
        let mut state = self.states[slot];
        let mut draft = Vec::new();
        for index in 0..max_tokens {
            state = mix(state, token);
            token = next_token(token, state, self.vocab as u64);
            if self.reject_at == Some(index) {
                token = (token + 1) % self.vocab as u32;
            }
            draft.push(token);
        }
        Ok(draft)
    }

    fn speculative_verify(&mut self, slot: usize, inputs: &[u32], _pos: usize) -> Result<Vec<Vec<f32>>> {
        // Как K decode-шагов подряд: state съедает все inputs, логиты на каждой позиции.
        let mut out = Vec::with_capacity(inputs.len());
        for &token in inputs {
            self.states[slot] = mix(self.states[slot], token);
            out.push(self.logits(token, self.states[slot]));
        }
        self.verify_inputs[slot] = Some(inputs.to_vec());
        Ok(out)
    }

    fn speculative_accept(&mut self, slot: usize, consumed: usize) -> Result<()> {
        let inputs = self.verify_inputs[slot]
            .take()
            .ok_or_else(|| anyhow!("accept without verify"))?;
        if consumed < inputs.len() {
            // Откат к checkpoint'у + re-run принятого префикса (как в адаптере).
            let base = self.checkpoint[slot].ok_or_else(|| anyhow!("no checkpoint"))?;
            self.states[slot] = base;
            for &token in &inputs[..consumed] {
                self.states[slot] = mix(self.states[slot], token);
            }
        }
        Ok(())
    }

    fn speculative_commit(&mut self, slot: usize) -> Result<()> {
        if self.failure == Failure::Commit {
            return Err(anyhow!("injected commit"));
        }
        self.checkpoint[slot] = None;
        Ok(())
    }

    fn speculative_rollback(&mut self, slot: usize) -> Result<()> {
        if let Some(state) = self.checkpoint[slot].take() {
            self.states[slot] = state;
        }
        self.verify_inputs[slot] = None;
        Ok(())
    }

    fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.states[slot] = 0;
        self.checkpoint[slot] = None;
        self.verify_inputs[slot] = None;
        Ok(())
    }
}

#[derive(Clone)]
struct TestSampler {
    states: Vec<u64>,
}

impl TestSampler {
    fn new(slots: usize) -> Self {
        Self {
            states: vec![1; slots],
        }
    }
}

impl Sampler for TestSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0 as u32
    }

    fn checkpoint(&self, slot: usize) -> SamplerCheckpoint {
        Box::new(self.states[slot])
    }

    fn restore(&mut self, slot: usize, checkpoint: SamplerCheckpoint) -> Result<()> {
        self.states[slot] = *checkpoint
            .downcast::<u64>()
            .map_err(|_| anyhow!("bad sampler checkpoint"))?;
        Ok(())
    }

    fn sample_indexed(&mut self, slot: usize, _generated: &[u32], logits: &[f32]) -> u32 {
        self.states[slot] = mix(self.states[slot], slot as u32 + 1);
        self.sample(logits)
    }
}

fn run(
    slots: usize,
    failure: Failure,
    reject_at: Option<usize>,
) -> (Vec<Vec<u32>>, Vec<SpeculativeMetrics>) {
    let vocab = 257;
    let mut model = TransactionModel::new(slots, vocab);
    model.failure = failure;
    model.reject_at = reject_at;
    let mut scheduler = BatchScheduler::new(model, slots, u32::MAX, vocab);
    scheduler.set_sampler(Box::new(TestSampler::new(slots)));
    let prompts: Vec<Vec<u32>> = (0..slots)
        .map(|slot| vec![slot as u32 + 3, slot as u32 + 9])
        .collect();
    let mut output = vec![Vec::new(); slots];
    let mut metrics = vec![SpeculativeMetrics::default(); slots];
    for (slot, prompt) in prompts.into_iter().enumerate() {
        scheduler.submit(prompt, 9);
        metrics[slot] = SpeculativeMetrics::default();
    }
    let mut order = vec![None; slots];
    let mut next = 0usize;
    loop {
        for slot in scheduler.slots_mut().iter() {
            if slot.status != qwen35_batch::slot::SlotStatus::Idle && order[slot.idx].is_none() {
                order[slot.idx] = Some(next);
                next += 1;
            }
        }
        let finished = scheduler
            .slots_mut()
            .iter()
            .filter(|slot| slot.status == qwen35_batch::slot::SlotStatus::Finished)
            .map(|slot| slot.idx)
            .collect::<Vec<_>>();
        for slot in finished {
            let output_index = order[slot].unwrap();
            output[output_index] = scheduler.slots_mut()[slot].generated_tokens().to_vec();
            metrics[output_index] = scheduler.speculative_metrics(slot).unwrap().clone();
            scheduler.slots_mut()[slot].reset();
            scheduler.model_mut().reset_slot(slot).unwrap();
            order[slot] = None;
        }
        if scheduler.step().unwrap() == qwen35_batch::scheduler::StepOutcome::Idle
            && scheduler
                .slots_mut()
                .iter()
                .all(|slot| slot.status == qwen35_batch::slot::SlotStatus::Idle)
        {
            break;
        }
    }
    (output, metrics)
}

#[test]
fn mtp_matches_baseline_for_b1_to_b4_and_mixed_fallback() {
    for slots in 1..=4 {
        let (actual, metrics) = run(slots, Failure::None, None);
        let baseline = (0..slots)
            .map(|slot| {
                let mut scheduler = BatchScheduler::new(
                    TransactionModel {
                        failure: Failure::Begin,
                        ..TransactionModel::new(1, 257)
                    },
                    1,
                    u32::MAX,
                    257,
                );
                scheduler.set_sampler(Box::new(TestSampler::new(1)));
                scheduler
                    .run_with_collection(vec![vec![slot as u32 + 3, slot as u32 + 9]], 9)
                    .unwrap()
                    .remove(0)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, baseline, "B={slots}");
        assert!(metrics[0].used);
        for slot in (1..slots).step_by(2) {
            assert!(!metrics[slot].enabled, "fallback slot {slot}");
        }
    }
}

#[test]
fn reject_and_failures_restore_then_match_baseline() {
    let (baseline, _) = run(4, Failure::Begin, None);
    for (failure, reject_at, expected) in [
        (Failure::None, Some(0), None),
        (Failure::None, Some(2), None),
        (Failure::Draft, None, Some(SpeculativeFallback::Draft)),
        (Failure::Commit, None, Some(SpeculativeFallback::Commit)),
    ] {
        let (actual, metrics) = run(4, failure, reject_at);
        assert_eq!(actual, baseline);
        if let Some(expected) = expected {
            assert_eq!(metrics[0].fallback, Some(expected));
        }
    }
}

#[test]
fn cancellation_before_verify_emits_nothing() {
    let mut scheduler = BatchScheduler::new(TransactionModel::new(1, 257), 1, u32::MAX, 257);
    scheduler.submit(vec![3, 9], 8);
    scheduler.step().unwrap();
    let mut cancel = |_: usize, _: &[u32]| true;
    scheduler.step_with(&mut cancel).unwrap();
    assert_eq!(scheduler.slots_mut()[0].generated_tokens().len(), 1);
    assert_eq!(
        scheduler.speculative_metrics(0).unwrap().fallback,
        Some(SpeculativeFallback::Cancelled)
    );
}
