use anyhow::Result;
use qwen35_batch::model::{
    mix, next_token, BatchModel, DecodeBatch, PrefillChunk, Sampler, SamplerCheckpoint,
    SpeculativeMetrics,
};
use qwen35_batch::scheduler::BatchScheduler;
use qwen35_batch::slot::SlotStatus;

struct ParityMockModel {
    vocab: usize,
    states: Vec<u64>,
    checkpoints: Vec<Option<u64>>,
    verify_inputs: Vec<Option<Vec<u32>>>,
    draft_enabled: bool,
    force_reject_step: Option<usize>,
    step_count: usize,
}

impl ParityMockModel {
    fn new(slots: usize, vocab: usize, draft_enabled: bool) -> Self {
        Self {
            vocab,
            states: vec![0; slots],
            checkpoints: vec![None; slots],
            verify_inputs: vec![None; slots],
            draft_enabled,
            force_reject_step: None,
            step_count: 0,
        }
    }

    fn logits(&self, token: u32, state: u64) -> Vec<f32> {
        let mut logits = vec![f32::NEG_INFINITY; self.vocab];
        logits[next_token(token, state, self.vocab as u64) as usize] = 1.0;
        logits
    }
}

impl BatchModel for ParityMockModel {
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
        self.step_count += 1;
        let mut out = Vec::with_capacity(batch.len());
        for item in &batch.items {
            self.states[item.slot_idx] = mix(self.states[item.slot_idx], item.token);
            out.push(self.logits(item.token, self.states[item.slot_idx]));
        }
        Ok(out)
    }

    fn speculative_available(&self, slot: usize) -> bool {
        self.draft_enabled && slot < self.states.len()
    }

    fn speculative_begin(&mut self, slot: usize) -> Result<()> {
        self.checkpoints[slot] = Some(self.states[slot]);
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
        let mut state = self.states[slot];
        let mut draft = Vec::new();
        for index in 0..max_tokens {
            state = mix(state, token);
            token = next_token(token, state, self.vocab as u64);
            if self.force_reject_step == Some(self.step_count + index) {
                token = (token + 1) % self.vocab as u32;
            }
            draft.push(token);
        }
        Ok(draft)
    }

    fn speculative_verify(&mut self, slot: usize, inputs: &[u32], _pos: usize) -> Result<Vec<Vec<f32>>> {
        // step_count двигается как при K decode-шагах — для force_reject_step.
        self.step_count += inputs.len();
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
            .ok_or_else(|| anyhow::anyhow!("accept without verify"))?;
        if consumed < inputs.len() {
            let base = self.checkpoints[slot]
                .ok_or_else(|| anyhow::anyhow!("no checkpoint"))?;
            self.states[slot] = base;
            for &token in &inputs[..consumed] {
                self.states[slot] = mix(self.states[slot], token);
            }
        }
        Ok(())
    }

    fn speculative_commit(&mut self, slot: usize) -> Result<()> {
        self.checkpoints[slot] = None;
        Ok(())
    }

    fn speculative_rollback(&mut self, slot: usize) -> Result<()> {
        if let Some(state) = self.checkpoints[slot].take() {
            self.states[slot] = state;
        }
        self.verify_inputs[slot] = None;
        Ok(())
    }

    fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.states[slot] = 0;
        self.checkpoints[slot] = None;
        self.verify_inputs[slot] = None;
        Ok(())
    }
}

#[derive(Clone)]
struct DeterministicSampler {
    states: Vec<u64>,
}

impl DeterministicSampler {
    fn new(_slots: usize, seeds: &[u64]) -> Self {
        Self {
            states: seeds.to_vec(),
        }
    }
}

impl Sampler for DeterministicSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap()
            .0 as u32
    }

    fn checkpoint(&self, slot_idx: usize) -> SamplerCheckpoint {
        Box::new(self.states[slot_idx])
    }

    fn restore(&mut self, slot_idx: usize, checkpoint: SamplerCheckpoint) -> Result<()> {
        self.states[slot_idx] = *checkpoint
            .downcast::<u64>()
            .map_err(|_| anyhow::anyhow!("bad sampler checkpoint"))?;
        Ok(())
    }

    fn sample_indexed(&mut self, slot_idx: usize, _generated: &[u32], logits: &[f32]) -> u32 {
        self.states[slot_idx] = mix(self.states[slot_idx], slot_idx as u32 + 17);
        self.sample(logits)
    }
}

fn execute_suite(
    slots: usize,
    prompts: Vec<Vec<u32>>,
    max_new: usize,
    draft_enabled: bool,
    force_reject_step: Option<usize>,
) -> (Vec<Vec<u32>>, Vec<SpeculativeMetrics>, Vec<u64>) {
    let vocab = 1000;
    let seeds = (0..slots).map(|i| (i as u64 + 1) * 101).collect::<Vec<_>>();
    let mut model = ParityMockModel::new(slots, vocab, draft_enabled);
    model.force_reject_step = force_reject_step;
    let mut scheduler = BatchScheduler::new(model, slots, u32::MAX, vocab);
    scheduler.set_sampler(Box::new(DeterministicSampler::new(slots, &seeds)));
    for prompt in &prompts {
        scheduler.submit(prompt.clone(), max_new);
    }
    let mut outputs = vec![Vec::new(); prompts.len()];
    let mut metrics = vec![SpeculativeMetrics::default(); prompts.len()];
    let mut order = vec![None; slots];
    let mut next = 0usize;
    loop {
        for slot in scheduler.slots_mut().iter() {
            if slot.status != SlotStatus::Idle && order[slot.idx].is_none() {
                order[slot.idx] = Some(next);
                next += 1;
            }
        }
        let finished = scheduler
            .slots_mut()
            .iter()
            .filter(|slot| slot.status == SlotStatus::Finished)
            .map(|slot| slot.idx)
            .collect::<Vec<_>>();
        for slot in finished {
            let output_index = order[slot].unwrap();
            outputs[output_index] = scheduler.slots_mut()[slot].generated_tokens().to_vec();
            metrics[output_index] = scheduler.speculative_metrics(slot).unwrap().clone();
            scheduler.slots_mut()[slot].reset();
            scheduler.model_mut().reset_slot(slot).unwrap();
            order[slot] = None;
        }
        if scheduler.step().unwrap() == qwen35_batch::scheduler::StepOutcome::Idle
            && scheduler.slots_mut().iter().all(|slot| slot.status == SlotStatus::Idle)
        {
            break;
        }
    }
    let final_model_states = scheduler.model().states.clone();
    (outputs, metrics, final_model_states)
}

#[test]
fn fixed_seed_multilingual_prompts_match_token_for_token_b1_to_b4() {
    let mock_prompts = vec![
        vec![101, 102, 103, 104], // EN text
        vec![201, 202, 203],      // RU Cyrillic OCR
        vec![301, 302, 303, 304], // EN Image
        vec![401, 402, 403],      // RU Video
    ];
    for b in 1..=4 {
        let current_prompts = mock_prompts[..b].to_vec();
        let (baseline_out, _, _) = execute_suite(b, current_prompts.clone(), 16, false, None);
        let (mtp_out, mtp_metrics, _) = execute_suite(b, current_prompts.clone(), 16, true, None);
        assert_eq!(
            mtp_out, baseline_out,
            "B={b} outputs must match baseline bit-exact"
        );
        for (i, m) in mtp_metrics.iter().enumerate() {
            assert!(m.used, "slot {i} in B={b} must use MTP");
            assert!(m.accepted > 0, "slot {i} in B={b} must accept draft tokens");
        }
    }
}

#[test]
fn parity_holds_across_injected_rejections_and_fallbacks() {
    let prompts = vec![vec![11, 12, 13], vec![21, 22]];
    let (baseline_out, _, _) = execute_suite(2, prompts.clone(), 12, false, None);
    for step in [1, 3, 5, 8] {
        let (mtp_out, mtp_metrics, _) = execute_suite(2, prompts.clone(), 12, true, Some(step));
        assert_eq!(
            mtp_out, baseline_out,
            "rejection at step {step} must not diverge from baseline"
        );
        assert!(mtp_metrics.iter().any(|m| m.drafted > m.accepted));
    }
}
