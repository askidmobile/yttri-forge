//! `BatchModel`-адаптер над реальной Qwen3.5-4B (`ModelWeights`).
//!
//! ## Архитектура (Phases 3+4+5: true batched decode)
//! Prefill — per-slot через `forward()` (seq_len>1, single-slot state:
//! `metal_ctx`/`cuda_ctx` + `kv_cache`). После prefill snapshot state слота и
//! мигрируется в batched decode buffers (`seed_slot_batched`) — DeltaNet state
//! в slot-регион batched GPU-буфера, attention KV-cache в `kv_cache_batched[slot]`.
//!
//! Decode — true batched: один `forward_decode_batch([B,1], positions)` для
//! B слотов одновременно (batched projections + batched delta_rule kernel с осью
//! slot + per-slot KV/RoPE/SDPA). Per-slot state живёт в batched buffers —
//! никаких snapshot shuffle (в отличие от time-multiplexed). Parity bit-exact
//! (каждый слот изолирован своим slot-регионом state, math идентичен single).
//!
//! Fallback: если batched GPU-контекст отсутствует (batched decode disabled),
//! `seed_slot_batched`/`forward_decode_batch` недоступны — адаптер откатывается
//! на time-multiplexed path (restore→forward→snapshot per slot).

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use std::path::Path;

use crate::model::{BatchModel, DecodeBatch, MultimodalPrefill, PrefillChunk};
use crate::real::model_profile::ModelProfile;
use crate::real::model_weights::{
    BatchedStateCheckpoint, ModelWeights, StateSnapshot, DECODE_BATCH_CAPACITY,
};
#[cfg(feature = "cuda")]
use crate::real::model_weights::GRAPH_MIN_FREE_BYTES;
use crate::real::mtp::Qwen35Mtp;
use crate::real::multimodal::{GridThw, PositionPlan};
use crate::real::vision::Qwen35Vision;

/// Незавершённый batched verify: K inputs съедены multi-token forward'ом,
/// hidden-строки всех K позиций ждут `speculative_accept` (выравнивание state
/// + отбор строк 0..consumed для MTP commit).
struct PendingVerify {
    inputs: Vec<u32>,
    pos: usize,
    hidden: Tensor,
}

/// Адаптер реальной Qwen3.5-4B над `BatchModel` (true batched decode).
struct InstalledMultimodal {
    token_ids: Vec<u32>,
    grids: Vec<GridThw>,
    patches: Tensor,
    mm_token_types: Vec<u8>,
    plan: PositionPlan,
    features: Option<Tensor>,
}

fn slice_position_plan(plan: &PositionPlan, start: usize, len: usize) -> Result<PositionPlan> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| anyhow!("position-plan range overflow"))?;
    let slice = |axis: &[u32]| -> Result<Vec<u32>> {
        Ok(axis
            .get(start..end)
            .ok_or_else(|| anyhow!("position-plan range is out of bounds"))?
            .to_vec())
    };
    Ok(PositionPlan {
        text_positions: slice(&plan.text_positions)?,
        rope_positions: [
            slice(&plan.rope_positions[0])?,
            slice(&plan.rope_positions[1])?,
            slice(&plan.rope_positions[2])?,
        ],
        decode_rope_delta: plan.decode_rope_delta,
    })
}

/// CUDA-graph replay состояние для decode-шага.
#[cfg(feature = "cuda")]
struct DecodeGraphState {
    /// Raw handles: cudarc CudaGraph wrapper требует flags-enum без валидного 0.
    exec: cudarc::driver::sys::CUgraphExec,
    cu_graph: cudarc::driver::sys::CUgraph,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    b: usize,
    slots: Vec<u32>,
    /// Persistent входы: [B, 1] U32 токены (htod до launch).
    ids_t: Tensor,
    /// Alias захваченных выходов (graph-pool память).
    logits_t: Tensor,
}

#[cfg(feature = "cuda")]
impl DecodeGraphState {
    fn launch(&self) -> Result<()> {
        let res = unsafe {
            cudarc::driver::sys::cuGraphLaunch(self.exec, self.stream.cu_stream())
        };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphLaunch failed: {res:?}"));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for DecodeGraphState {
    fn drop(&mut self) {
        unsafe {
            cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            cudarc::driver::sys::cuGraphDestroy(self.cu_graph);
        }
    }
}

/// Адаптер реальной Qwen3.5-4B над `BatchModel` (true batched decode).
pub struct Qwen35BatchAdapter {
    model: ModelWeights,
    device: Device,
    /// Валидированный profile модели (Phase 1 preflight).
    profile: ModelProfile,
    /// Per-slot snapshot после prefill — источник state для seed в batched buffers.
    /// Хранится и для time-multiplexed fallback (если batched decode disabled).
    slot_snaps: Vec<Option<StateSnapshot>>,
    /// Признак того, что слот уже засеян в batched buffers (после prefill).
    /// True = batched decode может использовать этот slot без повторного seed.
    slot_seeded: Vec<bool>,
    /// CUDA-graph replay состояние (env QWEN36_CUDA_GRAPHS=1).
    #[cfg(feature = "cuda")]
    decode_graph: Option<DecodeGraphState>,
    /// Per-slot: KV данные изменились (prefill/seed) и paged pool устарел.
    #[cfg(feature = "cuda")]
    paged_dirty: Vec<bool>,
    #[cfg(feature = "cuda")]
    graphs_enabled: bool,
    /// On-demand Vision component. Phase 8 owns TTL/load barrier; adapter only
    /// consumes explicitly loaded component and per-request payloads.
    vision: Option<Qwen35Vision>,
    mtp: Option<Qwen35Mtp>,
    target_transactions: Vec<Option<BatchedStateCheckpoint>>,
    verified_target_hidden: Vec<Vec<Tensor>>,
    verify_pending: Vec<Option<PendingVerify>>,
    transaction_snapshot_positions: Vec<Option<usize>>,
    multimodal: Vec<Option<InstalledMultimodal>>,
    rope_deltas: Vec<i64>,
    eos: u32,
    vocab: usize,
}

impl Qwen35BatchAdapter {
    /// Загрузить модель из GGUF (zero-copy на Metal) и подготовить N слотов.
    pub fn load(gguf_path: &Path, device: Device, num_slots: usize) -> Result<Self> {
        if num_slots > DECODE_BATCH_CAPACITY as usize {
            return Err(anyhow!(
                "num_slots {num_slots} exceeds decode capacity {DECODE_BATCH_CAPACITY}"
            ));
        }
        // Batched state ровно под num_slots (DeltaNet ~8MB/слой/слот на 27B).
        crate::real::model_weights::set_decode_capacity(num_slots as u32);
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda_dev) = &device {
            static RETAIN_ONCE: std::sync::Once = std::sync::Once::new();
            let disabled = std::env::var("QWEN36_NO_MEMPOOL_RETAIN").as_deref() == Ok("1");
            if !disabled {
                RETAIN_ONCE.call_once(|| {
                    // Ограниченный release threshold вместо u64::MAX:
                    // retain=MAX держит пиковые префилл-транзиенты (до 1 GiB)
                    // навсегда → VRAM 97%+ → WDDM spill → decode коллапс.
                    // 512 MiBthreshold: decode-аллокации переиспользуют пул,
                    // префилл-пики освобождаются на sync-точках.
                    let thresh_mib = std::env::var("QWEN36_MEMPOOL_THRESHOLD_MIB")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(256); // 256 MiB — баланс: decode reuse работает, префилл-пики >256 освобождаются
                    if let Err(e) = candle_core::cuda_backend::mem_pool::set_release_threshold_mib(cuda_dev, thresh_mib) {
                        log::warn!("[qwen35-batch] set_release_threshold_mib({thresh_mib}) failed: {e}");
                    }
                });
            }
        }
        use candle_core::quantized::gguf_file;
        use std::fs::File;
        use std::sync::Arc;

        let file = File::open(gguf_path).map_err(|e| anyhow!("open GGUF: {e}"))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| anyhow!("mmap GGUF: {e}"))?;
        let mmap = Arc::new(mmap);

        // Один проход чтения GGUF: EOS + vocab (из token_embd.weight shape[0]) + веса.
        let mut c = std::io::Cursor::new(mmap.as_ref());
        let ct = gguf_file::Content::read(&mut c).map_err(|e| anyhow!("read GGUF: {e}"))?;

        // Phase 1 preflight: validate architecture, metadata, and tensor contracts
        // BEFORE heavy tensor loading. Fail-fast with aggregated errors.
        let file_size = mmap.len() as u64;
        let profile = ModelProfile::read_and_validate(&ct, file_size)
            .map_err(|e| anyhow!("GGUF validation failed: {e}"))?;
        log::info!(
            "[qwen35-batch] profile: arch={:?} blocks={} hidden={} ctx={} quant_count={} fingerprint={}",
            profile.architecture,
            profile.block_count,
            profile.hidden_size,
            profile.context_length,
            profile.quant_set.len(),
            profile.fingerprint.hash,
        );

        let eos = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(151645);

        let vocab = ct
            .tensor_infos
            .get("token_embd.weight")
            .and_then(|info| info.shape.dims().first().copied())
            .unwrap_or(0);
        log::info!(
            "[qwen35-batch] GGUF: eos={eos}, vocab(shape)={vocab}, mmap={:.1} MB",
            mmap.len() as f64 / 1024.0 / 1024.0
        );

        // Загрузка весов (zero-copy на Metal, обычный путь на CPU).
        #[cfg(target_os = "macos")]
        let model = if matches!(device, Device::Metal(_)) {
            ModelWeights::from_gguf_zero_copy(ct, mmap, &device)
                .map_err(|e| anyhow!("load weights zero-copy: {e}"))?
        } else {
            ModelWeights::from_gguf(ct, mmap, &device).map_err(|e| anyhow!("load weights: {e}"))?
        };
        #[cfg(not(target_os = "macos"))]
        let model =
            ModelWeights::from_gguf(ct, mmap, &device).map_err(|e| anyhow!("load weights: {e}"))?;

        // После загрузки весов пул держит reserved-страницы от upload staging.
        // Trim → commit возвращается к фактическим весам (важно для 27B на 12GB).
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }

        #[cfg(feature = "cuda")]
        let graphs_on = {
            let want = std::env::var("QWEN36_CUDA_GRAPHS").as_deref() == Ok("1");
            if !want {
                false
            } else if let Device::Cuda(c) = &device {
                // Графы требуют +VRAM (graph pool + CUDA embedding). При нехватке
                // запаса после загрузки — отключаем: на 27B (11.0 GiB весов на
                // 12GB) лишние ~0.5 GiB уходят в sysmem и душат decode сильнее,
                // чем экономия на launch overhead. Порог общий с гейтом
                // эмбеддинга — см. GRAPH_MIN_FREE_BYTES в model_weights.rs.
                let free = c.cuda_stream().context().mem_get_info().map(|(f, _)| f).unwrap_or(0);
                if free < GRAPH_MIN_FREE_BYTES {
                    log::info!(
                        "[graphs] disabled: VRAM headroom {:.0} MiB < {} MiB",
                        free as f64 / 1048576.0,
                        GRAPH_MIN_FREE_BYTES / 1048576
                    );
                    false
                } else {
                    true
                }
            } else {
                true
            }
        };

        let mut a = Self {
            model,
            device,
            profile,
            slot_snaps: (0..num_slots).map(|_| None).collect(),
            slot_seeded: vec![false; num_slots],
            #[cfg(feature = "cuda")]
            decode_graph: None,
            #[cfg(feature = "cuda")]
            paged_dirty: vec![true; num_slots],
            #[cfg(feature = "cuda")]
            graphs_enabled: graphs_on,
            vision: None,
            mtp: None,
            target_transactions: (0..num_slots).map(|_| None).collect(),
            verified_target_hidden: (0..num_slots).map(|_| Vec::new()).collect(),
            verify_pending: (0..num_slots).map(|_| None).collect(),
            transaction_snapshot_positions: vec![None; num_slots],
            multimodal: (0..num_slots).map(|_| None).collect(),
            rope_deltas: vec![0; num_slots],
            eos,
            vocab,
        };
        let mut a = a; // keep mut for prepare below
        // Упреждающее f16-зеркало слота 0 (фикс WDDM-коллапса 2026-08-23):
        // выделяем сразу после загрузки весов, пока dedicated VRAM свободна,
        // иначе страницы уходят в WDDM shared и декод падает на PCIe.
        // yttri-forge stage1: F16-сайдкар тяжёлых проекций (dual-read prefill)
        a.model
            .attach_ytf16(gguf_path, &a.device)
            .map_err(|e| anyhow!("attach_ytf16: {e}"))?;

        if let Some(tokens) = std::env::var("QWEN36_KV_MIRROR_PREPARE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            if tokens > 0 {
                a.model.prepare_kv_mirror(tokens.min(a.model.context_length));
            }
        }
        Ok(a)
    }

    /// Загрузить с явным vocab (из GGUF metadata `tokenizer.ggml.tokens` len).
    pub fn load_with_vocab(
        gguf_path: &Path,
        device: Device,
        num_slots: usize,
        vocab: usize,
    ) -> Result<Self> {
        let mut a = Self::load(gguf_path, device, num_slots)?;
        a.vocab = vocab;
        Ok(a)
    }

    /// Prefix-cache инфраструктура: snapshot state последнего prefill'а слота.
    /// Сервер забирает его в LRU-кэш; при повторном prompt'е — inject + primed admit.
    pub fn slot_snapshot(&self, slot: usize) -> Option<StateSnapshot> {
        self.slot_snaps[slot].clone()
    }

    /// Внедрить snapshot слоту (prefix-cache hit): при следующем prefill_chunk
    /// с reset_first=false модель восстановит state из этого snapshot'а.
    pub fn inject_slot_snapshot(&mut self, slot: usize, snap: StateSnapshot) {
        self.slot_snaps[slot] = Some(snap);
        self.slot_seeded[slot] = false;
    }

    /// Делегированный доступ к модели (для profiling / debug_capture).
    pub fn model(&self) -> &ModelWeights {
        &self.model
    }

    /// Доступ к валидированному profile модели (Phase 1).
    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn load_vision(&mut self, gguf_path: &Path) -> Result<()> {
        self.vision = Some(
            Qwen35Vision::load(gguf_path, self.device.clone())
                .map_err(|error| anyhow!("load Vision component: {error}"))?,
        );
        Ok(())
    }

    /// BF16 correctness oracle only; production component loading stays Q8-strict.
    pub fn load_vision_reference(&mut self, gguf_path: &Path) -> Result<()> {
        self.vision = Some(
            Qwen35Vision::load_reference(gguf_path, self.device.clone())
                .map_err(|error| anyhow!("load reference Vision component: {error}"))?,
        );
        Ok(())
    }

    pub fn unload_vision(&mut self) {
        self.vision = None;
        for payload in &mut self.multimodal {
            *payload = None;
        }
    }

    pub fn load_mtp(&mut self, gguf_path: &Path) -> Result<()> {
        if self.target_transactions.iter().any(Option::is_some) {
            return Err(anyhow!("cannot load MTP during active transaction"));
        }
        self.mtp = Some(
            Qwen35Mtp::load(
                gguf_path,
                self.device.clone(),
                self.slot_snaps.len(),
                &self.profile,
                self.model.shared_output(),
            )
            .map_err(|error| anyhow!("load MTP component: {error}"))?,
        );
        Ok(())
    }

    pub fn unload_mtp(&mut self) -> Result<()> {
        if self.target_transactions.iter().any(Option::is_some) {
            return Err(anyhow!("cannot unload MTP during active transaction"));
        }
        self.mtp = None;
        Ok(())
    }

    /// Полный сброс (все слоты) — только для тестов / teardown.
    #[allow(dead_code)]
    fn clear_all_state(&mut self) {
        self.model.clear_state();
        self.model.clear_state_batched(&self.device);
        for s in self.slot_snaps.iter_mut() {
            *s = None;
        }
        for f in self.slot_seeded.iter_mut() {
            *f = false;
        }
        for delta in &mut self.rope_deltas {
            *delta = 0;
        }
        for payload in &mut self.multimodal {
            *payload = None;
        }
        self.mtp = None;
        for transaction in &mut self.target_transactions {
            *transaction = None;
        }
        for hidden in &mut self.verified_target_hidden {
            hidden.clear();
        }
        for pending in &mut self.verify_pending {
            *pending = None;
        }
        for position in &mut self.transaction_snapshot_positions {
            *position = None;
        }
    }

    /// RoPE-позиции decode/verify: cache_pos + per-slot multimodal rope delta.
    fn rope_positions_for(&self, slot: usize, start: usize, len: usize) -> Result<Vec<usize>> {
        (start..start + len)
            .map(|cache_position| {
                let rope_position = i64::try_from(cache_position)?
                    .checked_add(self.rope_deltas[slot])
                    .ok_or_else(|| anyhow!("decode RoPE position overflow"))?;
                usize::try_from(rope_position)
                    .map_err(|_| anyhow!("negative decode RoPE position"))
            })
            .collect()
    }
}

impl BatchModel for Qwen35BatchAdapter {
    fn vocab_size(&self) -> usize {
        if self.vocab != 0 {
            self.vocab
        } else {
            151943
        }
    }

    fn install_multimodal(&mut self, slot: usize, payload: MultimodalPrefill) -> Result<()> {
        if slot >= self.multimodal.len() {
            return Err(anyhow!("multimodal slot {slot} is out of range"));
        }
        if self.vision.is_none() {
            return Err(anyhow!("Vision component is not loaded"));
        }
        if payload.token_ids.len() != payload.mm_token_types.len()
            || payload
                .rope_positions
                .iter()
                .any(|axis| axis.len() != payload.token_ids.len())
            || payload.mm_token_types.iter().all(|kind| *kind == 0)
            || payload.mm_token_types.iter().any(|kind| *kind > 2)
        {
            return Err(anyhow!("multimodal token/position lengths differ or contain no media"));
        }
        if payload.patch_values.len()
            != payload
                .patch_rows
                .checked_mul(payload.patch_width)
                .ok_or_else(|| anyhow!("multimodal patch size overflow"))?
        {
            return Err(anyhow!("multimodal patch value count mismatch"));
        }
        let expected_features = payload
            .mm_token_types
            .iter()
            .filter(|kind| **kind != 0)
            .count();
        let grids: Vec<_> = payload
            .media_grids
            .into_iter()
            .map(|[t, h, w]| GridThw { t, h, w })
            .collect();
        let actual_features = grids.iter().try_fold(0usize, |total, grid| {
            if !grid.h.is_multiple_of(2) || !grid.w.is_multiple_of(2) {
                return Err(anyhow!("multimodal grid is not divisible by merge size"));
            }
            let count = grid
                .t
                .checked_mul(grid.h / 2)
                .and_then(|value| value.checked_mul(grid.w / 2))
                .ok_or_else(|| anyhow!("multimodal grid overflow"))?;
            total
                .checked_add(count)
                .ok_or_else(|| anyhow!("multimodal feature count overflow"))
        })?;
        if actual_features != expected_features {
            return Err(anyhow!(
                "multimodal feature/grid count {actual_features} != placeholder count {expected_features}"
            ));
        }
        let patches = Tensor::from_vec(
            payload.patch_values,
            (payload.patch_rows, payload.patch_width),
            &self.device,
        )?;
        let plan = PositionPlan {
            text_positions: (0..payload.token_ids.len())
                .map(u32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            rope_positions: payload.rope_positions,
            decode_rope_delta: payload.decode_rope_delta,
        };
        self.multimodal[slot] = Some(InstalledMultimodal {
            token_ids: payload.token_ids,
            grids,
            patches,
            mm_token_types: payload.mm_token_types,
            plan,
            features: None,
        });
        self.rope_deltas[slot] = payload.decode_rope_delta;
        Ok(())
    }

    fn prefill_chunk(&mut self, chunk: &PrefillChunk) -> Result<Vec<f32>> {
        let sidx = chunk.slot_idx;
        let pf_t0 = std::time::Instant::now();
        let mut pf_restore_ms = 0f64;
        if sidx >= self.slot_snaps.len() || chunk.tokens.is_empty() {
            return Err(anyhow!("prefill slot is out of range or chunk is empty"));
        }
        if chunk.reset_first {
            // Keep installed media for first chunk; reset only model state.
            self.model.clear_state();
            self.slot_snaps[sidx] = None;
            self.slot_seeded[sidx] = false;
            #[cfg(feature = "cuda")]
            {
                self.paged_dirty[sidx] = true;
            }
            if self.multimodal[sidx].is_none() {
                self.rope_deltas[sidx] = 0;
            }
        } else if let Some(snap) = self.slot_snaps[sidx].as_ref() {
            // Продолжение prefill после чанка: восстанавливаем single-slot state.
            self.model
                .restore_state(&self.device, snap)
                .map_err(|e| anyhow!("prefill restore: {e}"))?;
        } else {
            return Err(anyhow!(
                "prefill_chunk: slot {sidx} без snapshot и не reset"
            ));
        }

        let pf_restore = pf_t0.elapsed();
        let ids = Tensor::from_vec(
            chunk.tokens.clone(),
            (1usize, chunk.tokens.len()),
            &self.device,
        )?;
        let pf_fwd0 = std::time::Instant::now();
        let (logits, mtp_inputs) = if let Some(media) = self.multimodal[sidx].as_mut() {
            let end = chunk
                .start_pos
                .checked_add(chunk.tokens.len())
                .ok_or_else(|| anyhow!("prefill range overflow"))?;
            if media.token_ids.get(chunk.start_pos..end) != Some(chunk.tokens.as_slice()) {
                return Err(anyhow!(
                    "multimodal prefill tokens differ from installed prompt"
                ));
            }
            if media.features.is_none() {
                media.features = Some(
                    self.vision
                        .as_ref()
                        .ok_or_else(|| anyhow!("Vision component is not loaded"))?
                        .forward(&media.patches, &media.grids)
                        .map_err(|error| anyhow!("Vision forward: {error}"))?,
                );
            }
            let embeds = self.model.embed_tokens(&ids, &self.device)?;
            let mut feature_offset = media.mm_token_types[..chunk.start_pos]
                .iter()
                .filter(|kind| **kind != 0)
                .count();
            let features = media
                .features
                .as_ref()
                .ok_or_else(|| anyhow!("Vision features were not produced"))?;
            let mut cursor = 0usize;
            while cursor < chunk.tokens.len() {
                if media.mm_token_types[chunk.start_pos + cursor] == 0 {
                    cursor += 1;
                    continue;
                }
                let start = cursor;
                while cursor < chunk.tokens.len()
                    && media.mm_token_types[chunk.start_pos + cursor] != 0
                {
                    cursor += 1;
                }
                let len = cursor - start;
                embeds.slice_set(
                    &features.narrow(0, feature_offset, len)?.unsqueeze(0)?,
                    1,
                    start,
                )?;
                feature_offset += len;
            }
            let plan = slice_position_plan(&media.plan, chunk.start_pos, chunk.tokens.len())?;
            let (logits, hidden) = self
                .model
                .forward_embeds_mrope_with_hidden(&embeds, &plan, chunk.start_pos)
                .map_err(|error| anyhow!("multimodal prefill forward: {error}"))?;
            (logits, Some((embeds, hidden)))
        } else if self.mtp.is_some() {
            let embeds = self.model.embed_tokens(&ids, &self.device)?;
            let (logits, hidden) = self
                .model
                .forward_embeds_with_hidden(&embeds, chunk.start_pos)
                .map_err(|error| anyhow!("prefill forward: {error}"))?;
            (logits, Some((embeds, hidden)))
        } else {
            (
                self.model
                    .forward(&ids, chunk.start_pos)
                    .map_err(|error| anyhow!("prefill forward: {error}"))?,
                None,
            )
        };
        if let (Some(mtp), Some((embeds, hidden))) = (self.mtp.as_mut(), mtp_inputs) {
            let rope_positions = self.multimodal[sidx]
                .as_ref()
                .map(|media| slice_position_plan(&media.plan, chunk.start_pos, chunk.tokens.len()))
                .transpose()?
                .map(|plan| plan.rope_positions);
            mtp.catch_up(
                sidx,
                &embeds,
                &hidden,
                chunk.start_pos,
                rope_positions.as_ref(),
            )
            .map_err(|error| anyhow!("MTP prefill catch-up: {error}"))?;
        }
        let pf_fwd = pf_fwd0.elapsed();
        let pf_l0 = std::time::Instant::now();
        let logits_f32 = logits
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()
            .map_err(|e| anyhow!("prefill logits to_vec1: {e}"))?;
        let pf_logits = pf_l0.elapsed();

        // Сохранить snapshot state слота на позиции start_pos + tokens.len().
        let new_pos = chunk.start_pos + chunk.tokens.len();
        let pf_s0 = std::time::Instant::now();
        let snap = self
            .model
            .snapshot_state(&self.device, new_pos)
            .map_err(|e| anyhow!("prefill snapshot: {e}"))?;
        self.slot_snaps[sidx] = Some(snap);
        // Seed only final prompt boundary; intermediate chunks restore through
        // single-slot snapshot on next prefill call.
        if chunk.is_final {
            self.model
                .seed_slot_batched(
                    &self.device,
                    sidx,
                    self.slot_snaps[sidx]
                        .as_ref()
                        .ok_or_else(|| anyhow!("prefill snapshot disappeared"))?,
                )
                .map_err(|error| anyhow!("prefill seed slot {sidx}: {error}"))?;
            self.slot_seeded[sidx] = true;
            // Single-slot F16 KV больше не нужен (decode через batched q8):
            // освобождаем ~480 MiB @24K, иначе карта уходит в 97%+ и декод
            // падает в WDDM shared (обрыв 6K→12K, 2026-08-23).
            if std::env::var("QWEN36_KEEP_SINGLE_KV").as_deref() != Ok("1") {
                self.model.clear_single_slot_kv();
            }
            // Trim CUDA memory pool: prefill оставил пиковые F16/KV транзиенты
            // в driver pool (release threshold=512 MiB). Trim возвращает ОС
            // страницы сверх текущего usage → VRAM освобождается для decode.
            #[cfg(feature = "cuda")]
            if let Device::Cuda(c) = &self.device {
                let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
            }
        } else {
            self.slot_seeded[sidx] = false;
        }

        if crate::scheduler::trace_on() {
            eprintln!(
                "[pfa] chunk tok={} restore={:.1}ms fwd={}ms logits_d2h={:.1}ms snap+seed={:.1}ms total={:.1}ms",
                chunk.tokens.len(),
                if chunk.reset_first { 0.0 } else { pf_restore.as_secs_f64() * 1e3 },
                pf_fwd.as_secs_f64() * 1e3,
                pf_logits.as_secs_f64() * 1e3,
                pf_s0.elapsed().as_secs_f64() * 1e3,
                pf_t0.elapsed().as_secs_f64() * 1e3,
            );
        }
        let _ = pf_restore_ms;
        Ok(logits_f32)
    }

    /// CUDA-graph decode: один cuGraphLaunch вместо ~1700 драйверных вызовов.
    /// Возвращает Ok(None) — если шаг не графable (окно, состав, MTP) → eager fallback.
    fn decode_batch(&mut self, batch: &DecodeBatch) -> Result<Vec<Vec<f32>>> {
        let b = batch.items.len();
        if batch.items.iter().any(|item| item.slot_idx >= self.slot_snaps.len()) {
            return Err(anyhow!("decode slot is out of range"));
        }
        if b == 0 {
            return Ok(vec![]);
        }

        // Сначала seed всех слотов в batched buffers (если ещё не засеяны).
        for it in &batch.items {
            let sidx = it.slot_idx;
            if !self.slot_seeded[sidx] {
                let snap = self.slot_snaps[sidx]
                    .as_ref()
                    .ok_or_else(|| anyhow!("decode_batch: slot {sidx} без snapshot"))?;
                self.model
                    .seed_slot_batched(&self.device, sidx, snap)
                    .map_err(|e| anyhow!("decode seed slot {sidx}: {e}"))?;
                self.slot_seeded[sidx] = true;
                #[cfg(feature = "cuda")]
                {
                    self.paged_dirty[sidx] = true;
                    // Seed меняет состав state — graph состав-зависим, инвалидируем.
                    if self.decode_graph.is_some() {
                        self.decode_graph = None;
                    }
                    // Prefill оставил в пуле пиковые страницы интермедиатов
                    // (512-token chunk buffers). Trim при первом decode после
                    // prefill — иначе retained slack добивает VRAM на 27B.
                    if let Device::Cuda(c) = &self.device {
                        let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
                    }
                }
            }
        }

        // Собираем batched входы: tokens [B,1] + positions [B] + slots [B].
        // `slots` — отображение batch_idx → slot_idx: persistent state (ssm_state,
        // conv_state, kv_cache) адресуется по slot_idx, а не по batch_idx.
        // После сжатия батча (ранний EOS) это сохраняет привязку слота к его state.
        let mut tokens = Vec::with_capacity(b);
        let mut positions = Vec::with_capacity(b);
        let mut rope_positions = Vec::with_capacity(b);
        let mut slot_order = Vec::with_capacity(b); // (batch_idx, slot_idx)
        let mut slots = Vec::with_capacity(b);
        for it in &batch.items {
            tokens.push(it.token);
            positions.push(it.pos);
            let rope_position = i64::try_from(it.pos)?
                .checked_add(self.rope_deltas[it.slot_idx])
                .ok_or_else(|| anyhow!("decode RoPE position overflow"))?;
            rope_positions.push(
                usize::try_from(rope_position)
                    .map_err(|_| anyhow!("negative decode RoPE position"))?,
            );
            slot_order.push(it.slot_idx);
            slots.push(it.slot_idx as u32);
        }
        let ids = Tensor::from_vec(tokens.clone(), (b, 1usize), &self.device)?;

        // CUDA-graph decode: один cuGraphLaunch на весь forward. None → eager.
        #[cfg(feature = "cuda")]
        {
            match self.decode_batch_graphed(b, &tokens, &rope_positions, &slots, &positions) {
                Ok(Some(out)) => {
                    for i in 0..b {
                        let sidx = slot_order[i];
                        if let Some(snap) = self.slot_snaps[sidx].as_mut() {
                            snap.position = positions[i] + 1;
                        }
                    }
                    return Ok(out);
                }
                Ok(None) => {}
                Err(e) => {
                    eprintln!("[graphs] graphed decode failed, eager fallback: {e}");
                    self.decode_graph = None;
                    self.graphs_enabled = false;
                }
            }
        }
        // Hidden decode_batch'а больше не нужен MTP: verify идёт через
        // speculative_verify (multi-token), hidden собирает speculative_accept.
        let logits = self
            .model
            .forward_decode_batch(&ids, &positions, &rope_positions, &slots)
            .map_err(|e| anyhow!("decode_batch forward: {e}"))?;
        // One D2H transfer for [B, vocab], then split on host. Per-row to_vec1()
        // serialized four CUDA synchronizations/copies for B=4.
        let flat = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()
            .map_err(|e| anyhow!("decode_batch logits to_vec1: {e}"))?;
        let vocab = self.vocab_size();
        if flat.len() != b * vocab {
            return Err(anyhow!(
                "decode_batch logits length {} != batch {b} * vocab {vocab}",
                flat.len()
            ));
        }
        let out = flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect();

        // Обновить per-slot snapshot из batched state после decode (для последующего
        // prefill-продолжения и для повторного seed, если слот покинет batch и вернётся).
        // Дешевле: позиция продвинулась на 1; state живёт в batched buffers, но snapshot
        // нужен для prefill-restore (который использует single-slot path).
        // Полный re-snapshot из batched buffers не реализован (требует dtoh slot-region);
        // вместо этого помечаем slot как требующий re-seed перед следующим decode —
        // batched state уже актуален в буферах, snapshot устарел только по позиции.
        for i in 0..b {
            let sidx = slot_order[i];
            // Позиция в snapshot продвинулась; сам state валиден в batched buffers.
            // Обновляем только position (для future prefill-restore корректен только
            // если последующий prefill стартует с it.pos+1 и reset — обычный паттерн).
            if let Some(snap) = self.slot_snaps[sidx].as_mut() {
                snap.position = positions[i] + 1;
            }
            // slot остаётся seeded — batched buffers уже содержат обновлённый state.
        }

        Ok(out)
    }

    fn speculative_available(&self, slot: usize) -> bool {
        slot < self.slot_seeded.len()
            && self.mtp.is_some()
            && self.slot_seeded[slot]
            && self.target_transactions[slot].is_none()
    }

    fn speculative_begin(&mut self, slot: usize) -> Result<()> {
        if !self.speculative_available(slot) {
            return Err(anyhow!("MTP is unavailable for slot {slot}"));
        }
        let checkpoint = self
            .model
            .checkpoint_slot_batched(&self.device, slot)
            .map_err(|error| anyhow!("target checkpoint: {error}"))?;
        let mtp = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow!("MTP component is not loaded"))?;
        if let Err(error) = mtp.begin(slot) {
            self.model
                .restore_slot_batched(&self.device, &checkpoint)
                .map_err(|restore| anyhow!("MTP begin failed: {error}; target restore: {restore}"))?;
            return Err(anyhow!("MTP begin: {error}"));
        }
        self.transaction_snapshot_positions[slot] =
            self.slot_snaps[slot].as_ref().map(|snapshot| snapshot.position);
        self.target_transactions[slot] = Some(checkpoint);
        self.verified_target_hidden[slot].clear();
        Ok(())
    }

    fn speculative_draft(
        &mut self,
        slot: usize,
        token: u32,
        cache_pos: usize,
        rope_pos: usize,
        max_tokens: usize,
    ) -> Result<Vec<u32>> {
        if cache_pos != rope_pos && self.rope_deltas[slot] == 0 {
            return Err(anyhow!("MTP cache/RoPE position mismatch"));
        }
        self.mtp
            .as_mut()
            .ok_or_else(|| anyhow!("MTP component is not loaded"))?
            .draft(slot, token, rope_pos, max_tokens, &self.model)
            .map_err(|error| anyhow!("MTP draft: {error}"))
    }

    /// Один multi-token target-forward на K позиций одного слота. Батчевая ось
    /// переиспользуется как ось позиций: slots=[slot;K], positions=pos..pos+K.
    /// Attention batched decode обрабатывает строки по порядку (append по
    /// cache_len), DeltaNet сериализует одинаковые слоты — те же decode-ядра,
    /// что и K одиночных шагов, бит-эксактно.
    fn speculative_verify(
        &mut self,
        slot: usize,
        inputs: &[u32],
        pos: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if self.target_transactions[slot].is_none() {
            return Err(anyhow!("MTP transaction is not active for slot {slot}"));
        }
        if inputs.is_empty() {
            return Err(anyhow!("speculative verify requires inputs"));
        }
        let k = inputs.len();
        let cache_positions: Vec<usize> = (pos..pos + k).collect();
        let rope_positions = self.rope_positions_for(slot, pos, k)?;
        let ids = Tensor::from_vec(inputs.to_vec(), (k, 1usize), &self.device)?;
        let slots = vec![slot as u32; k];
        let (logits, hidden) = self
            .model
            .forward_decode_batch_with_hidden(&ids, &cache_positions, &rope_positions, &slots)
            .map_err(|error| anyhow!("speculative verify forward: {error}"))?;
        self.verify_pending[slot] = Some(PendingVerify {
            inputs: inputs.to_vec(),
            pos,
            hidden,
        });
        let flat = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(|error| anyhow!("speculative verify logits: {error}"))?;
        let vocab = self.vocab_size();
        if flat.len() != k * vocab {
            return Err(anyhow!(
                "speculative verify logits length {} != {k} * vocab {vocab}",
                flat.len()
            ));
        }
        Ok(flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect())
    }

    /// Выровнять state после verify: при consumed < K DeltaNet-state ушёл вперёд
    /// по отвергнутым inputs → откат к checkpoint'у транзакции + re-run принятого
    /// префикса одним чанком (логиты не читаем — без D2H). Hidden-строки
    /// 0..consumed уходят в verified_target_hidden для MTP commit.
    fn speculative_accept(&mut self, slot: usize, consumed: usize) -> Result<()> {
        let pending = self.verify_pending[slot]
            .take()
            .ok_or_else(|| anyhow!("speculative accept without verify for slot {slot}"))?;
        let k = pending.inputs.len();
        if consumed > k {
            return Err(anyhow!("speculative accept {consumed} exceeds verified {k}"));
        }
        if consumed < k {
            let checkpoint = self.target_transactions[slot]
                .as_ref()
                .ok_or_else(|| anyhow!("MTP transaction is not active for slot {slot}"))?;
            self.model
                .restore_slot_batched(&self.device, checkpoint)
                .map_err(|error| anyhow!("speculative accept restore: {error}"))?;
            if consumed > 0 {
                let cache_positions: Vec<usize> =
                    (pending.pos..pending.pos + consumed).collect();
                let rope_positions = self.rope_positions_for(slot, pending.pos, consumed)?;
                let ids = Tensor::from_vec(
                    pending.inputs[..consumed].to_vec(),
                    (consumed, 1usize),
                    &self.device,
                )?;
                let slots = vec![slot as u32; consumed];
                let _ = self
                    .model
                    .forward_decode_batch(&ids, &cache_positions, &rope_positions, &slots)
                    .map_err(|error| anyhow!("speculative accept re-run: {error}"))?;
            }
        }
        for row in 0..consumed {
            self.verified_target_hidden[slot].push(pending.hidden.i(row)?.unsqueeze(0)?);
        }
        if let Some(snap) = self.slot_snaps[slot].as_mut() {
            snap.position = pending.pos + consumed;
        }
        Ok(())
    }

    fn speculative_commit(&mut self, slot: usize) -> Result<()> {
        if self.target_transactions[slot].is_none() {
            return Err(anyhow!("MTP transaction is not active for slot {slot}"));
        }
        self.mtp
            .as_mut()
            .ok_or_else(|| anyhow!("MTP component is not loaded"))?
            .commit(slot, &self.verified_target_hidden[slot])
            .map_err(|error| anyhow!("MTP commit: {error}"))?;
        self.target_transactions[slot] = None;
        self.verified_target_hidden[slot].clear();
        self.transaction_snapshot_positions[slot] = None;
        Ok(())
    }

    fn speculative_rollback(&mut self, slot: usize) -> Result<()> {
        if slot >= self.target_transactions.len() {
            return Err(anyhow!("MTP rollback slot {slot} is out of range"));
        }
        if let Some(checkpoint) = self.target_transactions[slot].take() {
            self.model
                .restore_slot_batched(&self.device, &checkpoint)
                .map_err(|error| anyhow!("target rollback: {error}"))?;
            if let Some(position) = self.transaction_snapshot_positions[slot] {
                if let Some(snapshot) = self.slot_snaps[slot].as_mut() {
                    snapshot.position = position;
                }
            }
        }
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.rollback(slot)
                .map_err(|error| anyhow!("MTP rollback: {error}"))?;
        }
        self.verified_target_hidden[slot].clear();
        self.verify_pending[slot] = None;
        self.transaction_snapshot_positions[slot] = None;
        Ok(())
    }

    fn reset_slot(&mut self, idx: usize) -> Result<()> {
        if idx >= self.slot_snaps.len() {
            return Err(anyhow!("reset slot {idx} is out of range"));
        }
        self.slot_snaps[idx] = None;
        self.slot_seeded[idx] = false;
        self.multimodal[idx] = None;
        self.rope_deltas[idx] = 0;
        self.target_transactions[idx] = None;
        self.verified_target_hidden[idx].clear();
        self.verify_pending[idx] = None;
        self.transaction_snapshot_positions[idx] = None;
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset_slot(idx)
                .map_err(|error| anyhow!("reset MTP slot {idx}: {error}"))?;
        }
        Ok(())
    }
}

impl Qwen35BatchAdapter {
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_batch_graphed(
        &mut self,
        b: usize,
        tokens: &[u32],
        rope_positions: &[usize],
        slots: &[u32],
        positions: &[usize],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        if !self.graphs_enabled {
            return Ok(None);
        }
        let Device::Cuda(cuda_dev) = &self.device else {
            return Ok(None);
        };
        // MTP transactions читают hidden — graph-pool буфер перезаписывается
        // следующим launch, поэтому MTP-слоты → eager.
        if self.target_transactions.iter().any(|t| t.is_some()) {
            return Ok(None);
        }
        self.model
            .init_paged_decode(&self.device)
            .map_err(|e| anyhow!("init paged decode: {e}"))?;
        let window = self.model.paged_window();
        if window == 0 {
            return Ok(None);
        }
        let num_slots = self.slot_snaps.len();

        // Миграция KV из per-slot кэша в paged pool для dirty слотов.
        for &s in slots {
            let sidx = s as usize;
            if sidx >= num_slots {
                return Ok(None);
            }
            if self.paged_dirty[sidx] {
                let len = self
                    .model
                    .migrate_kv_to_paged(sidx)
                    .map_err(|e| anyhow!("migrate KV slot {sidx}: {e}"))?;
                if len > window {
                    return Ok(None);
                }
                self.paged_dirty[sidx] = false;
                // Синхронизируем device kv_len с мигрированным содержимым.
                let lens: Vec<u32> = {
                    let ctx = self.model.paged_ctx.as_ref().unwrap();
                    let mut v = ctx.kv_len_host.clone();
                    v[sidx] = len as u32;
                    v
                };
                self.model
                    .paged_ctx
                    .as_mut()
                    .unwrap()
                    .reset_kv_len(&lens)?;
            }
        }

        // Окно и позиции: позиция должна совпадать с device kv_len (host mirror).
        {
            let ctx = self.model.paged_ctx.as_ref().unwrap();
            for (i, &s) in slots.iter().enumerate() {
                let cur = ctx.kv_len_host[s as usize] as usize;
                if positions[i] != cur || cur + 1 > window {
                    return Ok(None);
                }
            }
        }

        // Block table: slot s владеет страницами [s*mb .. (s+1)*mb).
        let (max_blocks, block_table) = {
            let ctx = self.model.paged_ctx.as_ref().unwrap();
            let mb = ctx.max_blocks;
            let mut bt = vec![0u32; b * mb];
            for (bidx, &s) in slots.iter().enumerate() {
                for j in 0..mb {
                    bt[bidx * mb + j] = s * mb as u32 + j as u32;
                }
            }
            (mb, bt)
        };
        let _ = max_blocks;

        let need_capture = match &self.decode_graph {
            Some(g) => {
                let why = if g.b != b {
                    format!("b {}!={}", g.b, b)
                } else if g.slots != slots {
                    format!("slots {:?}!={:?}", g.slots, slots)
                } else {
                    String::new()
                };
                if crate::scheduler::trace_on() && !why.is_empty() {
                    eprintln!("[graphs] recapture reason: {why}");
                }
                !why.is_empty()
            }
            None => {
                if crate::scheduler::trace_on() {
                    eprintln!("[graphs] recapture reason: no graph");
                }
                true
            }
        };
        if crate::scheduler::trace_on() {
            eprintln!("[graphs] step b={b} need_capture={need_capture}");
        }

        if need_capture {
            self.decode_graph = None;
            // Eager graphed-forward: реальный результат шага + prime всех ядер/кэшей.
            // Guard включает htod param cache: params_from_vec идёт в кэш (prime),
            // а при захвате промах → явная ошибка вместо pageable memcpy в графе.
            let _htod_guard = cuda_dev.enable_cuda_graph_htod_cache();
            let ids_eager = Tensor::from_vec(tokens.to_vec(), (b, 1usize), &self.device)?;
            {
                let ctx = self.model.paged_ctx.as_mut().unwrap();
                ctx.stage_inputs(slots, rope_positions, &block_table)?;
            }
            let (logits, hidden) = self
                .model
                .forward_decode_batch_graphed(&ids_eager, slots)
                .map_err(|e| anyhow!("graphed forward (eager prime): {e}"))?;
            // Второй eager прогон (замер тёплого пути): ТОЛЬКО диагностика.
            // ВАЖНО: двигает device KV на +1 — рассинхронизирует scheduler
            // позиции после capture. Поэтому gprof+graphs даёт корректные
            // замеры только первого шага; последующие уходят в eager.
            // Не инкрементируем kv_len_host за этот прогон намеренно? Нет —
            // инкремент ниже (1206) покрывает оба прогона; расхождение
            // остаётся и это известное ограничение диагностики.
            if self.model.gprof_events.is_some() {
                {
                    let ctx = self.model.paged_ctx.as_mut().unwrap();
                    for &s in slots {
                        ctx.kv_len_host[s as usize] += 1;
                    }
                }
                let _ = self
                    .model
                    .forward_decode_batch_graphed(&ids_eager, slots)
                    .map_err(|e| anyhow!("graphed forward (eager prime 2): {e}"))?;
                // Компенсация: вернуть host mirror к состоянию ПОСЛЕ одного
                // реального шага, чтобы scheduler-позиции совпали с device
                // только на реальный инкремент (строка ниже).
                {
                    let ctx = self.model.paged_ctx.as_mut().unwrap();
                    for &s in slots {
                        ctx.kv_len_host[s as usize] -= 1;
                    }
                }
            }
            // GPU-сегменты eager-prime: та же последовательность ядер, что в графе.
            if let Some(ev) = self.model.gprof_events.as_ref() {
                cuda_dev.cuda_stream().synchronize()?;
                use cudarc::driver::result as cres;
                if let (Ok(e0), Ok(e1), Ok(e2), Ok(e3)) = (
                    unsafe { cres::event::elapsed(ev[0], ev[1]) },
                    unsafe { cres::event::elapsed(ev[1], ev[2]) },
                    unsafe { cres::event::elapsed(ev[2], ev[3]) },
                    unsafe { cres::event::elapsed(ev[0], ev[3]) },
                ) {
                    eprintln!(
                        "[gGPU-prime] emb={e0:.2}ms blocks={e1:.2}ms head={e2:.2}ms total={e3:.2}ms"
                    );
                } else {
                    eprintln!("[gGPU-prime] elapsed failed");
                }
            }
            if let Some(bevs) = self.model.gprof_block_events.as_ref() {
                cuda_dev.cuda_stream().synchronize()?;
                use cudarc::driver::result as cres;
                let n = bevs.len();
                let mut times: Vec<(usize, f32)> = Vec::with_capacity(n.saturating_sub(1));
                let mut dsum = 0f32;
                let mut dcount = 0usize;
                let mut asum = 0f32;
                let mut acount = 0usize;
                for i in 0..n.saturating_sub(1) {
                    if let Ok(t) = unsafe { cres::event::elapsed(bevs[i], bevs[i + 1]) } {
                        times.push((i, t));
                        let is_delta = self.model.blocks[i].is_deltanet();
                        if is_delta { dsum += t; dcount += 1; } else { asum += t; acount += 1; }
                    }
                }
                times.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top: Vec<String> = times.iter().take(8).map(|(i, t)| format!("b{i}={t:.2}")).collect();
                eprintln!("[gGPU-blocks] delta_sum={dsum:.1}ms/{dcount} attn_sum={asum:.1}ms/{acount} top: {}", top.join(" "));
            }
            // host mirror kv_len после инкремента
            {
                let ctx = self.model.paged_ctx.as_mut().unwrap();
                for &s in slots {
                    ctx.kv_len_host[s as usize] += 1;
                }
            }
            let flat = logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()
                .map_err(|e| anyhow!("graphed logits: {e}"))?;
            let vocab = self.vocab_size();
            if flat.len() != b * vocab {
                return Err(anyhow!("graphed logits length mismatch"));
            }
            let out: Vec<Vec<f32>> = flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect();
            // Capture для следующих шагов (захват не исполняет ядра).
            let stream = cuda_dev.cuda_stream();
            let capture_result = (|| -> Result<DecodeGraphState> {
                use cudarc::driver::{result as cres, sys as csys};
                // ВНИМАНИЕ: память, выделенная ВНУТРИ захвата, принадлежит graph pool —
                // её адреса валидны только внутри graph launch. Внешний D2H по ним →
                // illegal address. Поэтому выход копируем во ВНЕШНИЙ (default pool,
                // выделен ДО захвата) буфер D2D-нодой внутри графа.
                let ids_eager = Tensor::from_vec(tokens.to_vec(), (b, 1usize), &self.device)?;
                let logits_out = Tensor::zeros((b, self.vocab), DType::F32, &self.device)?;
                unsafe {
                    cres::stream::begin_capture(
                        stream.cu_stream(),
                        csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    )
                }
                .map_err(|e| anyhow!("begin_capture: {e}"))?;
                let forward_result = self
                    .model
                    .forward_decode_batch_graphed(&ids_eager, slots);
                let (logits_t, hidden_t) = match forward_result {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                        return Err(anyhow!("graphed forward (capture): {e}"));
                    }
                };
                // D2D копия внутри графа: graph-pool → внешний буфер.
                if let Err(e) = logits_out.slice_set(&logits_t, 0, 0) {
                    let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                    return Err(anyhow!("logits_out copy: {e}"));
                }
                drop(logits_t);
                drop(hidden_t);
                let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
                    .map_err(|e| anyhow!("end_capture: {e}"))?;
                if cu_graph.is_null() {
                    return Err(anyhow!("end_capture returned null graph"));
                }
                {
                    let mut nodes: usize = 0;
                    let res = unsafe {
                        csys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut nodes)
                    };
                    eprintln!("[graphs] captured nodes={nodes} res={res:?}");
                    if std::env::var("QWEN36_GRAPH_DOT").as_deref() == Ok("1") {
                        let path = std::ffi::CString::new("D:\\Projects\\yttri-build\\decode-graph.dot").unwrap();
                        let res = unsafe {
                            csys::cuGraphDebugDotPrint(cu_graph, path.as_ptr(), 1u32)
                        };
                        eprintln!("[graphs] dot dump res={res:?}");
                    }
                }
                let mut exec: csys::CUgraphExec = std::ptr::null_mut();
                let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
                if res != csys::CUresult::CUDA_SUCCESS || exec.is_null() {
                    unsafe { csys::cuGraphDestroy(cu_graph) };
                    return Err(anyhow!("graph instantiate failed: {res:?}"));
                }
                Ok(DecodeGraphState {
                    exec,
                    cu_graph,
                    stream: stream.clone(),
                    b,
                    slots: slots.to_vec(),
                    ids_t: ids_eager.clone(),
                    logits_t: logits_out,
                })
            })();
            match capture_result {
                Ok(state) => {
                    self.decode_graph = Some(state);
                    if crate::scheduler::trace_on() {
                        eprintln!("[graphs] captured decode graph B={b} slots={slots:?}");
                    }
                }
                Err(e) => {
                    // Захват не удался — продолжаем eager, логируем один раз.
                    eprintln!("[graphs] capture failed (eager fallback): {e}");
                    self.graphs_enabled = false;
                }
            }
            let _ = hidden;
            return Ok(Some(out));
        }

        // Replay path.
        let state = self.decode_graph.as_ref().unwrap();
        let stream = cuda_dev.cuda_stream();
        {
            let ctx = self.model.paged_ctx.as_mut().unwrap();
            ctx.stage_inputs(slots, rope_positions, &block_table)?;
        }
        // ids: htod через staging tensor + slice_set (вне графа).
        let ids_staging =
            Tensor::from_vec(tokens.to_vec(), (b, 1usize), &Device::Cpu)?.to_device(&self.device)?;
        state.ids_t.slice_set(&ids_staging, 0, 0)?;
        state.launch().map_err(|e| anyhow!("graph launch: {e}"))?;
        // Диагностика: sync сразу после launch, чтобы async-ошибка графа
        // привязывалась к этому шагу, а не всплывала sticky на следующем.
        if crate::scheduler::trace_on() || self.model.gprof_events.is_some() {
            cuda_dev
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow!("graph post-launch sync: {e}"))?;
        }
        if let Some(ev) = self.model.gprof_events.as_ref() {
            use cudarc::driver::result as cres;
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 3 || n % 32 == 0 {
                let e0 = unsafe { cres::event::elapsed(ev[0], ev[1]) };
                let e1 = unsafe { cres::event::elapsed(ev[1], ev[2]) };
                let e2 = unsafe { cres::event::elapsed(ev[2], ev[3]) };
                let e3 = unsafe { cres::event::elapsed(ev[0], ev[3]) };
                if let (Ok(e0), Ok(e1), Ok(e2), Ok(e3)) = (e0, e1, e2, e3) {
                    eprintln!(
                        "[gGPU] #{n} emb={e0:.2}ms blocks={e1:.2}ms head={e2:.2}ms total={e3:.2}ms"
                    );
                } else {
                    eprintln!("[gGPU] #{n} elapsed err: {e0:?} {e1:?} {e2:?} {e3:?}");
                }
            }
        }
        {
            let ctx = self.model.paged_ctx.as_mut().unwrap();
            for &s in slots {
                ctx.kv_len_host[s as usize] += 1;
            }
        }
        let _ = stream;
        let flat = state
            .logits_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()
            .map_err(|e| anyhow!("graph logits read: {e}"))?;
        let vocab = self.vocab_size();
        if flat.len() != b * vocab {
            return Err(anyhow!("graph logits length mismatch"));
        }
        Ok(Some(flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect()))
    }

    /// EOS token id модели.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Размер словаря (для BatchScheduler::new).
    pub fn vocab_size(&self) -> usize {
        if self.vocab != 0 {
            self.vocab
        } else {
            151943
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_position_plan_slices_fail_closed() {
        let plan = PositionPlan {
            text_positions: vec![0, 1, 2],
            rope_positions: [vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]],
            decode_rope_delta: -1,
        };
        let slice = slice_position_plan(&plan, 1, 2).unwrap();
        assert_eq!(slice.text_positions, vec![1, 2]);
        assert_eq!(slice.rope_positions[0], vec![1, 2]);
        assert!(slice_position_plan(&plan, 2, 2).is_err());
    }

    #[test]
    fn load_rejects_excess_slots_before_opening_gguf() {
        let missing = Path::new("this-model-must-not-exist.gguf");
        let error = match Qwen35BatchAdapter::load(
            missing,
            Device::Cpu,
            DECODE_BATCH_CAPACITY as usize + 1,
        ) {
            Ok(_) => panic!("excess slots unexpectedly accepted"),
            Err(error) => error,
        };
        assert_eq!(
            error.to_string(),
            format!(
                "num_slots {} exceeds decode capacity {DECODE_BATCH_CAPACITY}",
                DECODE_BATCH_CAPACITY + 1
            )
        );
    }
}
