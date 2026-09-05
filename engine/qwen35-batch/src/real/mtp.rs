//! Qwen3.5 single-head MTP runtime over thin mixed-Q8 GGUF.

use candle_core::{
    quantized::{gguf_file, QMatMul},
    DType, Device, IndexOp, Module, Result, Tensor,
};
use candle_nn::RmsNorm;
use std::{fs::File, path::Path, sync::Arc};

use super::{
    model_profile::{ModelProfile, MtpProfile},
    model_weights::ModelWeights,
    moe::{
        ForwardMode, MoeRouter, PackedExperts, Qwen35MoeBlock, Qwen35MoeConfig,
        select_backend, SharedExpert,
    },
    multimodal::MROPE_DIMENSION_SOURCES,
};

// hidden/heads/kv_heads — из MtpProfile (4B: 2560/16/4, Qwen3.8-27B: 5120/24/4).
// HEAD_DIM/ROPE_DIM/ROPE_BASE — инварианты семейства qwen35, валидируются
// в MtpProfile::read_and_validate.
const HEAD_DIM: usize = 256;
const ROPE_DIM: usize = 64;
const ROPE_BASE: f64 = 10_000_000.0;

/// KV головы: предвыделенные буферы с запасом по длине, строки дописываются
/// на месте. Раньше на каждом шаге черновика делался `Tensor::cat` всего KV —
/// O(контекст) копия на шаг. Чекпоинт транзакции разделяет буферы и держит
/// свою `len`: черновик пишет только за committed-префикс, откат — сдвиг длины,
/// так что общий буфер безопасен (та же логика, что у batched-кеша модели).
#[derive(Clone)]
struct MtpKv {
    k: Tensor, // [1, cap, KV_HEADS, HEAD_DIM], F16; валидны строки 0..len
    v: Tensor,
    len: usize,
}

/// Шаг роста буфера KV головы (строк). Черновик дописывает 1–8 строк на раунд,
/// префил — по чанку; 4096 строк × 4 головы × 256 × F16 = 8 МБ на K.
const MTP_KV_GROW: usize = 4096;

#[derive(Clone, Default)]
struct MtpSlot {
    kv: Option<MtpKv>,
    pending_target_hidden: Option<Tensor>, // normalized target hidden [1,HIDDEN]
}

#[derive(Clone)]
struct MtpSlotCheckpoint {
    kv: Option<MtpKv>,
    pending_target_hidden: Option<Tensor>,
}

struct MtpTransaction {
    checkpoint: MtpSlotCheckpoint,
}

/// Короткая проекция черновика: строки `output.weight` только для токенов из
/// списка. Черновик гадает, а не решает (неверный токен отвергнет проверка),
/// поэтому ему не нужен весь словарь: у Ornith-9B проекция на 248k строк —
/// 834 МБ Q6_K, 2.8 мс из 5 мс прохода головы; шортлист на 32k — 107 МБ.
struct ShortHead {
    proj: QMatMul,
    /// Обратное отображение: индекс в шортлисте → id токена (U32 [n] на устройстве).
    ids: Tensor,
    ids_host: Vec<u32>,
}

pub struct Qwen35Mtp {
    profile: MtpProfile,
    device: Device,
    hnorm: RmsNorm,
    enorm: RmsNorm,
    eh_proj: QMatMul,
    attn_norm: RmsNorm,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o: QMatMul,
    ffn_norm: RmsNorm,
    /// FR-011: FFN черновика — плотный или MoE (у 35B-A3B nextn-слой MoE).
    ffn: MtpFfn,
    head_norm: RmsNorm,
    shared_head: QMatMul,
    short_head: Option<ShortHead>,
    slots: Vec<MtpSlot>,
    transactions: Vec<Option<MtpTransaction>>,
    /// Граф прохода головы на слот (черновик): ключ — адрес и ёмкость кеша
    /// K/V слота, при их смене перезахват. При сбое захвата — eager до
    /// конца процесса.
    #[cfg(feature = "cuda")]
    draft_graphs: Vec<Option<DraftGraph>>,
    #[cfg(feature = "cuda")]
    draft_graph_failed: bool,
}

/// Стейджинг прохода головы: адреса стабильны между replay, всё, что меняется
/// от прохода к проходу, кладётся сюда ДО launch, выходы читаются ПОСЛЕ.
/// FFN черновика MTP: плотный (4B/Ornith/Qwen3.8) или MoE (35B-A3B, FR-011).
pub enum MtpFfn {
    Dense {
        gate: QMatMul,
        up: QMatMul,
        down: QMatMul,
    },
    Moe {
        block: Qwen35MoeBlock,
    },
}

#[derive(Clone)]
struct DraftStaging {
    emb_in: Tensor,    // [1,1,H] F32 — эмбеддинг токена (деквант на хосте, H2D)
    hidden_in: Tensor, // [1,1,H] F32 — вход; граф сам кладёт сюда pre_head для следующего прохода
    cos_in: Tensor,    // [1, ROPE_DIM/2] F32
    sin_in: Tensor,
    len_dev: Tensor,   // U32 [1] — длина кеша головы (граф инкрементирует)
    zero_slot: Tensor, // U32 [1] = [0] — «слот» для cumsum_seqlens
    one_u32: Tensor,   // U32 [1] = [1]
    seqlens_q: Tensor, // U32 [2] = [0, 1]
    seqlens_k: Tensor, // U32 [2] — заполняется в графе: [0, len + 1]
    out_id: Tensor,    // U32 [1] — argmax логитов головы
}

#[cfg(feature = "cuda")]
impl DraftStaging {
    fn new(device: &Device, hidden: usize) -> Result<Self> {
        Ok(Self {
            emb_in: Tensor::zeros((1, 1, hidden), DType::F32, device)?,
            hidden_in: Tensor::zeros((1, 1, hidden), DType::F32, device)?,
            cos_in: Tensor::zeros((1, ROPE_DIM / 2), DType::F32, device)?,
            sin_in: Tensor::zeros((1, ROPE_DIM / 2), DType::F32, device)?,
            len_dev: Tensor::zeros(1, DType::U32, device)?,
            zero_slot: Tensor::zeros(1, DType::U32, device)?,
            one_u32: Tensor::from_vec(vec![1u32], 1, device)?,
            seqlens_q: Tensor::from_vec(vec![0u32, 1u32], 2, device)?,
            seqlens_k: Tensor::zeros(2, DType::U32, device)?,
            out_id: Tensor::zeros(1, DType::U32, device)?,
        })
    }
}

#[cfg(feature = "cuda")]
struct DraftGraph {
    exec: cudarc::driver::sys::CUgraphExec,
    cu_graph: cudarc::driver::sys::CUgraph,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// Адрес и ёмкость кеша K/V, запечённые в граф.
    k_ptr: u64,
    cap: usize,
    st: DraftStaging,
}

#[cfg(feature = "cuda")]
impl Drop for DraftGraph {
    fn drop(&mut self) {
        unsafe {
            cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            cudarc::driver::sys::cuGraphDestroy(self.cu_graph);
        }
    }
}

/// MTP_GRAPH=0 — черновик eager (диагностика).
#[cfg(feature = "cuda")]
fn draft_graph_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MTP_GRAPH").as_deref() != Ok("0"))
}

/// MTP_GRAPH_RECAPTURE=1 — перезахватывать граф на каждом вызове
/// черновика (диагностика: исчез ли недетерминизм вместе с состоянием,
/// живущим внутри графа между вызовами).
#[cfg(feature = "cuda")]
fn draft_graph_recapture() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MTP_GRAPH_RECAPTURE").as_deref() == Ok("1"))
}

/// MTP_GRAPH_LEAK=1 — при перезахвате старый граф не разрушать и его
/// стейджинг не освобождать (утечка; диагностика: виновато ли разрушение).
#[cfg(feature = "cuda")]
fn draft_graph_leak() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MTP_GRAPH_LEAK").as_deref() == Ok("1"))
}

/// MTP_DRAFT_LOG=1 — печатать id каждого чернового токена. Нужен, чтобы
/// сравнить последовательности черновика между прогонами, не вмешиваясь в них:
/// D2H этого id в обычном пути и так есть.
#[cfg(feature = "cuda")]
fn draft_id_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MTP_DRAFT_LOG").as_deref() == Ok("1"))
}

/// MTP_GRAPH_CHECK=1 — после каждого replay повторить проход eager на
/// тех же входах и сравнить id и hidden; печать по проходу. Состояние после
/// проверки — от eager-прохода (строка кеша перезаписана тем же значением).
fn moe_meta_u32(content: &candle_core::quantized::gguf_file::Content, key: &str, default: u32) -> u32 {
    content
        .metadata
        .get(key)
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(default)
}

fn moe_n_experts(content: &candle_core::quantized::gguf_file::Content, _prefix: &str) -> usize {
    moe_meta_u32(content, "qwen35moe.expert_count", 256) as usize
}

fn moe_topk(content: &candle_core::quantized::gguf_file::Content, _prefix: &str) -> usize {
    moe_meta_u32(content, "qwen35moe.expert_used_count", 8) as usize
}

fn moe_intermediate(content: &candle_core::quantized::gguf_file::Content, _prefix: &str) -> usize {
    moe_meta_u32(content, "qwen35moe.expert_feed_forward_length", 512) as usize
}

fn moe_shared_intermediate(content: &candle_core::quantized::gguf_file::Content, _prefix: &str) -> usize {
    moe_meta_u32(content, "qwen35moe.expert_shared_feed_forward_length", 512) as usize
}

fn draft_graph_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MTP_GRAPH_CHECK").as_deref() == Ok("1"))
}

impl Qwen35Mtp {
    pub fn load(
        path: &Path,
        device: Device,
        slots: usize,
        text_profile: &ModelProfile,
        shared_head: QMatMul,
    ) -> Result<Self> {
        let file = File::open(path).map_err(candle_core::Error::wrap)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(candle_core::Error::wrap)?;
        let mmap = Arc::new(mmap);
        // Формат-независимо: GGUF или самостоятельный контейнер .ytf.
        let content = crate::real::ytf16::content_any(mmap.as_ref())?;
        let profile = MtpProfile::read_and_validate(&content, text_profile)?;
        let rms_norm_eps = profile.rms_norm_eps;
        let data: &[u8] = &mmap;
        let qmat = |name: &str| -> Result<QMatMul> {
            QMatMul::from_qtensor(content.tensor_from_slice(data, name, &device)?)
        };
        let norm = |name: &str| -> Result<RmsNorm> {
            let weight = content
                .tensor_from_slice(data, name, &Device::Cpu)?
                .dequantize(&Device::Cpu)?
                .to_device(&device)?;
            Ok(RmsNorm::new(weight, rms_norm_eps))
        };
        let prefix = format!("blk.{}", profile.mtp_block);
        // FR-011: MoE-вариант (35B-A3B): nextn-слой содержит routed experts
        // + router + shared expert вместо плотного FFN. Эксперты всегда
        // VRAM-резиденты (D-007, ~307 МиБ).
        let is_moe = content.tensor_infos.contains_key(&format!("{prefix}.ffn_gate_exps.weight"));
        let ffn = if is_moe {
            #[cfg(feature = "cuda")]
            {
                let moe_cfg = Qwen35MoeConfig {
                    hidden_size: profile.hidden_size,
                    n_experts: moe_n_experts(&content, &prefix),
                    n_experts_per_tok: moe_topk(&content, &prefix),
                    routed_intermediate: moe_intermediate(&content, &prefix),
                    shared_intermediate: moe_shared_intermediate(&content, &prefix),
                    norm_topk_prob: true,
                };
                // Router: ffn_gate_inp → dequantize → F32 Linear.
                let router_qt = content.tensor_from_slice(data, &format!("{prefix}.ffn_gate_inp.weight"), &device)?;
                let router_w = router_qt.dequantize(&device)?.to_dtype(DType::F32)?;
                let router = MoeRouter::new(
                    candle_nn::Linear::new(router_w, None),
                    moe_cfg.n_experts,
                    moe_cfg.n_experts_per_tok,
                    moe_cfg.norm_topk_prob,
                );
                // Routed experts: VRAM-резиденты (D-007).
                let gate_qt = content.tensor_from_slice(data, &format!("{prefix}.ffn_gate_exps.weight"), &device)?;
                let up_qt = content.tensor_from_slice(data, &format!("{prefix}.ffn_up_exps.weight"), &device)?;
                let down_qt = content.tensor_from_slice(data, &format!("{prefix}.ffn_down_exps.weight"), &device)?;
                let backend = select_backend(None, device.is_cuda(), gate_qt.dtype(), up_qt.dtype(), down_qt.dtype())?;
                let routed = PackedExperts {
                    gate: Arc::new(gate_qt),
                    up: Arc::new(up_qt),
                    down: Arc::new(down_qt),
                    n_experts: moe_cfg.n_experts,
                };
                // Shared expert.
                let shexp_gate_inp_qt = content.tensor_from_slice(data, &format!("{prefix}.ffn_gate_inp_shexp.weight"), &device)?;
                let shexp_gate_inp_w = shexp_gate_inp_qt.dequantize(&device)?.to_dtype(DType::F32)?;
                let shexp_gate_inp_w = if shexp_gate_inp_w.rank() == 1 {
                    let h = shexp_gate_inp_w.dim(0)?;
                    shexp_gate_inp_w.reshape((1, h))?
                } else {
                    shexp_gate_inp_w
                };
                let shared_expert = SharedExpert::new(
                    candle_nn::Linear::new(shexp_gate_inp_w, None),
                    qmat(&format!("{prefix}.ffn_gate_shexp.weight"))?,
                    qmat(&format!("{prefix}.ffn_up_shexp.weight"))?,
                    qmat(&format!("{prefix}.ffn_down_shexp.weight"))?,
                );
                let block = Qwen35MoeBlock::new(moe_cfg, router, routed, shared_expert, backend);
                MtpFfn::Moe { block }
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("MoE MTP требует CUDA-фичу");
            }
        } else {
            MtpFfn::Dense {
                gate: qmat(&format!("{prefix}.ffn_gate.weight"))?,
                up: qmat(&format!("{prefix}.ffn_up.weight"))?,
                down: qmat(&format!("{prefix}.ffn_down.weight"))?,
            }
        };
        let mut mtp = Self {
            profile,
            device: device.clone(),
            hnorm: norm(&format!("{prefix}.nextn.hnorm.weight"))?,
            enorm: norm(&format!("{prefix}.nextn.enorm.weight"))?,
            eh_proj: qmat(&format!("{prefix}.nextn.eh_proj.weight"))?,
            attn_norm: norm(&format!("{prefix}.attn_norm.weight"))?,
            q: qmat(&format!("{prefix}.attn_q.weight"))?,
            k: qmat(&format!("{prefix}.attn_k.weight"))?,
            v: qmat(&format!("{prefix}.attn_v.weight"))?,
            q_norm: norm(&format!("{prefix}.attn_q_norm.weight"))?,
            k_norm: norm(&format!("{prefix}.attn_k_norm.weight"))?,
            o: qmat(&format!("{prefix}.attn_output.weight"))?,
            ffn_norm: norm(&format!("{prefix}.post_attention_norm.weight"))?,
            ffn,
            head_norm: norm(&format!("{prefix}.nextn.shared_head_norm.weight"))?,
            shared_head,
            short_head: None,
            slots: vec![MtpSlot::default(); slots],
            transactions: (0..slots).map(|_| None).collect(),
            #[cfg(feature = "cuda")]
            draft_graphs: (0..slots).map(|_| None).collect(),
            #[cfg(feature = "cuda")]
            draft_graph_failed: false,
        };
        // Keep ownership construction after loaders stopped borrowing device.
        mtp.device = device;
        Ok(mtp)
    }

    pub fn profile(&self) -> &MtpProfile {
        &self.profile
    }

    /// Ограничить словарь черновика списком id: собрать проекцию из строк
    /// общей головы (байты квантованных блоков копируются построчно, деквант не
    /// нужен). Пустой список — полный словарь. Захваченные графы сбрасываются.
    pub fn set_vocab_shortlist(&mut self, ids: Vec<u32>) -> Result<()> {
        #[cfg(feature = "cuda")]
        for g in self.draft_graphs.iter_mut() {
            *g = None;
        }
        if ids.is_empty() {
            self.short_head = None;
            return Ok(());
        }
        let QMatMul::QTensor(full) = &self.shared_head else {
            candle_core::bail!("MTP shortlist: общая голова не квантованный тензор");
        };
        let dims = full.shape().dims().to_vec();
        if dims.len() != 2 {
            candle_core::bail!("MTP shortlist: форма головы {dims:?}, ожидалась [vocab, hidden]");
        }
        let (vocab, hidden) = (dims[0], dims[1]);
        let dtype = full.dtype();
        let row_bytes = hidden / dtype.block_size() * dtype.type_size();
        let data = full.data()?; // D2H один раз при загрузке
        if data.len() != vocab * row_bytes {
            candle_core::bail!(
                "MTP shortlist: размер данных головы {} != {vocab} × {row_bytes}",
                data.len()
            );
        }
        // Дополнение до кратности 256 повтором последнего id: ядра матвека и
        // MMQ обрабатывают строки группами, хвост не кратной длины — отдельная
        // ветка, которую проще не открывать; дубликат при равенстве логитов
        // проигрывает оригиналу (argmax берёт меньший индекс) и на выбор не
        // влияет.
        let mut ids = ids;
        while ids.len() % 256 != 0 {
            let last = *ids.last().expect("список не пуст");
            ids.push(last);
        }
        let mut bytes = Vec::with_capacity(ids.len() * row_bytes);
        for &id in &ids {
            let id = id as usize;
            if id >= vocab {
                candle_core::bail!("MTP shortlist: id {id} вне словаря {vocab}");
            }
            bytes.extend_from_slice(&data[id * row_bytes..(id + 1) * row_bytes]);
        }
        let short = candle_core::quantized::ggml_file::qtensor_from_ggml(
            dtype,
            &bytes,
            vec![ids.len(), hidden],
            &self.device,
        )?;
        let ids_t = Tensor::from_vec(ids.clone(), ids.len(), &self.device)?;
        let proj = QMatMul::from_qtensor(short)?;
        // MTP_SHORTLIST_CHECK=1: логиты пересобранной головы против
        // исходной на одном случайном входе — строки должны совпасть побитово
        // (те же байты блоков, то же ядро). Расхождение = дефект пересборки.
        if std::env::var("MTP_SHORTLIST_CHECK").as_deref() == Ok("1") {
            let x = Tensor::randn(0f32, 1f32, (1, hidden), &self.device)?;
            let full_logits = self.shared_head.forward(&x)?.flatten_all()?; // [vocab]
            let short_logits = proj.forward(&x)?.flatten_all()?; // [n]
            let picked = full_logits.index_select(&ids_t, 0)?; // [n]
            let diff = (short_logits.clone() - &picked)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let exact = short_logits
                .to_vec1::<f32>()?
                .iter()
                .zip(picked.to_vec1::<f32>()?.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "[mtp] shortlist check: n={} max|Δ|={diff:.3e} строк с побитовым расхождением={exact}",
                ids.len()
            );
        }
        self.short_head = Some(ShortHead {
            proj,
            ids: ids_t,
            ids_host: ids,
        });
        Ok(())
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.check_slot(slot)?;
        // Буфер K/V головы переживает сброс: его адрес запечён в граф
        // черновика, а строки за len никогда не читаются — перевыделять и
        // обнулять незачем. Перевыделение означало перезахват графа на каждом
        // запросе, адрес которого зависел от аллокатора (2026-08-28).
        if let Some(kv) = self.slots[slot].kv.as_mut() {
            kv.len = 0;
        }
        self.slots[slot].pending_target_hidden = None;
        self.transactions[slot] = None;
        Ok(())
    }

    pub fn begin(&mut self, slot: usize) -> Result<()> {
        self.check_slot(slot)?;
        if self.transactions[slot].is_some() {
            candle_core::bail!("MTP transaction already active for slot {slot}");
        }
        let state = &self.slots[slot];
        self.transactions[slot] = Some(MtpTransaction {
            checkpoint: MtpSlotCheckpoint {
                kv: state.kv.clone(),
                pending_target_hidden: state.pending_target_hidden.clone(),
            },
        });
        Ok(())
    }

    pub fn rollback(&mut self, slot: usize) -> Result<()> {
        self.check_slot(slot)?;
        if let Some(transaction) = self.transactions[slot].take() {
            self.slots[slot].kv = transaction.checkpoint.kv;
            self.slots[slot].pending_target_hidden =
                transaction.checkpoint.pending_target_hidden;
        }
        Ok(())
    }

    /// Keep only rows target actually verified and replace draft carry with
    /// target normalized hidden from last consumed input.
    pub fn commit(&mut self, slot: usize, verified_target_hidden: &[Tensor]) -> Result<()> {
        self.check_slot(slot)?;
        let transaction = self.transactions[slot]
            .take()
            .ok_or_else(|| candle_core::Error::Msg("MTP transaction is not active".into()))?;
        if verified_target_hidden.is_empty() {
            self.slots[slot].kv = transaction.checkpoint.kv;
            self.slots[slot].pending_target_hidden =
                transaction.checkpoint.pending_target_hidden;
            return Ok(());
        }
        let base = transaction
            .checkpoint
            .kv
            .as_ref()
            .map(|cache| cache.len)
            .unwrap_or(0);
        let committed = base
            .checked_add(verified_target_hidden.len())
            .ok_or_else(|| candle_core::Error::Msg("MTP committed length overflow".into()))?;
        let cache = self.slots[slot]
            .kv
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("MTP draft produced no KV state".into()))?;
        if committed > cache.len {
            candle_core::bail!("MTP committed length exceeds draft KV");
        }
        cache.len = committed;
        self.slots[slot].pending_target_hidden = Some(
            verified_target_hidden
                .last()
                .expect("checked non-empty")
                .clone(),
        );
        Ok(())
    }

    /// Catch target prompt state up. `embeddings` include post-Vision rows;
    /// `target_hidden` are normalized target rows for same chunk.
    pub fn catch_up(
        &mut self,
        slot: usize,
        embeddings: &Tensor,
        target_hidden: &Tensor,
        start_pos: usize,
        rope_positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<()> {
        self.check_slot(slot)?;
        // FR-011: embed_tokens возвращает rank-2 [n_tok, H]; unsqueeze до
        // [1, n_tok, H] для dims3() ниже (rank fix независимо от вызывающего).
        eprintln!("[mtp-dbg] catch_up: embeds rank={} shape={:?}", embeddings.dims().len(), embeddings.dims());
        let embeddings = if embeddings.dims().len() == 2 {
            embeddings.unsqueeze(0)?
        } else {
            embeddings.clone()
        };
        eprintln!("[mtp-dbg] catch_up: after unsqueeze rank={}", embeddings.dims().len());
        let (_, seq, hidden) = embeddings.dims3()?;
        if target_hidden.dims() != [1, seq, self.profile.hidden_size] || hidden != self.profile.hidden_size {
            candle_core::bail!("MTP catch-up shape mismatch");
        }
        let previous = self.slots[slot]
            .pending_target_hidden
            .clone()
            .unwrap_or(Tensor::zeros((1, self.profile.hidden_size), DType::F32, &self.device)?);
        let shifted = if seq == 1 {
            previous.unsqueeze(1)?
        } else {
            Tensor::cat(
                &[
                    &previous.unsqueeze(1)?,
                    &target_hidden.narrow(1, 0, seq - 1)?,
                ],
                1,
            )?
        };
        let _ = self.forward_rows(slot, &embeddings, &shifted, start_pos, rope_positions)?;
        self.slots[slot].pending_target_hidden =
            Some(target_hidden.i((.., seq - 1, ..))?);
        Ok(())
    }

    pub fn draft(
        &mut self,
        slot: usize,
        first_token: u32,
        start_pos: usize,
        width: usize,
        target: &ModelWeights,
    ) -> Result<Vec<u32>> {
        self.check_slot(slot)?;
        if self.transactions[slot].is_none() {
            candle_core::bail!("MTP draft requires active transaction");
        }
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() && !self.draft_graph_failed && draft_graph_enabled() {
            return self.draft_graphed(slot, first_token, start_pos, width, target);
        }
        // P0.5b: черновой контур полностью на GPU — id предыдущего шага
        // остаётся CUDA-тензором (H2D один раз в начале, D2H один раз в конце).
        // Раньше каждый шаг делал D2H-sync (argmax→u32) + H2D (from_vec),
        // дважды осушая pipeline на итерацию: ~3 мс × width на раунд.
        let mut token_t =
            Tensor::from_vec(vec![first_token], (1, 1), &self.device)?;
        let mut hidden = self.slots[slot]
            .pending_target_hidden
            .clone()
            .ok_or_else(|| candle_core::Error::Msg("MTP target hidden is not initialized".into()))?;
        let mut tokens_gpu: Vec<Tensor> = Vec::with_capacity(width);
        for offset in 0..width {
            let embedding = target.embed_tokens(&token_t, &self.device)?;
            let mtp_hidden = self.forward_rows(
                slot,
                &embedding,
                &hidden.unsqueeze(1)?,
                start_pos + offset,
                None,
            )?;
            let pre_head = mtp_hidden.i((.., 0, ..))?;
            // Разбор стоимости прохода головы: проекция на словарь против всего
            // остального. Синхронизация обязательна — иначе замеряем постановку
            // в очередь, а не ядро. Только при MTP_TIMING=1.
            // Device::synchronize() здесь не синхронизирует — первый замер дал
            // проекцию 0 мс, то есть время постановки в очередь. Ждать надо
            // именно стрим, как это делает sync_t в delta_rule_cuda.rs.
            let prof_dev = match (&self.device, crate::scheduler::mtp_timing_on()) {
                (Device::Cuda(d), true) => Some(d.clone()),
                _ => None,
            };
            let sync = |d: &Option<candle_core::CudaDevice>| {
                if let Some(d) = d {
                    #[cfg(feature = "cuda")]
                    let _ = d.cuda_stream().synchronize();
                    #[cfg(not(feature = "cuda"))]
                    let _ = d;
                }
                std::time::Instant::now()
            };
            let t_proj = sync(&prof_dev);
            let normalized = self.head_norm.forward(&pre_head)?;
            let logits = match &self.short_head {
                Some(sh) => sh.proj.forward(&normalized)?,
                None => self.shared_head.forward(&normalized)?,
            };
            let t_argmax = sync(&prof_dev);
            let d_proj = t_argmax.duration_since(t_proj);
            // Аргмакс на устройстве; тай-брейк на равных максимумах может
            // отличаться от host-argmax — на драфт не влияет (верифицирует target).
            if logits.device().is_cpu() {
                let idx = argmax(&logits.to_dtype(DType::F32)?.to_vec2()?[0]) as usize;
                let id = match &self.short_head {
                    Some(sh) => sh.ids_host[idx],
                    None => idx as u32,
                };
                tokens_gpu.push(Tensor::from_vec(vec![id], (1, 1), &self.device)?);
            } else {
                let idx = logits.argmax(candle_core::D::Minus1)?.to_dtype(DType::U32)?;
                let id = match &self.short_head {
                    Some(sh) => sh.ids.index_select(&idx.flatten_all()?, 0)?,
                    None => idx.flatten_all()?,
                };
                tokens_gpu.push(id.reshape((1, 1))?);
            }
            if prof_dev.is_some() {
                let d_argmax = sync(&prof_dev).duration_since(t_argmax);
                eprintln!(
                    "[mtp-head] проекция={:.3}мс argmax={:.3}мс",
                    d_proj.as_secs_f64() * 1000.0,
                    d_argmax.as_secs_f64() * 1000.0
                );
            }
            hidden = pre_head;
            token_t = tokens_gpu.last().unwrap().clone();
        }
        // Единственный D2H на весь draft.
        let mut out = Vec::with_capacity(width);
        for tg in &tokens_gpu {
            out.push(tg.flatten_all()?.to_vec1::<u32>()?[0]);
        }
        Ok(out)
    }

    fn forward_rows(
        &mut self,
        slot: usize,
        embeddings: &Tensor,
        target_hidden: &Tensor,
        start_pos: usize,
        rope_positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<Tensor> {
        let (_, seq, hidden) = embeddings.dims3()?;
        if target_hidden.dims() != [1, seq, hidden] || hidden != self.profile.hidden_size {
            candle_core::bail!("MTP input shape mismatch");
        }
        let e = self.enorm.forward(embeddings)?;
        let h = self.hnorm.forward(target_hidden)?;
        let projected = self.eh_proj.forward(&Tensor::cat(&[&e, &h], 2)?)?;
        let residual = &projected;
        let mixed = self.attention(
            slot,
            &self.attn_norm.forward(&projected)?,
            start_pos,
            rope_positions,
        )?;
        let after_attention = (mixed + residual)?;
        let ffn = self.ffn_norm.forward(&after_attention)?;
        let out = match &self.ffn {
            MtpFfn::Dense { gate, up, down } => {
                let activated = gate.forward(&ffn)?.silu_mul_direct(&up.forward(&ffn)?)?;
                down.forward(&activated)?
            }
            MtpFfn::Moe { block } => {
                // FR-011: MoE-черновик — те же ядра, что и ствол (T=1,
                // DecodeBatch). Эксперты VRAM-резиденты (D-007).
                let (batch, seq, n_embd) = ffn.dims3()?;
                let ffn_2d = ffn.reshape(((), n_embd))?;
                let mode = ForwardMode::DecodeBatch;
                let moe_out = block.forward(&ffn_2d, mode)?;
                moe_out.reshape((batch, seq, n_embd))?
            }
        };
        out + after_attention
    }

    fn attention(
        &mut self,
        slot: usize,
        xs: &Tensor,
        start_pos: usize,
        rope_positions: Option<&[Vec<u32>; 3]>,
    ) -> Result<Tensor> {
        let (_, seq, _) = xs.dims3()?;
        let qg = self.q.forward(xs)?.reshape((1, seq, self.profile.head_count, HEAD_DIM * 2))?;
        let q = qg
            .narrow(3, 0, HEAD_DIM)?
            .transpose(1, 2)?
            .contiguous()?;
        let gate = qg
            .narrow(3, HEAD_DIM, HEAD_DIM)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((1, seq, self.profile.kv_head_count, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((1, seq, self.profile.kv_head_count, HEAD_DIM))?
            .transpose(1, 2)?;
        let q = self
            .q_norm
            .forward(&q.flatten(0, 2)?)?
            .reshape((1, self.profile.head_count, seq, HEAD_DIM))?;
        let k = self
            .k_norm
            .forward(&k.flatten(0, 2)?)?
            .reshape((1, self.profile.kv_head_count, seq, HEAD_DIM))?;
        let (cos, sin) = match rope_positions {
            Some(positions) => mrope_tables(positions, &self.device)?,
            None => rope_tables(start_pos, seq, &self.device)?,
        };
        let q = apply_partial_rope(&q, &cos, &sin)?;
        let k = apply_partial_rope(&k, &cos, &sin)?;
        let k_new = k.transpose(1, 2)?.to_dtype(DType::F16)?.contiguous()?;
        let v_new = v.transpose(1, 2)?.to_dtype(DType::F16)?.contiguous()?;
        let (k_all, v_all, total) = self.append_kv(slot, &k_new, &v_new)?;
        let past = total - seq;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();

        // CUDA: flash-attn v2 — GQA нативно, F16 входы, F32 аккумулятор внутри.
        // Причинная маска при q_len < k_len выравнивается по правому-нижнему
        // углу: строка i видит 0..past+i — то же, что строила старая маска.
        // Без этого черновик на каждом шаге разворачивал KV на все головы в F32
        // (broadcast_as + contiguous: ~1.6 ГБ временных тензоров на шаг при 32K)
        // и строил маску seq×total на хосте — O(контекст) на каждый токен.
        #[cfg(feature = "cuda")]
        let mixed = if q.device().is_cuda() {
            let q_f = q.to_dtype(DType::F16)?.transpose(1, 2)?.contiguous()?; // [1, seq, H, hd]
            candle_flash_attn::flash_attn(&q_f, &k_all, &v_all, scale as f32, true)?
                .transpose(1, 2)?
                .to_dtype(DType::F32)?
        } else {
            self.attention_reference(&q, &k_all, &v_all, past, seq, total, scale)?
        };
        #[cfg(not(feature = "cuda"))]
        let mixed = self.attention_reference(&q, &k_all, &v_all, past, seq, total, scale)?;

        let mixed = (mixed * candle_nn::ops::sigmoid(&gate)?)?
            .transpose(1, 2)?
            .reshape((1, seq, self.profile.head_count * HEAD_DIM))?;
        self.o.forward(&mixed)
    }

    /// Дописать строки в KV головы на месте; вернуть (k, v) валидной длины.
    fn append_kv(
        &mut self,
        slot: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor, usize)> {
        let (_, seq, _kv_heads, _hd) = k_new.dims4()?;
        self.ensure_kv_capacity(slot, seq)?;
        let cache = self.slots[slot].kv.as_mut().expect("ensure_kv_capacity создаёт кеш");
        let past = cache.len;
        let total = past + seq;
        cache.k.slice_set(k_new, 1, past)?;
        cache.v.slice_set(v_new, 1, past)?;
        cache.len = total;
        Ok((cache.k.narrow(1, 0, total)?, cache.v.narrow(1, 0, total)?, total))
    }

    /// Кеш K/V головы вмещает ещё `extra` строк; рост — перевыделение с копией
    /// (адреса меняются: граф черновика слота перезахватывается по k_ptr).
    fn ensure_kv_capacity(&mut self, slot: usize, extra: usize) -> Result<()> {
        let kv_heads = self.profile.kv_head_count;
        let hd = HEAD_DIM;
        let past = self.slots[slot].kv.as_ref().map(|cache| cache.len).unwrap_or(0);
        let total = past + extra;
        let fits = self.slots[slot]
            .kv
            .as_ref()
            .map(|cache| cache.k.dim(1).map(|cap| cap >= total))
            .transpose()?
            .unwrap_or(false);
        if !fits {
            let cap = (total + MTP_KV_GROW).next_multiple_of(MTP_KV_GROW);
            let k = Tensor::zeros((1, cap, kv_heads, hd), DType::F16, &self.device)?;
            let v = Tensor::zeros((1, cap, kv_heads, hd), DType::F16, &self.device)?;
            if let Some(old) = self.slots[slot].kv.as_ref() {
                if old.len > 0 {
                    k.slice_set(&old.k.narrow(1, 0, old.len)?, 1, 0)?;
                    v.slice_set(&old.v.narrow(1, 0, old.len)?, 1, 0)?;
                }
            }
            self.slots[slot].kv = Some(MtpKv { k, v, len: past });
        }
        Ok(())
    }

    /// Эталонное внимание (Metal/CPU и откат): F32, разворот GQA на все головы,
    /// причинная маска на хосте. O(контекст) памяти на вызов — только там, где
    /// flash-attn недоступен.
    #[allow(clippy::too_many_arguments)]
    fn attention_reference(
        &self,
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        past: usize,
        seq: usize,
        total: usize,
        scale: f64,
    ) -> Result<Tensor> {
        let k = k_all.to_dtype(DType::F32)?.transpose(1, 2)?;
        let v = v_all.to_dtype(DType::F32)?.transpose(1, 2)?;
        let repeats = self.profile.head_count / self.profile.kv_head_count;
        let k = k
            .unsqueeze(2)?
            .broadcast_as((1, self.profile.kv_head_count, repeats, total, HEAD_DIM))?
            .contiguous()?
            .reshape((1, self.profile.head_count, total, HEAD_DIM))?;
        let v = v
            .unsqueeze(2)?
            .broadcast_as((1, self.profile.kv_head_count, repeats, total, HEAD_DIM))?
            .contiguous()?
            .reshape((1, self.profile.head_count, total, HEAD_DIM))?;
        let q_f32 = q.to_dtype(DType::F32)?.contiguous()?;
        let scores = (q_f32.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let mask = (0..seq)
            .flat_map(|row| {
                (0..total).map(move |column| {
                    if column <= past + row {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(mask, (1, 1, seq, total), &self.device)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        probs.contiguous()?.matmul(&v.contiguous()?)
    }

    fn check_slot(&self, slot: usize) -> Result<()> {
        if slot >= self.slots.len() {
            candle_core::bail!("MTP slot {slot} is out of range");
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Qwen35Mtp {
    /// Черновик графом: проход головы захвачен один раз на слот, дальше replay.
    /// Между проходами остаётся только зависимость по данным — id токена
    /// (4 байта D2H) и его эмбеддинг (деквант на хосте, H2D в стейджинг);
    /// hidden следующего прохода граф кладёт в стейджинг сам.
    fn draft_graphed(
        &mut self,
        slot: usize,
        first_token: u32,
        start_pos: usize,
        width: usize,
        target: &ModelWeights,
    ) -> Result<Vec<u32>> {
        use cudarc::driver::{result as cres, sys as csys};
        let cuda_dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => candle_core::bail!("draft_graphed: устройство не CUDA"),
        };
        // Ёмкость под все проходы — ДО захвата: рост меняет адреса кеша.
        self.ensure_kv_capacity(slot, width)?;
        let (k_ptr, cap, len) = {
            let cache = self.slots[slot].kv.as_ref().expect("ensure_kv_capacity создаёт кеш");
            (
                crate::real::paged_kv_cuda::tensor_cuda_ptr(&cache.k)?,
                cache.k.dim(1)?,
                cache.len,
            )
        };
        // Состояние на начало черновика (кладётся в стейджинг нулевого прохода).
        let hidden0 = self.slots[slot]
            .pending_target_hidden
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("MTP target hidden is not initialized".into()))?
            .unsqueeze(1)?; // [1,H] → [1,1,H]
        let len_t = Tensor::from_vec(vec![len as u32], 1, &self.device)?;

        let mut token = first_token;
        let mut out = Vec::with_capacity(width);
        for offset in 0..width {
            // Кеш перевыделен (рост ёмкости) — граф устарел.
            let stale = self.draft_graphs[slot]
                .as_ref()
                .is_some_and(|g| g.k_ptr != k_ptr || g.cap != cap);
            if (stale || draft_graph_recapture()) && self.draft_graphs[slot].is_some() {
                let old = self.draft_graphs[slot].take();
                if draft_graph_leak() {
                    std::mem::forget(old);
                } else {
                    // Разрушать только на пустом стриме и не выделять ничего до
                    // следующей синхронизации: освобождение внешних буферов графа
                    // рядом с instantiate соседнего ломало аллокатор (OQ-7).
                    cuda_dev.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
                    drop(old);
                    cuda_dev.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
                }
                if stale {
                    eprintln!("[mtp] граф черновика слота {slot} перезахвачен: кеш K/V перевыделен");
                }
            }
            let st = match self.draft_graphs[slot].as_ref() {
                Some(g) => g.st.clone(),
                None => DraftStaging::new(&self.device, self.profile.hidden_size)?,
            };
            if offset == 0 {
                st.hidden_in.slice_set(&hidden0, 0, 0)?;
                st.len_dev.slice_set(&len_t, 0, 0)?;
            }
            let emb = target.embed_for_graph(&[token], &self.device)?; // [1,1,H] F32
            st.emb_in.slice_set(&emb, 0, 0)?;
            let (cos, sin) = rope_tables(start_pos + offset, 1, &self.device)?;
            st.cos_in.slice_set(&cos, 0, 0)?;
            st.sin_in.slice_set(&sin, 0, 0)?;

            if self.draft_graphs[slot].is_none() {
                // Прайм (исполняется, результат — этого прохода) + захват
                // (не исполняется). Guard: параметры ядер — через htod-кэш,
                // иначе в захват попал бы pageable memcpy.
                let _guard = cuda_dev.enable_cuda_graph_htod_cache();
                self.draft_pass_body(slot, &st)?;
                let stream = cuda_dev.cuda_stream();
                let captured = (|| -> Result<(csys::CUgraphExec, csys::CUgraph)> {
                    unsafe {
                        cres::stream::begin_capture(
                            stream.cu_stream(),
                            csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                        )
                    }
                    .map_err(candle_core::Error::wrap)?;
                    if let Err(e) = self.draft_pass_body(slot, &st) {
                        let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                        return Err(e);
                    }
                    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
                        .map_err(candle_core::Error::wrap)?;
                    if cu_graph.is_null() {
                        candle_core::bail!("end_capture вернул пустой граф");
                    }
                    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
                    let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
                    if res != csys::CUresult::CUDA_SUCCESS || exec.is_null() {
                        unsafe { csys::cuGraphDestroy(cu_graph) };
                        candle_core::bail!("cuGraphInstantiate (mtp draft): {res:?}");
                    }
                    Ok((exec, cu_graph))
                })();
                match captured {
                    Ok((exec, cu_graph)) => {
                        self.draft_graphs[slot] = Some(DraftGraph {
                            exec,
                            cu_graph,
                            stream: stream.clone(),
                            k_ptr,
                            cap,
                            st: st.clone(),
                        });
                    }
                    Err(e) => {
                        eprintln!("[mtp] захват графа черновика не удался, черновик eager до перезапуска: {e}");
                        self.draft_graph_failed = true;
                    }
                }
            } else {
                // Снимки входов до replay — для проверки против eager.
                let check = draft_graph_check();
                let saved = if check {
                    Some((st.hidden_in.copy()?, st.len_dev.copy()?))
                } else {
                    None
                };
                let g = self.draft_graphs[slot].as_ref().expect("проверено выше");
                let res = unsafe { csys::cuGraphLaunch(g.exec, g.stream.cu_stream()) };
                if res != csys::CUresult::CUDA_SUCCESS {
                    candle_core::bail!("cuGraphLaunch (mtp draft) failed: {res:?}");
                }
                if let Some((hidden_before, len_before)) = saved {
                    let id_g = st.out_id.to_vec1::<u32>()?[0];
                    let hidden_g = st.hidden_in.copy()?; // pre_head из графа
                    let len_g = st.len_dev.to_vec1::<u32>()?[0];
                    // Строка K/V, которую записал граф (по длине до прохода).
                    let row = len_before.to_vec1::<u32>()?[0] as usize;
                    let (k_row_g, v_row_g) = {
                        let cache = self.slots[slot].kv.as_ref().expect("кеш есть");
                        (cache.k.narrow(1, row, 1)?.copy()?, cache.v.narrow(1, row, 1)?.copy()?)
                    };
                    // Тот же проход eager на тех же входах.
                    st.hidden_in.slice_set(&hidden_before, 0, 0)?;
                    st.len_dev.slice_set(&len_before, 0, 0)?;
                    self.draft_pass_body(slot, &st)?;
                    let id_e = st.out_id.to_vec1::<u32>()?[0];
                    let len_e = st.len_dev.to_vec1::<u32>()?[0];
                    let dh = (st.hidden_in.clone() - &hidden_g)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    let (dk, dv) = {
                        let cache = self.slots[slot].kv.as_ref().expect("кеш есть");
                        let k_e = cache.k.narrow(1, row, 1)?;
                        let v_e = cache.v.narrow(1, row, 1)?;
                        (
                            (k_e.to_dtype(DType::F32)? - k_row_g.to_dtype(DType::F32)?)?
                                .abs()?
                                .max_all()?
                                .to_scalar::<f32>()?,
                            (v_e.to_dtype(DType::F32)? - v_row_g.to_dtype(DType::F32)?)?
                                .abs()?
                                .max_all()?
                                .to_scalar::<f32>()?,
                        )
                    };
                    eprintln!(
                        "[mtp-graph-check] slot={slot} pass={offset} pos={} id graph={id_g} eager={id_e} {} len graph={len_g} eager={len_e} max|Δhidden|={dh:.3e} max|ΔK|={dk:.3e} max|ΔV|={dv:.3e}",
                        start_pos + offset,
                        if id_g == id_e { "ok" } else { "MISMATCH" }
                    );
                }
            }
            // Единственная синхронизация прохода: id токена.
            token = st
                .out_id
                .to_vec1::<u32>()
                .map_err(|e| candle_core::Error::Msg(format!("mtp draft out_id D2H: {e}")))?[0];
            out.push(token);
            // Лог id черновика: печатается там, где D2H уже произошёл, поэтому
            // сам замер синхронизации не добавляет и окно гонки не сдвигает.
            if draft_id_log() {
                eprintln!("[mtp-draft] slot={slot} pass={offset} pos={} id={token}", start_pos + offset);
            }
            // Зеркало длины на хосте (граф инкрементировал len_dev).
            self.slots[slot].kv.as_mut().expect("кеш есть").len += 1;
        }
        Ok(out)
    }

    /// Тело одного прохода головы над стейджингом: то же, что forward_rows +
    /// head_norm + shared_head + argmax, но все входы — из `st`, все выходы —
    /// в `st`, длина кеша — на устройстве. Захватывается графом целиком.
    fn draft_pass_body(&mut self, slot: usize, st: &DraftStaging) -> Result<()> {
        let e = self.enorm.forward(&st.emb_in)?;
        let h = self.hnorm.forward(&st.hidden_in)?;
        let projected = self.eh_proj.forward(&Tensor::cat(&[&e, &h], 2)?)?;
        let mixed = self.attention_graphed(slot, &self.attn_norm.forward(&projected)?, st)?;
        let after_attention = (mixed + &projected)?;
        let ffn = self.ffn_norm.forward(&after_attention)?;
        let pre = match &self.ffn {
            MtpFfn::Dense { gate, up, down } => {
                let activated = gate.forward(&ffn)?.silu_mul_direct(&up.forward(&ffn)?)?;
                down.forward(&activated)? + after_attention
            }
            MtpFfn::Moe { block } => {
                // FR-011: MoE-черновик — те же ядра, что и ствол (T=1).
                let (_, _, n_embd) = ffn.dims3()?;
                let ffn_2d = ffn.reshape(((), n_embd))?;
                let moe_out = block.forward(&ffn_2d, ForwardMode::DecodeBatch)?;
                moe_out.reshape((1, 1, n_embd))? + after_attention
            }
        }?; // [1,1,H] — unwrap Result<Tensor> для .i() ниже
        let pre_head = pre.i((.., 0, ..))?; // [1,H]
        let normalized = self.head_norm.forward(&pre_head)?;
        let logits = match &self.short_head {
            Some(sh) => sh.proj.forward(&normalized)?, // [1, n_short]
            None => self.shared_head.forward(&normalized)?, // [1, V]
        };
        let idx = logits
            .argmax(candle_core::D::Minus1)?
            .to_dtype(DType::U32)?
            .reshape(1)?;
        let id = match &self.short_head {
            Some(sh) => sh.ids.index_select(&idx, 0)?,
            None => idx,
        };
        st.out_id.slice_set(&id, 0, 0)?;
        // hidden следующего прохода = pre_head (как в eager-цикле).
        st.hidden_in.slice_set(&pre, 0, 0)?;
        // len += 1 — после внимания: ядро дописи и seqlens читали прежнюю длину.
        let next = st.len_dev.broadcast_add(&st.one_u32)?;
        st.len_dev.slice_set(&next, 0, 0)?;
        Ok(())
    }

    /// Внимание головы для одного токена с длиной кеша на устройстве: дописать
    /// строку по len_dev, seqlens_k = [0, len + 1], FA2 varlen по всему буферу
    /// ёмкости cap (реальную длину задаёт seqlens_k).
    fn attention_graphed(&mut self, slot: usize, xs: &Tensor, st: &DraftStaging) -> Result<Tensor> {
        let heads = self.profile.head_count;
        let kv_heads = self.profile.kv_head_count;
        let qg = self.q.forward(xs)?.reshape((1, 1, heads, HEAD_DIM * 2))?;
        let q = qg.narrow(3, 0, HEAD_DIM)?.transpose(1, 2)?.contiguous()?; // [1,H,1,hd]
        let gate = qg.narrow(3, HEAD_DIM, HEAD_DIM)?.transpose(1, 2)?.contiguous()?;
        let k = self
            .k
            .forward(xs)?
            .reshape((1, 1, kv_heads, HEAD_DIM))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((1, 1, kv_heads, HEAD_DIM))?
            .transpose(1, 2)?;
        let q = self
            .q_norm
            .forward(&q.flatten(0, 2)?)?
            .reshape((1, heads, 1, HEAD_DIM))?;
        let k = self
            .k_norm
            .forward(&k.flatten(0, 2)?)?
            .reshape((1, kv_heads, 1, HEAD_DIM))?;
        let q = apply_partial_rope(&q, &st.cos_in, &st.sin_in)?;
        let k = apply_partial_rope(&k, &st.cos_in, &st.sin_in)?;
        let k_new = k.transpose(1, 2)?.to_dtype(DType::F16)?.contiguous()?; // [1,1,KVH,hd]
        let v_new = v.transpose(1, 2)?.to_dtype(DType::F16)?.contiguous()?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => candle_core::bail!("attention_graphed: устройство не CUDA"),
        };
        let cache = self.slots[slot].kv.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("attention_graphed: кеш головы не создан".into())
        })?;
        let cap = cache.k.dim(1)?;
        crate::real::paged_kv_cuda::launch_kv_append_flat_f16(
            &dev, &cache.k, &cache.v, &k_new, &v_new, &st.len_dev, kv_heads, HEAD_DIM, cap,
        )?;
        crate::real::paged_kv_cuda::launch_seqlens_from_len(
            &dev, &st.len_dev, &st.zero_slot, &st.seqlens_k, 1,
        )?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let q_f = q
            .to_dtype(DType::F16)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, heads, HEAD_DIM))?; // [total_q=1, H, hd]
        let k_all = cache.k.reshape((cap, kv_heads, HEAD_DIM))?;
        let v_all = cache.v.reshape((cap, kv_heads, HEAD_DIM))?;
        let out = candle_flash_attn::flash_attn_varlen(
            &q_f,
            &k_all,
            &v_all,
            &st.seqlens_q,
            &st.seqlens_k,
            1,
            cap,
            scale as f32,
            true,
        )?; // [1, H, hd]
        let mixed = out
            .reshape((1, 1, heads, HEAD_DIM))?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?; // [1,H,1,hd]
        let mixed = (mixed * candle_nn::ops::sigmoid(&gate)?)?
            .transpose(1, 2)?
            .reshape((1, 1, heads * HEAD_DIM))?;
        self.o.forward(&mixed)
    }
}

fn rope_tables(start: usize, len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let mut cos = Vec::with_capacity(len * ROPE_DIM / 2);
    let mut sin = Vec::with_capacity(len * ROPE_DIM / 2);
    for position in start..start + len {
        for i in 0..ROPE_DIM / 2 {
            let frequency = 1.0 / ROPE_BASE.powf((2 * i) as f64 / ROPE_DIM as f64);
            let value = position as f64 * frequency;
            cos.push(value.cos() as f32);
            sin.push(value.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, ROPE_DIM / 2), device)?,
        Tensor::from_vec(sin, (len, ROPE_DIM / 2), device)?,
    ))
}

fn mrope_tables(positions: &[Vec<u32>; 3], device: &Device) -> Result<(Tensor, Tensor)> {
    let len = positions[0].len();
    if positions.iter().any(|axis| axis.len() != len) {
        candle_core::bail!("MTP MRoPE position lengths differ");
    }
    let frequencies = (0..ROPE_DIM)
        .step_by(2)
        .map(|index| 1.0 / ROPE_BASE.powf(index as f64 / ROPE_DIM as f64))
        .collect::<Vec<_>>();
    let mut cos = Vec::with_capacity(len * frequencies.len());
    let mut sin = Vec::with_capacity(len * frequencies.len());
    for token in 0..len {
        for (&axis, &frequency) in MROPE_DIMENSION_SOURCES.iter().zip(&frequencies) {
            let angle = positions[axis as usize][token] as f64 * frequency;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, ROPE_DIM / 2), device)?,
        Tensor::from_vec(sin, (len, ROPE_DIM / 2), device)?,
    ))
}

fn apply_partial_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, _, dim) = xs.dims4()?;
    let rotated = candle_nn::rotary_emb::rope(
        &xs.narrow(3, 0, ROPE_DIM)?.contiguous()?,
        cos,
        sin,
    )?;
    Tensor::cat(&[&rotated, &xs.narrow(3, ROPE_DIM, dim - ROPE_DIM)?], 3)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}
