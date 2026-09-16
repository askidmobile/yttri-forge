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
#[cfg(feature = "cuda")]
use crate::real::model_weights::GRAPH_MIN_FREE_BYTES;
use crate::real::model_weights::{
    BatchedStateCheckpoint, BlockStateSnap, ModelWeights, StateSnapshot, DECODE_BATCH_CAPACITY,
};
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
    /// Писал ли путь проверки теневые снимки DeltaNet. Их пишет ТОЛЬКО
    /// графовый `forward_verify_paged` (shadow=true и b>1); построчный
    /// `verify_eager` идёт с b=1, где `shadow_rows = b - 1 = 0`.
    /// `speculative_accept` при частичном приёме обязан смотреть на это, а не
    /// на `paged_authority`: иначе восстанавливается снимок, которого нет.
    shadow_written: bool,
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
    /// Persistent вход: [B, 1, H] F32 эмбеддинги (стейджатся htod до launch;
    /// деквант строк на хосте — копия token_embd на GPU не нужна).
    emb_t: Tensor,
    /// Внешний буфер логитов: выделен ДО захвата, граф пишет в него D2D-узлом
    /// (память graph-pool снаружи launch невалидна для D2H).
    logits_t: Tensor,
    /// Второй внешний буфер: hidden декодируемых позиций, тем же способом.
    /// Читается глубокой копией — буфер перезаписывается следующим launch.
    hidden_t: Tensor,
}

#[cfg(feature = "cuda")]
impl DecodeGraphState {
    fn launch(&self) -> Result<()> {
        let res = unsafe { cudarc::driver::sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphLaunch failed: {res:?}"));
        }
        Ok(())
    }
}

/// CUDA-graph replay состояние для prefill-чанка. Ключ LRU — (T, slot):
/// граф жёстко зашит под длину чанка и страницы слота.
#[cfg(feature = "cuda")]
struct PrefillGraphState {
    exec: cudarc::driver::sys::CUgraphExec,
    cu_graph: cudarc::driver::sys::CUgraph,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    t: usize,
    /// Persistent входы (стейджатся ВНЕ графа): эмбеддинги [1,T,H] F32
    /// (для проверки — [T,1,H]), позиции [T] U32.
    emb_t: Tensor,
    rope_pos_t: Tensor,
    /// Внешний буфер выхода (D2D-нода внутри графа) — graph-pool адреса
    /// невалидны для D2H снаружи launch.
    logits_t: Tensor,
    /// Второй внешний буфер: hidden всех T позиций чанка. Нужен MTP::catch_up,
    /// которому требуется [1, T, hidden] для построения собственного KV над
    /// префиксом. Без него графовый префил был закрыт для MTP гейтом PD-204.
    hidden_t: Tensor,
}

#[cfg(feature = "cuda")]
impl PrefillGraphState {
    fn launch(&self) -> Result<()> {
        let res = unsafe { cudarc::driver::sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) };
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(anyhow!("cuGraphLaunch (prefill) failed: {res:?}"));
        }
        Ok(())
    }
}

/// Графы не вытесняются: Drop графа (cudaFreeAsync его внешних буферов сразу
/// после instantiate следующего) ронял следующий шаг CUDA_ERROR_INVALID_VALUE
/// (OQ-7, матрица 2026-08-28: LEAK чисто, KEEP_HANDLES падает — виноваты
/// именно освобождения, не destroy). Ключи пулов без состава слотов держат их
/// маленькими; при полном пуле новую форму просто не захватываем. Drop
/// остаётся для clear() и завершения процесса.
#[cfg(feature = "cuda")]
impl Drop for PrefillGraphState {
    fn drop(&mut self) {
        unsafe {
            cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            cudarc::driver::sys::cuGraphDestroy(self.cu_graph);
        }
    }
}

/// Сколько графов префилла держим одновременно (PD-A5). Умолчание 8,
/// переопределяется PGRAPH_LRU.
#[cfg(feature = "cuda")]
fn pgraph_lru() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PGRAPH_LRU")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8)
    })
}

/// Чанки короче этого порога не захватываем в граф — считаем прогревочным
/// проходом и всё. Ключ пула — T, а длина хвостового чанка (prompt mod chunk)
/// у агентских сессий почти всегда новая: пул наполнялся хвостами и вытеснял
/// (2026-08-28, T=45 при lru=8 — первое вытеснение роняло следующий шаг).
/// Выигрыш графа на хвосте — десятки мс один раз на запрос, терять нечего;
/// а пул, забитый хвостами, не принял бы полный чанк. Умолчание — размер
/// чанка (захватываются только полные чанки); PGRAPH_MIN_T
/// переопределяет, 0 — захватывать всё (диагностика).
#[cfg(feature = "cuda")]
fn pgraph_min_capture_t() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PGRAPH_MIN_T")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| match crate::scheduler::prefill_chunk_size() {
                0 => usize::MAX,
                n => n,
            })
    })
}

/// Верхняя граница снимка prefix cache. На Ornith один Q8-снимок 64K занимает
/// около 1.1 ГиБ; более длинная временная копия вместе с рабочими буферами уже
/// может вытолкнуть рабочий набор 12-ГБ карты в WDDM shared memory.
/// `usize::MAX` сохраняет прежнее поведение на больших GPU.
fn prefix_cache_max_tokens() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("PREFIX_CACHE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(usize::MAX)
    })
}

/// Дополнительные checkpoint'ы для branch-point prefix cache.
/// По умолчанию включены; `PREFIX_CACHE_CHECKPOINTS=0` возвращает прежнее
/// поведение — только последняя граница чанка.
/// Минимальная позиция, на которой вообще имеет смысл снимать снимок границы.
///
/// Блок префикс-кэша — 64 токена (страница paged-пула), и `put` отвергает
/// снимки короче. На коротких промптах чанк всё равно кончается на позиции
/// меньше блока, а снимок стоит ~30 мс D2H состояния DeltaNet: замер
/// 2026-09-16 показал ровно эти 30 мс в фазе restore на 164-токенном
/// промпте при включённом кэше и 0 при выключенном.
fn prefix_cache_min_pos() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("PREFIX_CACHE_MIN_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(64)
    })
}

fn prefix_cache_checkpoints_enabled() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("PREFIX_CACHE_CHECKPOINTS")
            .map(|v| {
                !matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(true)
    })
}

/// Верхняя позиция дополнительного checkpoint'а. Степени двойки до этого
/// предела покрывают системные/tool-префиксы, не раздувая state cache.
fn prefix_cache_checkpoint_max() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("PREFIX_CACHE_CHECKPOINT_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8192)
    })
}

/// Сколько граничных снимков максимум хранить на один запрос. При чанке 512
/// и промпте 25K без потолка набралось бы пять десятков снимков.
fn prefix_cache_checkpoint_cap() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("PREFIX_CACHE_CHECKPOINT_CAP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(8)
    })
}
/// Размер пула декодных графов. Ключ — только ширина батча b (состав слотов
/// ядра читают из стейджинга PagedModelCtx::slots_dev), так что различных
/// форм не больше числа слотов; сверх пула не захватываем (DGRAPH_LRU).
const DGRAPH_LRU: usize = 8;

/// Максимальный размер батча, на котором ещё используются графы декода.
///
/// По умолчанию без ограничения. Предохранитель ставился, пока причина падений
/// многослотового граф-пути была неизвестна: тогда он повторял поведение
/// llama.cpp («disabling CUDA graphs due to batch size > 1»).
///
/// Обе причины найдены и устранены — кэш slot_ids, чей адрес запекался в узлы
/// графа (57c0ba5b), и освобождение тензоров вытесненного графа сразу за
/// instantiate; последнее снято структурно, составом слотов больше не
/// ключуется ни один пул, поэтому вытеснения не происходит вовсе (c910850b).
/// Проверено на 3060, 4 слота: 80 запросов без сбоев там, где до правки падало
/// на втором прогоне. Графы на четырёх слотах дают 12-20%.
///
/// Переменная оставлена как ручной предохранитель на случай, если похожее
/// всплывёт на другом железе.
fn graph_max_b() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("GRAPH_MAX_B")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(usize::MAX)
    })
}

fn dgraph_lru() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DGRAPH_LRU")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DGRAPH_LRU)
    })
}

/// Потолок сбоев графового пути на процесс (OQ-8): до него графы возвращаются
/// при следующем приёме запроса, после — выключены до перезапуска.
/// GRAPH_FAIL_CAP, умолчание 3.
fn graph_fail_cap() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("GRAPH_FAIL_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(3)
    })
}

/// Диагностика OQ-8: GRAPH_FAIL_INJECT=K — первые K захватов декодного
/// графа завершаются искусственным сбоем ещё до начала захвата (шаг уже
/// посчитан прогревочным проходом, состояние консистентно). Нет/0 — выключено.
fn graph_fail_inject() -> bool {
    use std::sync::atomic::{AtomicU32, Ordering};
    static LEFT: std::sync::OnceLock<AtomicU32> = std::sync::OnceLock::new();
    let left = LEFT.get_or_init(|| {
        AtomicU32::new(
            std::env::var("GRAPH_FAIL_INJECT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        )
    });
    left.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| v.checked_sub(1))
        .is_ok()
}

/// Режим paged/graph-prefill: `PGRAPH` = off | on | check.
///
/// Умолчание в коде — off, но это не про корректность. Подмена токенов, из-за
/// которой режим выключили («paged-prefill портит состояние»), оказалась
/// дефектом пре-токенизатора (регексп GPT-2 вместо Qwen2) и воспроизводилась
/// при выключенных графах. После починки реплей записанной сессии чист при
/// `on` (жадно и на пресете, три seed) и в многоходовой сессии с prefix cache;
/// `on` даёт декод ~3x и -866 МиБ VRAM. Прод работает с `PGRAPH=on`.
///
/// `check` прогоняет paged-путь, откатывает state и отдаёт eager-пересчёт;
/// его max |Δlogit| на поздних чанках — накопленный дрейф двух независимых
/// историй (пул против single-slot кэша), а логиты промежуточных чанков до
/// сэмплера не доходят — судить о корректности по ним нельзя.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PgraphMode {
    Off,
    On,
    Check,
}

#[cfg(feature = "cuda")]
fn pgraph_mode() -> PgraphMode {
    static MODE: std::sync::OnceLock<PgraphMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| parse_pgraph_mode(std::env::var("PGRAPH").ok().as_deref()))
}

/// Эффективный режим PGRAPH: при выгрузке экспертов графовый префил
/// принудительно выключается (FR-007) — явный PGRAPH=on не ошибка, а WARN.
#[cfg(feature = "cuda")]
fn pgraph_mode_effective(experts_ram: bool) -> PgraphMode {
    let mode = pgraph_mode();
    if experts_ram && mode != PgraphMode::Off {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            log::warn!("[pg] disabled: experts in ram (prefill needs per-layer promotion)")
        });
        PgraphMode::Off
    } else {
        mode
    }
}

#[cfg(feature = "cuda")]
fn parse_pgraph_mode(value: Option<&str>) -> PgraphMode {
    match value {
        Some("1" | "on") => PgraphMode::On,
        Some("check") => PgraphMode::Check,
        _ => PgraphMode::Off,
    }
}

/// Каким путём считается префил текущего промпта в слоте. Выбирается один раз
/// и до конца промпта не меняется.
///
/// Смешивание путей внутри одного промпта тихо теряет историю внимания в ОБЕ
/// стороны. Eager-чанк после paged: `forward_attn_with_rope` считает внимание
/// по single-slot кэшу, а тот после графовых чанков пуст — `kv_cache_len`
/// становится равен длине чанка, хотя позиции RoPE говорят `start_pos..`, и
/// весь префикс из внимания исчезает. Paged-чанк после eager: строки
/// eager-чанка в пул не попали, и append оставляет там дыру.
///
/// Ни то, ни другое ничем не диагностировалось: поле `pg_paged_only`, которое
/// по комментарию «закрывало путь явной ошибкой», ни разу не читалось.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefillPath {
    /// Первый чанк промпта ещё не посчитан — путь определят гейты.
    Undecided,
    /// KV промпта лежит ТОЛЬКО в страничном пуле.
    Paged,
    /// KV промпта лежит в single-slot кэше; в пуле его нет.
    Eager,
}

/// Можно ли увести очередной чанк на eager. Нельзя ровно в одном случае —
/// когда хотя бы один чанк этого промпта уже лёг в страничный пул.
#[cfg(feature = "cuda")]
fn eager_fallback_allowed(path: PrefillPath) -> bool {
    !matches!(path, PrefillPath::Paged)
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
    /// Снимки на границах чанков префила — для prefix cache.
    /// Раньше хранили только последнюю границу; этого мало для divergent
    /// branch, где расхождение происходит раньше последнего чанка. Теперь
    /// храним набор checkpoint'ов, ограниченный степенями двойки и env
    /// `PREFIX_CACHE_CHECKPOINT_MAX`.
    slot_prefix_snaps: Vec<Vec<(usize, StateSnapshot)>>,
    /// Снимать ли границу (включается сервером, когда кеш префикса активен):
    /// лишний снимок стоит копии всего KV в VRAM.
    capture_prefix: bool,
    /// Признак того, что слот уже засеян в batched buffers (после prefill).
    /// True = batched decode может использовать этот slot без повторного seed.
    slot_seeded: Vec<bool>,
    /// CUDA-graph replay состояние (env CUDA_GRAPHS=1).
    #[cfg(feature = "cuda")]
    /// Пул декодных графов по ключу (b, slots). Раньше хранился единственный
    /// экземпляр, и чередование обычного декода (b = число слотов) с проверкой
    /// спекуляции (b = K, все строки на один слот) пересобирало граф на каждом
    /// шаге. Захват дорог, поэтому держим небольшой пул с вытеснением, как у
    /// графов префила.
    decode_graphs: Vec<DecodeGraphState>,
    /// hidden последнего чанка, посчитанного графовым префилом. Нужен, чтобы
    /// графовая ветка заполнила mtp_inputs тем же способом, что и eager, и оба
    /// пути сошлись в ОДНУ точку вызова catch_up. Отдельный вызов внутри
    /// графовой функции приводил к тому, что MTP переставал приниматься.
    #[cfg(feature = "cuda")]
    pg_last_hidden: Option<Tensor>,
    /// LRU графов префилла (PGRAPH), ключ — (T, slot); хвост = свежий.
    #[cfg(feature = "cuda")]
    prefill_graphs: Vec<PrefillGraphState>,
    /// Графы проверки спекуляции: k строк одного слота на страничном пути,
    /// ключ (k, slot). Та же структура, что у префила, — это и есть префил-чанк
    /// длины k, только DeltaNet идёт по batched-состоянию слота.
    #[cfg(feature = "cuda")]
    verify_graphs: Vec<PrefillGraphState>,
    /// Слот, чьё состояние сейчас лежит в single-slot буферах (DeltaNet
    /// cuda_ctx + attention kv_cache), и позиция, на которой оно остановилось.
    /// Пока владелец не сменился, snapshot/restore между чанками не нужны:
    /// декод работает по batched-буферам и single-slot state не трогает.
    /// Замер 2026-08-25: снимок+восстановление стоили ~25 мс на чанк.
    state_owner: Option<(usize, usize)>,
    /// Путь префила текущего промпта в каждом слоте (см. `PrefillPath`):
    /// решается на первом чанке и до конца промпта не меняется.
    #[cfg(feature = "cuda")]
    prefill_path: Vec<PrefillPath>,
    /// Per-slot: KV данные изменились (prefill/seed) и paged pool устарел.
    #[cfg(feature = "cuda")]
    paged_dirty: Vec<bool>,
    /// Per-slot: гейт декодного графа (позиция/окно) уже отчитался, что уводит
    /// шаги на eager. Печатаем один раз на запрос: молчаливый Ok(None) стоил
    /// прогона и двух неверных гипотез 2026-08-27 (окно пула 32768 при
    /// промпте 32768 — декод весь запрос eager, а замер выглядел валидным).
    #[cfg(feature = "cuda")]
    graph_gate_warned: Vec<bool>,
    #[cfg(feature = "cuda")]
    graphs_enabled: bool,
    /// Графы разрешены конфигурацией (CUDA_GRAPHS=1 и запас VRAM). После
    /// сбоя `graphs_enabled` гаснет до следующего приёма запроса и
    /// возвращается, пока сбоев меньше потолка `graph_fail_cap()` (OQ-8).
    #[cfg(feature = "cuda")]
    graphs_configured: bool,
    #[cfg(feature = "cuda")]
    graph_failures: u32,
    /// On-demand Vision component. Phase 8 owns TTL/load barrier; adapter only
    /// consumes explicitly loaded component and per-request payloads.
    vision: Option<Qwen35Vision>,
    mtp: Option<Qwen35Mtp>,
    /// Per-slot MTP state matches the target state. Prefix-cache snapshots
    /// currently contain only the target model; after injecting one, the MTP
    /// head has neither its attention KV nor the previous target hidden row.
    /// Speculation must stay disabled for that request until MTP snapshots are
    /// added to the cache format.
    mtp_slot_aligned: Vec<bool>,
    /// Диагностический буфер (DEBUG_ALLOC_MB), см. load_mtp.
    debug_alloc: Option<Tensor>,
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
    /// FR-020: состояние выгрузки экспертов для /v1/models (capabilities.moe).
    /// None — модель без MoE или эксперты резидентны (vram).
    #[cfg(feature = "cuda")]
    pub fn moe_summary(&self) -> Option<super::expert_store::MoeRuntimeSummary> {
        let rt = self.model.moe_runtime()?;
        let (cache_mib, cache_slots, hit_rate) = rt
            .cache
            .get()
            .map(|c| (c.cache_mib, c.capacity, c.hit_rate()))
            .unwrap_or((0, 0, 0.0));
        Some(super::expert_store::MoeRuntimeSummary {
            experts_ram: true,
            pinned_bytes: rt.pinned_bytes,
            staging_bytes: rt.staging_bytes() as u64,
            cache_mib,
            cache_slots,
            hit_rate,
        })
    }

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
            let disabled = std::env::var("NO_MEMPOOL_RETAIN").as_deref() == Ok("1");
            if !disabled {
                RETAIN_ONCE.call_once(|| {
                    // Ограниченный release threshold вместо u64::MAX:
                    // retain=MAX держит пиковые префилл-транзиенты (до 1 GiB)
                    // навсегда → VRAM 97%+ → WDDM spill → decode коллапс.
                    // 512 MiBthreshold: decode-аллокации переиспользуют пул,
                    // префилл-пики освобождаются на sync-точках.
                    let thresh_mib = std::env::var("MEMPOOL_THRESHOLD_MIB")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(256); // 256 MiB — баланс: decode reuse работает, префилл-пики >256 освобождаются
                    if let Err(e) = candle_core::cuda_backend::mem_pool::set_release_threshold_mib(
                        cuda_dev, thresh_mib,
                    ) {
                        log::warn!(
                            "[qwen35-batch] set_release_threshold_mib({thresh_mib}) failed: {e}"
                        );
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

        // Самостоятельный контейнер .ytf (v2) против GGUF. Дальше по коду
        // разницы нет: у контейнера тензоры уже лежат готовыми GGML-блоками,
        // а метаданные и токенизатор развёрнуты в тот же `Content`.
        let standalone = crate::real::ytf16::container_version(mmap.as_ref())
            == Some(crate::real::ytf16::VERSION_STANDALONE);
        if standalone {
            log::info!("[qwen35-batch] самостоятельный контейнер .ytf — GGUF не нужен");
        }
        // Один проход чтения: EOS + vocab (из token_embd.weight shape[0]) + веса.
        let ct = crate::real::ytf16::content_any(mmap.as_ref())
            .map_err(|e| anyhow!("read model: {e}"))?;

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
        let mut model = if matches!(device, Device::Metal(_)) {
            ModelWeights::from_gguf_zero_copy(ct, mmap, &device)
                .map_err(|e| anyhow!("load weights zero-copy: {e}"))?
        } else {
            ModelWeights::from_gguf(ct, mmap, &device).map_err(|e| anyhow!("load weights: {e}"))?
        };
        #[cfg(not(target_os = "macos"))]
        let mut model =
            ModelWeights::from_gguf(ct, mmap, &device).map_err(|e| anyhow!("load weights: {e}"))?;

        // После загрузки весов пул держит reserved-страницы от upload staging.
        // Trim → commit возвращается к фактическим весам (важно для 27B на 12GB).
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }
        // PD-010: при выгрузке экспертов пул KV и стейджинг создаются сразу
        // при загрузке — иначе кэш (фаза 4) заберёт память пула.
        #[cfg(feature = "cuda")]
        let moe_experts_ram = model.experts_ram();
        #[cfg(feature = "cuda")]
        if moe_experts_ram {
            model
                .prepare_expert_offload(&device)
                .map_err(|e| anyhow!("prepare expert offload: {e}"))?;
        }


        // F16-GEMM сайдкара по умолчанию аккумулирует в F32 (как в pytorch).
        // На Ampere это вдвое медленнее F16-аккумуляции. F16_FAST_ACC=1
        // включает CUBLAS_COMPUTE_16F — быстрее, но копит ошибку по K=2560;
        // сторож [pfa] WARN non-finite logits ловит срыв.
        #[cfg(feature = "cuda")]
        if std::env::var("F16_FAST_ACC").as_deref() == Ok("1") {
            candle_core::cuda_backend::set_gemm_reduced_precision_f16(true);
            log::info!("[ytf] F16 GEMM: аккумуляция F16 (fast)");
        }

        #[cfg(feature = "cuda")]
        let graphs_on = {
            let want = std::env::var("CUDA_GRAPHS").as_deref() == Ok("1");
            if !want {
                false
            } else if let Device::Cuda(c) = &device {
                // Графы требуют +VRAM (graph pool + CUDA embedding). При нехватке
                // запаса после загрузки — отключаем: на 27B (11.0 GiB весов на
                // 12GB) лишние ~0.5 GiB уходят в sysmem и душат decode сильнее,
                // чем экономия на launch overhead. Порог общий с гейтом
                // эмбеддинга — см. GRAPH_MIN_FREE_BYTES в model_weights.rs.
                let free = c
                    .cuda_stream()
                    .context()
                    .mem_get_info()
                    .map(|(f, _)| f)
                    .unwrap_or(0);
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
            slot_prefix_snaps: (0..num_slots).map(|_| Vec::new()).collect(),
            capture_prefix: false,
            slot_seeded: vec![false; num_slots],
            #[cfg(feature = "cuda")]
            decode_graphs: Vec::new(),
            #[cfg(feature = "cuda")]
            pg_last_hidden: None,
            #[cfg(feature = "cuda")]
            paged_dirty: vec![true; num_slots],
            #[cfg(feature = "cuda")]
            graph_gate_warned: vec![false; num_slots],
            state_owner: None,
            #[cfg(feature = "cuda")]
            prefill_graphs: Vec::new(),
            #[cfg(feature = "cuda")]
            verify_graphs: Vec::new(),
            #[cfg(feature = "cuda")]
            prefill_path: vec![PrefillPath::Undecided; num_slots],
            #[cfg(feature = "cuda")]
            graphs_enabled: graphs_on,
            #[cfg(feature = "cuda")]
            graphs_configured: graphs_on,
            #[cfg(feature = "cuda")]
            graph_failures: 0,
            vision: None,
            mtp: None,
            mtp_slot_aligned: vec![true; num_slots],
            debug_alloc: None,
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
                       // Сайдкар нужен только GGUF-пути: у самостоятельного контейнера веса
                       // уже нашего формата, вторая копия в VRAM была бы бессмысленной.
        if !standalone {
            a.model
                .attach_ytf16(gguf_path, &a.device)
                .map_err(|e| anyhow!("attach_ytf16: {e}"))?;
        }
        // Q8_0-квантование сайдкара идёт через F16-тензор на GPU: после
        // сжатия его страницы остаются в driver pool (~1.2 ГиБ на 4B).
        // Trim возвращает их ОС — иначе экономия VRAM съедается слаком.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(c) = &a.device {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
        }

        if let Some(tokens) = std::env::var("KV_MIRROR_PREPARE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            if tokens > 0 {
                a.model
                    .prepare_kv_mirror(tokens.min(a.model.context_length));
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

    /// Включить снятие снимка на границе последнего чанка префила.
    pub fn set_prefix_capture(&mut self, on: bool) {
        self.capture_prefix = on;
    }

    /// Забрать checkpoint'ы границ: (позиция, состояние). Одноразово.
    pub fn take_prefix_snapshots(&mut self, slot: usize) -> Vec<(usize, StateSnapshot)> {
        self.slot_prefix_snaps
            .get_mut(slot)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Внедрить snapshot слоту (prefix-cache hit): при следующем prefill_chunk
    /// с reset_first=false модель восстановит state из этого snapshot'а.
    pub fn inject_slot_snapshot(&mut self, slot: usize, snap: StateSnapshot) {
        self.slot_snaps[slot] = Some(snap);
        self.slot_seeded[slot] = false;
        self.mtp_slot_aligned[slot] = false;
        if self.state_owner.map(|(owner, _)| owner) == Some(slot) {
            self.state_owner = None;
        }
        if self.mtp.is_some() {
            eprintln!(
                "[mtp] slot {slot}: prefix-cache snapshot has no MTP state; speculation disabled for this request"
            );
        }
    }

    /// Делегированный доступ к модели (для profiling / debug_capture).
    /// Устройство, на котором загружена модель (перенос снимков префикс-кеша).
    pub fn device(&self) -> &Device {
        &self.device
    }

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
        let mut mtp = Qwen35Mtp::load(
            gguf_path,
            self.device.clone(),
            self.slot_snaps.len(),
            &self.profile,
            self.model.shared_output(),
        )
        .map_err(|error| anyhow!("load MTP component: {error}"))?;
        // Шортлист словаря черновика: MTP_VOCAB_SHORTLIST=<файл, id по
        // строке> либо MTP_VOCAB_TOP=N (первые N id — у BPE это
        // примерно порядок частоты слияний). Не задано — полный словарь.
        // Пустое значение переменной — «нет шортлиста», а не путь: иначе
        // загрузка MTP молча падала, и контрольные замеры шли без MTP.
        let shortlist_path = std::env::var("MTP_VOCAB_SHORTLIST")
            .ok()
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty());
        let shortlist: Vec<u32> = if let Some(path) = shortlist_path {
            let text =
                std::fs::read_to_string(&path).map_err(|e| anyhow!("MTP shortlist {path}: {e}"))?;
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| {
                    l.parse::<u32>()
                        .map_err(|e| anyhow!("MTP shortlist {path}: «{l}»: {e}"))
                })
                .collect::<Result<Vec<u32>>>()?
        } else if let Some(v) = std::env::var("MTP_VOCAB_TOP")
            .ok()
            .filter(|v| !v.trim().is_empty())
        {
            // Ошибка разбора — ошибкой, не тихим полным словарём: значение с
            // пробелом или CR из батника иначе выглядело бы как «не задано».
            let n: u32 = v
                .trim()
                .parse()
                .map_err(|e| anyhow!("MTP_VOCAB_TOP=«{v}»: {e}"))?;
            (0..n).collect()
        } else {
            Vec::new()
        };
        if !shortlist.is_empty() {
            let n = shortlist.len();
            mtp.set_vocab_shortlist(shortlist)
                .map_err(|e| anyhow!("MTP vocab shortlist: {e}"))?;
            eprintln!("[mtp] словарь черновика ограничен шортлистом из {n} токенов");
        }
        // Диагностика раскладки памяти: DEBUG_ALLOC_MB=N держит лишний
        // буфер N МБ с момента загрузки MTP. Если один он меняет вывод при
        // полном словаре — где-то читается неинициализированная память, и
        // шортлист лишь сдвигал раскладку.
        if let Some(mb) = std::env::var("DEBUG_ALLOC_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
        {
            let t = Tensor::zeros(mb * 1024 * 1024, DType::U8, &self.device)
                .map_err(|e| anyhow!("debug alloc {mb} MB: {e}"))?;
            eprintln!("[mtp] debug: удерживаю лишний буфер {mb} МБ");
            self.debug_alloc = Some(t);
        }
        self.mtp = Some(mtp);
        for (slot, aligned) in self.mtp_slot_aligned.iter_mut().enumerate() {
            *aligned = self.slot_snaps[slot].is_none() && !self.slot_seeded[slot];
        }
        Ok(())
    }

    pub fn unload_mtp(&mut self) -> Result<()> {
        if self.target_transactions.iter().any(Option::is_some) {
            return Err(anyhow!("cannot unload MTP during active transaction"));
        }
        self.mtp = None;
        self.mtp_slot_aligned.fill(true);
        Ok(())
    }

    /// Полный сброс (все слоты) — только для тестов / teardown.
    #[allow(dead_code)]
    fn clear_all_state(&mut self) {
        self.state_owner = None;
        self.model.clear_state();
        self.model.clear_state_batched(&self.device);
        for s in self.slot_snaps.iter_mut() {
            *s = None;
        }
        for f in self.slot_seeded.iter_mut() {
            *f = false;
        }
        self.mtp_slot_aligned.fill(true);
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
                usize::try_from(rope_position).map_err(|_| anyhow!("negative decode RoPE position"))
            })
            .collect()
    }
}

/// Trim пула драйвера (возврат освобождённых страниц ОС). Вызов блокирующий и
/// дорогой, а в prefill-пути он попадает прямо в TTFT запроса. `YTTRI_TRIM=0`
/// выключает его для A/B — по умолчанию поведение прежнее.
#[cfg(feature = "cuda")]
fn trim_pool_cuda(device: &candle_core::Device) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("YTTRI_TRIM").map(|v| v != "0").unwrap_or(true));
    if !on {
        return;
    }
    if let candle_core::Device::Cuda(c) = device {
        let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
    }
}

/// Пиннированный host-буфер под логиты (на Ornith — 248 320 f32 = 993 КБ).
///
/// Штатный `Tensor::to_vec1()` на CUDA копирует через pageable staging: замер
/// `h2d_probe` дал 0.436 мс на этот объём против **0.091 мс** при копировании
/// в pinned-память (`cuMemHostAlloc` + `cuMemcpyDtoH`). На декоде это ~0.35 мс
/// на токен, то есть ~2 % шага. Буфер один на поток и переиспользуется.
#[cfg(feature = "cuda")]
mod pinned_logits {
    use candle_core::cuda_backend::cudarc::driver::sys as csys;
    use candle_core::{DType, Device, Result, Tensor};
    use std::cell::RefCell;
    use std::ffi::c_void;

    struct Buf {
        ptr: *mut c_void,
        bytes: usize,
    }

    impl Drop for Buf {
        fn drop(&mut self) {
            unsafe {
                let _ = csys::cuMemFreeHost(self.ptr);
            }
        }
    }

    thread_local! {
        static BUF: RefCell<Option<Buf>> = const { RefCell::new(None) };
    }

    /// F32-логиты устройства → `Vec<f32>` через pinned-буфер.
    pub fn to_vec_f32(device: &Device, t: &Tensor) -> Result<Vec<f32>> {
        let t = t.to_dtype(DType::F32)?.flatten_all()?.contiguous()?;
        let n = t.elem_count();
        let bytes = n * std::mem::size_of::<f32>();
        let src = crate::real::paged_kv_cuda::tensor_cuda_ptr(&t)?;
        let mut out = vec![0f32; n];
        BUF.with(|cell| -> Result<()> {
            let mut cell = cell.borrow_mut();
            let need_alloc = match cell.as_ref() {
                Some(b) => b.bytes < bytes,
                None => true,
            };
            if need_alloc {
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let res = unsafe { csys::cuMemHostAlloc(&mut ptr, bytes, 0) };
                if res != csys::CUresult::CUDA_SUCCESS {
                    candle_core::bail!("cuMemHostAlloc({bytes}) failed: {res:?}");
                }
                *cell = Some(Buf { ptr, bytes });
            }
            let buf = cell.as_ref().expect("буфер только что выделен");
            let res = unsafe { csys::cuMemcpyDtoH_v2(buf.ptr, src, bytes) };
            if res != csys::CUresult::CUDA_SUCCESS {
                candle_core::bail!("cuMemcpyDtoH({bytes}) failed: {res:?}");
            }
            unsafe {
                std::ptr::copy_nonoverlapping(buf.ptr as *const f32, out.as_mut_ptr(), n);
            }
            Ok(())
        })?;
        let _ = device;
        Ok(out)
    }
}

impl BatchModel for Qwen35BatchAdapter {
    /// Граница раунда для `MTP_TIMING=1` (см. `BatchModel::speculative_timing_sync`):
    /// применяется только диагностикой, поэтому в рабочем режиме не зовётся.
    fn speculative_timing_sync(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda) = &self.device {
            cuda.cuda_stream()
                .synchronize()
                .map_err(|error| anyhow!("speculative timing sync: {error:?}"))?;
        }
        Ok(())
    }

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
            return Err(anyhow!(
                "multimodal token/position lengths differ or contain no media"
            ));
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
            #[cfg(feature = "cuda")]
            self.graphs_reenable_on_admit();
            // Keep installed media for first chunk; reset only model state.
            self.state_owner = None;
            self.model.clear_state();
            self.slot_snaps[sidx] = None;
            self.slot_prefix_snaps[sidx].clear();
            self.slot_seeded[sidx] = false;
            if let Some(mtp) = self.mtp.as_mut() {
                mtp.reset_slot(sidx)
                    .map_err(|error| anyhow!("reset MTP slot {sidx} before prefill: {error}"))?;
            }
            self.mtp_slot_aligned[sidx] = true;
            #[cfg(feature = "cuda")]
            {
                self.paged_dirty[sidx] = true;
                self.prefill_path[sidx] = PrefillPath::Undecided;
                self.graph_gate_warned[sidx] = false;
            }
            if self.multimodal[sidx].is_none() {
                self.rope_deltas[sidx] = 0;
            }
        } else if self.state_owner.map(|(s, _)| s) == Some(sidx) {
            // Состояние слота всё ещё в буферах — восстанавливать нечего.
        } else if self.slot_snaps[sidx].is_some() {
            // Владелец сменился: сначала сохраняем состояние прежнего слота
            // (его снимок отложен), затем восстанавливаем своё.
            if let Some((prev, prev_pos)) = self.state_owner {
                let snap = self
                    .model
                    .snapshot_slot_state(&self.device, prev, prev_pos)
                    .map_err(|e| {
                        anyhow!("prefill snapshot (передача владения слоту {sidx}): {e}")
                    })?;
                self.slot_snaps[prev] = Some(snap);
            }
            let snap = self.slot_snaps[sidx].as_ref().unwrap();
            let pool_ok = self
                .model
                .restore_slot_state(&self.device, sidx, snap)
                .map_err(|e| anyhow!("prefill restore: {e}"))?;
            // Перезалитый пул снова авторитетен: graph-префилл продолжит
            // append с kv_len_host = cache_len снимка. Если пул недоступен
            // (не CUDA / int8) — авторитет остаётся за single-slot/batched.
            // Само поле существует только на CUDA: вне её страничного пула
            // нет, и авторитет всегда за single-slot/batched.
            #[cfg(feature = "cuda")]
            {
                self.paged_dirty[sidx] = !pool_ok;
            }
            #[cfg(not(feature = "cuda"))]
            let _ = pool_ok;
            if pool_ok {
                // K/V уже скопирован в постоянный paged pool. Не держим вторую
                // копию длиной со весь префикс во время suffix-prefill/decode.
                if let Some(snap) = self.slot_snaps[sidx].as_mut() {
                    snap.clear_attention_payloads();
                }
                #[cfg(feature = "cuda")]
                if let Device::Cuda(c) = &self.device {
                    let _ = c.cuda_stream().synchronize();
                    let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
                }
            }
        } else {
            return Err(anyhow!(
                "prefill_chunk: slot {sidx} без snapshot и не reset"
            ));
        }

        // Снимок границы для prefix cache: состояние слота здесь отвечает
        // ровно chunk.start_pos, и эта позиция кратна размеру чанка.
        let prefix_limit = prefix_cache_max_tokens();
        // Снимок на КАЖДОЙ границе чанка: именно эти позиции лежат на стыке
        // истории и генерационного суффикса, поэтому переиспользуются следую-
        // щим ходом диалога. Число снимков ограничено, чтобы host RAM не рос
        // при мелком чанке.
        let capture_boundary = prefix_cache_checkpoints_enabled()
            && chunk.start_pos > 0
            && self.slot_prefix_snaps[sidx].len() < prefix_cache_checkpoint_cap();
        let capture_final = chunk.is_final && chunk.start_pos > 0;
        let capture_position = chunk.start_pos >= prefix_cache_min_pos()
            && chunk.start_pos <= prefix_limit
            && (capture_boundary || capture_final);
        let already_captured = self.slot_prefix_snaps[sidx].iter().any(|(pos, _)| *pos == chunk.start_pos);
        if self.capture_prefix && capture_position && !already_captured {
            let device_snap = self
                .model
                .snapshot_slot_state(&self.device, sidx, chunk.start_pos)
                .map_err(|e| anyhow!("prefix snapshot: {e}"))?;
            // Переносим в RAM сразу, до forward финального чанка. Раньше
            // device-снимок границы пересекался по времени с финальным
            // slot-snapshot той же длины, удваивая пик VRAM.
            let host_snap = device_snap
                .to_host()
                .map_err(|e| anyhow!("prefix snapshot to host: {e}"))?;
            drop(device_snap);
            #[cfg(feature = "cuda")]
            {
                if let Device::Cuda(c) = &self.device {
                    let _ = c.cuda_stream().synchronize();
                }
                trim_pool_cuda(&self.device);
            }
            self.slot_prefix_snaps[sidx].push((chunk.start_pos, host_snap));
        }

        let pf_restore = pf_t0.elapsed();
        let ids = Tensor::from_vec(
            chunk.tokens.clone(),
            (1usize, chunk.tokens.len()),
            &self.device,
        )
        .map_err(|e| anyhow!("prefill ids from_vec T={}: {e}", chunk.tokens.len()))?;
        // yttri-forge: CUDA-graph prefill (PGRAPH=on|check).
        #[cfg(feature = "cuda")]
        let (pg_logits, pg_used) = self.prefill_try_graphed(chunk)?;
        #[cfg(not(feature = "cuda"))]
        let (pg_logits, pg_used): (Option<Vec<f32>>, bool) = (None, false);

        let pf_fwd0 = std::time::Instant::now();
        let (logits, mtp_inputs) = if pg_used {
            // Графовый префил посчитал hidden всех позиций — собираем такие же
            // mtp_inputs, как в eager-ветке, чтобы штатный catch_up ниже отработал.
            // Без этого MTP остаётся без префикса, черновики не принимаются, и
            // адаптивная ширина молча выключает спекуляцию (drafted=2 accepted=0).
            #[cfg(feature = "cuda")]
            let mi = match self.pg_last_hidden.take() {
                Some(hidden) if self.multimodal[sidx].is_none() && self.mtp_slot_aligned[sidx] => {
                    // Именно embed_for_graph, а не embed_tokens: при GPU_ONLY=1
                    // хостовая таблица намеренно опустошается ради освобождения
                    // mmap на весь GGUF, и её forward обязан не вызываться. Этот
                    // путь его вызывал и падал с «token_id N out of range
                    // (vocab_size=0)» — на 4B с MTP при графовом префиле.
                    let embeds = self
                        .model
                        .embed_for_graph(&chunk.tokens, &self.device)
                        .map_err(|e| anyhow!("pgraph embed for MTP: {e}"))?;
                    Some((embeds, hidden))
                }
                _ => None,
            };
            #[cfg(not(feature = "cuda"))]
            let mi = None;
            (None, mi)
        } else if let Some(media) = self.multimodal[sidx].as_mut() {
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
            (Some(logits), Some((embeds, hidden)))
        } else if self.mtp.is_some() && self.mtp_slot_aligned[sidx] {
            let embeds = self.model.embed_tokens(&ids, &self.device)?;
            // FR-011: catch_up ожидает [1, seq, H]. `ids` здесь уже rank-2
            // ([1, T]), поэтому embedding отдаёт rank-3 и лишний unsqueeze
            // давал rank-4 [1,1,T,H] — forward падал с «unexpected rank ... got: 4».
            // Приводим форму явно, не завязываясь на ранг входа.
            let embeds_3d = embeds.reshape((1usize, chunk.tokens.len(), self.model.hidden_size()))?;
            let (logits, hidden) = self
                .model
                .forward_embeds_with_hidden(&embeds_3d, chunk.start_pos)
                .map_err(|error| anyhow!("prefill forward: {error}"))?;
            (Some(logits), Some((embeds_3d, hidden)))
        } else {
            (
                Some(
                    self.model
                        .forward(&ids, chunk.start_pos)
                        .map_err(|error| anyhow!("prefill forward: {error}"))?,
                ),
                None,
            )
        };
        if self.mtp_slot_aligned[sidx] {
            if let (Some(mtp), Some((embeds, hidden))) = (self.mtp.as_mut(), mtp_inputs) {
                let rope_positions = self.multimodal[sidx]
                    .as_ref()
                    .map(|media| {
                        slice_position_plan(&media.plan, chunk.start_pos, chunk.tokens.len())
                    })
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
        }
        let pf_fwd = pf_fwd0.elapsed();
        let pf_l0 = std::time::Instant::now();
        let logits_f32 = match logits {
            Some(l) => l
                .squeeze(0)
                .map_err(|e| anyhow!("prefill logits squeeze: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("prefill logits to_dtype: {e}"))?
                .to_vec1()
                .map_err(|e| anyhow!("prefill logits to_vec1: {e}"))?,
            None => pg_logits
                .clone()
                .ok_or_else(|| anyhow!("prefill: логиты не посчитаны"))?,
        };
        let pf_logits = pf_l0.elapsed();

        // Топ-2 логита финального чанка: по ним видно, ничья ли решает выбор
        // первого токена, когда два варианта ядра дают разный текст.
        if chunk.is_final && crate::scheduler::trace_on() {
            let mut top: Vec<(usize, f32)> = logits_f32.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if top.len() >= 2 {
                eprintln!(
                    "[pfa] top1={} ({:.4}) top2={} ({:.4}) отрыв={:.4}",
                    top[0].0,
                    top[0].1,
                    top[1].0,
                    top[1].1,
                    top[0].1 - top[1].1
                );
            }
        }

        // Не-финитные логиты префилла = сэмплер выдаст мусорный первый токен.
        // Дёшево (один проход по vocab на чанк) и ловит целый класс поломок.
        if let Some(idx) = logits_f32.iter().position(|v| !v.is_finite()) {
            let bad = logits_f32.iter().filter(|v| !v.is_finite()).count();
            eprintln!(
                "[pfa] WARN non-finite logits: {bad}/{} (первый idx={idx}) T={} pos={} graph={}",
                logits_f32.len(),
                chunk.tokens.len(),
                chunk.start_pos,
                pg_used
            );
        }

        // check-режим: граф vs eager на одном и том же чанке (гейт Phase 2).
        if let (Some(g), false) = (pg_logits.as_ref(), pg_used) {
            let n = g.len().min(logits_f32.len());
            let (mut sum, mut max) = (0f64, 0f32);
            for i in 0..n {
                let d = (g[i] - logits_f32[i]).abs();
                sum += d as f64;
                max = max.max(d);
            }
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
                        if x > a.1 {
                            (i, x)
                        } else {
                            a
                        }
                    })
                    .0
            };
            let (ag, ae) = (argmax(g), argmax(&logits_f32));
            // pos обязателен: без него по логу не отличить чанк 0 (чистое
            // сравнение одного шага) от чанка 49 (накопленный дрейф двух
            // независимых историй — графовой в пуле и eager в single-slot
            // кэше). Ровно эта переменная решает «баг или дрейф», а печаталось
            // только T, и разбор 2026-09-04 упёрся в её отсутствие.
            eprintln!(
                "[pg] parity pos={} T={} len={n} mae={:.3e} max={max:.3e} argmax g={ag}({:.2}/{:.2}) e={ae}({:.2}/{:.2}) {}",
                chunk.start_pos,
                chunk.tokens.len(),
                sum / n.max(1) as f64,
                g[ag],
                logits_f32[ag],
                g[ae],
                logits_f32[ae],
                if ag == ae { "OK" } else { "MISMATCH" }
            );
        }

        // Промежуточные чанки снимок НЕ делают: состояние остаётся в буферах,
        // владелец — этот слот. Снимок берётся лениво — либо на финальном чанке
        // (нужен для seed), либо при передаче владения другому слоту.
        let new_pos = chunk.start_pos + chunk.tokens.len();
        self.state_owner = Some((sidx, new_pos));
        let pf_s0 = std::time::Instant::now();
        if chunk.is_final {
            // При graph-prefill attention K/V уже находится в paged pool.
            // Полный снимок создавал ещё одну копию, линейную по контексту,
            // хотя seed использовал из неё только recurrent-state и длину.
            let snap = if pg_used {
                self.model
                    .snapshot_slot_recurrent_state(&self.device, new_pos)
                    .map_err(|e| anyhow!("prefill recurrent snapshot: {e}"))?
            } else {
                self.model
                    .snapshot_slot_state(&self.device, sidx, new_pos)
                    .map_err(|e| anyhow!("prefill snapshot: {e}"))?
            };
            self.slot_snaps[sidx] = Some(snap);
            self.model
                .seed_slot_batched(
                    &self.device,
                    sidx,
                    self.slot_snaps[sidx]
                        .as_ref()
                        .ok_or_else(|| anyhow!("prefill snapshot disappeared"))?,
                )
                .map_err(|error| anyhow!("prefill seed slot {sidx}: {error}"))?;
            #[cfg(feature = "cuda")]
            if pg_used {
                // Attention payload намеренно отсутствует: пул уже содержит
                // строки, но host-зеркало длины нужно eager fallback/checkpoint.
                self.model.set_kv_len_batched(sidx, new_pos);
            }
            self.slot_seeded[sidx] = true;
            // Владение снимается: ниже KV single-slot пути очищается, и
            // пропускать restore для следующего чанка этого слота нельзя.
            self.state_owner = None;
            // Single-slot F16 KV больше не нужен (decode через batched q8):
            // освобождаем ~480 MiB @24K, иначе карта уходит в 97%+ и декод
            // падает в WDDM shared (обрыв 6K→12K, 2026-08-23).
            if std::env::var("KEEP_SINGLE_KV").as_deref() != Ok("1") {
                self.model.clear_single_slot_kv();
            }
            // Trim CUDA memory pool: prefill оставил пиковые F16/KV транзиенты
            // в driver pool (release threshold=512 MiB). Trim возвращает ОС
            // страницы сверх текущего usage → VRAM освобождается для decode.
            #[cfg(feature = "cuda")]
            {
                if let Device::Cuda(c) = &self.device {
                    let _ = c.cuda_stream().synchronize();
                }
                trim_pool_cuda(&self.device);
            }
        } else {
            self.slot_seeded[sidx] = false;
        }

        #[cfg(feature = "cuda")]
        if pg_used {
            // KV чанка записан прямо в paged pool: миграция из batched-кэша
            // (он пуст после graph-префилла) затёрла бы его.
            self.paged_dirty[sidx] = false;
            self.prefill_path[sidx] = PrefillPath::Paged;
            // Отмечаем длину: batched-кэша нет, но пул её знает, и обратная
            // миграция должна знать, сколько восстанавливать, если понадобится
            // eager-путь.
            self.model
                .set_kv_len_batched(sidx, chunk.start_pos + chunk.tokens.len());
        } else if pg_logits.is_none() {
            // Paged-проход не исполнялся (гейты/ошибка): чанк ушёл в обычный
            // eager single-slot forward, строк этого чанка в пуле нет —
            // авторитет single-slot/batched. Случай pg_logits.is_some() без
            // pg_used — paged-прогрев без захвата (t < порога): пул УЖЕ
            // содержит строки чанка, трогать dirty нельзя.
            self.paged_dirty[sidx] = true;
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
        if batch
            .items
            .iter()
            .any(|item| item.slot_idx >= self.slot_snaps.len())
        {
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
                    // Seed из pool-снимка (from_pool) НЕ меняет содержимое пула:
                    // attention не материализуется в batched-кэш, пул остаётся
                    // авторитетным. Seed из single-slot-снимка кладёт строки в
                    // batched-кэш, а в пуле их нет → пул устарел (dirty).
                    let pool_authoritative = self.model.paged_ctx.is_some()
                        && snap.blocks.iter().all(|b| match b {
                            BlockStateSnap::Attention(Some(s)) => s.from_pool,
                            _ => true,
                        });
                    self.paged_dirty[sidx] = !pool_authoritative;
                    // Seed меняет состав state — graph состав-зависим, инвалидируем.
                    if !self.decode_graphs.is_empty() {
                        self.decode_graphs.clear();
                    }
                    // Prefill оставил в пуле пиковые страницы интермедиатов
                    // (512-token chunk buffers). Trim при первом decode после
                    // prefill — иначе retained slack добивает VRAM на 27B.
                    trim_pool_cuda(&self.device);
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
        // Первая CUDA-операция шага (выделение + htod) — с контекстом: после
        // захвата и вытеснения графа сюда приходил голый INVALID_VALUE.
        let ids = Tensor::from_vec(tokens.clone(), (b, 1usize), &self.device)
            .map_err(|e| anyhow!("decode ids from_vec b={b}: {e}"))?;

        // FR-006/PD-004: кэш горячих экспертов — D2H следа прошлого шага,
        // LRU, подъёмы промахов на боковом потоке (перед графом шага).
        #[cfg(feature = "cuda")]
        let dec_t0 = std::time::Instant::now();
        #[cfg(feature = "cuda")]
        self.model
            .before_moe_step(&self.device)
            .map_err(|e| anyhow!("moe before_step: {e}"))?;
        #[cfg(feature = "cuda")]
        let dec_t1 = std::time::Instant::now();
        // CUDA-graph decode: один cuGraphLaunch на весь forward. None → eager.
        #[cfg(feature = "cuda")]
        {
            let graphed = self.decode_batch_graphed(b, &tokens, &rope_positions, &slots, &positions);
            if crate::scheduler::trace_on() {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n % 32 == 0 {
                    eprintln!(
                        "[dec] #{n} before_step={:.2}ms graphed={:.2}ms",
                        (dec_t1 - dec_t0).as_secs_f64() * 1e3,
                        dec_t1.elapsed().as_secs_f64() * 1e3
                    );
                }
            }
            match graphed {
                Ok(Some((out, _hidden))) => {
                    for i in 0..b {
                        let sidx = slot_order[i];
                        if let Some(snap) = self.slot_snaps[sidx].as_mut() {
                            snap.position = positions[i] + 1;
                        }
                        // Хостовое зеркало длины batched-кеша: по нему
                        // rehydrate_kv_from_paged собирает кеш из пула при
                        // eager-откате и checkpoint_slot_batched снимает длину.
                        // Без этого после графовых шагов оба видели бы длину
                        // на момент миграции.
                        self.model.set_kv_len_batched(sidx, positions[i] + 1);
                    }
                    return Ok(out);
                }
                Ok(None) => {}
                Err(e) => {
                    self.decode_graphs.clear();
                    self.graphs_failed("graphed decode failed, eager fallback", &e);
                }
            }
        }
        // Hidden decode_batch'а больше не нужен MTP: verify идёт через
        // speculative_verify (multi-token), hidden собирает speculative_accept.
        // Авторитетная копия KV может жить только в пуле: после миграции
        // batched-кэш освобождён, а после graph-префилла его и не было.
        // Eager-путь без кэша не работает — собираем его из пула. Ничего не
        // делает, если кэш на месте.
        #[cfg(feature = "cuda")]
        for it in batch.items.iter() {
            self.model
                .rehydrate_kv_from_paged(it.slot_idx)
                .map_err(|e| anyhow!("rehydrate KV slot {}: {e}", it.slot_idx))?;
            self.prefill_path[it.slot_idx] = PrefillPath::Eager;
        }
        let logits = self
            .model
            .forward_decode_batch(&ids, &positions, &rope_positions, &slots)
            .map_err(|e| anyhow!("decode_batch forward: {e}"))?;
        // One D2H transfer for [B, vocab], then split on host. Per-row to_vec1()
        // serialized four CUDA synchronizations/copies for B=4.
        let flat = logits
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("decode_batch logits to_dtype: {e}"))?
            .flatten_all()
            .map_err(|e| anyhow!("decode_batch logits flatten: {e}"))?
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
            && self.mtp_slot_aligned[slot]
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
                .map_err(|restore| {
                    anyhow!("MTP begin failed: {error}; target restore: {restore}")
                })?;
            return Err(anyhow!("MTP begin: {error}"));
        }
        self.transaction_snapshot_positions[slot] = self.slot_snaps[slot]
            .as_ref()
            .map(|snapshot| snapshot.position);
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
        let slots = vec![slot as u32; k];
        #[cfg(feature = "cuda")]
        let (flat, hidden, shadow_written) = if self.paged_authority(slot) {
            // Пул — единственная копия KV слота: проверка идёт по нему графом.
            // k строк одного слота — это префил-чанк длины k (append k позиций
            // подряд + FA2 varlen с причинной маской), а НЕ декод B=k: у
            // декодного пути одна позиция на слот, и k строк легли бы в одну —
            // на этом сорвалась попытка 2026-08-27 через декодный граф.
            // Уходить на eager здесь нельзя: batched-кеша нет, а его сборка из
            // пула развела бы два хранилища, и следующий графовый шаг видел бы
            // устаревшую длину.
            let cur = self.model.paged_ctx.as_ref().unwrap().kv_len_host[slot] as usize;
            let window = self.model.paged_window();
            if cur != pos || pos + k > window {
                return Err(anyhow!(
                    "speculative verify: позиция {pos} против длины пула {cur}, окно {window}"
                ));
            }
            let rope_host: Vec<u32> = rope_positions.iter().map(|&p| p as u32).collect();
            let (flat, hidden, _) = self.paged_graph_run(true, slot, pos, inputs, rope_host)?;
            self.model.paged_ctx.as_mut().unwrap().kv_len_host[slot] = (pos + k) as u32;
            // Графовый путь зовёт forward_verify_paged с shadow=true: при k>1
            // теневые снимки записаны, откат может на них опереться.
            (flat, hidden, k > 1)
        } else {
            let (flat, hidden) =
                self.verify_eager(inputs, &cache_positions, &rope_positions, &slots)?;
            (flat, hidden, false)
        };
        #[cfg(not(feature = "cuda"))]
        let (flat, hidden, shadow_written) = {
            let (flat, hidden) =
                self.verify_eager(inputs, &cache_positions, &rope_positions, &slots)?;
            (flat, hidden, false)
        };
        self.verify_pending[slot] = Some(PendingVerify {
            inputs: inputs.to_vec(),
            pos,
            hidden,
            shadow_written,
        });
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
            return Err(anyhow!(
                "speculative accept {consumed} exceeds verified {k}"
            ));
        }
        #[cfg(feature = "cuda")]
        let paged = self.paged_authority(slot);
        #[cfg(not(feature = "cuda"))]
        let paged = false;
        if consumed < k {
            // Снимки пишет только графовая проверка (`forward_verify_paged`,
            // shadow=true при b>1). Построчный `verify_eager` их не пишет, и
            // без этой проверки restore_slot_from_shadow подсунул бы состояние
            // чужого раунда.
            if paged && consumed > 0 && pending.shadow_written {
                // Ф3: без перепрогона. DeltaNet возвращается к теневому снимку
                // после строки consumed-1 — он снят внутри графа проверки
                // (delta_rule_batched_cuda::dispatch_delta_rule_batched_seq,
                // shadow_rows). Строки 0..consumed первой проверки бит-в-бит
                // равны перепрогону: их входы зависят только от префикса и самих
                // себя. Attention: строки pos..pos+consumed в пуле верны, откат
                // отвергнутых — сдвиг длины (FR-007). Раньше здесь был restore на
                // начало раунда и повторный графовый прогон принятых строк по
                // цене целой проверки в ~27% раундов.
                #[cfg(feature = "cuda")]
                {
                    self.model
                        .restore_slot_from_shadow(slot, consumed - 1)
                        .map_err(|e| anyhow!("speculative accept shadow restore: {e}"))?;
                    let ctx = self.model.paged_ctx.as_mut().unwrap();
                    let mut lens = ctx.kv_len_host.clone();
                    lens[slot] = (pending.pos + consumed) as u32;
                    ctx.reset_kv_len(&lens)
                        .map_err(|e| anyhow!("accept shadow reset_kv_len: {e}"))?;
                }
            } else {
                let checkpoint = self.target_transactions[slot]
                    .as_ref()
                    .ok_or_else(|| anyhow!("MTP transaction is not active for slot {slot}"))?;
                self.model
                    .restore_slot_batched(&self.device, checkpoint)
                    .map_err(|error| anyhow!("speculative accept restore: {error}"))?;
                if paged {
                    // consumed == 0: длина пула на начало раунда.
                    #[cfg(feature = "cuda")]
                    {
                        let ctx = self.model.paged_ctx.as_mut().unwrap();
                        let mut lens = ctx.kv_len_host.clone();
                        lens[slot] = pending.pos as u32;
                        ctx.reset_kv_len(&lens)
                            .map_err(|e| anyhow!("accept reset_kv_len: {e}"))?;
                    }
                } else if consumed > 0 {
                    #[cfg(feature = "cuda")]
                    self.model
                        .rehydrate_kv_from_paged(slot)
                        .map_err(|e| anyhow!("speculative accept rehydrate: {e}"))?;
                    // Построчно, а не одним батчем: геометрия ядра матвека
                    // зависит от размера батча, поэтому перепрогон пачкой
                    // оставлял бы состояние, не равное состоянию обычного
                    // декода — та же причина, что и у батчевой проверки
                    // (см. verify_eager). Наружу это выглядело как выпавший
                    // токен: «index. html» вместо «index.html».
                    let rope_positions = self.rope_positions_for(slot, pending.pos, consumed)?;
                    for row in 0..consumed {
                        let ids = Tensor::from_vec(
                            vec![pending.inputs[row]],
                            (1usize, 1usize),
                            &self.device,
                        )?;
                        let _ = self
                            .model
                            .forward_decode_batch(
                                &ids,
                                &[pending.pos + row],
                                &rope_positions[row..=row],
                                &[slot as u32],
                            )
                            .map_err(|error| anyhow!("speculative accept re-run: {error}"))?;
                    }
                }
            }
        }
        #[cfg(feature = "cuda")]
        if paged {
            // Зеркало длины batched-кеша (см. decode_batch): restore вернул
            // длину на начало раунда, а eager-откат и следующий checkpoint
            // должны видеть длину после принятых строк.
            self.model.set_kv_len_batched(slot, pending.pos + consumed);
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
                // Проверка могла дописать k строк в пул до срыва раунда —
                // длина слота возвращается на начало раунда (FR-007).
                #[cfg(feature = "cuda")]
                if self.paged_authority(slot) {
                    let ctx = self.model.paged_ctx.as_mut().unwrap();
                    let mut lens = ctx.kv_len_host.clone();
                    lens[slot] = position as u32;
                    ctx.reset_kv_len(&lens)
                        .map_err(|e| anyhow!("rollback reset_kv_len: {e}"))?;
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
        if self.state_owner.map(|(owner, _)| owner) == Some(idx) {
            self.state_owner = None;
        }
        self.multimodal[idx] = None;
        self.rope_deltas[idx] = 0;
        self.target_transactions[idx] = None;
        self.verified_target_hidden[idx].clear();
        self.verify_pending[idx] = None;
        self.transaction_snapshot_positions[idx] = None;
        // Отдать видеопамять слота сразу, а не держать до конца процесса.
        // Буфер имел размер самого длинного запроса на этом слоте; на карте
        // 12 ГБ два запроса на 30K подряд съедали её до отказа.
        //
        // На страничном пути это пусто: KV живёт в пуле, batched-кэша нет.
        // Возвращать страницы драйверу через trim здесь пробовали — он отдаёт
        // ровно ноль (замер: свободно 685 → 685 МиБ, и с синхронизацией потока
        // тоже). Свободных страниц в пуле нет: удерживаемая память — это сам
        // пул, живой и занятый. Поэтому trim убран, осталось освобождение.
        // Бисекция 28.08.2026: падение на четырёх слотах воспроизводится и
        // без этой строки, значит виновата не она, а графы. Возвращено.
        self.model.free_slot_kv_batched(idx);
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset_slot(idx)
                .map_err(|error| anyhow!("reset MTP slot {idx}: {error}"))?;
        }
        self.mtp_slot_aligned[idx] = true;
        Ok(())
    }
}

impl Qwen35BatchAdapter {
    /// Обёртка над `prefill_chunk_graphed`: снимает snapshot для check-режима,
    /// глотает ошибку графа (PD-205 — чанк уходит в eager, графы остаются), но
    /// только пока промпт ещё не пошёл по пулу.
    /// Возвращает (логиты графа, использовать_ли_их_как_результат).
    #[cfg(feature = "cuda")]
    fn prefill_try_graphed(&mut self, chunk: &PrefillChunk) -> Result<(Option<Vec<f32>>, bool)> {
        #[cfg(feature = "cuda")]
        let moe_ram = self.model.experts_ram();
        #[cfg(feature = "cuda")]
        let mode = pgraph_mode_effective(moe_ram);
        #[cfg(not(feature = "cuda"))]
        let mode = pgraph_mode();
        // FR-007: при выгрузке эффективный режим Off (захвата префила нет),
        // но пейджед-прогрев обязан выполняться (KV в пуле) — потому bail
        // только вне выгрузки; внутренний гейт решает остальное.
        if mode == PgraphMode::Off && !moe_ram {
            self.prefill_path[chunk.slot_idx] = PrefillPath::Eager;
            return Ok((None, false));
        }
        let pre = if mode == PgraphMode::Check {
            Some(
                self.model
                    .snapshot_state(&self.device, chunk.start_pos)
                    .map_err(|e| anyhow!("pg pre-snapshot: {e}"))?,
            )
        } else {
            None
        };
        let logits = match self.prefill_chunk_graphed(chunk) {
            Ok(v) => v,
            // PD-205 действует, пока откат на eager безопасен. После первого
            // paged-чанка он уже не безопасен: префикс лежит только в пуле,
            // и eager досчитал бы промпт без него.
            Err(e) if self.prefill_path[chunk.slot_idx] != PrefillPath::Paged => {
                eprintln!("[pg] ошибка, чанк уходит в eager: {e}");
                self.prefill_path[chunk.slot_idx] = PrefillPath::Eager;
                None
            }
            Err(e) => {
                return Err(anyhow!(
                    "paged prefill слота {} прервался, когда промпт уже шёл по пулу: {e}",
                    chunk.slot_idx
                ))
            }
        };
        if logits.is_none() {
            return Ok((None, false));
        }
        match pre {
            // check: откатываем state на дографовый — эталон считает eager-путь.
            Some(pre) => {
                self.model
                    .restore_state(&self.device, &pre)
                    .map_err(|e| anyhow!("pg restore: {e}"))?;
                Ok((logits, false))
            }
            None => Ok((logits, true)),
        }
    }

    /// Гейт увёл чанк с графового пути на eager.
    ///
    /// До первого paged-чанка это законно: весь промпт пойдёт eager. После —
    /// нет: строки предыдущих чанков лежат только в пуле, single-slot кэш пуст,
    /// и eager посчитал бы внимание вообще без префикса. Раньше здесь был
    /// молчаливый `Ok(None)`, и такой промпт досчитывался мусором.
    #[cfg(feature = "cuda")]
    fn prefill_gate(&mut self, slot: usize, reason: &str) -> Result<Option<Vec<f32>>> {
        if !eager_fallback_allowed(self.prefill_path[slot]) {
            return Err(anyhow!(
                "paged prefill: слот {slot} уже считает промпт по страничному пулу, \
                 а этот чанк уходит на eager ({reason}). Single-slot кэш пуст — \
                 eager потерял бы весь префикс промпта. Для этой конфигурации \
                 задайте PGRAPH=off: весь префил пойдёт eager с первого чанка"
            ));
        }
        if self.prefill_path[slot] == PrefillPath::Undecided {
            eprintln!("[pg] слот {slot}: префил идёт eager ({reason})");
        }
        self.prefill_path[slot] = PrefillPath::Eager;
        Ok(None)
    }

    /// CUDA-graph prefill одного чанка: один cuGraphLaunch вместо ~1000 мс
    /// хостовых launch'ей. `Ok(None)` — чанк не graphable (гейты), вызывающий
    /// идёт eager. Ошибка внутри → eager этот чанк, графы НЕ выключаются
    /// (PD-205: отличие от decode).
    ///
    /// Вне графа стейджится вся динамика: ids, позиции RoPE, kv_len[slot],
    /// seqlens_q=[0,T], block_table. Внутри — только device-операции.
    #[cfg(feature = "cuda")]
    fn prefill_chunk_graphed(&mut self, chunk: &PrefillChunk) -> Result<Option<Vec<f32>>> {
        let slot = chunk.slot_idx;
        #[cfg(feature = "cuda")]
        let moe_ram = self.model.experts_ram();
        #[cfg(feature = "cuda")]
        // FR-007: захвата графового префила при выгрузке нет (WARN уже дан),
        // но KV обязан идти в paged pool — иначе миграция int8 запрещена и
        // графы декода не захватятся. Поэтому при выгрузке пейджед-прогрев
        // без захвата выполняется независимо от PGRAPH.
        if pgraph_mode_effective(moe_ram) == PgraphMode::Off && !moe_ram {
            self.prefill_path[slot] = PrefillPath::Eager;
            return Ok(None);
        }
        let Device::Cuda(cuda_dev) = &self.device else {
            return Ok(None);
        };
        let t = chunk.tokens.len();
        // Промпт уже пошёл по eager — назад дороги нет: строки посчитанных
        // чанков в пул не попали, и append оставил бы там дыру.
        // чанков в пул не попали, и append оставил бы там дыру.
        if self.prefill_path[slot] == PrefillPath::Eager {
            return Ok(None);
        }
        // PD-204: MTP из гейта снят 2026-08-27 — проверка спекуляции идёт по
        // страничному пулу (`speculative_verify` → `paged_graph_run`), поэтому
        // загруженные веса MTP больше не отключают графовый префил всему
        // движку. Vision остаётся: mrope требует своих позиций, которых у
        // графа нет.
        if self.multimodal[slot].is_some() {
            return self.prefill_gate(slot, "multimodal");
        }
        // graphs_enabled гаснет при сбое графа в любом слоте (OQ-8) — то есть
        // ровно посреди чужого промпта.
        if !self.graphs_enabled {
            return self.prefill_gate(slot, "графы выключены после сбоя");
        }
        self.model
            .init_paged_decode(&self.device)
            .map_err(|e| anyhow!("pgraph init paged: {e}"))?;
        let window = self.model.paged_window();
        if window == 0 {
            return self.prefill_gate(slot, "страничный пул не создан");
        }
        // Окно пула = min(VRAM, окно модели, CTX). Когда оно меньше промпта,
        // слоту физически некуда писать хвост: его страницы — ровно `window`
        // токенов. Это конфигурация, а не сбой — движок уже печатает
        // «WARN: окно меньше контекста» при создании пула.
        if chunk.start_pos + t > window {
            return self.prefill_gate(slot, &format!("чанк за окном пула ({window})"));
        }

        let t_run = std::time::Instant::now();
        let rope_host: Vec<u32> = (chunk.start_pos..chunk.start_pos + t)
            .map(|p| p as u32)
            .collect();
        let (flat, hidden, hit_flag) =
            self.paged_graph_run(false, slot, chunk.start_pos, &chunk.tokens, rope_host)?;

        // hidden отдаём наружу, а не вызываем catch_up здесь: пусть графовая
        // ветка prefill_chunk заполнит mtp_inputs тем же способом, что и eager,
        // и оба пути сойдутся в одну точку вызова.
        self.pg_last_hidden = Some(hidden);

        // Device kv_len продвинут ядром на +T — синхронизируем хостовое зеркало.
        self.model.paged_ctx.as_mut().unwrap().kv_len_host[slot] = (chunk.start_pos + t) as u32;
        eprintln!(
            "[pg] chunk T={t} slot={slot} pos={} hit={} lru={} run={:.1}ms",
            chunk.start_pos,
            u8::from(hit_flag),
            self.prefill_graphs.len(),
            t_run.elapsed().as_secs_f64() * 1e3,
        );
        Ok(Some(flat))
    }

    /// Проверка спекуляции по batched-кешу (eager): k строк одного слота одним
    /// multi-token forward — батчевая ось как ось позиций, те же decode-ядра,
    /// что и k одиночных шагов. Возвращает (плоские F32-логиты, hidden [k,H]).
    fn verify_eager(
        &mut self,
        inputs: &[u32],
        cache_positions: &[usize],
        rope_positions: &[usize],
        slots: &[u32],
    ) -> Result<(Vec<f32>, Tensor)> {
        // После отказа графов batched-кеш мог быть освобождён миграцией —
        // собрать обратно (пусто, если кеш на месте). Без этого
        // forward_attn_decode_batch отказывает, а планировщик до 2026-08-27
        // глотал отказ молча — так MTP «не принимался» при графовом префиле.
        #[cfg(feature = "cuda")]
        self.model
            .rehydrate_kv_from_paged(slots[0] as usize)
            .map_err(|e| anyhow!("speculative verify rehydrate: {e}"))?;

        // Строки идут по одной, а не одним батчем: геометрия запуска MMVQ
        // зависит от размера батча (`mul_mat_vec_via_q8_1`: b_size=1 даёт
        // (nrows,4), 2..=4 даёт (nrows/2,4)), а разная раскладка потоков меняет
        // порядок суммирования — сложение f32 не ассоциативно. Одиночный декод
        // всегда идёт b_size=1, поэтому батчевая проверка возвращала ДРУГИЕ
        // логиты: замер qwen35_verify_rows дал max|Δlogit| до 6.5 при зазоре
        // 1.2, что переворачивало argmax и роняло токен из выдачи (наружу это
        // выглядело как «index. html» вместо «index.html»). Построчно
        // расхождение ровно нулевое.
        //
        // Цена: k запусков вместо одного. Веса всё равно читаются из кеша L2,
        // а спекуляция выигрывает на пропущенных шагах декода, не на батче
        // проверки. Откат — VERIFY_BATCHED=1.
        let batched = std::env::var("VERIFY_BATCHED").as_deref() == Ok("1");
        if batched || inputs.len() == 1 {
            let ids = Tensor::from_vec(inputs.to_vec(), (inputs.len(), 1usize), &self.device)?;
            let (logits, hidden) = self
                .model
                .forward_decode_batch_with_hidden(&ids, cache_positions, rope_positions, slots)
                .map_err(|error| anyhow!("speculative verify forward: {error}"))?;
            let flat = logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
                .map_err(|error| anyhow!("speculative verify logits: {error}"))?;
            return Ok((flat, hidden));
        }

        let mut flat: Vec<f32> = Vec::new();
        let mut hidden_rows: Vec<Tensor> = Vec::with_capacity(inputs.len());
        for row in 0..inputs.len() {
            let ids = Tensor::from_vec(vec![inputs[row]], (1usize, 1usize), &self.device)?;
            let (logits, hidden) = self
                .model
                .forward_decode_batch_with_hidden(
                    &ids,
                    &cache_positions[row..=row],
                    &rope_positions[row..=row],
                    &slots[row..=row],
                )
                .map_err(|error| anyhow!("speculative verify forward: {error}"))?;
            flat.extend(
                logits
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .map_err(|error| anyhow!("speculative verify logits: {error}"))?,
            );
            hidden_rows.push(hidden);
        }
        // Hidden склеиваем в ту же форму [k, H], что даёт батчевый путь:
        // speculative_accept читает строки по индексу.
        let hidden = Tensor::cat(&hidden_rows, 0)
            .map_err(|e| anyhow!("speculative verify hidden concat: {e}"))?;
        Ok((flat, hidden))
    }

    /// Хранилище KV слота — страничный пул (графы включены, пул актуален),
    /// а не batched-кеш. Выбирает путь проверки спекуляции и отката.
    ///
    /// ВНИМАНИЕ: графовая проверка (`paged_graph_run` с `verify=true`) считает
    /// k строк одним прогоном, и её логиты не совпадают с одиночным декодом —
    /// геометрия ядра матвека зависит от размера батча. Построчная починка в
    /// `verify_eager` этот путь НЕ покрывает. Замер хешами при temperature=0
    /// (Ornith Q6_K, int8-пул, графы вкл): MTP даёт 85acedb03922 против
    /// эталона 31c86d896851.
    ///
    /// Просто запретить этот путь нельзя: eager требует `rehydrate_kv_from_paged`,
    /// который при int8-пуле отказывает, и спекуляция падает до accepted=0
    /// (замер: 0 из 798 драфтов, 1368 срывов проверки). Чинить надо сам
    /// графовый путь — построчным режимом либо порядком редукции в ядре.
    #[cfg(feature = "cuda")]
    fn paged_authority(&self, slot: usize) -> bool {
        self.graphs_enabled && self.model.paged_ctx.is_some() && !self.paged_dirty[slot]
    }

    /// Прогон графа на страничном пути для T токенов одного слота: префил-чанк
    /// (`verify=false`, пул `prefill_graphs`) или проверка спекуляции
    /// (`verify=true`, пул `verify_graphs`). Вне графа стейджится вся динамика:
    /// ids, позиции RoPE, kv_len[slot]=start_pos, seqlens_q=[0,T], block_table.
    /// Граф ищется по ключу (T, slot); при промахе — прогрев eager-проходом
    /// (он же результат) и захват. Возвращает (плоские F32-логиты, hidden,
    /// попадание в пул): для префила логиты последней позиции и hidden [1,T,H],
    /// для проверки — логиты всех T строк и hidden [T,H]. hidden — всегда
    /// глубокая копия: внешний буфер графа перезаписывается следующим launch,
    /// а проверка держит его между вызовами (PendingVerify, MTP commit).
    /// После вызова device kv_len[slot] = start_pos + T; хостовое зеркало
    /// обновляет вызывающий.
    #[cfg(feature = "cuda")]
    fn paged_graph_run(
        &mut self,
        verify: bool,
        slot: usize,
        start_pos: usize,
        tokens: &[u32],
        rope_host: Vec<u32>,
    ) -> Result<(Vec<f32>, Tensor, bool)> {
        let Device::Cuda(cuda_dev) = &self.device else {
            return Err(anyhow!("paged graph: не CUDA-устройство"));
        };
        let t = tokens.len();
        let kind = if verify { "verify" } else { "prefill" };
        // Слот владеет страницами [slot*mb .. (slot+1)*mb).
        let block_table = {
            let ctx = self.model.paged_ctx.as_ref().unwrap();
            let mb = ctx.max_blocks as u32;
            (0..mb).map(|j| slot as u32 * mb + j).collect::<Vec<u32>>()
        };
        let slot_u = slot as u32;
        // Однопроходная проверка (VERIFY_ONEPASS) читает seqlens_q
        // как [0, k*ngroups] (строки запроса свёрнуты по GQA-группам);
        // построчный и слитый пути этот буфер не читают, префилльный —
        // читает, но у него verify=false и стейджится обычное t.
        // (Считается до блока с ctx: там paged_ctx занят mut-заёмом.)
        let q_rows = if verify {
            t * self.model.attn_kv_groups()
        } else {
            t
        };
        {
            let ctx = self.model.paged_ctx.as_mut().unwrap();
            // rope_pos_t декода здесь не используется (позиции идут своим
            // буфером [T]), но slots/block_table нужны append-ядру.
            ctx.stage_inputs(&[slot as u32], &[start_pos], &block_table)
                .map_err(|e| anyhow!("pgraph {kind} stage_inputs slot={slot}: {e}"))?;
            let mut lens = ctx.kv_len_host.clone();
            lens[slot] = start_pos as u32;
            ctx.reset_kv_len(&lens)
                .map_err(|e| anyhow!("pgraph {kind} reset_kv_len: {e}"))?;
            ctx.set_prefill_seqlens_q(q_rows)
                .map_err(|e| anyhow!("pgraph {kind} seqlens_q: {e}"))?;
        }
        let emb_shape = if verify {
            (t, 1usize, self.model.hidden_size())
        } else {
            (1usize, t, self.model.hidden_size())
        };
        let fwd = |m: &mut ModelWeights, emb: &Tensor, rope: &Tensor| -> Result<(Tensor, Tensor)> {
            let r = if verify {
                m.forward_verify_graphed(emb, rope, t, slot_u)
            } else {
                m.forward_prefill_graphed(emb, rope, t)
            };
            r.map_err(|e| anyhow!("pgraph {kind} forward: {e}"))
        };
        let pool = if verify {
            &mut self.verify_graphs
        } else {
            &mut self.prefill_graphs
        };
        let hit = pool.iter().position(|g| g.t == t);
        let pool_full = pool.len() >= pgraph_lru();
        match hit {
            Some(i) => {
                {
                    let g = &pool[i];
                    let emb_staging = self
                        .model
                        .embed_for_graph(tokens, &self.device)
                        .map_err(|e| anyhow!("pgraph {kind} embed: {e}"))?
                        .reshape(emb_shape)
                        .map_err(|e| anyhow!("pgraph {kind} embed reshape: {e}"))?;
                    g.emb_t
                        .slice_set(&emb_staging, 0, 0)
                        .map_err(|e| anyhow!("pgraph {kind} emb slice_set: {e}"))?;
                    let rope_staging = Tensor::from_vec(rope_host, t, &Device::Cpu)
                        .map_err(|e| anyhow!("pgraph {kind} rope from_vec: {e}"))?
                        .to_device(&self.device)
                        .map_err(|e| anyhow!("pgraph {kind} rope to_device: {e}"))?;
                    g.rope_pos_t
                        .slice_set(&rope_staging, 0, 0)
                        .map_err(|e| anyhow!("pgraph {kind} rope slice_set: {e}"))?;
                    g.launch()
                        .map_err(|e| anyhow!("pgraph {kind} launch T={t} slot={slot}: {e}"))?;
                }
                // LRU: свежий — в хвост.
                let g = pool.remove(i);
                let flat = g
                    .logits_t
                    .to_dtype(DType::F32)
                    .map_err(|e| anyhow!("pgraph {kind} logits to_dtype (первая синхронная точка после launch): {e}"))?
                    .flatten_all()
                    .map_err(|e| anyhow!("pgraph {kind} logits flatten: {e}"))?
                    .to_vec1::<f32>()
                    .map_err(|e| anyhow!("pgraph {kind} logits read: {e}"))?;
                let hidden =
                    Tensor::zeros(g.hidden_t.shape().clone(), g.hidden_t.dtype(), &self.device)
                        .map_err(|e| anyhow!("pgraph {kind} hidden zeros: {e}"))?;
                hidden
                    .slice_set(&g.hidden_t, 0, 0)
                    .map_err(|e| anyhow!("pgraph {kind} hidden copy: {e}"))?;
                pool.push(g);
                Ok((flat, hidden, true))
            }
            None => {
                // htod-кэш: промах параметров при захвате → явная ошибка вместо
                // pageable memcpy внутри графа.
                // Эмбеддинги — до guard: буфер больше порога кэша htod.
                let emb_t = self
                    .model
                    .embed_for_graph(tokens, &self.device)?
                    .reshape(emb_shape)?;
                let _htod_guard = cuda_dev.enable_cuda_graph_htod_cache();
                let rope_pos_t = Tensor::from_vec(rope_host, t, &self.device)?;
                // 1) Eager-прогон: это и есть результат (захват ядра не
                //    исполняет) + прогрев ядер/htod-кэша.
                let (logits, prime_hidden) = fwd(&mut self.model, &emb_t, &rope_pos_t)?;
                let hid_shape = prime_hidden.shape().clone();
                let hid_dtype = prime_hidden.dtype();
                let flat = logits
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .map_err(|e| anyhow!("pgraph {kind} prime logits: {e}"))?;
                let out_shape = logits.shape().clone();
                let out_dtype = logits.dtype();
                drop(logits);
                // Хвост короче порога: результат уже посчитан прогревочным
                // проходом, KV в пуле, состояние консистентно — захват не нужен.
                // FR-007: при выгрузке экспертов захват подавлен всегда —
                // каждый чанк идёт пейджед-проходом (KV в пуле, MoE-слои через
                // стейджинг), декод-графы захватываются без миграции int8.
                if !verify && (t < pgraph_min_capture_t() || self.model.experts_ram()) {
                    return Ok((flat, prime_hidden, false));
                }
                // Пул полон — новую форму не захватываем: вытеснение роняло
                // следующий шаг (см. комментарий у Drop).
                if pool_full {
                    return Ok((flat, prime_hidden, false));
                }
                let stream = cuda_dev.cuda_stream();
                let captured = (|| -> Result<PrefillGraphState> {
                    use cudarc::driver::{result as cres, sys as csys};
                    let logits_out = Tensor::zeros(out_shape, out_dtype, &self.device)?;
                    let hidden_out = Tensor::zeros(hid_shape.clone(), hid_dtype, &self.device)?;
                    unsafe {
                        cres::stream::begin_capture(
                            stream.cu_stream(),
                            csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                        )
                    }
                    .map_err(|e| anyhow!("pgraph begin_capture: {e}"))?;
                    let (logits_t, hidden_t) = match fwd(&mut self.model, &emb_t, &rope_pos_t) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                            return Err(anyhow!("pgraph capture: {e}"));
                        }
                    };
                    if let Err(e) = logits_out.slice_set(&logits_t, 0, 0) {
                        let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                        return Err(anyhow!("pgraph logits_out copy: {e}"));
                    }
                    if let Err(e) = hidden_out.slice_set(&hidden_t, 0, 0) {
                        let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                        return Err(anyhow!("pgraph hidden_out copy: {e}"));
                    }
                    drop(logits_t);
                    drop(hidden_t);
                    let cu_graph = unsafe { cres::stream::end_capture(stream.cu_stream()) }
                        .map_err(|e| anyhow!("pgraph end_capture: {e}"))?;
                    if cu_graph.is_null() {
                        return Err(anyhow!("pgraph end_capture returned null graph"));
                    }
                    let mut nodes: usize = 0;
                    unsafe { csys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut nodes) };
                    let mut exec: csys::CUgraphExec = std::ptr::null_mut();
                    let res = unsafe { csys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, 0) };
                    if res != csys::CUresult::CUDA_SUCCESS || exec.is_null() {
                        unsafe { csys::cuGraphDestroy(cu_graph) };
                        return Err(anyhow!("pgraph instantiate: {res:?}"));
                    }
                    eprintln!("[pg] captured {kind} T={t} slot={slot} nodes={nodes}");
                    Ok(PrefillGraphState {
                        exec,
                        cu_graph,
                        stream: stream.clone(),
                        t,
                        emb_t: emb_t.clone(),
                        rope_pos_t: rope_pos_t.clone(),
                        logits_t: logits_out,
                        hidden_t: hidden_out,
                    })
                })();
                match captured {
                    Ok(g) => {
                        pool.push(g);
                    }
                    // Захват не удался — результат уже посчитан eager-прогоном,
                    // состояние консистентно; графы остаются включёнными.
                    Err(e) => eprintln!(
                        "[pg] {kind} capture failed (результат отдан eager-прогоном): {e}"
                    ),
                }
                // При захвате операции записываются, а не исполняются, поэтому
                // буфер графа пуст: настоящие hidden даёт прогревочный проход.
                Ok((flat, prime_hidden, false))
            }
        }
    }

    /// Сбой графового пути: графы гаснут до следующего приёма запроса, сбои
    /// считаются; с потолка — до перезапуска процесса (OQ-8).
    #[cfg(feature = "cuda")]
    fn graphs_failed(&mut self, what: &str, e: &anyhow::Error) {
        self.graph_failures += 1;
        self.graphs_enabled = false;
        let cap = graph_fail_cap();
        let until = if self.graph_failures >= cap {
            "до перезапуска процесса"
        } else {
            "до следующего приёма запроса"
        };
        eprintln!(
            "[graphs] {what}: {e} — сбой {}/{cap}, графы выключены {until}",
            self.graph_failures
        );
    }

    /// Приём нового запроса: вернуть графы после сбоя, пока потолок не достигнут.
    #[cfg(feature = "cuda")]
    fn graphs_reenable_on_admit(&mut self) {
        if self.graphs_enabled || !self.graphs_configured {
            return;
        }
        let cap = graph_fail_cap();
        if self.graph_failures < cap {
            self.graphs_enabled = true;
            eprintln!(
                "[graphs] включены снова при приёме запроса (сбоев {}/{cap})",
                self.graph_failures
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn decode_batch_graphed(
        &mut self,
        b: usize,
        tokens: &[u32],
        rope_positions: &[usize],
        slots: &[u32],
        positions: &[usize],
    ) -> Result<Option<(Vec<Vec<f32>>, Tensor)>> {
        if !self.graphs_enabled {
            return Ok(None);
        }
        // Батч больше гейта уходит на eager: многослотовый граф-путь не готов.
        if b > graph_max_b() {
            return Ok(None);
        }
        let Device::Cuda(cuda_dev) = &self.device else {
            return Ok(None);
        };
        // Гейт транзакций снят 2026-08-27. Его причиной было то, что MTP читает
        // hidden, а буфер графа перезаписывается следующим launch. Теперь hidden
        // выводится во ВНЕШНИЙ буфер и возвращается ГЛУБОКОЙ копией, поэтому
        // держать его между вызовами безопасно.
        //
        // Оговорка про многослотовый режим: раньше здесь стоял any() по всем
        // слотам, то есть транзакция одного слота уводила на eager шаги всех
        // остальных. Многослотовый режим отложен, но если вернётся — проверять
        // надо слоты текущего вызова, а не все.

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
                // Освобождённый batched-кэш возвращаем ОС. Порядок важен:
                // cuMemFreeAsync упорядочен по потоку, и до синхронизации
                // освобождать ещё нечего — trim без неё не делает ничего.
                #[cfg(feature = "cuda")]
                if let Device::Cuda(c) = &self.device {
                    let _ = c.cuda_stream().synchronize();
                    let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(c);
                }
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
                    .reset_kv_len(&lens)
                    .map_err(|e| anyhow!("migrate reset_kv_len slot {sidx}: {e}"))?;
            }
        }

        // Окно и позиции: позиция должна совпадать с device kv_len (host mirror).
        {
            let ctx = self.model.paged_ctx.as_ref().unwrap();
            for (i, &s) in slots.iter().enumerate() {
                let cur = ctx.kv_len_host[s as usize] as usize;
                if positions[i] != cur || cur + 1 > window {
                    if !self.graph_gate_warned[s as usize] {
                        self.graph_gate_warned[s as usize] = true;
                        eprintln!(
                            "[graphs] slot {s}: декод уходит на eager — позиция {} против kv_len {cur}, окно пула {window} (GRAPH_WINDOW / served_ctx); дальше по этому запросу молчу",
                            positions[i]
                        );
                    }
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

        let hit = self.decode_graphs.iter().position(|g| g.b == b);
        let need_capture = match hit {
            Some(_) => {
                let why = String::new();
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
            // пул не чистим: другие формы остаются валидными
            // Eager graphed-forward: реальный результат шага + prime всех ядер/кэшей.
            // Guard включает htod param cache: params_from_vec идёт в кэш (prime),
            // а при захвате промах → явная ошибка вместо pageable memcpy в графе.
            // Эмбеддинги — ДО guard кэша htod: буфер больше его порога, а
            // захвату он и не нужен (стейджится снаружи).
            let emb_eager = self
                .model
                .embed_for_graph(tokens, &self.device)
                .map_err(|e| anyhow!("graph capture embed: {e}"))?
                .reshape((b, 1usize, self.model.hidden_size()))
                .map_err(|e| anyhow!("graph capture embed reshape: {e}"))?;
            let _htod_guard = cuda_dev.enable_cuda_graph_htod_cache();
            {
                let ctx = self.model.paged_ctx.as_mut().unwrap();
                ctx.stage_inputs(slots, rope_positions, &block_table)
                    .map_err(|e| {
                        anyhow!("graph capture stage_inputs b={b} slots={slots:?}: {e}")
                    })?;
            }
            let (logits, hidden) = self
                .model
                .forward_decode_batch_graphed(&emb_eager, slots)
                .map_err(|e| anyhow!("graphed forward (eager prime): {e}"))?;
            // Форма hidden для внешнего буфера захвата: берём с прогрева, он и так
            // выполняется перед захватом. Раньше hidden прогрева использовался
            // только как результат шага и в граф не выводился.
            let hid_shape = hidden.shape().clone();
            let hid_dtype = hidden.dtype();
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
                    .forward_decode_batch_graphed(&emb_eager, slots)
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
                        if is_delta {
                            dsum += t;
                            dcount += 1;
                        } else {
                            asum += t;
                            acount += 1;
                        }
                    }
                }
                times.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top: Vec<String> = times
                    .iter()
                    .take(8)
                    .map(|(i, t)| format!("b{i}={t:.2}"))
                    .collect();
                eprintln!("[gGPU-blocks] delta_sum={dsum:.1}ms/{dcount} attn_sum={asum:.1}ms/{acount} top: {}", top.join(" "));
            }
            // host mirror kv_len после инкремента
            {
                let ctx = self.model.paged_ctx.as_mut().unwrap();
                for &s in slots {
                    ctx.kv_len_host[s as usize] += 1;
                }
            }
            let flat = pinned_logits::to_vec_f32(&self.device, &logits)
                .map_err(|e| anyhow!("graphed logits (pinned): {e}"))?;
            let vocab = self.vocab_size();
            if flat.len() != b * vocab {
                return Err(anyhow!("graphed logits length mismatch"));
            }
            let out: Vec<Vec<f32>> = flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect();
            // Пул полон — не захватываем (вытеснение роняло следующий шаг).
            if self.decode_graphs.len() >= dgraph_lru() {
                return Ok(Some((out, hidden)));
            }
            // Capture для следующих шагов (захват не исполняет ядра).
            let stream = cuda_dev.cuda_stream();
            let capture_result = (|| -> Result<DecodeGraphState> {
                use cudarc::driver::{result as cres, sys as csys};
                if graph_fail_inject() {
                    return Err(anyhow!("искусственный сбой захвата (GRAPH_FAIL_INJECT)"));
                }
                // ВНИМАНИЕ: память, выделенная ВНУТРИ захвата, принадлежит graph pool —
                // её адреса валидны только внутри graph launch. Внешний D2H по ним →
                // illegal address. Поэтому выход копируем во ВНЕШНИЙ (default pool,
                // выделен ДО захвата) буфер D2D-нодой внутри графа.
                let logits_out = Tensor::zeros((b, self.vocab), DType::F32, &self.device)?;
                let hidden_out = Tensor::zeros(hid_shape.clone(), hid_dtype, &self.device)?;
                unsafe {
                    cres::stream::begin_capture(
                        stream.cu_stream(),
                        csys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                    )
                }
                .map_err(|e| anyhow!("begin_capture: {e}"))?;
                let forward_result = self.model.forward_decode_batch_graphed(&emb_eager, slots);
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
                if let Err(e) = hidden_out.slice_set(&hidden_t, 0, 0) {
                    let _ = unsafe { cres::stream::end_capture(stream.cu_stream()) };
                    return Err(anyhow!("hidden_out copy: {e}"));
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
                    if std::env::var("GRAPH_DOT").as_deref() == Ok("1") {
                        let path =
                            std::ffi::CString::new("D:\\Projects\\yttri-build\\decode-graph.dot")
                                .unwrap();
                        let res =
                            unsafe { csys::cuGraphDebugDotPrint(cu_graph, path.as_ptr(), 1u32) };
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
                    emb_t: emb_eager.clone(),
                    logits_t: logits_out,
                    hidden_t: hidden_out,
                })
            })();
            match capture_result {
                Ok(state) => {
                    self.decode_graphs.push(state);
                    if crate::scheduler::trace_on() {
                        eprintln!("[graphs] captured decode graph B={b} slots={slots:?}");
                    }
                }
                Err(e) => {
                    // Захват не удался — продолжаем eager (результат отдан
                    // прогревочным проходом).
                    self.graphs_failed("capture failed (eager fallback)", &e);
                }
            }
            // Путь захвата: hidden получен прогревочным проходом, это обычный
            // тензор вне графового буфера — копировать не нужно.
            return Ok(Some((out, hidden)));
        }

        // Replay path.
        let idx = self
            .decode_graphs
            .iter()
            .position(|g| g.b == b)
            .expect("декодный граф только что захвачен или найден");
        let state = &self.decode_graphs[idx];
        let stream = cuda_dev.cuda_stream();
        // Контекст на каждой операции: редкий CUDA_ERROR_INVALID_VALUE на
        // многослотовом сервере (2026-08-28) приходил голым DriverError, и
        // операцию было не назвать.
        {
            let ctx = self.model.paged_ctx.as_mut().unwrap();
            ctx.stage_inputs(slots, rope_positions, &block_table)
                .map_err(|e| anyhow!("graph replay stage_inputs b={b} slots={slots:?}: {e}"))?;
        }
        // Эмбеддинги: деквант на хосте + htod в persistent-буфер (вне графа).
        let emb_staging = self
            .model
            .embed_for_graph(tokens, &self.device)
            .map_err(|e| anyhow!("graph replay embed: {e}"))?
            .reshape((b, 1usize, self.model.hidden_size()))
            .map_err(|e| anyhow!("graph replay embed reshape: {e}"))?;
        state
            .emb_t
            .slice_set(&emb_staging, 0, 0)
            .map_err(|e| anyhow!("graph replay emb slice_set: {e}"))?;
        state
            .launch()
            .map_err(|e| anyhow!("graph launch b={b} slots={slots:?}: {e}"))?;
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
        let flat = pinned_logits::to_vec_f32(&self.device, &state.logits_t)
            .map_err(|e| anyhow!("graph logits read (pinned): {e}"))?;
        let vocab = self.vocab_size();
        if flat.len() != b * vocab {
            return Err(anyhow!("graph logits length mismatch"));
        }
        // Глубокая копия, а не clone: clone разделяет хранилище, а внешний буфер
        // перезаписывается следующим launch, тогда как проверка спекуляции держит
        // hidden между вызовами (PendingVerify). Для [K, hidden] это десятки
        // килобайт — цена пренебрежимая.
        let hidden_copy = {
            let src = &self.decode_graphs[idx].hidden_t;
            let dst = Tensor::zeros(src.shape().clone(), src.dtype(), &self.device)
                .map_err(|e| anyhow!("graph hidden zeros: {e}"))?;
            dst.slice_set(src, 0, 0)
                .map_err(|e| anyhow!("graph hidden copy: {e}"))?;
            dst
        };
        Ok(Some((
            flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect(),
            hidden_copy,
        )))
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

    /// Смешивание paged- и eager-чанков внутри одного промпта теряет префикс:
    /// после первого чанка в пуле откат на eager обязан быть ошибкой, а не
    /// молчаливым `Ok(None)`.
    #[cfg(feature = "cuda")]
    #[test]
    fn eager_fallback_blocked_once_prompt_is_paged() {
        assert!(eager_fallback_allowed(PrefillPath::Undecided));
        assert!(eager_fallback_allowed(PrefillPath::Eager));
        assert!(!eager_fallback_allowed(PrefillPath::Paged));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn paged_prefill_is_fail_closed_by_default() {
        assert!(matches!(parse_pgraph_mode(None), PgraphMode::Off));
        assert!(matches!(parse_pgraph_mode(Some("off")), PgraphMode::Off));
        assert!(matches!(parse_pgraph_mode(Some("0")), PgraphMode::Off));
        assert!(matches!(parse_pgraph_mode(Some("on")), PgraphMode::On));
        assert!(matches!(parse_pgraph_mode(Some("1")), PgraphMode::On));
        assert!(matches!(
            parse_pgraph_mode(Some("check")),
            PgraphMode::Check
        ));
    }

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
