//! qwen35-batch — standalone continuous-batching prototype для Qwen3.5 на Candle.
//!
//! Архитектура (см. docs/research/2026-08-01 в Yttri для обоснования):
//!
//! - **Per-slot state**: каждый из N слотов имеет собственное рекуррентное
//!   состояние DeltaNet (conv_buf + ssm_state, fp32) и собственный KV-cache
//!   attention-слоёв. Веса модели — одни на все слоты (shared weight copy).
//! - **Batched decode step**: на каждом шаге все активные Decoding-слоты
//!   обрабатываются в одном батче — QMatMul-проекции амортизируют чтение весов
//!   (доминирующая cost на bandwidth-bound Apple Silicon). Рекуррентный
//!   delta_rule-шаг и attention выполняются per-slot (state не имеет batch-оси
//!   в Candle-пути; это потолок без переписывания Metal-ядер).
//! - **Continuous batching scheduler**: слоты IDLE→PREFILL→DECODE; prefill
//!   admit'ится последовательно (как у LM Studio MLX / llama.cpp prefill vs
//!   decode), decode-шаг батчит все DECODE-слоты одновременно; завершённый
//!   слот освобождается для нового запроса.
//!
//! Hypothesis (bandwidth-bound на Pro-class): ×1.15–1.35 агрегатного decode
//! gain при B=4 vs sequential. Прототип должен это подтвердить/опровергнуть
//! реальными числами на 4B GGUF.

pub mod model;
pub mod scheduler;
pub mod slot;

/// Реальная GGUF-модель Qwen3.5 (порт из Yttri) — только за фичей `real-model`.
#[cfg(feature = "real-model")]
pub mod real;

pub use model::{BatchModel, DecodeBatch, MultimodalPrefill, PrefillChunk};
pub use scheduler::{BatchScheduler, SchedulerStats};
pub use slot::{Slot, SlotStatus};

/// Базовое значение batch-размера (4 параллельных потока), как ordered.
pub const DEFAULT_NUM_SLOTS: usize = 4;
