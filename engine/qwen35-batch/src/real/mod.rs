//! Реальная модель Qwen3.5-4B (GGUF) — порт из Yttri без Tauri-зависимостей.
//!
//! Компилируется за фичей `real-model`; backend выбирается `metal` или `cuda`.
//! Источники скопированы из Yttri `frontend/src-tauri/src/modules/ai/local_llm/`
//! (см. REAL_MODEL.md §1–3). Metal и CUDA используют те же production kernels,
//! но standalone crate не зависит от Tauri.

pub mod adapter;
#[cfg(feature = "cuda")]
pub mod delta_rule_cuda;
/// Phase 2: true batched decode (ось slot B) — CUDA.
#[cfg(feature = "cuda")]
pub mod delta_rule_batched_cuda;
pub mod ytf16;
/// Split-K flash-decoding (длинный KV, seqlen_q=1).
#[cfg(feature = "cuda")]
pub mod flash_decode_cuda;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub mod metal_utils;
pub mod model_profile;
pub mod model_weights;
pub mod moe;
pub mod mtp;
pub mod multimodal;
/// Paged KV decode + CUDA graph plumbing.
#[cfg(feature = "cuda")]
pub mod paged_kv_cuda;
#[cfg(feature = "real-model")]
pub mod tokenizer;
#[cfg(feature = "real-model")]
pub mod vision;

pub use adapter::Qwen35BatchAdapter;
pub use model_profile::{Architecture, ModelFingerprint, ModelProfile, MtpProfile};
pub use model_weights::ModelWeights;
