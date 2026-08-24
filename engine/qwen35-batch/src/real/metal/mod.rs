// metal/ — Metal compute shaders для ускорения DeltaNet inference
//
// Модуль содержит:
// - delta_rule.metal — Metal Shading Language ядра
// - delta_rule_metal.rs — Rust bindings для dispatch

#[cfg(target_os = "macos")]
pub mod delta_rule_metal;

/// Phase 1: true batched decode (ось slot B).
#[cfg(target_os = "macos")]
pub mod delta_rule_batched_metal;

#[cfg(target_os = "macos")]
pub mod gated_delta_net_fused;
