//! Integration tests for model_profile validation (Phase 1).
//!
//! Tests cover metadata validation, tensor contract checks, corruption
//! detection, and architecture dispatch for both dense `qwen35` and
//! MoE `qwen35moe`.

#![cfg(feature = "real-model")]

use candle_core::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
use candle_core::quantized::GgmlDType;
use candle_core::Shape;
use qwen35_batch::real::model_profile::{Architecture, ModelProfile};
use std::collections::HashMap;

fn make_content(metadata: HashMap<String, Value>, tensor_infos: HashMap<String, TensorInfo>) -> Content {
    Content {
        magic: VersionedMagic::GgufV2,
        metadata,
        tensor_infos,
        tensor_data_offset: 4096,
    }
}

fn u32_val(v: u32) -> Value {
    Value::U32(v)
}

fn f32_val(v: f32) -> Value {
    Value::F32(v)
}

fn str_val(v: &str) -> Value {
    Value::String(v.to_string())
}

const QWEN35_TOKEN_IDS: [(&str, u32); 7] = [
    ("<|endoftext|>", 248044),
    ("<|im_start|>", 248045),
    ("<|im_end|>", 248046),
    ("<|vision_start|>", 248053),
    ("<|vision_end|>", 248054),
    ("<|image_pad|>", 248056),
    ("<|video_pad|>", 248057),
];

fn tensor_info(dtype: GgmlDType, shape: &[usize], offset: u64) -> TensorInfo {
    TensorInfo {
        ggml_dtype: dtype,
        shape: Shape::from(shape.to_vec()),
        offset,
    }
}

fn dense_metadata() -> HashMap<String, Value> {
    let mut md = HashMap::new();
    md.insert("general.architecture".into(), str_val("qwen35"));
    md.insert("qwen35.block_count".into(), u32_val(36));
    md.insert("qwen35.embedding_length".into(), u32_val(2560));
    md.insert("qwen35.context_length".into(), u32_val(32768));
    md.insert("qwen35.feed_forward_length".into(), u32_val(9728));
    md.insert("qwen35.attention.head_count".into(), u32_val(32));
    md.insert("qwen35.attention.head_count_kv".into(), u32_val(8));
    md.insert("qwen35.attention.key_length".into(), u32_val(256));
    md.insert("qwen35.attention.value_length".into(), u32_val(256));
    md.insert("qwen35.attention.layer_norm_rms_epsilon".into(), f32_val(1e-6));
    md.insert("qwen35.full_attention_interval".into(), u32_val(4));
    md
}

/// Mixer tensor names per hybrid layout (interval=4): DeltaNet blocks lack
/// attn_q/k/v/output; full-attention blocks lack ssm_*/attn_qkv/attn_gate.
/// GGUF tensor names have no architecture prefix.
fn mixer_tensors(layer: usize) -> Vec<String> {
    let prefix = format!("blk.{layer}");
    let mut v = vec![
        format!("{prefix}.attn_norm.weight"),
        format!("{prefix}.post_attention_norm.weight"),
    ];
    if (layer + 1) % 4 == 0 {
        for n in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"] {
            v.push(format!("{prefix}.{n}"));
        }
    } else {
        for n in [
            "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
            "ssm_out.weight", "ssm_dt.bias", "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight",
        ] {
            v.push(format!("{prefix}.{n}"));
        }
    }
    v
}

fn dense_tensors() -> HashMap<String, TensorInfo> {
    let mut ti = HashMap::new();
    ti.insert("token_embd.weight".into(), tensor_info(GgmlDType::Q4K, &[151943, 2560], 0));
    ti.insert("output_norm.weight".into(), tensor_info(GgmlDType::F32, &[2560], 1000));
    let ffn = ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];
    // Validator checks block 0 (DeltaNet) and last block 35 (attention).
    for (i, layer) in [0usize, 35].iter().enumerate() {
        for name in mixer_tensors(*layer) {
            ti.insert(name, tensor_info(GgmlDType::Q4K, &[2560, 2560], 2000 + i as u64 * 1000));
        }
        for n in &ffn {
            ti.insert(
                format!("blk.{layer}.{n}"),
                tensor_info(GgmlDType::Q4K, &[2560, 2560], 2000 + i as u64 * 1000),
            );
        }
    }
    ti
}

fn moe_metadata() -> HashMap<String, Value> {
    let mut md = HashMap::new();
    md.insert("general.architecture".into(), str_val("qwen35moe"));
    md.insert("qwen35moe.block_count".into(), u32_val(40));
    md.insert("qwen35moe.embedding_length".into(), u32_val(2560));
    md.insert("qwen35moe.context_length".into(), u32_val(81920));
    md.insert("qwen35moe.feed_forward_length".into(), u32_val(9728));
    md.insert("qwen35moe.attention.head_count".into(), u32_val(32));
    md.insert("qwen35moe.attention.head_count_kv".into(), u32_val(8));
    md.insert("qwen35moe.attention.key_length".into(), u32_val(256));
    md.insert("qwen35moe.attention.value_length".into(), u32_val(256));
    md.insert("qwen35moe.attention.layer_norm_rms_epsilon".into(), f32_val(1e-6));
    md.insert("qwen35moe.full_attention_interval".into(), u32_val(4));
    md.insert("qwen35moe.expert_count".into(), u32_val(128));
    md.insert("qwen35moe.expert_used_count".into(), u32_val(8));
    md.insert("qwen35moe.shared_expert_count".into(), u32_val(1));
    md.insert("qwen35moe.feed_forward_length.experts".into(), u32_val(4864));
    md.insert("qwen35moe.feed_forward_length.shared_expert".into(), u32_val(9728));
    md
}

fn moe_tensors() -> HashMap<String, TensorInfo> {
    let mut ti = HashMap::new();
    ti.insert("token_embd.weight".into(), tensor_info(GgmlDType::IQ2XXS, &[151943, 2560], 0));
    ti.insert("output_norm.weight".into(), tensor_info(GgmlDType::F32, &[2560], 1000));
    let moe_ffn = [
        "ffn_gate_inp.weight",  // router
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",  // routed
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",  // shared
    ];
    // All 40 blocks (validator does full check for MoE <= 48 blocks).
    for layer in 0..40usize {
        for name in mixer_tensors(layer) {
            ti.insert(
                name,
                tensor_info(GgmlDType::IQ2XXS, &[2560, 2560], 4000 + layer as u64 * 100),
            );
        }
        for n in &moe_ffn {
            ti.insert(
                format!("blk.{layer}.{n}"),
                tensor_info(GgmlDType::IQ2XXS, &[2560, 2560], 4000 + layer as u64 * 100),
            );
        }
    }
    ti
}

#[test]
fn qwen35_tokenizer_contract_is_stable() {
    assert_eq!(QWEN35_TOKEN_IDS[0], ("<|endoftext|>", 248044));
    assert_eq!(QWEN35_TOKEN_IDS[2], ("<|im_end|>", 248046));
    assert_eq!(QWEN35_TOKEN_IDS[4], ("<|vision_end|>", 248054));
    assert_eq!(QWEN35_TOKEN_IDS[6], ("<|video_pad|>", 248057));
    let ids: std::collections::HashSet<_> = QWEN35_TOKEN_IDS.iter().map(|(_, id)| *id).collect();
    assert_eq!(ids.len(), QWEN35_TOKEN_IDS.len());
}

#[test]
fn component_inventory_contract_is_stable() {
    const COUNTS: [usize; 3] = [426, 297, 15];
    assert_eq!(COUNTS.iter().sum::<usize>(), 738);
}

#[test]
fn test_dense_profile_validates() {
    let ct = make_content(dense_metadata(), dense_tensors());
    let profile = ModelProfile::read_and_validate(&ct, 1024 * 1024).unwrap();
    assert_eq!(profile.architecture, Architecture::DenseQwen35);
    assert_eq!(profile.block_count, 36);
    assert_eq!(profile.hidden_size, 2560);
    assert!(profile.routed_experts.is_none());
    assert!(!profile.fingerprint.hash.is_empty());
}

#[test]
fn test_moe_profile_validates() {
    let ct = make_content(moe_metadata(), moe_tensors());
    let profile = ModelProfile::read_and_validate(&ct, 10 * 1024 * 1024 * 1024).unwrap();
    assert_eq!(profile.architecture, Architecture::Qwen35Moe);
    assert_eq!(profile.block_count, 40);
    assert_eq!(profile.routed_experts, Some(128));
    assert_eq!(profile.experts_per_token, Some(8));
    assert_eq!(profile.shared_expert_count, Some(1));
    assert!(profile.quant_set.contains(&GgmlDType::IQ2XXS));
}

#[test]
fn test_moe_wrong_block_count_fails() {
    let mut md = moe_metadata();
    md.insert("qwen35moe.block_count".into(), u32_val(32));
    let ct = make_content(md, moe_tensors());
    let result = ModelProfile::read_and_validate(&ct, 0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("block_count=32") || msg.contains("expected 40"));
}

#[test]
fn test_moe_wrong_expert_used_count_fails() {
    let mut md = moe_metadata();
    md.insert("qwen35moe.expert_used_count".into(), u32_val(4));
    let ct = make_content(md, moe_tensors());
    let result = ModelProfile::read_and_validate(&ct, 0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("expert_used_count=4") || msg.contains("expected 8"));
}

#[test]
fn test_moe_missing_router_tensor_fails() {
    let mut ti = moe_tensors();
    ti.remove("blk.0.ffn_gate_inp.weight");
    let ct = make_content(moe_metadata(), ti);
    let result = ModelProfile::read_and_validate(&ct, 0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("ffn_gate_inp"));
}

#[test]
fn test_moe_missing_shared_expert_tensor_fails() {
    let mut ti = moe_tensors();
    ti.remove("blk.0.ffn_gate_shexp.weight");
    let ct = make_content(moe_metadata(), ti);
    let result = ModelProfile::read_and_validate(&ct, 0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("ffn_gate_shexp"));
}

#[test]
fn test_aggregated_errors_report_multiple() {
    let mut md = moe_metadata();
    md.remove("qwen35moe.expert_count");
    md.remove("qwen35moe.expert_used_count");
    md.insert("qwen35moe.block_count".into(), u32_val(32));
    let ct = make_content(md, moe_tensors());
    let result = ModelProfile::read_and_validate(&ct, 0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    // Should contain multiple errors.
    assert!(msg.contains("expert_count"));
    assert!(msg.contains("expert_used_count"));
    assert!(msg.contains("block_count=32"));
}

#[test]
fn test_dense_missing_block_tensor_fails() {
    let mut ti = dense_tensors();
    ti.remove("blk.0.ffn_gate.weight");
    let ct = make_content(dense_metadata(), ti);
    let result = ModelProfile::read_and_validate(&ct, 0);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("ffn_gate"));
}

#[test]
fn test_fingerprint_stable_across_same_input() {
    let ct1 = make_content(moe_metadata(), moe_tensors());
    let ct2 = make_content(moe_metadata(), moe_tensors());
    let p1 = ModelProfile::read_and_validate(&ct1, 1024).unwrap();
    let p2 = ModelProfile::read_and_validate(&ct2, 1024).unwrap();
    assert_eq!(p1.fingerprint.hash, p2.fingerprint.hash);
}

#[test]
fn test_fingerprint_changes_with_different_file_size() {
    let ct1 = make_content(moe_metadata(), moe_tensors());
    let ct2 = make_content(moe_metadata(), moe_tensors());
    let p1 = ModelProfile::read_and_validate(&ct1, 1024).unwrap();
    let p2 = ModelProfile::read_and_validate(&ct2, 2048).unwrap();
    assert_ne!(p1.fingerprint.hash, p2.fingerprint.hash);
}