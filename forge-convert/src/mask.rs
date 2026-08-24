//! Heavy-маска: резолв имён тензоров safetensors → GGUF-имена.
//!
//! Qwen3.5/3.8 (DeltaNet-гибрид):
//!   DeltaNet слои (linear_attn, ~3 из 4):
//!     GGUF blk.{i}.attn_qkv.weight  ← language_model.layers.{i}.linear_attn.in_proj_qkv.weight
//!     GGUF blk.{i}.attn_z.weight    ← …in_proj_z.weight
//!     GGUF blk.{i}.attn_a.weight    ← …in_proj_a.weight
//!     GGUF blk.{i}.attn_b.weight    ← …in_proj_b.weight
//!     GGUF blk.{i}.attn_out.weight  ← …linear_attn.out_proj.weight
//!   Attention слои (self_attn, каждый 4-й):
//!     GGUF blk.{i}.attn_q/k/v/o     ← self_attn.{q,k,v,o}_proj.weight

/// Суффикс → GGUF-имя внутри слоя. None = тензор не входит в heavy-маску.
pub fn resolve(layer_kind: LayerKind, st_suffix: &str) -> Option<&'static str> {
    match layer_kind {
        LayerKind::DeltaNet => match st_suffix {
            "linear_attn.in_proj_qkv.weight" => Some("blk.{i}.attn_qkv.weight"),
            "linear_attn.in_proj_z.weight" => Some("blk.{i}.attn_z.weight"),
            "linear_attn.in_proj_a.weight" => Some("blk.{i}.attn_a.weight"),
            "linear_attn.in_proj_b.weight" => Some("blk.{i}.attn_b.weight"),
            "linear_attn.out_proj.weight" => Some("blk.{i}.attn_out.weight"),
            _ => None,
        },
        LayerKind::Attention => match st_suffix {
            "self_attn.q_proj.weight" => Some("blk.{i}.attn_q.weight"),
            "self_attn.k_proj.weight" => Some("blk.{i}.attn_k.weight"),
            "self_attn.v_proj.weight" => Some("blk.{i}.attn_v.weight"),
            "self_attn.o_proj.weight" => Some("blk.{i}.attn_o.weight"),
            _ => None,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    DeltaNet,
    Attention,
}

/// Классифицировать тензор safetensors по имени.
/// Возвращает (layer_index, kind, суффикс) или None если вне слоёв.
pub fn classify(name: &str) -> Option<(u32, LayerKind, String)> {
    let rest = name.strip_prefix("language_model.layers.")?;
    let dot = rest.find('.')?;
    let idx: u32 = rest[..dot].parse().ok()?;
    let suffix = &rest[dot + 1..];
    // Порядок важен: self_attn проверяем первым (attention слой)
    if suffix.starts_with("self_attn.") {
        return Some((idx, LayerKind::Attention, suffix.to_string()));
    }
    if suffix.starts_with("linear_attn.") {
        return Some((idx, LayerKind::DeltaNet, suffix.to_string()));
    }
    None
}

/// Итоговое GGUF-имя тензора для сайдкара.
pub fn gguf_name(layer_index: u32, gguf_template: &str) -> String {
    gguf_template.replace("{i}", &layer_index.to_string())
}
