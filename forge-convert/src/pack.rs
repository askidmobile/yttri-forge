//! Упаковка safetensors → самостоятельный контейнер .ytf (формат v2).
//!
//! Отличие от сайдкара v1: контейнер несёт ВСЮ языковую модель, каждый тензор
//! лежит готовыми GGML-блоками своего типа, гиперпараметры и токенизатор едут
//! внутри. Рантайму GGUF не нужен — ни для весов, ни для метаданных, ни для
//! токенизатора. Это снимает и вторую копию весов в VRAM, и переквантование
//! на CPU при загрузке.
//!
//! Эталонный GGUF нужен ТОЛЬКО на этапе упаковки: из него берутся типы
//! тензоров и метаданные, чтобы первый контейнер был сопоставим с текущим
//! путём один-в-один и расхождение было видно сразу.

use crate::container;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

/// HF-имя тензора → имя, по которому его спрашивает конструктор модели.
///
/// Имена сверены по `build_model_common`, а не по маске сайдкара: у сайдкара
/// своё пространство имён (`attn_z`, `attn_a`…), а конструктор просит
/// `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`, `attn_output`.
/// Видео-башня (`model.visual.*`) и MTP (`mtp.*`) в v2 пока не входят.
pub fn hf_to_gguf(hf: &str) -> Option<String> {
    match hf {
        "model.language_model.embed_tokens.weight" => return Some("token_embd.weight".into()),
        "model.language_model.norm.weight" => return Some("output_norm.weight".into()),
        _ => {}
    }
    let rest = hf.strip_prefix("model.language_model.layers.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<u32>().ok()?;
    let g = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        // DeltaNet
        "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
        "linear_attn.in_proj_z.weight" => "attn_gate.weight",
        "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
        "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
        "linear_attn.out_proj.weight" => "ssm_out.weight",
        "linear_attn.dt_bias" => "ssm_dt.bias",
        "linear_attn.A_log" => "ssm_a",
        "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
        "linear_attn.norm.weight" => "ssm_norm.weight",
        // Внимание
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        _ => return None,
    };
    Some(format!("blk.{idx}.{g}"))
}

pub fn dtype_name(dt: GgmlDType) -> &'static str {
    match dt {
        GgmlDType::F32 => "F32",
        GgmlDType::F16 => "F16",
        GgmlDType::Q4_0 => "Q4_0",
        GgmlDType::Q4_1 => "Q4_1",
        GgmlDType::Q5_0 => "Q5_0",
        GgmlDType::Q5_1 => "Q5_1",
        GgmlDType::Q8_0 => "Q8_0",
        GgmlDType::Q8_1 => "Q8_1",
        GgmlDType::Q2K => "Q2_K",
        GgmlDType::Q3K => "Q3_K",
        GgmlDType::Q4K => "Q4_K",
        GgmlDType::Q5K => "Q5_K",
        GgmlDType::Q6K => "Q6_K",
        GgmlDType::Q8K => "Q8_K",
        GgmlDType::BF16 => "BF16",
        // IQ-типы (2-битные и прочие) в контейнер пока не пакуем: у читателя
        // движка их нет в разборе, и молчаливо переименовать нельзя.
        other => panic!("тип {other:?} пока не поддержан контейнером v2"),
    }
}

/// Значение метаданных GGUF → JSON для манифеста. Читатель разворачивает
/// целые обратно в U32, дробные в F32 — ровно те типы, что просит конструктор.
fn value_to_json(v: &gguf_file::Value) -> Option<serde_json::Value> {
    use gguf_file::Value as V;
    Some(match v {
        V::U8(x) => (*x as u64).into(),
        V::U16(x) => (*x as u64).into(),
        V::U32(x) => (*x as u64).into(),
        V::U64(x) => (*x).into(),
        V::I8(x) if *x >= 0 => (*x as u64).into(),
        V::I16(x) if *x >= 0 => (*x as u64).into(),
        V::I32(x) if *x >= 0 => (*x as u64).into(),
        V::I64(x) if *x >= 0 => (*x as u64).into(),
        V::F32(x) => serde_json::Number::from_f64(*x as f64).map(Into::into)?,
        V::F64(x) => serde_json::Number::from_f64(*x).map(Into::into)?,
        V::Bool(x) => (*x).into(),
        V::String(s) => s.clone().into(),
        _ => return None,
    })
}

/// Ключи метаданных, которые нужны конструктору модели. Массивы токенизатора
/// не переносим — вместо них внутрь кладётся готовый tokenizer.json.
fn wanted_metadata(key: &str) -> bool {
    if key.starts_with("tokenizer.") {
        // chat_template — строка, она нужна; остальное (tokens/merges) заменяет блоб.
        return key == "tokenizer.chat_template"
            || key == "tokenizer.ggml.bos_token_id"
            || key == "tokenizer.ggml.eos_token_id";
    }
    key.starts_with("general.")
        || key.starts_with("qwen35.")
        || key.starts_with("qwen35moe.")
}

pub struct PackStats {
    pub tensors: usize,
    pub bytes: u64,
    pub by_dtype: BTreeMap<String, usize>,
    pub skipped: Vec<String>,
}

/// Квантовать f32-тензор в целевой GGML-тип и вернуть сырые блоки.
fn quantize_bytes(values: Vec<f32>, shape: &[usize], dt: GgmlDType) -> Result<Vec<u8>, String> {
    let t = Tensor::from_vec(values, shape, &Device::Cpu)
        .map_err(|e| format!("tensor {shape:?}: {e}"))?;
    let q = QTensor::quantize(&t, dt).map_err(|e| format!("quantize {dt:?}: {e}"))?;
    let data = q.data().map_err(|e| format!("data {dt:?}: {e}"))?;
    Ok(data.into_owned())
}

/// Собрать самостоятельный контейнер.
///
/// `ref_gguf` задаёт типы тензоров и метаданные; `tokenizer_json` встраивается
/// блобом. Формы сверяются с эталоном: расхождение — ошибка, а не тихий сдвиг.
#[allow(clippy::too_many_arguments)]
pub fn pack(
    shards: &[(String, memmap2::Mmap)],
    hf_names: &[String],
    ref_gguf: &Path,
    tokenizer_json: &Path,
    out_path: &Path,
    delta_layout: Option<crate::DeltaLayout>,
    read_tensor_f32: &dyn Fn(&[(String, memmap2::Mmap)], &str) -> Result<(Vec<f32>, Vec<usize>), String>,
    repack: &dyn Fn(&str, Vec<u8>, &[usize], Option<crate::DeltaLayout>) -> Vec<u8>,
) -> Result<PackStats, String> {
    let mut gf = std::fs::File::open(ref_gguf)
        .map_err(|e| format!("open ref gguf {}: {e}", ref_gguf.display()))?;
    let ct = gguf_file::Content::read(&mut gf)
        .map_err(|e| format!("read ref gguf: {e}"))?;

    let mut config: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for (k, v) in ct.metadata.iter() {
        if wanted_metadata(k) {
            if let Some(j) = value_to_json(v) {
                config.insert(k.clone(), j);
            }
        }
    }
    if config.is_empty() {
        return Err("эталонный GGUF не дал ни одного ключа метаданных".into());
    }

    let out_file = std::fs::File::create(out_path)
        .map_err(|e| format!("create {}: {e}", out_path.display()))?;
    let pre = container::ManifestPre {
        gguf_sha256: String::new(), // v2 самостоятелен, привязки к GGUF нет
        mask: "standalone".into(),
    };
    let mut w = container::ContainerWriter::create_standalone(out_file, pre, config)
        .map_err(|e| format!("container create: {e}"))?;

    let mut stats = PackStats {
        tensors: 0,
        bytes: 0,
        by_dtype: BTreeMap::new(),
        skipped: Vec::new(),
    };

    for hf in hf_names {
        let Some(gguf_name) = hf_to_gguf(hf) else {
            stats.skipped.push(hf.clone());
            continue;
        };
        let info = ct.tensor_infos.get(&gguf_name).ok_or_else(|| {
            format!("{hf} → {gguf_name}: в эталонном GGUF такого тензора нет")
        })?;
        let (values, shape) = read_tensor_f32(shards, hf)?;
        if shape != info.shape.dims() {
            return Err(format!(
                "{gguf_name}: форма {shape:?} не совпадает с эталоном {:?}",
                info.shape.dims()
            ));
        }
        // Перепаковка раскладки DeltaNet работает по байтам F16 — применяем её
        // до квантования, на F16-представлении.
        let values = if delta_layout.is_some() && shape.len() == 2 {
            let f16le: Vec<u8> = values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect();
            let packed = repack(&gguf_name, f16le, &shape, delta_layout);
            packed
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        } else {
            values
        };
        let bytes = quantize_bytes(values, &shape, info.ggml_dtype)?;
        let dn = dtype_name(info.ggml_dtype);
        w.add_typed(&gguf_name, &shape, dn, &bytes);
        stats.tensors += 1;
        stats.bytes += bytes.len() as u64;
        *stats.by_dtype.entry(dn.to_string()).or_insert(0) += 1;
    }

    let tok = std::fs::read(tokenizer_json)
        .map_err(|e| format!("read {}: {e}", tokenizer_json.display()))?;
    w.add_typed(
        crate::pack::TOKENIZER_BLOB,
        &[tok.len()],
        "RAW",
        &tok,
    );
    stats.bytes += tok.len() as u64;

    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(stats)
}

/// Имя блоба токенизатора внутри data-секции (совпадает с читателем движка).
pub const TOKENIZER_BLOB: &str = "__tokenizer_json";

#[cfg(test)]
mod tests {
    use super::hf_to_gguf;

    /// Карта имён сверена по `build_model_common`, а не по маске сайдкара.
    /// Разница неочевидная и молчаливая: у сайдкара `attn_z`/`attn_a`/`attn_b`/
    /// `attn_out`/`attn_o`, у конструктора модели — совсем другие имена.
    /// Ошибка здесь даёт не падение, а мусорные веса.
    #[test]
    fn delta_names_match_the_model_builder_not_the_sidecar_mask() {
        let l = |s: &str| hf_to_gguf(&format!("model.language_model.layers.3.{s}"));
        assert_eq!(l("linear_attn.in_proj_z.weight").as_deref(), Some("blk.3.attn_gate.weight"));
        assert_eq!(l("linear_attn.in_proj_a.weight").as_deref(), Some("blk.3.ssm_alpha.weight"));
        assert_eq!(l("linear_attn.in_proj_b.weight").as_deref(), Some("blk.3.ssm_beta.weight"));
        assert_eq!(l("linear_attn.out_proj.weight").as_deref(), Some("blk.3.ssm_out.weight"));
        assert_eq!(l("self_attn.o_proj.weight").as_deref(), Some("blk.3.attn_output.weight"));
        assert_eq!(l("linear_attn.A_log").as_deref(), Some("blk.3.ssm_a"));
        assert_eq!(l("linear_attn.dt_bias").as_deref(), Some("blk.3.ssm_dt.bias"));
        assert_eq!(
            l("post_attention_layernorm.weight").as_deref(),
            Some("blk.3.post_attention_norm.weight")
        );
    }

    #[test]
    fn globals_and_unmapped() {
        assert_eq!(
            hf_to_gguf("model.language_model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            hf_to_gguf("model.language_model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        // Видео-башня и MTP в v2 не входят — должны отсеиваться, а не падать.
        assert_eq!(hf_to_gguf("model.visual.blocks.0.attn.qkv.weight"), None);
        assert_eq!(hf_to_gguf("mtp.fc.weight"), None);
        // Незнакомый суффикс языкового слоя тоже None — вызывающий это заметит
        // и остановится, потому что языковой тензор терять нельзя.
        assert_eq!(hf_to_gguf("model.language_model.layers.1.something.weight"), None);
    }
}
