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
        // Отдельная выходная голова. У 4B её нет (tie_word_embeddings=true) и
        // конструктор откатывается на token_embd; у 9B она есть, и без этой
        // строки контейнер молча остался бы со связанными эмбеддингами.
        "lm_head.weight" => return Some("output.weight".into()),
        "model.lm_head.weight" => return Some("output.weight".into()),
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

/// HF-имя тензора видео-башни → имена в mmproj.
///
/// Возвращает список, потому что `patch_embed.proj.weight` — это Conv3d
/// [1024, 3, 2, 16, 16], а загрузчик ждёт его расщеплённым по временной оси
/// на два тензора [1024, 3, 16, 16] (`v.patch_embd.weight` и `.weight.1`).
pub fn hf_to_gguf_vision(hf: &str) -> Option<Vec<String>> {
    match hf {
        "model.visual.patch_embed.proj.weight" => {
            return Some(vec![
                "v.patch_embd.weight".into(),
                "v.patch_embd.weight.1".into(),
            ])
        }
        "model.visual.patch_embed.proj.bias" => return Some(vec!["v.patch_embd.bias".into()]),
        "model.visual.pos_embed.weight" => return Some(vec!["v.position_embd.weight".into()]),
        "model.visual.merger.norm.weight" => return Some(vec!["v.post_ln.weight".into()]),
        "model.visual.merger.norm.bias" => return Some(vec!["v.post_ln.bias".into()]),
        "model.visual.merger.linear_fc1.weight" => return Some(vec!["mm.0.weight".into()]),
        "model.visual.merger.linear_fc1.bias" => return Some(vec!["mm.0.bias".into()]),
        "model.visual.merger.linear_fc2.weight" => return Some(vec!["mm.2.weight".into()]),
        "model.visual.merger.linear_fc2.bias" => return Some(vec!["mm.2.bias".into()]),
        _ => {}
    }
    let rest = hf.strip_prefix("model.visual.blocks.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<u32>().ok()?;
    let g = match suffix {
        "norm1.weight" => "ln1.weight",
        "norm1.bias" => "ln1.bias",
        "norm2.weight" => "ln2.weight",
        "norm2.bias" => "ln2.bias",
        "attn.qkv.weight" => "attn_qkv.weight",
        "attn.qkv.bias" => "attn_qkv.bias",
        "attn.proj.weight" => "attn_out.weight",
        "attn.proj.bias" => "attn_out.bias",
        "mlp.linear_fc1.weight" => "ffn_up.weight",
        "mlp.linear_fc1.bias" => "ffn_up.bias",
        "mlp.linear_fc2.weight" => "ffn_down.weight",
        "mlp.linear_fc2.bias" => "ffn_down.bias",
        _ => return None,
    };
    Some(vec![format!("v.blk.{idx}.{g}")])
}

/// HF-имя тензора MTP → имя в эталонном MTP-GGUF.
///
/// `block` — индекс слоя MTP (в эталоне это blk.32: block_count включает
/// nextn-слой). Формы подтверждают карту: `nextn.eh_proj` имеет вход 5120 =
/// 2×2560, то есть это и есть `mtp.fc`, склеивающий эмбеддинг с hidden.
pub fn hf_to_gguf_mtp(hf: &str, block: u32) -> Option<Vec<String>> {
    let one = |s: String| Some(vec![s]);
    match hf {
        "mtp.fc.weight" => return one(format!("blk.{block}.nextn.eh_proj.weight")),
        "mtp.pre_fc_norm_embedding.weight" => {
            return one(format!("blk.{block}.nextn.enorm.weight"))
        }
        "mtp.pre_fc_norm_hidden.weight" => return one(format!("blk.{block}.nextn.hnorm.weight")),
        "mtp.norm.weight" => return one(format!("blk.{block}.nextn.shared_head_norm.weight")),
        _ => {}
    }
    let rest = hf.strip_prefix("mtp.layers.")?;
    let (idx, suffix) = rest.split_once('.')?;
    idx.parse::<u32>().ok()?;
    let g = match suffix {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        _ => return None,
    };
    one(format!("blk.{block}.{g}"))
}

/// Индекс слоя MTP в эталонном GGUF: единственный блок с тензорами `nextn.*`.
pub fn mtp_block_of(ct: &gguf_file::Content) -> Option<u32> {
    ct.tensor_infos
        .keys()
        .filter_map(|n| {
            let rest = n.strip_prefix("blk.")?;
            let (idx, tail) = rest.split_once('.')?;
            tail.starts_with("nextn.").then(|| idx.parse::<u32>().ok())?
        })
        .max()
}

/// Срез Conv3d по временной оси: [O, C, T, H, W] → T тензоров [O, C, H, W].
/// Данные в HF идут row-major, поэтому элемент (o,c,t,h,w) лежит по индексу
/// (((o*C + c)*T + t)*H + h)*W + w — простой копией срез не возьмёшь.
fn split_conv3d_temporal(values: &[f32], shape: &[usize]) -> Result<Vec<Vec<f32>>, String> {
    if shape.len() != 5 {
        return Err(format!("patch_embed: ожидалась форма [O,C,T,H,W], пришла {shape:?}"));
    }
    let (o, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    if values.len() != o * c * t * h * w {
        return Err("patch_embed: длина не бьётся с формой".into());
    }
    let hw = h * w;
    let mut out = vec![Vec::with_capacity(o * c * hw); t];
    for oi in 0..o {
        for ci in 0..c {
            for ti in 0..t {
                let base = (((oi * c + ci) * t + ti) * hw) as usize;
                out[ti].extend_from_slice(&values[base..base + hw]);
            }
        }
    }
    Ok(out)
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
        // Знаковые: неотрицательные уходят как u64 (читатель развернёт их в
        // U32, как ждёт конструктор), отрицательные — как i64.
        V::I8(x) => i64::from(*x).into(),
        V::I16(x) => i64::from(*x).into(),
        V::I32(x) => i64::from(*x).into(),
        V::I64(x) => (*x).into(),
        V::F32(x) => serde_json::Number::from_f64(*x as f64).map(Into::into)?,
        V::F64(x) => serde_json::Number::from_f64(*x).map(Into::into)?,
        V::Bool(x) => (*x).into(),
        V::String(s) => s.clone().into(),
        // Массивы нужны как есть: у видео-башни это image_mean/image_std —
        // константы нормализации, без них препроцессинг неверен.
        V::Array(items) => serde_json::Value::Array(
            items.iter().filter_map(value_to_json).collect(),
        ),
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


/// Перестановка v-голов по f32: HF чередует головы внутри k-группы, GGUF
/// хранит их подряд. Та же перестановка, что в `deinterleave_blocks`, но без
/// промежуточного F16 — у скаляров DeltaNet цель F32, и round-trip через
/// половинную точность их бы огрубил.
fn reorder_v_heads_f32(src: &[f32], n_k: usize, n_per_k: usize, block: usize) -> Vec<f32> {
    let heads = n_k * n_per_k;
    if src.len() != heads * block {
        return src.to_vec();
    }
    let mut out = Vec::with_capacity(src.len());
    for j in 0..n_per_k {
        for g in 0..n_k {
            let hf = g * n_per_k + j;
            out.extend_from_slice(&src[hf * block..(hf + 1) * block]);
        }
    }
    out
}

/// Преобразования весов, которые конвертер llama.cpp применяет при записи GGUF
/// (`convert_hf_to_gguf.py`, `Qwen3NextModel.modify_tensors`). Без них
/// контейнер грузится и даже проходит проверки имён и форм, но модель выдаёт
/// мусор: расхождение сидит в скалярах DeltaNet и в нормах.
///
///   .A_log            → -exp(x), затем перестановка v-голов
///   .dt_bias          → перестановка v-голов
///   *norm.weight      → x + 1, КРОМЕ linear_attn.norm.weight
///   conv1d            → squeeze + перестановка только v-канальной части
fn apply_hf_transforms(
    hf: &str,
    mut values: Vec<f32>,
    shape: &[usize],
    lay: Option<crate::DeltaLayout>,
) -> Vec<f32> {
    // Нормы MTP-головы называются иначе: pre_fc_norm_embedding.weight и
    // pre_fc_norm_hidden.weight. Проверка по "norm.weight" их не ловила, и они
    // уходили в артефакт без сдвига — сверка показывала 185% и 119% расхождения,
    // а голова давала ноль принятых черновиков.
    let is_norm = hf.ends_with("norm.weight")
        || hf.ends_with("pre_fc_norm_embedding.weight")
        || hf.ends_with("pre_fc_norm_hidden.weight");
    if is_norm && !hf.ends_with("linear_attn.norm.weight") {
        for v in values.iter_mut() {
            *v += 1.0;
        }
        return values;
    }
    if hf.ends_with(".A_log") {
        for v in values.iter_mut() {
            *v = -v.exp();
        }
    }
    let Some(l) = lay else { return values };
    let n_per_k = l.n_per_k();
    if n_per_k <= 1 {
        return values;
    }
    if hf.ends_with(".A_log") || hf.ends_with(".dt_bias") {
        // 1-D по числу v-голов: блок в один элемент.
        return reorder_v_heads_f32(&values, l.n_k, n_per_k, 1);
    }
    if hf.ends_with("conv1d.weight") {
        // [C, K] после схлопывания: q/k-часть остаётся, переставляется хвост v.
        let cols = *shape.last().unwrap_or(&1);
        let qk = l.hk * l.n_k * 2 * cols;
        if qk >= values.len() {
            return values;
        }
        let mut out = values[..qk].to_vec();
        out.extend(reorder_v_heads_f32(
            &values[qk..],
            l.n_k,
            n_per_k,
            l.hv * cols,
        ));
        return out;
    }
    values
}

/// Рецепт квантования: тип на каждый тензор.
///
/// Декод упирается в полосу памяти (68% GPU-времени шага — чтение весов),
/// поэтому скорость определяют байты, а не формат вычислений. Значит выбор
/// рецепта — это торговля «перплексия против байтов», и мерить надо оба.
///
/// Одномерные тензоры (нормы, ssm_a, dt_bias) в GGUF лежат в F32 и рецептом
/// не трогаются: они крошечные, а их огрубление бьёт по качеству сильнее
/// всего.
pub fn recipe_dtype(recipe: &str, gguf_name: &str, mirror: GgmlDType) -> GgmlDType {
    if matches!(mirror, GgmlDType::F32 | GgmlDType::F16) {
        return mirror;
    }
    let is = |suffix: &str| gguf_name.ends_with(suffix);
    let is_delta_proj = is("attn_qkv.weight")
        || is("attn_gate.weight")
        || is("ssm_beta.weight")
        || is("ssm_alpha.weight")
        || is("ssm_out.weight");
    let is_ffn = is("ffn_gate.weight") || is("ffn_up.weight") || is("ffn_down.weight");
    match recipe {
        // Как в эталонном GGUF — базовая точка отсчёта.
        "mirror" => mirror,
        // Всё в Q4_K: минимум байтов, максимум скорости декода.
        "q4" => GgmlDType::Q4K,
        // Всё в Q8_0. Для MTP-головы: квант черновика входит в ускорение
        // множителем через долю принятия, а сама голова мала (15 тензоров
        // одного блока), поэтому экономить на её точности невыгодно. Замер на
        // Q4-голове при Q8-модели дал принятие 66% на 8K и 37% на 32K.
        "q8" => GgmlDType::Q8_0,
        // Внимание и DeltaNet точнее, FFN (основной объём весов) в Q4_K.
        "q5-attn" => {
            if is_ffn {
                GgmlDType::Q4K
            } else {
                GgmlDType::Q5K
            }
        }
        // Голова точнее остального: она читается каждый шаг целиком.
        "q6-head" => {
            if gguf_name == "token_embd.weight" {
                GgmlDType::Q6K
            } else {
                mirror
            }
        }
        // Точечный int8 на проекциях DeltaNet — идея «int8 бесплатен по
        // компьюту»; проверяем, стоит ли он удвоения байтов на этой группе.
        "q8-delta" => {
            if is_delta_proj {
                GgmlDType::Q8_0
            } else {
                mirror
            }
        }
        _ => mirror,
    }
}

/// Сверка формы с эталоном. HF держит conv1d как [C, 1, K], GGUF — как [C, K].
/// Единичные измерения схлопываем, но только если порядок и число элементов
/// совпали: иначе это настоящее расхождение раскладки, и молчать нельзя.
fn reconcile_shape(name: &str, shape: &[usize], ref_dims: &[usize]) -> Result<Vec<usize>, String> {
    if shape == ref_dims {
        return Ok(shape.to_vec());
    }
    let squeeze = |d: &[usize]| -> Vec<usize> { d.iter().copied().filter(|x| *x != 1).collect() };
    if squeeze(shape) == squeeze(ref_dims)
        && shape.iter().product::<usize>() == ref_dims.iter().product::<usize>()
    {
        Ok(ref_dims.to_vec())
    } else {
        Err(format!("{name}: форма {shape:?} не совпадает с эталоном {ref_dims:?}"))
    }
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
    fuse_in_proj: bool,
    recipe: &str,
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

    // Значения тензора, готовые к квантованию: чтение + преобразования
    // llama.cpp + перепаковка раскладки + сверка формы с эталоном.
    let prepare = |hf: &str, ref_dims: &[usize]| -> Result<(Vec<f32>, Vec<usize>), String> {
        let (values, shape) = read_tensor_f32(shards, hf)?;
        let shape = reconcile_shape(hf, &shape, ref_dims)?;
        let values = apply_hf_transforms(hf, values, &shape, delta_layout);
        let values = if delta_layout.is_some() && shape.len() == 2 {
            let gguf_name = hf_to_gguf(hf).unwrap_or_default();
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
        Ok((values, shape))
    };

    // hf-имя каждого записанного тензора — нужно слитой проекции.
    let mut written: BTreeMap<String, String> = BTreeMap::new();

    for hf in hf_names {
        let Some(gguf_name) = hf_to_gguf(hf) else {
            stats.skipped.push(hf.clone());
            continue;
        };
        let info = ct.tensor_infos.get(&gguf_name).ok_or_else(|| {
            format!("{hf} → {gguf_name}: в эталонном GGUF такого тензора нет")
        })?;
        let (values, shape) = read_tensor_f32(shards, hf)?;
        let shape = reconcile_shape(&gguf_name, &shape, info.shape.dims())?;
        // Преобразования llama.cpp (нормы, A_log, dt_bias, conv1d) — по f32.
        let values = apply_hf_transforms(hf, values, &shape, delta_layout);
        // Перепаковка проекций DeltaNet работает по байтам F16 — применяем её
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
        let dt = recipe_dtype(recipe, &gguf_name, info.ggml_dtype);
        let bytes = quantize_bytes(values, &shape, dt)?;
        let dn = dtype_name(dt);
        w.add_typed(&gguf_name, &shape, dn, &bytes)
            .map_err(|e| format!("{gguf_name}: запись: {e}"))?;
        written.insert(gguf_name.clone(), hf.clone());
        stats.tensors += 1;
        stats.bytes += bytes.len() as u64;
        *stats.by_dtype.entry(dn.to_string()).or_insert(0) += 1;
    }

    if fuse_in_proj {
        // Слитая проекция qkv+z+b+a: одна матрица вместо четырёх.
        // Дополняем число строк до кратности 128 — иначе матмуль уходит с
        // MMA-пути (гейт n % 128 == 0) и становится медленнее раздельных:
        // замер на M=512 Q8_0 дал -67.5% без дополнения и +27.8% с ним.
        let parts = ["attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"];
        let mut layers = 0usize;
        for blk in 0..1024usize {
            let names: Vec<String> = parts.iter().map(|p| format!("blk.{blk}.{p}")).collect();
            if !names.iter().all(|n| written.contains_key(n)) {
                continue;
            }
            let mut fused: Vec<f32> = Vec::new();
            let mut cols = 0usize;
            let mut offsets: Vec<usize> = Vec::new();
            let mut dt = GgmlDType::Q4K;
            for (i, n) in names.iter().enumerate() {
                let hf = &written[n];
                let info = ct.tensor_infos.get(n).unwrap();
                if i == 0 {
                    dt = recipe_dtype(recipe, n, info.ggml_dtype);
                    cols = info.shape.dims()[1];
                }
                let (v, sh) = prepare(hf, info.shape.dims())?;
                if sh[1] != cols {
                    return Err(format!("{n}: ширина {} != {cols}", sh[1]));
                }
                offsets.push(fused.len() / cols);
                fused.extend_from_slice(&v);
            }
            let rows = fused.len() / cols;
            let padded = rows.div_ceil(128) * 128;
            fused.resize(padded * cols, 0.0);
            let bytes = quantize_bytes(fused, &[padded, cols], dt)?;
            let name = format!("blk.{blk}.attn_in_proj.weight");
            w.add_typed(&name, &[padded, cols], dtype_name(dt), &bytes)
                .map_err(|e| format!("{name}: запись: {e}"))?;
            stats.bytes += bytes.len() as u64;
            stats.tensors += 1;
            if layers == 0 {
                println!(
                    "слитая проекция: [{padded}, {cols}] (строк {rows}, дополнено {}), смещения {offsets:?}, тип {}",
                    padded - rows,
                    dtype_name(dt)
                );
            }
            layers += 1;
        }
        println!("слитых проекций записано: {layers}");
    }

    let tok = std::fs::read(tokenizer_json)
        .map_err(|e| format!("read {}: {e}", tokenizer_json.display()))?;
    w.add_typed(crate::pack::TOKENIZER_BLOB, &[tok.len()], "RAW", &tok)
        .map_err(|e| format!("токенизатор: запись: {e}"))?;
    stats.bytes += tok.len() as u64;

    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(stats)
}

/// Имя блоба токенизатора внутри data-секции (совпадает с читателем движка).
pub const TOKENIZER_BLOB: &str = "__tokenizer_json";

#[cfg(test)]
mod tests {
    use super::{hf_to_gguf, recipe_dtype};
    use candle_core::quantized::GgmlDType;

    /// Одномерные тензоры (нормы, ssm_a, dt_bias) лежат в F32 и рецептом не
    /// трогаются: они крошечные, а огрубление бьёт по качеству сильнее всего.
    #[test]
    fn recipes_never_quantize_f32_scalars() {
        for r in ["mirror", "q4", "q5-attn", "q6-head", "q8-delta"] {
            assert_eq!(
                recipe_dtype(r, "blk.0.attn_norm.weight", GgmlDType::F32),
                GgmlDType::F32,
                "рецепт {r} тронул норму"
            );
            assert_eq!(
                recipe_dtype(r, "blk.0.ssm_a", GgmlDType::F32),
                GgmlDType::F32,
                "рецепт {r} тронул ssm_a"
            );
        }
    }

    #[test]
    fn recipes_do_what_they_claim() {
        let ffn = "blk.7.ffn_down.weight";
        let qkv = "blk.7.attn_qkv.weight";
        let head = "token_embd.weight";
        // mirror ничего не меняет
        assert_eq!(recipe_dtype("mirror", ffn, GgmlDType::Q6K), GgmlDType::Q6K);
        // q4 сводит всё квантуемое к Q4_K
        assert_eq!(recipe_dtype("q4", ffn, GgmlDType::Q6K), GgmlDType::Q4K);
        // q5-attn: FFN остаётся Q4_K, внимание и DeltaNet поднимаются до Q5_K
        assert_eq!(recipe_dtype("q5-attn", ffn, GgmlDType::Q6K), GgmlDType::Q4K);
        assert_eq!(recipe_dtype("q5-attn", qkv, GgmlDType::Q4K), GgmlDType::Q5K);
        // q6-head трогает только голову
        assert_eq!(recipe_dtype("q6-head", head, GgmlDType::Q4K), GgmlDType::Q6K);
        assert_eq!(recipe_dtype("q6-head", ffn, GgmlDType::Q4K), GgmlDType::Q4K);
        // q8-delta — точечный int8 только на проекциях DeltaNet
        assert_eq!(recipe_dtype("q8-delta", qkv, GgmlDType::Q4K), GgmlDType::Q8_0);
        assert_eq!(recipe_dtype("q8-delta", ffn, GgmlDType::Q4K), GgmlDType::Q4K);
        // неизвестное имя не должно молча портить модель — ведёт себя как mirror
        assert_eq!(recipe_dtype("нет-такого", ffn, GgmlDType::Q6K), GgmlDType::Q6K);
    }

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

    /// У 4B эмбеддинги связаны и головы нет, у 9B она отдельная. Без этой
    /// карты контейнер 9B молча остался бы со связанными эмбеддингами —
    /// падения не будет, будет тихо худшее качество.
    #[test]
    fn separate_output_head_is_mapped() {
        assert_eq!(hf_to_gguf("lm_head.weight").as_deref(), Some("output.weight"));
        assert_eq!(
            hf_to_gguf("model.lm_head.weight").as_deref(),
            Some("output.weight")
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

/// Сверить готовый контейнер с эталонным GGUF потензорно.
///
/// Оба файла квантованы из одних весов, поэтому расхождение должно быть на
/// уровне шума квантования (единицы процентов). Десятки и сотни процентов
/// означают, что тензор попал не туда или не в той раскладке — именно такую
/// ошибку не ловят проверки имён и форм.
pub fn verify(ytf: &Path, ref_gguf: &Path, top: usize) -> Result<(), String> {
    let cont = container::Reader::open(ytf)?;
    let mut gf = std::fs::File::open(ref_gguf)
        .map_err(|e| format!("open {}: {e}", ref_gguf.display()))?;
    let ct = gguf_file::Content::read(&mut gf).map_err(|e| format!("read ref gguf: {e}"))?;
    let gguf_mmap = {
        let f = std::fs::File::open(ref_gguf).map_err(|e| format!("open: {e}"))?;
        unsafe { memmap2::MmapOptions::new().map(&f) }.map_err(|e| format!("mmap: {e}"))?
    };

    let dev = Device::Cpu;
    let mut rows: Vec<(f64, String, String)> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for entry in &cont.manifest.tensors {
        if entry.dtype == "RAW" {
            continue;
        }
        let Some(info) = ct.tensor_infos.get(&entry.name) else {
            missing.push(entry.name.clone());
            continue;
        };
        let want = info
            .read_from_slice(&gguf_mmap, ct.tensor_data_offset, &dev)
            .map_err(|e| format!("{}: чтение эталона: {e}", entry.name))?
            .dequantize(&dev)
            .map_err(|e| format!("{}: деквант эталона: {e}", entry.name))?
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("{}: вектор эталона: {e}", entry.name))?;

        let (bytes, shape) = cont
            .tensor(&entry.name)
            .ok_or_else(|| format!("{}: нет в контейнере", entry.name))?;
        let dt = match entry.dtype.as_str() {
            "F32" => GgmlDType::F32,
            "F16" => GgmlDType::F16,
            "BF16" => GgmlDType::BF16,
            "Q4_0" => GgmlDType::Q4_0,
            "Q8_1" => GgmlDType::Q8_1,
            "Q4_K" => GgmlDType::Q4K,
            "Q5_K" => GgmlDType::Q5K,
            "Q6_K" => GgmlDType::Q6K,
            "Q8_0" => GgmlDType::Q8_0,
            other => return Err(format!("{}: тип {other} не поддержан сверкой", entry.name)),
        };
        let got = candle_core::quantized::ggml_file::qtensor_from_ggml(dt, bytes, shape.to_vec(), &dev)
            .map_err(|e| format!("{}: разбор блоков: {e}", entry.name))?
            .dequantize(&dev)
            .map_err(|e| format!("{}: деквант: {e}", entry.name))?
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| format!("{}: вектор: {e}", entry.name))?;

        if got.len() != want.len() {
            rows.push((f64::INFINITY, entry.name.clone(), format!("длина {} vs {}", got.len(), want.len())));
            continue;
        }
        let mut num = 0f64;
        let mut den = 0f64;
        for (a, b) in got.iter().zip(want.iter()) {
            num += (*a as f64 - *b as f64).abs();
            den += (*b as f64).abs();
        }
        let rel = 100.0 * num / den.max(1e-12);
        rows.push((rel, entry.name.clone(), format!("{:?}", shape)));
    }

    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!("сверено тензоров: {}", rows.len());
    if !missing.is_empty() {
        println!("нет в эталоне: {} ({:?})", missing.len(), &missing[..missing.len().min(5)]);
    }
    println!("худшие расхождения (относительная ошибка значений):");
    for (rel, name, shape) in rows.iter().take(top) {
        println!("  {rel:8.2}%  {name}  {shape}");
    }
    let bad = rows.iter().filter(|r| r.0 > 15.0).count();
    println!(
        "медиана {:.2}%, выше 15%: {bad}",
        rows.get(rows.len() / 2).map(|r| r.0).unwrap_or(0.0)
    );
    if bad > 0 {
        return Err(format!("{bad} тензоров расходятся с эталоном сильнее шума квантования"));
    }
    Ok(())
}

/// Напечатать тензоры GGUF: имя, тип, форма. Карта имён строится по факту.
pub fn list_gguf(path: &Path) -> Result<(), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let ct = gguf_file::Content::read(&mut f).map_err(|e| format!("read gguf: {e}"))?;
    let mut names: Vec<&String> = ct.tensor_infos.keys().collect();
    names.sort();
    println!("тензоров: {}", names.len());
    for n in names {
        let i = &ct.tensor_infos[n];
        println!("  {:<40} {:<6} {:?}", n, dtype_name(i.ggml_dtype), i.shape.dims());
    }
    let mut keys: Vec<&String> = ct.metadata.keys().collect();
    keys.sort();
    println!("метаданных: {}", keys.len());
    Ok(())
}

/// Собрать контейнер компонента (видео-башня, MTP) из safetensors.
///
/// Эталон — GGUF компонента: из него берутся типы тензоров и метаданные.
/// Токенизатор компонентам не нужен. Формы сверяются с эталоном, как и у
/// языковой модели: расхождение раскладки — ошибка, а не тихий сдвиг.
///
/// `map` возвращает список имён, потому что одна HF-матрица может давать
/// несколько тензоров эталона (Conv3d видео-башни режется по временной оси).
/// `require_all` — падать, если в контейнер не попал тензор эталона; для MTP
/// это неверно: его эталон содержит всю модель, а нам нужен только nextn-слой.
#[allow(clippy::too_many_arguments)]
pub fn pack_component(
    shards: &[(String, memmap2::Mmap)],
    hf_names: &[String],
    ref_gguf: &Path,
    out_path: &Path,
    recipe: &str,
    mask: &str,
    map: &dyn Fn(&str, &gguf_file::Content) -> Option<Vec<String>>,
    require_all: bool,
    read_tensor_f32: &dyn Fn(
        &[(String, memmap2::Mmap)],
        &str,
    ) -> Result<(Vec<f32>, Vec<usize>), String>,
) -> Result<PackStats, String> {
    let mut gf = std::fs::File::open(ref_gguf)
        .map_err(|e| format!("open {}: {e}", ref_gguf.display()))?;
    let ct = gguf_file::Content::read(&mut gf).map_err(|e| format!("read эталон: {e}"))?;

    let mut config: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut dropped: Vec<String> = Vec::new();
    for (k, v) in ct.metadata.iter() {
        match value_to_json(v) {
            Some(j) => {
                config.insert(k.clone(), j);
            }
            // Массивы (image_mean/std и подобное) в манифест не переносятся —
            // печатаем их, чтобы потеря была видна.
            None => dropped.push(k.clone()),
        }
    }
    if !dropped.is_empty() {
        dropped.sort();
        return Err(format!("метаданные не перенесены: {dropped:?}"));
    }

    let out_file = std::fs::File::create(out_path)
        .map_err(|e| format!("create {}: {e}", out_path.display()))?;
    let pre = container::ManifestPre {
        gguf_sha256: String::new(),
        mask: mask.to_string(),
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
        let Some(targets) = map(hf, &ct) else {
            stats.skipped.push(hf.clone());
            continue;
        };
        let (values, shape) = read_tensor_f32(shards, hf)?;
        // Преобразования HF → GGUF обязательны и для компонентов, не только для
        // основной модели: нормы хранятся со сдвигом и требуют +1. Без этого
        // упакованная MTP-голова давала ноль принятых черновиков — сверка с
        // эталоном показывала расхождение 44-83% именно на нормах при 2-8% на
        // матрицах, то есть на уровне шума квантования.
        let values = apply_hf_transforms(hf, values, &shape, None);
        // Одна HF-матрица может давать несколько тензоров эталона: Conv3d
        // patch_embed режется по временной оси.
        let parts: Vec<Vec<f32>> = if targets.len() > 1 {
            split_conv3d_temporal(&values, &shape)?
        } else {
            vec![values]
        };
        if parts.len() != targets.len() {
            return Err(format!(
                "{hf}: срезов {} против {} имён",
                parts.len(),
                targets.len()
            ));
        }
        for (name, part) in targets.iter().zip(parts) {
            let info = ct
                .tensor_infos
                .get(name)
                .ok_or_else(|| format!("{hf} → {name}: в эталоне такого тензора нет"))?;
            let dims = info.shape.dims().to_vec();
            if part.len() != dims.iter().product::<usize>() {
                return Err(format!(
                    "{name}: элементов {} против эталонных {dims:?}",
                    part.len()
                ));
            }
            let dt = recipe_dtype(recipe, name, info.ggml_dtype);
            let bytes = quantize_bytes(part, &dims, dt)?;
            let dn = dtype_name(dt);
            w.add_typed(name, &dims, dn, &bytes)
                .map_err(|e| format!("{name}: запись: {e}"))?;
            stats.tensors += 1;
            stats.bytes += bytes.len() as u64;
            *stats.by_dtype.entry(dn.to_string()).or_insert(0) += 1;
        }
    }

    // Всё, что есть в эталоне, должно быть в контейнере: иначе компонент не
    // соберётся, а узнаем мы об этом только при загрузке.
    if require_all {
        let mut missing: Vec<&String> = ct
            .tensor_infos
            .keys()
            .filter(|n| !w.has_tensor(n))
            .collect();
        if !missing.is_empty() {
            missing.sort();
            return Err(format!("в контейнер не попали тензоры эталона: {missing:?}"));
        }
    }

    w.finalize().map_err(|e| format!("finalize: {e}"))?;
    Ok(stats)
}
