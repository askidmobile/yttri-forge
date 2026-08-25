//! Читатель .ytf16 сайдкара (формат YTF1, v1).
//!
//! Контейнер производит forge-convert (репозиторий yttri-forge):
//!   [0..4)   magic "YTF1"
//!   [4..8)   version u32 = 1
//!   [8..12)  manifest_len u32 (резервированное окно, JSON + нулевой паддинг)
//!   [12..12+mlen)      манифест JSON: { gguf_sha256, mask, tensors:[{name,shape,offset,len}] }
//!   далее              F16 LE буферы тензоров (выравнивание 64B)
//!
//! Dual-read инвариант: этот модуль используется ТОЛЬКО prefill-путём
//! (см. спеку yttri-forge этапа 1, R-DUAL). Decode читает GGUF-квант.

use candle_core::Result;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"YTF1";
pub const SUPPORTED_VERSION: u32 = 1;

#[derive(Debug)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    /// Смещение от начала data-секции
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug)]
pub struct Manifest {
    pub gguf_sha256: String,
    pub mask: String,
}

fn parse_manifest(bytes: &[u8]) -> Result<(Manifest, Vec<(String, TensorInfo)>)> {
    let v: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| candle_core::Error::Msg(format!("ytf16 manifest parse: {e}")))?;
    let obj = v.as_object().ok_or_else(|| {
        candle_core::Error::Msg("ytf16 manifest: not an object".into())
    })?;
    let gguf_sha256 = obj
        .get("gguf_sha256")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let mask = obj
        .get("mask")
        .and_then(|x| x.as_str())
        .unwrap_or_default()
        .to_string();
    let mut infos = Vec::new();
    if let Some(arr) = obj.get("tensors").and_then(|x| x.as_array()) {
        for t in arr {
            let name = t.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let offset = t.get("offset").and_then(|x| x.as_u64()).unwrap_or(0);
            let len = t.get("len").and_then(|x| x.as_u64()).unwrap_or(0);
            let shape: Vec<usize> = t
                .get("shape")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|d| d.as_u64().map(|v| v as usize)).collect())
                .unwrap_or_default();
            infos.push((name, TensorInfo { shape, offset, len }));
        }
    }
    Ok((Manifest { gguf_sha256, mask }, infos))
}

pub struct Ytf16Sidecar {
    mmap: memmap2::Mmap,
    pub manifest: Manifest,
    data_start: u64,
    index: std::collections::HashMap<String, TensorInfo>,
}

impl Ytf16Sidecar {
    /// Открыть и провалидировать контейнер.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| candle_core::Error::Msg(format!("ytf16 open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| {
            candle_core::Error::Msg(format!("ytf16 mmap {}: {e}", path.display()))
        })?;
        if mmap.len() < 12 || &mmap[0..4] != MAGIC {
            return Err(candle_core::Error::Msg(format!(
                "ytf16 {}: not a YTF1 container",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != SUPPORTED_VERSION {
            return Err(candle_core::Error::Msg(format!(
                "ytf16 {}: unsupported version {version}",
                path.display()
            )));
        }
        let mlen = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let end = 12usize.checked_add(mlen).ok_or_else(|| {
            candle_core::Error::Msg("ytf16: manifest length overflow".into())
        })?;
        if mmap.len() < end {
            return Err(candle_core::Error::Msg("ytf16: truncated manifest".into()));
        }
        let raw = &mmap[12..end];
        let json_end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let (manifest, tensor_infos) = parse_manifest(&raw[..json_end])?;
        let mut index = std::collections::HashMap::with_capacity(tensor_infos.len());
        for (name, info) in tensor_infos {
            index.insert(name, info);
        }
        Ok(Self { mmap, manifest, data_start: end as u64, index })
    }

    /// sha256 GGUF из манифеста.
    pub fn gguf_sha256(&self) -> &str {
        &self.manifest.gguf_sha256
    }

    /// Байты тензора (F16 LE) по имени.
    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], &[usize])> {
        let info = self.index.get(name)?;
        let start = self.data_start + info.offset;
        let end = start + info.len;
        if end > self.mmap.len() as u64 {
            return None;
        }
        Some((&self.mmap[start as usize..end as usize], &info.shape))
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }
}

// ══════════════════ v2: самостоятельная модель ══════════════════
//
// В отличие от сайдкара (v1), контейнер v2 несёт всю модель: у каждого
// тензора свой GGML-тип, данные лежат готовыми блоками, гиперпараметры и
// токенизатор едут внутри. GGUF рядом не нужен, dual-read не нужен —
// один контур.
//
// Загрузка сводится к синтезу `gguf_file::Content` из манифеста: поля у него
// публичные, а конструктор модели (`build_model_common`) обращается к
// метаданным и тензорам только через них. Данные при этом не конвертируются:
// `qtensor_from_ggml` кладёт блоки прямо в VRAM.

use candle_core::quantized::gguf_file;
use candle_core::quantized::GgmlDType;
use std::collections::HashMap;

pub const VERSION_STANDALONE: u32 = 2;

/// Имя блоба с встроенным HF-токенизатором внутри data-секции.
pub const TOKENIZER_BLOB: &str = "__tokenizer_json";

fn ggml_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name {
        "F32" => GgmlDType::F32,
        "F16" => GgmlDType::F16,
        "Q4_0" => GgmlDType::Q4_0,
        "Q4_1" => GgmlDType::Q4_1,
        "Q5_0" => GgmlDType::Q5_0,
        "Q5_1" => GgmlDType::Q5_1,
        "Q8_0" => GgmlDType::Q8_0,
        "Q8_1" => GgmlDType::Q8_1,
        "Q2_K" => GgmlDType::Q2K,
        "Q3_K" => GgmlDType::Q3K,
        "Q4_K" => GgmlDType::Q4K,
        "Q5_K" => GgmlDType::Q5K,
        "Q6_K" => GgmlDType::Q6K,
        "Q8_K" => GgmlDType::Q8K,
        other => {
            return Err(candle_core::Error::Msg(format!(
                "ytf v2: неизвестный тип тензора {other}"
            )))
        }
    })
}

/// JSON-значение конфига → значение GGUF-метаданных. Целое без дробной части
/// становится U32, дробное — F32: ровно те типы, которые читает конструктор.
fn config_value(v: &serde_json::Value) -> Option<gguf_file::Value> {
    match v {
        serde_json::Value::String(s) => Some(gguf_file::Value::String(s.clone())),
        serde_json::Value::Bool(b) => Some(gguf_file::Value::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Some(gguf_file::Value::U32(u as u32))
            } else {
                n.as_f64().map(|f| gguf_file::Value::F32(f as f32))
            }
        }
        _ => None,
    }
}

/// Версия контейнера без разбора манифеста (дешёвая проверка при выборе пути).
pub fn container_version(data: &[u8]) -> Option<u32> {
    if data.len() < 12 || &data[0..4] != MAGIC {
        return None;
    }
    Some(u32::from_le_bytes(data[4..8].try_into().ok()?))
}

/// Собрать `gguf_file::Content` из контейнера v2.
///
/// `data` — весь mmap файла. Возвращённый Content ссылается на тот же буфер:
/// `tensor_data_offset` указывает на начало data-секции, смещения тензоров
/// отсчитываются от неё, как и в GGUF.
pub fn content_from_standalone(data: &[u8]) -> Result<gguf_file::Content> {
    match container_version(data) {
        Some(v) if v == VERSION_STANDALONE => {}
        Some(v) => {
            return Err(candle_core::Error::Msg(format!(
                "ytf: версия {v} не самостоятельная модель (нужна {VERSION_STANDALONE})"
            )))
        }
        None => return Err(candle_core::Error::Msg("ytf: не тот контейнер".into())),
    }
    let mlen = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let end = 12usize
        .checked_add(mlen)
        .filter(|e| *e <= data.len())
        .ok_or_else(|| candle_core::Error::Msg("ytf: манифест обрезан".into()))?;
    // Окно манифеста дополнено нулями — режем по первому \0.
    let raw = &data[12..end];
    let json_end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    let manifest: serde_json::Value = serde_json::from_slice(&raw[..json_end])
        .map_err(|e| candle_core::Error::Msg(format!("ytf: манифест: {e}")))?;

    let mut metadata: HashMap<String, gguf_file::Value> = HashMap::new();
    if let Some(cfg) = manifest.get("config").and_then(|c| c.as_object()) {
        for (k, v) in cfg {
            if let Some(val) = config_value(v) {
                metadata.insert(k.clone(), val);
            }
        }
    }

    let data_start = end as u64;
    let mut tensor_infos: HashMap<String, gguf_file::TensorInfo> = HashMap::new();
    let mut tokenizer_span: Option<(usize, usize)> = None;
    let arr = manifest
        .get("tensors")
        .and_then(|t| t.as_array())
        .ok_or_else(|| candle_core::Error::Msg("ytf: нет списка тензоров".into()))?;
    for t in arr {
        let name = t.get("name").and_then(|x| x.as_str()).unwrap_or("");
        let offset = t.get("offset").and_then(|x| x.as_u64()).unwrap_or(0);
        let len = t.get("len").and_then(|x| x.as_u64()).unwrap_or(0);
        let start = (data_start + offset) as usize;
        let stop = start + len as usize;
        if stop > data.len() {
            return Err(candle_core::Error::Msg(format!(
                "ytf: тензор {name} выходит за файл ({stop} > {})",
                data.len()
            )));
        }
        if name == TOKENIZER_BLOB {
            tokenizer_span = Some((start, stop));
            continue;
        }
        let shape: Vec<usize> = t
            .get("shape")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|d| d.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_default();
        let dtype = ggml_dtype(t.get("dtype").and_then(|x| x.as_str()).unwrap_or("F16"))?;
        tensor_infos.insert(
            name.to_string(),
            gguf_file::TensorInfo {
                ggml_dtype: dtype,
                shape: candle_core::Shape::from_dims(&shape),
                offset,
            },
        );
    }

    // Токенизатор кладём в метаданные тем же ключом, который уже умеет читать
    // построитель токенизатора (`tokenizer.huggingface.json`).
    if let Some((s, e)) = tokenizer_span {
        match std::str::from_utf8(&data[s..e]) {
            Ok(json) => {
                metadata.insert(
                    "tokenizer.huggingface.json".to_string(),
                    gguf_file::Value::String(json.to_string()),
                );
            }
            Err(err) => {
                return Err(candle_core::Error::Msg(format!(
                    "ytf: блоб токенизатора не UTF-8: {err}"
                )))
            }
        }
    }

    Ok(gguf_file::Content {
        magic: gguf_file::VersionedMagic::GgufV3,
        metadata,
        tensor_infos,
        tensor_data_offset: data_start,
    })
}
