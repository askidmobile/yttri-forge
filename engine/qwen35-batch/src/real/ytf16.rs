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
