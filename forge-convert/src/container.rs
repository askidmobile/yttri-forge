//! Контейнер .ytf16 (формат v1).
//!
//! Layout:
//!   [0..4)   magic "YTF1" (LE)
//!   [4..8)   version u32 = 1
//!   [8..12)  manifest_len u32 (= RESERVE_MANIFEST, окно с нулевым паддингом)
//!   [12..12+mlen)      manifest JSON (UTF-8), патчится при finalize
//!   [12+mlen..)        данные тензоров: F16 LE, выравнивание буферов 64B
//!
//! Двухпроходная схема: заголовок резервирует фиксированное окно манифеста,
//! finalize перезаписывает окно полным списком entries. Стриминг без знания
//! полного списка заранее.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"YTF1";
pub const VERSION: u32 = 1;
pub const RESERVE_MANIFEST: u32 = 16 * 1024 * 1024;
const ALIGN: u64 = 64;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    /// Смещение от начала data-секции (после заголовка+манифеста)
    pub offset: u64,
    /// Размер в байтах
    pub len: u64,
}

#[derive(serde::Deserialize, Debug, Default, Clone)]
pub struct ManifestPre {
    pub gguf_sha256: String,
    pub mask: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct FinalManifest {
    pub gguf_sha256: String,
    pub mask: String,
    #[serde(default)]
    pub source_dtype_counts: BTreeMap<String, u32>,
    #[serde(default)]
    pub clamped_bf16: u64,
    pub tensors: Vec<TensorEntry>,
}

/// Стриминговый писатель контейнера.
pub struct ContainerWriter<W: Write> {
    file: W,
    pre: ManifestPre,
    dtype_counts: BTreeMap<String, u32>,
    clamped_bf16: u64,
    entries: Vec<TensorEntry>,
    buffer: Vec<u8>,
}

impl<W: Write + Seek> ContainerWriter<W> {
    pub fn create(mut file: W, pre: ManifestPre) -> std::io::Result<Self> {
        let mut mb = serde_json::to_vec_pretty(&FinalManifest {
            gguf_sha256: pre.gguf_sha256.clone(),
            mask: pre.mask.clone(),
            source_dtype_counts: BTreeMap::new(),
            clamped_bf16: 0,
            tensors: Vec::new(),
        })
        .expect("serialize");
        assert!(
            mb.len() <= RESERVE_MANIFEST as usize,
            "manifest reserve overflow"
        );
        mb.extend(std::iter::repeat(0u8).take(RESERVE_MANIFEST as usize - mb.len()));
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&RESERVE_MANIFEST.to_le_bytes())?;
        file.write_all(&mb)?;
        Ok(Self {
            file,
            pre,
            dtype_counts: BTreeMap::new(),
            clamped_bf16: 0,
            entries: Vec::new(),
            buffer: Vec::with_capacity(128 << 20),
        })
    }

    /// Добавить тензор F16 LE. Возвращает data-offset.
    pub fn add_tensor(&mut self, name: &str, shape: &[usize], f16_le: Vec<u8>) -> u64 {
        let pad = ((ALIGN - (self.buffer.len() as u64 % ALIGN)) % ALIGN) as usize;
        self.buffer.extend(std::iter::repeat(0u8).take(pad));
        let off = self.buffer.len() as u64;
        self.buffer.extend_from_slice(&f16_le);
        self.entries.push(TensorEntry {
            name: name.to_string(),
            shape: shape.to_vec(),
            offset: off,
            len: f16_le.len() as u64,
        });
        off
    }

    pub fn note_dtype(&mut self, dt: &str) {
        *self.dtype_counts.entry(dt.to_string()).or_insert(0) += 1;
    }

    pub fn note_clamped(&mut self, n: u64) {
        self.clamped_bf16 += n;
    }

    pub fn finalize(mut self) -> std::io::Result<()> {
        let final_manifest = FinalManifest {
            gguf_sha256: self.pre.gguf_sha256.clone(),
            mask: self.pre.mask.clone(),
            source_dtype_counts: self.dtype_counts,
            clamped_bf16: self.clamped_bf16,
            tensors: self.entries,
        };
        let mut mb = serde_json::to_vec_pretty(&final_manifest)?;
        assert!(
            mb.len() <= RESERVE_MANIFEST as usize,
            "manifest overflow at finalize"
        );
        mb.extend(std::iter::repeat(0u8).take(RESERVE_MANIFEST as usize - mb.len()));
        self.file.seek(SeekFrom::Start(12))?;
        // mlen уже записан в create() — пишем только JSON-окно
        self.file.write_all(&mb)?;
        self.file.seek(SeekFrom::End(0))?;
        self.file.write_all(&self.buffer)?;
        self.file.flush()?;
        Ok(())
    }
}

/// Читатель .ytf16 поверх mmap.
pub struct Reader {
    mmap: memmap2::Mmap,
    pub manifest: FinalManifest,
    data_start: u64,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| format!("mmap: {e}"))?;
        if mmap.len() < 12 || &mmap[0..4] != MAGIC {
            return Err("not a YTF1 container".into());
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(format!("unsupported ytf16 version {version}"));
        }
        let mlen = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let end = 12 + mlen;
        if mmap.len() < end {
            return Err("truncated manifest".into());
        }
        // Окно манифеста дополнено нулями — режем по первому \0
        let raw = &mmap[12..end];
        let json_end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let manifest: FinalManifest = serde_json::from_slice(&raw[..json_end])
            .map_err(|e| format!("manifest parse: {e}"))?;
        Ok(Self { mmap, manifest, data_start: end as u64 })
    }

    /// Байты тензора по имени (F16 LE) + форма.
    pub fn tensor(&self, name: &str) -> Option<(&[u8], &[usize])> {
        let t = self.manifest.tensors.iter().find(|t| t.name == name)?;
        let start = self.data_start + t.offset;
        let end = start + t.len;
        Some((&self.mmap[start as usize..end as usize], &t.shape))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.manifest.tensors.iter().map(|t| t.name.as_str())
    }
}
