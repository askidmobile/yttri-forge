//! Контейнер .ytf16.
//!
//! Layout:
//!   [0..4)   magic "YTF1" (LE)
//!   [4..8)   version u32: 1 — сайдкар, 2 — самостоятельная модель
//!   [8..12)  manifest_len u32 (= RESERVE_MANIFEST, окно с нулевым паддингом)
//!   [12..12+mlen)      manifest JSON (UTF-8), патчится при finalize
//!   [12+mlen..)        данные тензоров, выравнивание буферов 64B
//!
//! Двухпроходная схема: заголовок резервирует фиксированное окно манифеста,
//! finalize перезаписывает окно полным списком entries. Стриминг без знания
//! полного списка заранее.
//!
//! v1 — F16-дамп избранных проекций поверх GGUF (сайдкар, dual-read).
//! v2 — самостоятельная модель: у каждого тензора свой GGML-тип, данные лежат
//! готовыми блоками, поэтому загрузка это mmap → VRAM без переквантования на
//! CPU. Гиперпараметры и токенизатор едут внутри, GGUF рядом не нужен.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"YTF1";
pub const VERSION: u32 = 1;
/// Самостоятельная модель: типизированные тензоры + конфиг + токенизатор.
pub const VERSION_STANDALONE: u32 = 2;
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
    /// GGML-тип данных: "F16" (умолчание, v1), "F32", "Q4_K", "Q6_K", "Q8_0"…
    #[serde(default = "dtype_f16")]
    pub dtype: String,
}

fn dtype_f16() -> String {
    "F16".to_string()
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
    /// v2: гиперпараметры модели в терминах ключей GGUF-метаданных
    /// (`qwen35.block_count` и т.д.). Пусто у сайдкара.
    #[serde(default)]
    pub config: BTreeMap<String, serde_json::Value>,
    pub tensors: Vec<TensorEntry>,
}

/// Стриминговый писатель контейнера.
pub struct ContainerWriter<W: Write> {
    file: W,
    pre: ManifestPre,
    version: u32,
    config: BTreeMap<String, serde_json::Value>,
    dtype_counts: BTreeMap<String, u32>,
    clamped_bf16: u64,
    entries: Vec<TensorEntry>,
    /// Сколько байтов данных уже записано (offset следующего тензора).
    /// Данные пишутся сразу в файл: копить контейнер в памяти нельзя —
    /// на 27B это десятки гигабайт.
    data_len: u64,
}

impl<W: Write + Seek> ContainerWriter<W> {
    pub fn create(file: W, pre: ManifestPre) -> std::io::Result<Self> {
        Self::create_versioned(file, pre, VERSION)
    }

    /// Писатель самостоятельной модели (v2): типизированные тензоры + конфиг.
    pub fn create_standalone(
        file: W,
        pre: ManifestPre,
        config: BTreeMap<String, serde_json::Value>,
    ) -> std::io::Result<Self> {
        let mut w = Self::create_versioned(file, pre, VERSION_STANDALONE)?;
        w.config = config;
        Ok(w)
    }

    fn create_versioned(mut file: W, pre: ManifestPre, version: u32) -> std::io::Result<Self> {
        let mut mb = serde_json::to_vec_pretty(&FinalManifest {
            gguf_sha256: pre.gguf_sha256.clone(),
            mask: pre.mask.clone(),
            source_dtype_counts: BTreeMap::new(),
            clamped_bf16: 0,
            config: BTreeMap::new(),
            tensors: Vec::new(),
        })
        .expect("serialize");
        assert!(
            mb.len() <= RESERVE_MANIFEST as usize,
            "manifest reserve overflow"
        );
        mb.extend(std::iter::repeat(0u8).take(RESERVE_MANIFEST as usize - mb.len()));
        file.write_all(MAGIC)?;
        file.write_all(&version.to_le_bytes())?;
        file.write_all(&RESERVE_MANIFEST.to_le_bytes())?;
        file.write_all(&mb)?;
        Ok(Self {
            file,
            pre,
            version,
            config: BTreeMap::new(),
            dtype_counts: BTreeMap::new(),
            clamped_bf16: 0,
            entries: Vec::new(),
            data_len: 0,
        })
    }

    /// Добавить тензор F16 LE. Возвращает data-offset.
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        f16_le: Vec<u8>,
    ) -> std::io::Result<u64> {
        self.add_typed(name, shape, "F16", &f16_le)
    }

    /// Добавить тензор произвольного GGML-типа (v2). Возвращает data-offset.
    ///
    /// Пишет сразу в файл: заголовок и окно манифеста уже зарезервированы, а
    /// сам манифест патчится при finalize. Промежуточный буфер в памяти
    /// означал бы весь контейнер в RAM — для 27B это неподъёмно.
    pub fn add_typed(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: &str,
        bytes: &[u8],
    ) -> std::io::Result<u64> {
        let pad = ((ALIGN - (self.data_len % ALIGN)) % ALIGN) as usize;
        if pad > 0 {
            self.file.write_all(&[0u8; ALIGN as usize][..pad])?;
            self.data_len += pad as u64;
        }
        let off = self.data_len;
        self.file.write_all(bytes)?;
        self.data_len += bytes.len() as u64;
        self.entries.push(TensorEntry {
            name: name.to_string(),
            shape: shape.to_vec(),
            offset: off,
            len: bytes.len() as u64,
            dtype: dtype.to_string(),
        });
        Ok(off)
    }

    /// Записан ли уже тензор с таким именем.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
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
            config: self.config,
            tensors: self.entries,
        };
        let mut mb = serde_json::to_vec_pretty(&final_manifest)?;
        assert!(
            mb.len() <= RESERVE_MANIFEST as usize,
            "manifest overflow at finalize"
        );
        mb.extend(std::iter::repeat(0u8).take(RESERVE_MANIFEST as usize - mb.len()));
        self.file.seek(SeekFrom::Start(12))?;
        // mlen уже записан в create() — пишем только JSON-окно.
        // Данные тензоров уже в файле (add_typed пишет их сразу).
        self.file.write_all(&mb)?;
        self.file.flush()?;
        Ok(())
    }
}

/// Читатель .ytf16 поверх mmap.
pub struct Reader {
    mmap: memmap2::Mmap,
    pub manifest: FinalManifest,
    pub version: u32,
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
        if version != VERSION && version != VERSION_STANDALONE {
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
        Ok(Self { mmap, manifest, version, data_start: end as u64 })
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
