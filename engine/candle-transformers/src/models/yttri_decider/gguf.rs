//! Читалка контракта GGUF decider (`docs/integration-yttri.md` §10 в yttri-decider): архитектура `yttri-decider`,
//! раскладка 1, KV строго u32 / f32 / строки, 117 тензоров на боевом конфиге (Q8_0 и F32), порядок и ширины голов
//! в самом файле. Файл отображается в память: матрицы энкодера деквантуются в f32 при загрузке по одной, таблица
//! эмбеддингов (97 из 117 МБ) остаётся в `Mmap` и читается построчно.
//!
//! Строгость — как у референсной читалки `export_gguf.read_gguf`: набор, формы и типы тензоров совпадают с
//! раскладкой точно (лишний или недостающий тензор — отказ), согласованность `pad_id`, `max_tokens` и таблицы
//! позиций проверяется здесь, а не в рантайме. Прочие KV игнорируются.

use super::kern;
use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::GgmlDType;
use candle::{bail, Result};
use memmap2::Mmap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

const ARCH: &str = "yttri-decider";
const PREFIX: &str = "yttri-decider.";
const LAYOUT: u32 = 1;
/// Элементов в блоке Q8_0: ширина строки каждой квантованной матрицы и таблицы обязана на него делиться.
const BLOCK: usize = 32;
/// Байт в блоке Q8_0: f16 масштаб + 32 × int8.
const BLOCK_BYTES: usize = 34;

/// Архитектура, версии и головы из KV файла (имена полей — по ключам `yttri-decider.*`).
#[derive(Debug, Clone, PartialEq)]
pub struct DeciderConfig {
    /// `model_version` пакета (`0.2.0`, `0.2.0-smoke`): клиент сверяет с manifest.
    pub model_version: String,
    pub registry_rev: u32,
    pub state_version: u32,
    /// Рецепт происхождения весов, сейчас только `q8_0`.
    pub quant: String,
    /// Слоёв энкодера (`block_count`).
    pub layers: usize,
    /// Ширина скрытого состояния (`embedding_length`).
    pub hidden: usize,
    /// Ширина FFN (`feed_forward_length`).
    pub intermediate: usize,
    /// Голов внимания (`attention.head_count`); `head_dim = hidden / attention_heads`.
    pub attention_heads: usize,
    /// eps LayerNorm (`attention.layer_norm_epsilon`): в tiny-фикстуре 1e-2, в боевой модели 1e-5.
    pub eps: f32,
    /// Строк таблицы позиций (`context_length`), не `max_tokens`.
    pub positions: usize,
    /// Строк таблицы эмбеддингов (`vocab_size`).
    pub vocab: usize,
    /// `pad_token_id`: задаёт и позиции (HF), и смещение таблицы позиций.
    pub pad_id: u32,
    /// Усечение входа делает клиент; рантайм принимает входы до `positions - pad_id - 1` включительно.
    pub max_tokens: usize,
    /// Головы в порядке файла: идентификатор вопроса и число выходов.
    pub heads: Vec<(String, usize)>,
}

impl DeciderConfig {
    /// Ширина одной головы внимания.
    pub fn head_dim(&self) -> usize {
        self.hidden / self.attention_heads
    }

    /// Всего выходов по всем головам.
    pub fn total_outputs(&self) -> usize {
        self.heads.iter().map(|(_, n)| n).sum()
    }
}

fn get<'a>(c: &'a Content, key: &str) -> Result<&'a Value> {
    match c.metadata.get(key) {
        Some(v) => Ok(v),
        None => bail!("gguf: нет ключа {key}"),
    }
}

fn get_u32(c: &Content, key: &str) -> Result<u32> {
    get(c, key)?
        .to_u32()
        .map_err(|e| e.context(format!("gguf: ключ {key}")))
}

fn get_f32(c: &Content, key: &str) -> Result<f32> {
    get(c, key)?
        .to_f32()
        .map_err(|e| e.context(format!("gguf: ключ {key}")))
}

fn get_str<'a>(c: &'a Content, key: &str) -> Result<&'a str> {
    get(c, key)?
        .to_string()
        .map(|s| s.as_str())
        .map_err(|e| e.context(format!("gguf: ключ {key}")))
}

fn own(name: &str) -> String {
    format!("{PREFIX}{name}")
}

fn parse_config(c: &Content) -> Result<DeciderConfig> {
    let arch = get_str(c, "general.architecture")?;
    if arch != ARCH {
        bail!("gguf: general.architecture = {arch:?}, ожидается {ARCH:?}");
    }
    let layout = get_u32(c, &own("layout"))?;
    if layout != LAYOUT {
        bail!("gguf: раскладка {layout}, читалка знает {LAYOUT}");
    }
    let quant = get_str(c, &own("quant"))?.to_string();
    if quant != "q8_0" {
        bail!("gguf: рецепт {quant:?}, поддержан только \"q8_0\"");
    }
    for (key, want) in [("pooling", "cls"), ("hidden_act", "gelu_erf")] {
        let got = get_str(c, &own(key))?;
        if got != want {
            bail!("gguf: {key} = {got:?}, контракт — {want:?}");
        }
    }
    let usize_of = |key: &str| -> Result<usize> { Ok(get_u32(c, &own(key))? as usize) };
    let mut heads = Vec::new();
    for qid in get_str(c, &own("heads"))?.split(',') {
        let ok = qid.chars().next().is_some_and(|ch| ch.is_ascii_lowercase())
            && qid
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_');
        if !ok || heads.iter().any(|(q, _): &(String, usize)| q == qid) {
            bail!("gguf: heads должен быть списком различных идентификаторов вопросов через запятую, получено {qid:?}");
        }
        let n = usize_of(&format!("head.{qid}"))?;
        if n == 0 {
            bail!("gguf: head.{qid} = 0");
        }
        heads.push((qid.to_string(), n));
    }
    let cfg = DeciderConfig {
        model_version: get_str(c, &own("model_version"))?.to_string(),
        registry_rev: get_u32(c, &own("registry_rev"))?,
        state_version: get_u32(c, &own("state_version"))?,
        quant,
        layers: usize_of("block_count")?,
        hidden: usize_of("embedding_length")?,
        intermediate: usize_of("feed_forward_length")?,
        attention_heads: usize_of("attention.head_count")?,
        eps: get_f32(c, &own("attention.layer_norm_epsilon"))?,
        positions: usize_of("context_length")?,
        vocab: usize_of("vocab_size")?,
        pad_id: get_u32(c, &own("pad_token_id"))?,
        max_tokens: usize_of("max_tokens")?,
        heads,
    };
    if cfg.layers == 0
        || cfg.hidden == 0
        || cfg.intermediate == 0
        || cfg.attention_heads == 0
        || cfg.vocab == 0
    {
        bail!("gguf: нулевой размер архитектуры: {cfg:?}");
    }
    if !cfg.hidden.is_multiple_of(cfg.attention_heads) {
        bail!(
            "gguf: embedding_length {} не делится на attention.head_count {}",
            cfg.hidden,
            cfg.attention_heads
        );
    }
    for (what, n) in [
        ("embedding_length", cfg.hidden),
        ("feed_forward_length", cfg.intermediate),
    ] {
        if !n.is_multiple_of(BLOCK) {
            bail!("gguf: {what} = {n} не кратен {BLOCK}: строка Q8_0 не делится на блоки");
        }
    }
    if !(cfg.eps > 0.0 && cfg.eps < 1.0) {
        bail!("gguf: layer_norm_epsilon = {} вне (0, 1)", cfg.eps);
    }
    if cfg.pad_id as usize >= cfg.vocab {
        bail!(
            "gguf: pad_token_id {} вне словаря из {} строк",
            cfg.pad_id,
            cfg.vocab
        );
    }
    // последняя позиция входа без <pad> равна pad_id + L, в таблице `positions` строк
    if cfg.max_tokens == 0 || cfg.max_tokens + cfg.pad_id as usize + 1 > cfg.positions {
        bail!(
            "gguf: max_tokens {} не помещается в таблицу из {} позиций при pad_id {}",
            cfg.max_tokens,
            cfg.positions,
            cfg.pad_id
        );
    }
    Ok(cfg)
}

/// Тензор раскладки: имя, форма как в torch, квантован ли (Q8_0) или F32.
type Spec = (String, Vec<usize>, bool);

fn tensor_specs(cfg: &DeciderConfig) -> Vec<Spec> {
    let (h, ff) = (cfg.hidden, cfg.intermediate);
    let mut out: Vec<Spec> = vec![
        (
            "embeddings.word_embeddings.weight".into(),
            vec![cfg.vocab, h],
            true,
        ),
        (
            "embeddings.position_embeddings.weight".into(),
            vec![cfg.positions, h],
            false,
        ),
        (
            "embeddings.token_type_embeddings.weight".into(),
            vec![1, h],
            false,
        ),
        ("embeddings.LayerNorm.weight".into(), vec![h], false),
        ("embeddings.LayerNorm.bias".into(), vec![h], false),
    ];
    for i in 0..cfg.layers {
        let base = format!("encoder.layer.{i}.");
        for (part, rows, cols) in [
            ("attention.self.query", h, h),
            ("attention.self.key", h, h),
            ("attention.self.value", h, h),
            ("attention.output.dense", h, h),
            ("intermediate.dense", ff, h),
            ("output.dense", h, ff),
        ] {
            out.push((format!("{base}{part}.weight"), vec![rows, cols], true));
            out.push((format!("{base}{part}.bias"), vec![rows], false));
        }
        for part in ["attention.output.LayerNorm", "output.LayerNorm"] {
            out.push((format!("{base}{part}.weight"), vec![h], false));
            out.push((format!("{base}{part}.bias"), vec![h], false));
        }
    }
    for (qid, n) in &cfg.heads {
        out.push((format!("heads.{qid}.dense.weight"), vec![h, h], false));
        out.push((format!("heads.{qid}.dense.bias"), vec![h], false));
        out.push((format!("heads.{qid}.out_proj.weight"), vec![*n, h], false));
        out.push((format!("heads.{qid}.out_proj.bias"), vec![*n], false));
    }
    out
}

/// Открытый файл: заголовок, проверенный по контракту, и отображение данных.
pub(super) struct Gguf {
    pub(super) cfg: DeciderConfig,
    content: Content,
    mmap: Mmap,
}

impl Gguf {
    pub(super) fn open(path: &Path) -> Result<Arc<Self>> {
        let name = path.display();
        let f = File::open(path).map_err(|e| candle::Error::msg(format!("gguf: {name}: {e}")))?;
        let content = Content::read(&mut BufReader::new(&f))
            .map_err(|e| e.context(format!("gguf: {name}")))?;
        // SAFETY: файл пакета неизменяем, пока модель загружена (его заменяют только после выгрузки)
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| candle::Error::msg(format!("gguf: mmap {name}: {e}")))?;
        let cfg = parse_config(&content).map_err(|e| e.context(format!("gguf: {name}")))?;
        let specs = tensor_specs(&cfg);
        if content.tensor_infos.len() != specs.len() {
            bail!(
                "gguf: {name}: {} тензоров, раскладка требует {}",
                content.tensor_infos.len(),
                specs.len()
            );
        }
        for (tname, shape, quantized) in &specs {
            let Some(info) = content.tensor_infos.get(tname) else {
                bail!("gguf: {name}: нет тензора {tname}");
            };
            let want = if *quantized {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            if info.ggml_dtype != want || info.shape.dims() != shape.as_slice() {
                bail!(
                    "gguf: {name}: тензор {tname}: {:?} {:?}, раскладка требует {want:?} {shape:?}",
                    info.ggml_dtype,
                    info.shape.dims()
                );
            }
            let (start, size) = content.tensor_byte_range(tname)?;
            if start.checked_add(size).is_none_or(|end| end > mmap.len()) {
                bail!("gguf: {name}: тензор {tname} выходит за пределы файла");
            }
        }
        Ok(Arc::new(Self { cfg, content, mmap }))
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        let (start, size) = self.content.tensor_byte_range(name)?;
        Ok(&self.mmap[start..start + size])
    }

    /// F32-тензор целиком (позиции, LayerNorm, bias, головы).
    pub(super) fn f32_vec(&self, name: &str) -> Result<Vec<f32>> {
        if self.content.tensor_dtype(name)? != GgmlDType::F32 {
            bail!("gguf: тензор {name} не F32");
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Матрица Q8_0 → f32 `[rows, cols]` напрямую из отображения.
    pub(super) fn q8_to_f32(&self, name: &str) -> Result<Vec<f32>> {
        if self.content.tensor_dtype(name)? != GgmlDType::Q8_0 {
            bail!("gguf: тензор {name} не Q8_0");
        }
        let shape = self.content.tensor_shape(name)?;
        let (rows, cols) = (shape[0], shape[1]);
        let row_bytes = cols / BLOCK * BLOCK_BYTES;
        let bytes = self.bytes(name)?;
        let mut out = vec![0f32; rows * cols];
        for (r, o) in out.chunks_exact_mut(cols).enumerate() {
            kern::dequant_q8_0_row(&bytes[r * row_bytes..(r + 1) * row_bytes], o);
        }
        Ok(out)
    }

    pub(super) fn table(self: &Arc<Self>) -> Result<Table> {
        let (start, size) = self
            .content
            .tensor_byte_range("embeddings.word_embeddings.weight")?;
        let row_bytes = self.cfg.hidden / BLOCK * BLOCK_BYTES;
        if size != self.cfg.vocab * row_bytes {
            bail!(
                "gguf: таблица эмбеддингов: {size} байт, ожидается {}",
                self.cfg.vocab * row_bytes
            );
        }
        Ok(Table {
            g: self.clone(),
            off: start,
            rows: self.cfg.vocab,
            row_bytes,
        })
    }
}

/// Таблица эмбеддингов Q8_0 в отображении файла: строка = `hidden/32` блоков по 34 байта, деквантуется по запросу.
pub(super) struct Table {
    g: Arc<Gguf>,
    off: usize,
    rows: usize,
    row_bytes: usize,
}

impl Table {
    /// Строка таблицы в f32. `id >= rows` — ошибка: чтение мимо таблицы дало бы чужие байты файла.
    #[inline]
    pub(super) fn row_into(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let id = id as usize;
        if id >= self.rows {
            bail!("id {id} вне словаря из {} строк", self.rows);
        }
        let s = self.off + id * self.row_bytes;
        kern::dequant_q8_0_row(&self.g.mmap[s..s + self.row_bytes], out);
        Ok(())
    }
}
