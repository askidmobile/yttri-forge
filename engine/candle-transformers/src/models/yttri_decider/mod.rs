//! Энкодер decider Yttri (T-750, D-035, BD-009): XLM-R-подобный энкодер (MiniLM-L6/H384 в боевой модели) и головы
//! `out_proj(tanh(dense(h[0])))` по вопросам реестра, веса — GGUF `yttri-decider` (Q8_0 + F32).
//!
//! Контракт файла, позиций и границ входа — `docs/integration-yttri.md` §10 в yttri-decider. Здесь живёт Rust-реализация
//! для Windows и Linux; на macOS ту же модель считает Swift-сайдкар MLX. Модуль исполняется в процессе вызывающего
//! (как e5 и реранкер), сайдкар yforge для него не используется (BD-009).
//!
//! ```no_run
//! use candle_transformers::models::yttri_decider::YttriDecider;
//! # fn main() -> candle::Result<()> {
//! let model = YttriDecider::load("model.gguf", &candle::Device::Cpu)?;
//! let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
//! // ids — полная токенизация state токенайзером e5, усечённая клиентом до max_tokens: [0, ..., 2], без паддинга
//! let logits = pool.install(|| model.forward(&[0, 100, 2]))?; // логиты по головам в порядке model.outputs()
//! # let _ = logits; Ok(()) }
//! ```
//!
//! # Устройства
//!
//! * `Device::Cpu` — lean-путь (`lean.rs`): веса f32, `gemm` напрямую, слитые ядра, AVX2+FMA с проверкой в рантайме
//!   и запасом на базовом уровне. На i7-9700, L = 256, пул из 4 потоков: p95 ≈ 19 мс.
//! * `Device::Cuda` — тензорный путь (`tensor.rs`). **Модуль CUDA сам не поднимает**: устройство создаёт и передаёт
//!   вызывающий, когда среда CUDA процесса уже активна; на свежей среде это +365 МБ RSS и до 10 с на пустом JIT-кэше.
//! * остальные устройства — ошибка.
//!
//! # Потоки
//!
//! Пул потоков задаёт вызывающий: `forward` распараллеливается в текущем пуле rayon (`ThreadPool::install`).
//! Вызовы одной модели сериализуются внутренним замком (рабочие буферы общие).
//!
//! # Вход
//!
//! B = 1, без паддинга и маски. Ошибка (а не чтение мимо таблиц): пустой вход, `id >= vocab_size`,
//! `pad_id + L >= context_length`. Структуру `[<s> … </s>]` и усечение до `max_tokens` обеспечивает клиент.
//! Литерал `<pad>` внутри входа занимает позицию `pad_id` и не сдвигает счёт следующих (HF `create_position_ids_from_input_ids`).

mod gguf;
mod kern;
mod lean;

use candle::{bail, Device, Result};
use std::path::Path;

pub use gguf::DeciderConfig;

enum Engine {
    Lean(Box<lean::Lean>),
}

/// Загруженный decider: веса, конфигурация и вычислитель под выбранное устройство.
pub struct YttriDecider {
    cfg: DeciderConfig,
    outputs: Vec<(String, usize)>,
    engine: Engine,
}

/// Имя из замороженного API плана (PD-050); тот же тип.
pub type DeciderModel = YttriDecider;

impl YttriDecider {
    /// Читает и проверяет `model.gguf` по контракту и готовит модель под устройство.
    /// `Device::Cpu` даёт lean-путь, `Device::Cuda` — тензорный; CUDA этим вызовом не инициализируется.
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::load_with(path.as_ref(), device, kern::best_isa())
    }

    /// Как [`load`](Self::load), но ядра CPU-пути на базовом уровне инструкций (SSE2 на x86-64) даже там, где есть AVX2:
    /// для проверки запасного варианта на машине с AVX2. `gemm` выбирает набор инструкций сам и этим не управляется.
    #[doc(hidden)]
    pub fn load_with_baseline_kernels(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::load_with(path.as_ref(), device, kern::Isa::Base)
    }

    fn load_with(path: &Path, device: &Device, isa: kern::Isa) -> Result<Self> {
        let g = gguf::Gguf::open(path)?;
        let engine = match device {
            Device::Cpu => Engine::Lean(Box::new(lean::Lean::load(&g, isa)?)),
            Device::Cuda(_) => bail!("yttri_decider: путь CUDA появится отдельным шагом (C5)"),
            _ => bail!(
                "yttri_decider: устройство {:?} не поддерживается (Cpu и Cuda)",
                device.location()
            ),
        };
        let cfg = g.cfg.clone();
        let outputs = cfg
            .heads
            .iter()
            .map(|(qid, n)| (format!("logits_{qid}"), *n))
            .collect();
        Ok(Self {
            cfg,
            outputs,
            engine,
        })
    }

    /// Конфигурация из KV файла: версии (для сверки с manifest), `pad_id`, `max_tokens`, головы.
    pub fn config(&self) -> &DeciderConfig {
        &self.cfg
    }

    /// Выходы в порядке голов файла: имя `logits_<qid>` и число значений. `forward` возвращает логиты в том же порядке.
    pub fn outputs(&self) -> &[(String, usize)] {
        &self.outputs
    }

    /// Байт, которые модель держит резидентно: веса f32 и рабочие буферы на хосте (CPU) или веса на устройстве (CUDA).
    /// Не включает отображённый файл: таблица эмбеддингов читается из него страницами по мере обращения.
    pub fn resident_bytes(&self) -> usize {
        match &self.engine {
            Engine::Lean(m) => m.resident_bytes(),
        }
    }

    /// Логиты по головам (порядок — [`outputs`](Self::outputs)) для одного входа `ids` (B = 1).
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.cfg;
        if ids.is_empty() {
            bail!("yttri_decider: пустой вход");
        }
        // последняя позиция входа без <pad> равна pad_id + L, в таблице context_length строк
        if cfg.pad_id as usize + ids.len() >= cfg.positions {
            bail!(
                "yttri_decider: {} id не помещаются в таблицу из {} позиций (pad_id {})",
                ids.len(),
                cfg.positions,
                cfg.pad_id
            );
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= cfg.vocab) {
            bail!("yttri_decider: id {id} вне словаря из {} строк", cfg.vocab);
        }
        let mut flat = vec![0f32; cfg.total_outputs()];
        match &self.engine {
            Engine::Lean(m) => m.forward(ids, &mut flat)?,
        }
        if !flat.iter().all(|v| v.is_finite()) {
            bail!("yttri_decider: логиты не конечны (NaN или Inf)");
        }
        let mut rest = flat.as_slice();
        Ok(cfg
            .heads
            .iter()
            .map(|(_, n)| {
                let (head, tail) = rest.split_at(*n);
                rest = tail;
                head.to_vec()
            })
            .collect())
    }
}
