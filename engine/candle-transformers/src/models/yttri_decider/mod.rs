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
//!   и запасом на базовом уровне. На i7-9700, L = 256, пул из 4 потоков: p50 ≈ 16 мс, p95 ≈ 17–18 мс (пул из 2 потоков —
//!   p95 ≈ 30 мс); резидентно ≈ 47 МБ плюс страницы таблицы эмбеддингов из отображённого файла (RSS ≈ 67 МБ после
//!   загрузки, ≈ 100 МБ после тысяч прогонов). Это номинал на тихой машине: под фоном p95 растёт — до 25–29 мс при 4 занятых
//!   соседних потоках на 8 ядрах и до 50–53 мс при 6 (10 потоков на 8 ядер, выше D-032): работу `gemm` пул делит статически,
//!   и один вытесненный ОС поток задерживает весь проход. Поэтому размер пула вызывающий выбирает по свободным ядрам.
//!   `gemm` выбирает ядро сам и по одному признаку: есть FMA — ядро x86-64-v3 (AVX2, BMI2…), иначе скалярное (SIMD-ширина 1,
//!   не SSE2). CPU с FMA, но без остального v3 отклоняется на загрузке (иначе паника `gemm` в каждом `forward`). Без FMA
//!   считает верно, но медленно: на трансляции без AVX/FMA (Rosetta 2) L = 256 даёт p95 ≈ 75 мс на пуле из 4 потоков и
//!   ≈ 140 мс на пуле из 2, то есть D-032 там не выполняется; политику для таких CPU решает G8.
//! * `Device::Cuda` — тензорный путь (`tensor.rs`), f32. **Модуль CUDA сам не поднимает**: устройство создаёт и передаёт
//!   вызывающий, когда среда CUDA процесса уже активна; на свежей среде это +365 МБ RSS и до 10 с на пустом JIT-кэше, на
//!   активной — +48 МБ RSS после загрузки и около +70 МБ после сотен прогонов, +40 МБ VRAM (веса 43 МБ), готовность около
//!   20 мс. Подряд (L = 256) проход занимает ≈ 3,3 мс, но видеокарта сбрасывает такты (P8) уже при паузе между вызовами
//!   около 50 мс (p50 ≈ 28 мс), а с 100 мс и дольше тот же проход идёт ≈ 31 мс (p95 ≈ 31–36): медленнее lean-CPU, в норме
//!   p95 ≤ 40 мс. Для разреженного потока писем GPU-путь почти всегда в этом режиме. Свойство железа: документируется,
//!   не лечится.
//! * остальные устройства — ошибка.
//!
//! # Потоки
//!
//! Пул потоков задаёт вызывающий: `forward` распараллеливается в текущем пуле rayon (`ThreadPool::install`).
//!
//! * CPU-путь: вызовы из нескольких потоков идут параллельно, у каждого свои рабочие буферы (из списка свободных, не
//!   больше двух хранится). Замка поверх вычисления нет нарочно: с ним два `pool.install(|| forward)` на одном пуле
//!   вставали бы навсегда — рабочий поток, ждущий вложенные задачи, берёт из очереди чужой вызов, а тот ждёт замок первого.
//! * CUDA-путь: вызовы одной модели идут по очереди (замок на весь `forward`; одно устройство и общий handle cuBLAS),
//!   rayon на этом пути не используется, поэтому ожидание замка пулу не вредит.
//!
//! # Вход
//!
//! B = 1, без паддинга и маски. Ошибка (а не чтение мимо таблиц): пустой вход, `id >= vocab_size`,
//! `pad_id + L >= context_length`. Структуру `[<s> … </s>]` и усечение до `max_tokens` обеспечивает клиент.
//! Литерал `<pad>` внутри входа занимает позицию `pad_id` и не сдвигает счёт следующих (HF `create_position_ids_from_input_ids`).

mod gguf;
mod kern;
mod lean;
mod tensor;

use candle::{bail, Device, Result};
use std::path::Path;

pub use gguf::DeciderConfig;

enum Engine {
    Lean(Box<lean::Lean>),
    Tensor(tensor::TensorPath),
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

    /// Как [`load`](Self::load), но собственные ядра CPU-пути (GELU, softmax, LayerNorm, деквантизация) на базовом уровне
    /// инструкций (SSE2 на x86-64) даже там, где есть AVX2: для проверки запасного варианта на машине с AVX2.
    /// `gemm` выбирает набор инструкций сам и этим не управляется.
    #[doc(hidden)]
    pub fn load_with_baseline_kernels(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::load_with(path.as_ref(), device, kern::Isa::Base)
    }

    fn load_with(path: &Path, device: &Device, isa: kern::Isa) -> Result<Self> {
        let g = gguf::Gguf::open(path)?;
        let engine = match device {
            Device::Cpu => {
                kern::check_gemm_cpu()
                    .map_err(|e| candle::Error::msg(format!("yttri_decider: {e}")))?;
                Engine::Lean(Box::new(lean::Lean::load(&g, isa)?))
            }
            Device::Cuda(_) => Engine::Tensor(tensor::TensorPath::load(&g, device)?),
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

    /// Байт, которые модель держит резидентно: веса f32 и свободные рабочие буферы на хосте (CPU) или веса на устройстве
    /// (CUDA). Буферы вызовов, идущих в этот момент, не учитываются. Не включает отображённый файл: таблица эмбеддингов
    /// читается из него страницами по мере обращения.
    pub fn resident_bytes(&self) -> usize {
        match &self.engine {
            Engine::Lean(m) => m.resident_bytes(),
            Engine::Tensor(m) => m.resident_bytes(),
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
            Engine::Tensor(m) => m.forward(ids, &mut flat)?,
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
