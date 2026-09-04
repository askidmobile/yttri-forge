//! Выгрузка маршрутизируемых экспертов MoE в pinned host-память (FR-001).
//!
//! План 2026-09-04-moe-expert-offload, фаза 2. Эксперты каждого MoE-слоя
//! (`ffn_{gate,up,down}_exps`) читаются из GGUF по смещениям в pinned
//! DEVICEMAP-память (PD-006, по буферу на (слой, матрица)); на устройстве
//! лежит только таблица указателей на экспертов (FR-002). Ядра
//! `indexed_moe_forward*` читают промахи прямо по PCIe (zero-copy, FR-004).
//!
//! Фаза 2 — без кэша: весь префил идёт через стейджинг (PD-011), декод —
//! чистый zero-copy. Кэш горячих экспертов (`SlotPool`, `CacheDirectory`,
//! `ExpertCacheController`) добавляется в фазе 4.
//!
//! Fail-closed (FR-008): неподдерживаемый dtype, нехватка host-памяти,
//! невозможность отобразить буфер — ошибка старта с числами, без молчаливого
//! отката на VRAM.

use candle_core::quantized::GgmlDType;
use candle_core::Result;
use std::sync::Arc;

// ─── Размещение (FR-009) ──────────────────────────────────────────────────────

/// Запрошенное размещение маршрутизируемых экспертов.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPlacement {
    /// Эксперты резидентны в VRAM (как до выгрузки).
    Vram,
    /// Эксперты в pinned host-памяти, device-mapped.
    Ram,
    /// Решение по свободной VRAM при загрузке (FR-021).
    Auto,
}

/// Разобрать значение `MOE_EXPERTS` (default `auto`). Невалидное значение —
/// ошибка старта (fail-closed).
pub fn parse_placement(value: Option<&str>) -> Result<ExpertPlacement> {
    match value {
        None | Some("") => Ok(ExpertPlacement::Auto),
        Some("vram") => Ok(ExpertPlacement::Vram),
        Some("ram") => Ok(ExpertPlacement::Ram),
        Some("auto") => Ok(ExpertPlacement::Auto),
        Some(other) => candle_core::bail!(
            "MOE_EXPERTS={other:?} не разобрать: ожидается vram|ram|auto (default auto)"
        ),
    }
}

/// Разрешённое (фактическое) размещение после решения при загрузке.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedPlacement {
    Vram,
    Ram,
}

static RESOLVED: std::sync::OnceLock<ResolvedPlacement> = std::sync::OnceLock::new();

/// Зафиксировать разрешённое размещение (вызывается загрузчиком один раз).
pub(crate) fn set_resolved_placement(p: ResolvedPlacement) {
    let _ = RESOLVED.set(p);
}

/// Разрешённое размещение для scheduler/adapter (до загрузки — Vram).
pub fn resolved_placement() -> ResolvedPlacement {
    RESOLVED.get().copied().unwrap_or(ResolvedPlacement::Vram)
}

/// Эксперты выгружены в RAM?
pub fn experts_ram() -> bool {
    resolved_placement() == ResolvedPlacement::Ram
}

// ─── Объединение выбранных экспертов чанка (FR-005) ──────────────────────────

/// Отсортированное объединение выбранных экспертов по ids `[n_tokens][k]`
/// (плоский массив). Чистая функция — покрыта host-тестом.
pub fn union_experts(ids: &[u32], n_experts: usize) -> Result<Vec<usize>> {
    let mut seen = vec![false; n_experts];
    let mut out = Vec::new();
    for &id in ids {
        let e = id as usize;
        if e >= n_experts {
            candle_core::bail!("id эксперта {e} вне диапазона 0..{n_experts} — роутер сломан");
        }
        if !seen[e] {
            seen[e] = true;
            out.push(e);
        }
    }
    out.sort_unstable();
    Ok(out)
}

// ─── Pinned host-буфер (PD-006) ───────────────────────────────────────────────

/// Pinned DEVICEMAP-буфер: host-память, отображённая в адресное пространство
/// устройства. Флаги выбраны воротами 0 (2026-09-04): DEVICEMAP без
/// WRITECOMBINED — полоса та же, UVA dev_ptr == host ptr.
pub(crate) struct HostBuf {
    ptr: *mut u8,
    dev_ptr: cudarc::driver::sys::CUdeviceptr,
    bytes: usize,
}

// Буфер живёт в однопоточном загрузчике; device читает по dev_ptr.
unsafe impl Send for HostBuf {}
unsafe impl Sync for HostBuf {}

impl HostBuf {
    pub fn alloc(bytes: usize) -> Result<Self> {
        use cudarc::driver::sys;
        let raw = unsafe {
            cudarc::driver::result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP)
        }
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "pinned-хранилище экспертов: malloc_host({bytes} байт, DEVICEMAP) не удался: {e:?} — \
                     недостаточно свободной host-памяти (fail-closed, FR-008)"
                ))
            })?;
        if raw.is_null() {
            candle_core::bail!("pinned-хранилище экспертов: malloc_host вернул null для {bytes} байт");
        }
        let ptr = raw as *mut u8;
        let mut dev_ptr: cudarc::driver::sys::CUdeviceptr = 0;
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, raw, 0) };
        if rc != sys::CUresult::CUDA_SUCCESS || dev_ptr == 0 {
            candle_core::bail!(
                "pinned-хранилище экспертов: cuMemHostGetDevicePointer_v2 → {rc:?} — \
                 отображение host-памяти для устройства не работает (fail-closed, FR-008)"
            );
        }
        Ok(Self {
            ptr,
            dev_ptr,
            bytes,
        })
    }

    pub fn dev_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Полный буфер как u8-срез (заполнение при загрузке).
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }

    /// Срез одного эксперта (для подъёма в стейджинг/кэш).
    pub fn expert_slice(&self, expert: usize, expert_bytes: usize) -> &[u8] {
        assert!(
            (expert + 1) * expert_bytes <= self.bytes,
            "expert {expert} вне буфера ({} байт)",
            self.bytes
        );
        unsafe { std::slice::from_raw_parts(self.ptr.add(expert * expert_bytes), expert_bytes) }
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        unsafe {
            // FR-012: pinned-память освобождается при выгрузке модели
            // (Drop срабатывает до сброса CUDA-контекста в пути смены модели).
            let _ = cudarc::driver::result::free_host(self.ptr as *mut _);
        }
    }
}

// ─── Матрица экспертов одного слоя ────────────────────────────────────────────

/// Одна матрица (`ffn_{gate,up,down}_exps`) MoE-слоя при размещении в RAM:
/// pinned-данные на хосте + таблица указателей [n_experts] на устройстве.
pub struct ExpertMatrix {
    pub dtype: GgmlDType,
    /// Упаковка [n_experts, n, k]: n строк выхода, k ширина входа.
    pub n_experts: usize,
    pub n: usize,
    pub k: usize,
    /// Байты одного эксперта: n · row_bytes.
    pub expert_bytes: usize,
    host: HostBuf,
    table: cudarc::driver::CudaSlice<u64>,
}

impl ExpertMatrix {
    /// Загрузка из GGUF-среза `data` по диапазону `[start, start+size)`.
    pub fn load(
        dev: &candle_core::CudaDevice,
        dtype: GgmlDType,
        n_experts: usize,
        n: usize,
        k: usize,
        data: &[u8],
        start: usize,
        size: usize,
    ) -> Result<Self> {
        if !matches!(
            dtype,
            GgmlDType::IQ2S
                | GgmlDType::IQ2XS
                | GgmlDType::IQ2XXS
                | GgmlDType::IQ3S
                | GgmlDType::IQ3XXS
                | GgmlDType::IQ4XS
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            // Fail-closed: матрица dtype, который PTX-ядра не поддерживают
            // (матрица select_backend), при размещении в RAM — ошибка старта.
            candle_core::bail!(
                "MOE_EXPERTS=ram: dtype {dtype:?} маршрутизируемых экспертов не поддерживается PTX-ядрами (fail-closed, FR-008)"
            );
        }
        let block_size = dtype.block_size();
        if k % block_size != 0 {
            candle_core::bail!(
                "MOE_EXPERTS=ram: k={k} не кратно block_size {block_size} для {dtype:?}"
            );
        }
        let expert_bytes = n * (k / block_size * dtype.type_size());
        let bytes = n_experts
            .checked_mul(expert_bytes)
            .ok_or_else(|| candle_core::Error::Msg("expert bytes overflow".into()))?;
        if size != bytes {
            candle_core::bail!(
                "размер тензора в файле {size} != расчётному {bytes} (dtype {dtype:?}, [{n_experts},{n},{k}])"
            );
        }
        if start + size > data.len() {
            candle_core::bail!(
                "диапазон экспертов {start}..{} вне среза GGUF (len {})",
                start + size,
                data.len()
            );
        }
        let mut host = HostBuf::alloc(bytes)?;
        host.as_slice_mut().copy_from_slice(&data[start..start + size]);
        let table = alloc_host_table(dev, host.dev_ptr(), n_experts, expert_bytes)?;
        log::info!(
            "[moe] matrix: [{n_experts},{n},{k}] {dtype:?} — {:.1} МиБ pinned",
            bytes as f64 / 1024.0 / 1024.0
        );
        Ok(Self {
            dtype,
            n_experts,
            n,
            k,
            expert_bytes,
            host,
            table,
        })
    }

    pub fn table(&self) -> &cudarc::driver::CudaSlice<u64> {
        &self.table
    }

    /// Таблица по умолчанию: все эксперты читаются из pinned host-памяти
    /// (zero-copy декод, FR-004).
    fn fill_host_table(&mut self, dev: &candle_core::CudaDevice) -> Result<()> {
        let host_vec = host_table_entries(self.host.dev_ptr(), self.n_experts, self.expert_bytes);
        dev.memcpy_htod(&host_vec, &mut self.table)?;
        Ok(())
    }

    /// Таблица на время слоя префила: эксперты union → стейджинг
    /// (ядро префила читает только VRAM, FR-005); остальные записи — host
    /// (не используются ядром на этом слое, но остаются консистентными).
    fn fill_staging_table(
        &mut self,
        dev: &candle_core::CudaDevice,
        staging_base: cudarc::driver::sys::CUdeviceptr,
        union: &[usize],
    ) -> Result<()> {
        let mut host_vec =
            host_table_entries(self.host.dev_ptr(), self.n_experts, self.expert_bytes);
        for &e in union {
            host_vec[e] = staging_base + e as u64 * self.expert_bytes as u64;
        }
        dev.memcpy_htod(&host_vec, &mut self.table)?;
        Ok(())
    }
}

/// Таблица указателей: каждая запись — адрес эксперта в pinned-памяти.
fn alloc_host_table(
    dev: &candle_core::CudaDevice,
    base: cudarc::driver::sys::CUdeviceptr,
    n_experts: usize,
    expert_bytes: usize,
) -> Result<cudarc::driver::CudaSlice<u64>> {
    let mut table = unsafe { dev.alloc::<u64>(n_experts) }?;
    let host_vec = host_table_entries(base, n_experts, expert_bytes);
    dev.memcpy_htod(&host_vec, &mut table)?;
    Ok(table)
}

// ─── Хост-логика таблиц (чистые функции для тестов) ──────────────────────────

/// Хост-модель таблицы: entry[e] = base + e·stride.
pub fn host_table_entries(base: u64, n_experts: usize, expert_bytes: usize) -> Vec<u64> {
    (0..n_experts)
        .map(|e| base + e as u64 * expert_bytes as u64)
        .collect()
}

/// Смешанная таблица: эксперты union читаются из staging, остальные — из host.
pub fn mixed_table_entries(
    host_base: u64,
    staging_base: u64,
    n_experts: usize,
    expert_bytes: usize,
    union: &[usize],
) -> Vec<u64> {
    let mut out = host_table_entries(host_base, n_experts, expert_bytes);
    for &e in union {
        out[e] = staging_base + e as u64 * expert_bytes as u64;
    }
    out
}

// ─── Стейджинг (D-013): один буфер на модель, размером полный слой ────────────

/// VRAM-буфер размером в полный слой экспертов (максимум по слоям): эксперты
/// слоя префила поднимаются сюда на время слоя; эксперты кладутся
/// expert-major (`эксперт e → ptr + e·expert_bytes`), поэтому запись таблицы —
/// просто адрес с тем же stride.
pub struct Staging {
    pub gate: std::sync::Mutex<cudarc::driver::CudaSlice<u8>>,
    pub up: std::sync::Mutex<cudarc::driver::CudaSlice<u8>>,
    pub down: std::sync::Mutex<cudarc::driver::CudaSlice<u8>>,
    /// [gate, up, down] — байты полного слоя (максимум по слоям).
    pub layer_bytes: [usize; 3],
}

impl Staging {
    /// Суммарный размер стейджинга в байтах.
    pub fn total_bytes(&self) -> usize {
        self.gate.lock().expect("staging gate").len()
            + self.up.lock().expect("staging up").len()
            + self.down.lock().expect("staging down").len()
    }

    /// Заблокированный буфер матрицы.
    fn buf(&self, kind: MatrixKind) -> std::sync::MutexGuard<'_, cudarc::driver::CudaSlice<u8>> {
        match kind {
            MatrixKind::Gate => self.gate.lock().expect("staging gate"),
            MatrixKind::Up => self.up.lock().expect("staging up"),
            MatrixKind::Down => self.down.lock().expect("staging down"),
        }
    }
}

// ─── След маршрутизации (FR-004, PD-003) ─────────────────────────────────────

/// Постоянный device-буфер следа: `[n_layers][cap_tokens][k]` u32. Ядра
/// роутера пишут ids текущего шага копированием внутри графа (PD-003); хост
/// читает после шага (фаза 4, before_step).
pub struct TraceBuf {
    ids: std::sync::Mutex<cudarc::driver::CudaSlice<u32>>,
    n_layers: usize,
    /// Токенов на слой: B слотов × (ширина драфта + 1).
    cap_tokens: usize,
    k: usize,
}

impl TraceBuf {
    pub fn alloc(
        dev: &candle_core::CudaDevice,
        n_layers: usize,
        cap_tokens: usize,
        k: usize,
    ) -> Result<Self> {
        let len = n_layers
            .checked_mul(cap_tokens)
            .and_then(|v| v.checked_mul(k))
            .ok_or_else(|| candle_core::Error::Msg("trace size overflow".into()))?;
        let ids = unsafe { dev.alloc::<u32>(len) }?;
        log::info!(
            "[moe] route trace: {n_layers} слоёв × {cap_tokens} ток × k={k} ({:.1} КиБ)",
            len as f64 * 4.0 / 1024.0
        );
        Ok(Self {
            ids: std::sync::Mutex::new(ids),
            n_layers,
            cap_tokens,
            k,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn cap_tokens(&self) -> usize {
        self.cap_tokens
    }

    /// Копия ids шага декода в след слоя (внутри графа — захватывается).
    /// `src` — ids текущего шага `[tokens × k]`, tokens ≤ cap_tokens.
    pub fn copy_in(
        &self,
        dev: &candle_core::CudaDevice,
        layer: usize,
        src: &cudarc::driver::CudaSlice<u32>,
        tokens: usize,
    ) -> Result<()> {
        if layer >= self.n_layers {
            candle_core::bail!("trace: слой {layer} вне 0..{}", self.n_layers);
        }
        if tokens > self.cap_tokens {
            candle_core::bail!(
                "trace: {tokens} ток > cap {cap} — расширьте след (B×(W+1))",
                cap = self.cap_tokens
            );
        }
        let mut ids = self.ids.lock().expect("trace mutex");
        let len = self.cap_tokens * self.k;
        let start = layer * len;
        let mut dst = ids.slice_mut(start..start + tokens * self.k);
        dev.cuda_stream()
            .memcpy_dtod(src, &mut dst)
            .map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}

// ─── Хранилище одного MoE-слоя ────────────────────────────────────────────────

/// Эксперты одного MoE-слоя при размещении в RAM (FR-001): три pinned-матрицы
/// + три таблицы указателей. Живёт в блоке за `Arc<Mutex<…>>` — таблицы
/// меняются на префиле (подготовка слоя → ядра → возврат).
pub struct ExpertLayerStore {
    pub idx: usize,
    pub gate: ExpertMatrix,
    pub up: ExpertMatrix,
    pub down: ExpertMatrix,
}

#[derive(Clone, Copy)]
enum MatrixKind {
    Gate,
    Up,
    Down,
}

impl ExpertLayerStore {
    pub fn shapes(&self) -> ((usize, usize, usize), (usize, usize, usize)) {
        let g = (self.gate.n_experts, self.gate.n, self.gate.k);
        let d = (self.down.n_experts, self.down.n, self.down.k);
        (g, d)
    }

    pub fn pinned_bytes(&self) -> u64 {
        [self.gate.expert_bytes, self.up.expert_bytes, self.down.expert_bytes]
            .iter()
            .map(|&b| b as u64 * self.gate.n_experts as u64)
            .sum()
    }

    /// Подготовка слоя префила (FR-005, PD-011 — весь чанк в стейджинг):
    /// 1) union экспертов чанка; 2) подъём выбранных экспертов из pinned в
    /// стейджинг (expert-major); 3) таблицы слоя → стейджинг.
    pub fn prefill_prepare_layer(
        &mut self,
        dev: &candle_core::CudaDevice,
        staging: &Staging,
        ids_host: &[u32],
    ) -> Result<Vec<usize>> {
        let union = union_experts(ids_host, self.gate.n_experts)?;
        copy_union_to_staging(dev, &self.gate, staging, &union, MatrixKind::Gate)?;
        copy_union_to_staging(dev, &self.up, staging, &union, MatrixKind::Up)?;
        copy_union_to_staging(dev, &self.down, staging, &union, MatrixKind::Down)?;
        // Базы стейджинга для таблиц (значения указателей стабильны, пока
        // живы аллокации стейджинга — guard'ы можно бросать сразу).
        use cudarc::driver::DevicePtr;
        let stream = dev.cuda_stream();
        let staging_ptr = |buf: &std::sync::Mutex<cudarc::driver::CudaSlice<u8>>| {
            let g = buf.lock().expect("staging buf");
            let (ptr, _rec) = g.device_ptr(&stream);
            ptr
        };
        let staging_ptrs: [cudarc::driver::sys::CUdeviceptr; 3] = [
            staging_ptr(&staging.gate),
            staging_ptr(&staging.up),
            staging_ptr(&staging.down),
        ];
        self.gate.fill_staging_table(dev, staging_ptrs[0], &union)?;
        self.up.fill_staging_table(dev, staging_ptrs[1], &union)?;
        self.down.fill_staging_table(dev, staging_ptrs[2], &union)?;
        Ok(union)
    }

    /// Возврат таблиц слоя на pinned host (после экспертов слоя; до первого
    /// шага декода таблицы обязаны указывать на host — иначе декод прочитает
    /// затёртый стейджинг).
    pub fn prefill_release_layer(&mut self, dev: &candle_core::CudaDevice) -> Result<()> {
        self.gate.fill_host_table(dev)?;
        self.up.fill_host_table(dev)?;
        self.down.fill_host_table(dev)?;
        Ok(())
    }
}

fn copy_union_to_staging(
    dev: &candle_core::CudaDevice,
    matrix: &ExpertMatrix,
    staging: &Staging,
    union: &[usize],
    kind: MatrixKind,
) -> Result<()> {
    let mut buf = staging.buf(kind);
    if matrix.expert_bytes * matrix.n_experts > buf.len() {
        candle_core::bail!(
            "стейджинг меньше слоя: нужно {} байт, буфер {}",
            matrix.expert_bytes * matrix.n_experts,
            buf.len()
        );
    }
    for &e in union {
        if e >= matrix.n_experts {
            candle_core::bail!("union: эксперт {e} вне 0..{}", matrix.n_experts);
        }
        let src = matrix.host.expert_slice(e, matrix.expert_bytes);
        let start = e * matrix.expert_bytes;
        let mut dst = buf.slice_mut(start..start + matrix.expert_bytes);
        dev.memcpy_htod(src, &mut dst)?;
    }
    Ok(())
}

// ─── Рантайм модели (одно на модель) ─────────────────────────────────────────

/// Общий рантайм MoE при выгрузке: след маршрутизации + стейджинг + статистика.
/// Живёт в модели за Arc; блоки получают клон Arc.
pub struct MoeRuntime {
    pub n_layers: usize,
    pub k: usize,
    pub trace: TraceBuf,
    /// Стейджинг выделяется после paged-пула (PD-010: пул → стейджинг → кэш).
    staging: std::sync::OnceLock<Staging>,
    /// Суммарный размер pinned-хранилища (для лога/наблюдаемости).
    pub pinned_bytes: u64,
}

impl MoeRuntime {
    pub fn new(
        dev: &candle_core::CudaDevice,
        n_layers: usize,
        k: usize,
        cap_tokens: usize,
        pinned_bytes: u64,
    ) -> Result<Arc<Self>> {
        let trace = TraceBuf::alloc(dev, n_layers, cap_tokens, k)?;
        Ok(Arc::new(Self {
            n_layers,
            k,
            trace,
            staging: std::sync::OnceLock::new(),
            pinned_bytes,
        }))
    }

    /// Выделить стейджинг после paged-пула. Fail-closed при нехватке VRAM.
    pub fn alloc_staging(
        &self,
        dev: &candle_core::CudaDevice,
        layer_bytes: [usize; 3], // [gate, up, down] — максимум по слоям
    ) -> Result<&Staging> {
        if let Some(s) = self.staging.get() {
            return Ok(s);
        }
        let alloc = |bytes: usize, name: &str| -> Result<cudarc::driver::CudaSlice<u8>> {
            unsafe { dev.alloc::<u8>(bytes) }.map_err(|e| {
                candle_core::Error::Msg(format!(
                    "стейджинг {name}: не хватило {bytes} байт VRAM: {e} — \
                     пул KV должен оставить место (fail-closed, FR-006)"
                ))
            })
        };
        let staging = Staging {
            gate: std::sync::Mutex::new(alloc(layer_bytes[0], "gate")?),
            up: std::sync::Mutex::new(alloc(layer_bytes[1], "up")?),
            down: std::sync::Mutex::new(alloc(layer_bytes[2], "down")?),
            layer_bytes,
        };
        log::info!(
            "[moe] staging: {:.0} МиБ (полные слои {}+{}+{} МиБ)",
            staging.total_bytes() as f64 / 1024.0 / 1024.0,
            layer_bytes[0] / 1024 / 1024,
            layer_bytes[1] / 1024 / 1024,
            layer_bytes[2] / 1024 / 1024,
        );
        let _ = self.staging.set(staging);
        Ok(self.staging.get().expect("staging just set"))
    }

    /// Стейджинг готов? (префил fail-closed без него.)
    pub fn staging(&self) -> Option<&Staging> {
        self.staging.get()
    }

    pub fn staging_bytes(&self) -> usize {
        self.staging.get().map(|s| s.total_bytes()).unwrap_or(0)
    }
}

// ─── Решение auto (FR-021) ────────────────────────────────────────────────────

/// Эксперты уходят в RAM, только если ствол+эксперты+пул KV на объявленный
/// CTX+запас не помещаются в свободную VRAM (FR-021: «та же формула пула, что
/// у движка»), а ствол+пул помещаются. Возвращает (ram, reason).
pub fn resolve_auto_needs(
    free_vram: u64,
    need_resident: u64,
    need_ram: u64,
    experts_bytes: u64,
) -> (bool, String) {
    let mib = |b: u64| b as f64 / 1024.0 / 1024.0;
    if free_vram >= need_resident {
        (
            false,
            format!(
                "всё (ствол+эксперты+пул+запас = {:.0} МиБ) помещается в свободных {:.0} МиБ — резидентно",
                mib(need_resident),
                mib(free_vram)
            ),
        )
    } else if free_vram >= need_ram {
        (
            true,
            format!(
                "нужно {:.0} МиБ резидентно (эксперты {:.0}), свободных {:.0} МиБ — при выгрузке нужно {:.0} МиБ, помещается — эксперты в RAM",
                mib(need_resident),
                mib(experts_bytes),
                mib(free_vram),
                mib(need_ram)
            ),
        )
    } else {
        (
            true,
            format!(
                "свободных {:.0} МиБ мало даже при выгрузке ({:.0} МиБ) — тесно, эксперты всё равно в RAM",
                mib(free_vram),
                mib(need_ram)
            ),
        )
    }
}

/// Упрощённое решение (без пула KV) — для тестов.
pub fn resolve_auto(free_vram: u64, trunk_bytes: u64, experts_bytes: u64) -> (bool, String) {
    resolve_auto_needs(
        free_vram,
        trunk_bytes.saturating_add(experts_bytes),
        trunk_bytes,
        experts_bytes,
    )
}

/// Проверка свободной физической памяти под pinned-хранилище (§6): свободной
/// должно быть ≥ эксперты + PREFIX_CACHE_MIB + 2 ГиБ. Windows-only: на Metal/CPU
/// `ram` отвергается раньше.
pub fn check_host_ram_free(experts_bytes: u64, prefix_cache_mib: u64) -> Result<()> {
    let needed = experts_bytes + prefix_cache_mib * 1024 * 1024 + 2 * 1024 * 1024 * 1024;
    let avail = windows_available_phys();
    match avail {
        Some(avail) if avail as u64 >= needed => Ok(()),
        Some(avail) => candle_core::bail!(
            "MOE_EXPERTS=ram: свободной физической памяти {:.0} МиБ < нужных {:.0} МиБ \
             (эксперты {:.0} + prefix cache + 2 ГиБ); pinned не свопится (fail-closed, §6)",
            avail as f64 / 1024.0 / 1024.0,
            needed as f64 / 1024.0 / 1024.0,
            experts_bytes as f64 / 1024.0 / 1024.0
        ),
        None => Ok(()),
    }
}

#[cfg(windows)]
fn windows_available_phys() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    unsafe {
        if GlobalMemoryStatusEx(&mut status) != 0 {
            Some(status.ull_avail_phys)
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
fn windows_available_phys() -> Option<u64> {
    None
}

// ─── Сводка для наблюдаемости (FR-020) ───────────────────────────────────────

/// Снимок состояния выгрузки для /v1/models (сервер маппит в MoeInfo).
#[derive(Debug, Clone, Copy)]
pub struct MoeRuntimeSummary {
    pub experts_ram: bool,
    pub pinned_bytes: u64,
    pub staging_bytes: u64,
}

// ─── Клон-безопасный хэндл хранилища слоя для `Qwen35MoeBlock` ────────────────

/// Таблицы слоя меняются на префиле (подготовка → ядра → возврат), декод
/// читает их по &self — Mutex достаточен (форварды одно-поточны).
pub type SharedLayerStore = Arc<std::sync::Mutex<ExpertLayerStore>>;
