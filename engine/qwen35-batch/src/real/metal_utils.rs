//! Metal GPU safety utilities
//!
//! Предотвращает SIGSEGV в Apple Metal GPU driver (AGXMetalG16X::fillBuffer)
//! при использовании candle ML framework на Apple Silicon.
//!
//! Корневая причина: candle Metal backend использует пул command buffers
//! с агрессивным переиспользованием. При большом количестве in-flight операций
//! (Conformer 16 слоёв × ~15 ops = ~240 ops) возникает race condition —
//! GPU ещё использует буфер, а пул уже переиспользует его.
//!
//! Три уровня защиты:
//! 1. `configure_metal_env()` — снижает количество ops на command buffer (50 → 16)
//! 2. `metal_probe()` — тестирует GPU перед загрузкой тяжёлой модели
//! 3. `metal_sync()` — sync barrier для сброса command buffer pool в runtime

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use log::{debug, info};

/// Количество compute dispatches на один Metal command buffer.
/// Дефолт candle = 50. Ранее снижали до 16 из-за SIGSEGV в Conformer,
/// потом подняли обратно до 50.
///
/// Для LLM prefill 15K токенов 50 ops означает что в command queue одновременно
/// in-flight ~3-4 decoder layers (один layer ~15 ops: 4 attention QMatMul,
/// 3 MLP QMatMul, softmax/silu/etc). Каждый QMatMul на Metal выдаёт F32-буфер
/// размера [seq, n_out], для MLP up_proj это [1, 15K, 10240] = 615 МБ.
/// 4 слоя × ~1.9 ГБ buffers = 7-8 ГБ in-flight в Metal heap — основной источник
/// 17 ГБ Activity Monitor peak (при ps RSS 3.1 ГБ).
///
/// Снижение до 15 = один commit на layer, in-flight buffers = ~1 layer = 1.9 ГБ.
/// Sync overhead на M4 Pro минимальный (~50мкс per commit × 22 layers = 1мс/forward).
const SAFE_COMPUTE_PER_BUFFER: &str = "15";

/// Настроить Metal command buffer pool для стабильной работы.
///
/// Устанавливает `CANDLE_METAL_COMPUTE_PER_BUFFER` до создания Metal device.
/// Candle читает эту переменную в `Device::new_metal()`.
/// Не перезаписывает если пользователь установил вручную.
pub fn configure_metal_env() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        if std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
            // SAFETY: вызывается один раз через Once, до создания Metal device
            unsafe {
                std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", SAFE_COMPUTE_PER_BUFFER);
            }
            debug!(
                "Metal: CANDLE_METAL_COMPUTE_PER_BUFFER={}",
                SAFE_COMPUTE_PER_BUFFER
            );
        }
        // Inference pool: 1 entry вместо дефолтных 5.
        // Каждый flush_and_wait итерирует ВСЕ pool entries (semaphore + mutex на каждый).
        // При serial inference с 24 DeltaNet layers × 1 sync = 24 flush_and_wait/token,
        // 4 лишних entries дают ~0.5мс × 24 = ~12мс overhead.
        // Pool_size=1 убирает это — ноль итераций по пустым entries.
        // Для inference (single-thread, serial ops) pipelining пула не нужен.
        if std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE").is_err() {
            unsafe {
                std::env::set_var("CANDLE_METAL_COMMAND_POOL_SIZE", "1");
            }
            debug!("Metal: CANDLE_METAL_COMMAND_POOL_SIZE=1 (inference optimized)");
        }

        // ПРИМЕЧАНИЕ: CANDLE_DEQUANTIZE_ALL_F16 НЕ активируется автоматически.
        // Полный F16 baseline ломает Qwen3.5 (echo вместо translation):
        // llama.cpp принципиально использует F32 для RMSNorm/softmax/GDN internal
        // compute, F16 только в transit между блоками. Selective F16 подход —
        // F32 baseline + F16 ТОЛЬКО внутри больших batch buffers DeltaNet GDN
        // и chunked attention — реализован через cast на границе dispatch
        // функций в quantized_qwen35.rs. Native GPU dequantize_f16 в candle-fork
        // остаётся как готовая капабилити (PR-ready) для будущих model integrations.
    });
}

/// Пробное вычисление для проверки работоспособности Metal GPU.
///
/// Тестирует критические пути которые вызывают SIGSEGV:
/// 1. matmul (compute pipeline)
/// 2. Tensor::zeros (blit fill_buffer — проблемный путь)
/// 3. Большой matmul 768×768 (типичный для Conformer)
/// 4. readback на CPU (memory mapping)
///
/// Вызывается перед загрузкой тяжёлой модели (~843MB).
/// При ошибке — рекомендуется fallback на CPU.
pub fn metal_probe(device: &Device) -> Result<()> {
    if !device.is_metal() {
        return Ok(());
    }
    debug!("Metal: running probe computation to verify GPU...");

    // 1. Тест compute: matmul 8×8 + readback
    let a = Tensor::from_vec(vec![1.0f32; 64], (8, 8), device)
        .map_err(|e| anyhow::anyhow!("metal probe: create tensor A: {}", e))?;
    let b = Tensor::from_vec(vec![1.0f32; 64], (8, 8), device)
        .map_err(|e| anyhow::anyhow!("metal probe: create tensor B: {}", e))?;
    let c = a
        .matmul(&b)
        .map_err(|e| anyhow::anyhow!("metal probe: matmul 8x8: {}", e))?;
    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("metal probe: sync after matmul: {}", e))?;
    let _data: Vec<Vec<f32>> = c
        .to_vec2()
        .map_err(|e| anyhow::anyhow!("metal probe: readback matmul: {}", e))?;

    // 2. Тест allocate_zeros: Tensor::zeros (blit fill_buffer — проблемный путь)
    let z = Tensor::zeros((8, 8), DType::F32, device)
        .map_err(|e| anyhow::anyhow!("metal probe: zeros: {}", e))?;
    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("metal probe: sync after zeros: {}", e))?;
    let _zdata: Vec<Vec<f32>> = z
        .to_vec2()
        .map_err(|e| anyhow::anyhow!("metal probe: readback zeros: {}", e))?;

    // 3. Тест крупных операций: matmul 768×768 (типичный размер для Conformer)
    let x = Tensor::from_vec(vec![0.5f32; 768], (1, 768), device)
        .map_err(|e| anyhow::anyhow!("metal probe: create tensor X: {}", e))?;
    let w = Tensor::from_vec(vec![0.1f32; 768 * 768], (768, 768), device)
        .map_err(|e| anyhow::anyhow!("metal probe: create tensor W: {}", e))?;
    let y = x
        .matmul(
            &w.t()
                .map_err(|e| anyhow::anyhow!("metal probe: transpose W: {}", e))?,
        )
        .map_err(|e| anyhow::anyhow!("metal probe: matmul 768x768: {}", e))?;
    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("metal probe: sync after large matmul: {}", e))?;
    let _ydata: Vec<Vec<f32>> = y
        .to_vec2()
        .map_err(|e| anyhow::anyhow!("metal probe: readback large matmul: {}", e))?;

    info!("Metal: probe computation successful — GPU is operational");
    Ok(())
}

/// Sync barrier для Metal GPU.
///
/// Вызывает `device.synchronize()` — commit текущего command buffer
/// и ожидание завершения всех GPU операций. На CPU — no-op.
///
/// Используется в Conformer inference для предотвращения накопления
/// слишком большого количества in-flight буферов.
/// Overhead: ~50-200 мкс на вызов.
#[inline]
pub fn metal_sync(device: &Device) -> Result<()> {
    if device.is_metal() {
        device
            .synchronize()
            .map_err(|e| anyhow::anyhow!("Metal sync failed: {}", e))?;
    }
    Ok(())
}

/// Очистить Metal GPU ресурсы после завершения серии инференсов.
///
/// Вызывает `synchronize` + `flush_buffers` + `purge_buffers` на Metal device
/// и логирует статистику buffer pool / command pool / kernel cache до и после.
///
/// **ВАЖНО: только для teardown (выгрузка engine, завершение записи, eviction).**
/// НЕ вызывать в per-chunk hot path! `purge_buffers()` очищает kernel cache,
/// что вызывает перекомпиляцию ~15 Metal shaders при следующем inference
/// (~100мс overhead + ~5 MB Metal driver state residual на каждый цикл).
/// Для hot path использовать `device.synchronize()` + `device.flush_buffers()`.
///
/// На CPU — no-op. Безопасен для повторных вызовов (идемпотентен).
pub fn metal_engine_cleanup(device: &Device, engine_name: &str) {
    #[cfg(target_os = "macos")]
    {
        if let Device::Metal(dev) = device {
            let before_pool = dev.buffer_pool_stats().ok();
            let before_cmd = dev.command_pool_stats().ok();
            let before_kernels = dev.kernel_cache_stats().ok();
            let (before_alloc, before_ws_limit) = dev.metal_memory_stats();

            let _ = device.synchronize();
            let _ = device.flush_buffers();
            let _ = device.purge_buffers();

            let after_pool = dev.buffer_pool_stats().ok();
            let after_cmd = dev.command_pool_stats().ok();
            let after_kernels = dev.kernel_cache_stats().ok();
            let (after_alloc, _) = dev.metal_memory_stats();

            info!(
                "{} cleanup (metal): alloc={}MB->{}MB ws_limit={}MB pool={:?}->{:?} cmd={:?}->{:?} kernels={:?}->{:?}",
                engine_name,
                before_alloc / (1024 * 1024),
                after_alloc / (1024 * 1024),
                before_ws_limit / (1024 * 1024),
                before_pool,
                after_pool,
                before_cmd,
                after_cmd,
                before_kernels,
                after_kernels
            );
            return;
        }
    }
    // Non-Metal (CPU / CUDA): базовая очистка
    let _ = device.synchronize();
    info!("{} cleanup (cpu/cuda): synchronize complete", engine_name);
}

/// Выполнить `f` внутри ObjC autorelease-пула (macOS; иначе прозрачный вызов).
///
/// Metal-драйвер создаёт autoreleased временные объекты на каждый submit.
/// На raw Rust-потоке БЕЗ пула objc4 складывает их в TLS-страницы
/// (autoreleaseNoPage) и освобождает ТОЛЬКО при выходе потока — на
/// долгоживущем воркере это утечка. Инцидент 2026-07-06: live-STT рос
/// ~200 МБ/мин (+33 МБ на транскрипцию, репро в nemotron_engine tests),
/// −5 ГБ возвращались лишь при остановке записи (выход потока воркера).
///
/// Оборачивай КАЖДЫЙ ML-инференс (candle/Metal) на долгоживущем потоке.
/// Паника внутри `f` безопасна: пул дренируется в Drop при unwind.
#[inline]
pub fn with_autoreleasepool<T>(f: impl FnOnce() -> T) -> T {
    #[cfg(target_os = "macos")]
    {
        objc2::rc::autoreleasepool(|_| f())
    }
    #[cfg(not(target_os = "macos"))]
    {
        f()
    }
}

/// Периодическая очистка Metal buffer pool В ХОДЕ серии инференсов (hot path).
///
/// В отличие от [`metal_engine_cleanup`] (teardown) вызывается периодически во
/// время длинной записи. Чтобы не платить рекомпиляцией ~15 шейдеров на КАЖДОМ
/// вызове, полный `purge_buffers` (чистит pool + kernel cache) запускается ТОЛЬКО
/// когда пул перерос `purge_threshold_bytes`; иначе — дешёвый `flush_buffers`.
///
/// Почему недостаточно flush: `flush_buffers` удаляет лишь буферы со
/// `strong_count == 1`, а в completion-aware пуле candle буферы держатся со
/// `strong_count > 1`, поэтому пул растёт неограниченно (инцидент 2026-06-27:
/// 655 буферов / 2.55 ГБ за 12 мин, при том что flush звался на каждый чанк).
/// `purge_buffers` (`wait_until_completed` + `buffers.clear`) — единственный, кто
/// реально освобождает пул.
///
/// Возвращает `true`, если был выполнен полный purge. На CPU — no-op → `false`.
pub fn metal_periodic_purge(
    device: &Device,
    purge_threshold_bytes: usize,
    engine_name: &str,
) -> bool {
    #[cfg(target_os = "macos")]
    {
        if let Device::Metal(dev) = device {
            let _ = device.synchronize();
            let pool_bytes = dev.buffer_pool_stats().map(|(_, b, _, _)| b).unwrap_or(0);
            if pool_bytes > purge_threshold_bytes {
                let _ = device.purge_buffers();
                log::debug!(
                    "[{}] periodic Metal purge: pool {} МБ > порог {} МБ — очищен",
                    engine_name,
                    pool_bytes / (1024 * 1024),
                    purge_threshold_bytes / (1024 * 1024)
                );
                return true;
            }
            let _ = device.flush_buffers();
            return false;
        }
    }
    // Non-Metal: дешёвая синхронизация, purge не нужен.
    let _ = device.synchronize();
    let _ = engine_name;
    let _ = purge_threshold_bytes;
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configure_metal_env() {
        // Не должен паниковать
        configure_metal_env();
    }

    #[test]
    fn test_metal_sync_cpu() {
        // На CPU — no-op, не должен паниковать
        let device = Device::Cpu;
        metal_sync(&device).unwrap();
    }

    #[test]
    fn test_metal_probe_cpu() {
        // На CPU — сразу Ok
        let device = Device::Cpu;
        metal_probe(&device).unwrap();
    }
}
