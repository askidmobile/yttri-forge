#![cfg(feature = "metal")]
//! Чтение результатов GPU на CPU при приватных буферах. С #3416 пуловый
//! `new_buffer` отдаёт `StorageModePrivate`, у которого `contents()` — NULL:
//! zero-copy-чтение и fused-ядра выборки обязаны идти через blit, а не падать
//! на `assert!` в `read_to_vec` (Yttri, паники выборки 2026-08-29 и 2026-09-04).

use candle_core::{Device, Result, Tensor};

#[test]
fn readback_from_private_buffers() -> Result<()> {
    let device = Device::new_metal(0)?;
    // Выход op'а живёт в пуловом `new_buffer` → приватный буфер.
    let logits = Tensor::arange(0f32, 4096f32, &device)?.affine(1.0, 0.5)?;
    if let candle_core::Storage::Metal(s) = &*logits.storage_and_layout().0 {
        assert!(s.buffer().is_private(), "предусловие: выход op'а должен быть приватным");
    }
    let expected = logits.to_vec1::<f32>()?;
    assert_eq!(logits.to_vec1_zero_copy::<f32>()?, expected);

    // Fused-ядра выборки читают свой выход на CPU тем же `read_to_vec`.
    assert_eq!(logits.argmax_suppressed(&[4095])?, 4094);
    let top = logits.topk_suppressed(&[4095, 4094], 3)?;
    let ids: Vec<u32> = top.iter().map(|(id, _)| *id).collect();
    assert_eq!(top.len(), 3, "{top:?}");
    assert_eq!(ids[0], 4093, "{top:?}");
    assert!(!ids.contains(&4095) && !ids.contains(&4094), "{top:?}");
    Ok(())
}
