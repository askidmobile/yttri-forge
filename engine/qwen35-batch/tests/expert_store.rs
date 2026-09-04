//! Host-логика выгрузки экспертов без GPU (фаза 2 плана
//! 2026-09-04-moe-expert-offload): разбор MOE_EXPERTS (FR-009), объединение
//! выбранных экспертов чанка (FR-005), математика таблиц указателей (FR-002),
//! решение auto (FR-021). GPU-части (pinned, стейджинг, след) проверяются
//! CUDA-тестами и на стенде.

#![cfg(feature = "cuda")]

use qwen35_batch::real::expert_store::{
    host_table_entries, mixed_table_entries, parse_placement, resolve_auto, union_experts,
    ExpertPlacement,
};

#[test]
fn placement_parsing() {
    assert_eq!(parse_placement(None).unwrap(), ExpertPlacement::Auto);
    assert_eq!(parse_placement(Some("")).unwrap(), ExpertPlacement::Auto);
    assert_eq!(parse_placement(Some("auto")).unwrap(), ExpertPlacement::Auto);
    assert_eq!(parse_placement(Some("vram")).unwrap(), ExpertPlacement::Vram);
    assert_eq!(parse_placement(Some("ram")).unwrap(), ExpertPlacement::Ram);
    // Fail-closed (FR-009): мусор — ошибка старта, а не молчаливый auto.
    assert!(parse_placement(Some("RAM")).is_err());
    assert!(parse_placement(Some("yes")).is_err());
}

#[test]
fn union_of_chunk_ids() {
    // Дубликаты схлопываются, порядок сортированный.
    assert_eq!(union_experts(&[5, 2, 5, 9, 2, 0], 256).unwrap(), vec![0, 2, 5, 9]);
    // Пустой чанк — пустой union.
    assert_eq!(union_experts(&[], 256).unwrap(), Vec::<usize>::new());
    // Вне диапазона — fail-closed ошибка (роутер сломан).
    assert!(union_experts(&[255], 256).is_ok());
    assert!(union_experts(&[256], 256).is_err());
    assert!(union_experts(&[u32::MAX], 256).is_err());
}

#[test]
fn table_entries_math() {
    // host: entry[e] = base + e*expert_bytes (та же формула, что в ядре было).
    let entries = host_table_entries(0x1000, 4, 2112);
    assert_eq!(entries, vec![0x1000, 0x1000 + 2112, 0x1000 + 2 * 2112, 0x1000 + 3 * 2112]);

    // mixed: union читается из staging, остальные — из host.
    let mixed = mixed_table_entries(0x1000, 0x9000, 4, 2112, &[1, 3]);
    assert_eq!(mixed[0], 0x1000);
    assert_eq!(mixed[1], 0x9000 + 2112);
    assert_eq!(mixed[2], 0x1000 + 2 * 2112);
    assert_eq!(mixed[3], 0x9000 + 3 * 2112);
}

#[test]
fn auto_resolution() {
    // trunk+exps влезает — резидентно.
    let (ram, reason) = resolve_auto(100_000, 60_000, 30_000);
    assert!(!ram, "{reason}");
    // exps не влезают, trunk влезает — выгрузка.
    let (ram, reason) = resolve_auto(70_000, 60_000, 30_000);
    assert!(ram, "{reason}");
    // не влезает даже trunk — всё равно выгрузка (fail-closed дальше).
    let (ram, _) = resolve_auto(10_000, 60_000, 30_000);
    assert!(ram);
}
