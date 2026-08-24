//! Совместимость .ytf16: независимый writer (тестовый билд заголовка вручную)
//! → Reader из real::ytf16. Гарантирует, что формат форка совпадает с
//! forge-convert (yttri-forge).

use candle_core::Result;
use std::io::Write;

fn build_container(path: &std::path::Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"YTF1")?;
    f.write_all(&1u32.to_le_bytes())?;
    let reserve = 4096u32;
    // Манифест
    let manifest = r#"{
  "gguf_sha256": "abc123",
  "mask": "heavy",
  "tensors": [
    {"name": "blk.0.attn_q.weight", "shape": [1, 4], "offset": 0, "len": 8},
    {"name": "blk.0.attn_qkv.weight", "shape": [2, 512], "offset": 64, "len": 2048}
  ]
}"#;
    let mlen = reserve as usize;
    f.write_all(&(reserve).to_le_bytes())?;
    let mut mb = manifest.as_bytes().to_vec();
    mb.resize(mlen, 0);
    f.write_all(&mb)?;
    // Данные: тензор1 на data_start+0 (8 байт), тензор2 на data_start+64 (2048)
    let pad = vec![0u8; 64 - 8];
    f.write_all(&[9u8, 8, 7, 6, 5, 4, 3, 2])?;
    f.write_all(&pad)?;
    f.write_all(&vec![1u8; 2048])?;
    f.flush()?;
    Ok(())
}

#[test]
fn ytf16_reader_matches_forge_format() {
    let dir = std::env::temp_dir().join("ytf16_fork_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.ytf16");
    build_container(&path).unwrap();

    use qwen35_batch::real::ytf16::Ytf16Sidecar;
    let r = Ytf16Sidecar::open(&path).expect("open");
    assert_eq!(r.gguf_sha256(), "abc123");
    assert_eq!(r.tensor_names().len(), 2);

    // Тензор 1: данные на data_start+0
    let (d1, s1) = r.tensor_bytes("blk.0.attn_q.weight").expect("t1");
    assert_eq!(s1, &[1, 4]);
    assert_eq!(d1, &[9u8, 8, 7, 6, 5, 4, 3, 2]);

    // Тензор 2: выровнен на 64 внутри data-секции
    let (d2, s2) = r.tensor_bytes("blk.0.attn_qkv.weight").expect("t2");
    assert_eq!(s2, &[2, 512]);
    assert_eq!(d2.len(), 2048);
    assert!(d2.iter().all(|&b| b == 1));

    assert!(r.tensor_bytes("nope").is_none());
    std::fs::remove_file(&path).ok();
}

#[test]
fn ytf16_rejects_bad_magic() {
    let dir = std::env::temp_dir().join("ytf16_fork_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bad.ytf16");
    std::fs::write(&path, b"NOPE12345678").unwrap();
    use qwen35_batch::real::ytf16::Ytf16Sidecar;
    assert!(Ytf16Sidecar::open(&path).is_err());
}
