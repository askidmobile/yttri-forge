use forge_convert::container::{ContainerWriter, ManifestPre, Reader};
use std::fs::File;
use std::io::Write;

#[test]
fn ytf16_roundtrip() {
    let path = std::env::temp_dir().join("ytf16_test.ytf16");
    {
        let f = File::create(&path).unwrap();
        let pre = ManifestPre {
            gguf_sha256: "deadbeef".repeat(4),
            mask: "heavy".into(),
        };
        let mut w = ContainerWriter::create(f, pre).unwrap();
        // Тензор 1: 3 элемента (6 байт) — проверка выравнивания следующего
        let off0 = w.add_tensor("blk.0.attn_q.weight", &[1, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(off0, 0);
        // Тензор 2: большой, должен быть выровнен на 64
        let big = vec![7u8; 1000];
        let off1 = w.add_tensor("blk.0.attn_out.weight", &[500], big).unwrap();
        assert_eq!(off1 % 64, 0, "offset must be 64-aligned");
        w.finalize().unwrap();
    }
    let r = Reader::open(&path).unwrap();
    assert_eq!(r.manifest.mask, "heavy");
    assert_eq!(r.manifest.gguf_sha256.len(), 32);
    let names: Vec<&str> = r.tensor_names().collect();
    assert_eq!(names, vec!["blk.0.attn_q.weight", "blk.0.attn_out.weight"]);
    let (data, shape) = r.tensor("blk.0.attn_q.weight").unwrap();
    assert_eq!(shape, &[1, 3]);
    assert_eq!(data, &[1, 2, 3, 4, 5, 6]);
    let (data2, _) = r.tensor("blk.0.attn_out.weight").unwrap();
    assert_eq!(data2[0], 7);
    assert_eq!(data2.len(), 1000);
    assert!(r.tensor("nonexistent").is_none());
    std::fs::remove_file(&path).ok();
}

/// Данные пишутся потоком, а не копятся в памяти: контейнер 27B в RAM не
/// поместится. Проверяем, что файл растёт по мере добавления тензоров, а не
/// только на finalize.
#[test]
fn tensor_data_is_written_streaming_not_buffered() {
    let path = std::env::temp_dir().join("ytf_stream_test.ytf");
    let f = File::create(&path).unwrap();
    let pre = ManifestPre {
        gguf_sha256: String::new(),
        mask: "standalone".into(),
    };
    let mut w = ContainerWriter::create(f, pre).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();
    w.add_typed("big", &[1 << 20], "F16", &vec![9u8; 1 << 20]).unwrap();
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(
        after >= before + (1 << 20),
        "данные должны быть в файле сразу: было {before}, стало {after}"
    );
    w.finalize().unwrap();
    let r = Reader::open(&path).unwrap();
    let (data, _) = r.tensor("big").unwrap();
    assert_eq!(data.len(), 1 << 20);
    assert_eq!(data[0], 9);
    std::fs::remove_file(&path).ok();
}
