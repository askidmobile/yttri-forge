//! CPU-путь decider в сборке с `--features cuda` работает без единой библиотеки CUDA в процессе (T-750, Ф3c, C5):
//! среда CUDA не поднимается (cudarc с dynamic-loading открывает nvcuda / cublas только при создании CUDA-устройства),
//! поэтому decider на CPU не платит +365 МБ RSS и до 10 с на пустом JIT-кэше.
//!
//! Отдельный файл — отдельный процесс: в одном процессе с GPU-тестами образы CUDA уже были бы загружены.

use candle::Device;
use candle_transformers::models::yttri_decider::YttriDecider;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/decider-tiny")
}

/// Загруженные образы процесса (Windows: ToolHelp, Linux: /proc/self/maps); на прочих платформах пусто.
fn loaded_images() -> Vec<String> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ModuleEntry {
            size: u32,
            module_id: u32,
            process_id: u32,
            glblcnt: u32,
            proccnt: u32,
            base: *mut u8,
            base_size: u32,
            handle: isize,
            name: [u16; 256],
            path: [u16; 260],
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcessId() -> u32;
            fn CreateToolhelp32Snapshot(flags: u32, pid: u32) -> isize;
            fn Module32FirstW(h: isize, m: *mut ModuleEntry) -> i32;
            fn Module32NextW(h: isize, m: *mut ModuleEntry) -> i32;
            fn CloseHandle(h: isize) -> i32;
        }
        let mut out = Vec::new();
        // SAFETY: структура и сигнатуры kernel32 описаны по документации ToolHelp; снимок закрывается
        unsafe {
            let snap = CreateToolhelp32Snapshot(0x8 | 0x10, GetCurrentProcessId());
            if snap == -1 || snap == 0 {
                return out;
            }
            let mut m: ModuleEntry = std::mem::zeroed();
            m.size = std::mem::size_of::<ModuleEntry>() as u32;
            let mut more = Module32FirstW(snap, &mut m);
            while more != 0 {
                let n = m.name.iter().position(|&c| c == 0).unwrap_or(m.name.len());
                out.push(String::from_utf16_lossy(&m.name[..n]));
                more = Module32NextW(snap, &mut m);
            }
            CloseHandle(snap);
        }
        out
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/maps")
            .map(|s| {
                s.lines()
                    .filter_map(|l| l.split_whitespace().nth(5).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        Vec::new()
    }
}

/// Загрузка на `Device::Cpu` и 14 прогонов golden не подтягивают ни одной библиотеки CUDA.
#[test]
fn cpu_path_loads_no_cuda_libraries() {
    let m = YttriDecider::load(dir().join("model.gguf"), &Device::Cpu).unwrap();
    let g: Value =
        serde_json::from_slice(&std::fs::read(dir().join("golden.json")).unwrap()).unwrap();
    for c in g["cases"].as_array().unwrap() {
        let ids: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        m.forward(&ids).unwrap();
    }
    let images = loaded_images();
    let cuda: Vec<_> = images
        .iter()
        .filter(|n| {
            let n = n.to_lowercase();
            [
                "nvcuda", "cublas", "cudart", "curand", "nvrtc", "cudnn", "libcuda", "nvptxjit",
            ]
            .iter()
            .any(|p| n.contains(p))
        })
        .collect();
    eprintln!(
        "образов в процессе: {}; библиотеки CUDA: {cuda:?}",
        images.len()
    );
    assert!(cuda.is_empty(), "CPU-путь подтянул CUDA: {cuda:?}");
    #[cfg(any(windows, target_os = "linux"))]
    assert!(
        !images.is_empty(),
        "список образов пуст: проверка ничего не доказывает"
    );
}
