//! Фаза 1 плана 2026-09-04-moe-expert-offload (FR-002, FR-003): ядра
//! `indexed_moe_forward*` адресуют эксперта через таблицу указателей. Тест
//! проверяет бит в бит, что при одном и том же квантованном входе выход
//! одинаков для трёх раскладок:
//! (а) упакованная VRAM-раскладка (ленивая таблица `base + id·stride`),
//! (б) таблица с перестановкой слотов в VRAM (каждый эксперт — отдельный буфер),
//! (в) таблица на device-mapped pinned host-памяти.
//! Проверяется `_q8_1` (plain) и `_dual` (gate+up) для шести dtype.
//!
//! Запуск: `cargo test --features cuda -p candle-core moe_table`

#![cfg(feature = "cuda")]

use candle_core::cuda_backend::cudarc::driver::{result, sys, DevicePtr};
use candle_core::cuda_backend::cudarc;
use candle_core::cuda_backend::CudaDevice;
use candle_core::quantized::gguf_file;
use candle_core::quantized::{
    indexed_moe_forward_dual_table, indexed_moe_forward_table, GgmlDType, QTensor,
};
use candle_core::{Device, Tensor};

const N_EXPERTS: usize = 4;
const N: usize = 32; // строк выхода на эксперта
const K: usize = 256; // ширина входа (ровно один блок всех перечисленных dtype)
const TOPK: usize = 2;
const BATCH: usize = 3;

/// Перестановка слотов: slot[i] хранит эксперта PERM[i].
const PERM: [usize; N_EXPERTS] = [2, 0, 3, 1];

struct HostMapped {
    ptr: *mut u8,
    dev_ptr: sys::CUdeviceptr,
    bytes: usize,
}

impl HostMapped {
    unsafe fn alloc_fill(bytes: usize, data: &[u8]) -> Self {
        assert_eq!(bytes, data.len());
        let raw = result::malloc_host(bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP)
            .expect("malloc_host DEVICEMAP");
        assert!(!raw.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), raw as *mut u8, bytes);
        }
        let mut dev_ptr: sys::CUdeviceptr = 0;
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, raw, 0) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer_v2");
        assert_ne!(dev_ptr, 0);
        Self {
            ptr: raw as *mut u8,
            dev_ptr,
            bytes,
        }
    }
}

impl Drop for HostMapped {
    fn drop(&mut self) {
        unsafe {
            let _ = result::free_host(self.ptr as *mut _);
        }
    }
}

fn test_dtypes() -> Vec<GgmlDType> {
    vec![
        GgmlDType::IQ2XXS,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ2S,
        GgmlDType::IQ4XS,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
    ]
}

/// Синтез байтов квантованных данных (LCG). CPU-квантователя для IQ-типов в
/// форке нет, а для бит-в-бит сравнения раскладок содержимое блоков должно
/// лишь быть валидной памятью и совпадать между раскладками — LCG даёт
/// детерминированный байт-в-байт паттерн.
fn synth_bytes(seed: u32, len: usize) -> Vec<u8> {
    let mut x = 0x1234_5678u32 ^ (seed.wrapping_mul(0x9E37_79B9));
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 16) as u8
        })
        .collect()
}

/// QTensor на устройстве из точных байтов (без перекодирования) — через
/// gguf-механику чтения по смещениям.
fn qtensor_from_bytes(bytes: &[u8], dtype: GgmlDType, device: &Device) -> QTensor {
    let content = gguf_file::Content {
        magic: gguf_file::VersionedMagic::GgufV3,
        metadata: std::collections::HashMap::new(),
        tensor_infos: std::iter::once((
            "w".to_string(),
            gguf_file::TensorInfo {
                ggml_dtype: dtype,
                shape: (N_EXPERTS, N, K).into(),
                offset: 0,
            },
        ))
        .collect(),
        tensor_data_offset: 0,
    };
    content.tensor_from_slice(bytes, "w", device).unwrap()
}

fn row_bytes(dtype: GgmlDType) -> usize {
    K / dtype.block_size() * dtype.type_size()
}

fn input_tensor(device: &Device, dim1: usize) -> Tensor {
    let data: Vec<f32> = (0..BATCH * dim1 * K)
        .map(|i| ((i % 23) as f32) * 0.05 - 0.5)
        .collect();
    Tensor::from_vec(data, (BATCH, dim1, K), device).unwrap()
}

fn ids_tensor(device: &Device) -> Tensor {
    // В пределах N_EXPERTS, с повторами — как реальный роутинг.
    let ids: Vec<u32> = vec![0, 3, 2, 1, 3, 0];
    Tensor::from_vec(ids, (BATCH, TOPK), device).unwrap()
}

fn assert_bit_eq(tag: &str, a: &Tensor, b: &Tensor) {
    let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let vb = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(va.len(), vb.len(), "{tag}: длина выхода");
    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{tag}: выход разошёлся в элементе {i}: {x} vs {y}"
        );
    }
    assert!(
        va.iter().any(|v| v.to_bits() != 0),
        "{tag}: выход весь нулевой — ядро не считало?"
    );
}

/// Таблица указателей: slot[i] хранит эксперта PERM[i], поэтому
/// table[PERM[i]] = адрес слота i. Эксперт ищется ядром по table[expert_id].
fn build_table(device: &CudaDevice, ptrs: &[sys::CUdeviceptr; N_EXPERTS]) -> cudarc::driver::CudaSlice<u64> {
    let mut host = vec![0u64; N_EXPERTS];
    for (slot, &expert) in PERM.iter().enumerate() {
        host[expert] = ptrs[slot] as u64;
    }
    let mut tbl = unsafe { device.alloc::<u64>(N_EXPERTS) }.unwrap();
    device.memcpy_htod(&host, &mut tbl).unwrap();
    tbl
}

fn run_case(device: &Device, cdev: &CudaDevice, dtype: GgmlDType) {
    let stream = cdev.cuda_stream();

    // ── (а) упакованная раскладка: QTensor прямо на CUDA ──
    let row = row_bytes(dtype);
    let ebytes = N * row;
    let gate_bytes = synth_bytes(1, N_EXPERTS * ebytes);
    let up_bytes = synth_bytes(2, N_EXPERTS * ebytes);
    let gate_packed = qtensor_from_bytes(&gate_bytes, dtype, device);
    let up_packed = qtensor_from_bytes(&up_bytes, dtype, device);

    // ── (б) перестановка слотов в VRAM: отдельный буфер на эксперта ──
    let mut gate_slots: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
    let mut gate_ptrs = [0u64 as sys::CUdeviceptr; N_EXPERTS];
    for slot in 0..N_EXPERTS {
        let e = PERM[slot]; // в слот slot кладём эксперта PERM[slot]
        let chunk = &gate_bytes[e * ebytes..(e + 1) * ebytes];
        let mut ws = unsafe { cdev.alloc::<u8>(ebytes) }.unwrap();
        cdev.memcpy_htod(chunk, &mut ws).unwrap();
        // Указатель достаём в узком скоупе: guard лишь ведёт учёт событий
        // чтения, а валидность адреса определяется жизнью ws.
        gate_ptrs[slot] = {
            let (ptr, _guard) = DevicePtr::<u8>::device_ptr(&ws, &stream);
            ptr
        };
        gate_slots.push(ws);
        // _guard живёт до конца итерации; значение ptr стабильно, пока жив ws.
    }
    let mut up_slots: Vec<cudarc::driver::CudaSlice<u8>> = Vec::new();
    let mut up_ptrs = [0u64 as sys::CUdeviceptr; N_EXPERTS];
    for slot in 0..N_EXPERTS {
        let e = PERM[slot];
        let chunk = &up_bytes[e * ebytes..(e + 1) * ebytes];
        let mut ws = unsafe { cdev.alloc::<u8>(ebytes) }.unwrap();
        cdev.memcpy_htod(chunk, &mut ws).unwrap();
        up_ptrs[slot] = {
            let (ptr, _guard) = DevicePtr::<u8>::device_ptr(&ws, &stream);
            ptr
        };
        up_slots.push(ws);
    }

    // ── (в) device-mapped pinned host-память ──
    let mut gate_host: Vec<HostMapped> = Vec::new();
    let mut gate_host_ptrs = [0u64 as sys::CUdeviceptr; N_EXPERTS];
    for slot in 0..N_EXPERTS {
        let e = PERM[slot];
        let chunk = &gate_bytes[e * ebytes..(e + 1) * ebytes];
        let h = unsafe { HostMapped::alloc_fill(ebytes, chunk) };
        gate_host_ptrs[slot] = h.dev_ptr;
        gate_host.push(h);
    }
    let mut up_host: Vec<HostMapped> = Vec::new();
    let mut up_host_ptrs = [0u64 as sys::CUdeviceptr; N_EXPERTS];
    for slot in 0..N_EXPERTS {
        let e = PERM[slot];
        let chunk = &up_bytes[e * ebytes..(e + 1) * ebytes];
        let h = unsafe { HostMapped::alloc_fill(ebytes, chunk) };
        up_host_ptrs[slot] = h.dev_ptr;
        up_host.push(h);
    }

    let table_gate_vram = build_table(cdev, &gate_ptrs);
    let table_up_vram = build_table(cdev, &up_ptrs);
    let table_gate_host = build_table(cdev, &gate_host_ptrs);
    let table_up_host = build_table(cdev, &up_host_ptrs);

    // ── plain `_q8_1` ──
    let input1 = input_tensor(device, 1); // [batch, 1, k] — общий вход на экспертов
    let ids = ids_tensor(device);

    let out_packed = gate_packed
        .indexed_moe_forward_cuda(&input1, &ids)
        .unwrap();
    let out_vram = indexed_moe_forward_table(
        cdev, dtype, (N_EXPERTS, N, K), &table_gate_vram, &input1, &ids,
    )
    .unwrap();
    let out_host = indexed_moe_forward_table(
        cdev, dtype, (N_EXPERTS, N, K), &table_gate_host, &input1, &ids,
    )
    .unwrap();
    assert_bit_eq("plain packed vs vram-permuted", &out_packed, &out_vram);
    assert_bit_eq("plain packed vs host-mapped", &out_packed, &out_host);

    // ── dual (gate+up) ──
    let input2 = input_tensor(device, TOPK); // [batch, topk, k]
    let (g_packed, u_packed) = gate_packed
        .indexed_moe_forward_dual_cuda(&up_packed, &input2, &ids)
        .unwrap();
    let (g_vram, u_vram) = indexed_moe_forward_dual_table(
        cdev, dtype, (N_EXPERTS, N, K), &table_gate_vram, &table_up_vram, &input2, &ids,
    )
    .unwrap();
    let (g_host, u_host) = indexed_moe_forward_dual_table(
        cdev, dtype, (N_EXPERTS, N, K), &table_gate_host, &table_up_host, &input2, &ids,
    )
    .unwrap();
    assert_bit_eq("dual gate packed vs vram-permuted", &g_packed, &g_vram);
    assert_bit_eq("dual up packed vs vram-permuted", &u_packed, &u_vram);
    assert_bit_eq("dual gate packed vs host-mapped", &g_packed, &g_host);
    assert_bit_eq("dual up packed vs host-mapped", &u_packed, &u_host);
}

#[test]
fn moe_table_bit_exact_across_placements() {
    let device = Device::new_cuda(0).expect("CUDA устройство");
    let cdev = device.as_cuda_device().unwrap().clone();
    for dtype in test_dtypes() {
        println!("moe_table: {dtype:?}");
        run_case(&device, &cdev, dtype);
    }
}
