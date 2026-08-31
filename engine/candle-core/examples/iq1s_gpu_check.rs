//! iq1s_gpu_check — 2 блока с сильно различающимися значениями.
use candle_core::quantized::{ggml_file, iq1s, GgmlDType};
use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let mut data = vec![0u8; iq1s::BLOCK_IQ1S_BYTES * 2];
    data[0..2].copy_from_slice(&half::f16::to_le_bytes(half::f16::ONE));
    data[2] = 0x01;
    data[50..52].copy_from_slice(&half::f16::to_le_bytes(half::f16::from_f32(6.0)));
    data[52] = 0x99;
    data[53] = 0x77;
    let qh: u16 = 2 << 12;
    data[50 + 34..50 + 36].copy_from_slice(&qh.to_le_bytes());
    let cpu_ref = iq1s::dequantize_iq1_s(&data, 2 * iq1s::QK_K);
    let q = ggml_file::qtensor_from_ggml(GgmlDType::IQ1S, &data, vec![2 * iq1s::QK_K], &dev)?;
    let g = q.dequantize(&dev)?.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    println!("BLOCK A (gpu):       {:?}", &g[0..8]);
    println!("BLOCK A (cpu):       {:?}", &cpu_ref[0..8]);
    println!("BLOCK B (gpu):       {:?}", &g[256..264]);
    println!("BLOCK B (cpu):       {:?}", &cpu_ref[256..264]);
    let mut maxdiff = 0f32;
    for i in 0..512 { let d = (g[i]-cpu_ref[i]).abs(); if d > maxdiff { maxdiff = d; } }
    println!("maxdiff = {maxdiff}");
    let _ = Tensor::zeros((), candle_core::DType::F32, &dev)?;
    Ok(())
}
