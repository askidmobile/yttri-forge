//! iq1s_check — деквант первого блока реального GGUF для сверки с python-эталоном.
use candle_core::quantized::iq1s::dequantize_iq1_s;
use std::io::{Read, Seek};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("нужен путь к GGUF");
    let start: u64 = args.next().expect("нужен offset тензора").parse()?;
    let mut f = std::fs::File::open(&path)?;
    f.seek(std::io::SeekFrom::Start(start))?;
    let mut buf = vec![0u8; 50];
    f.read_exact(&mut buf)?;
    println!("raw hex: {}", buf.iter().map(|b| format!("{:02x}", b)).collect::<String>());
    let d = half::f16::from_le_bytes([buf[0], buf[1]]).to_f32();
    println!("d = {}", d);
    let y = dequantize_iq1_s(&buf, 256);
    let vals: Vec<f32> = y[..16].iter().map(|v| (v * 1000.0).round() / 1000.0).collect();
    println!("rust first 16: {:?}", vals);
    Ok(())
}
