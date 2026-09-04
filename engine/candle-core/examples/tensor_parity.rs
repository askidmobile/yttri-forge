//! tensor_parity — один тензор GGUF: деквант и матвек на CUDA против CPU.
//!
//! Запуск: tensor_parity <model.gguf> <имя тензора> [<имя тензора> ...]
//!
//! Для каждого тензора: деквантование на обоих устройствах (max|Δ|, rel L2,
//! первая позиция расхождения) и матвек QMatMul со случайным входом при
//! b=1 (MMVQ) и b=4. Нужен, чтобы отличить «ядро деквантования врёт» от
//! «ошибка накапливается в модели»: послойный дамп 2026-09-04 показал, что
//! выход блока расходится на 3.7% сразу после одной матрицы (ssm_out), тогда
//! как её вход — на 0.5%.
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Module, Tensor};

fn stats(a: &[f32], b: &[f32]) -> (f32, f64, usize) {
    let mut mx = 0f32;
    let mut first = usize::MAX;
    let (mut nd, mut nb) = (0f64, 0f64);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > mx {
            mx = d;
        }
        if first == usize::MAX && d > 1e-3 * (1.0 + y.abs()) {
            first = i;
        }
        nd += (d as f64) * (d as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (mx, if nb > 0.0 { (nd / nb).sqrt() } else { 0.0 }, first)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("путь к GGUF");
    let names: Vec<String> = args.collect();
    let cuda = Device::new_cuda(0)?;
    let cpu = Device::Cpu;
    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    for name in &names {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("нет тензора {name}"))?;
        let dims: Vec<usize> = info.shape.dims().to_vec();
        println!("== {name}  {:?}  {:?}", dims, info.ggml_dtype);
        let q_cpu = std::sync::Arc::new(content.tensor(&mut file, name, &cpu)?);
        let q_gpu = std::sync::Arc::new(content.tensor(&mut file, name, &cuda)?);
        let d_cpu = q_cpu.dequantize(&cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let d_gpu = q_gpu.dequantize(&cuda)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (mx, rel, first) = stats(&d_gpu, &d_cpu);
        println!("   деквант   : max|Δ|={mx:.3e} rel_L2={rel:.3e} первое расхождение={}", if first == usize::MAX { "нет".to_string() } else { first.to_string() });
        // Матвек: y = W x, W [n, k]; x [b, k]
        let k = *dims.last().unwrap();
        for &b in &[1usize, 4] {
            let x = Tensor::randn(0f32, 1f32, (b, k), &cpu)?;
            let y_cpu = QMatMul::from_arc(q_cpu.clone())?.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let y_gpu = QMatMul::from_arc(q_gpu.clone())?
                .forward(&x.to_device(&cuda)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // Эталон: плотный f32 матвек по CPU-деквантованным весам.
            let w = Tensor::from_vec(d_cpu.clone(), (dims[0], k), &cpu)?;
            let y_ref = x.matmul(&w.t()?)?.flatten_all()?.to_vec1::<f32>()?;
            let (_, r_cpu, _) = stats(&y_cpu, &y_ref);
            let (_, r_gpu, _) = stats(&y_gpu, &y_ref);
            let (mx_gc, r_gc, _) = stats(&y_gpu, &y_cpu);
            // Эталонный путь CUDA: деквант + cuBLAS без квантования активаций.
            candle_core::quantized::cuda::set_force_dmmv(true);
            let y_dmmv = QMatMul::from_arc(q_gpu.clone())?
                .forward(&x.to_device(&cuda)?)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            candle_core::quantized::cuda::set_force_dmmv(false);
            let (_, r_dmmv, _) = stats(&y_dmmv, &y_ref);
            println!("   матвек b={b}: CPU/ref rel={r_cpu:.3e}  CUDA(mmvq)/ref rel={r_gpu:.3e}  CUDA(dmmv)/ref rel={r_dmmv:.3e}  CUDA/CPU rel={r_gc:.3e} max|Δ|={mx_gc:.3e}");
        }
    }
    Ok(())
}
