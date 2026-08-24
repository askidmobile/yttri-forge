#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("qwen35_vision_probe requires --features real-model");
    std::process::exit(2);
}

#[cfg(feature = "real-model")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use qwen35_batch::real::{multimodal::GridThw, vision::Qwen35Vision};
    use std::path::Path;

    let mut args = std::env::args().skip(1);
    let gguf = args.next().ok_or_else(|| anyhow::anyhow!("usage: qwen35_vision_probe VISION.gguf [cpu|cuda|metal] [patches.bin T H W]"))?;
    let backend = args.next().unwrap_or_else(|| "cpu".into());
    let reference = std::env::var("QWEN35_VISION_REFERENCE").as_deref() == Ok("1");
    let patch_dtype = match std::env::var("QWEN35_VISION_PATCH_DTYPE").as_deref() {
        Err(_) | Ok("f32") => DType::F32,
        Ok("f16") => DType::F16,
        Ok("bf16") => DType::BF16,
        Ok(value) => anyhow::bail!("unsupported patch dtype {value:?}"),
    };
    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(0)?,
        other => anyhow::bail!("unsupported backend {other:?}"),
    };
    let model = if reference {
        Qwen35Vision::load_reference(Path::new(&gguf), device.clone())?
    } else {
        Qwen35Vision::load(Path::new(&gguf), device.clone())?
    };
    let Some(patch_path) = args.next() else {
        println!(
            "profile=pass blocks={} hidden={} projection={} quant={:?}",
            model.profile().block_count,
            model.profile().hidden_size,
            model.profile().projection_dim,
            model.profile().quant_set,
        );
        return Ok(());
    };
    let t: usize = args.next().ok_or_else(|| anyhow::anyhow!("missing T"))?.parse()?;
    let h: usize = args.next().ok_or_else(|| anyhow::anyhow!("missing H"))?.parse()?;
    let w: usize = args.next().ok_or_else(|| anyhow::anyhow!("missing W"))?.parse()?;
    let bytes = std::fs::read(&patch_path)?;
    if bytes.len() % 4 != 0 {
        anyhow::bail!("patch file length is not divisible by 4");
    }
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let rows = t.checked_mul(h).and_then(|v| v.checked_mul(w)).ok_or_else(|| anyhow::anyhow!("grid overflow"))?;
    let patches = Tensor::from_vec(values, (rows, 3 * 2 * 16 * 16), &device)?
        .to_dtype(patch_dtype)?;
    let output = model.forward(&patches, &[GridThw { t, h, w }])?;
    device.synchronize()?;
    let values = output.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let path = format!("{patch_path}.{backend}.output.f32");
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(&path, bytes)?;
    println!(
        "shape={:?} finite={} l2={norm:.9} output={path}",
        output.dims(),
        values.iter().all(|value| value.is_finite())
    );
    Ok(())
}
