#![cfg(feature = "real-model")]

use candle_core::{quantized::gguf_file, Device, Result, Tensor};
use qwen35_batch::real::{model_profile::VisionProfile, vision::Qwen35Vision};
use std::{fs::File, io::BufReader, path::Path};

fn vision_path() -> Option<&'static Path> {
    std::env::var_os("QWEN35_VISION_GGUF")
        .map(|path| Box::leak(Path::new(&path).to_path_buf().into_boxed_path()) as &'static Path)
}

#[test]
fn official_vision_profile_is_strict() -> Result<()> {
    let Some(path) = vision_path() else {
        return Ok(());
    };
    let mut reader = BufReader::new(File::open(path)?);
    let content = gguf_file::Content::read(&mut reader)?;
    let profile = VisionProfile::read_and_validate(&content)?;
    assert_eq!(profile.block_count, 24);
    assert_eq!(profile.hidden_size, 1024);
    assert_eq!(profile.projection_dim, 2560);
    assert_eq!(content.tensor_infos.len(), 298);
    for layer in 0..24 {
        let dtype = content.tensor_infos[&format!("v.blk.{layer}.ffn_down.weight")].ggml_dtype;
        if (1..20).contains(&layer) {
            assert_eq!(dtype, candle_core::quantized::GgmlDType::Q8_0);
        } else {
            assert!(matches!(
                dtype,
                candle_core::quantized::GgmlDType::BF16
                    | candle_core::quantized::GgmlDType::F32
            ));
        }
    }
    Ok(())
}

#[test]
fn tiny_vision_forward_is_finite() -> Result<()> {
    let Some(path) = vision_path() else {
        return Ok(());
    };
    let model = Qwen35Vision::load(path, Device::Cpu)?;
    let patches = Tensor::zeros((4, 3 * 2 * 16 * 16), candle_core::DType::F32, &Device::Cpu)?;
    let output = model.forward(
        &patches,
        &[qwen35_batch::real::multimodal::GridThw { t: 1, h: 2, w: 2 }],
    )?;
    assert_eq!(output.dims(), [1, 2560]);
    assert!(output.flatten_all()?.to_vec1::<f32>()?.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn official_vision_loads_without_dequantizing_matrices() -> Result<()> {
    let Some(path) = vision_path() else {
        return Ok(());
    };
    let model = Qwen35Vision::load(path, Device::Cpu)?;
    assert_eq!(model.profile().block_count, 24);
    assert_eq!(model.profile().projection_dim, 2560);
    Ok(())
}
