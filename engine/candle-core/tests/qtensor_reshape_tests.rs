use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    Device, Module, Result, Tensor,
};

#[test]
fn q8_reshape_preserves_matmul_and_checks_layout() -> Result<()> {
    let device = Device::Cpu;
    let weights = Tensor::from_iter((0..256).map(|v| (v as f32 - 127.0) / 64.0), &device)?
        .reshape((2, 2, 2, 32))?;
    let quantized = QTensor::quantize(&weights, GgmlDType::Q8_0)?.reshape((2, 128))?;
    let input = Tensor::from_iter((0..128).map(|v| (v as f32 - 63.0) / 32.0), &device)?
        .reshape((1, 128))?;
    let expected = input.matmul(&weights.reshape((2, 128))?.t()?)?;
    let actual = QMatMul::from_qtensor(quantized)?.forward(&input)?;
    let max_abs = (&expected - actual)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(max_abs < 0.5, "Q8 reshape matmul max abs {max_abs}");

    let wrong_count = QTensor::quantize(&weights, GgmlDType::Q8_0)?.reshape((2, 127));
    assert!(wrong_count.is_err());
    let wrong_block = QTensor::quantize(&weights, GgmlDType::Q8_0)?.reshape((8, 4, 8));
    assert!(wrong_block.is_err());
    Ok(())
}
