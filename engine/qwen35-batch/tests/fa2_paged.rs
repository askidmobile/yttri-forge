#![cfg(feature = "cuda")]
//! FA2 varlen paged vs flash_attn: точные размеры Qwen3.5-4B decode (GQA 24/8, hd=128, page=64).

use candle_core::{DType, Device, Result, Tensor};

#[test]
fn cuda_fa2_paged_qwen35_dims() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (b, n_head, n_kv, hd, page, len) = (1usize, 24usize, 8usize, 128usize, 64usize, 17usize);
    let num_blocks = 4usize;

    let q = Tensor::randn(0f32, 1.0, (b, n_head, hd), &device)?.to_dtype(DType::F16)?;
    let k_seq = Tensor::randn(0f32, 1.0, (len, n_kv, hd), &device)?.to_dtype(DType::F16)?;
    let v_seq = Tensor::randn(0f32, 1.0, (len, n_kv, hd), &device)?.to_dtype(DType::F16)?;
    let pad = Tensor::zeros((num_blocks * page - len, n_kv, hd), DType::F16, &device)?;
    let k_pool = Tensor::cat(&[&k_seq, &pad], 0)?.reshape((num_blocks, page, n_kv, hd))?;
    let v_pool = Tensor::cat(&[&v_seq, &pad], 0)?.reshape((num_blocks, page, n_kv, hd))?;

    let seqlens_q = Tensor::from_vec(vec![0u32, 1u32], 2, &device)?;
    let seqlens_k = Tensor::from_vec(vec![0u32, len as u32], 2, &device)?;
    let block_table = Tensor::from_vec(
        (0..num_blocks as u32).collect::<Vec<_>>(),
        (1, num_blocks),
        &device,
    )?;

    let scale = 1.0 / (hd as f64).sqrt();
    let out_paged = candle_flash_attn::flash_attn_varlen_paged_windowed(
        &q,
        &k_pool,
        &v_pool,
        &seqlens_q,
        &seqlens_k,
        &block_table,
        None,
        1,
        len,
        scale as f32,
        None,
        None,
        page,
        None,
    )?;
    let q_ref = q.unsqueeze(1)?; // [1, 1, n_head, hd] = (b, seq, heads, hd)
    let k_ref = k_seq.unsqueeze(0)?;
    let v_ref = v_seq.unsqueeze(0)?;
    let out_ref = candle_flash_attn::flash_attn(&q_ref, &k_ref, &v_ref, scale as f32, true)?;
    let out_ref = out_ref.squeeze(1)?;
    let diff = (&out_paged.to_dtype(DType::F32)? - &out_ref.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff.is_finite() && diff < 1e-3, "paged FA2 mismatch: {diff}");
    Ok(())
}
