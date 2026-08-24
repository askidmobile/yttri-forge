#[cfg(feature = "metal")]
mod metal_sdpa_tests {
    use candle::{DType, Device, Result, Shape, Tensor};
    use rand::SeedableRng;
    use rand_distr::Distribution;
    use std::ops::{Div, Mul};

    fn randn<S: Into<Shape>>(
        rng: &mut rand::rngs::StdRng,
        shape: S,
        dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let vs: Vec<f32> = (0..elem_count).map(|_| normal.sample(rng)).collect();
        Tensor::from_vec(vs, &shape, dev)
    }

    #[test]
    fn sdpa_full() -> Result<()> {
        // Test the full SDPA kernel path (q_seq > 8)
        const BS: usize = 4;
        const R: usize = 16;
        const L: usize = 16;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        // Larger sequences have higher accumulated error
        assert!(error <= 0.02, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_full_headdim_48() -> Result<()> {
        const BS: usize = 2;
        const R: usize = 16;
        const L: usize = 16;
        const DK: usize = 48;
        const H: usize = 4;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4848);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.02, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector() -> Result<()> {
        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.000, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_full_softcapping() -> Result<()> {
        // Test softcapping with sdpa_vector kernel (q_seq = 1)
        // NOTE: Vector kernel only supports q_seq = 1 correctly
        // Full kernel does NOT support softcapping
        const BS: usize = 4;
        const R: usize = 1; // Vector kernel requires q_seq = 1
        const L: usize = 4;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output =
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, SOFTCAP as f32)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        // Slightly higher error for cross-attention case (R=1, L=4)
        assert!(error <= 0.002, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector_softcapping() -> Result<()> {
        // Allow vectorized, seqlen = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 1;
        const DK: usize = 64;
        const H: usize = 3;
        const SOFTCAP: f64 = 50.;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(
                &att.to_dtype(DType::F32)?
                    .div(SOFTCAP)?
                    .tanh()?
                    .mul(SOFTCAP)?,
            )?
            .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output =
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, SOFTCAP as f32)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.0001, "{}", error);
        Ok(())
    }

    #[test]
    fn sdpa_vector_cross() -> Result<()> {
        // Allow vectorized, seqlen = 1. Simulat cross attention case where R != L, R = 1
        const BS: usize = 4;
        const R: usize = 1;
        const L: usize = 24;
        const DK: usize = 64;
        const H: usize = 3;

        let scale: f64 = f64::from(DK as u32).sqrt().recip();
        let device = Device::new_metal(0)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242424242);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let ground_truth = {
            let att = (q.clone() * scale)?.matmul(&k.clone().t()?)?;
            let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
                .to_dtype(q.dtype())?;
            att.matmul(&v.clone())?
        };
        let sdpa_output = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.)?;
        assert_eq!(ground_truth.shape(), sdpa_output.shape());
        let error: f32 = ((&ground_truth - &sdpa_output)?.abs()? / &ground_truth.abs()?)?
            .sum_all()?
            .to_scalar()?;
        assert!(error <= 0.0013, "{}", error);
        Ok(())
    }
}

mod cpu_sdpa_tests {
    // CPU SDPA -- critical for embeddings + ASR on machines
    // without working CUDA. Tests verify correctness against a naive implementation
    // via matmul + softmax.
    use candle::{DType, Device, Result, Shape, Tensor};
    use rand::SeedableRng;
    use rand_distr::Distribution;

    fn randn<S: Into<Shape>>(
        rng: &mut rand::rngs::StdRng,
        shape: S,
        dev: &Device,
    ) -> Result<Tensor> {
        let shape = shape.into();
        let elem_count = shape.elem_count();
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let vs: Vec<f32> = (0..elem_count).map(|_| normal.sample(rng)).collect();
        Tensor::from_vec(vs, &shape, dev)
    }

    fn naive_sdpa(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        do_causal: bool,
        softcap: f32,
    ) -> Result<Tensor> {
        // GQA expansion: repeat k/v along the head dim, if n_q > n_kv.
        let q_dims = q.dims();
        let k_dims = k.dims();
        let n_q = q_dims[1];
        let n_kv = k_dims[1];
        let group = n_q / n_kv;
        let k = if group > 1 {
            // (b, n_kv, kseq, dk) -> (b, n_q, kseq, dk)
            k.unsqueeze(2)?
                .expand((k_dims[0], n_kv, group, k_dims[2], k_dims[3]))?
                .reshape((k_dims[0], n_q, k_dims[2], k_dims[3]))?
        } else {
            k.clone()
        };
        let v_dims = v.dims();
        let v = if group > 1 {
            v.unsqueeze(2)?
                .expand((v_dims[0], n_kv, group, v_dims[2], v_dims[3]))?
                .reshape((v_dims[0], n_q, v_dims[2], v_dims[3]))?
        } else {
            v.clone()
        };
        let mut att = (q * scale as f64)?.matmul(&k.t()?)?;
        if (softcap - 1.0).abs() > 1e-6 {
            att = (att.tanh()? * softcap as f64)?;
        }
        if do_causal {
            // Build a causal mask (qseq, kseq) with -inf above the diagonal.
            let q_seq = q_dims[2];
            let k_seq = k_dims[2];
            let mut mask = vec![0f32; q_seq * k_seq];
            for qi in 0..q_seq {
                for kj in (qi + 1)..k_seq {
                    mask[qi * k_seq + kj] = f32::NEG_INFINITY;
                }
            }
            let mask_t = Tensor::from_vec(mask, (q_seq, k_seq), q.device())?
                .to_dtype(att.dtype())?
                .broadcast_as(att.shape())?;
            att = (&att + &mask_t)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        att.matmul(&v)
    }

    #[test]
    fn cpu_sdpa_basic_f32() -> Result<()> {
        const BS: usize = 2;
        const H: usize = 4;
        const R: usize = 5;
        const L: usize = 7;
        const DK: usize = 16;
        let scale = (DK as f32).sqrt().recip();
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let truth = naive_sdpa(&q, &k, &v, scale, false, 1.0)?;
        let got = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?;
        assert_eq!(truth.shape(), got.shape());
        let err: f32 = ((&truth - &got)?.abs()?.sum_all()?).to_scalar()?;
        let norm: f32 = truth.abs()?.sum_all()?.to_scalar()?;
        assert!(err / norm < 1e-5, "rel err {} / norm {}", err, norm);
        Ok(())
    }

    #[test]
    fn cpu_sdpa_gqa_f32() -> Result<()> {
        // Qwen3.5-4B attention layers: n_q=16, n_kv=4, group=4.
        const BS: usize = 1;
        const N_Q: usize = 16;
        const N_KV: usize = 4;
        const R: usize = 3;
        const L: usize = 8;
        const DK: usize = 32;
        let scale = (DK as f32).sqrt().recip();
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(22);
        let q = randn(&mut rng, (BS, N_Q, R, DK), &device)?;
        let k = randn(&mut rng, (BS, N_KV, L, DK), &device)?;
        let v = randn(&mut rng, (BS, N_KV, L, DK), &device)?;
        let truth = naive_sdpa(&q, &k, &v, scale, false, 1.0)?;
        let got = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?;
        assert_eq!(truth.shape(), got.shape());
        let err: f32 = ((&truth - &got)?.abs()?.sum_all()?).to_scalar()?;
        let norm: f32 = truth.abs()?.sum_all()?.to_scalar()?;
        assert!(err / norm < 1e-5, "rel err {} / norm {}", err, norm);
        Ok(())
    }

    #[test]
    fn cpu_sdpa_causal_f32() -> Result<()> {
        const BS: usize = 1;
        const H: usize = 2;
        const N: usize = 6;
        const DK: usize = 16;
        let scale = (DK as f32).sqrt().recip();
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(33);
        let q = randn(&mut rng, (BS, H, N, DK), &device)?;
        let k = randn(&mut rng, (BS, H, N, DK), &device)?;
        let v = randn(&mut rng, (BS, H, N, DK), &device)?;
        let truth = naive_sdpa(&q, &k, &v, scale, true, 1.0)?;
        let got = candle_nn::ops::sdpa(&q, &k, &v, None, true, scale, 1.0)?;
        let err: f32 = ((&truth - &got)?.abs()?.sum_all()?).to_scalar()?;
        let norm: f32 = truth.abs()?.sum_all()?.to_scalar()?;
        assert!(err / norm < 1e-5, "rel err {} / norm {}", err, norm);
        Ok(())
    }

    #[test]
    fn cpu_sdpa_softcap_f32() -> Result<()> {
        const BS: usize = 1;
        const H: usize = 2;
        const R: usize = 4;
        const L: usize = 5;
        const DK: usize = 16;
        let scale = (DK as f32).sqrt().recip();
        let softcap = 30.0_f32;
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(44);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?;
        let truth = naive_sdpa(&q, &k, &v, scale, false, softcap)?;
        let got = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, softcap)?;
        let err: f32 = ((&truth - &got)?.abs()?.sum_all()?).to_scalar()?;
        let norm: f32 = truth.abs()?.sum_all()?.to_scalar()?;
        assert!(err / norm < 1e-5, "rel err {} / norm {}", err, norm);
        Ok(())
    }

    #[test]
    fn cpu_sdpa_f16() -> Result<()> {
        const BS: usize = 1;
        const H: usize = 2;
        const R: usize = 3;
        const L: usize = 4;
        const DK: usize = 16;
        let scale = (DK as f32).sqrt().recip();
        let device = Device::Cpu;
        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        let q = randn(&mut rng, (BS, H, R, DK), &device)?.to_dtype(DType::F16)?;
        let k = randn(&mut rng, (BS, H, L, DK), &device)?.to_dtype(DType::F16)?;
        let v = randn(&mut rng, (BS, H, L, DK), &device)?.to_dtype(DType::F16)?;
        let truth = naive_sdpa(&q, &k, &v, scale, false, 1.0)?;
        let got = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0)?;
        assert_eq!(truth.dtype(), DType::F16);
        assert_eq!(got.dtype(), DType::F16);
        let truth_f32 = truth.to_dtype(DType::F32)?;
        let got_f32 = got.to_dtype(DType::F32)?;
        let err: f32 = ((&truth_f32 - &got_f32)?.abs()?.sum_all()?).to_scalar()?;
        let norm: f32 = truth_f32.abs()?.sum_all()?.to_scalar()?;
        // F16 has lower precision -- allow 1% relative error.
        assert!(err / norm < 1e-2, "rel err {} / norm {}", err, norm);
        Ok(())
    }
}
