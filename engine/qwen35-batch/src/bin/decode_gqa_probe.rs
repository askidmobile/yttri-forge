//! Parity и микробенч paged GQA decode-attention без загрузки всей модели.
//! Запуск: decode_gqa_probe [kv_len] [iters]

use candle_core::{DType, Device, Tensor};

fn check_gqa_heads(
    dev: &Device,
    cuda: &candle_core::CudaDevice,
    kv_len: usize,
) -> candle_core::Result<()> {
    use qwen35_batch::real::flash_decode_cuda::dispatch_flash_decode_paged;

    const HEADS: usize = 16;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const SCALE: f32 = 1.0 / 16.0;

    let blocks = kv_len.div_ceil(64);
    let mut q_host = vec![0f32; HEADS * HEAD_DIM];
    let mut expected = vec![0f32; HEADS * HEAD_DIM];
    let mut k_host = vec![0f32; blocks * 64 * KV_HEADS * HEAD_DIM];
    let mut v_host = vec![0f32; blocks * 64 * KV_HEADS * HEAD_DIM];
    for h in 0..HEADS {
        let x = h as f32 - 4.0;
        q_host[h * HEAD_DIM] = 16.0 * x;
        expected[h * HEAD_DIM] = 1.0 / (1.0 + (-2.0 * x).exp());
    }
    for p in 0..kv_len {
        let positive = p % 2 == 1;
        for kvh in 0..KV_HEADS {
            let idx = (p * KV_HEADS + kvh) * HEAD_DIM;
            k_host[idx] = if positive { 1.0 } else { -1.0 };
            v_host[idx] = f32::from(positive as u8);
        }
    }

    let q = Tensor::from_vec(q_host, (1, HEADS, HEAD_DIM), dev)?.to_dtype(DType::F16)?;
    let k =
        Tensor::from_vec(k_host, (blocks, 64, KV_HEADS, HEAD_DIM), dev)?.to_dtype(DType::F16)?;
    let v =
        Tensor::from_vec(v_host, (blocks, 64, KV_HEADS, HEAD_DIM), dev)?.to_dtype(DType::F16)?;
    let table = Tensor::from_vec((0..blocks as u32).collect(), (1, blocks), dev)?;
    let lens = Tensor::from_vec(vec![0u32, kv_len as u32], (2,), dev)?;
    let got = dispatch_flash_decode_paged(cuda, &q, &k, &v, &lens, &table, SCALE)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut reference = expected;
    if kv_len % 2 != 0 {
        // Нечётная длина проверяет хвост; из-за неодинакового числа sign-строк
        // эталоном служит прежнее scalar-ядро, а не аналитическая sigmoid.
        let old_fold = std::env::var_os("DECODE_GQA_FOLD");
        std::env::set_var("DECODE_GQA_FOLD", "0");
        reference = dispatch_flash_decode_paged(cuda, &q, &k, &v, &lens, &table, SCALE)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        std::env::set_var("DECODE_GQA_FOLD", "1");
        match old_fold {
            Some(value) => std::env::set_var("DECODE_GQA_FOLD", value),
            None => std::env::remove_var("DECODE_GQA_FOLD"),
        }
    }
    let max_error = got
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    if max_error > 1e-3 {
        candle_core::bail!(
            "GQA head parity kv={kv_len}: max|delta|={max_error}, got first/group={:?}, expected={:.6}",
            got.iter().step_by(HEAD_DIM).take(HEADS).collect::<Vec<_>>(),
            reference[0]
        );
    }
    eprintln!("[decode-gqa] kv={kv_len} head parity max|delta|={max_error:.6}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    use candle_core::Result;
    use qwen35_batch::real::flash_decode_cuda::dispatch_flash_decode_paged;

    fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let kv_len: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(131072);
        let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
        let dev = Device::new_cuda(0)?;
        let cuda = dev.as_cuda_device()?.clone();
        check_gqa_heads(&dev, &cuda, 64)?;
        check_gqa_heads(&dev, &cuda, 257)?;
        let blocks = kv_len.div_ceil(64);
        let k = Tensor::randn(0f32, 1f32, (blocks, 64, 2, 256), &dev)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1f32, (blocks, 64, 2, 256), &dev)?.to_dtype(DType::F16)?;
        let q = Tensor::randn(0f32, 1f32, (1, 16, 256), &dev)?.to_dtype(DType::F16)?;
        let table: Vec<u32> = (0..blocks as u32).collect();
        let table = Tensor::from_vec(table, (1, blocks), &dev)?;
        let lens = Tensor::from_vec(vec![0u32, kv_len as u32], (2,), &dev)?;
        dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
        cuda.cuda_stream()
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
        }
        cuda.cuda_stream()
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        eprintln!(
            "kv={kv_len} iters={iters} splits={} ms={:.3}",
            std::env::var("DECODE_GQA_SPLITS").unwrap_or_else(|_| "auto".into()),
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        );
        Ok(())
    }

    run().map_err(Into::into)
}
