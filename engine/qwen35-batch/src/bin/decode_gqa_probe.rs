//! Микробенч paged GQA decode-attention без загрузки всей модели.
//! Запуск: decode_gqa_probe [kv_len] [iters]

use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    use candle_core::Result;
    use qwen35_batch::real::flash_decode_cuda::dispatch_flash_decode_paged;

    fn run() -> Result<()> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let kv_len: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(131072);
        let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
        let dev = Device::new_cuda(0)?;
        let cuda = dev.as_cuda_device()?.clone();
        let blocks = kv_len.div_ceil(64);
        let k = Tensor::randn(0f32, 1f32, (blocks, 64, 2, 256), &dev)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1f32, (blocks, 64, 2, 256), &dev)?.to_dtype(DType::F16)?;
        let q = Tensor::randn(0f32, 1f32, (1, 16, 256), &dev)?.to_dtype(DType::F16)?;
        let table: Vec<u32> = (0..blocks as u32).collect();
        let table = Tensor::from_vec(table, (1, blocks), &dev)?;
        let lens = Tensor::from_vec(vec![0u32, kv_len as u32], (2,), &dev)?;
        dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
        }
        cuda.cuda_stream().synchronize().map_err(candle_core::Error::wrap)?;
        eprintln!(
            "kv={kv_len} iters={iters} splits={} ms={:.3}",
            std::env::var("DECODE_GQA_SPLITS").unwrap_or_else(|_| "auto".into()),
            t.elapsed().as_secs_f64() * 1e3 / iters as f64
        );
        Ok(())
    }

    run().map_err(Into::into)
}
