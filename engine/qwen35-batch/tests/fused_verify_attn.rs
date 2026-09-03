#![cfg(all(feature = "cuda", feature = "real-model"))]

//! Быстрая сверка attention-тензоров MTP verify без загрузки модели.
//! Построчный paged FA2 — эталон; слитый путь подаёт те же позиции одним batch.

use candle_core::{DType, Device, Result, Tensor};
use qwen35_batch::real::paged_attn::PagedAttn;
use qwen35_batch::real::paged_kv_cuda::{PagedModelCtx, PAGE_SIZE};

const N_HEAD: usize = 24;
const N_KV_HEAD: usize = 4;
const HEAD_DIM: usize = 256;
const VERIFY_ROWS: usize = 2;
const CONTEXT_WINDOW: usize = 131_072;

fn kv_scales<'a>() -> Option<(&'a Tensor, &'a Tensor)> {
    None
}

fn run_parity(prefix_len: usize) -> Result<f32> {
    let device = Device::new_cuda(0)?;
    device.set_seed(7)?;

    let total_len = prefix_len + VERIFY_ROWS;
    let pool_blocks = total_len.div_ceil(PAGE_SIZE);
    let max_blocks = CONTEXT_WINDOW / PAGE_SIZE;
    let q = Tensor::randn(0f32, 0.35, (VERIFY_ROWS, N_HEAD, HEAD_DIM), &device)?
        .to_dtype(DType::F16)?;
    let k_pool = Tensor::randn(
        0f32,
        0.35,
        (pool_blocks, PAGE_SIZE, N_KV_HEAD, HEAD_DIM),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let v_pool = Tensor::randn(
        0f32,
        0.35,
        (pool_blocks, PAGE_SIZE, N_KV_HEAD, HEAD_DIM),
        &device,
    )?
    .to_dtype(DType::F16)?;

    let cuda = device.as_cuda_device()?;
    let mut ctx = PagedModelCtx::new(cuda, 1, max_blocks)?;
    let mut pages = vec![0u32; max_blocks];
    for (logical, page) in pages.iter_mut().take(pool_blocks).enumerate() {
        *page = logical as u32;
    }
    ctx.stage_inputs(&[0], &[prefix_len], &pages)?;
    ctx.reset_kv_len(&[prefix_len as u32])?;

    let block_table = ctx.block_table(1)?;
    let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;
    let out_ref = Tensor::zeros(q.shape(), DType::F16, &device)?;
    let seqlens_q_row = ctx.seqlens_q(1)?;
    let lse_row = Tensor::zeros((N_HEAD, 1), DType::F32, &device)?;

    for row in 0..VERIFY_ROWS {
        let seqlens_k_row = ctx.seqlens_k_for_prefill(1, row + 1)?;
        let q_row = q.narrow(0, row, 1)?;
        let out_row = out_ref.narrow(0, row, 1)?;
        PagedAttn {
            q: &q_row,
            k_pool: &k_pool,
            v_pool: &v_pool,
            kv_scales: kv_scales(),
            seqlens_q: &seqlens_q_row,
            seqlens_k: &seqlens_k_row,
            block_table: &block_table,
            out: &out_row,
            softmax_lse: &lse_row,
            b: 1,
            h: N_HEAD,
            h_k: N_KV_HEAD,
            d: HEAD_DIM,
            max_seqlen_q: 1,
            max_seqlen_k: CONTEXT_WINDOW,
            softmax_scale: scale,
            window_left: -1,
            window_right: -1,
            page_block_size: PAGE_SIZE,
            shared_block_table: false,
            rows_per_position: 0,
        }
        .forward()?;
    }

    let seqlens_q = ctx.seqlens_q_verify(VERIFY_ROWS)?;
    let seqlens_k = ctx.seqlens_k_for_verify(VERIFY_ROWS)?;
    let out_fused = Tensor::zeros(q.shape(), DType::F16, &device)?;
    let lse = Tensor::zeros((N_HEAD, VERIFY_ROWS), DType::F32, &device)?;
    PagedAttn {
        q: &q,
        k_pool: &k_pool,
        v_pool: &v_pool,
        kv_scales: kv_scales(),
        seqlens_q: &seqlens_q,
        seqlens_k: &seqlens_k,
        block_table: &block_table,
        out: &out_fused,
        softmax_lse: &lse,
        b: VERIFY_ROWS,
        h: N_HEAD,
        h_k: N_KV_HEAD,
        d: HEAD_DIM,
        max_seqlen_q: 1,
        max_seqlen_k: CONTEXT_WINDOW,
        softmax_scale: scale,
        window_left: -1,
        window_right: -1,
        page_block_size: PAGE_SIZE,
        shared_block_table: true,
        rows_per_position: 0,
    }
    .forward()?;

    (&out_ref.to_dtype(DType::F32)? - &out_fused.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()
}

/// Однопроходный путь (VERIFY_ONEPASS): строки запроса свёрнуты по
/// GQA-группам — [k*ngroups, h_k, d], строка r = (позиция r/ngroups, группа
/// r%ngroups); причинная граница по позиции (rows_per_position), один проход
/// по KV. Требуется побитовое совпадение с построчным эталоном.
fn run_onepass_parity(prefix_len: usize) -> Result<f32> {
    let device = Device::new_cuda(0)?;
    device.set_seed(7)?;

    let ngroups = N_HEAD / N_KV_HEAD;
    let rows = VERIFY_ROWS * ngroups;
    let total_len = prefix_len + VERIFY_ROWS;
    let pool_blocks = total_len.div_ceil(PAGE_SIZE);
    let max_blocks = CONTEXT_WINDOW / PAGE_SIZE;
    let q = Tensor::randn(0f32, 0.35, (VERIFY_ROWS, N_HEAD, HEAD_DIM), &device)?
        .to_dtype(DType::F16)?;
    let k_pool = Tensor::randn(
        0f32,
        0.35,
        (pool_blocks, PAGE_SIZE, N_KV_HEAD, HEAD_DIM),
        &device,
    )?
    .to_dtype(DType::F16)?;
    let v_pool = Tensor::randn(
        0f32,
        0.35,
        (pool_blocks, PAGE_SIZE, N_KV_HEAD, HEAD_DIM),
        &device,
    )?
    .to_dtype(DType::F16)?;

    let cuda = device.as_cuda_device()?;
    let mut ctx = PagedModelCtx::new(cuda, 1, max_blocks)?;
    let mut pages = vec![0u32; max_blocks];
    for (logical, page) in pages.iter_mut().take(pool_blocks).enumerate() {
        *page = logical as u32;
    }
    ctx.stage_inputs(&[0], &[prefix_len], &pages)?;
    ctx.reset_kv_len(&[prefix_len as u32])?;

    let block_table = ctx.block_table(1)?;
    let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;

    // Построчный эталон — тот же, что в run_parity.
    let out_ref = Tensor::zeros(q.shape(), DType::F16, &device)?;
    let seqlens_q_row = ctx.seqlens_q(1)?;
    let lse_row = Tensor::zeros((N_HEAD, 1), DType::F32, &device)?;
    for row in 0..VERIFY_ROWS {
        let seqlens_k_row = ctx.seqlens_k_for_prefill(1, row + 1)?;
        let q_row = q.narrow(0, row, 1)?;
        let out_row = out_ref.narrow(0, row, 1)?;
        PagedAttn {
            q: &q_row,
            k_pool: &k_pool,
            v_pool: &v_pool,
            kv_scales: kv_scales(),
            seqlens_q: &seqlens_q_row,
            seqlens_k: &seqlens_k_row,
            block_table: &block_table,
            out: &out_row,
            softmax_lse: &lse_row,
            b: 1,
            h: N_HEAD,
            h_k: N_KV_HEAD,
            d: HEAD_DIM,
            max_seqlen_q: 1,
            max_seqlen_k: CONTEXT_WINDOW,
            softmax_scale: scale,
            window_left: -1,
            window_right: -1,
            page_block_size: PAGE_SIZE,
            shared_block_table: false,
            rows_per_position: 0,
        }
        .forward()?;
    }

    // Свёртка Q: [k, nh, d] -> [k*ngroups, h_k, d]; адрес строки —
    // ((p*h_k + hk)*ngroups + g)*d.
    let q_op = q
        .reshape((VERIFY_ROWS, N_KV_HEAD, ngroups, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((rows, N_KV_HEAD, HEAD_DIM))?;
    let out_op = Tensor::zeros(q_op.shape(), DType::F16, &device)?;
    let seqlens_q_op = Tensor::from_vec(vec![0u32, rows as u32], 2, &device)?;
    // kv0 + k: строки уже в пуле, kv_len не инкрементирован.
    let seqlens_k_op = Tensor::from_vec(vec![0u32, total_len as u32], 2, &device)?;
    let lse = Tensor::zeros((N_KV_HEAD, rows), DType::F32, &device)?;
    PagedAttn {
        q: &q_op,
        k_pool: &k_pool,
        v_pool: &v_pool,
        kv_scales: kv_scales(),
        seqlens_q: &seqlens_q_op,
        seqlens_k: &seqlens_k_op,
        block_table: &block_table,
        out: &out_op,
        softmax_lse: &lse,
        b: 1,
        h: N_KV_HEAD,
        h_k: N_KV_HEAD,
        d: HEAD_DIM,
        max_seqlen_q: rows,
        max_seqlen_k: CONTEXT_WINDOW,
        softmax_scale: scale,
        window_left: -1,
        // Правое окно 0 — причинность; граница по позиции (rows_per_position).
        window_right: 0,
        page_block_size: PAGE_SIZE,
        shared_block_table: false,
        rows_per_position: ngroups,
    }
    .forward()?;

    // Обратная переукладка [k*ngroups, h_k, d] -> [k, nh, d].
    let out_op = out_op
        .reshape((VERIFY_ROWS, ngroups, N_KV_HEAD, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((VERIFY_ROWS, N_HEAD, HEAD_DIM))?;

    (&out_ref.to_dtype(DType::F32)? - &out_op.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()
}

#[test]
fn cuda_fused_verify_matches_per_row_short_and_long() -> Result<()> {
    for prefix_len in [452usize, 32_685usize] {
        let max_abs_diff = run_parity(prefix_len)?;
        eprintln!("[fused-verify-attn] prefix={prefix_len} max_abs_diff={max_abs_diff:.9e}");
        assert!(
            max_abs_diff <= 1e-3,
            "prefix={prefix_len}: max_abs_diff={max_abs_diff}"
        );
    }
    Ok(())
}

#[test]
fn cuda_onepass_verify_bitwise_matches_per_row() -> Result<()> {
    for prefix_len in [452usize, 32_685usize] {
        let max_abs_diff = run_onepass_parity(prefix_len)?;
        eprintln!("[onepass-verify-attn] prefix={prefix_len} max_abs_diff={max_abs_diff:.9e}");
        assert_eq!(
            max_abs_diff, 0.0,
            "prefix={prefix_len}: однопроходный путь обязан совпадать с построчным побитово, got {max_abs_diff}"
        );
    }
    Ok(())
}
