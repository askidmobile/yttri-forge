//! Paged KV decode path (CUDA) + CUDA-graph capture plumbing.
//!
//! Все step-varying скаляры живут в device-буферах (kv_len, rope positions,
//! slots, block_table), обновляемых ДО cuGraphLaunch. Внутри графа только
//! device-side операции — это и делает decode-шаг захватываемым целиком.

use candle_core::{CudaDevice, DType, Device, Result, Tensor};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

pub const PAGE_SIZE: usize = 64;

/// Разделяемый контекст paged decode на уровне модели (общий для attention слоёв).
///
/// Буферы, читаемые candle-ops (FA2, RoPE gather), живут как persistent Tensor
/// (identity важна для replay: граф пишет/читает по захваченным адресам).
/// Буферы для кастомных ядер — raw CudaSlice (передаются по raw pointer).
pub struct PagedModelCtx {
    pub dev: CudaDevice,
    /// Per-slot заполненность KV (device truth), [capacity_b] u32.
    /// Инкремент — отдельным ядром в конце шага.
    pub kv_len_dev: CudaSlice<u32>,
    /// batch_idx → slot_idx: [capacity_b] u32.
    pub slots_dev: CudaSlice<u32>,
    /// Static seqlens_q: [0, 1, ..., capacity_b] i32 — persistent Tensor.
    pub seqlens_q_t: Tensor,
    /// Cumulative seqlens_k (kernel-written per step): [capacity_b + 1] i32.
    pub seqlens_k_t: Tensor,
    /// Block table: [capacity_b, max_blocks] u32 (bidx → slot pages).
    pub block_table_t: Tensor,
    /// RoPE positions: [capacity_b] u32.
    pub rope_pos_t: Tensor,
    pub capacity_b: usize,
    pub max_blocks: usize,
    /// Хостовые зеркала для fallback-логики (window eviction, seed).
    pub kv_len_host: Vec<u32>,
    pub slots_host: Vec<u32>,
}

fn tensor_cuda_ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("tensor is not CUDA"),
    };
    if !layout.is_contiguous() {
        candle_core::bail!("tensor is not contiguous");
    }
    macro_rules! ptr_of {
        ($ty:ty) => {{
            let slice = cuda.as_cuda_slice::<$ty>()?;
            let stream = slice.stream();
            let slice = slice.slice(layout.start_offset()..);
            let (ptr, _guard) = cudarc::driver::DevicePtr::device_ptr(&slice, stream);
            ptr
        }};
    }
    Ok(match t.dtype() {
        DType::U8 => ptr_of!(u8),
        DType::U32 => ptr_of!(u32),
        DType::I32 => ptr_of!(i32),
        DType::I64 => ptr_of!(i64),
        DType::F16 => ptr_of!(half::f16),
        DType::BF16 => ptr_of!(half::bf16),
        DType::F32 => ptr_of!(f32),
        DType::F64 => ptr_of!(f64),
        d => candle_core::bail!("tensor_cuda_ptr: unsupported dtype {:?}", d),
    })
}

impl PagedModelCtx {
    pub fn new(dev: &CudaDevice, capacity_b: usize, max_blocks: usize) -> Result<Self> {
        let device = Device::Cuda(dev.clone());
        let kv_len_dev = dev.alloc_zeros::<u32>(capacity_b)?;
        let slots_dev = dev.alloc_zeros::<u32>(capacity_b)?;
        let seqlens_q_host: Vec<u32> = (0..=capacity_b as u32).collect();
        let seqlens_q_t = Tensor::from_vec(seqlens_q_host, capacity_b + 1, &device)?;
        let seqlens_k_t = Tensor::zeros(capacity_b + 1, DType::U32, &device)?;
        let block_table_t = Tensor::zeros((capacity_b, max_blocks), DType::U32, &device)?;
        let rope_pos_t = Tensor::zeros(capacity_b, DType::U32, &device)?;
        Ok(Self {
            dev: dev.clone(),
            kv_len_dev,
            slots_dev,
            seqlens_q_t,
            seqlens_k_t,
            block_table_t,
            rope_pos_t,
            capacity_b,
            max_blocks,
            kv_len_host: vec![0; capacity_b],
            slots_host: vec![0; capacity_b],
        })
    }

    /// htod-обновления входов — строго ВНЕ графа (перед cuGraphLaunch).
    /// Использует Tensor::slice_set (D2D из staging tensor).
    pub fn stage_inputs(
        &mut self,
        slots: &[u32],
        rope_positions: &[usize],
        block_table: &[u32],
    ) -> Result<()> {
        let device = Device::Cuda(self.dev.clone());
        let b = slots.len();
        // slots_dev — raw buffer (htod напрямую)
        self.dev.memcpy_htod(slots, &mut self.slots_dev)?;
        // rope_pos_t
        let rope_u32: Vec<u32> = rope_positions.iter().map(|&p| p as u32).collect();
        let rope_staging = Tensor::from_vec(rope_u32.clone(), b, &Device::Cpu)?.to_device(&device)?;
        self.rope_pos_t.narrow(0, 0, b)?.slice_set(&rope_staging, 0, 0)?;
        // block_table_t
        let bt_staging = Tensor::from_vec(
            block_table.to_vec(),
            (b, self.max_blocks),
            &Device::Cpu,
        )?
        .to_device(&device)?;
        self.block_table_t
            .narrow(0, 0, b)?
            .slice_set(&bt_staging, 0, 0)?;
        self.slots_host = slots.to_vec();
        Ok(())
    }

    /// Сброс kv_len на device (после seed/restore) — вне графа.
    pub fn reset_kv_len(&mut self, lens: &[u32]) -> Result<()> {
        self.dev.memcpy_htod(lens, &mut self.kv_len_dev)?;
        self.kv_len_host = lens.to_vec();
        Ok(())
    }

    /// cumsum seqlens_k = kv_len + 1 (строка текущего шага уже включена).
    /// Один раз в начале forward (stream-ordered до attention слоёв).
    pub fn launch_cumsum(&self, b: usize) -> Result<()> {
        let func = self.dev.get_or_load_func(
            "cumsum_seqlens_from_kvlen",
            &candle_core::cuda_backend::kernels::QUANTIZED,
        )?;
        let seqlens_k_ptr = tensor_cuda_ptr(&self.seqlens_k_t)?;
        let b_i32 = b as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&self.kv_len_dev);
        builder.arg(&self.slots_dev);
        builder.arg(&seqlens_k_ptr);
        builder.arg(&b_i32);
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    /// Инкремент kv_len активных слотов — один раз в конце forward.
    pub fn launch_increment(&self, b: usize) -> Result<()> {
        let func = self.dev.get_or_load_func(
            "kv_len_increment",
            &candle_core::cuda_backend::kernels::QUANTIZED,
        )?;
        let b_i32 = b as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&self.kv_len_dev);
        builder.arg(&self.slots_dev);
        builder.arg(&b_i32);
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    /// Узкие view под текущий batch B для FA2.
pub fn launch_increment_t(&self, b: usize, t: usize) -> Result<()> {
        let func = self.dev.get_or_load_func(
            "kv_len_increment_t",
            &candle_core::cuda_backend::kernels::QUANTIZED,
        )?;
        let t_i32 = t as i32;
        let b_i32 = b as i32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&self.kv_len_dev);
        builder.arg(&self.slots_dev);
        builder.arg(&b_i32);
        builder.arg(&t_i32);
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    /// Узкие view под текущий batch B для FA2.
    pub fn seqlens_q(&self, b: usize) -> Result<Tensor> {
        self.seqlens_q_t.narrow(0, 0, b + 1)
    }
    pub fn seqlens_k(&self, b: usize) -> Result<Tensor> {
        self.seqlens_k_t.narrow(0, 0, b + 1)
    }
    pub fn block_table(&self, b: usize) -> Result<Tensor> {
        self.block_table_t.narrow(0, 0, b)
    }
    /// Prefill varlen: seqlens_k[i] = kv_len[slot]+t. Graph-capturable.
    pub fn seqlens_k_for_prefill(&self, b: usize, t: usize) -> Result<Tensor> {
        let func = self.dev.get_or_load_func(
            "cumsum_seqlens_from_kvlen_offset",
            &candle_core::cuda_backend::kernels::QUANTIZED,
        )?;
        let out = self.seqlens_k_t.narrow(0, 0, b + 1)?;
        let stream = self.dev.cuda_stream();
        let (kv_len_ptr, _g1) = cudarc::driver::DevicePtr::device_ptr(&self.kv_len_dev, &stream);
        let (slots_ptr, _g2) = cudarc::driver::DevicePtr::device_ptr(&self.slots_dev, &stream);
        let out_ptr = tensor_cuda_ptr(&out)?;
        let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 };
        let mut builder = func.builder();
        builder.arg(&kv_len_ptr);
        builder.arg(&slots_ptr);
        builder.arg(&out_ptr);
        builder.arg(&(b as i32));
        builder.arg(&(t as i32));
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        Ok(out)
    }

    pub fn rope_pos(&self, b: usize) -> Result<Tensor> {
        self.rope_pos_t.narrow(0, 0, b)
    }
}

/// Per-layer paged KV pool (k/v).
#[derive(Debug, Clone)]
pub struct PagedKvPool {
    pub k_pool: Tensor, // [num_blocks, page_size, n_kv, hd] F16
    pub v_pool: Tensor,
}

impl PagedKvPool {
    pub fn new(dev: &Device, num_blocks: usize, n_kv: usize, hd: usize) -> Result<Self> {
        let cuda_dev = dev
            .as_cuda_device()
            .map_err(|_| candle_core::Error::Msg("paged pool requires CUDA".into()))?;
        let shape = (num_blocks, PAGE_SIZE, n_kv, hd);
        let total_elems = num_blocks * PAGE_SIZE * n_kv * hd;
        let k_slice = unsafe { cuda_dev.alloc::<half::f16>(total_elems)? };
        let v_slice = unsafe { cuda_dev.alloc::<half::f16>(total_elems)? };
        let k_pool = Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                k_slice,
                cuda_dev.clone(),
            )),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        );
        let v_pool = Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                v_slice,
                cuda_dev.clone(),
            )),
            shape,
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Self { k_pool, v_pool })
    }

    /// Append текущих K/V строк [B, n_kv, hd] F16 в pool по device kv_len.
    pub fn launch_append(
        &self,
        ctx: &PagedModelCtx,
        k_rows: &Tensor,
        v_rows: &Tensor,
        b: usize,
        n_kv: usize,
        hd: usize,
        window: usize,
    ) -> Result<()> {
        let k_pool_ptr = tensor_cuda_ptr(&self.k_pool)?;
        let v_pool_ptr = tensor_cuda_ptr(&self.v_pool)?;
        let k_rows_ptr = tensor_cuda_ptr(k_rows)?;
        let v_rows_ptr = tensor_cuda_ptr(v_rows)?;
        let kv_len_ptr = ctx.dev.cuda_stream();
        let (kv_len_ptr, _g1) = cudarc::driver::DevicePtr::device_ptr(&ctx.kv_len_dev, &kv_len_ptr);
        let slots_ptr = ctx.dev.cuda_stream();
        let (slots_ptr, _g2) = cudarc::driver::DevicePtr::device_ptr(&ctx.slots_dev, &slots_ptr);
        let block_table_ptr = tensor_cuda_ptr(&ctx.block_table_t)?;
        let func = ctx.dev.get_or_load_func(
            "kv_append_paged_f16",
            &candle_core::cuda_backend::kernels::QUANTIZED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_kv as u32, b as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let b_i32 = b as i32;
        let n_kv_i32 = n_kv as i32;
        let hd_i32 = hd as i32;
        let page_i32 = PAGE_SIZE as i32;
        let max_blocks_i32 = ctx.max_blocks as i32;
        let window_i32 = window as i32;
        let mut builder = func.builder();
        builder.arg(&k_pool_ptr);
        builder.arg(&v_pool_ptr);
        builder.arg(&k_rows_ptr);
        builder.arg(&v_rows_ptr);
        builder.arg(&block_table_ptr);
        builder.arg(&slots_ptr);
        builder.arg(&kv_len_ptr);
        builder.arg(&b_i32);
        builder.arg(&n_kv_i32);
        builder.arg(&hd_i32);
        builder.arg(&page_i32);
        builder.arg(&max_blocks_i32);
        builder.arg(&window_i32);
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        Ok(())
    }

    /// Prefill: append T строк на позиции [kv_len .. kv_len+T). Инкремент kv_len
    /// отдельным kernel'ом (kv_len_increment_t).
    pub fn launch_append_multi(
        &self,
        ctx: &PagedModelCtx,
        k_rows: &Tensor,   // [B, T, n_kv, hd]
        v_rows: &Tensor,
        b: usize,
        t: usize,
        n_kv: usize,
        hd: usize,
        window: usize,
    ) -> Result<()> {
        let k_pool_ptr = tensor_cuda_ptr(&self.k_pool)?;
        let v_pool_ptr = tensor_cuda_ptr(&self.v_pool)?;
        let k_rows_ptr = tensor_cuda_ptr(k_rows)?;
        let v_rows_ptr = tensor_cuda_ptr(v_rows)?;
        let stream = ctx.dev.cuda_stream();
        let (kv_len_ptr, _g1) = cudarc::driver::DevicePtr::device_ptr(&ctx.kv_len_dev, &stream);
        let (slots_ptr, _g2) = cudarc::driver::DevicePtr::device_ptr(&ctx.slots_dev, &stream);
        let block_table_ptr = tensor_cuda_ptr(&ctx.block_table_t)?;
        let func = ctx.dev.get_or_load_func(
            "kv_append_paged_f16_multi",
            &candle_core::cuda_backend::kernels::QUANTIZED,
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n_kv as u32, b as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&k_pool_ptr);
        builder.arg(&v_pool_ptr);
        builder.arg(&k_rows_ptr);
        builder.arg(&v_rows_ptr);
        builder.arg(&block_table_ptr);
        builder.arg(&slots_ptr);
        builder.arg(&kv_len_ptr);
        builder.arg(&(b as i32));
        builder.arg(&(t as i32));
        builder.arg(&(n_kv as i32));
        builder.arg(&(hd as i32));
        builder.arg(&(PAGE_SIZE as i32));
        builder.arg(&(ctx.max_blocks as i32));
        builder.arg(&(window as i32));
        unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
