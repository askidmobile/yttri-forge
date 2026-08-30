//! Постраничное внимание своим вызовом, мимо op-механики candle.
//!
//! Ядру нужны указатели, а не типизированные тензоры. Автоград, `CustomOp3` и
//! дженерики по dtype здесь не дают ничего, зато последние прямо мешают: они
//! требуют, чтобы K/V были того же типа, что Q, и байтовый int8-пул пришлось бы
//! протаскивать обходом. Свой вызов снимает это ограничение, убирает слой
//! обёрток на каждом шаге декода и оставляет выходной буфер за нами — а он
//! должен быть постоянным, иначе CUDA-графы каждый replay ловят новый адрес.

use crate::real::paged_kv_cuda::tensor_cuda_ptr;
use candle_core::{DType, Result, Tensor};

/// Что нужно ядру для одного вызова постраничного внимания.
pub struct PagedAttn<'a> {
    /// Запросы [total_q, h, d] F16 (varlen: строки идут подряд).
    pub q: &'a Tensor,
    /// Пулы K/V: F16 [n_tokens, h_k, d] либо int8 (U8) той же формы.
    pub k_pool: &'a Tensor,
    pub v_pool: &'a Tensor,
    /// Масштабы int8-пула [n_tokens, h_k] F16. None — пул в F16.
    pub kv_scales: Option<(&'a Tensor, &'a Tensor)>,
    /// Кумулятивные длины: [b+1] i32/u32.
    pub seqlens_q: &'a Tensor,
    pub seqlens_k: &'a Tensor,
    /// Таблица блоков [b, max_blocks] u32.
    pub block_table: &'a Tensor,
    /// Выходной буфер [total_q, h, d] F16 — постоянный, переиспользуется.
    pub out: &'a Tensor,
    /// Буфер logsumexp [h, total_q] F32.
    pub softmax_lse: &'a Tensor,
    pub b: usize,
    pub h: usize,
    pub h_k: usize,
    pub d: usize,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub softmax_scale: f32,
    /// Окна как у FA2: отрицательное — без ограничения. Причинность это частный
    /// случай (левое < 0, правое = 0), и её надо задавать именно так: при
    /// q_len < k_len FA2 выравнивает маску по правому-нижнему углу, иначе
    /// токены чанка увидят будущее.
    pub window_left: i32,
    pub window_right: i32,
    pub page_block_size: usize,
    /// Все элементы batch читают одну строку block_table. Нужен слитой
    /// MTP-проверке: позиции одного слота имеют разные длины K, но общий KV-пул.
    pub shared_block_table: bool,
    /// Строк запроса на одну позицию (свёртка GQA, one-pass verify): строка r
    /// запроса [k*ngroups, h_k, d] — это (позиция r/ngroups, группа r%ngroups),
    /// и причинная граница маски считается по позиции. 0 — обычная раскладка.
    pub rows_per_position: usize,
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

impl PagedAttn<'_> {
    /// Проверки, которые дешевле сделать здесь, чем ловить мусор на выходе.
    fn validate(&self) -> Result<()> {
        if self.q.dtype() != DType::F16 || self.out.dtype() != DType::F16 {
            candle_core::bail!("paged_attn: q и out должны быть F16");
        }
        let pool_dtype = self.k_pool.dtype();
        if pool_dtype != self.v_pool.dtype() {
            candle_core::bail!("paged_attn: типы пулов K и V разошлись");
        }
        match (pool_dtype, self.kv_scales.is_some()) {
            (DType::F16, false) => {}
            (DType::U8, true) => {}
            (dt, has) => candle_core::bail!(
                "paged_attn: пул {dt:?} и масштабы={has} несовместимы \
                 (F16 без масштабов либо U8 с масштабами)"
            ),
        }
        if self.h % self.h_k != 0 {
            candle_core::bail!("paged_attn: h={} не кратно h_k={}", self.h, self.h_k);
        }
        if self.d % 8 != 0 {
            candle_core::bail!("paged_attn: head_dim={} должен быть кратен 8", self.d);
        }
        // Ядро с int8-KV инстанцируется только для hdim=256 (см. kQ8Allowed в
        // flash_fwd_launch_template.h): собирать q8-вариант для всех девяти
        // размерностей головы стоило часов компиляции. Проверяем здесь, потому что
        // C10_CUDA_CHECK в форке — пустышка, и тихий откат на F16 прочитал бы
        // int8-пул как f16, то есть выдал бы правдоподобный мусор вместо ошибки.
        if self.kv_scales.is_some() && self.d != 256 {
            candle_core::bail!(
                "paged_attn: int8-KV поддержан только при head_dim=256, получено {}",
                self.d
            );
        }
        let (q_rows, q_heads, q_dim) = self.q.dims3()?;
        if (q_heads, q_dim) != (self.h, self.d) {
            candle_core::bail!(
                "paged_attn: q shape {:?}, ожидалось [total_q, {}, {}]",
                self.q.dims(),
                self.h,
                self.d
            );
        }
        if self.out.dims3()? != (q_rows, self.h, self.d) {
            candle_core::bail!(
                "paged_attn: out shape {:?} не совпадает с q {:?}",
                self.out.dims(),
                self.q.dims()
            );
        }
        let (table_rows, _) = self.block_table.dims2()?;
        let required_rows = if self.shared_block_table { 1 } else { self.b };
        if table_rows < required_rows {
            candle_core::bail!(
                "paged_attn: block_table имеет {table_rows} строк, нужно {required_rows}"
            );
        }
        Ok(())
    }

    /// Один вызов ядра. Результат пишется в `out` на месте.
    pub fn forward(&self) -> Result<()> {
        self.validate()?;
        let (b, h, h_k, d) = (self.b, self.h, self.h_k, self.d);
        // Шаги пулов считаем из формы: [n_tokens, h_k, d] — они одинаковы в
        // элементах и для F16, и для int8, потому что раскладка та же.
        let kv_row_stride = (h_k * d) as u32;
        let kv_head_stride = d as u32;
        let q_row_stride = (h * d) as u32;
        let q_head_stride = d as u32;

        let (k_scale_ptr, v_scale_ptr, scale_row_stride, kv_is_q8) = match self.kv_scales {
            Some((ks, vs)) => (
                tensor_cuda_ptr(ks)? as *const std::ffi::c_void,
                tensor_cuda_ptr(vs)? as *const std::ffi::c_void,
                h_k as u32,
                1,
            ),
            None => (std::ptr::null(), std::ptr::null(), 0u32, 0),
        };
        // Шаг по блоку у масштабов — как у пула, но без измерения head_dim.
        let scale_batch_stride = (self.page_block_size * h_k) as u32;

        let dev = self.q.device().as_cuda_device()?;
        let stream = dev.cuda_stream();
        unsafe {
            candle_flash_attn::ffi::run_mha(
                tensor_cuda_ptr(self.q)? as *const std::ffi::c_void,
                tensor_cuda_ptr(self.k_pool)? as *const std::ffi::c_void,
                tensor_cuda_ptr(self.v_pool)? as *const std::ffi::c_void,
                tensor_cuda_ptr(self.out)? as *const std::ffi::c_void,
                tensor_cuda_ptr(self.softmax_lse)? as *const std::ffi::c_void,
                std::ptr::null(), // alibi
                tensor_cuda_ptr(self.seqlens_q)? as *const i32,
                tensor_cuda_ptr(self.seqlens_k)? as *const i32,
                // varlen: батчевых шагов нет, строки идут подряд
                0,
                (self.page_block_size * h_k * d) as u32,
                (self.page_block_size * h_k * d) as u32,
                0,
                0,
                q_row_stride,
                kv_row_stride,
                kv_row_stride,
                q_row_stride,
                q_head_stride,
                kv_head_stride,
                kv_head_stride,
                q_head_stride,
                b as u32,
                h as u32,
                h_k as u32,
                d as u32,
                round_multiple(d, 32) as u32,
                self.softmax_scale,
                self.max_seqlen_q as u32,
                self.max_seqlen_k as u32,
                round_multiple(self.max_seqlen_q, 128) as u32,
                round_multiple(self.max_seqlen_k, 128) as u32,
                self.q.dim(0)? as u32,
                0, // is_bf16
                if self.window_left < 0 && self.window_right == 0 {
                    1
                } else {
                    0
                },
                1, // unpadded_lse: varlen
                if self.window_left < 0 && self.window_right >= 0 {
                    self.max_seqlen_k as i32
                } else {
                    self.window_left
                },
                self.window_right,
                0.0, // softcap
                tensor_cuda_ptr(self.block_table)? as *const i32,
                if self.shared_block_table {
                    0
                } else {
                    self.block_table.dim(1)? as u32
                },
                self.page_block_size as i32,
                std::ptr::null(),
                0,
                0,
                k_scale_ptr,
                v_scale_ptr,
                scale_batch_stride,
                scale_row_stride,
                scale_batch_stride,
                scale_row_stride,
                kv_is_q8,
                self.rows_per_position as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }
}
