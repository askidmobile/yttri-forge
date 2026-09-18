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
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

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

        // QK на int8-тензорах: Q квантуется в int8 + построчные масштабы.
        // QK_INT8=1 — декод и MTP-проверка (остаётся opt-in: на шаге декода
        // выигрыша нет, а лишнее квантование Q стоит времени).
        //
        // Префил при int8-пуле — другое дело: он идёт тем же split-KV ядром,
        // K читается из int8-staging без распаковки, а Q квантуется ядром.
        // Замер 2026-09-18 на Qwen3.8-27B Q8_0 / RTX 4090 (f16-QK → int8-QK):
        // 8K 2066 → 2090, 32K 2156 → 2226, 131072 1617 → 1796,
        // 262144 1167 → 1388 t/s. На 128K/256K это выше llama.cpp на тех же
        // точках (1788/1356); greedy-выдача совпала побайтово (443 символа).
        // Поэтому для префила int8-QK — поведение по умолчанию, откат
        // QK_INT8_PREFILL=0 оставлен для A/B и отладки.
        let qk_int8_on = self.kv_scales.is_some()
            && if self.max_seqlen_q <= 8 {
                std::env::var("QK_INT8").as_deref() == Ok("1")
            } else {
                std::env::var("QK_INT8_PREFILL").as_deref() != Ok("0")
            };
        let (q8, qs) = if qk_int8_on {
            let (q8, qs) = quantize_q_int8_fast(self.q)?;
            (Some(q8), Some(qs))
        } else {
            (None, None)
        };
        let (q_int8_ptr, q_scale_ptr) = match (&q8, &qs) {
            (Some(a), Some(b)) => (
                tensor_cuda_ptr(a)? as *const std::ffi::c_void,
                tensor_cuda_ptr(b)? as *const std::ffi::c_void,
            ),
            _ => (std::ptr::null(), std::ptr::null()),
        };

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
                q_int8_ptr,
                q_scale_ptr,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }
}

/// Квантование Q в int8 для QK на int8-тензорных ядрах (спайк §91, план §92).
///
/// Масштаб считается на строку `(token, head)`: `s = max|q| / 127` (для пустой
/// строки — минимальный положительный, чтобы деление не давало inf), значения
/// округляются и зажимаются в `[-127, 127]`, после чего упаковываются в `U8`
/// как дополнительный код (`x < 0` → `x + 256`) — ровно так их читает ядро
/// через `reinterpret_cast<const int8_t *>`.
///
/// Возвращает `(q_int8, scale_f16)`; форма `q_int8` совпадает с входной,
/// `scale` сохраняет последнюю ось (годится для broadcast при проверках).
pub fn quantize_q_int8(q: &Tensor) -> Result<(Tensor, Tensor)> {
    let q32 = q.to_dtype(DType::F32)?;
    let amax = q32.abs()?.max_keepdim(candle_core::D::Minus1)?;
    // +1e-8: пустая строка (все нули) не должна давать деление на ноль.
    let scale = (amax / 127.0)?;
    let scale = (scale + 1e-8)?.to_dtype(DType::F16)?;
    let scaled = q32.broadcast_div(&scale.to_dtype(DType::F32)?)?;
    let rounded = scaled.round()?.clamp(-127.0, 127.0)?;
    // Дополнительный код: отрицательные -> x + 256, получаем 0..255 в U8.
    let as_u8 = rounded
        .lt(0.0)?
        .where_cond(&(rounded.clone() + 256.0)?, &rounded)?;
    Ok((as_u8.to_dtype(DType::U8)?, scale))
}

/// Кэш буферов Q-int8: форма между шагами не меняется, поэтому аллокации
/// делаются один раз и не попадают в захват CUDA-графа.
fn q8_cache() -> &'static Mutex<HashMap<(usize, usize, usize), (Tensor, Tensor)>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), (Tensor, Tensor)>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}


/// Быстрое квантование Q в int8 одним CUDA-ядром (`q_int8_quantize_rows`).
/// Форма входа [rows, h, d] F16; результат — (U8 [rows, h, d], F16 [rows, h])
/// в переиспользуемых буферах (см. `q8_cache`).
#[cfg(feature = "cuda")]
pub fn quantize_q_int8_fast(q: &Tensor) -> Result<(Tensor, Tensor)> {
    let (rows, h, d) = q.dims3()?;
    let dev = q.device().as_cuda_device()?;
    // Декодные размеры (граф) кэшируем: адрес буфера должен быть стабилен
    // между replay. Префильные тайлы (сотни+ строк) НЕ кэшируем: их форма
    // зависит от длины промпта, и кэш рос бы без границ (найдено 2026-09-18 —
    // +11 записей за пять промптов, вплоть до 33 МиБ каждая).
    let (q8, scales) = if rows <= 256 {
        let key = (rows, h, d);
        let mut cache = q8_cache().lock().expect("q8 cache");
        match cache.get(&key) {
            Some((a, b)) => (a.clone(), b.clone()),
            None => {
                let q8 = Tensor::zeros((rows, h, d), DType::U8, q.device())?;
                let scales = Tensor::zeros((rows, h), DType::F16, q.device())?;
                cache.insert(key, (q8.clone(), scales.clone()));
                (q8, scales)
            }
        }
    } else {
        (
            Tensor::zeros((rows, h, d), DType::U8, q.device())?,
            Tensor::zeros((rows, h), DType::F16, q.device())?,
        )
    };
    let q_ptr = tensor_cuda_ptr(q)?;
    let q8_ptr = tensor_cuda_ptr(&q8)?;
    let s_ptr = tensor_cuda_ptr(&scales)?;
    let d_i = d as i32;
    let func = dev.get_or_load_func("q_int8_quantize_rows", &candle_kernels::FLASH_DECODE)?;
    let cfg = LaunchConfig {
        grid_dim: ((rows * h) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    builder.arg(&q_ptr);
    builder.arg(&q8_ptr);
    builder.arg(&s_ptr);
    builder.arg(&d_i);
    unsafe { builder.launch(cfg) }.map_err(candle_core::Error::wrap)?;
    Ok((q8, scales))
}

#[cfg(test)]
mod q_int8_tests {
    use super::quantize_q_int8;
    use candle_core::{DType, Device, Result, Tensor};

    /// Эталонное квантование строки: s = max|q|/127, x8 = round(q/s).
    fn reference(vals: &[f32]) -> (f32, Vec<i32>) {
        let amax = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
        // Как в реализации: +1e-8 и приведение к F16 (иначе 63.5 округляется
        // по-разному: f32 даёт 63, f16-масштаб сдвигает границу).
        let s = half::f16::from_f32(amax / 127.0 + 1e-8).to_f32();
        let q8 = vals
            .iter()
            .map(|v| (v / s).round().clamp(-127.0, 127.0) as i32)
            .collect();
        (s, q8)
    }

    #[test]
    fn quantize_matches_reference_and_is_int8_encoded() -> Result<()> {
        let dev = Device::Cpu;
        let vals: Vec<f32> = vec![1.0, -2.0, 0.5, 0.0, 3.0, -0.25, 0.75, -1.5];
        let q = Tensor::from_vec(vals.clone(), (2, 4), &dev)?;
        let (q8, scale) = quantize_q_int8(&q)?;
        assert_eq!(q8.dtype(), DType::U8);
        assert_eq!(q8.dims(), &[2, 4]);
        assert_eq!(scale.dtype(), DType::F16);
        let bytes = q8.flatten_all()?.to_vec1::<u8>()?;
        let scales = scale.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        for (row, chunk) in vals.chunks(4).enumerate() {
            let (s_ref, q8_ref) = reference(chunk);
            assert!((scales[row] - s_ref as f32).abs() < 1e-3, "масштаб строки {row}");
            for (i, expect) in q8_ref.iter().enumerate() {
                let got = bytes[row * 4 + i] as i8 as i32;
                assert_eq!(got, *expect, "строка {row}, элемент {i}");
            }
        }
        Ok(())
    }

    #[test]
    fn empty_row_does_not_divide_by_zero() -> Result<()> {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 8), DType::F32, &dev)?;
        let (q8, _scale) = quantize_q_int8(&q)?;
        assert!(q8.flatten_all()?.to_vec1::<u8>()?.iter().all(|b| *b == 0));
        Ok(())
    }
}
