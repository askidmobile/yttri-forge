// candle_mmq_dp4a.cu — те же MMQ-ядра, но собранные на dp4a-пути (без
// тензорных ядер). Нужны для малого M: mma-вариант всегда считает 128
// столбцов и держит 96 n-блоков, из-за чего на 16 реальных столбцах форма
// [12288,4096] выдаёт 2.4–3.8 TFLOPS. llama.cpp для batches ≤ 64
// (MMQ_DP4A_MAX_BATCH_SIZE) берёт именно dp4a-ветку.
//
// Имена ядер отличаются префиксом, чтобы не конфликтовать с mma-инстансами:
// candle_mmq_dp4a_<tag>_x<mmq_x>. Выбор — на стороне Rust по m_total.

#define YTTRI_MMQ_DP4A 1

#include "mmq_common.cuh"
#include "mmq_gguf.cuh"

#define DEFINE_MMQ_DP4A(ggml_type_const, tag, MMQX) \
    extern "C" __global__ void __launch_bounds__(256) \
    candle_mmq_dp4a_##tag##_x##MMQX( \
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst, \
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, \
        const int ncols_y, const int stride_col_dst, const int ncols_max) { \
        mul_mat_q_impl<ggml_type_const, MMQX, false>( \
            x, y, nullptr, nullptr, dst, nullptr, \
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, stride_col_dst, \
            1, 1, 0, 0, 0, \
            1, 1, 0, 0, 0, \
            ncols_max); \
    }

DEFINE_MMQ_DP4A(GGML_TYPE_Q2_K, q2_k, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q2_K, q2_k, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q2_K, q2_k, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q3_K, q3_k, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q3_K, q3_k, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q3_K, q3_k, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_K, q4_k, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_K, q4_k, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_K, q4_k, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q5_K, q5_k, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q5_K, q5_k, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q5_K, q5_k, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q6_K, q6_k, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q6_K, q6_k, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q6_K, q6_k, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q8_0, q8_0, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q8_0, q8_0, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q8_0, q8_0, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_0, q4_0, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_0, q4_0, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_0, q4_0, 128)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_1, q4_1, 32)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_1, q4_1, 64)
DEFINE_MMQ_DP4A(GGML_TYPE_Q4_1, q4_1, 128)
