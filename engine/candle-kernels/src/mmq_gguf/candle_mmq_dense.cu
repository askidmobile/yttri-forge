// candle_mmq_dense.cu
// ВНИМАНИЕ: cudaforge кэширует nvcc-компиляцию по mtime/содержимому самого
// .cu — изменения в инклюдах (mmq_common.cuh/mmq_gguf.cuh) кэш НЕ сбрасывают,
// даже при сработавшем rerun-if-changed. Меняешь .cuh — бампни этот файл.
// cache-bust: 2026-08-18 real IQ grid tables in mmq_common.cuh
// extern "C" __global__ обёртки над llama.cpp mul_mat_q (Tensor-Core MMA) для
// плотного prefill. Вызываются из candle-core fast_mmq.rs через cudarc.
// Включён в PTX-сборку (имя не совпадает с exclude "mmq_*.cu").

#include "mmq_common.cuh"
#include "mmq_gguf.cuh"

// ---------------------------------------------------------------------------
// Quantize q8_1 (MMQ layout) — __device__ шаблон (копия из mmq_quantize.cu,
// чтобы не тянуть его хост-лаунчеры с <<<>>>).
// ---------------------------------------------------------------------------
template <mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void quantize_mmq_q8_1_impl(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1);
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4*QK8_1);

    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}

// ---------------------------------------------------------------------------
// Quantize q8_1 (MMQ layout) — DENSE обёртки (ids=null, ne2=1, s02=s03=0).
// grid=(ne1, ceil(ne0/512), 1), block=(128). ne00=k, s01=k, ne0=pad(k,512).
// ---------------------------------------------------------------------------
#define DEFINE_QUANT(layout_const, name) \
    extern "C" __global__ void candle_mmq_quant_##name( \
        const float * __restrict__ x, void * __restrict__ vy, \
        const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) { \
        quantize_mmq_q8_1_impl<layout_const>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1); \
    }

DEFINE_QUANT(MMQ_Q8_1_DS_LAYOUT_D4,   d4)
DEFINE_QUANT(MMQ_Q8_1_DS_LAYOUT_DS4,  ds4)
DEFINE_QUANT(MMQ_Q8_1_DS_LAYOUT_D2S6, d2s6)

// ---------------------------------------------------------------------------
// DENSE-обёртки mul_mat_q: ids_dst=null, expert_bounds=null, tmp_fixup=null,
// channel/sample параметры захардкожены (nchannels=nsamples=1, все stride=0).
// Упрощает Rust-сторону и избегает передачи null-указателей через cudarc.
// ---------------------------------------------------------------------------
#define DEFINE_MMQ_DENSE(ggml_type_const, tag, MMQX) \
    extern "C" __global__ void __launch_bounds__(256) \
    candle_mmq_##tag##_x##MMQX( \
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

// K-quants + базовые, mmq_x ∈ {32, 64, 128}
DEFINE_MMQ_DENSE(GGML_TYPE_Q2_K, q2_k, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q2_K, q2_k, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q2_K, q2_k, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_Q3_K, q3_k, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q3_K, q3_k, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q3_K, q3_k, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_Q4_K, q4_k, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q4_K, q4_k, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q4_K, q4_k, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_Q5_K, q5_k, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q5_K, q5_k, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q5_K, q5_k, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_Q6_K, q6_k, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q6_K, q6_k, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q6_K, q6_k, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_Q4_0, q4_0, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q4_0, q4_0, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q4_0, q4_0, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_Q8_0, q8_0, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_Q8_0, q8_0, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_Q8_0, q8_0, 128)

// IQ-кванты (27B/35B UD-модели): load_tiles_iq* деквантуют тайл в q8-формат
// в shared, дальше тот же MMA-путь. Tile: iq2_xs/iq2_s -> Q3_K (84),
// остальные -> Q8_0 (76); q8_1-layout активаций у всех IQ -> D4
// (mmq_get_q8_1_ds_layout).
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_XXS, iq2_xxs, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_XXS, iq2_xxs, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_XXS, iq2_xxs, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_XS, iq2_xs, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_XS, iq2_xs, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_XS, iq2_xs, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_S, iq2_s, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_S, iq2_s, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ2_S, iq2_s, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ3_XXS, iq3_xxs, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ3_XXS, iq3_xxs, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ3_XXS, iq3_xxs, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ3_S, iq3_s, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ3_S, iq3_s, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ3_S, iq3_s, 128)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ4_XS, iq4_xs, 32)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ4_XS, iq4_xs, 64)
DEFINE_MMQ_DENSE(GGML_TYPE_IQ4_XS, iq4_xs, 128)
