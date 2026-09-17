// add_rmsnorm_f32 — фьюжн остаточной суммы и RMSNorm в один запуск.
//
// Зачем: в декодном шаге на каждый слой приходится две пары
// «residual add → rmsnorm», то есть 64 лишних запуска на шаг из 745
// (см. docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md §56/§58).
//
// Схема: один блок на строку, block = 256 потоков, два прохода по строке.
// Первый проход пишет сумму в out_sum и копит сумму квадратов; второй —
// нормирует прочитанное из out_sum (попадает в L2). Без smem на строку,
// поэтому ограничения на cols нет.
//
// out_sum  = a + b                 (нужен как следующий residual)
// out_norm = (out_sum / rms) * w   (rms = sqrt(mean(s^2) + eps))

#include "cuda_utils.cuh"

extern "C" __global__ void add_rmsnorm_f32(
    const float * __restrict__ a,
    const float * __restrict__ b,
    const float * __restrict__ w,
    float * __restrict__ out_sum,
    float * __restrict__ out_norm,
    const int cols,
    const float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const size_t off = (size_t)row * (size_t)cols;
    const float * ar = a + off;
    const float * br = b + off;
    float * sr = out_sum + off;
    float * nr = out_norm + off;

    float acc = 0.0f;
    for (int i = tid; i < cols; i += nt) {
        const float v = ar[i] + br[i];
        sr[i] = v;
        acc = fmaf(v, v, acc);
    }

    // Блочная редукция суммы квадратов (warp shuffle + один шаг через smem).
    __shared__ float red[32];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        const int nw = (nt + 31) >> 5;
        float v = (lane < nw) ? red[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();

    const float mean = red[0] / (float)cols;
    const float scale = rsqrtf(mean + eps);
    for (int i = tid; i < cols; i += nt) {
        nr[i] = sr[i] * scale * w[i];
    }
}
