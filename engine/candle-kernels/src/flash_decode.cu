// Split-K flash-decoding для decode (seqlen_q=1, длинный KV).
// FA2 при q=1 держит на GPU только n_head блоков (24 для 27B) — latency-bound
// на длинном KV. Здесь: KV режется на S чанков, каждый блок считает частичный
// online-softmax, второй kernel объединяет. Grid A = (n_head, S).
//
// Layout: q [n_head, hd] F16; k/v [kv_len, n_kv, hd] F16 (head-last, как наш
// Q8 batched cache после dequant). GQA: kv_head = h / (n_head / n_kv).

#include <cuda_fp16.h>
#include "cuda_utils.cuh"

struct FlashDecodeParams {
    unsigned int n_head;    // 24
    unsigned int n_kv;      // 4
    unsigned int hd;        // 256
    unsigned int kv_len;
    unsigned int splits;    // S
    float scale;            // 1/sqrt(hd)
};

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Kernel A: частичный attention по чанку KV.
// grid=(n_head, S), block=128 (4 warps). КАЖДЫЙ warp — независимый сплиттер:
// свой непрерывный под-диапазон позиций, свои m/l в регистрах (lane-uniform
// через shfl, НИКАКИХ shared-скаляров — shared m/l между warps был гонкой и
// ломал генерацию после ~2K токенов). Каждый warp пишет свой partial:
// partials layout [n_head, S, NWARP, 2 + hd].
extern "C" __global__ void flash_decode_partial(
    const __half* __restrict__ q,      // [n_head * hd]
    const __half* __restrict__ k,      // [kv_len * n_kv * hd]
    const __half* __restrict__ v,      // [kv_len * n_kv * hd]
    float* __restrict__ partials,      // [n_head * S * 4 * (2 + hd)]
    const FlashDecodeParams params
) {
    const unsigned int h = blockIdx.x;
    const unsigned int s = blockIdx.y;
    const unsigned int hd = params.hd;
    const unsigned int n_kv = params.n_kv;
    const unsigned int kvh = h / (params.n_head / n_kv);
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int nwarp = blockDim.x / 32;

    // Диапазон чанка, далее под-диапазон warp'а (непрерывный).
    const unsigned int chunk = (params.kv_len + params.splits - 1) / params.splits;
    const unsigned int p0 = s * chunk;
    const unsigned int p1 = min(p0 + chunk, params.kv_len);
    const unsigned int wchunk = (p1 - p0 + nwarp - 1) / nwarp;
    const unsigned int w_start = p0 + warp * wchunk;
    const unsigned int w_end = min(w_start + wchunk, p1);

    // q головы в shared (f32) — только чтение, гонок нет.
    __shared__ float q_sh[256];
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        q_sh[d] = __half2float(q[h * hd + d]);
    }
    __syncthreads();

    // m/l — WARP-UNIFORM: одинаковые на всех lanes (shfl broadcast после reduce).
    // Раньше: lane-uniform → расхождение m/l → поломка softmax → мусор.
    float m_l = -INFINITY;
    float l_l = 0.0f;
    // acc: lane владеет 8 выходными dim своего warp'а (8×32=256).
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (unsigned int p = w_start; p < w_end; p++) {
        const __half* kp = k + ((size_t)p * n_kv + kvh) * hd;
        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int d = lane * 8 + i;
            if (d < hd) part += q_sh[d] * __half2float(kp[d]);
        }
        // warp reduce → все lanes знают dot (butterfly).
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
        const float dot = part * params.scale;

        // m/l — warp-uniform (все lanes имеют ОДИНАКОВЫЕ m_l/l_l
        // т.к. dot warp-uniform, а m_l/l_l инициализированы одинаково).
        const float m_new = fmaxf(m_l, dot);
        const float corr = __expf(m_l - m_new);
        const float p_exp = __expf(dot - m_new);
        l_l = l_l * corr + p_exp;
        m_l = m_new;

        const __half* vp = v + ((size_t)p * n_kv + kvh) * hd;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int d = lane * 8 + i;
            if (d < hd) acc[i] = acc[i] * corr + p_exp * __half2float(vp[d]);
        }
    }

    // Partial на (h, s, warp): [m, l, acc[hd]].
    float* out = partials +
        (((size_t)h * params.splits + s) * nwarp + warp) * (2 + hd);
    if (lane == 0) {
        out[0] = (l_l > 0.0f) ? m_l : -INFINITY;
        out[1] = l_l;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int d = lane * 8 + i;
        if (d < hd) out[2 + d] = acc[i];
    }
}

// Kernel B: combine partials → final. grid=(n_head), block=128.
// Читает S×4 partials (по одному на warp каждого сплита).
extern "C" __global__ void flash_decode_combine(
    const float* __restrict__ partials, // [n_head * S * 4 * (2 + hd)]
    __half* __restrict__ out,           // [n_head * hd]
    const FlashDecodeParams params
) {
    const unsigned int h = blockIdx.x;
    const unsigned int hd = params.hd;
    const unsigned int NS = params.splits * 4; // splits × warps
    const size_t stride = 2 + hd;
    const float* base = partials + (size_t)h * NS * stride;

    __shared__ float m_g, l_g;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < NS; s++) {
            m = fmaxf(m, base[s * stride]);
        }
        float l = 0.0f;
        for (unsigned int s = 0; s < NS; s++) {
            const float* p = base + s * stride;
            if (p[1] > 0.0f) l += p[1] * __expf(p[0] - m);
        }
        m_g = m;
        l_g = l;
    }
    __syncthreads();

    const float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < NS; s++) {
            const float* p = base + s * stride;
            if (p[1] > 0.0f) {
                acc += p[2 + d] * __expf(p[0] - m_g);
            }
        }
        out[h * hd + d] = __float2half(acc * inv_l);
    }
}
