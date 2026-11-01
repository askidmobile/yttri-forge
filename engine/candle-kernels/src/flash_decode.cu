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

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Q -> int8 с масштабом на строку (строка = (token, head)), одним ядром.
// amax -> scale = max(amax/127, 1e-8) в f16; q8 = clamp(round(v/scale), -127, 127).
// Раскладка входа [rows, d] F16, выхода [rows, d] int8, масштабов [rows] F16.
extern "C" __global__ void q_int8_quantize_rows(
    const __half* __restrict__ q,
    int8_t* __restrict__ q8,
    __half* __restrict__ scales,
    const int d)
{
    __shared__ float red[8];
    const int row = blockIdx.x;
    const __half* x = q + (size_t)row * d;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = blockDim.x >> 5;
    float amax = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(__half2float(x[i])));
    }
    amax = warp_max(amax);
    if (lane == 0) red[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nwarp) ? red[lane] : 0.0f;
        t = warp_max(t);
        if (lane == 0) red[0] = t;
    }
    __syncthreads();
    const __half sh = __float2half(fmaxf(red[0] / 127.0f, 1e-8f));
    const float s = __half2float(sh);
    if (tid == 0) scales[row] = sh;
    const float inv = 1.0f / s;
    int8_t* yo = q8 + (size_t)row * d;
    for (int i = tid; i < d; i += blockDim.x) {
        int r = __float2int_rn(__half2float(x[i]) * inv);
        r = max(-127, min(127, r));
        yo[i] = (int8_t)r;
    }
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

// Graph-safe split-K decode for the paged KV pool. All step-varying values stay
// in device buffers so the same launches can be captured into a CUDA graph.
struct PagedFlashDecodeParams {
    unsigned int n_head;
    unsigned int n_kv;
    unsigned int hd;
    unsigned int splits;
    float scale;
    unsigned int page_size;
    unsigned int block_table_stride;
};

extern "C" __global__ void flash_decode_paged_partial(
    const __half* __restrict__ q,                    // [B, n_head, hd]
    const __half* __restrict__ k_pool,               // [tokens, n_kv, hd]
    const __half* __restrict__ v_pool,               // [tokens, n_kv, hd]
    const unsigned int* __restrict__ seqlens_k,      // [B + 1]
    const unsigned int* __restrict__ block_table,    // [B, max_blocks]
    float* __restrict__ partials,
    const PagedFlashDecodeParams params
) {
    const unsigned int h = blockIdx.x;
    const unsigned int s = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int hd = params.hd;
    const unsigned int n_kv = params.n_kv;
    const unsigned int kv_len = seqlens_k[b + 1] - seqlens_k[b];
    const unsigned int kvh = h / (params.n_head / n_kv);
    const unsigned int lane = threadIdx.x % 32;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int nwarp = blockDim.x / 32;
    const unsigned int *bt = block_table + (size_t)b * params.block_table_stride;

    const unsigned int chunk = (kv_len + params.splits - 1) / params.splits;
    const unsigned int p0 = s * chunk;
    float *out = partials + (((size_t)b * params.n_head + h) * params.splits + s) * (2 + hd);
    if (p0 >= kv_len) {
        if (threadIdx.x == 0) {
            out[0] = -INFINITY;
            out[1] = 0.0f;
        }
        for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) out[2 + d] = 0.0f;
        return;
    }
    const unsigned int p1 = min(p0 + chunk, kv_len);
    const unsigned int wchunk = (p1 - p0 + nwarp - 1) / nwarp;
    const unsigned int w_start = p0 + warp * wchunk;
    const unsigned int w_end = min(w_start + wchunk, p1);

    __shared__ float q_sh[256];
    const __half *qh = q + ((size_t)b * params.n_head + h) * hd;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) q_sh[d] = __half2float(qh[d]);
    __syncthreads();

    float m_l = -INFINITY;
    float l_l = 0.0f;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (unsigned int p = w_start; p < w_end; p++) {
        const unsigned int page = bt[p / params.page_size];
        const unsigned int row = page * params.page_size + p % params.page_size;
        const __half *kp = k_pool + ((size_t)row * n_kv + kvh) * hd;
        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int d = lane * 8 + i;
            if (d < hd) part += q_sh[d] * __half2float(kp[d]);
        }
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
        const float dot = part * params.scale;

        const float m_new = fmaxf(m_l, dot);
        const float corr = __expf(m_l - m_new);
        const float p_exp = __expf(dot - m_new);
        l_l = l_l * corr + p_exp;
        m_l = m_new;

        const __half *vp = v_pool + ((size_t)row * n_kv + kvh) * hd;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int d = lane * 8 + i;
            if (d < hd) acc[i] = acc[i] * corr + p_exp * __half2float(vp[d]);
        }
    }

    __shared__ float warp_partial[4][2 + 256];
    if (lane == 0) {
        warp_partial[warp][0] = l_l > 0.0f ? m_l : -INFINITY;
        warp_partial[warp][1] = l_l;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int d = lane * 8 + i;
        if (d < hd) warp_partial[warp][2 + d] = acc[i];
    }
    __syncthreads();

    __shared__ float block_max, block_sum;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (unsigned int w = 0; w < nwarp; ++w) m = fmaxf(m, warp_partial[w][0]);
        float l = 0.0f;
        for (unsigned int w = 0; w < nwarp; ++w) {
            if (warp_partial[w][1] > 0.0f) {
                l += warp_partial[w][1] * __expf(warp_partial[w][0] - m);
            }
        }
        block_max = m;
        block_sum = l;
        out[0] = l > 0.0f ? m : -INFINITY;
        out[1] = l;
    }
    __syncthreads();
    // ВАЖНО: частичный выход — НЕнормированный числитель. Combine-ядро само
    // делит на глобальный l после взвешивания сплитов. Раньше здесь стояло
    // `value * inv_l` — двойная нормировка уменьшала выход attention в ~l_s
    // раз (найдено пробником prefill_sm75 parity 2026-10-31:
    // |dec|/|cpu| = 1/13…1/500).
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        float value = 0.0f;
        for (unsigned int w = 0; w < nwarp; ++w) {
            if (warp_partial[w][1] > 0.0f) {
                value += warp_partial[w][2 + d] * __expf(warp_partial[w][0] - block_max);
            }
        }
        out[2 + d] = value;
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
// Компактный свизл для одно-warp decode-MMA. В отличие от prefill-ядра здесь
// мало строк, поэтому номер 8x64-тайла не должен оставлять дыры по столбцам.
__device__ __forceinline__ int fd_mma_smem_idx(int r, int c, int col_groups) {
    return (((r >> 3) * col_groups + (c >> 6)) * 512)
         + ((r & 7) << 6)
         + ((c & 63) ^ ((r & 7) << 3));
}

__device__ __forceinline__ unsigned fd_mma_smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void fd_mma_ldmatrix_x2(
    unsigned& r0, unsigned& r1, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void fd_mma_ldmatrix_x1(
    unsigned& r0, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(r0) : "r"(addr));
}

__device__ __forceinline__ void fd_mma_ldmatrix_x1_trans(
    unsigned& r0, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                 : "=r"(r0) : "r"(addr));
}

__device__ __forceinline__ void fd_mma_m16n8k8(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned b0,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5},{%6},{%7,%8,%9,%10};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(b0),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void fd_load_query_heads(
    __half* __restrict__ dst,
    const __half* __restrict__ q,
    unsigned int b,
    unsigned int qh0,
    unsigned int n_head,
    int lane)
{
    constexpr int HD = 256;
    constexpr int COL_GROUPS = HD / 64;
    constexpr int VECS_PER_HEAD = HD / 8;
    for (int v = lane; v < 8 * VECS_PER_HEAD; v += 32) {
        const int h = v / VECS_PER_HEAD;
        const int c = (v % VECS_PER_HEAD) * 8;
        const __half* src = q + ((size_t)b * n_head + qh0 + h) * HD + c;
        int4* out = reinterpret_cast<int4*>(
            &dst[fd_mma_smem_idx(h, c, COL_GROUPS)]);
        *out = *reinterpret_cast<const int4*>(src);
    }
}

__device__ __forceinline__ void fd_load_pool_tile16(
    __half* __restrict__ dst,
    const __half* __restrict__ pool,
    const unsigned int* __restrict__ bt,
    unsigned int n_kv,
    unsigned int kvh,
    unsigned int base,
    unsigned int k_end,
    unsigned int page_size,
    int lane)
{
    constexpr int HD = 256;
    constexpr int COL_GROUPS = HD / 64;
    const int c = lane * 8;
    if (base >= k_end) {
        int4 zero = make_int4(0, 0, 0, 0);
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            *reinterpret_cast<int4*>(&dst[fd_mma_smem_idx(r, c, COL_GROUPS)]) = zero;
        }
        return;
    }
    // chunk выровнен по page_size, поэтому все 16 строк tile принадлежат одной
    // физической странице. Один lookup и развёрнутые чтения дают независимые
    // memory requests вместо цепочки load -> lookup -> load на каждую строку.
    const unsigned int page = bt[base / page_size];
    const unsigned int phys_base = page * page_size + base % page_size;
    const unsigned int rows = min(16u, k_end - base);
    #pragma unroll
    for (unsigned int r = 0; r < 16; ++r) {
        int4 val = make_int4(0, 0, 0, 0);
        if (r < rows) {
            const __half* src = pool
                + ((size_t)(phys_base + r) * n_kv + kvh) * HD + c;
            val = *reinterpret_cast<const int4*>(src);
        }
        *reinterpret_cast<int4*>(&dst[fd_mma_smem_idx((int)r, c, COL_GROUPS)]) = val;
    }
}
#endif

// Tensor-core partial для decode: один warp обслуживает все восемь Q-heads
// одной KV-head. QK^T считается как K[16×256] × Q^T[256×8], затем P^T[16×16]
// умножается на V[16×256]. Верхние восемь строк P содержат Q-heads, нижние
// восемь нулевые: это цена Turing m16n8k8 при восьми реальных головах.
extern "C" __global__ void flash_decode_paged_gqa_mma_partial(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const unsigned int* __restrict__ seqlens_k,
    const unsigned int* __restrict__ block_table,
    float* __restrict__ partials,
    const PagedFlashDecodeParams params)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    constexpr unsigned int Q_GROUPS = 8;
    constexpr unsigned int HD = 256;
    constexpr unsigned int COL_GROUPS = HD / 64;

    const unsigned int kvh = blockIdx.x;
    const unsigned int s = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int lane = threadIdx.x;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3;
    const unsigned int kv_len = seqlens_k[b + 1] - seqlens_k[b];
    const unsigned int *bt = block_table + (size_t)b * params.block_table_stride;
    const unsigned int chunk_raw = (kv_len + params.splits - 1) / params.splits;
    const unsigned int chunk = ((chunk_raw + params.page_size - 1) / params.page_size)
        * params.page_size;
    const unsigned int p0 = s * chunk;
    const unsigned int p1 = min(p0 + chunk, kv_len);

    if (p0 >= kv_len) {
        for (unsigned int i = lane; i < Q_GROUPS * (2 + HD); i += 32) {
            const unsigned int h = i / (2 + HD);
            const unsigned int d = i % (2 + HD);
            float *po = partials
                + (((size_t)b * params.n_head + kvh * Q_GROUPS + h)
                    * params.splits + s) * (2 + HD);
            po[d] = 0.0f;
        }
        __syncwarp();
        if (g == 0) {
            float *o0 = partials
                + (((size_t)b * params.n_head + kvh * Q_GROUPS + 2 * t)
                    * params.splits + s) * (2 + HD);
            float *o1 = o0 + (size_t)params.splits * (2 + HD);
            o0[0] = -INFINITY; o0[1] = 0.0f;
            o1[0] = -INFINITY; o1[1] = 0.0f;
        }
        return;
    }

    __shared__ __align__(16) __half sQ[8 * HD];
    __shared__ __align__(16) __half sKV[16 * HD];
    // Свизл хранит каждый 8x64-тайл целиком, даже если реально нужны 16 колонок.
    __shared__ __align__(16) __half sP[2 * 512];

    fd_load_query_heads(sQ, q, b, kvh * Q_GROUPS, params.n_head, (int)lane);
    for (unsigned int i = lane; i < 16 * 16; i += 32) {
        const unsigned int r = i / 16;
        const unsigned int c = i % 16;
        sP[fd_mma_smem_idx((int)r, (int)c, 1)] = __float2half(0.0f);
    }
    __syncwarp();

    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    float O[32][4];
    #pragma unroll
    for (int dc = 0; dc < 32; dc++) {
        O[dc][0] = 0.0f; O[dc][1] = 0.0f;
        O[dc][2] = 0.0f; O[dc][3] = 0.0f;
    }

    for (unsigned int base = p0; base < p1; base += 16) {
        fd_load_pool_tile16(
            sKV, k_pool, bt, params.n_kv, kvh, base, p1,
            params.page_size, (int)lane);
        __syncwarp();

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        #pragma unroll
        for (int kc = 0; kc < 32; kc++) {
            unsigned a0, a1, bq;
            fd_mma_ldmatrix_x2(
                a0, a1,
                fd_mma_smem_u32(&sKV[fd_mma_smem_idx(
                    (int)(lane & 15), 8 * kc, COL_GROUPS)]));
            fd_mma_ldmatrix_x1(
                bq,
                fd_mma_smem_u32(&sQ[fd_mma_smem_idx(
                    (int)(lane & 7), 8 * kc, COL_GROUPS)]));
            float d0, d1, d2, d3;
            fd_mma_m16n8k8(d0, d1, d2, d3, a0, a1, bq, s0, s1, s2, s3);
            s0 = d0; s1 = d1; s2 = d2; s3 = d3;
        }

        const bool row0_valid = base + g < p1;
        const bool row1_valid = base + g + 8 < p1;
        s0 = row0_valid ? s0 * params.scale : -INFINITY;
        s1 = row0_valid ? s1 * params.scale : -INFINITY;
        s2 = row1_valid ? s2 * params.scale : -INFINITY;
        s3 = row1_valid ? s3 * params.scale : -INFINITY;

        float tile_m0 = fmaxf(s0, s2);
        float tile_m1 = fmaxf(s1, s3);
        #pragma unroll
        for (int mask = 4; mask <= 16; mask <<= 1) {
            tile_m0 = fmaxf(tile_m0, __shfl_xor_sync(0xffffffffu, tile_m0, mask));
            tile_m1 = fmaxf(tile_m1, __shfl_xor_sync(0xffffffffu, tile_m1, mask));
        }
        const float mn0 = fmaxf(m0, tile_m0);
        const float mn1 = fmaxf(m1, tile_m1);
        const float corr0 = (m0 > -INFINITY) ? __expf(m0 - mn0) : 0.0f;
        const float corr1 = (m1 > -INFINITY) ? __expf(m1 - mn1) : 0.0f;
        m0 = mn0; m1 = mn1;

        float p0v = (s0 > -INFINITY && mn0 > -INFINITY) ? __expf(s0 - mn0) : 0.0f;
        float p1v = (s1 > -INFINITY && mn1 > -INFINITY) ? __expf(s1 - mn1) : 0.0f;
        float p2v = (s2 > -INFINITY && mn0 > -INFINITY) ? __expf(s2 - mn0) : 0.0f;
        float p3v = (s3 > -INFINITY && mn1 > -INFINITY) ? __expf(s3 - mn1) : 0.0f;
        sP[fd_mma_smem_idx((int)(2 * t), (int)g, 1)] = __float2half_rn(p0v);
        sP[fd_mma_smem_idx((int)(2 * t + 1), (int)g, 1)] = __float2half_rn(p1v);
        sP[fd_mma_smem_idx((int)(2 * t), (int)g + 8, 1)] = __float2half_rn(p2v);
        sP[fd_mma_smem_idx((int)(2 * t + 1), (int)g + 8, 1)] = __float2half_rn(p3v);

        float sum0 = __half2float(__float2half_rn(p0v))
                   + __half2float(__float2half_rn(p2v));
        float sum1 = __half2float(__float2half_rn(p1v))
                   + __half2float(__float2half_rn(p3v));
        #pragma unroll
        for (int mask = 4; mask <= 16; mask <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, mask);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, mask);
        }
        l0 = l0 * corr0 + sum0;
        l1 = l1 * corr1 + sum1;

        const float ce = __shfl_sync(0xffffffffu, corr0, (int)(g >> 1));
        const float co = __shfl_sync(0xffffffffu, corr1, (int)(g >> 1));
        const float corr_head = (g & 1) ? co : ce;
        #pragma unroll
        for (int dc = 0; dc < 32; dc++) {
            O[dc][0] *= corr_head; O[dc][1] *= corr_head;
            O[dc][2] *= corr_head; O[dc][3] *= corr_head;
        }
        __syncwarp();

        fd_load_pool_tile16(
            sKV, v_pool, bt, params.n_kv, kvh, base, p1,
            params.page_size, (int)lane);
        __syncwarp();

        #pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            unsigned pa0, pa1;
            fd_mma_ldmatrix_x2(
                pa0, pa1,
                fd_mma_smem_u32(&sP[fd_mma_smem_idx(
                    (int)(lane & 15), 8 * kc, 1)]));
            #pragma unroll
            for (int dc = 0; dc < 32; dc++) {
                unsigned bv;
                fd_mma_ldmatrix_x1_trans(
                    bv,
                    fd_mma_smem_u32(&sKV[fd_mma_smem_idx(
                        (int)(8 * kc + (lane & 7)), 8 * dc, COL_GROUPS)]));
                float d0, d1, d2, d3;
                fd_mma_m16n8k8(
                    d0, d1, d2, d3, pa0, pa1, bv,
                    O[dc][0], O[dc][1], O[dc][2], O[dc][3]);
                O[dc][0] = d0; O[dc][1] = d1;
                O[dc][2] = d2; O[dc][3] = d3;
            }
        }
        __syncwarp();
    }

    if (g == 0) {
        float *o0 = partials
            + (((size_t)b * params.n_head + kvh * Q_GROUPS + 2 * t)
                * params.splits + s) * (2 + HD);
        float *o1 = o0 + (size_t)params.splits * (2 + HD);
        o0[0] = l0 > 0.0f ? m0 : -INFINITY; o0[1] = l0;
        o1[0] = l1 > 0.0f ? m1 : -INFINITY; o1[1] = l1;
    }
    float *out = partials
        + (((size_t)b * params.n_head + kvh * Q_GROUPS + g)
            * params.splits + s) * (2 + HD);
    #pragma unroll
    for (int dc = 0; dc < 32; dc++) {
        out[2 + 8 * dc + 2 * t] = O[dc][0];
        out[2 + 8 * dc + 2 * t + 1] = O[dc][1];
    }
#endif
}

// GQA-свёрнутая варианта partial-ядра: один CTA обслуживает все Q-головы
// одной KV-головы. K/V tile загружается в shared один раз и переиспользуется
// восьмью warp'ами; partials сохраняют прежний layout [B, n_head, splits, 2+hd],
// поэтому существующий combine не меняется.
extern "C" __global__ void flash_decode_paged_gqa_partial(
    const __half* __restrict__ q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const unsigned int* __restrict__ seqlens_k,
    const unsigned int* __restrict__ block_table,
    float* __restrict__ partials,
    const PagedFlashDecodeParams params
) {
    constexpr unsigned int TILE = 32;
    const unsigned int kvh = blockIdx.x;
    const unsigned int s = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int hd = params.hd;
    const unsigned int n_kv = params.n_kv;
    const unsigned int ngroups = params.n_head / n_kv;
    const unsigned int kv_len = seqlens_k[b + 1] - seqlens_k[b];
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int warp = tid >> 5;
    const unsigned int *bt = block_table + (size_t)b * params.block_table_stride;

    const unsigned int chunk = (kv_len + params.splits - 1) / params.splits;
    const unsigned int p0 = s * chunk;
    const unsigned int qh = kvh * ngroups + warp;
    float *out = partials +
        (((size_t)b * params.n_head + qh) * params.splits + s) * (2 + hd);
    if (p0 >= kv_len) {
        if (warp == 0 && lane < ngroups) {
            const unsigned int ph = kvh * ngroups + lane;
            float *po = partials +
                (((size_t)b * params.n_head + ph) * params.splits + s) * (2 + hd);
            po[0] = -INFINITY;
            po[1] = 0.0f;
        }
        for (unsigned int i = tid; i < ngroups * hd; i += blockDim.x) {
            const unsigned int g = i / hd;
            const unsigned int d = i % hd;
            const unsigned int ph = kvh * ngroups + g;
            float *po = partials +
                (((size_t)b * params.n_head + ph) * params.splits + s) * (2 + hd);
            po[2 + d] = 0.0f;
        }
        return;
    }

    const unsigned int p1 = min(p0 + chunk, kv_len);
    const unsigned int d0 = lane * 8;
    const __half *qhead = q + ((size_t)b * params.n_head + qh) * hd;
    float qreg[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) qreg[i] = __half2float(qhead[d0 + i]);

    float m_l = -INFINITY;
    float l_l = 0.0f;
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    __shared__ __align__(16) __half kv_sh[TILE * 256];

    const unsigned int vecs_per_row = hd / 8;
    const unsigned int tile_vecs = TILE * vecs_per_row;
    for (unsigned int base = p0; base < p1; base += TILE) {
        const unsigned int rows = min(TILE, p1 - base);
        // K tile: int4 = 8 half.
        for (unsigned int v = tid; v < tile_vecs; v += blockDim.x) {
            const unsigned int r = v / vecs_per_row;
            const unsigned int c = (v % vecs_per_row) * 8;
            int4 val = make_int4(0, 0, 0, 0);
            if (r < rows) {
                const unsigned int p = base + r;
                const unsigned int page = bt[p / params.page_size];
                const unsigned int row = page * params.page_size + p % params.page_size;
                const __half *src = k_pool + ((size_t)row * n_kv + kvh) * hd + c;
                val = *reinterpret_cast<const int4*>(src);
            }
            *reinterpret_cast<int4*>(&kv_sh[r * 256 + c]) = val;
        }
        __syncthreads();

        float pexp[TILE];
        float corr[TILE];
        for (unsigned int r = 0; r < rows; ++r) {
            const __half *kp = &kv_sh[r * 256 + d0];
            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) part += qreg[i] * __half2float(kp[i]);
            for (int o = 16; o > 0; o >>= 1)
                part += __shfl_xor_sync(0xffffffff, part, o);
            const float dot = part * params.scale;
            const float m_new = fmaxf(m_l, dot);
            const float cr = __expf(m_l - m_new);
            const float pe = __expf(dot - m_new);
            l_l = l_l * cr + pe;
            m_l = m_new;
            pexp[r] = pe;
            corr[r] = cr;
        }
        __syncthreads();

        // V tile переиспользует тот же shared.
        for (unsigned int v = tid; v < tile_vecs; v += blockDim.x) {
            const unsigned int r = v / vecs_per_row;
            const unsigned int c = (v % vecs_per_row) * 8;
            int4 val = make_int4(0, 0, 0, 0);
            if (r < rows) {
                const unsigned int p = base + r;
                const unsigned int page = bt[p / params.page_size];
                const unsigned int row = page * params.page_size + p % params.page_size;
                const __half *src = v_pool + ((size_t)row * n_kv + kvh) * hd + c;
                val = *reinterpret_cast<const int4*>(src);
            }
            *reinterpret_cast<int4*>(&kv_sh[r * 256 + c]) = val;
        }
        __syncthreads();
        for (unsigned int r = 0; r < rows; ++r) {
            const __half *vp = &kv_sh[r * 256 + d0];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[i] = acc[i] * corr[r] + pexp[r] * __half2float(vp[i]);
        }
        __syncthreads();
    }

    // Каждый warp считает отдельную Q-head. Сохраняем его partial напрямую:
    // объединение warp'ов здесь смешало бы восемь независимых softmax.
    if (lane == 0) {
        out[0] = l_l > 0.0f ? m_l : -INFINITY;
        out[1] = l_l;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[2 + d0 + i] = acc[i];
}

extern "C" __global__ void flash_decode_paged_combine(
    const float* __restrict__ partials,
    __half* __restrict__ out,
    const PagedFlashDecodeParams params
) {
    const unsigned int h = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int hd = params.hd;
    const unsigned int NS = params.splits;
    const size_t stride = 2 + hd;
    const float *base = partials + ((size_t)b * params.n_head + h) * NS * stride;

    __shared__ float m_g, l_g;
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (unsigned int s = 0; s < NS; s++) m = fmaxf(m, base[s * stride]);
        float l = 0.0f;
        for (unsigned int s = 0; s < NS; s++) {
            const float *p = base + s * stride;
            if (p[1] > 0.0f) l += p[1] * __expf(p[0] - m);
        }
        m_g = m;
        l_g = l;
    }
    __syncthreads();

    const float inv_l = l_g > 0.0f ? 1.0f / l_g : 0.0f;
    __half *dst = out + ((size_t)b * params.n_head + h) * hd;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < NS; s++) {
            const float *p = base + s * stride;
            if (p[1] > 0.0f) acc += p[2 + d] * __expf(p[0] - m_g);
        }
        dst[d] = __float2half(acc * inv_l);
    }
}

// --- Fused attention-prep для paged-декода (YTTRI_ATTN_PREP_FUSED=1) -------
// Один запуск вместо ~15 мелких: q/k RMSNorm, partial RoPE (первые rope_dim из
// hd), раскладка head-last для FA2, извлечение gate-половины q-проекции и каст
// v в F16. Математика идентична цепочке q_norm -> apply_partial_rotary_emb_devpos
// -> to_dtype(F16) из model_weights.rs, отличие только в порядке суммирования.
//
// qg: [B, n_head, 2*hd] F32 (первая половина — q, вторая — gate)
// k, v: [B, n_kv, hd] F32; qw/kw: [hd] F32; cos/sin: [max_pos, rope_half] F32
// pos: [B] u32; выходы q_out [B,n_head,hd] F16, gate_out [B,n_head,hd] F32,
// k_out/v_out [B,n_kv,hd] F16.
// grid = (n_head + 2*n_kv, B), block = 128.
extern "C" __global__ void attn_prepare_decode(
    const float* __restrict__ qg,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ qw,
    const float* __restrict__ kw,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    const unsigned int* __restrict__ pos,
    __half* __restrict__ q_out,
    float* __restrict__ gate_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int n_head,
    const int n_kv,
    const int hd,
    const int rope_dim,
    const int rope_half,
    const float eps)
{
    __shared__ float red[8];
    const int b = blockIdx.y;
    const int bx = blockIdx.x;
    const int tid = threadIdx.x;
    const int nwarp = blockDim.x >> 5;
    const unsigned int p = pos[b];
    const float* cs = cos_t + (size_t)p * rope_half;
    const float* sn = sin_t + (size_t)p * rope_half;

    if (bx < n_head) {
        // ---- Q: RMSNorm + partial RoPE + каст, параллельно выгружаем gate ----
        const int h = bx;
        const float* x = qg + ((size_t)b * n_head + h) * (2 * hd);
        float ssq = 0.0f;
        for (int d = tid; d < hd; d += blockDim.x) {
            const float t = x[d];
            ssq = fmaf(t, t, ssq);
        }
        ssq = warp_sum(ssq);
        const int warp = tid >> 5;
        const int lane = tid & 31;
        if (lane == 0) red[warp] = ssq;
        __syncthreads();
        if (warp == 0) {
            float t = (lane < nwarp) ? red[lane] : 0.0f;
            t = warp_sum(t);
            if (lane == 0) red[0] = t;
        }
        __syncthreads();
        const float scale = rsqrtf(red[0] / (float)hd + eps);
        __half* yo = q_out + ((size_t)b * n_head + h) * hd;
        float* go = gate_out + ((size_t)b * n_head + h) * hd;
        for (int d = tid; d < hd; d += blockDim.x) {
            float val = x[d] * scale * qw[d];
            if (d < rope_dim) {
                if (d < rope_half) {
                    const float x2 = x[d + rope_half] * scale * qw[d + rope_half];
                    val = fmaf(-x2, sn[d], val * cs[d]);
                } else {
                    const int e = d - rope_half;
                    const float x1 = x[e] * scale * qw[e];
                    val = fmaf(x1, sn[e], val * cs[e]);
                }
            }
            yo[d] = __float2half(val);
            go[d] = x[hd + d];
        }
    } else if (bx < n_head + n_kv) {
        // ---- K: RMSNorm + partial RoPE + каст (та же схема, свой вес) ----
        const int h = bx - n_head;
        const float* x = k + ((size_t)b * n_kv + h) * hd;
        float ssq = 0.0f;
        for (int d = tid; d < hd; d += blockDim.x) {
            const float t = x[d];
            ssq = fmaf(t, t, ssq);
        }
        ssq = warp_sum(ssq);
        const int warp = tid >> 5;
        const int lane = tid & 31;
        if (lane == 0) red[warp] = ssq;
        __syncthreads();
        if (warp == 0) {
            float t = (lane < nwarp) ? red[lane] : 0.0f;
            t = warp_sum(t);
            if (lane == 0) red[0] = t;
        }
        __syncthreads();
        const float scale = rsqrtf(red[0] / (float)hd + eps);
        __half* yo = k_out + ((size_t)b * n_kv + h) * hd;
        for (int d = tid; d < hd; d += blockDim.x) {
            float val = x[d] * scale * kw[d];
            if (d < rope_dim) {
                if (d < rope_half) {
                    const float x2 = x[d + rope_half] * scale * kw[d + rope_half];
                    val = fmaf(-x2, sn[d], val * cs[d]);
                } else {
                    const int e = d - rope_half;
                    const float x1 = x[e] * scale * kw[e];
                    val = fmaf(x1, sn[e], val * cs[e]);
                }
            }
            yo[d] = __float2half(val);
        }
    } else {
        // ---- V: только каст в F16 ----
        const int h = bx - n_head - n_kv;
        const float* x = v + ((size_t)b * n_kv + h) * hd;
        __half* yo = v_out + ((size_t)b * n_kv + h) * hd;
        for (int d = tid; d < hd; d += blockDim.x) {
            yo[d] = __float2half(x[d]);
        }
    }
}
