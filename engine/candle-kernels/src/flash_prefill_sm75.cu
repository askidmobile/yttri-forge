// Тайловый prefill-attention для Turing (sm_75).
//
// Проект: docs/design/2026-10-31-sm75-prefill-kernel.md.
// Фаза 1: служебное ядро flash_prefill_sm75_layout_probe — проверяет
// раскладки ldmatrix (x2 для A, x1 для K→B, x1.trans для V→B), свизл smem
// и одну инструкцию mma.m16n8k8 против CPU-эталона. Боевое ядро добавится
// Фазой 2.
//
// m16n8k16 на sm_75 нет; m16n8k8 — есть (PTX ISA: sm_75+, макс. k8 для f16).

#include <cuda_fp16.h>

// Свизл smem: атом 8 строк × 64 столбца, гранула 16 Б.
// Формула проверяется layout-пробником; боевое ядро использует её же.
__device__ __forceinline__ int yf_smem_idx(int r, int c) {
    return ((r >> 3) + ((c >> 6) << 3)) * 512
         + ((r & 7) << 6)
         + ((c & 63) ^ ((r & 7) << 3));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ unsigned yf_smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void yf_ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void yf_ldmatrix_x1(unsigned& r0, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
                 : "=r"(r0) : "r"(addr));
}

__device__ __forceinline__ void yf_ldmatrix_x1_trans(unsigned& r0, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                 : "=r"(r0) : "r"(addr));
}

__device__ __forceinline__ void yf_mma_m16n8k8(
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
#endif

// Входы — логические матрицы в row-major:
//   a: [16][8]  A[m][k]
//   k: [8][8]   K[n][k]   (K^T-режим: результируй = A × K^T)
//   v: [8][8]   V[j][d]   (P×V-режим: результат = A × V)
// Выходы float:
//   d_qk: [16][8]  Σ_k A[m][k]·K[n][k]
//   d_pv: [16][8]  Σ_j A[m][j]·V[j][d]
extern "C" __global__ void flash_prefill_sm75_layout_probe(
    const __half* __restrict__ a,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ d_qk,
    float* __restrict__ d_pv)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    // Буферы размечены под ту же формулу, что боевые 64×64-тайлы
    // (атом 8×64), поэтому индексы могут выходить за 16×8 — размер с запасом.
    __shared__ __align__(16) __half sA[16 * 64];
    __shared__ __align__(16) __half sK[8 * 64];
    __shared__ __align__(16) __half sV[8 * 64];

    const int tid = threadIdx.x;
    const int lane = tid & 31;

    for (int i = tid; i < 128; i += blockDim.x) {
        sA[yf_smem_idx(i / 8, i % 8)] = a[i];
    }
    for (int i = tid; i < 64; i += blockDim.x) {
        sK[yf_smem_idx(i / 8, i % 8)] = k[i];
        sV[yf_smem_idx(i / 8, i % 8)] = v[i];
    }
    __syncthreads();

    // A-фрагмент 16×8 (два регистра: строки g и g+8, столбцы 2t,2t+1).
    unsigned a0, a1;
    yf_ldmatrix_x2(a0, a1, yf_smem_u32(&sA[yf_smem_idx(lane % 16, 0)]));

    // K→B: строка smem — головной индекс n, ldmatrix.x1 без .trans.
    unsigned b0;
    yf_ldmatrix_x1(b0, yf_smem_u32(&sK[yf_smem_idx(lane % 8, 0)]));

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float qk0, qk1, qk2, qk3;
    yf_mma_m16n8k8(qk0, qk1, qk2, qk3, a0, a1, b0, 0.f, 0.f, 0.f, 0.f);

    const int g = lane >> 2;
    const int t = lane & 3;
    // C-фрагмент: (row=g, col=2t/2t+1) в c0/c1, (row=g+8) в c2/c3.
    d_qk[g * 8 + 2 * t] = qk0;
    d_qk[g * 8 + 2 * t + 1] = qk1;
    d_qk[(g + 8) * 8 + 2 * t] = qk2;
    d_qk[(g + 8) * 8 + 2 * t + 1] = qk3;

    // V→B: строка smem — токен j, ldmatrix.x1.trans даёт столбцовую
    // раскладку B[j][d] для P×V.
    unsigned bv;
    yf_ldmatrix_x1_trans(bv, yf_smem_u32(&sV[yf_smem_idx(lane % 8, 0)]));

    float pv0, pv1, pv2, pv3;
    yf_mma_m16n8k8(pv0, pv1, pv2, pv3, a0, a1, bv, c0, c1, c2, c3);

    d_pv[g * 8 + 2 * t] = pv0;
    d_pv[g * 8 + 2 * t + 1] = pv1;
    d_pv[(g + 8) * 8 + 2 * t] = pv2;
    d_pv[(g + 8) * 8 + 2 * t + 1] = pv3;
#endif
}

// ─── Боевое ядро (Фаза 2: фиксированная форма, без маски) ────────────────────

struct FlashPrefillSm75Params {
    unsigned int n_head;
    unsigned int n_kv;
    unsigned int hd;                 // 256 (контракт диспетчера)
    unsigned int t;                  // строк Q в чанке
    unsigned int b;                  // batch
    unsigned int page_size;          // 64
    unsigned int block_table_stride;
    unsigned int causal;             // 1 — каузальная маска, 0 — все ключи
    float scale;
};

#define YF_BR 64
#define YF_BC 64
#define YF_HD 256

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ unsigned yf_pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&h);
}

// Загрузка 64×256 F16 тайла из пула в свизленный smem с zero-fill
// невидимых токенов (alloc пула не инициализирован: 0·NaN отравил бы O).
__device__ __forceinline__ void yf_load_pool_tile(
    __half* __restrict__ dst,
    const __half* __restrict__ pool,
    unsigned int phys_page,
    unsigned int n_kv,
    unsigned int kvh,
    unsigned int kv0_base,      // первый логический токен страницы
    unsigned int k_end,
    int threadIdx_flat)
{
    const unsigned int r_in_group = threadIdx_flat >> 3;   // 0..15
    const unsigned int g = threadIdx_flat & 7;             // гранула внутри 64
    #pragma unroll
    for (int r0 = 0; r0 < 4; r0++) {
        const unsigned int row = 16 * (unsigned int)r0 + r_in_group;
        const unsigned int tok = kv0_base + row;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int col = 64u * (unsigned int)c + 8u * g;
            int4* d = reinterpret_cast<int4*>(&dst[yf_smem_idx((int)row, (int)col)]);
            if (tok < k_end) {
                const __half* src = pool
                    + (((size_t)phys_page * YF_BC + row) * n_kv + kvh) * YF_HD + col;
                *d = *reinterpret_cast<const int4*>(src);
            } else {
                *d = make_int4(0, 0, 0, 0);
            }
        }
    }
}

// Загрузка/выкладка Q-тайла 64×256 (row-major со свизлом).
__device__ __forceinline__ void yf_load_q_tile(
    __half* __restrict__ dst,
    const __half* __restrict__ q,
    unsigned int row_base,      // q0 строк в чанке
    unsigned int t,
    unsigned int n_head,
    unsigned int h,
    int threadIdx_flat)
{
    const unsigned int r_in_group = threadIdx_flat >> 3;
    const unsigned int g = threadIdx_flat & 7;
    #pragma unroll
    for (int r0 = 0; r0 < 4; r0++) {
        const unsigned int row = 16 * (unsigned int)r0 + r_in_group;
        const bool valid = (row_base + row) < t;
        const __half* src = q + ((size_t)(row_base + row) * n_head + h) * YF_HD;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int col = 64u * (unsigned int)c + 8u * g;
            int4* d = reinterpret_cast<int4*>(&dst[yf_smem_idx((int)row, (int)col)]);
            *d = valid ? *reinterpret_cast<const int4*>(src + col) : make_int4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void yf_store_q_tile(
    const __half* __restrict__ src_smem,
    __half* __restrict__ out,
    unsigned int row_base,
    unsigned int t,
    unsigned int n_head,
    unsigned int h,
    int threadIdx_flat)
{
    const unsigned int r_in_group = threadIdx_flat >> 3;
    const unsigned int g = threadIdx_flat & 7;
    #pragma unroll
    for (int r0 = 0; r0 < 4; r0++) {
        const unsigned int row = 16 * (unsigned int)r0 + r_in_group;
        if ((row_base + row) >= t) {
            continue;
        }
        __half* dst = out + ((size_t)(row_base + row) * n_head + h) * YF_HD;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int col = 64u * (unsigned int)c + 8u * g;
            *reinterpret_cast<int4*>(dst + col) =
                *reinterpret_cast<const int4*>(&src_smem[yf_smem_idx((int)row, (int)col)]);
        }
    }
}
#endif

// Тайловый paged prefill-attention: Br=64 (4 warp × 16 строк), Bc=64 = PAGE.
// grid = (ceil(T/64), n_head, b), block = 128, dynamic smem = 65536 Б.
// out допускает out == q (запись поверх Q после чтения тайла).
extern "C" __global__ void flash_prefill_sm75(
    const __half* q,
    const __half* __restrict__ k_pool,
    const __half* __restrict__ v_pool,
    const unsigned int* __restrict__ kv_len,
    const unsigned int* __restrict__ slots,
    const unsigned int* __restrict__ block_table,
    __half* out,
    float* lse,
    const FlashPrefillSm75Params params)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    extern __shared__ __align__(16) __half smem[];
    __half* sQ = smem;                    // 64×256 half = 32 КиБ
    __half* sKV = smem + YF_BR * YF_HD;   // 64×256 half = 32 КиБ

    const unsigned int h = blockIdx.y;
    const unsigned int b = blockIdx.z;
    const unsigned int q0 = blockIdx.x * YF_BR;
    const unsigned int n_head = params.n_head;
    const unsigned int n_kv = params.n_kv;
    const unsigned int kvh = h / (n_head / n_kv);
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31;
    const unsigned int w = tid >> 5;
    const unsigned int g = lane >> 2;     // 0..7
    const unsigned int tg = lane & 3;     // 0..3

    const unsigned int n_rows = min(YF_BR, params.t - q0);
    const unsigned int kv0 = kv_len[slots[b]];
    const unsigned int *bt = block_table + (size_t)b * params.block_table_stride;
    unsigned int k_end = kv0 + q0 + n_rows;
    const unsigned int pool_cap = params.block_table_stride * params.page_size;
    if (k_end > pool_cap) {
        k_end = pool_cap;
    }
    const unsigned int n_pages = (k_end + YF_BC - 1) / YF_BC;

    // --- Q-тайл ---
    yf_load_q_tile(sQ, q + (size_t)b * params.t * n_head * YF_HD, q0, params.t, n_head, h,
                   (int)tid);
    __syncthreads();

    float O[32][4];
    #pragma unroll
    for (int dc = 0; dc < 32; dc++) {
        O[dc][0] = 0.f; O[dc][1] = 0.f; O[dc][2] = 0.f; O[dc][3] = 0.f;
    }
    float m_x[2] = {-INFINITY, -INFINITY};
    float m_ref[2] = {-INFINITY, -INFINITY};
    float l_x[2] = {0.f, 0.f};

    for (unsigned int p = 0; p < n_pages; p++) {
        const unsigned int page = bt[p];
        yf_load_pool_tile(sKV, k_pool, page, n_kv, kvh, p * YF_BC, k_end, (int)tid);
        __syncthreads();

        // --- QK^T: S[nc][4] ---
        float S[8][4];
        #pragma unroll
        for (int nc = 0; nc < 8; nc++) {
            S[nc][0] = 0.f; S[nc][1] = 0.f; S[nc][2] = 0.f; S[nc][3] = 0.f;
        }
        #pragma unroll
        for (int kc = 0; kc < 32; kc++) {
            unsigned a0, a1;
            yf_ldmatrix_x2(a0, a1,
                yf_smem_u32(&sQ[yf_smem_idx((int)(16 * w + (lane & 15)), 8 * kc)]));
            #pragma unroll
            for (int nc = 0; nc < 8; nc++) {
                unsigned b0;
                yf_ldmatrix_x1(b0,
                    yf_smem_u32(&sKV[yf_smem_idx((int)(8 * nc + (lane & 7)), 8 * kc)]));
                float d0, d1, d2, d3;
                yf_mma_m16n8k8(d0, d1, d2, d3, a0, a1, b0,
                               S[nc][0], S[nc][1], S[nc][2], S[nc][3]);
                S[nc][0] = d0; S[nc][1] = d1; S[nc][2] = d2; S[nc][3] = d3;
            }
        }

        // --- scale + маска; максимум строки по тайлу ---
        float m_tile[2] = {-INFINITY, -INFINITY};
        #pragma unroll
        for (int nc = 0; nc < 8; nc++) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int x = i >> 1;
                const unsigned int row = 16 * w + g + 8 * x;
                const unsigned int tok = p * YF_BC + 8 * nc + 2 * tg + (i & 1);
                const bool vis = (row < n_rows) &&
                    (params.causal == 0 || tok <= kv0 + q0 + row);
                S[nc][i] = vis ? S[nc][i] * params.scale : -INFINITY;
                m_tile[x] = fmaxf(m_tile[x], S[nc][i]);
            }
        }
        #pragma unroll
        for (int x = 0; x < 2; x++) {
            m_tile[x] = fmaxf(m_tile[x], __shfl_xor_sync(0xffffffffu, m_tile[x], 1));
            m_tile[x] = fmaxf(m_tile[x], __shfl_xor_sync(0xffffffffu, m_tile[x], 2));
        }

        // --- online softmax: норма на максимум тайла (см. дизайн §4.5) ---
        float corr[2];
        #pragma unroll
        for (int x = 0; x < 2; x++) {
            m_ref[x] = fmaxf(m_ref[x], m_tile[x]);
            const float m_new = (m_tile[x] > -INFINITY)
                ? fmaxf(m_tile[x], m_ref[x] - 40.f)
                : m_x[x];
            corr[x] = (m_x[x] > -INFINITY) ? __expf(m_x[x] - m_new) : 0.f;
            l_x[x] *= corr[x];
            m_x[x] = m_new;
        }
        #pragma unroll
        for (int dc = 0; dc < 32; dc++) {
            O[dc][0] *= corr[0]; O[dc][1] *= corr[0];
            O[dc][2] *= corr[1]; O[dc][3] *= corr[1];
        }

        // Фаза D: l должна суммироваться из ТОГО ЖЕ значения P, которое
        // получит mma (F16-округление), иначе O и l расходятся на ~5e-4 на
        // тайл. Округляем p -> half -> float, и эти же half уходят в P.
        unsigned P0[8], P1[8];
        float sum[2] = {0.f, 0.f};
        #pragma unroll
        for (int nc = 0; nc < 8; nc++) {
            float p[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                p[i] = (S[nc][i] > -INFINITY) ? __expf(S[nc][i] - m_x[i >> 1]) : 0.f;
            }
            P0[nc] = yf_pack_half2(p[0], p[1]);
            P1[nc] = yf_pack_half2(p[2], p[3]);
            {
                __half2 h2 = *reinterpret_cast<__half2*>(&P0[nc]);
                float2 f2 = __half22float2(h2);
                sum[0] += f2.x + f2.y;
            }
            {
                __half2 h2 = *reinterpret_cast<__half2*>(&P1[nc]);
                float2 f2 = __half22float2(h2);
                sum[1] += f2.x + f2.y;
            }
        }
        #pragma unroll
        for (int x = 0; x < 2; x++) {
            sum[x] += __shfl_xor_sync(0xffffffffu, sum[x], 1);
            sum[x] += __shfl_xor_sync(0xffffffffu, sum[x], 2);
            l_x[x] += sum[x];
        }

        // --- V этой же страницы ---
        __syncthreads();
        yf_load_pool_tile(sKV, v_pool, page, n_kv, kvh, p * YF_BC, k_end, (int)tid);
        __syncthreads();

        // --- P·V ---
        #pragma unroll
        for (int nc = 0; nc < 8; nc++) {
            #pragma unroll
            for (int dc = 0; dc < 32; dc++) {
                unsigned bv;
                yf_ldmatrix_x1_trans(bv,
                    yf_smem_u32(&sKV[yf_smem_idx((int)(8 * nc + (lane & 7)), 8 * dc)]));
                float d0, d1, d2, d3;
                yf_mma_m16n8k8(d0, d1, d2, d3, P0[nc], P1[nc], bv,
                               O[dc][0], O[dc][1], O[dc][2], O[dc][3]);
                O[dc][0] = d0; O[dc][1] = d1; O[dc][2] = d2; O[dc][3] = d3;
            }
        }
        __syncthreads();
    }

    // --- эпилог: O/l → F16 в sQ, затем в out ---
    #pragma unroll
    for (int dc = 0; dc < 32; dc++) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int x = i >> 1;
            const unsigned int row = 16 * w + g + 8 * x;
            const unsigned int d = 8 * dc + 2 * tg + (i & 1);
            if (row < n_rows) {
                const float inv = (l_x[x] > 0.f) ? (1.f / l_x[x]) : 0.f;
                sQ[yf_smem_idx((int)row, (int)d)] = __float2half_rn(O[dc][i] * inv);
            }
        }
    }
    __syncthreads();
    yf_store_q_tile(sQ, out + (size_t)b * params.t * n_head * YF_HD, q0, params.t, n_head,
                    h, (int)tid);

    if (lse != nullptr) {
        #pragma unroll
        for (int x = 0; x < 2; x++) {
            const unsigned int row = 16 * w + g + 8 * x;
            if (row < n_rows && tg == 0) {
                lse[h * (params.b * params.t) + b * params.t + q0 + row] =
                    (l_x[x] > 0.f) ? (m_x[x] + logf(l_x[x])) : -INFINITY;
            }
        }
    }
#endif
}

// ─── Диагностика (PREFILL_NAN_TRACE=1) ───────────────────────────────────────
// Счётчик NaN/Inf в F16-тензоре: ищем, где именно ломается 128k-префил.
extern "C" __global__ void yf_count_nonfinite_f16(
    const __half* __restrict__ x,
    unsigned int n,
    unsigned int* __restrict__ out)
{
    unsigned int c = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = __half2float(x[i]);
        if (isnan(v) || isinf(v)) c++;
    }
    if (c) atomicAdd(out, c);
}

extern "C" __global__ void yf_count_nonfinite_f32(
    const float* __restrict__ x,
    unsigned int n,
    unsigned int* __restrict__ out)
{
    unsigned int c = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        const float v = x[i];
        if (isnan(v) || isinf(v)) c++;
    }
    if (c) atomicAdd(out, c);
}
