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
