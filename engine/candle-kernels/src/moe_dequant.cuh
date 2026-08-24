// Warp-level dequant functions for IQ types — shared between quantized.cu and moe_quantized.cu.
// Each function dequantizes one block (QK_K=256 elements) using 256 threads, writing to dst_t* yy.
//
// Block structs and lookup tables are defined in quantized.cu and shared via this header
// by including quantized.cu's declarations. Since quantized.cu is a single translation unit,
// moe_quantized.cu must NOT re-define the structs/tables — it includes this header which
// forward-declares them as extern.
//
// IMPORTANT: these functions are __forceinline__ __device__ and ported directly from
// quantized.cu:1764-1938. The math is identical — any divergence breaks llama.cpp parity.

#pragma once

#include "cuda_fp16.h"
#include <stdint.h>

#ifndef QK_K
#define QK_K 256
#endif

// ─── Block struct declarations (defined in quantized.cu, shared here) ─────────
// These are duplicated from quantized.cu. If quantized.cu and moe_quantized.cu are
// in the same PTX module, they share __device__ tables. If separate modules,
// moe_quantized.cu must define its own copies (see below).

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    half2 dm;
} moe_block_q2_K;

typedef struct {
    half2 dm;
    uint8_t scales[3*QK_K/64];
    uint8_t qs[QK_K/2];
} moe_block_q4_K;

typedef struct {
    half d;
    uint8_t qs[3*QK_K/8];
} moe_block_iq3_xxs;

typedef struct {
    half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} moe_block_iq2_s;

typedef struct {
    half d;
    uint16_t qs[QK_K/8];
} moe_block_iq2_xxs;

// ─── Lookup tables (must match quantized.cu exactly) ──────────────────────────

static const __device__ uint8_t moe_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static const __device__ uint8_t moe_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const __device__ float moe_kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

// IQ2_XXS grid (256 entries) — included from quantized.cu's table via separate compilation.
// For PTX module isolation, we define our own copy here.
// NOTE: This file is included by moe_quantized.cu only. The grid tables are large;
// they are defined directly in moe_quantized.cu to avoid header bloat.

// ─── Dequant functions ────────────────────────────────────────────────────────
// Each takes a raw block pointer and writes QK_K elements to yy.
// Uses blockIdx.x as block index i and threadIdx.x as position 0..255.
// CALLER must set blockIdx.x = block index, blockDim.x = 256.

// Q2_K dequantize (ported from quantized.cu:1467-1498)
template<typename dst_t>
static __device__ __forceinline__ void moe_dequant_q2_K(
    const void* __restrict__ vx,
    int block_idx,
    dst_t* __restrict__ yy
) {
    const moe_block_q2_K* x = (const moe_block_q2_K*)vx;
    const int tid = threadIdx.x;
    const int n = tid / 32;
    const int l = tid - 32 * n;
    const int is = 8 * n + l / 16;

    const uint8_t q = x[block_idx].qs[32 * n + l];
    dst_t* y = yy + block_idx * QK_K + 128 * n;

    float dall = __low2half(x[block_idx].dm);
    float dmin = __high2half(x[block_idx].dm);
    y[l + 0] = dall * (x[block_idx].scales[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (x[block_idx].scales[is + 0] >> 4);
    y[l + 32] = dall * (x[block_idx].scales[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (x[block_idx].scales[is + 2] >> 4);
    y[l + 64] = dall * (x[block_idx].scales[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (x[block_idx].scales[is + 4] >> 4);
    y[l + 96] = dall * (x[block_idx].scales[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (x[block_idx].scales[is + 6] >> 4);
}

// Q4_K dequantize (ported from quantized.cu:1566-1604)
static __device__ __forceinline__ void moe_get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void moe_dequant_q4_K(
    const void* __restrict__ vx,
    int block_idx,
    dst_t* __restrict__ yy
) {
    const moe_block_q4_K* x = (const moe_block_q4_K*)vx;
    const int tid = threadIdx.x;
    const int il = tid / 8;
    const int ir = tid % 8;
    const int is = 2 * il;
    const int n = 4;

    dst_t* y = yy + block_idx * QK_K + 64 * il + n * ir;

    const float dall = __low2half(x[block_idx].dm);
    const float dmin = __high2half(x[block_idx].dm);

    const uint8_t* q = x[block_idx].qs + 32 * il + n * ir;

    uint8_t sc, m;
    moe_get_scale_min_k4(is + 0, x[block_idx].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    moe_get_scale_min_k4(is + 1, x[block_idx].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l + 32] = d2 * (q[l] >> 4) - m2;
    }
}

// IQ3_XXS dequantize (ported from quantized.cu:1764-1791)
template<typename dst_t>
static __device__ __forceinline__ void moe_dequant_iq3_xxs(
    const void* __restrict__ vx,
    int block_idx,
    dst_t* __restrict__ yy,
    const uint32_t* __restrict__ iq3xxs_grid  // [256] — passed as param for module isolation
) {
    const int pos = threadIdx.x;  // 0..255
    const moe_block_iq3_xxs* x = (const moe_block_iq3_xxs*)vx;

    const int ib32 = pos / 32;
    const int sub_pos = pos % 32;
    const int il = sub_pos / 16;
    const int j = sub_pos % 16;
    const int half = j / 8;
    const int pair = (j % 8) / 4;
    const int elem = j % 4;

    const uint8_t* q3 = x[block_idx].qs + 8 * ib32;
    const uint16_t* gas = (const uint16_t*)(x[block_idx].qs + QK_K / 4) + 2 * ib32;
    const uint32_t aux32 = (uint32_t)gas[0] | ((uint32_t)gas[1] << 16);

    const float d = __half2float(x[block_idx].d);
    const float dl = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

    const uint8_t grid_idx = q3[4 * il + 2 * half + pair];
    const uint8_t* grid = (const uint8_t*)(iq3xxs_grid + grid_idx);
    const uint8_t signs = moe_ksigns_iq2xs[(aux32 >> (14 * il + 7 * half)) & 127];
    const uint8_t sign_mask = moe_kmask_iq2xs[elem + 4 * pair];

    const float value = dl * (float)grid[elem] * (signs & sign_mask ? -1.f : 1.f);
    yy[block_idx * QK_K + pos] = value;
}

// IQ2_S dequantize (ported from quantized.cu:1796-1821)
template<typename dst_t>
static __device__ __forceinline__ void moe_dequant_iq2_s(
    const void* __restrict__ vx,
    int block_idx,
    dst_t* __restrict__ yy,
    const uint64_t* __restrict__ iq2s_grid  // [1024] — passed as param for module isolation
) {
    const int pos = threadIdx.x;  // 0..255
    const moe_block_iq2_s* x = (const moe_block_iq2_s*)vx;

    const int ib32 = pos / 32;
    const int sub_pos = pos % 32;
    const int il = sub_pos / 16;
    const int j = sub_pos % 16;
    const int half = j / 8;
    const int elem = j % 8;

    const uint8_t* qs = x[block_idx].qs + 4 * ib32 + 2 * il;
    const uint8_t* signs = x[block_idx].qs + QK_K / 8 + 4 * ib32 + 2 * il;
    const uint8_t qh = x[block_idx].qh[ib32] >> (4 * il);

    const float d = __half2float(x[block_idx].d);
    const float dl = d * (0.5f + (float)((x[block_idx].scales[ib32] >> (4 * il)) & 0xf)) * 0.25f;

    const int grid_idx = qs[half] | ((qh << (8 - 2 * half)) & 0x300);
    const uint8_t* grid = (const uint8_t*)(iq2s_grid + grid_idx);
    const uint8_t sign_mask = moe_kmask_iq2xs[elem];

    const float value = dl * (float)grid[elem] * ((signs[half] & sign_mask) ? -1.f : 1.f);
    yy[block_idx * QK_K + pos] = value;
}

// IQ2_XXS dequantize (ported from quantized.cu:1885-1912)
template<typename dst_t>
static __device__ __forceinline__ void moe_dequant_iq2_xxs(
    const void* __restrict__ vx,
    int block_idx,
    dst_t* __restrict__ yy,
    const uint64_t* __restrict__ iq2xxs_grid  // [256] — passed as param for module isolation
) {
    const int pos = threadIdx.x;  // 0..255
    const moe_block_iq2_xxs* x = (const moe_block_iq2_xxs*)vx;

    const int ib32 = pos / 32;
    const int sub_pos = pos % 32;
    const int il = sub_pos / 16;
    const int j = sub_pos % 16;
    const int half = j / 8;
    const int elem = j % 8;

    const uint16_t* q2 = x[block_idx].qs + 4 * ib32;
    const uint32_t aux32_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux32_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    const uint8_t* aux8 = (const uint8_t*)&aux32_g;

    const float d = __half2float(x[block_idx].d);
    const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;

    const int grid_idx = aux8[2 * il + half];
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + grid_idx);
    const uint8_t signs = moe_ksigns_iq2xs[(aux32_s >> (14 * il + 7 * half)) & 127];
    const uint8_t sign_mask = moe_kmask_iq2xs[elem];

    const float value = dl * (float)grid[elem] * ((signs & sign_mask) ? -1.f : 1.f);
    yy[block_idx * QK_K + pos] = value;
}

// ─── Quant type IDs (must match GgmlDType::to_u32() in candle-core) ───────────
// These are used by moe_quantized.cu dispatch. Rust launcher passes dtype.to_u32().
#define MOE_QTYPE_Q4_K    12
#define MOE_QTYPE_Q2_K    10
#define MOE_QTYPE_IQ3_XXS  18
#define MOE_QTYPE_IQ2_S   22
#define MOE_QTYPE_IQ2_XXS  16
