#include <cstdio>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "kernels.h"
#include "kernel_helpers.h"
#include "flash_fwd_launch_template.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
      BOOL_SWITCH(params.is_causal, Is_causal, [&] {
          if (params.block_table != nullptr) {
              // Only the splitkv kernel understands paged KV; head dims here match the host-side gate.
              if (params.d <= 64) {
                  run_mha_fwd_splitkv_paged_<elem_type, 64, Is_causal>(params, stream);
              } else if (params.d <= 128) {
                  run_mha_fwd_splitkv_paged_<elem_type, 128, Is_causal>(params, stream);
              } else if (params.d <= 256) {
                  run_mha_fwd_splitkv_paged_<elem_type, 256, Is_causal>(params, stream);
              } else {
                  run_mha_fwd_splitkv_paged_<elem_type, 512, Is_causal>(params, stream);
              }
          } else {
              HEADDIM_SWITCH(params.d, [&] {
                  run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
              });
          }
      });
  });
}


// ── Split-KV: выбор числа сплитов и буферы-аккумуляторы ──
// На декоде (seqlen_q=1) сетка без сплитов вырождается в batch*heads блоков —
// для 16 голов это 16-32 блока на 28 SM, половина карты простаивает, и ядро
// идёт в 10 раз выше предела памяти. Эвристика — из апстрима flash-attn.
static inline int fa_ceildiv(int a, int b) { return (a + b - 1) / b; }

static int fa_num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min(std::min(max_splits, num_SMs), num_n_blocks);
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto is_split_eligible = [&](int ns) {
        return ns == 1 || fa_ceildiv(num_n_blocks, ns) != fa_ceildiv(num_n_blocks, ns - 1);
    };
    for (int ns = 1; ns <= max_splits; ns++) {
        if (!is_split_eligible(ns)) { efficiency.push_back(0.f); continue; }
        float n_waves = float(batch_nheads_mblocks * ns) / num_SMs;
        float eff = n_waves / std::ceil(n_waves);
        if (eff > max_efficiency) { max_efficiency = eff; }
        efficiency.push_back(eff);
    }
    for (int ns = 1; ns <= max_splits; ns++) {
        if (!is_split_eligible(ns)) { continue; }
        if (efficiency[ns - 1] >= 0.85f * max_efficiency) { return ns; }
    }
    return 1;
}

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,
    uint32_t total_q,

    int is_bf16,
    int is_causal,
    int unpadded_lse,

    int window_size_left,
    int window_size_right,

    float softcap,

    int32_t *block_table_ptr,
    uint32_t block_table_batch_stride,
    int page_block_size,

    int32_t *mm_prefix_ranges_ptr,
    uint32_t mm_prefix_range_batch_stride,
    int max_mm_prefix_ranges,

    // int8 постраничный KV: масштабы на пару (токен, голова). Раскладка
    // зеркалит пул без измерения head_dim. Нули = обычный F16-пул.
    void *k_scale_ptr,
    void *v_scale_ptr,
    uint32_t k_scale_batch_stride,
    uint32_t k_scale_row_stride,
    uint32_t v_scale_batch_stride,
    uint32_t v_scale_row_stride,
    int kv_is_q8,

    void *stream_ptr
) {
    Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.total_q = total_q;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr; // used for `return_softmax`.
    params.seqused_k = nullptr;
    params.block_table = block_table_ptr;
    params.block_table_batch_stride = block_table_batch_stride;
    params.page_block_size = page_block_size;
    params.mm_prefix_ranges = mm_prefix_ranges_ptr;
    params.mm_prefix_range_batch_stride = mm_prefix_range_batch_stride;
    params.max_mm_prefix_ranges = max_mm_prefix_ranges;

    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.k_scale_ptr = k_scale_ptr;
    params.v_scale_ptr = v_scale_ptr;
    params.k_scale_batch_stride = k_scale_batch_stride;
    params.k_scale_row_stride = k_scale_row_stride;
    params.k_scale_head_stride = 1;
    params.v_scale_batch_stride = v_scale_batch_stride;
    params.v_scale_row_stride = v_scale_row_stride;
    params.v_scale_head_stride = 1;
    params.kv_is_q8 = kv_is_q8 != 0;

    params.is_seqlens_k_cumulative = true;
    params.unpadded_lse = unpadded_lse;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Упаковка GQA в измерение запросов (приём seqlenq_ngroups_swapped из FA2;
    // поддержка в ядрах уже есть, не хватало только хостовой части).
    //
    // В декоде запрос ровно один, а M-тайл ядра равен 64 — 63 строки из 64
    // считаются впустую. Вдобавок при h=16 и h_k=4 четыре блока читают один
    // и тот же KV, то есть трафик вчетверо лишний. Если подставить группу GQA
    // в seqlen_q, обе потери уходят разом: данные не двигаются, меняются
    // только страйды.
    static const bool gqa_swap_enabled = [] {
        const char* e = std::getenv("QWEN36_FA_GQA");
        return e == nullptr || std::atoi(e) != 0;
    }();
    const bool is_decode = (seqlen_q == 1);
    // При одном запросе маска ничего не режет: позиция запроса последняя,
    // так что и is_causal, и правое окно 0 эквивалентны полному вниманию.
    // Настоящее скользящее окно (window_size_left >= 0) так свернуть нельзя.
    const bool mask_is_noop = (window_size_left < 0) && (window_size_right <= 0);
    if (gqa_swap_enabled && is_decode && h > h_k && mask_is_noop &&
        alibi_slopes_ptr == nullptr && d % 8 == 0) {
        const int ngroups = h / h_k;
        const auto q_head = params.q_head_stride;
        const auto o_head = params.o_head_stride;
        params.q_batch_stride = params.q_row_stride;
        params.q_row_stride = q_head;
        params.q_head_stride = q_head * ngroups;
        params.o_batch_stride = params.o_row_stride;
        params.o_row_stride = o_head;
        params.o_head_stride = o_head * ngroups;
        params.cu_seqlens_q = nullptr;
        params.seqlen_q = ngroups;
        params.seqlen_q_rounded = ((ngroups + 127) / 128) * 128;
        params.h = h_k;
        params.h_h_k_ratio = 1;
        params.total_q = b * ngroups;
        params.is_causal = false;
        params.window_size_left = -1;
        params.window_size_right = -1;
        params.seqlenq_ngroups_swapped = true;
    }

    // Число сплитов по K. QWEN36_FA_SPLITS=1 возвращает прежнее поведение.
    params.num_splits = 1;
    // QWEN36_FA_SPLITS: 1 — прежнее поведение без сплитов, N>1 — форсировать N,
    // не задано — эвристика апстрима.
    static const int forced_splits = [] {
        const char* e = std::getenv("QWEN36_FA_SPLITS");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    const bool force_single = (forced_splits == 1);
    void* oaccum = nullptr;
    void* lseaccum = nullptr;
    if (!force_single) {
        // Число SM спрашиваем один раз: на каждый вызов внимания эти два
        // хостовых запроса стоили заметно (префилл просел на 10%).
        static const int num_sms = [] {
            int dev = 0;
            cudaGetDevice(&dev);
            int n = 0;
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
            return n > 0 ? n : 1;
        }();
        const int block_n = d <= 64 ? 256 : (d <= 128 ? 128 : 64);
        const int num_n_blocks = fa_ceildiv(seqlen_k, block_n);
        const int num_m_blocks = fa_ceildiv(params.seqlen_q, 64);
        int ns;
        if (is_decode) {
            // Декод: запрос один, поэтому без сплитов сетка это b*h блоков
            // (16-32 на 28 SM) и карта простаивает. Апстримная эвристика
            // считает «волны» в предположении, что блок делает много работы,
            // и даёт всего 3. Замер на 12K (Qwen3.5-4B, RTX 3060):
            // 1 сплит 45.0 ток/с, 3 — 46.0, 8 — 52.0, 14 — 53.0, 16 — 55.4,
            // 32 — 53.0. Целимся в 9 волн (на b=1 h=16 это ровно 16 сплитов),
            // оставляя каждому сплиту не меньше двух блоков ключей.
            const int target = 9 * num_sms;
            const int denom = std::max(1, params.b * params.h * num_m_blocks);
            ns = fa_ceildiv(target, denom);
            ns = std::min(ns, num_n_blocks / 2);
            ns = std::min(ns, 128);
            ns = std::max(ns, 1);
        } else {
            ns = fa_num_splits_heuristic(params.b * params.h * num_m_blocks, num_sms, num_n_blocks, 128);
        }
        if (forced_splits > 1 && is_decode) {
            ns = std::min(forced_splits, num_n_blocks);
        }
        // Печатаем по одному разу для декода (seqlen_q==1) и для префилла.
        static bool reported_decode = false;
        static bool reported_prefill = false;
        bool& reported = is_decode ? reported_decode : reported_prefill;
        if (!reported && std::getenv("QWEN36_FA_DEBUG") != nullptr) {
            reported = true;
            fprintf(stderr, "[fa] splits: b=%d h=%d->%d seqlen_q=%d->%d seqlen_k=%d n_blocks=%d sms=%d gqa_swap=%d -> num_splits=%d\n",
                    b, h, params.h, seqlen_q, params.seqlen_q, seqlen_k, num_n_blocks, num_sms,
                    int(params.seqlenq_ngroups_swapped), ns);
        }
        if (ns > 1) {
            const size_t lse_elems = (size_t)ns * params.b * params.h * params.seqlen_q;
            const size_t o_elems = lse_elems * d_rounded;
            if (cudaMallocAsync(&lseaccum, lse_elems * sizeof(float), stream) == cudaSuccess &&
                cudaMallocAsync(&oaccum, o_elems * sizeof(float), stream) == cudaSuccess) {
                params.num_splits = ns;
                params.softmax_lseaccum_ptr = lseaccum;
                params.oaccum_ptr = oaccum;
            } else {
                if (lseaccum) { cudaFreeAsync(lseaccum, stream); lseaccum = nullptr; }
                if (oaccum) { cudaFreeAsync(oaccum, stream); oaccum = nullptr; }
            }
        }
    }

    run_mha_fwd(params, stream);

    if (lseaccum) { cudaFreeAsync(lseaccum, stream); }
    if (oaccum) { cudaFreeAsync(oaccum, stream); }
}
