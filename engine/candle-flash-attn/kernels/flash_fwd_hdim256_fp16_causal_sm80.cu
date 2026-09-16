// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"
#include <cstdlib>

// Плотный (не пейдженный) префильный FA2 для hdim=256. На sm86 штатная ветка
// `run_mha_fwd_hdim256` выбирает (kBlockM=64, kBlockN=64, 4 варпа): smem 96 КБ,
// один CTA на SM, 128 потоков. Замер 2026-09-16: префильное внимание идёт на
// ~7 TFLOPS (ncu: занятость ~26 % от per-SM пика) и занимает 29 % времени
// префила — это главный проигрыш llama.cpp на длинном контексте.
//
// Здесь добавлена вторая инстанциация (kBlockM=128, kBlockN=32, 8 варпов):
// тот же smem (2*d*(M+2N) = 96 КБ), но вдвое больше варпов на SM и вдвое
// крупный M-тайл. Выбор — runtime, через QWEN36_FA_PREFILL_TILE:
//   0 (по умолчанию) — штатная ветка, 1 — 128x32x8.
template<>
void run_mha_fwd_<cutlass::half_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream) {
    // По умолчанию — вариант 6 (fp16-аккумулятор, см. ниже): изолированный
    // замер 2026-09-16 дал 56.6 мс против 88.5 у f32 на T=16384 (38.9 против
    // 24.9 TFLOPS), ошибка относительно f32-референса выросла с 3e-5 до 1.5e-4
    // при среднем |выходе| 0.104. Откат — QWEN36_FA_PREFILL_TILE=1 (или 0).
    static const int variant = [] {
        const char *e = std::getenv("QWEN36_FA_PREFILL_TILE");
        return e != nullptr ? std::atoi(e) : 6;
    }();
    if (variant == 1) {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<256, 128, 32, 8, false, false, cutlass::half_t>,
                          Is_dropout, true>(params, stream);
        });
        return;
    }
    // Вариант 6 — тот же тайл (128x32x8), но тензорное ядро с fp16-
    // аккумулятором (f16.f16.f16.f16): на GA10x это вдвое больший темп.
    // Продакшн-ветка (вариант 1) считает в f32; сравнение — тот же бинарник.
    if (variant == 6) {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_fwd<Flash_fwd_kernel_traits<256, 128, 32, 8, false, false, cutlass::half_t, true>,
                          Is_dropout, true>(params, stream);
        });
        return;
    }
    run_mha_fwd_hdim256<cutlass::half_t, true>(params, stream);
}
