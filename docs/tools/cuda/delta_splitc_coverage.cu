// Автономная проверка покрытия столбцов состояния в
// delta_rule_kernel_batched_splitc (delta_rule_batched.cu).
//
// Проверяем ровно тот класс ошибки, из-за которого DELTA_COLS=16 давал
// ложные +5%: если номер столбца берётся из константы, а не из blockDim,
// покрывается лишь часть состояния. Стенд:
//   A) block=(32,4), grid.z=4  -> эталон (как до ручек DELTA_COLS)
//   B) block=(16,4), grid.z=8 -> новая раскладка через blockDim (фикс)
//   C) block=(16,4), grid.z=4 -> контроль чувствительности: половина
//      столбцов не считается, тест обязан это заметить
// Сравнение с последовательным CPU-эталоном по формуле самого ядра.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "delta_rule_batched.cu"

struct Case { const char* name; unsigned cols; unsigned rowgrp; unsigned gridz_expected; };

static void cpu_reference(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, const std::vector<float>& beta,
                          const std::vector<float>& gate, std::vector<float>& state,
                          std::vector<float>& out, const DeltaParams& p, unsigned B,
                          unsigned head) {
    const unsigned hd = p.head_v_dim;
    for (unsigned b = 0; b < B; b++) {
        const unsigned vb = b * p.n_v_heads * hd + head * hd;
        const unsigned sh = b * p.n_v_heads + head;
        float* st = state.data() + (b * p.n_v_heads + head) * hd * hd;
        const float ge = expf(gate[sh]);
        const float be = beta[sh];
        for (unsigned c = 0; c < hd; c++) {
            float sk = 0.0f;
            for (unsigned r = 0; r < hd; r++) sk += st[r * hd + c] * ge * k[vb + r];
            const float d = (v[vb + c] - sk) * be;
            float o = 0.0f;
            for (unsigned r = 0; r < hd; r++) {
                const float sv = st[r * hd + c] * ge + k[vb + r] * d;
                st[r * hd + c] = sv;
                o += sv * q[vb + r];
            }
            out[vb + c] = o;
        }
    }
}

static bool run_case(const Case& cs, const std::vector<float>& hq, const std::vector<float>& hk,
                     const std::vector<float>& hv, const std::vector<float>& hbeta,
                     const std::vector<float>& hgate, const std::vector<float>& hstate0,
                     const DeltaParams& p, unsigned B) {
    const unsigned hd = p.head_v_dim;
    const size_t nq = (size_t)B * p.n_v_heads * hd;
    const size_t nst = (size_t)B * p.n_v_heads * hd * hd;

    std::vector<float> ref_state = hstate0;
    std::vector<float> ref_out(nq, 0.0f);
    for (unsigned head = 0; head < p.n_v_heads; head++)
        cpu_reference(hq, hk, hv, hbeta, hgate, ref_state, ref_out, p, B, head);

    float *dq, *dk, *dv, *db, *dg, *ds, *do_; unsigned* dslots_u;
    cudaMalloc(&dq, nq * 4); cudaMalloc(&dk, nq * 4); cudaMalloc(&dv, nq * 4);
    cudaMalloc(&db, (size_t)B * p.n_v_heads * 4); cudaMalloc(&dg, (size_t)B * p.n_v_heads * 4);
    cudaMalloc(&ds, nst * 4); cudaMalloc(&do_, nq * 4);
    cudaMalloc(&dslots_u, B * 4);
    std::vector<unsigned> slots(B); for (unsigned b = 0; b < B; b++) slots[b] = b;
    cudaMemcpy(dq, hq.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hbeta.data(), (size_t)B * p.n_v_heads * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, hgate.data(), (size_t)B * p.n_v_heads * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hstate0.data(), nst * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dslots_u, slots.data(), B * 4, cudaMemcpyHostToDevice);

    dim3 grid(p.n_v_heads, B, cs.gridz_expected);
    dim3 block(cs.cols, cs.rowgrp, 1);
    const unsigned smem = cs.cols * cs.rowgrp * 4;
    delta_rule_kernel_batched_splitc<<<grid, block, smem>>>(dq, dk, dv, db, dg, ds, do_, p, dslots_u);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%-6s LAUNCH ERROR: %s\n", cs.name, cudaGetErrorString(err)); return false; }

    std::vector<float> gstate(nst), gout(nq);
    cudaMemcpy(gstate.data(), ds, nst * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gout.data(), do_, nq * 4, cudaMemcpyDeviceToHost);

    double dsmax = 0.0, domax = 0.0;
    for (size_t i = 0; i < nst; i++) dsmax = std::max(dsmax, (double)fabsf(gstate[i] - ref_state[i]));
    for (size_t i = 0; i < nq; i++)  domax = std::max(domax, (double)fabsf(gout[i] - ref_out[i]));
    const bool ok = (dsmax < 1e-3 && domax < 1e-3);
    printf("%-6s cols=%-3u rowgrp=%u grid.z=%u | dstate=%.3e dout=%.3e -> %s\n",
           cs.name, cs.cols, cs.rowgrp, cs.gridz_expected, dsmax, domax, ok ? "MATCH" : "MISMATCH");
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(db); cudaFree(dg); cudaFree(ds); cudaFree(do_); cudaFree(dslots_u);
    return ok;
}

int main() {
    DeltaParams p{};
    p.n_k_heads = 16; p.n_v_heads = 32; p.head_k_dim = 128; p.head_v_dim = 128;
    p.key_dim = 16 * 128; p.value_dim = 32 * 128; p.channels = p.key_dim * 2 + p.value_dim;
    p.conv_kernel = 4; p.q_scale = 1.0f / sqrtf(128.0f); p.rms_norm_eps = 1e-6f;
    p.heads_per_kv = 2; p.batch_size = 1;
    const unsigned B = 1, hd = 128;
    const size_t nq = (size_t)B * p.n_v_heads * hd;
    const size_t nst = (size_t)B * p.n_v_heads * hd * hd;
    const size_t nh = (size_t)B * p.n_v_heads;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), ub(0.0f, 1.0f);
    std::vector<float> hq(nq), hk(nq), hv(nq), hstate0(nst), hbeta(nh), hgate(nh);
    for (auto& x : hq) x = u(rng);
    for (auto& x : hk) x = u(rng);
    for (auto& x : hv) x = u(rng);
    for (auto& x : hstate0) x = 0.1f * u(rng);
    for (auto& x : hbeta) x = ub(rng);
    for (auto& x : hgate) x = -0.5f + 0.1f * u(rng);

    printf("=== delta_rule_kernel_batched_splitc: покрытие столбцов ===\n");
    bool a = run_case({"A-ref",  32, 4, 4}, hq, hk, hv, hbeta, hgate, hstate0, p, B);
    bool b = run_case({"B-fix",  16, 4, 8}, hq, hk, hv, hbeta, hgate, hstate0, p, B);
    bool c = run_case({"C-half", 16, 4, 4}, hq, hk, hv, hbeta, hgate, hstate0, p, B);
    printf("\nИтог: A=%s B=%s, контроль чувствительности C обязан быть MISMATCH -> %s\n",
           a ? "ok" : "FAIL", b ? "ok" : "FAIL", (!c) ? "ok (тест ловит неполное покрытие)" : "СЛЕПОЙ ТЕСТ");
    return (a && b && !c) ? 0 : 1;
}
