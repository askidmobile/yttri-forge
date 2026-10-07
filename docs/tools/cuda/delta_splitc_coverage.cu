// Автономная GPU-проверка раскладки в delta_rule_kernel_batched_split(c)
// (delta_rule_batched.cu) против последовательного CPU-эталона.
//
// Проверяем ровно тот класс ошибки, из-за которого DELTA_COLS=16 дал ложные
// +5% декода: если номер столбца/группы строк берётся из константы, а не из
// blockDim, покрывается лишь часть состояния, ядро работает меньше и
// «ускоряется» при мусорном результате.
//
// Стенд #include-ит РАБОЧИЙ delta_rule_batched.cu и запускает настоящие ядра,
// поэтому файл лежит вне src/ (иначе попал бы в сборку PTX как отдельный .cu).
//
// Порядок case'ов ниже: эталон -> вариант после фикса -> контроль
// чувствительности. Контроль обязан дать MISMATCH, иначе тест слепой.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include "delta_rule_batched.cu"

static void cpu_reference(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, const std::vector<float>& beta,
                          const std::vector<float>& gate, std::vector<float>& state,
                          std::vector<float>& out, const DeltaParams& p, unsigned B,
                          const std::vector<unsigned>& slots) {
    const unsigned hd = p.head_v_dim;
    for (unsigned b = 0; b < B; b++) {
        const unsigned vb = b * p.n_v_heads * hd;
        const unsigned sh = b * p.n_v_heads;
        for (unsigned head = 0; head < p.n_v_heads; head++) {
            float* st = state.data() + ((size_t)slots[b] * p.n_v_heads + head) * hd * hd;
            const float ge = expf(gate[sh + head]);
            const float be = beta[sh + head];
            const unsigned off = vb + head * hd;
            for (unsigned c = 0; c < hd; c++) {
                float sk = 0.0f;
                for (unsigned r = 0; r < hd; r++) sk += st[r * hd + c] * ge * k[off + r];
                const float d = (v[off + c] - sk) * be;
                float o = 0.0f;
                for (unsigned r = 0; r < hd; r++) {
                    const float sv = st[r * hd + c] * ge + k[off + r] * d;
                    st[r * hd + c] = sv;
                    o += sv * q[off + r];
                }
                out[off + c] = o;
            }
        }
    }
}

struct Case {
    const char* name;
    const char* kernel;   // "splitc" | "split" | "plain"
    unsigned cols;        // blockDim.x (только splitc)
    unsigned rowgrp;      // blockDim.y
    unsigned gridz;       // grid.z (только splitc)
};

static bool run_case(const Case& cs, const std::vector<float>& hq, const std::vector<float>& hk,
                     const std::vector<float>& hv, const std::vector<float>& hbeta,
                     const std::vector<float>& hgate, const std::vector<float>& hstate0,
                     const std::vector<unsigned>& slots, const DeltaParams& p) {
    const unsigned hd = p.head_v_dim;
    const unsigned B = (unsigned)slots.size();
    const size_t nq = (size_t)B * p.n_v_heads * hd;
    const size_t nst = 4 * p.n_v_heads * hd * hd; // CAP=4 слота под индирекцию

    std::vector<float> ref_state = hstate0;
    std::vector<float> ref_out(nq, 0.0f);
    cpu_reference(hq, hk, hv, hbeta, hgate, ref_state, ref_out, p, B, slots);

    float *dq, *dk, *dv, *db, *dg, *ds, *do_; unsigned* dslots_u;
    cudaMalloc(&dq, nq * 4); cudaMalloc(&dk, nq * 4); cudaMalloc(&dv, nq * 4);
    cudaMalloc(&db, (size_t)B * p.n_v_heads * 4); cudaMalloc(&dg, (size_t)B * p.n_v_heads * 4);
    cudaMalloc(&ds, nst * 4); cudaMalloc(&do_, nq * 4); cudaMalloc(&dslots_u, B * 4);
    cudaMemcpy(dq, hq.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dk, hk.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, hv.data(), nq * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hbeta.data(), (size_t)B * p.n_v_heads * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, hgate.data(), (size_t)B * p.n_v_heads * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hstate0.data(), nst * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dslots_u, slots.data(), B * 4, cudaMemcpyHostToDevice);

    unsigned smem;
    if (strcmp(cs.kernel, "splitc") == 0) {
        dim3 grid(p.n_v_heads, B, cs.gridz);
        dim3 block(cs.cols, cs.rowgrp, 1);
        smem = cs.cols * cs.rowgrp * 4;
        delta_rule_kernel_batched_splitc<<<grid, block, smem>>>(dq, dk, dv, db, dg, ds, do_, p, dslots_u);
    } else if (strcmp(cs.kernel, "split") == 0) {
        dim3 grid(p.n_v_heads, B, 1);
        dim3 block(hd, cs.rowgrp, 1);
        smem = hd * cs.rowgrp * 4;
        delta_rule_kernel_batched_split<<<grid, block, smem>>>(dq, dk, dv, db, dg, ds, do_, p, dslots_u);
    } else {
        dim3 grid(p.n_v_heads, B, 1);
        dim3 block(hd, 1, 1);
        smem = 0;
        delta_rule_kernel_batched<<<grid, block, smem>>>(dq, dk, dv, db, dg, ds, do_, p, dslots_u);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("%-7s LAUNCH ERROR: %s\n", cs.name, cudaGetErrorString(err)); return false; }

    std::vector<float> gstate(nst), gout(nq);
    cudaMemcpy(gstate.data(), ds, nst * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gout.data(), do_, nq * 4, cudaMemcpyDeviceToHost);

    double dsmax = 0.0, domax = 0.0;
    for (size_t i = 0; i < nst; i++) dsmax = std::max(dsmax, (double)fabsf(gstate[i] - ref_state[i]));
    for (size_t i = 0; i < nq; i++)  domax = std::max(domax, (double)fabsf(gout[i] - ref_out[i]));
    const bool ok = (dsmax < 1e-3 && domax < 1e-3);
    printf("%-7s %-6s B=%u cols=%-3u rowgrp=%-2u grid.z=%u | dstate=%.3e dout=%.3e -> %s\n",
           cs.name, cs.kernel, B, cs.cols, cs.rowgrp, cs.gridz, dsmax, domax, ok ? "MATCH" : "MISMATCH");
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(db); cudaFree(dg); cudaFree(ds); cudaFree(do_); cudaFree(dslots_u);
    return ok;
}

int main() {
    DeltaParams p{};
    p.n_k_heads = 16; p.n_v_heads = 32; p.head_k_dim = 128; p.head_v_dim = 128;
    p.key_dim = 16 * 128; p.value_dim = 32 * 128; p.channels = p.key_dim * 2 + p.value_dim;
    p.conv_kernel = 4; p.q_scale = 1.0f / sqrtf(128.0f); p.rms_norm_eps = 1e-6f;
    p.heads_per_kv = 2; p.batch_size = 4;
    const unsigned hd = 128;

    // CAP=4 слота; B=1 и B=3 с нетривиальной индирекцией slot_ids.
    const std::vector<unsigned> slots1 = {0};
    const std::vector<unsigned> slots3 = {2, 0, 3};
    const size_t nq3 = (size_t)slots3.size() * p.n_v_heads * hd;
    const size_t nst = 4 * p.n_v_heads * hd * hd;
    const size_t nh3 = (size_t)slots3.size() * p.n_v_heads;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), ub(0.0f, 1.0f);
    std::vector<float> hq(nq3), hk(nq3), hv(nq3);
    for (size_t i = 0; i < nq3; i++) { hq[i] = u(rng); hk[i] = u(rng); hv[i] = u(rng); }
    std::vector<float> hbeta(nh3), hgate(nh3);
    for (size_t i = 0; i < nh3; i++) { hbeta[i] = ub(rng); hgate[i] = -0.5f + 0.1f * u(rng); }
    std::vector<float> hstate0(nst); for (auto& e : hstate0) e = 0.1f * u(rng);

    // Публичная сетка дефолтов: COLS=32, ROWGRP=4 (splitc), ROWGRP=4 (split).
    printf("=== delta_rule_kernel_batched_split(c): раскладка против CPU-эталона ===\n");
    // Корректные раскладки обязаны совпасть с CPU-эталоном.
    bool good = true;
    good &= run_case({"A-ref",   "plain",  hd, 1, 1}, hq, hk, hv, hbeta, hgate, hstate0, slots1, p); // базовое ядро как эталон раскладки
    good &= run_case({"B-spl",   "split",  hd, 4, 1}, hq, hk, hv, hbeta, hgate, hstate0, slots1, p); // после фикса, ROWGRP=4
    good &= run_case({"C-spl8",  "split",  hd, 8, 1}, hq, hk, hv, hbeta, hgate, hstate0, slots1, p); // после фикса, ROWGRP=8
    good &= run_case({"D-c32",   "splitc", 32, 4, 4}, hq, hk, hv, hbeta, hgate, hstate0, slots1, p); // после фикса, дефолт
    good &= run_case({"E-c16",   "splitc", 16, 4, 8}, hq, hk, hv, hbeta, hgate, hstate0, slots1, p); // после фикса, DELTA_COLS=16
    good &= run_case({"G-b3",    "splitc", 16, 4, 8}, hq, hk, hv, hbeta, hgate, hstate0, slots3, p); // B=3 + индирекция слотов
    good &= run_case({"H-b3row8","split",  hd, 8, 1}, hq, hk, hv, hbeta, hgate, hstate0, slots3, p); // split ROWGRP=8, B=3
    // Контроль чувствительности обязан НЕ совпасть: иначе стенд слеп к тому
    // самому классу ошибки, который он должен ловить.
    const bool blind_guard_fired = !run_case({"F-c16bad","splitc", 16, 4, 4}, hq, hk, hv, hbeta, hgate, hstate0, slots1, p);
    printf("\nКорректных раскладок MATCH: %s; контроль F поймал неполное покрытие: %s\n",
           good ? "да" : "НЕТ", blind_guard_fired ? "да" : "НЕТ (стенд слеп)");
    const bool pass = good && blind_guard_fired;
    printf("Итог стенда: %s\n", pass ? "ok" : "FAIL");
    return pass ? 0 : 1;
}
