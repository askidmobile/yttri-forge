// DeltaNet decode: боевые ядра движка против приёма llama.cpp + вариант
// «транспонировать только в shared» (сохраняет глобальный layout состояния).
//
// Запускает РЕАЛЬНЫЕ ядра из delta_rule_batched.cu (через #include), поэтому
// числа сопоставимы с продовым профилем, а не с огрублённой копией.
//
// Сравниваются:
//   prod_split  — delta_rule_kernel_batched_split, block=(hd,ROWGRP)  [боевой дефолт]
//   prod_splitc — delta_rule_kernel_batched_splitc, block=(COLS,ROWGRP) [DELTA_DECODE=cols]
//   B (llama)   — warp на столбец, состояние ТРАНСПОНИРОВАНО (M[col*hd+row])
//   D (shared)  — глобальный layout тот же, но блок сам транспонирует тайл
//                 через shared и дальше работает как B внутри shared
// Математика у всех одна, проверка — против CPU-эталона.

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include "delta_rule_batched.cu"

// ---------- B: приём llama.cpp (gated_delta_net.cu) ----------
template<int SV>
__global__ void __launch_bounds__(SV, 2) kB(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ gate, float* __restrict__ M,   // M[col*SV + row]
    float* __restrict__ out, const DeltaParams p, const unsigned* __restrict__ slots)
{
    const unsigned hd = p.head_v_dim, n_v = p.n_v_heads;
    const unsigned head = blockIdx.x, seq = blockIdx.y;
    const int lane = threadIdx.x;
    const int col  = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= (int)hd) return;
    const unsigned vb = seq * n_v * hd + head * hd;
    float* m = M + ((size_t)slots[seq] * n_v + head) * hd * hd;
    const float g = __expf(gate[seq * n_v + head]);
    const float b = beta[seq * n_v + head];
    constexpr int RP = SV / 32;
    float s[RP];
#pragma unroll
    for (int r = 0; r < RP; r++) s[r] = m[col * SV + r * 32 + lane];
    float kreg[RP], qreg[RP];
#pragma unroll
    for (int r = 0; r < RP; r++) { kreg[r] = k[vb + r * 32 + lane]; qreg[r] = q[vb + r * 32 + lane]; }
    float kvp = 0.f;
#pragma unroll
    for (int r = 0; r < RP; r++) kvp += s[r] * kreg[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kvp += __shfl_down_sync(0xffffffffu, kvp, o);
    kvp = __shfl_sync(0xffffffffu, kvp, 0);
    const float d = (v[vb + col] - g * kvp) * b;
    float op = 0.f;
#pragma unroll
    for (int r = 0; r < RP; r++) { s[r] = g * s[r] + kreg[r] * d; op += s[r] * qreg[r]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) op += __shfl_down_sync(0xffffffffu, op, o);
    if (lane == 0) out[vb + col] = op * p.q_scale;
#pragma unroll
    for (int r = 0; r < RP; r++) m[col * SV + r * 32 + lane] = s[r];
}


// ---------- C: warp на столбец, состояние ROW-MAJOR (глобальный layout боевой) ----------
// Отличие от B только в раскладке чтений/записей состояния. При warps=32 блок
// покрывает 32 подряд идущих столбца, поэтому строка состояния, которую читают
// все 32 warpa вместе, попадает в кэш-линию целиком.
template<int SV>
__global__ void __launch_bounds__(1024, 2) kC(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ gate, float* __restrict__ S,   // S[row*SV + col]
    float* __restrict__ out, const DeltaParams p, const unsigned* __restrict__ slots)
{
    const unsigned hd = p.head_v_dim, n_v = p.n_v_heads;
    const unsigned head = blockIdx.x, seq = blockIdx.y;
    const int lane = threadIdx.x;
    const int col  = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= (int)hd) return;
    const unsigned vb = seq * n_v * hd + head * hd;
    float* s = S + ((size_t)slots[seq] * n_v + head) * hd * hd;
    const float g = __expf(gate[seq * n_v + head]);
    const float b = beta[seq * n_v + head];
    constexpr int RP = SV / 32;
    float st[RP];
#pragma unroll
    for (int r = 0; r < RP; r++) st[r] = s[(r * 32 + lane) * SV + col];
    float kreg[RP], qreg[RP];
#pragma unroll
    for (int r = 0; r < RP; r++) { kreg[r] = k[vb + r * 32 + lane]; qreg[r] = q[vb + r * 32 + lane]; }
    float kvp = 0.f;
#pragma unroll
    for (int r = 0; r < RP; r++) kvp += st[r] * kreg[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kvp += __shfl_down_sync(0xffffffffu, kvp, o);
    kvp = __shfl_sync(0xffffffffu, kvp, 0);
    const float d = (v[vb + col] - g * kvp) * b;
    float op = 0.f;
#pragma unroll
    for (int r = 0; r < RP; r++) { st[r] = g * st[r] + kreg[r] * d; op += st[r] * qreg[r]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) op += __shfl_down_sync(0xffffffffu, op, o);
    if (lane == 0) out[vb + col] = op * p.q_scale;
#pragma unroll
    for (int r = 0; r < RP; r++) s[(r * 32 + lane) * SV + col] = st[r];
}


// ---------- D: shared-транспозиция тайла, ГЛОБАЛЬНЫЙ layout не меняется ----------
// grid=(H, B, hd/32), block=(32,32). Блок берёт 32 столбца состояния,
// транспонирует их в shared ([col][row]), считает рекурренцию как llama.cpp
// (warp на столбец, shuffle-редукции, без барьеров внутри шага), затем пишет
// тайл обратно в тот же row-major layout. Глобальный формат состояния и все
// снапшоты/prefix-cache остаются прежними.
template<int SV, int CW, int WR>
__global__ void __launch_bounds__(32*WR, 2) kD(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ gate, float* __restrict__ S,   // S[row*SV + col]
    float* __restrict__ out, const DeltaParams p, const unsigned* __restrict__ slots)
{
    extern __shared__ float tile[];                 // [32][SV+1] (паддинг против bank conflict)
    constexpr int TS = SV + 1;                       // паддинг строки тайла
    const unsigned hd = p.head_v_dim, n_v = p.n_v_heads;
    const unsigned head = blockIdx.x, seq = blockIdx.y;
    const int lane = threadIdx.x, w = threadIdx.y;
    const int col0 = blockIdx.z * CW;
    const unsigned vb = seq * n_v * hd + head * hd;
    float* s = S + ((size_t)slots[seq] * n_v + head) * hd * hd;

    // Coalesced загрузка: линейный tid обходит [hd][CW], соседние lane читают
    // соседние столбцы -> 128-байтовые транзакции. В shared кладём транспонировано.
    {
        const int tid = w * 32 + lane;
        for (int e = tid; e < (int)hd * CW; e += 32*WR) {
            const int r = e / CW;
            const int c = e % CW;
            tile[c * TS + r] = s[r * hd + (col0 + c)];
        }
    }
    __syncthreads();

    const float g = __expf(gate[seq * n_v + head]);
    const float b = beta[seq * n_v + head];
    constexpr int RP = SV / 32;
    for (int c = w; c < CW; c += WR) {
        float st[RP];
#pragma unroll
            for (int r = 0; r < RP; r++) st[r] = tile[c * TS + r * 32 + lane];
        float kreg[RP], qreg[RP];
#pragma unroll
        for (int r = 0; r < RP; r++) { kreg[r] = k[vb + r * 32 + lane]; qreg[r] = q[vb + r * 32 + lane]; }
        float kvp = 0.f;
#pragma unroll
        for (int r = 0; r < RP; r++) kvp += st[r] * kreg[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) kvp += __shfl_down_sync(0xffffffffu, kvp, o);
        kvp = __shfl_sync(0xffffffffu, kvp, 0);
        const float d = (v[vb + col0 + c] - g * kvp) * b;
        float op = 0.f;
#pragma unroll
        for (int r = 0; r < RP; r++) { st[r] = g * st[r] + kreg[r] * d; op += st[r] * qreg[r]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) op += __shfl_down_sync(0xffffffffu, op, o);
        if (lane == 0) out[vb + col0 + c] = op * p.q_scale;
#pragma unroll
        for (int r = 0; r < RP; r++) tile[c * TS + r * 32 + lane] = st[r];
    }
    __syncthreads();
    {
        const int tid = w * 32 + lane;
        for (int e = tid; e < (int)hd * CW; e += 32*WR) {
            const int r = e / CW;
            const int c = e % CW;
            s[r * hd + (col0 + c)] = tile[c * TS + r];
        }
    }
}

static void cpu_ref(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,
                    const std::vector<float>&be,const std::vector<float>&ga,
                    std::vector<float>&S,std::vector<float>&o,const DeltaParams&p,unsigned B,
                    const std::vector<unsigned>&slots){
    for(unsigned b=0;b<B;b++){
        const unsigned vb=b*p.n_v_heads*p.head_v_dim, sh=b*p.n_v_heads;
        for(unsigned h=0;h<p.n_v_heads;h++){
            float* s=S.data()+((size_t)slots[b]*p.n_v_heads+h)*p.head_v_dim*p.head_v_dim;
            const float g=expf(ga[sh+h]), bt=be[sh+h];
            const unsigned off=vb+h*p.head_v_dim;
            for(unsigned c=0;c<p.head_v_dim;c++){
                float kv=0.f; for(unsigned r=0;r<p.head_v_dim;r++) kv+=g*s[r*p.head_v_dim+c]*k[off+r];
                const float d=(v[off+c]-kv)*bt; float oo=0.f;
                for(unsigned r=0;r<p.head_v_dim;r++){ float sv=g*s[r*p.head_v_dim+c]+k[off+r]*d; s[r*p.head_v_dim+c]=sv; oo+=sv*q[off+r]; }
                o[off+c]=oo*p.q_scale;
            }
        }
    }
}

static void set_smem(const void* fn, int bytes){
    if (bytes > 48*1024) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
}
static void chk(const char* where){
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  !! LAUNCH ERROR at %s: %s\n", where, cudaGetErrorString(e)); }
}

int main(){
    DeltaParams p{}; p.n_k_heads=16; p.n_v_heads=32; p.head_k_dim=128; p.head_v_dim=128;
    p.key_dim=2048; p.value_dim=4096; p.channels=8192; p.conv_kernel=4;
    p.q_scale=1.0f; p.rms_norm_eps=1e-6f; p.heads_per_kv=2; p.batch_size=4;
    const unsigned H=32,HD=128,CAP=4;
    std::mt19937 rng(11); std::uniform_real_distribution<float> U(-1,1);
    const unsigned B=1; std::vector<unsigned> slots={1};
    const size_t nq=(size_t)B*H*HD, nst=(size_t)CAP*H*HD*HD;
    std::vector<float> q(nq),k(nq),v(nq),be((size_t)B*H),ga((size_t)B*H);
    for(auto&x:q)x=U(rng); for(auto&x:k)x=U(rng); for(auto&x:v)x=U(rng);
    for(size_t i=0;i<be.size();i++){be[i]=0.3f+0.2f*U(rng); ga[i]=-0.6f+0.05f*U(rng);}
    std::vector<float> S0(nst); for(auto&x:S0)x=0.1f*U(rng);
    std::vector<float> Sr=S0, Or(nq,0.f); cpu_ref(q,k,v,be,ga,Sr,Or,p,B,slots);

    float *dq,*dk,*dv,*db,*dg,*dS,*dO; unsigned* dsl;
    cudaMalloc(&dq,nq*4);cudaMalloc(&dk,nq*4);cudaMalloc(&dv,nq*4);
    cudaMalloc(&db,be.size()*4);cudaMalloc(&dg,ga.size()*4);
    cudaMalloc(&dS,nst*4);cudaMalloc(&dO,nq*4);cudaMalloc(&dsl,B*4);
    cudaMemcpy(dq,q.data(),nq*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dk,k.data(),nq*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dv,v.data(),nq*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,be.data(),be.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dg,ga.data(),ga.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dsl,slots.data(),B*4,cudaMemcpyHostToDevice);

    // транспонированное состояние для B
    std::vector<float> Mt(nst);
    for(unsigned sl=0;sl<CAP;sl++)for(unsigned h=0;h<H;h++)for(unsigned c=0;c<HD;c++)for(unsigned r=0;r<HD;r++)
        Mt[(size_t)sl*H*HD*HD + h*HD*HD + c*HD + r] = S0[(size_t)sl*H*HD*HD + h*HD*HD + r*HD + c];

    const int iters=2000, reps=7;
    auto median=[&](std::vector<double> xs){ std::sort(xs.begin(),xs.end()); return xs[xs.size()/2]; };
    auto timeit=[&](auto launch,std::vector<float>* init)->double{
        std::vector<double> r;
        for(int rep=0;rep<reps;rep++){
            cudaMemcpy(dS,init->data(),nst*4,cudaMemcpyHostToDevice);
            launch(); chk("warmup"); cudaDeviceSynchronize();
            cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1);
            cudaEventRecord(e0);
            for(int i=0;i<iters;i++) launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms=0; cudaEventElapsedTime(&ms,e0,e1); r.push_back(ms/iters);
            cudaEventDestroy(e0);cudaEventDestroy(e1);
        }
        return median(r);
    };
    auto checkit=[&](auto launch,std::vector<float>* init,const char* nm){
        cudaMemcpy(dS,init->data(),nst*4,cudaMemcpyHostToDevice);
        launch(); cudaDeviceSynchronize();
        std::vector<float> Og(nq); cudaMemcpy(Og.data(),dO,nq*4,cudaMemcpyDeviceToHost);
        double m=0; for(size_t i=0;i<nq;i++) m=fmax(m,fabs(Og[i]-Or[i]));
        printf("%-34s max|dout|=%.3e  %s\n", nm, m, m<1e-4?"MATCH":"MISMATCH");
    };

    set_smem((const void*)kD<128,128,32>, 128*129*4);
    // B=4: 4 слота, индирекция
    const unsigned B4=4; const size_t nq4=(size_t)B4*H*HD, nst4=(size_t)CAP*H*HD*HD;
    std::vector<float> q4(nq4),k4(nq4),v4(nq4),be4((size_t)B4*H),ga4((size_t)B4*H);
    for(auto&x:q4)x=U(rng); for(auto&x:k4)x=U(rng); for(auto&x:v4)x=U(rng);
    for(size_t i=0;i<be4.size();i++){be4[i]=0.3f+0.2f*U(rng); ga4[i]=-0.6f+0.05f*U(rng);}
    float *dq4,*dk4,*dv4,*db4,*dg4; unsigned* dsl4;
    cudaMalloc(&dq4,nq4*4);cudaMalloc(&dk4,nq4*4);cudaMalloc(&dv4,nq4*4);
    cudaMalloc(&db4,be4.size()*4);cudaMalloc(&dg4,ga4.size()*4);cudaMalloc(&dsl4,B4*4);
    cudaMemcpy(dq4,q4.data(),nq4*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dk4,k4.data(),nq4*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dv4,v4.data(),nq4*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db4,be4.data(),be4.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dg4,ga4.data(),ga4.size()*4,cudaMemcpyHostToDevice);
    { std::vector<unsigned> sl={3,0,2,1}; cudaMemcpy(dsl4,sl.data(),B4*4,cudaMemcpyHostToDevice); }
    auto dsl4_v = dsl4;
    auto L_split =[&]{ delta_rule_kernel_batched_split <<<dim3(p.n_v_heads,B,1),dim3(HD,4,1),HD*4*4>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };
    auto L_splitc=[&]{ delta_rule_kernel_batched_splitc<<<dim3(p.n_v_heads,B,HD/32),dim3(32,4,1),32*4*4>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };
    auto L_b2    =[&]{ kB<128><<<dim3(H,B,HD/2),dim3(32,2,1)>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };
    auto L_b4    =[&]{ kB<128><<<dim3(H,B,HD/4),dim3(32,4,1)>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };
    auto L_d32   =[&]{ kD<128,32,32> <<<dim3(H,B,HD/32),dim3(32,32,1),32*129*4>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };
    auto L_d64   =[&]{ kD<128,64,32> <<<dim3(H,B,HD/64),dim3(32,32,1),64*129*4>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };
    auto L_repo  =[&]{ delta_rule_kernel_batched_tile<<<dim3(H,B,HD/64),dim3(32,32,1),64*129*4>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("repo"); };
    auto L_d128  =[&]{ kD<128,128,32><<<dim3(H,B,1),dim3(32,32,1),128*129*4>>>(dq,dk,dv,db,dg,dS,dO,p,dsl); chk("L"); };

    double t_split=timeit(L_split,&S0), t_splitc=timeit(L_splitc,&S0);
    double t_b2=timeit(L_b2,&Mt), t_b4=timeit(L_b4,&Mt);
    double t_d32=timeit(L_d32,&S0), t_d64=timeit(L_d64,&S0), t_d128=timeit(L_d128,&S0);
    double t_repo=timeit(L_repo,&S0);

    printf("=== DeltaNet decode, 1 layer (H=%u hd=%u B=%u), state %.1f MiB ===\n",H,HD,B,(double)nst*4/1048576);
    printf("prod split        block=128x4 grid=(32,1,1)   : %8.4f ms\n", t_split);
    printf("prod splitc       block=32x4  grid=(32,1,4)   : %8.4f ms\n", t_splitc);
    printf("B  transposed     warps=2 grid=(32,1,64)      : %8.4f ms  (%.2fx vs prod)\n", t_b2, t_split/t_b2);
    printf("B  transposed     warps=4 grid=(32,1,32)      : %8.4f ms  (%.2fx)\n", t_b4, t_split/t_b4);
    printf("D  smem-transpose CW=32  WR=32 grid=(32,1,4)  : %8.4f ms  (%.2fx)\n", t_d32, t_split/t_d32);
    printf("D  smem-transpose CW=64  WR=32 grid=(32,1,2)  : %8.4f ms  (%.2fx)\n", t_d64, t_split/t_d64);
    printf("D  smem-transpose CW=128 WR=32 grid=(32,1,1)  : %8.4f ms  (%.2fx)\n", t_d128, t_split/t_d128);
    printf("REPO delta_rule_kernel_batched_tile (CW=64)   : %8.4f ms  (%.2fx)\n", t_repo, t_split/t_repo);

    // B=4 отдельно: свой выходной буфер, чтобы не переполнять dO (1 слот).
    float* dO4=nullptr; cudaMalloc(&dO4, nq4*4);
    std::vector<double> b4r;
    for(int rep=0;rep<7;rep++){
        cudaMemcpy(dS,S0.data(),(size_t)4*H*HD*HD*4,cudaMemcpyHostToDevice);
        delta_rule_kernel_batched_tile<<<dim3(H,4,HD/64),dim3(32,32,1),64*129*4>>>(dq4,dk4,dv4,db4,dg4,dS,dO4,p,dsl4);
        chk("repoB4"); cudaDeviceSynchronize();
        cudaEvent_t e0,e1; cudaEventCreate(&e0);cudaEventCreate(&e1); cudaEventRecord(e0);
        for(int i=0;i<2000;i++) delta_rule_kernel_batched_tile<<<dim3(H,4,HD/64),dim3(32,32,1),64*129*4>>>(dq4,dk4,dv4,db4,dg4,dS,dO4,p,dsl4);
        cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1);
        b4r.push_back(ms/2000); cudaEventDestroy(e0);cudaEventDestroy(e1);
    }
    double t_repo_b4=median(b4r);
    printf("REPO tile, B=4 slots                          : %8.4f ms\n", t_repo_b4);

    printf("\n--- корректность против CPU-эталона ---\n");
    checkit(L_split,&S0,"prod split");  checkit(L_splitc,&S0,"prod splitc");
    checkit(L_b4,&Mt,"B transposed");  checkit(L_d32,&S0,"D smem-transpose CW=32");
    checkit(L_d64,&S0,"D smem-transpose CW=64"); checkit(L_d128,&S0,"D smem-transpose CW=128");
    checkit(L_repo,&S0,"REPO tile kernel CW=64 (1 slot)");
    return 0;
}
