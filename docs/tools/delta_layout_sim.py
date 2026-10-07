#!/usr/bin/env python3
"""CPU-эмуляция ядра delta_rule_kernel_batched_splitc: старый (константы
DR_COLS/DR_ROWGRP) против нового (blockDim.x/y) разбора раскладки.

Математика ядра (одна голова, один слот, hd x hd состояние):
  sk[col]  = sum_r S[r][col] * k[r]           (редукция по группам строк)
  d[col]   = (v[col] - sk[col]) * beta
  S[r][col] = S[r][col] + k[r] * d[col]
  out[col] = sum_r S[r][col] * q[r]

Роль раскладки — какие (row, col) вообще трогает ядро. Старый вариант брал
col-страйд и число групп из констант; новый — из фактического blockDim.
"""
import random

HD = 128

def kernel(q, k, v, beta, S, cols_launch, rows_launch, use_const, dr_cols=32, dr_rowgrp=4):
    """Возвращает (out, S_new, touched) — touched = множество (row,col),
    реально обновлённых ядром."""
    cols = dr_cols if use_const else cols_launch
    rows = dr_rowgrp if use_const else rows_launch
    S = [row[:] for row in S]
    out = [0.0] * HD
    touched = set()
    nblocks_z = HD // cols_launch if cols_launch else 0
    rows_per = HD // rows if rows else 0
    for bz in range(nblocks_z):
        for tx in range(cols_launch):
            col = bz * cols + tx
            if col >= HD:
                continue
            # sk: редукция по группам строк
            sk = [0.0] * rows_launch
            for ty in range(rows_launch):
                acc = 0.0
                for r in range(rows_per):
                    row = ty * rows_per + r
                    if 0 <= row < HD:
                        acc += S[row][col] * k[row]
                        touched.add((row, col))
                sk[ty] = acc
            sk_val = sum(sk)
            d = (v[col] - sk_val) * beta
            o = 0.0
            for ty in range(rows_launch):
                for r in range(rows_per):
                    row = ty * rows_per + r
                    if 0 <= row < HD:
                        S[row][col] = S[row][col] + k[row] * d
                        o += S[row][col] * q[row]
            out[col] = o
    return out, S, touched


random.seed(7)
q = [random.uniform(-1, 1) for _ in range(HD)]
k = [random.uniform(-1, 1) for _ in range(HD)]
v = [random.uniform(-1, 1) for _ in range(HD)]
beta = 0.6
S0 = [[random.uniform(-1, 1) for _ in range(HD)] for _ in range(HD)]

# эталон: полное покрытие, однопоточная математика
ref_out, ref_S, _ = kernel(q, k, v, beta, S0, HD, 1, use_const=False)

def maxdiff(A, B):
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))

print("hd =", HD)
print()
print("=== COLS=32 (дефолт) ===")
for uc in (True, False):
    out, S, touched = kernel(q, k, v, beta, S0, 32, 4, use_const=uc)
    full = HD * HD
    tag = "старый (const DR_*)" if uc else "новый (blockDim)"
    print(f"  {tag:22s} покрытие {len(touched):6d}/{full} = {100*len(touched)/full:5.1f}%  "
          f"ΔS к эталону {maxdiff(S, ref_S):.3e}  Δout {max(range(HD), key=lambda i: abs(out[i]-ref_out[i])) and max(abs(out[i]-ref_out[i]) for i in range(HD)):.3e}")

print()
print("=== COLS=16 (тот самый эксперимент) ===")
for uc in (True, False):
    out, S, touched = kernel(q, k, v, beta, S0, 16, 4, use_const=uc)
    full = HD * HD
    tag = "старый (const DR_*)" if uc else "новый (blockDim)"
    ds = maxdiff(S, ref_S)
    do = max(abs(out[i] - ref_out[i]) for i in range(HD))
    print(f"  {tag:22s} покрытие {len(touched):6d}/{full} = {100*len(touched)/full:5.1f}%  "
          f"ΔS к эталону {ds:.3e}  Δout {do:.3e}"
          f"  {'<-- ЛОЖНОЕ УСКОРЕНИЕ' if len(touched) < full else ''}")

print()
print("=== ROWGRP=8 (латентный баг в delta_rule_kernel_batched_split) ===")
for uc in (True, False):
    out, S, touched = kernel(q, k, v, beta, S0, 32, 8, use_const=uc)
    full = HD * HD
    tag = "старый (const DR_*) " if uc else "новый (blockDim)   "
    ds = maxdiff(S, ref_S)
    print(f"  {tag:22s} покрытие {len(touched):6d}/{full} = {100*len(touched)/full:5.1f}%  ΔS {ds:.3e}"
          f"  {'<-- выход за границы головы' if uc else ''}")
