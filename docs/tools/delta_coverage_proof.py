#!/usr/bin/env python3
"""Доказательство класса бага «константа против blockDim» в splitc/splitr.

Моделируем, какие (row, col) пишет ядро состояния [hd x hd]:
  const-вариант:  col = blockIdx.z * DR_COLS + threadIdx.x   (DR_COLS=32)
  blockDim-вариант: col = blockIdx.z * blockDim.x + threadIdx.x
и какие строки трогает группа threadIdx.y при rowgrp (const) против blockDim.y.
Проверяем полноту покрытия всех hd*hd элементов состояния.
"""

def cover_splitc(hd, cols_launch, rows_launch, dr_cols=32, dr_rowgrp=4, use_const=True):
    cols = dr_cols if use_const else cols_launch
    rows = dr_rowgrp if use_const else rows_launch
    touched = set()
    nblocks_z = hd // cols_launch if cols_launch else 0
    for bz in range(nblocks_z):
        for tx in range(cols_launch):
            col = bz * cols + tx
            if col >= hd:
                continue
            for ty in range(rows_launch):
                rows_per = hd // rows
                if rows_per == 0:
                    continue
                row0 = ty * rows_per
                for r in range(rows_per):
                    row = row0 + r
                    if row < hd:
                        touched.add((row, col))
    return touched


def report(hd, cols_launch, rows_launch, use_const, label):
    t = cover_splitc(hd, cols_launch, rows_launch, use_const=use_const)
    full = hd * hd
    print(f"{label:44s} покрыто {len(t):6d}/{full:6d} = {100*len(t)/full:5.1f}%"
          f"  {'ПОЛНОЕ' if len(t)==full else 'НЕПОЛНОЕ <-- баг'}")


HD = 128
print("=== splitc: hd=128, launch blockDim=(COLS, ROWGRP) ===")
report(HD, 32, 4, True,  "COLS=32 (дефолт), const DR_COLS=32")
report(HD, 32, 4, False, "COLS=32 (дефолт), blockDim.x=32")
report(HD, 16, 4, True,  "COLS=16, const DR_COLS=32  <-- прежний баг")
report(HD, 16, 4, False, "COLS=16, blockDim.x=16  (после починки)")
report(HD,  8, 4, False, "COLS=8,  blockDim.x=8")
print()
print("=== splitr/split: то же для строк (DR_ROWGRP=4 против blockDim.y) ===")
report(HD, 32,  8, True,  "ROWGRP=8, const DR_ROWGRP=4 <-- латентный баг")
report(HD, 32,  8, False, "ROWGRP=8, blockDim.y=8")
report(HD, 32,  4, False, "ROWGRP=4 (дефолт), blockDim.y=4")
print()
print("Вывод: при дефолтах (COLS=32, ROWGRP=4) правка численно ничего не меняет,")
print("а при COLS=16 прежний код покрывал половину столбцов и потому 'ускорялся'.")
