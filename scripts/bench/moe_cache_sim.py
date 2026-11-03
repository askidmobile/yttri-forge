#!/usr/bin/env python3
"""Симуляция VRAM-кэша экспертов по трассе маршрутизации."""
import collections, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moe_ids.csv"
per_layer = collections.defaultdict(list)
with open(path) as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        layer = int(parts[0])
        ids = [int(x) for x in parts[1:] if x]
        per_layer[layer].append(ids)

tokens = min(len(v) for v in per_layer.values())
print(f"layers={len(per_layer)} tokens={tokens}")

def lru_hits(seq, n):
    cache, order, hits = set(), collections.OrderedDict(), 0
    for ids in seq:
        for e in ids:
            if e in cache:
                hits += 1
                order.move_to_end(e)
            else:
                if len(cache) >= n:
                    old, _ = order.popitem(last=False)
                    cache.discard(old)
                cache.add(e)
                order[e] = None
    return hits / (len(seq) * len(seq[0]))

def static_hits(train, test, n):
    cnt = collections.Counter(e for ids in train for e in ids)
    hot = {e for e, _ in cnt.most_common(n)}
    hits = sum(1 for ids in test for e in ids if e in hot)
    return hits / (len(test) * len(test[0]))

for n in (8, 16, 24, 32, 48, 64):
    lru = sum(lru_hits(v, n) for v in per_layer.values()) / len(per_layer)
    st = sum(static_hits(v[: tokens // 2], v[tokens // 2:], n) for v in per_layer.values()) / len(per_layer)
    print(f"N={n:3d}  LRU={lru*100:5.1f}%  static={st*100:5.1f}%")
