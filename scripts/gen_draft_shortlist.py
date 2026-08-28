#!/usr/bin/env python3
"""Шортлист словаря для черновика MTP (QWEN36_MTP_VOCAB_SHORTLIST).

Словарь Qwen сгруппирован по языкам: кириллица лежит около id 200000, CJK — у
100000 и 248000. Поэтому глобальная отсечка QWEN36_MTP_VOCAB_TOP=N вырезает
целые письменности и обрушивает принятие (замер: TOP=32768 дал E[m] 1.455
против 1.685 на полном словаре).

Внутри же одного языкового блока порядок id — это порядок слияний BPE, то есть
приближение частоты. Отсюда конструкция: берём кириллицу целиком, а от
латиницы — только частотную голову.

Использование:
    python gen_draft_shortlist.py model.gguf shortlist.txt [размер_головы]
"""
import json, struct, sys

def read_tokens(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "не GGUF"
    _, _, n_kv = struct.unpack("<IQQ", f.read(20))
    SZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    FMT = {0:"<B", 1:"<b", 2:"<H", 3:"<h", 4:"<I", 5:"<i", 6:"<f", 7:"<?",
           10:"<Q", 11:"<q", 12:"<d"}
    def rd_str():
        return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
    def rd_val(t):
        if t == 8:
            return rd_str()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rd_val(et) for _ in range(n)]
        return struct.unpack(FMT[t], f.read(SZ[t]))[0]
    for _ in range(n_kv):
        k = rd_str()
        v = rd_val(struct.unpack("<I", f.read(4))[0])
        if k.endswith("tokenizer.ggml.tokens"):
            return v
    raise SystemExit("в GGUF нет tokenizer.ggml.tokens")

def byte_decoder():
    """Обратная таблица byte-level BPE (как в GPT-2/Qwen)."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    head = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
    toks, dec_map = read_tokens(sys.argv[1]), byte_decoder()

    def decode(t):
        try:
            return bytes(dec_map[c] for c in t).decode("utf-8", "replace")
        except (KeyError, UnicodeDecodeError):
            return None

    def is_ours(ch):
        o = ord(ch)
        return o < 0x80 or 0x0400 <= o < 0x0500 or 0x2010 <= o < 0x2060 or o == 0xFFFD

    cyrillic, latin = [], []
    for i, t in enumerate(toks):
        s = decode(t)
        if s is None or not all(is_ours(c) for c in s):
            continue          # байтовые фолбэки и чужие письменности
        (cyrillic if any(0x0400 <= ord(c) < 0x0500 for c in s) else latin).append(i)

    keep = sorted(set(cyrillic) | set(latin[:head]))
    with open(sys.argv[2], "w") as f:
        f.write("\n".join(map(str, keep)) + "\n")
    print(f"словарь {len(toks)} → шортлист {len(keep)} "
          f"(кириллица {len(cyrillic)}, латиница {min(head, len(latin))})")

if __name__ == "__main__":
    main()
