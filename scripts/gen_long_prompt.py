#!/usr/bin/env python3
"""Генератор длинных промптов для A/B-тестов (детерминированный).

Пример: python3 scripts/gen_long_prompt.py alpha 130900 /tmp/p128k.txt
       python3 scripts/gen_long_prompt.py realistic 130900 /tmp/real130k.txt
Печатает sha256 и фактическое число слов (≈ токенов у BPE Qwen для alpha).
"""
import argparse
import hashlib
import pathlib

SUFFIX = "\nIgnore the repeated text. Count from 1 to 120 separated by commas. Do not explain."
PARAGRAPH = (
    "В инженерной практике длинный контекст нужен для чтения спецификаций, логов и переписки. "
    "Модель должна находить в тексте редкие детали, не теряя общий смысл документа. "
    "Проверка качества на длинных входах требует воспроизводимых промптов и фиксированных настроек. "
)


def build(mode: str, n: int) -> str:
    suffix_words = len(SUFFIX.split())
    body_n = max(1, n - suffix_words)
    if mode == "alpha":
        body = ("alpha " * body_n).rstrip()
    else:
        parts = []
        total = 0
        i = 0
        while total < body_n:
            i += 1
            piece = f"[{i}] " + PARAGRAPH
            parts.append(piece)
            total += len(piece.split())
        body = " ".join(parts)
    return body + SUFFIX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["alpha", "realistic"])
    ap.add_argument("words", type=int)
    ap.add_argument("out", type=pathlib.Path)
    args = ap.parse_args()
    text = build(args.mode, args.words)
    args.out.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f"{args.out} words={len(text.split())} sha256={digest}")


if __name__ == "__main__":
    main()
