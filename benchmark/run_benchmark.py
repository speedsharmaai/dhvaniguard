#!/usr/bin/env python3
"""
Reproduce the README comparison: DhvaniGuard vs the English deberta-v3 prompt-
injection classifier (the model LLM Guard ships) on the same held-out set of
attacks + benign requests in English / Hinglish / Hindi.

Run:
    DHVANIGUARD_MODEL=/path/to/dhvaniguard-v0 python run_benchmark.py
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "eval_set.jsonl")
ENGLISH_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def score_table(name, classify, rows):
    det = defaultdict(lambda: [0, 0])
    fp = defaultdict(lambda: [0, 0])
    for r in rows:
        flagged = classify(r["text"]) == "injection"
        bucket = det if r["is_attack"] else fp
        bucket[r["lang"]][1] += 1
        bucket[r["lang"]][0] += flagged
    print(f"\n{name}")
    print(f"  {'lang':<10}{'attack detect':>16}{'benign FP':>14}")
    for lang in ["en", "hinglish", "hi"]:
        d_f, d_t = det[lang]; f_f, f_t = fp[lang]
        dp = d_f / d_t * 100 if d_t else 0
        fpp = f_f / f_t * 100 if f_t else 0
        print(f"  {lang:<10}{f'{dp:.0f}%':>16}{f'{fpp:.0f}%':>14}")


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from dhvaniguard import DhvaniGuard

    rows = [json.loads(l) for l in open(EVAL) if l.strip()]

    # English baseline
    tok = AutoTokenizer.from_pretrained(ENGLISH_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(ENGLISH_MODEL)
    mdl.eval()

    def english(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            i = int(torch.argmax(mdl(**enc).logits, dim=-1))
        return "injection" if "INJ" in mdl.config.id2label[i].upper() else "safe"

    guard = DhvaniGuard()
    print("=" * 44)
    print(" DhvaniGuard vs English deberta-v3 (LLM Guard)")
    print("=" * 44)
    score_table("English deberta-v3 (LLM Guard):", english, rows)
    score_table("DhvaniGuard:", lambda t: guard.check(t).verdict, rows)


if __name__ == "__main__":
    main()
