#!/usr/bin/env python3
"""
Evaluate DhvaniGuard on the SAME held-out 36-case set the English deberta failed,
and print the same attack-detection + false-positive table for an apples-to-apples
comparison. The 36 cases use different wordings than the training set.
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "..", "model", "dhvaniguard-v0")
EVAL = os.path.join(HERE, "..", "validation", "attacks.jsonl")


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    id2label = model.config.id2label

    def classify(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**enc).logits
        idx = int(torch.argmax(logits, dim=-1))
        return "INJECTION" if "INJ" in id2label[idx].upper() else "SAFE"

    rows = [json.loads(l) for l in open(EVAL) if l.strip()]
    det = defaultdict(lambda: [0, 0])
    fp = defaultdict(lambda: [0, 0])
    for r in rows:
        v = classify(r["text"])
        flagged = v == "INJECTION"
        if r["is_attack"]:
            det[r["lang"]][1] += 1
            det[r["lang"]][0] += flagged
        else:
            fp[r["lang"]][1] += 1
            fp[r["lang"]][0] += flagged

    order = ["en", "hinglish", "hi"]
    label = {"en": "English", "hinglish": "Hinglish (roman)", "hi": "Hindi (Devanagari)"}
    print("=" * 62)
    print(" DhvaniGuard (MuRIL fine-tune) on the held-out 36-case set")
    print("=" * 62)
    print(f"\n {'language':<22}{'attack detection':>18}{'false positives':>18}")
    print(" " + "-" * 58)
    for lang in order:
        d_f, d_t = det[lang]; f_f, f_t = fp[lang]
        dp = d_f / d_t * 100 if d_t else 0
        fpp = f_f / f_t * 100 if f_t else 0
        print(f" {label[lang]:<22}{f'{d_f}/{d_t} ({dp:.0f}%)':>18}{f'{f_f}/{f_t} ({fpp:.0f}%)':>18}")

    print("\n COMPARISON vs English deberta-v3 (LLM Guard):")
    print("   English deberta:  Hinglish benign FP = 100%, Hindi benign FP = 100%")
    hin_fp = fp['hinglish'][0] / fp['hinglish'][1] * 100 if fp['hinglish'][1] else 0
    hi_fp = fp['hi'][0] / fp['hi'][1] * 100 if fp['hi'][1] else 0
    print(f"   DhvaniGuard:      Hinglish benign FP = {hin_fp:.0f}%, Hindi benign FP = {hi_fp:.0f}%")


if __name__ == "__main__":
    main()
