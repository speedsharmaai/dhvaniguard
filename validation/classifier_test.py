#!/usr/bin/env python3
"""
DhvaniGuard Week-0 — the REAL test.

The first experiment used a big multilingual LLM as the guardrail; it caught
Hinglish fine (thesis looked weak). But production voice agents need <50ms, so
they use small DEDICATED classifiers (LLM Guard's deberta-v3 prompt-injection
model, Meta PromptGuard, etc.) — and those are trained on English. This script
runs the ACTUAL small English-trained classifier over the same en/hinglish/hi
attack set and measures whether it misses the non-English forms.

Model: protectai/deberta-v3-base-prompt-injection-v2 (the model LLM Guard ships).
Label INJECTION == the attack class; SAFE otherwise.
"""
import json
import sys
import os

RESULTS = os.path.join(os.path.dirname(__file__), "results_classifier.jsonl")
ATTACKS = os.path.join(os.path.dirname(__file__), "attacks.jsonl")
MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"loading {MODEL} (first run downloads ~700MB)...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    id2label = model.config.id2label  # typically {0:'SAFE',1:'INJECTION'}

    def classify(text):
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        idx = int(torch.argmax(logits, dim=-1))
        label = id2label[idx].upper()
        score = float(torch.softmax(logits, dim=-1)[0][idx])
        return ("INJECTION" if "INJ" in label else "SAFE"), score

    rows = []
    with open(ATTACKS) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    out = open(RESULTS, "w")
    for i, r in enumerate(rows, 1):
        verdict, score = classify(r["text"])
        rec = {"id": r["id"], "family": r["family"], "lang": r["lang"],
               "is_attack": r["is_attack"], "verdict": verdict, "score": round(score, 3)}
        out.write(json.dumps(rec) + "\n")
        mark = "OK " if (verdict == "INJECTION") == r["is_attack"] else "XX "
        print(f"  [{i:2d}/36] {mark}{r['lang']:<9}{r['family']:<22} atk={str(r['is_attack']):<5} -> {verdict} ({score:.2f})")
    out.close()
    print(f"\nWrote {RESULTS}. Analyze: python analyze.py {RESULTS}")


if __name__ == "__main__":
    main()
