#!/usr/bin/env python3
"""
Fine-tune DhvaniGuard: a code-mixed-aware prompt-injection classifier.

Base: google/muril-base-cased (BERT pretrained on 17 Indian languages + their
transliterations — i.e. it natively understands Hinglish/romanized Indic, the
exact thing English deberta lacks).

Trains a 2-class head (SAFE=0 / INJECTION=1) on the balanced en/hinglish/hi set,
then saves the model so the eval script can run it on the held-out 36-case set.
CPU-friendly: small model, few epochs, short sequences.
"""
import json
import os
import random
import numpy as np

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

HERE = os.path.dirname(__file__)
# Train on whatever data files exist: the AI-generated multi-language set is the
# main source; the small hand-authored set is merged in if present.
DATA_FILES = [
    os.path.join(HERE, "train_synthetic.jsonl"),
    os.path.join(HERE, "train.jsonl"),
]
OUTDIR = os.path.join(HERE, "..", "model", "dhvaniguard-v0")
BASE = "google/muril-base-cased"


def main():
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    torch.manual_seed(SEED)

    rows = []
    seen = set()
    for path in DATA_FILES:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["text"].lower(), r["label"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    random.shuffle(rows)
    # hold out 15% as an internal dev split (separate from the 36-case eval set)
    n_dev = max(20, int(0.15 * len(rows)))
    dev, tr = rows[:n_dev], rows[n_dev:]
    print(f"loaded {len(rows)} rows | train={len(tr)} dev={len(dev)}")

    tok = AutoTokenizer.from_pretrained(BASE)

    class DS(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            r = self.data[i]
            enc = tok(r["text"], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            return {"input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "labels": torch.tensor(r["label"])}

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, id2label={0: "SAFE", 1: "INJECTION"}, label2id={"SAFE": 0, "INJECTION": 1})

    def metrics(p):
        import scipy.special as sp
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = (preds == labels).mean()
        # mean confidence in the predicted class — we want this high (>=0.95)
        probs = sp.softmax(p.predictions, axis=1)
        conf = float(probs.max(axis=1).mean())
        return {"accuracy": float(acc), "mean_confidence": round(conf, 3)}

    args = TrainingArguments(
        output_dir=os.path.join(HERE, "..", "model", "_ckpt"),
        # more epochs + a higher peak LR with warmup + weight decay -> the model
        # actually commits to a decision instead of sitting at ~0.57 confidence.
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=SEED,
    )
    trainer = Trainer(model=model, args=args, train_dataset=DS(tr),
                      eval_dataset=DS(dev), compute_metrics=metrics)
    trainer.train()
    print("dev eval:", trainer.evaluate())

    os.makedirs(OUTDIR, exist_ok=True)
    model.save_pretrained(OUTDIR)
    tok.save_pretrained(OUTDIR)
    print(f"saved DhvaniGuard model -> {OUTDIR}")


if __name__ == "__main__":
    main()
