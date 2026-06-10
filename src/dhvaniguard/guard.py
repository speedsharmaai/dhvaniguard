"""DhvaniGuard — code-mixed-aware prompt-injection guardrail for voice agents."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["safe", "injection"]

# Published weights live on the Hugging Face Hub; override with DHVANIGUARD_MODEL
# (e.g. a local path) for development before the Hub upload.
DEFAULT_MODEL = os.environ.get("DHVANIGUARD_MODEL", "speedsharma/dhvaniguard-v0")


@dataclass
class GuardResult:
    verdict: Verdict
    score: float
    """confidence in `verdict`, 0-1."""
    latency_ms: float
    blocked: bool
    """True when verdict is 'injection' and score >= threshold."""


class DhvaniGuard:
    """A small multilingual classifier that flags prompt-injection / jailbreak
    attempts in code-mixed Indian speech (Hinglish, romanized Indic, Devanagari)
    without blocking genuine Hindi/Hinglish customers the way English-only
    guardrails do.

    Built to sit on a voice agent's live transcript stream. Lazy-loads the model
    on first call; subsequent calls are a single forward pass (CPU-friendly).
    """

    def __init__(self, model: str = DEFAULT_MODEL, threshold: float = 0.5, device: str | None = None):
        self.model_name = model
        self.threshold = threshold
        self._device = device
        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        if self._device:
            self._model.to(self._device)
        self._torch = torch

    def check(self, text: str) -> GuardResult:
        """Classify a single transcript utterance."""
        self._load()
        torch = self._torch
        t0 = time.perf_counter()
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=64)
        if self._device:
            enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        label = self._model.config.id2label[idx].upper()
        is_injection = "INJ" in label
        score = float(probs[idx])
        latency = (time.perf_counter() - t0) * 1000
        # block only when it's an injection AND we're confident enough
        inj_score = float(probs[1]) if "INJ" in self._model.config.id2label[1].upper() else 1 - score
        blocked = is_injection and inj_score >= self.threshold
        return GuardResult(
            verdict="injection" if is_injection else "safe",
            score=score,
            latency_ms=round(latency, 2),
            blocked=blocked,
        )

    def is_safe(self, text: str) -> bool:
        """Convenience: True if the utterance is safe to pass to the agent."""
        return not self.check(text).blocked
