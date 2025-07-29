"""Broca generation model loader & response generator (Sprint-0).

Uses a causal-LM (~450 M params target). For CI we default to a very small
model to avoid heavy downloads (`sshleifer/tiny-gpt2`).

API:
    generate(prompt: str, max_tokens: int = 50) -> str
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["BrocaModel", "generate"]

_DEFAULT_MODEL = "sshleifer/tiny-gpt2"
_MAX_NEW_TOKENS = 50


class BrocaModel:
    """Lazy-loaded singleton wrapper around a causal LM."""

    def __init__(self) -> None:
        model_name = os.getenv("BROCA_MODEL_NAME", _DEFAULT_MODEL)
        quantize = os.getenv("BROCA_QUANTIZE", "1") == "1"

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if quantize and torch.backends.quantized.engine != "none" and torch.get_default_dtype() == torch.float32:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        # Optional TorchCompile for speed
        if os.getenv("TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, backend="inductor")
            except Exception:  # noqa: BLE001
                pass
        self.model = model.eval()
        self._load_ms = (time.perf_counter() - t0) * 1000

    @torch.inference_mode()
    def generate(self, prompt: str, max_tokens: int = _MAX_NEW_TOKENS) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            **tokens,
            max_new_tokens=min(max_tokens, _MAX_NEW_TOKENS),
            do_sample=True,
            top_p=0.9,
        )
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated[len(prompt) :].strip()


@lru_cache(maxsize=1)
def _get_broca() -> BrocaModel:
    return BrocaModel()


def generate(prompt: str, max_tokens: int = _MAX_NEW_TOKENS) -> str:
    """Convenience wrapper around singleton Broca instance."""
    return _get_broca().generate(prompt, max_tokens)
