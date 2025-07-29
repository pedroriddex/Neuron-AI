"""Wernicke comprehension model loader & inference (Sprint-0).

This module downloads a Transformer encoder model (~350 M params) from
🤗 Hugging Face, applies dynamic int8 quantization *optionally* and exposes a
simple `infer()` API returning a fixed-size embedding (d=768).

During CI/tests we default to a tiny model (`sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english`) to avoid large downloads. In production, set
`WERNICKE_MODEL_NAME` env-var to the desired checkpoint.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

__all__ = ["WernickeModel", "infer"]

_DEFAULT_MODEL = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"
_HIDDEN_SIZE = 768  # embedding dimension expected by downstream components


class WernickeModel:
    """Lazy-loaded singleton wrapper around a Transformer encoder."""

    def __init__(self) -> None:
        model_name = os.getenv("WERNICKE_MODEL_NAME", _DEFAULT_MODEL)
        quantize = os.getenv("WERNICKE_QUANTIZE", "1") == "1"

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

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
    def encode(self, text: str) -> torch.Tensor:  # shape (hidden,)
        """Return a single embedding vector for *text* (d=768)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            # Mean-pooling
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        else:  # some models expose `pooler_output`
            emb = outputs.pooler_output.squeeze(0)
        # Ensure fixed dimension
        if emb.shape[-1] != _HIDDEN_SIZE:
            emb = torch.nn.Linear(emb.shape[-1], _HIDDEN_SIZE)(emb)
        return emb


@lru_cache(maxsize=1)
def _get_wernicke() -> WernickeModel:
    return WernickeModel()


def infer(texts: List[str]) -> torch.Tensor:
    """Batch inference returning embeddings tensor (B, 768)."""
    model = _get_wernicke()
    embeddings = [model.encode(t) for t in texts]
    return torch.stack(embeddings)
