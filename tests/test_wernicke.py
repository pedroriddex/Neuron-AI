import time

import torch

from wernicke import infer, WernickeModel


def test_infer_latency_and_shape():
    texts = ["hello", "world"]
    # Warm-up (model download & first pass)
    _ = infer(["warm-up"])
    start = time.perf_counter_ns()
    embeddings = infer(texts)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    # Shape (B, 768)
    assert embeddings.shape == (2, 768), embeddings.shape
    assert embeddings.dtype in (torch.float32, torch.float16, torch.int8)

    # Latency < 100 ms after warm-up on CPU
    assert elapsed_ms < 100, f"Latency too high: {elapsed_ms:.2f} ms"
