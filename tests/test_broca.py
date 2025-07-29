import time

from broca import generate


def test_generate_latency_and_length():
    prompt = "Hello, how are you?"
    _ = generate("warm-up", max_tokens=10)
    start = time.perf_counter_ns()
    response = generate(prompt, max_tokens=50)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    assert isinstance(response, str)
    # length <= 60 tokens approx (simple char check)
    assert len(response.split()) <= 60
    assert elapsed_ms < 300, f"Generation latency too high: {elapsed_ms:.2f} ms"
