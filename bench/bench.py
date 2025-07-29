"""Simple async benchmark harness for Neuron `/chat` endpoint.

Runs a local uvicorn server (api.main:app) in a subprocess, then fires
concurrent chat requests (100 QPS) for a configurable duration and computes
P95 latency. Fails (exit 1) if P95 ≥ 100 ms, as required in Sprint-0.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = os.getenv("NEURON_HOST", "http://127.0.0.1:8000")
DURATION_S = int(os.getenv("NEURON_BENCH_DURATION", "30"))
QPS = int(os.getenv("NEURON_BENCH_QPS", "100"))
PROMPT = "Hello, Neuron!"
TARGET_P95_MS = 100
REPORT_PATH = PROJECT_ROOT / "bench" / "report.json"


async def worker(client: httpx.AsyncClient, latencies: List[float], end_t: float):
    """Send requests at max possible rate; use shared list to record latencies."""
    while time.perf_counter() < end_t:
        start = time.perf_counter()
        try:
            r = await client.post("/chat", json={"prompt": PROMPT}, timeout=30)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            print(f"Request failed: {exc}", file=sys.stderr)
        finally:
            latencies.append((time.perf_counter() - start) * 1000)
        # crude pacing to approx QPS per task handled globally elsewhere


async def run_load_test():
    concurrency = QPS  # 1 req per task per second average
    latencies: List[float] = []
    async with httpx.AsyncClient(base_url=HOST) as client:
        end_t = time.perf_counter() + DURATION_S
        tasks = [worker(client, latencies, end_t) for _ in range(concurrency)]
        await asyncio.gather(*tasks)
    return latencies


def main() -> None:  # noqa: C901
    server_proc: subprocess.Popen | None = None
    # Launch uvicorn server if HOST seems local default
    if HOST.startswith("http://127.0.0.1") or HOST.startswith("http://localhost"):
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
        server_cmd = [sys.executable, "-m", "uvicorn", "api.main:app"]
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        # wait for server to boot
        time.sleep(3)
    try:
        latencies = asyncio.run(run_load_test())
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

    if not latencies:
        print("No latencies recorded", file=sys.stderr)
        sys.exit(1)

    p95 = statistics.quantiles(latencies, n=100)[94]
    result = {
        "p95_ms": p95,
        "count": len(latencies),
        "target_ms": TARGET_P95_MS,
        "pass": p95 < TARGET_P95_MS,
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if p95 >= TARGET_P95_MS:
        print("P95 latency target not met", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
