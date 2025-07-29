"""Locust performance test for Neuron `/chat` endpoint.

Run:
    locust -f bench/locustfile.py --headless -u 100 -r 10 -t 30s --host=http://localhost:8000

This will issue up to 100 concurrent users, ramping 10 users/s for 30 seconds.
Metrics:
    - Locust summary outputs P95 latency.
    - Prometheus exporter is enabled via `/metrics` if app run with `prometheus_multiproc_dir`.
"""
from locust import HttpUser, task, between
import random

PROMPTS = [
    "Hello, how are you?",
    "Tell me a joke about computers.",
    "Summarize the importance of modular AI architectures.",
]


class ChatUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task
    def chat(self):
        prompt = random.choice(PROMPTS)
        self.client.post("/chat", json={"prompt": prompt})
