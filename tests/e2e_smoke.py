"""E2E smoke test: FastAPI TestClient simulates `/chat` request."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_chat_endpoint():
    payload = {"prompt": "Hello Neuron!"}
    r = client.post("/chat", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "response" in data and isinstance(data["response"], str)
    assert "expert" in data
    assert "latency_ms" in data
