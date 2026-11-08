import pytest
from fastapi.testclient import TestClient

import server

client = TestClient(server.app)


def submission(**over):
    body = {
        "timestamp": 1700000000, "os": "Windows 11", "cpu": "Ryzen 5", "gpu": "NVIDIA GeForce RTX 4060",
        "results": [{"model": "llama3.1:8b", "parameter_size": "8.0B", "avg_tokens_per_sec": 40.0,
                     "avg_ttft_ms": 120.0, "runs": 3, "gpu_offload_pct": 100}],
    }
    body.update(over)
    return body


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    server._uploads.clear()


def post(body, ip="1.2.3.4"):
    return client.post("/api/submissions", json=body, headers={"x-forwarded-for": ip})


def test_version():
    assert client.get("/api/version").json()["app"] == "llm-bench"


def test_submit_and_read_back():
    r = post(submission(gpu="AMD Radeon RX 7600"))
    assert r.status_code == 201
    rows = client.get("/api/results", params={"gpu": "AMD Radeon RX 7600"}).json()
    assert any(row["model"] == "llama3.1:8b" for row in rows)


@pytest.mark.parametrize("bad", [
    {"results": []},
    {"gpu": ""},
    {"results": [{"model": "x", "avg_tokens_per_sec": -1, "avg_ttft_ms": 1, "runs": 1}]},
    {"results": [{"model": "x", "avg_tokens_per_sec": 99999, "avg_ttft_ms": 1, "runs": 1}]},
    {"results": [{"model": "x", "avg_tokens_per_sec": 10, "avg_ttft_ms": 1, "runs": 0}]},
    {"results": [{"model": "x", "avg_tokens_per_sec": 10, "avg_ttft_ms": 1, "runs": 1, "gpu_offload_pct": 150}]},
])
def test_rejects_impossible_submissions(bad):
    assert post(submission(**bad)).status_code == 422


def test_rate_limit_per_ip():
    for _ in range(server.UPLOAD_LIMIT):
        assert post(submission(), ip="9.9.9.9").status_code == 201
    assert post(submission(), ip="9.9.9.9").status_code == 429
    assert post(submission(), ip="8.8.8.8").status_code == 201


def test_predictions():
    r = client.get("/api/predictions", params={"gpu": "RTX 4060"})
    assert r.status_code == 200 and r.json()["models"]
    assert client.get("/api/predictions", params={"gpu": "nope"}).status_code == 404


def test_estimate_validates_inputs():
    ok = client.get("/api/estimate", params={"gpu": "RTX 4060", "model": "llama3.1:8b", "bits": 4, "ctx": 4096})
    assert ok.status_code == 200
    assert client.get("/api/estimate", params={"gpu": "RTX 4060", "model": "llama3.1:8b", "bits": 40}).status_code == 400
    assert client.get("/api/estimate", params={"gpu": "RTX 4060", "model": "nope"}).status_code == 404


def test_local_endpoints_are_off_on_the_public_site():
    assert client.post("/api/local/pull", json={"model": "llama3.2:3b"}).status_code == 403
    assert client.post("/api/local/share-all").status_code == 403


def test_homepage_served():
    r = client.get("/")
    assert r.status_code == 200 and "LLM Bench" in r.text
