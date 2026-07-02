from fastapi import FastAPI
from fastapi.testclient import TestClient

from uaisearch.api import RateLimitMiddleware


def test_rate_limit_blocks_after_threshold_and_sets_headers():
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, requests_per_minute=2)

    @test_app.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(test_app)
    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    third = client.get("/ping")
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_rate_limit_buckets_by_forwarded_client_ip():
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

    @test_app.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(test_app)
    # First IP 203.0.113.1 — first request succeeds
    resp1 = client.get("/ping", headers={"X-Forwarded-For": "203.0.113.1"})
    assert resp1.status_code == 200
    # Second request from same IP is rate-limited
    resp2 = client.get("/ping", headers={"X-Forwarded-For": "203.0.113.1"})
    assert resp2.status_code == 429
    # Different IP 203.0.113.2 gets its own bucket — succeeds
    resp3 = client.get("/ping", headers={"X-Forwarded-For": "203.0.113.2"})
    assert resp3.status_code == 200


def test_rate_limit_keys_on_last_forwarded_hop_not_spoofable_first_hop():
    # simulate post-caddy header: attacker-chosen first hop + caddy-appended real IP.
    # rotating the first hop must NOT get a fresh bucket — the real (last) hop buckets them together.
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

    @test_app.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(test_app)
    r1 = client.get("/ping", headers={"X-Forwarded-For": "1.1.1.1, 203.0.113.9"})
    r2 = client.get("/ping", headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.9"})
    assert r1.status_code == 200
    assert r2.status_code == 429   # same real client 203.0.113.9 despite different spoofed first hop
