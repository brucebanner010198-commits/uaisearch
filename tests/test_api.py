import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from uaisearch.api import (
    _CONVERSATIONS,
    RateLimitMiddleware,
    _client_dependency,
    _get_llm_client,
    app,
)
from uaisearch.models import Answer, Chunk


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


def test_search_endpoint_returns_results_without_internal_ranking_fields():
    fake_chunk = Chunk(
        url="https://a.example/1", title="A", domain="a.example",
        chunk_text="bees need hives and regular inspection", embedding=[],
        ad_ratio=0.4, domain_quality=0.6, crawl_date="2026-07-01",
    )
    app.dependency_overrides[_client_dependency] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", return_value=[fake_chunk]):
            client = TestClient(app)
            response = client.get("/api/v1/search", params={"q": "beekeeping"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["url"] == "https://a.example/1"
    assert body["results"][0]["snippet"].startswith("bees need hives")
    assert "ad_ratio" not in body["results"][0]
    assert "domain_quality" not in body["results"][0]


def _parse_sse_events(body: str) -> list[dict]:
    return [
        json.loads(chunk[len("data: "):])
        for chunk in body.strip().split("\n\n")
        if chunk.startswith("data: ")
    ]


def test_answer_endpoint_streams_verified_tokens_then_final_event():
    fake_chunk = Chunk(
        url="https://a.example/1", title="A", domain="a.example",
        chunk_text="bees need hives", embedding=[], ad_ratio=0.0,
        domain_quality=1.0, crawl_date="2026-07-01",
    )
    fake_answer = Answer(text="Bees need hives [1].", citations=[1], sources=[fake_chunk])

    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", return_value=[fake_chunk]), \
             patch("uaisearch.api.synthesize_answer", return_value=fake_answer), \
             patch("uaisearch.api.generate_related_questions",
                   return_value=["Why do bees need hives?"]):
            client = TestClient(app)
            response = client.post("/api/v1/answer", params={"query": "why hives"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    streamed_text = "".join(e["token"] for e in events if "token" in e)
    assert streamed_text.strip() == "Bees need hives [1]."
    final = next(e for e in events if e.get("done"))
    assert final["citations"] == [1]
    assert final["sources"] == ["https://a.example/1"]
    assert final["related_questions"] == ["Why do bees need hives?"]
    # internal ranking signals must never leak into the SSE final event
    final_json = json.dumps(final)
    assert "ad_ratio" not in final_json
    assert "domain_quality" not in final_json


def test_answer_endpoint_defers_related_questions_until_after_answer_tokens():
    # SWOT latency fix: generate_related_questions (a second LLM round-trip) must run
    # AFTER the answer tokens have been produced, so it never delays time-to-first-token.
    #
    # Output ORDER alone can't prove this: a regression that awaits related questions
    # BEFORE the token loop still emits tokens-then-final and would pass the test above.
    # We must assert call TIMING. starlette's TestClient buffers the whole streaming
    # body before yielding any line, so mid-stream call_count inspection can't observe
    # the deferral either. Instead we record the order of two server-side events into a
    # shared list and assert the token loop ran first:
    #   - "answer_tokens": the token loop begins iterating result.text.split(" ")
    #   - "related": generate_related_questions is invoked
    # Correct (deferred) code produces ["answer_tokens", "related"]; moving the
    # related-questions call above the token loop flips the order and fails this test.
    order: list[str] = []

    class _RecordingText(str):
        def split(self, *args, **kwargs):
            order.append("answer_tokens")
            return str(self).split(*args, **kwargs)

    fake_chunk = Chunk(
        url="https://a.example/1", title="A", domain="a.example",
        chunk_text="bees need hives", embedding=[], ad_ratio=0.0,
        domain_quality=1.0, crawl_date="2026-07-01",
    )
    fake_answer = Answer(
        text=_RecordingText("Bees need hives [1]."), citations=[1], sources=[fake_chunk],
    )

    def _record_related(*args, **kwargs):
        order.append("related")
        return ["Why do bees need hives?"]

    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", return_value=[fake_chunk]), \
             patch("uaisearch.api.synthesize_answer", return_value=fake_answer), \
             patch("uaisearch.api.generate_related_questions",
                   side_effect=_record_related) as mock_related:
            client = TestClient(app)
            response = client.post("/api/v1/answer", params={"query": "why hives"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    # related questions ran exactly once, and only AFTER the answer tokens
    assert mock_related.call_count == 1
    assert order == ["answer_tokens", "related"]


def test_answer_endpoint_uses_conversation_history_for_follow_up_queries():
    fake_chunk = Chunk(
        url="https://a.example/1", title="A", domain="a.example",
        chunk_text="bees need hives", embedding=[], ad_ratio=0.0,
        domain_quality=1.0, crawl_date="2026-07-01",
    )
    fake_answer = Answer(text="Bees need hives [1].", citations=[1], sources=[fake_chunk])
    captured_queries = []

    def fake_retrieve(client, query, limit=8, **kwargs):
        captured_queries.append(query)
        return [fake_chunk]

    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", side_effect=fake_retrieve), \
             patch("uaisearch.api.synthesize_answer", return_value=fake_answer), \
             patch("uaisearch.api.generate_related_questions", return_value=[]):
            client = TestClient(app)
            client.post("/api/v1/answer",
                        params={"query": "what do bees need", "conversation_id": "conv-1"})
            client.post("/api/v1/answer",
                        params={"query": "how many hives", "conversation_id": "conv-1"})
    finally:
        app.dependency_overrides.clear()

    assert "what do bees need" in captured_queries[0]
    assert "Q: what do bees need" in captured_queries[1]
    assert "how many hives" in captured_queries[1]


def test_rate_limit_ignores_unvalidated_api_key_rotation():
    # No key issuance system exists, so rotating random X-Api-Key values must NOT
    # mint fresh rate-limit buckets — the real (last) XFF hop buckets them together.
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, requests_per_minute=1)

    @test_app.get("/ping")
    async def ping():
        return {"ok": True}

    client = TestClient(test_app)
    r1 = client.get("/ping", headers={
        "X-Forwarded-For": "1.1.1.1, 203.0.113.9", "X-Api-Key": "key-a",
    })
    r2 = client.get("/ping", headers={
        "X-Forwarded-For": "1.1.1.1, 203.0.113.9", "X-Api-Key": "key-b",
    })
    assert r1.status_code == 200
    assert r2.status_code == 429


def test_answer_endpoint_rejects_empty_and_oversized_queries():
    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        client = TestClient(app)
        empty = client.post("/api/v1/answer", params={"query": ""})
        oversized = client.post("/api/v1/answer", params={"query": "x" * 501})
    finally:
        app.dependency_overrides.clear()

    assert empty.status_code == 422
    assert oversized.status_code == 422


def test_answer_endpoint_bounds_conversation_history_per_id():
    fake_chunk = Chunk(
        url="https://a.example/1", title="A", domain="a.example",
        chunk_text="bees need hives", embedding=[], ad_ratio=0.0,
        domain_quality=1.0, crawl_date="2026-07-01",
    )
    fake_answer = Answer(text="Bees need hives [1].", citations=[1], sources=[fake_chunk])

    _CONVERSATIONS.clear()
    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", return_value=[fake_chunk]), \
             patch("uaisearch.api.synthesize_answer", return_value=fake_answer), \
             patch("uaisearch.api.generate_related_questions", return_value=[]):
            client = TestClient(app)
            for i in range(5):
                client.post("/api/v1/answer",
                            params={"query": f"question {i}", "conversation_id": "conv-bound"})
        history = list(_CONVERSATIONS["conv-bound"])
    finally:
        app.dependency_overrides.clear()
        _CONVERSATIONS.clear()

    assert len(history) == 3
    assert [q for q, _ in history] == ["question 2", "question 3", "question 4"]


def test_answer_endpoint_streams_explicit_not_enough_information_answer():
    no_info = Answer(text="not enough information", citations=[], sources=[])

    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", return_value=[]), \
             patch("uaisearch.api.synthesize_answer", return_value=no_info), \
             patch("uaisearch.api.generate_related_questions", return_value=[]):
            client = TestClient(app)
            response = client.post("/api/v1/answer", params={"query": "unknown topic"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    streamed_text = "".join(e["token"] for e in events if "token" in e)
    assert streamed_text.strip() == "not enough information"
    final = next(e for e in events if e.get("done"))
    assert final["sources"] == []


def test_answer_endpoint_still_sends_final_event_when_related_questions_fail():
    fake_chunk = Chunk(
        url="https://a.example/1", title="A", domain="a.example",
        chunk_text="bees need hives", embedding=[], ad_ratio=0.0,
        domain_quality=1.0, crawl_date="2026-07-01",
    )
    fake_answer = Answer(text="Bees need hives [1].", citations=[1], sources=[fake_chunk])

    app.dependency_overrides[_client_dependency] = lambda: None
    app.dependency_overrides[_get_llm_client] = lambda: None
    try:
        with patch("uaisearch.api.retrieve_and_rerank", return_value=[fake_chunk]), \
             patch("uaisearch.api.synthesize_answer", return_value=fake_answer), \
             patch("uaisearch.api.generate_related_questions",
                   side_effect=RuntimeError("boom")):
            client = TestClient(app)
            response = client.post("/api/v1/answer", params={"query": "why hives"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    streamed_text = "".join(e["token"] for e in events if "token" in e)
    assert "Bees need hives" in streamed_text
    final = next(e for e in events if e.get("done"))
    assert final["done"] is True
    assert final["related_questions"] == []
    assert final["sources"] == ["https://a.example/1"]
