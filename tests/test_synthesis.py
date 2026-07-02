import httpx

from uaisearch.models import Chunk
from uaisearch.synthesis import LLMClient, build_messages


async def test_chat_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test"  # api_key passed to the client
        body = _json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "what is the answer"}]
        assert body["temperature"] == 0.2  # the default from chat()
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    client = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await client.chat([{"role": "user", "content": "what is the answer"}])
    assert result == "42"


SSE_BODY = (
    'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
    'data: [DONE]\n\n'
)


async def test_chat_stream_yields_content_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test"
        body = _json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["temperature"] == 0.2
        assert body["stream"] is True
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    client = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tokens = [t async for t in client.chat_stream([{"role": "user", "content": "hi"}])]
    assert "".join(tokens) == "Hello world"


SSE_BODY_WITH_EDGE_CASES = (
    'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    'data: {"choices": []}\n\n'
    'data: {not json\n\n'
    'data: {"choices": [{"delta": {}}]}\n\n'
    'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
    'data: [DONE]\n\n'
)


async def test_chat_stream_skips_malformed_and_empty_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test"
        return httpx.Response(
            200, text=SSE_BODY_WITH_EDGE_CASES,
            headers={"content-type": "text/event-stream"},
        )

    client = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tokens = [t async for t in client.chat_stream([{"role": "user", "content": "hi"}])]
    assert tokens == ["Hello", " world"]


def test_build_messages_includes_numbered_sources_and_question():
    chunk = Chunk(
        url="https://a.example", title="A", domain="a.example",
        chunk_text="bees need hives", embedding=[], ad_ratio=0.0,
        domain_quality=1.0, crawl_date="2026-07-01",
    )
    messages = build_messages("how do bees live", [chunk])
    assert messages[0]["role"] == "system"
    assert "[1] bees need hives (source: https://a.example)" in messages[1]["content"]
    assert "how do bees live" in messages[1]["content"]


def test_system_prompt_treats_sources_as_untrusted_data():
    from uaisearch.synthesis import SYSTEM_PROMPT
    low = SYSTEM_PROMPT.lower()
    assert "untrusted" in low
    assert "instruction" in low  # must tell the model not to follow instructions inside sources
