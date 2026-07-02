import httpx

from uaisearch.synthesis import LLMClient


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
