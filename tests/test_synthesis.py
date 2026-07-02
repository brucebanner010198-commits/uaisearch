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
