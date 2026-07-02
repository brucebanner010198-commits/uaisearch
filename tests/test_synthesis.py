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


def test_verify_citations_keeps_supported_sentences_and_drops_unsupported():
    from uaisearch.indexer import embed
    from uaisearch.synthesis import verify_citations

    source_text = "Bees communicate through a waggle dance that encodes direction and distance."
    chunk = Chunk(
        url="https://a.example", title="A", domain="a.example", chunk_text=source_text,
        embedding=embed(source_text), ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01",
    )
    raw_text = (
        "Bees communicate through a waggle dance that encodes direction and distance [1]. "
        "The moon landing happened in 1969 [1]. "
        "The Eiffel Tower is in Paris."
    )
    answer = verify_citations(raw_text, [chunk])
    assert "waggle dance" in answer.text
    assert "moon landing" not in answer.text
    assert "Eiffel Tower" not in answer.text
    assert answer.citations == [1]


async def test_synthesize_answer_returns_verified_answer():
    from uaisearch.indexer import embed
    from uaisearch.synthesis import synthesize_answer

    source_text = "Bees communicate through a waggle dance that encodes direction and distance."
    chunk = Chunk(
        url="https://a.example", title="A", domain="a.example", chunk_text=source_text,
        embedding=embed(source_text), ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # The mocked LLM emits one SUPPORTED sentence (backed by the chunk) and one
        # UNSUPPORTED fabrication. synthesize_answer is the "no bypass" point: it MUST
        # run verify_citations on the raw output, so the fabrication must be stripped.
        return httpx.Response(200, json={"choices": [{"message": {
            "content": (
                "Bees communicate through a waggle dance that encodes direction and distance [1]. "
                "The Eiffel Tower is in Paris."
            ),
        }}]})

    llm = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    answer = await synthesize_answer("how do bees communicate", [chunk], llm)
    assert "waggle dance" in answer.text        # supported -> kept
    assert "Eiffel Tower" not in answer.text     # unsupported -> stripped by verify_citations
    assert answer.citations == [1]
    assert answer.sources == [chunk]


async def test_generate_related_questions_parses_numbered_list():
    from uaisearch.synthesis import generate_related_questions

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"].startswith("Bearer ")
        body = _json.loads(request.content)
        assert body["model"] == "test-model"
        assert isinstance(body["messages"], list) and body["messages"]
        # Mixed numbering: "1." period style and "2)" paren style must both parse,
        # proving the regex ^\s*\d+[.)]\s* strips either delimiter.
        return httpx.Response(200, json={"choices": [{"message": {"content": (
            "1. Why do bees waggle dance?\n"
            "2) How far can bees communicate distance?\n"
            "3. Do all bee species dance?"
        )}}]})

    llm = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    questions = await generate_related_questions("how do bees communicate", "they dance", llm, count=3)
    assert questions == [
        "Why do bees waggle dance?",
        "How far can bees communicate distance?",
        "Do all bee species dance?",
    ]
