# Answer Synthesis & Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a cited answer from retrieved chunks using any OpenAI-compatible LLM backend, then verify every citation against its source chunk so unsupported claims are dropped before a user ever sees them.

**Architecture:** `LLMClient` is a thin async wrapper over the OpenAI-compatible `/chat/completions` endpoint (works unmodified against Claude, GPT, Gemini, or a self-hosted vLLM/Ollama server — swapping backends is a `base_url`/`model` config change). `build_messages()` turns retrieved chunks into a numbered-sources prompt. `synthesize_answer()` calls the LLM and pipes the result through `verify_citations()`, which re-embeds each generated sentence and checks it against the chunk it claims to cite — sentences that don't match closely enough are dropped. `generate_related_questions()` is a separate, smaller follow-up call.

**Tech Stack:** Python 3.12, `httpx` (already a dependency from the Crawler plan), `numpy` + `sentence-transformers`'s `embed()` (already dependencies from the Indexer plan) — no new packages.

## Global Constraints

- `LLMClient` must work against any backend speaking the OpenAI-compatible chat-completions schema — no vendor-specific SDK, no hardcoded provider. Backend selection is `base_url` + `api_key` + `model`, passed in at construction.
- Only standard, safety-tuned instruct models are in scope. Do not add any option, flag, or code path for safety-stripped ("abliterated") models — this was an explicit, deliberate exclusion from the approved design (see `docs/superpowers/specs/2026-07-01-uncensored-search-engine-design.md`).
- `verify_citations` is the accuracy safeguard for "fully sourced, accuracy prioritized" — it must run on every generated answer before that answer reaches a user; there is no bypass path.
- Depends on the Indexer plan being implemented first (`embed` in `uaisearch.indexer`, `Chunk`/`Answer` in `uaisearch.models`).

---

### Task 1: `LLMClient.chat()` — non-streaming chat completion

**Files:**
- Create: `src/uaisearch/synthesis.py`
- Create: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: none.
- Produces: `LLMClient(base_url: str, api_key: str, model: str, http_client: httpx.AsyncClient | None = None)` with `async .chat(messages: list[dict], temperature: float = 0.2) -> str`. Used by Task 5's `synthesize_answer` and Task 6's `generate_related_questions`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthesis.py
import httpx

from uaisearch.synthesis import LLMClient


async def test_chat_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    client = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await client.chat([{"role": "user", "content": "what is the answer"}])
    assert result == "42"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uaisearch.synthesis'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/uaisearch/synthesis.py
import httpx


class LLMClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = http_client or httpx.AsyncClient(timeout=60.0)

    async def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        response = await self._http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/synthesis.py tests/test_synthesis.py
git commit -m "feat: add model-agnostic LLMClient.chat"
```

---

### Task 2: `LLMClient.chat_stream()` — streamed tokens for SSE

**Files:**
- Modify: `src/uaisearch/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: none.
- Produces: `async LLMClient.chat_stream(messages: list[dict], temperature: float = 0.2) -> AsyncIterator[str]`, yielding content deltas. Used by the API plan's `/api/v1/answer` SSE route.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_synthesis.py
SSE_BODY = (
    'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
    'data: [DONE]\n\n'
)


async def test_chat_stream_yields_content_deltas():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"})

    client = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tokens = [t async for t in client.chat_stream([{"role": "user", "content": "hi"}])]
    assert "".join(tokens) == "Hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py -v -k chat_stream`
Expected: FAIL with `AttributeError: 'LLMClient' object has no attribute 'chat_stream'`

- [ ] **Step 3: Write the minimal implementation**

Add `import json` and `from collections.abc import AsyncIterator` to the top of `src/uaisearch/synthesis.py`, then add a `chat_stream` method to the `LLMClient` class written in Task 1 — the full class now reads:

```python
# src/uaisearch/synthesis.py (LLMClient, updated)
import json
from collections.abc import AsyncIterator

import httpx


class LLMClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._http = http_client or httpx.AsyncClient(timeout=60.0)

    async def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        response = await self._http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": temperature},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def chat_stream(
        self, messages: list[dict], temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        async with self._http.stream(
            "POST", f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model, "messages": messages,
                "temperature": temperature, "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v -k chat_stream`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/synthesis.py tests/test_synthesis.py
git commit -m "feat: add LLMClient.chat_stream for SSE token streaming"
```

---

### Task 3: Citation-forcing prompt

**Files:**
- Modify: `src/uaisearch/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: `Chunk` (`uaisearch.models`, Indexer plan Task 1).
- Produces: `SYSTEM_PROMPT: str`, `build_messages(query: str, chunks: list[Chunk]) -> list[dict]`. Used by Task 5's `synthesize_answer`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_synthesis.py
from uaisearch.models import Chunk
from uaisearch.synthesis import build_messages


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py -v -k build_messages`
Expected: FAIL with `ImportError: cannot import name 'build_messages'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/synthesis.py
from uaisearch.models import Chunk

SYSTEM_PROMPT = (
    "Answer using ONLY the numbered sources below.\n"
    "Cite every claim inline as [n]. If the sources don't cover the question, "
    'say "not enough information" rather than guessing.'
)


def build_messages(query: str, chunks: list[Chunk]) -> list[dict]:
    sources_block = "\n".join(
        f"[{i + 1}] {c.chunk_text} (source: {c.url})" for i, c in enumerate(chunks)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{sources_block}\n\nQuestion: {query}"},
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v -k build_messages`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/synthesis.py tests/test_synthesis.py
git commit -m "feat: add citation-forcing system prompt and message builder"
```

---

### Task 4: Citation verification

**Files:**
- Modify: `src/uaisearch/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: `embed` (`uaisearch.indexer`, Indexer plan), `Chunk`, `Answer` (`uaisearch.models`).
- Produces: `cosine(a: list[float], b: list[float]) -> float`, `verify_citations(raw_text: str, chunks: list[Chunk], threshold: float = 0.3) -> Answer`. Used by Task 5's `synthesize_answer`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_synthesis.py
from uaisearch.indexer import embed
from uaisearch.synthesis import verify_citations


def test_verify_citations_keeps_supported_sentences_and_drops_unsupported():
    source_text = "Bees communicate through a waggle dance that encodes direction and distance."
    chunk = Chunk(
        url="https://a.example", title="A", domain="a.example", chunk_text=source_text,
        embedding=embed(source_text), ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01",
    )
    raw_text = (
        "Bees communicate through a waggle dance that encodes direction and distance [1]. "
        "The moon landing happened in 1969 [1]."
    )
    answer = verify_citations(raw_text, [chunk])
    assert "waggle dance" in answer.text
    assert "moon landing" not in answer.text
    assert answer.citations == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py -v -k verify_citations`
Expected: FAIL with `ImportError: cannot import name 'verify_citations'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/synthesis.py
import re

import numpy as np

from uaisearch.indexer import embed
from uaisearch.models import Answer

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def cosine(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return float(np.dot(vec_a, vec_b) / denom) if denom else 0.0


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def verify_citations(raw_text: str, chunks: list[Chunk], threshold: float = 0.3) -> Answer:
    kept_sentences = []
    citations: list[int] = []
    for sentence in _split_sentences(raw_text):
        cited_indices = [int(m) for m in CITATION_PATTERN.findall(sentence)]
        if not cited_indices:
            kept_sentences.append(sentence)
            continue
        supported = False
        for cited_index in cited_indices:
            if 1 <= cited_index <= len(chunks):
                chunk = chunks[cited_index - 1]
                if cosine(embed(sentence), chunk.embedding) >= threshold:
                    supported = True
                    citations.append(cited_index)
        if supported:
            kept_sentences.append(sentence)
    return Answer(
        text=" ".join(kept_sentences),
        citations=sorted(set(citations)),
        sources=chunks,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v -k verify_citations`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/synthesis.py tests/test_synthesis.py
git commit -m "feat: add embedding-based citation verification"
```

---

### Task 5: `synthesize_answer()` end-to-end

**Files:**
- Modify: `src/uaisearch/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: `LLMClient.chat` (Task 1), `build_messages` (Task 3), `verify_citations` (Task 4).
- Produces: `async synthesize_answer(query: str, chunks: list[Chunk], llm: LLMClient) -> Answer`. Consumed by the API plan's `/api/v1/answer` route.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_synthesis.py
from uaisearch.synthesis import synthesize_answer


async def test_synthesize_answer_returns_verified_answer():
    source_text = "Bees communicate through a waggle dance that encodes direction and distance."
    chunk = Chunk(
        url="https://a.example", title="A", domain="a.example", chunk_text=source_text,
        embedding=embed(source_text), ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {
            "content": "Bees communicate through a waggle dance that encodes direction and distance [1].",
        }}]})

    llm = LLMClient(
        base_url="https://api.example/v1", api_key="test", model="test-model",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    answer = await synthesize_answer("how do bees communicate", [chunk], llm)
    assert "waggle dance" in answer.text
    assert answer.citations == [1]
    assert answer.sources == [chunk]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py -v -k synthesize_answer`
Expected: FAIL with `ImportError: cannot import name 'synthesize_answer'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/synthesis.py
async def synthesize_answer(query: str, chunks: list[Chunk], llm: LLMClient) -> Answer:
    raw_text = await llm.chat(build_messages(query, chunks), temperature=0.1)
    return verify_citations(raw_text, chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v -k synthesize_answer`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/synthesis.py tests/test_synthesis.py
git commit -m "feat: add synthesize_answer end-to-end pipeline"
```

---

### Task 6: Related-question generation

**Files:**
- Modify: `src/uaisearch/synthesis.py`
- Modify: `tests/test_synthesis.py`

**Interfaces:**
- Consumes: `LLMClient.chat` (Task 1).
- Produces: `async generate_related_questions(query: str, answer_text: str, llm: LLMClient, count: int = 3) -> list[str]`. Consumed by the API plan's `/api/v1/answer` route, which attaches the result to `Answer.related_questions` before responding.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_synthesis.py
from uaisearch.synthesis import generate_related_questions


async def test_generate_related_questions_parses_numbered_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": (
            "1. Why do bees waggle dance?\n"
            "2. How far can bees communicate distance?\n"
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesis.py -v -k related_questions`
Expected: FAIL with `ImportError: cannot import name 'generate_related_questions'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/synthesis.py
async def generate_related_questions(
    query: str, answer_text: str, llm: LLMClient, count: int = 3,
) -> list[str]:
    prompt = (
        f"Based on this question and answer, suggest {count} short, distinct follow-up "
        "questions a curious reader might ask next. Return them as a plain numbered list, "
        f"no extra commentary.\n\nQuestion: {query}\nAnswer: {answer_text}"
    )
    raw = await llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
    questions = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions[:count]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesis.py -v -k related_questions`
Expected: 1 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests across the Indexer, Crawler, Retrieval, and Synthesis plans pass.

- [ ] **Step 6: Commit**

```bash
git add src/uaisearch/synthesis.py tests/test_synthesis.py
git commit -m "feat: add related-question generation"
```

## Verification

1. `pytest -v` from the repo root passes with 0 failures.
2. Point `LLMClient` at a real backend (a hosted API key, or a local vLLM/Ollama server serving an OpenAI-compatible endpoint) in a scratch script, run `synthesize_answer` against a couple of real `Chunk`s from the Retrieval plan's index, and read the output — confirm citations look right and unsupported claims genuinely get dropped, not just in the synthetic unit test.
