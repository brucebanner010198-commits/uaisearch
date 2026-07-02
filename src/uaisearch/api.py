import json
import os
import time
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from opensearchpy import OpenSearch
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from uaisearch.opensearch_client import get_client
from uaisearch.retrieval import retrieve_and_rerank
from uaisearch.synthesis import (
    LLMClient,
    generate_related_questions,
    synthesize_answer,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    # ponytail: single-process in-memory window; move to a shared store (e.g. Redis)
    # if the app ever runs as more than one instance behind a load balancer
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._windows: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))

    async def dispatch(self, request: Request, call_next):
        key = request.headers.get("x-api-key") or (
            # LAST hop of X-Forwarded-For: caddy is the one trusted proxy (the app
            # binds loopback-only) and APPENDS the real peer IP, so the rightmost
            # value is the one caddy observed. The leftmost is client-supplied and
            # spoofable — keying on it would let a caller rotate XFF to evade the limit.
            (request.headers.get("x-forwarded-for", "").split(",")[-1].strip() or None)
            or (request.client.host if request.client else "unknown")
        )
        now = time.monotonic()
        if len(self._windows) > 10_000:
            # ponytail: single-process memory bound — drop windows whose 60s elapsed;
            # switch to redis with TTL if this ever runs multi-instance
            self._windows = defaultdict(
                lambda: (0, 0.0),
                {k: v for k, v in self._windows.items() if now - v[1] < 60},
            )
        count, window_start = self._windows[key]
        if now - window_start >= 60:
            count, window_start = 0, now
        count += 1
        self._windows[key] = (count, window_start)
        if count > self.requests_per_minute:
            retry_after = max(0, 60 - (now - window_start))
            return JSONResponse(
                {"error": "rate limit exceeded"}, status_code=429,
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.requests_per_minute - count))
        return response


app = FastAPI(title="uaisearch")
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)


def _client_dependency() -> OpenSearch:
    # ponytail: new client per request; swap for an app-lifespan-scoped
    # singleton if connection overhead becomes measurable
    return get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )


def _get_llm_client() -> LLMClient:
    return LLMClient(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ["LLM_MODEL"],
    )


_CONVERSATIONS: dict[str, list[tuple[str, str]]] = defaultdict(list)


def _expand_query_with_history(conversation_id: str | None, query: str) -> str:
    # .get() never inserts — subscripting the defaultdict here would insert the
    # key on read and permanently defeat the eviction guard's `not in` check
    history = _CONVERSATIONS.get(conversation_id) if conversation_id else None
    if not history:
        return query
    context = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-3:])
    return f"{context}\nQ: {query}"


@app.get("/api/v1/search")
async def search(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(8, ge=1, le=50),
    include_fulltext: bool = False,
    client: OpenSearch = Depends(_client_dependency),
):
    hits = await run_in_threadpool(retrieve_and_rerank, client, q, limit=limit)
    return {
        "query": q,
        "results": [
            {
                "url": h.url, "title": h.title, "domain": h.domain,
                "snippet": h.chunk_text[:200], "published": h.crawl_date,
                **({"fulltext": h.chunk_text} if include_fulltext else {}),
            }
            for h in hits
        ],
    }


@app.post("/api/v1/answer")
async def answer(
    query: str, conversation_id: str | None = None,
    client: OpenSearch = Depends(_client_dependency),
    llm: LLMClient = Depends(_get_llm_client),
):
    expanded_query = _expand_query_with_history(conversation_id, query)
    chunks = await run_in_threadpool(retrieve_and_rerank, client, expanded_query, limit=8)
    result = await synthesize_answer(expanded_query, chunks, llm)
    if conversation_id and conversation_id not in _CONVERSATIONS and len(_CONVERSATIONS) >= 1000:
        _CONVERSATIONS.pop(next(iter(_CONVERSATIONS)))  # ponytail: FIFO eviction; LRU if churn matters
    if conversation_id:
        _CONVERSATIONS[conversation_id].append((query, result.text))

    async def event_stream():
        for word in result.text.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        # second LLM call happens after the visible answer has streamed
        related = await generate_related_questions(query, result.text, llm)
        final_event = {
            "done": True,
            "citations": result.citations,
            "sources": [c.url for c in result.sources],
            "related_questions": related,
        }
        yield f"data: {json.dumps(final_event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


_BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(_BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(_BASE_DIR / "static")), name="static")


@app.get("/")
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})
