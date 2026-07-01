# Retrieval & Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a user query into a small, relevance-ranked, domain-diverse set of `Chunk`s ready for answer synthesis — implementing the design doc's ranking allow-list (BM25 + embedding similarity + domain quality + freshness, minus ad ratio) with no advertiser/editorial signal anywhere in the formula.

**Architecture:** `fetch_candidates()` runs a BM25 match query and a kNN vector query against OpenSearch, merges and score-normalizes the results into `Chunk`s. `rerank()` re-scores that candidate pool with a cross-encoder for real semantic relevance. `apply_domain_cap()` walks the reranked list and stops any one domain from taking more than its share. `retrieve_and_rerank()` chains all three and is the single entry point the Synthesis and API plans call.

**Tech Stack:** Python 3.12, `opensearch-py`, `sentence-transformers` (embeddings + `CrossEncoder`, already a dependency from the Indexer plan — no new packages needed), `pytest`.

## Global Constraints

- The ranking formula in `score()` is an explicit allow-list: BM25, cosine similarity, domain quality, ad-ratio penalty, freshness. No advertiser bid, partner deal, manual trust score, or user-identity signal may be added to it.
- Depends on the Indexer plan being implemented first (`INDEX_NAME`, `create_index`, `embed`, `Chunk`, `get_client`).

---

### Task 1: Freshness decay and the ranking formula

**Files:**
- Create: `src/uaisearch/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Chunk` (`uaisearch.models`, Indexer plan Task 1).
- Produces: `freshness_decay(crawl_date: str, half_life_days: int = 180) -> float`, `score(bm25_norm: float, cosine_norm: float, chunk: Chunk) -> float`. Used by Task 2's `fetch_candidates`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrieval.py
from datetime import date, timedelta

from uaisearch.models import Chunk
from uaisearch.retrieval import freshness_decay, score


def test_freshness_decay_is_one_for_todays_date():
    assert freshness_decay(date.today().isoformat()) == 1.0


def test_freshness_decay_is_half_at_half_life():
    old_date = (date.today() - timedelta(days=180)).isoformat()
    assert abs(freshness_decay(old_date, half_life_days=180) - 0.5) < 1e-9


def test_score_rewards_relevance_and_penalizes_ad_ratio():
    clean_chunk = Chunk(
        url="a", title="a", domain="a.example", chunk_text="x", embedding=[],
        ad_ratio=0.0, domain_quality=1.0, crawl_date=date.today().isoformat(),
    )
    ad_heavy_chunk = Chunk(
        url="b", title="b", domain="b.example", chunk_text="x", embedding=[],
        ad_ratio=0.9, domain_quality=0.1, crawl_date=date.today().isoformat(),
    )
    assert score(0.8, 0.8, clean_chunk) > score(0.8, 0.8, ad_heavy_chunk)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'uaisearch.retrieval'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/uaisearch/retrieval.py
from datetime import date

from uaisearch.models import Chunk


def freshness_decay(crawl_date: str, half_life_days: int = 180) -> float:
    days_old = (date.today() - date.fromisoformat(crawl_date)).days
    return 0.5 ** (days_old / half_life_days)


def score(bm25_norm: float, cosine_norm: float, chunk: Chunk) -> float:
    return (
        0.35 * bm25_norm
        + 0.30 * cosine_norm
        + 0.20 * chunk.domain_quality
        - 0.15 * chunk.ad_ratio
        + freshness_decay(chunk.crawl_date)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/retrieval.py tests/test_retrieval.py
git commit -m "feat: add freshness decay and ranking formula"
```

---

### Task 2: Candidate fetch (BM25 + kNN) from OpenSearch

**Files:**
- Modify: `src/uaisearch/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `INDEX_NAME`, `embed` (`uaisearch.indexer`, Indexer plan), `score` (Task 1), `get_client` (`uaisearch.opensearch_client`).
- Produces: `fetch_candidates(client: OpenSearch, query: str, limit: int = 30) -> list[Chunk]`, sorted by `.score` descending. Used by Task 5's `retrieve_and_rerank`.

Requires the OpenSearch dev instance from the Indexer plan running (`docker ps` shows `uaisearch-opensearch`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_retrieval.py
from uaisearch.indexer import INDEX_NAME, create_index, embed
from uaisearch.opensearch_client import get_client
from uaisearch.retrieval import fetch_candidates


def test_fetch_candidates_ranks_relevant_chunk_above_irrelevant():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)

    relevant_text = "backyard beekeeping hive management for beginners"
    irrelevant_text = "stock market inflation report quarterly earnings"
    for i, (url, domain, text) in enumerate([
        ("https://a.example/1", "a.example", relevant_text),
        ("https://b.example/1", "b.example", irrelevant_text),
    ]):
        client.index(index=INDEX_NAME, body={
            "url": url, "domain": domain, "title": domain,
            "chunk_text": text, "embedding": embed(text),
            "ad_ratio": 0.0, "domain_quality": 1.0,
            "crawl_date": date.today().isoformat(), "simhash": i,
        })
    client.indices.refresh(index=INDEX_NAME)

    candidates = fetch_candidates(client, "how do I start beekeeping", limit=10)
    assert candidates[0].url == "https://a.example/1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v -k fetch_candidates`
Expected: FAIL with `ImportError: cannot import name 'fetch_candidates'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/retrieval.py
from opensearchpy import OpenSearch

from uaisearch.indexer import INDEX_NAME, embed


def _bm25_candidates(client: OpenSearch, query: str, size: int) -> dict[str, tuple[dict, float]]:
    response = client.search(index=INDEX_NAME, body={
        "size": size, "query": {"match": {"chunk_text": query}},
    })
    return {hit["_id"]: (hit["_source"], hit["_score"]) for hit in response["hits"]["hits"]}


def _knn_candidates(client: OpenSearch, query_emb: list[float], size: int) -> dict[str, tuple[dict, float]]:
    response = client.search(index=INDEX_NAME, body={
        "size": size, "query": {"knn": {"embedding": {"vector": query_emb, "k": size}}},
    })
    return {hit["_id"]: (hit["_source"], hit["_score"]) for hit in response["hits"]["hits"]}


def fetch_candidates(client: OpenSearch, query: str, limit: int = 30) -> list[Chunk]:
    query_emb = embed(query)
    bm25_hits = _bm25_candidates(client, query, limit)
    knn_hits = _knn_candidates(client, query_emb, limit)

    max_bm25 = max((s for _, s in bm25_hits.values()), default=1.0) or 1.0
    max_knn = max((s for _, s in knn_hits.values()), default=1.0) or 1.0

    merged: dict[str, dict] = {}
    for doc_id, (source, raw_score) in bm25_hits.items():
        merged.setdefault(doc_id, {"source": source, "bm25": 0.0, "knn": 0.0})
        merged[doc_id]["bm25"] = raw_score / max_bm25
    for doc_id, (source, raw_score) in knn_hits.items():
        merged.setdefault(doc_id, {"source": source, "bm25": 0.0, "knn": 0.0})
        merged[doc_id]["knn"] = raw_score / max_knn

    candidates = []
    for entry in merged.values():
        source = entry["source"]
        chunk = Chunk(
            url=source["url"], title=source["title"], domain=source["domain"],
            chunk_text=source["chunk_text"], embedding=source["embedding"],
            ad_ratio=source["ad_ratio"], domain_quality=source["domain_quality"],
            crawl_date=source["crawl_date"],
        )
        chunk.score = score(entry["bm25"], entry["knn"], chunk)
        candidates.append(chunk)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v -k fetch_candidates`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/retrieval.py tests/test_retrieval.py
git commit -m "feat: add BM25 + kNN candidate fetch from OpenSearch"
```

---

### Task 3: Cross-encoder rerank

**Files:**
- Modify: `src/uaisearch/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Chunk` (`uaisearch.models`).
- Produces: `rerank(query: str, candidates: list[Chunk], top_k: int = 8) -> list[Chunk]`. Used by Task 5's `retrieve_and_rerank`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_retrieval.py
from uaisearch.retrieval import rerank


def test_rerank_orders_by_relevance_and_truncates():
    query = "how do I start beekeeping"
    relevant = Chunk(
        url="a", title="a", domain="a.example",
        chunk_text="backyard beekeeping hive management for beginners",
        embedding=[], ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01",
    )
    irrelevant = Chunk(
        url="b", title="b", domain="b.example",
        chunk_text="stock market inflation report",
        embedding=[], ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01",
    )
    result = rerank(query, [irrelevant, relevant], top_k=1)
    assert len(result) == 1
    assert result[0].url == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v -k test_rerank`
Expected: FAIL with `ImportError: cannot import name 'rerank'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/retrieval.py
from sentence_transformers import CrossEncoder

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, candidates: list[Chunk], top_k: int = 8) -> list[Chunk]:
    if not candidates:
        return []
    pairs = [(query, c.chunk_text) for c in candidates]
    cross_scores = _get_cross_encoder().predict(pairs)
    for chunk, cross_score in zip(candidates, cross_scores):
        chunk.score = float(cross_score)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v -k test_rerank`
Expected: 1 passed (first run downloads the `cross-encoder/ms-marco-MiniLM-L-6-v2` model — allow extra time)

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/retrieval.py tests/test_retrieval.py
git commit -m "feat: add cross-encoder rerank"
```

---

### Task 4: Domain diversity cap

**Files:**
- Modify: `src/uaisearch/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Chunk`.
- Produces: `apply_domain_cap(chunks: list[Chunk], max_per_domain: int = 2) -> list[Chunk]`. Used by Task 5's `retrieve_and_rerank`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_retrieval.py
from uaisearch.retrieval import apply_domain_cap


def test_apply_domain_cap_limits_chunks_per_domain():
    def make(url, domain):
        return Chunk(url=url, title="", domain=domain, chunk_text="", embedding=[],
                      ad_ratio=0.0, domain_quality=1.0, crawl_date="2026-07-01")

    chunks = [make(f"a{i}", "a.example") for i in range(3)] + [make("b1", "b.example")]
    capped = apply_domain_cap(chunks, max_per_domain=2)
    assert sum(1 for c in capped if c.domain == "a.example") == 2
    assert sum(1 for c in capped if c.domain == "b.example") == 1
    assert len(capped) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v -k domain_cap`
Expected: FAIL with `ImportError: cannot import name 'apply_domain_cap'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/retrieval.py
def apply_domain_cap(chunks: list[Chunk], max_per_domain: int = 2) -> list[Chunk]:
    counts: dict[str, int] = {}
    capped = []
    for chunk in chunks:
        counts.setdefault(chunk.domain, 0)
        if counts[chunk.domain] < max_per_domain:
            capped.append(chunk)
            counts[chunk.domain] += 1
    return capped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v -k domain_cap`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/uaisearch/retrieval.py tests/test_retrieval.py
git commit -m "feat: add domain diversity cap"
```

---

### Task 5: `retrieve_and_rerank()` end-to-end

**Files:**
- Modify: `src/uaisearch/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `fetch_candidates` (Task 2), `rerank` (Task 3), `apply_domain_cap` (Task 4).
- Produces: `retrieve_and_rerank(client: OpenSearch, query: str, limit: int = 8, candidate_pool: int = 30) -> list[Chunk]`. Consumed by the Synthesis plan's `synthesize_answer` and the API plan's `/api/v1/search` and `/api/v1/answer` routes.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_retrieval.py
from uaisearch.retrieval import retrieve_and_rerank


def test_retrieve_and_rerank_orders_relevant_chunk_first_and_respects_limit():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)

    texts = [
        ("https://a.example/1", "a.example", "backyard beekeeping hive management for beginners"),
        ("https://b.example/1", "b.example", "stock market inflation report quarterly earnings"),
        ("https://c.example/1", "c.example", "beekeeping smoker tools and honey extraction basics"),
    ]
    for i, (url, domain, text) in enumerate(texts):
        client.index(index=INDEX_NAME, body={
            "url": url, "domain": domain, "title": domain,
            "chunk_text": text, "embedding": embed(text),
            "ad_ratio": 0.0, "domain_quality": 1.0,
            "crawl_date": date.today().isoformat(), "simhash": i,
        })
    client.indices.refresh(index=INDEX_NAME)

    results = retrieve_and_rerank(client, "how do I start beekeeping", limit=2)
    assert len(results) <= 2
    assert results[0].domain in {"a.example", "c.example"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v -k retrieve_and_rerank`
Expected: FAIL with `ImportError: cannot import name 'retrieve_and_rerank'`

- [ ] **Step 3: Write the minimal implementation**

```python
# append to src/uaisearch/retrieval.py
def retrieve_and_rerank(
    client: OpenSearch, query: str, limit: int = 8, candidate_pool: int = 30,
) -> list[Chunk]:
    candidates = fetch_candidates(client, query, limit=candidate_pool)
    reranked = rerank(query, candidates, top_k=candidate_pool)
    return apply_domain_cap(reranked, max_per_domain=2)[:limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v -k retrieve_and_rerank`
Expected: 1 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests across the Indexer, Crawler, and Retrieval plans pass.

- [ ] **Step 6: Commit**

```bash
git add src/uaisearch/retrieval.py tests/test_retrieval.py
git commit -m "feat: add retrieve_and_rerank end-to-end pipeline"
```

## Verification

1. `pytest -v` from the repo root passes with 0 failures.
2. Manually call `retrieve_and_rerank(get_client(), "<a real question about content you've indexed>")` in a scratch script and read the results — confirm the top chunk is actually relevant and no single domain dominates.
