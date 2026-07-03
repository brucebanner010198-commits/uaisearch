from datetime import date

from opensearchpy import OpenSearch
from sentence_transformers import CrossEncoder

from uaisearch.indexer import INDEX_NAME, embed
from uaisearch.models import Chunk

_cross_encoder: CrossEncoder | None = None


def freshness_decay(crawl_date: str, half_life_days: int = 180) -> float:
    days_old = (date.today() - date.fromisoformat(crawl_date)).days
    days_old = max(0, days_old)
    return 0.5 ** (days_old / half_life_days)


def score(bm25_norm: float, cosine_norm: float, chunk: Chunk) -> float:
    return (
        0.35 * bm25_norm
        + 0.30 * cosine_norm
        + 0.20 * chunk.domain_quality
        - 0.15 * chunk.ad_ratio
        + freshness_decay(chunk.crawl_date)
    )


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


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(query: str, candidates: list[Chunk], top_k: int = 8) -> list[Chunk]:
    if not candidates:
        return []
    pairs = [(query, c.chunk_text) for c in candidates]
    cross_scores = [float(s) for s in _get_cross_encoder().predict(pairs)]

    def normalize(values: list[float]) -> list[float]:
        lo, span = min(values), (max(values) - min(values)) or 1.0
        return [(v - lo) / span for v in values]

    cross_norms = normalize(cross_scores)
    prior_norms = normalize([c.score for c in candidates])
    for chunk, cross_norm, prior_norm in zip(candidates, cross_norms, prior_norms):
        # ponytail: 0.7/0.3 keeps semantic relevance dominant while the composite
        # prior (ad ratio, domain quality, freshness) still separates ties
        chunk.score = 0.7 * cross_norm + 0.3 * prior_norm
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def apply_domain_cap(chunks: list[Chunk], max_per_domain: int = 2) -> list[Chunk]:
    counts: dict[str, int] = {}
    capped = []
    for chunk in chunks:
        counts.setdefault(chunk.domain, 0)
        if counts[chunk.domain] < max_per_domain:
            capped.append(chunk)
            counts[chunk.domain] += 1
    return capped


def retrieve_and_rerank(
    client: OpenSearch, query: str, limit: int = 8, candidate_pool: int = 30,
) -> list[Chunk]:
    candidates = fetch_candidates(client, query, limit=candidate_pool)
    reranked = rerank(query, candidates, top_k=candidate_pool)
    return apply_domain_cap(reranked, max_per_domain=2)[:limit]
