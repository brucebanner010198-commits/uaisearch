def chunk_text(text: str, size: int = 450, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


from opensearchpy import OpenSearch

INDEX_NAME = "pages"

INDEX_MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "url": {"type": "keyword"},
            "domain": {"type": "keyword"},
            "title": {"type": "text"},
            "chunk_text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
            },
            "ad_ratio": {"type": "float"},
            "domain_quality": {"type": "float"},
            "crawl_date": {"type": "date", "format": "yyyy-MM-dd"},
            "simhash": {"type": "long"},
        }
    },
}


def create_index(client: OpenSearch) -> None:
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


from simhash import Simhash, SimhashIndex


def load_simhash_index(client: OpenSearch) -> SimhashIndex:
    # ponytail: loads every simhash into memory at once; re-load periodically
    # or switch to a sharded index if the corpus outgrows a single process's memory
    objs: list[tuple[str, Simhash]] = []
    seen: set[int] = set()
    response = client.search(
        index=INDEX_NAME,
        body={"query": {"match_all": {}}, "_source": ["simhash"], "size": 1000},
        scroll="2m",
    )
    while response["hits"]["hits"]:
        for hit in response["hits"]["hits"]:
            value = hit["_source"]["simhash"]
            if value not in seen:  # every chunk of a page shares its simhash — load once
                seen.add(value)
                objs.append((str(value), Simhash(value)))
        response = client.scroll(scroll_id=response["_scroll_id"], scroll="2m")
    client.clear_scroll(scroll_id=response["_scroll_id"])
    return SimhashIndex(objs, k=3)


def is_near_duplicate(index: SimhashIndex, simhash_value: int) -> bool:
    return len(index.get_near_dups(Simhash(simhash_value))) > 0


from urllib.parse import urlparse


def is_blocked(url: str, blocklist: set[str]) -> bool:
    domain = urlparse(url).netloc
    return url in blocklist or domain in blocklist


from uaisearch.models import ExtractedPage


def index_page(
    client: OpenSearch, page: ExtractedPage, dedup_index: SimhashIndex,
    blocklist: set[str] = frozenset(),
) -> int:
    if is_blocked(page.url, blocklist) or is_near_duplicate(dedup_index, page.simhash):
        return 0
    # ponytail: per-page heuristic; swap for a domain-level rolling average
    # if ad_ratio proves too noisy at the individual-page level
    domain_quality = round(1.0 - page.ad_ratio, 4)
    indexed = 0
    for chunk in chunk_text(page.text):
        client.index(index=INDEX_NAME, body={
            "url": page.url,
            "domain": page.domain,
            "title": page.title,
            "chunk_text": chunk,
            "embedding": embed(chunk),
            "ad_ratio": page.ad_ratio,
            "domain_quality": domain_quality,
            "crawl_date": page.crawl_date,
            "simhash": page.simhash,
        })
        indexed += 1
    dedup_index.add(page.url, Simhash(page.simhash))
    return indexed


def purge_blocked(client: OpenSearch, blocklist: set[str]) -> int:
    if not blocklist:
        return 0
    entries = list(blocklist)
    response = client.delete_by_query(index=INDEX_NAME, body={
        "query": {"bool": {"should": [
            {"terms": {"domain": entries}},
            {"terms": {"url": entries}},
        ]}},
    })
    return response["deleted"]
