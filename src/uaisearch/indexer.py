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
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    # One batched encode call: the model amortizes tokenization and forward
    # passes across the batch, which is the difference between CPU-bound
    # per-chunk calls and saturating the accelerator during backfills.
    if not texts:
        return []
    return _get_model().encode(texts, batch_size=64, normalize_embeddings=True).tolist()


from opensearchpy import OpenSearch, helpers

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
            # Simhash values are unsigned 64-bit; "long" is signed and rejects
            # values above 2**63-1, which real page content produces routinely.
            "simhash": {"type": "unsigned_long"},
            # LSH bands over the simhash (see simhash_bands) for flat-memory
            # near-duplicate lookups against the stored corpus.
            "simhash_bands": {"type": "keyword"},
        }
    },
}


def create_index(client: OpenSearch) -> None:
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)


DEDUP_HAMMING_DISTANCE = 3


def simhash_bands(value: int, k: int = DEDUP_HAMMING_DISTANCE) -> list[str]:
    # Pigeonhole: two 64-bit hashes within hamming distance k agree exactly on
    # at least one of k+1 equal-width blocks, so an exact-term match on any
    # band is a complete candidate filter for near-duplicates.
    n_bands = k + 1
    width = 64 // n_bands
    mask = (1 << width) - 1
    return [f"{i}:{(value >> (i * width)) & mask:04x}" for i in range(n_bands)]


class DedupIndex:
    """Near-duplicate detection with flat memory.

    The stored corpus is queried in OpenSearch through simhash LSH bands;
    hashes indexed during this run are also kept in a small in-run list
    because OpenSearch only makes documents searchable after a refresh.
    Replaces the old load-every-simhash-into-RAM SimhashIndex, which grew
    with corpus size and capped ingestion at single-process memory.
    """

    def __init__(self, client: OpenSearch):
        self._client = client
        self._session_hashes: list[int] = []

    def is_near_duplicate(self, simhash_value: int) -> bool:
        if any(
            (simhash_value ^ h).bit_count() <= DEDUP_HAMMING_DISTANCE
            for h in self._session_hashes
        ):
            return True
        response = self._client.search(index=INDEX_NAME, body={
            "query": {"terms": {"simhash_bands": simhash_bands(simhash_value)}},
            "_source": ["simhash"],
            "size": 200,
        })
        return any(
            (simhash_value ^ int(hit["_source"]["simhash"])).bit_count() <= DEDUP_HAMMING_DISTANCE
            for hit in response["hits"]["hits"]
        )

    def add(self, simhash_value: int) -> None:
        self._session_hashes.append(simhash_value)


from uaisearch.crawler import is_dark_web


from uaisearch.models import ExtractedPage


def index_page(client: OpenSearch, page: ExtractedPage, dedup_index: DedupIndex) -> int:
    return index_pages(client, [page], dedup_index)


def index_pages(client: OpenSearch, pages: list[ExtractedPage], dedup_index: DedupIndex) -> int:
    """Batch ingest: one cross-page embedding batch and one bulk write.

    Embedding compute dominates ingest cost, and the encoder only amortizes
    well over large batches — per-page batches of 2-3 chunks leave most of
    the accelerator idle. Callers with many pages should pass them together.
    """
    accepted: list[tuple[ExtractedPage, list[str]]] = []
    for page in pages:
        # Dark web is the only content exclusion — enforced at every ingestion path.
        if is_dark_web(page.url) or dedup_index.is_near_duplicate(page.simhash):
            continue
        chunks = chunk_text(page.text)
        if not chunks:
            continue
        accepted.append((page, chunks))
        dedup_index.add(page.simhash)  # registered now so in-batch duplicates collapse
    if not accepted:
        return 0

    embeddings = embed_batch([chunk for _, chunks in accepted for chunk in chunks])
    actions = []
    position = 0
    for page, chunks in accepted:
        # ponytail: per-page heuristic; swap for a domain-level rolling average
        # if ad_ratio proves too noisy at the individual-page level
        domain_quality = round(1.0 - page.ad_ratio, 4)
        bands = simhash_bands(page.simhash)
        for chunk in chunks:
            actions.append({
                "_index": INDEX_NAME,
                "url": page.url,
                "domain": page.domain,
                "title": page.title,
                "chunk_text": chunk,
                "embedding": embeddings[position],
                "ad_ratio": page.ad_ratio,
                "domain_quality": domain_quality,
                "crawl_date": page.crawl_date,
                "simhash": page.simhash,
                "simhash_bands": bands,
            })
            position += 1
    helpers.bulk(client, actions, chunk_size=500)
    return len(actions)

