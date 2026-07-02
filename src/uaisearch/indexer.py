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
