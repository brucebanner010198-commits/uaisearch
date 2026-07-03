from simhash import Simhash

from uaisearch.indexer import INDEX_NAME, create_index, is_near_duplicate, load_simhash_index
from uaisearch.opensearch_client import get_client


def test_is_near_duplicate_true_for_known_hash_false_for_new_hash():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    client.index(index=INDEX_NAME, body={
        "url": "https://example.com/seed", "domain": "example.com", "title": "Seed",
        "chunk_text": "seed chunk", "embedding": [0.1] * 384,
        "ad_ratio": 0.0, "domain_quality": 1.0, "crawl_date": "2026-07-01",
        "simhash": 123456789,
    })
    client.indices.refresh(index=INDEX_NAME)

    dedup_index = load_simhash_index(client)
    assert is_near_duplicate(dedup_index, 123456789) is True
    assert is_near_duplicate(dedup_index, Simhash("completely different unrelated content").value) is False
