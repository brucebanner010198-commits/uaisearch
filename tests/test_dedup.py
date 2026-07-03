from simhash import Simhash

from uaisearch.indexer import (
    INDEX_NAME,
    DedupIndex,
    create_index,
    simhash_bands,
)
from uaisearch.opensearch_client import get_client


def test_simhash_bands_shared_band_within_hamming_distance():
    value = Simhash("bees communicate through a waggle dance").value
    near = value ^ 0b101  # flip 2 bits -> within distance 3
    far = ~value & 0xFFFFFFFFFFFFFFFF  # flip all 64 bits
    assert set(simhash_bands(value)) & set(simhash_bands(near))
    assert not set(simhash_bands(value)) & set(simhash_bands(far))


def _seed_document(client, simhash_value: int) -> None:
    client.index(index=INDEX_NAME, body={
        "url": "https://example.com/seed", "domain": "example.com", "title": "Seed",
        "chunk_text": "seed chunk", "embedding": [0.1] * 384,
        "ad_ratio": 0.0, "domain_quality": 1.0, "crawl_date": "2026-07-01",
        "simhash": simhash_value, "simhash_bands": simhash_bands(simhash_value),
    })
    client.indices.refresh(index=INDEX_NAME)


def test_is_near_duplicate_queries_stored_corpus_via_bands():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    stored = Simhash("bees communicate through a waggle dance").value
    _seed_document(client, stored)

    dedup = DedupIndex(client)
    assert dedup.is_near_duplicate(stored) is True
    assert dedup.is_near_duplicate(stored ^ 0b111) is True  # 3 bits off
    assert dedup.is_near_duplicate(Simhash("completely different unrelated content").value) is False


def test_is_near_duplicate_sees_same_run_hashes_before_refresh():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)

    dedup = DedupIndex(client)
    value = Simhash("fresh page indexed this run").value
    assert dedup.is_near_duplicate(value) is False
    dedup.add(value)  # not yet searchable in OpenSearch (no refresh) — session cache must catch it
    assert dedup.is_near_duplicate(value) is True
