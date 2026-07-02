from uaisearch.indexer import INDEX_NAME, create_index, index_page, load_simhash_index
from uaisearch.models import ExtractedPage
from uaisearch.opensearch_client import get_client


def test_index_page_indexes_chunks_and_skips_near_duplicates():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = load_simhash_index(client)

    page = ExtractedPage(
        url="https://example.com/first", domain="example.com", title="First",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.1, crawl_date="2026-07-01", simhash=123456789,
    )
    count = index_page(client, page, dedup_index)
    assert count == 2  # 500 words, size=450/overlap=50 -> 2 chunks
    client.indices.refresh(index=INDEX_NAME)

    duplicate_page = ExtractedPage(
        url="https://example.com/copy", domain="example.com", title="Copy",
        text="different words but flagged as a duplicate via simhash",
        ad_ratio=0.1, crawl_date="2026-07-01", simhash=123456789,
    )
    dup_count = index_page(client, duplicate_page, dedup_index)
    assert dup_count == 0
