from uaisearch.indexer import INDEX_NAME, DedupIndex, create_index, index_page
from uaisearch.models import ExtractedPage
from uaisearch.opensearch_client import get_client


def test_index_page_indexes_chunks_and_skips_near_duplicates():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = DedupIndex(client)

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


def test_index_page_skips_dark_web_urls_as_only_content_exclusion():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = DedupIndex(client)

    page = ExtractedPage(
        url="http://abcdefonionhost.onion/page", domain="abcdefonionhost.onion", title="Hidden",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=999,
    )
    count = index_page(client, page, dedup_index)
    assert count == 0
    client.indices.refresh(index=INDEX_NAME)
    assert client.count(index=INDEX_NAME)["count"] == 0


def test_index_page_accepts_unsigned_64bit_simhash_values():
    # Real content routinely produces simhash values above 2**63-1; a signed
    # "long" mapping rejects them with mapper_parsing_exception.
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = DedupIndex(client)

    page = ExtractedPage(
        url="https://big-hash.example/post", domain="big-hash.example", title="Big",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=9448101097055934206,
    )
    count = index_page(client, page, dedup_index)
    assert count == 2
    client.indices.refresh(index=INDEX_NAME)
    # A fresh DedupIndex must find it through the stored corpus (bands query),
    # proving the value round-trips through OpenSearch intact.
    assert DedupIndex(client).is_near_duplicate(9448101097055934206) is True
