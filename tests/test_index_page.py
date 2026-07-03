from uaisearch.indexer import INDEX_NAME, create_index, index_page, load_simhash_index, purge_blocked
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


def test_index_page_skips_blocklisted_domains():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = load_simhash_index(client)

    page = ExtractedPage(
        url="https://blocked.example/post", domain="blocked.example", title="Blocked",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=999,
    )
    count = index_page(client, page, dedup_index, blocklist={"blocked.example"})
    assert count == 0
    client.indices.refresh(index=INDEX_NAME)
    assert client.count(index=INDEX_NAME)["count"] == 0


def test_index_page_skips_exact_url_blocklist_entry_and_keeps_index_empty():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = load_simhash_index(client)

    page = ExtractedPage(
        url="https://mixed.example/infringing-page", domain="mixed.example", title="Infringing",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=555,
    )
    count = index_page(
        client, page, dedup_index, blocklist={"https://mixed.example/infringing-page"},
    )
    assert count == 0
    client.indices.refresh(index=INDEX_NAME)
    assert client.count(index=INDEX_NAME)["count"] == 0


def test_index_page_skips_blocklisted_domain_regardless_of_url_case():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = load_simhash_index(client)

    page = ExtractedPage(
        url="https://Blocked.Example/post", domain="blocked.example", title="Blocked",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=444,
    )
    count = index_page(client, page, dedup_index, blocklist={"blocked.example"})
    assert count == 0
    client.indices.refresh(index=INDEX_NAME)
    assert client.count(index=INDEX_NAME)["count"] == 0


def test_purge_blocked_deletes_indexed_content():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = load_simhash_index(client)

    page = ExtractedPage(
        url="https://takedown.example/content", domain="takedown.example", title="Takedown",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=888,
    )
    count = index_page(client, page, dedup_index)
    assert count == 2

    control_page = ExtractedPage(
        url="https://keep.example/a", domain="keep.example", title="Keep",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=777,
    )
    control_count = index_page(client, control_page, dedup_index)
    assert control_count == 2
    client.indices.refresh(index=INDEX_NAME)

    deleted = purge_blocked(client, {"takedown.example"})
    assert deleted == 2
    client.indices.refresh(index=INDEX_NAME)
    assert client.count(
        index=INDEX_NAME, body={"query": {"term": {"domain": "takedown.example"}}},
    )["count"] == 0
    assert client.count(
        index=INDEX_NAME, body={"query": {"term": {"domain": "keep.example"}}},
    )["count"] == 2


def test_purge_blocked_deletes_by_exact_url_and_spares_others():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    dedup_index = load_simhash_index(client)

    page = ExtractedPage(
        url="https://takedown2.example/infringing", domain="takedown2.example", title="Infringing",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=444,
    )
    count = index_page(client, page, dedup_index)
    assert count == 2

    control_page = ExtractedPage(
        url="https://keep2.example/a", domain="keep2.example", title="Keep",
        text=" ".join(f"w{i}" for i in range(500)),
        ad_ratio=0.0, crawl_date="2026-07-01", simhash=333,
    )
    control_count = index_page(client, control_page, dedup_index)
    assert control_count == 2
    client.indices.refresh(index=INDEX_NAME)

    deleted = purge_blocked(client, {"https://takedown2.example/infringing"})
    assert deleted == 2
    client.indices.refresh(index=INDEX_NAME)
    assert client.count(
        index=INDEX_NAME, body={"query": {"term": {"domain": "takedown2.example"}}},
    )["count"] == 0
    assert client.count(
        index=INDEX_NAME, body={"query": {"term": {"domain": "keep2.example"}}},
    )["count"] == 2
