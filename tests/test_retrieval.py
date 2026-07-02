from datetime import date, timedelta

from uaisearch.models import Chunk
from uaisearch.retrieval import freshness_decay, score


def test_freshness_decay_is_one_for_todays_date():
    assert freshness_decay(date.today().isoformat()) == 1.0


def test_freshness_decay_is_half_at_half_life():
    old_date = (date.today() - timedelta(days=180)).isoformat()
    assert abs(freshness_decay(old_date, half_life_days=180) - 0.5) < 1e-9


def test_freshness_decay_future_date_capped_at_one():
    future = (date.today() + timedelta(days=30)).isoformat()
    assert freshness_decay(future) == 1.0


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


def test_score_ad_ratio_penalty_is_isolated_and_correctly_signed():
    # identical except ad_ratio; only the ad penalty can separate them
    clean = Chunk(url="c", title="", domain="c", chunk_text="", embedding=[],
                  ad_ratio=0.0, domain_quality=0.5, crawl_date="2026-07-01")
    ad_heavy = Chunk(url="a", title="", domain="a", chunk_text="", embedding=[],
                     ad_ratio=0.9, domain_quality=0.5, crawl_date="2026-07-01")
    assert score(0.5, 0.5, clean) > score(0.5, 0.5, ad_heavy)


from uaisearch.indexer import INDEX_NAME, create_index, embed
from uaisearch.opensearch_client import get_client
from uaisearch.retrieval import fetch_candidates, rerank


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


def test_rerank_blends_composite_prior_so_ad_heavy_loses_ties():
    query = "how do I start beekeeping"
    text = "backyard beekeeping hive management for beginners"
    clean = Chunk(url="clean", title="", domain="clean.example", chunk_text=text,
                  embedding=[], ad_ratio=0.0, domain_quality=1.0,
                  crawl_date="2026-07-01", score=1.0)
    ad_heavy = Chunk(url="ads", title="", domain="ads.example", chunk_text=text,
                     embedding=[], ad_ratio=0.9, domain_quality=0.1,
                     crawl_date="2026-07-01", score=0.2)
    result = rerank(query, [ad_heavy, clean], top_k=2)
    assert result[0].url == "clean"
