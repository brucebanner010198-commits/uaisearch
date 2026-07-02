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
