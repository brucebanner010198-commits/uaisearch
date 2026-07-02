from datetime import date

from uaisearch.models import Chunk


def freshness_decay(crawl_date: str, half_life_days: int = 180) -> float:
    days_old = (date.today() - date.fromisoformat(crawl_date)).days
    days_old = max(0, days_old)
    return 0.5 ** (days_old / half_life_days)


def score(bm25_norm: float, cosine_norm: float, chunk: Chunk) -> float:
    return (
        0.35 * bm25_norm
        + 0.30 * cosine_norm
        + 0.20 * chunk.domain_quality
        - 0.15 * chunk.ad_ratio
        + freshness_decay(chunk.crawl_date)
    )
