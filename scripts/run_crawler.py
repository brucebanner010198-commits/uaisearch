import asyncio
import os

import httpx

from uaisearch.crawler import Frontier, SeedManager, run_crawl_cycle
from uaisearch.indexer import create_index, index_page, load_simhash_index
from uaisearch.opensearch_client import get_client

SEED_URLS = [u.strip() for u in os.environ.get("SEED_URLS", "").split(",") if u.strip()]
FRONTIER_STATE_PATH = os.environ.get("FRONTIER_STATE_PATH", "/data/frontier.json")


async def main() -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    create_index(client)
    dedup_index = load_simhash_index(client)

    seeds = SeedManager.load(FRONTIER_STATE_PATH, default_seeds=SEED_URLS)
    frontier = Frontier()
    async with httpx.AsyncClient() as http_client:
        pages = await run_crawl_cycle(
            seeds, frontier, http_client,
            max_pages=int(os.environ.get("MAX_PAGES", "50")),
        )

    total_chunks = sum(
        index_page(client, page, dedup_index)
        for page in pages
    )
    print(f"Crawled {len(pages)} pages, indexed {total_chunks} chunks.")

    # Save after indexing — a crash mid-index re-fetches this cycle instead of losing pages forever.
    seeds.save(FRONTIER_STATE_PATH)


if __name__ == "__main__":
    asyncio.run(main())
