import os
import sys
from datetime import date

from uaisearch.common_crawl import build_page_from_wet, iter_wet_records
from uaisearch.indexer import create_index, index_page, load_simhash_index
from uaisearch.opensearch_client import get_client

TARGET_DOMAINS = {d.strip().lower() for d in os.environ.get("TARGET_DOMAINS", "").split(",") if d.strip()}


def main(s3_key: str, crawl_date: str) -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    create_index(client)
    dedup_index = load_simhash_index(client)

    total_chunks = 0
    for url, domain, text in iter_wet_records(s3_key, TARGET_DOMAINS):
        page = build_page_from_wet(url, domain, text, crawl_date=crawl_date)
        total_chunks += index_page(client, page, dedup_index)
    print(f"Backfilled {total_chunks} chunks from {s3_key}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: backfill_common_crawl.py <wet-segment-s3-key> [crawl-date YYYY-MM-DD]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat())
