import argparse
import os
from datetime import date
from multiprocessing import get_context

from uaisearch.common_crawl import build_page_from_wet, iter_wet_records
from uaisearch.indexer import INDEX_NAME, DedupIndex, create_index, index_pages
from uaisearch.metrics import IngestStats
from uaisearch.opensearch_client import get_client

TARGET_DOMAINS = {d.strip().lower() for d in os.environ.get("TARGET_DOMAINS", "").split(",") if d.strip()}


def _client():
    return get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )


BATCH_PAGES = 64  # cross-page embedding batches keep the accelerator saturated


def backfill_segment(args: tuple[str, str, int | None]) -> tuple[int, int]:
    s3_key, crawl_date, max_records = args
    client = _client()
    dedup = DedupIndex(client)
    stats = IngestStats()

    def flush(batch: list) -> None:
        stats.chunks += index_pages(client, batch, dedup)
        stats.pages += len(batch)
        batch.clear()

    buffer = []
    for record_number, (url, domain, text) in enumerate(iter_wet_records(s3_key, TARGET_DOMAINS)):
        if max_records is not None and record_number >= max_records:
            break
        buffer.append(build_page_from_wet(url, domain, text, crawl_date=crawl_date))
        if len(buffer) >= BATCH_PAGES:
            flush(buffer)
    if buffer:
        flush(buffer)
    print(f"[{s3_key}] {stats.summary()}", flush=True)
    return stats.pages, stats.chunks


def main(s3_keys: list[str], crawl_date: str, workers: int, max_records: int | None) -> None:
    client = _client()
    create_index(client)

    stats = IngestStats()
    jobs = [(key, crawl_date, max_records) for key in s3_keys]
    if workers > 1 and len(s3_keys) > 1:
        # spawn: each worker loads its own embedding model and OpenSearch client
        with get_context("spawn").Pool(min(workers, len(s3_keys))) as pool:
            results = pool.map(backfill_segment, jobs)
    else:
        results = [backfill_segment(job) for job in jobs]

    stats.pages = sum(pages for pages, _ in results)
    stats.chunks = sum(chunks for _, chunks in results)
    client.indices.refresh(index=INDEX_NAME)
    total_docs = client.count(index=INDEX_NAME)["count"]
    print(f"Backfilled {stats.chunks} chunks from {len(s3_keys)} segment(s).")
    print(stats.summary(index_docs=total_docs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill the index from Common Crawl WET segments.")
    parser.add_argument("s3_keys", nargs="+", help="WET segment S3 keys (one per worker job)")
    parser.add_argument("--crawl-date", default=date.today().isoformat(), help="ISO date stamped on ingested pages")
    parser.add_argument("--workers", type=int, default=1, help="parallel segment workers")
    parser.add_argument("--max-records", type=int, default=None, help="cap records per segment (for controlled runs)")
    cli = parser.parse_args()
    main(cli.s3_keys, cli.crawl_date, cli.workers, cli.max_records)
