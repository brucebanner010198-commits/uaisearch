import os

from uaisearch.indexer import purge_blocked
from uaisearch.opensearch_client import get_client

BLOCKED_DOMAINS = {d.strip().lower() for d in os.environ.get("BLOCKED_DOMAINS", "").split(",") if d.strip()}


def main() -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    deleted = purge_blocked(client, BLOCKED_DOMAINS)
    print(f"Purged {deleted} chunks for {len(BLOCKED_DOMAINS)} blocked entries.")


if __name__ == "__main__":
    main()
