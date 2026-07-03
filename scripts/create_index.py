import os

from uaisearch.indexer import create_index
from uaisearch.opensearch_client import get_client


def main() -> None:
    client = get_client(
        host=os.environ.get("OPENSEARCH_HOST", "localhost"),
        port=int(os.environ.get("OPENSEARCH_PORT", "9200")),
    )
    create_index(client)
    print("Index ready.")


if __name__ == "__main__":
    main()
