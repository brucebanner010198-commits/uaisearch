from uaisearch.indexer import INDEX_NAME, create_index
from uaisearch.opensearch_client import get_client


def test_create_index_is_idempotent():
    client = get_client()
    client.indices.delete(index=INDEX_NAME, ignore=[404])
    create_index(client)
    assert client.indices.exists(index=INDEX_NAME)
    create_index(client)  # must not raise on second call
    assert client.indices.exists(index=INDEX_NAME)
