from uaisearch.opensearch_client import get_client


def test_get_client_connects():
    client = get_client()
    info = client.info()
    assert "version" in info
