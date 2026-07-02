from opensearchpy import OpenSearch


def get_client(host: str = "localhost", port: int = 9200) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
    )
