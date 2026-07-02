from io import BytesIO

from warcio.warcwriter import BufferWARCWriter

from uaisearch.common_crawl import parse_wet_stream


def _build_test_wet_stream(url: str, content: str) -> BytesIO:
    writer = BufferWARCWriter()
    record = writer.create_warc_record(
        url, "conversion", payload=BytesIO(content.encode("utf-8")),
        warc_headers_dict={"WARC-Target-URI": url},
    )
    writer.write_record(record)
    return BytesIO(writer.get_contents())


def test_parse_wet_stream_yields_matching_target_domain():
    stream = _build_test_wet_stream("https://niche-blog.example/post", "hello from a niche blog")
    results = list(parse_wet_stream(stream, target_domains={"niche-blog.example"}))
    assert len(results) == 1
    url, domain, text = results[0]
    assert domain == "niche-blog.example"
    assert "hello from a niche blog" in text


def test_parse_wet_stream_skips_non_target_domains():
    stream = _build_test_wet_stream("https://other.example/post", "irrelevant content")
    results = list(parse_wet_stream(stream, target_domains={"niche-blog.example"}))
    assert results == []


def test_parse_wet_stream_excludes_dark_web_even_when_targeted():
    stream = _build_test_wet_stream("http://abcdefonionhost.onion/page", "dark web content")
    results = list(parse_wet_stream(stream, target_domains={"abcdefonionhost.onion"}))
    assert results == []


def test_parse_wet_stream_normalizes_host_dropping_port_and_case():
    stream = _build_test_wet_stream("https://EXAMPLE.com:8443/page", "normalized content")
    results = list(parse_wet_stream(stream, target_domains={"example.com"}))
    assert len(results) == 1
    url, domain, text = results[0]
    assert domain == "example.com"
