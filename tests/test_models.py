from uaisearch.models import ExtractedPage, Chunk, Answer


def test_extracted_page_construction():
    page = ExtractedPage(
        url="https://example.com/a",
        domain="example.com",
        title="A",
        text="hello world",
        ad_ratio=0.1,
        crawl_date="2026-07-01",
        simhash=12345,
    )
    assert page.url == "https://example.com/a"
    assert page.ad_ratio == 0.1


def test_chunk_construction_defaults_score_zero():
    chunk = Chunk(
        url="https://example.com/a", title="A", domain="example.com",
        chunk_text="hello", embedding=[0.1, 0.2], ad_ratio=0.1,
        domain_quality=0.9, crawl_date="2026-07-01",
    )
    assert chunk.score == 0.0


def test_answer_defaults_related_questions_empty():
    answer = Answer(text="hi", citations=[1], sources=[])
    assert answer.related_questions == []
