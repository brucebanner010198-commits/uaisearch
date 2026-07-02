import pytest

from uaisearch.indexer import chunk_text


def test_chunk_text_single_chunk_when_short():
    text = ("word " * 100).strip()
    chunks = chunk_text(text, size=450, overlap=50)
    assert len(chunks) == 1


def test_chunk_text_splits_long_text_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, size=450, overlap=50)
    assert len(chunks) == 3
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("", size=450, overlap=50) == []


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("some words here", size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("some words here", size=50, overlap=60)
