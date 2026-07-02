import pytest
import numpy as np

from uaisearch.indexer import chunk_text, embed


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


def test_embed_returns_384_dim_vector():
    vec = embed("hello world")
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_embed_similar_texts_have_higher_similarity_than_unrelated():
    a = np.array(embed("the cat sat on the mat"))
    b = np.array(embed("a cat was sitting on a mat"))
    c = np.array(embed("stock market crashes amid inflation fears"))
    assert float(np.dot(a, b)) > float(np.dot(a, c))
