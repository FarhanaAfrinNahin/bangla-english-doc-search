
from src.chunker import chunk_pages


def make_page(n_words):
    text = " ".join(f"w{i}" for i in range(n_words))
    return {"source": "a.txt", "page": 1, "text": text}


def test_short_page_is_skipped():
    assert chunk_pages([make_page(3)]) == []


def test_chunk_keeps_source_and_page():
    chunks = chunk_pages([make_page(50)])
    assert len(chunks) == 1
    assert chunks[0]["source"] == "a.txt"
    assert chunks[0]["page"] == 1


def test_long_page_makes_overlapping_chunks():
    chunks = chunk_pages([make_page(300)], chunk_words=120, overlap=20)
    assert len(chunks) > 1
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-20:] == second[:20]


def test_bangla_text_is_kept():
    text = "বাংলাদেশ একটি সুন্দর দেশ যেখানে নদী অনেক"
    page = {"source": "b.txt", "page": 1, "text": text}
    chunks = chunk_pages([page])
    assert chunks[0]["text"] == text
