
import zlib

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api as api
import src.indexer as indexer
import src.search as search_module

DIM = 16
DHAKA = b"Dhaka is the capital city of Bangladesh"
COOKING = b"Rice and lentils are cooked with turmeric and salt"


def fake_embed(texts):
    """A tiny fake embedder so tests are fast and need no model."""
    vectors = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            vectors[row, zlib.crc32(word.encode("utf-8")) % DIM] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


@pytest.fixture
def client(tmp_path, monkeypatch):
    index = faiss.IndexFlatIP(DIM)
    chunks = []
    monkeypatch.setattr(indexer, "embed_texts", fake_embed)
    monkeypatch.setattr(search_module, "embed_texts", fake_embed)
    monkeypatch.setattr(api, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(api, "get_index", lambda: (index, chunks))
    monkeypatch.setattr(search_module, "get_index", lambda: (index, chunks))
    monkeypatch.setattr(api, "save_index", lambda *args, **kwargs: None)
    return TestClient(api.app)


def upload(client, name, content, content_type="text/plain"):
    return client.post("/upload", files={"file": (name, content, content_type)})


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upload_txt_works(client):
    r = upload(client, "dhaka.txt", DHAKA)
    assert r.status_code == 200
    assert r.json()["chunks_added"] == 1


def test_duplicate_upload_is_rejected(client):
    upload(client, "dhaka.txt", DHAKA)
    r = upload(client, "dhaka.txt", DHAKA)
    assert r.status_code == 409


def test_unsupported_file_type_is_rejected(client):
    r = upload(client, "pic.png", b"x", "image/png")
    assert r.status_code == 400


def test_search_finds_the_right_document(client):
    upload(client, "dhaka.txt", DHAKA)
    upload(client, "cooking.txt", COOKING)
    r = client.get("/search", params={"q": "capital city of Bangladesh"})
    assert r.status_code == 200
    assert r.json()["results"][0]["source"] == "dhaka.txt"


def test_empty_query_is_rejected(client):
    r = client.get("/search", params={"q": ""})
    assert r.status_code == 422


def test_list_and_delete_document(client):
    upload(client, "dhaka.txt", DHAKA)
    r = client.get("/documents")
    assert r.json()["documents"] == [{"filename": "dhaka.txt", "chunks": 1}]

    assert client.delete("/documents/dhaka.txt").status_code == 200
    assert client.get("/documents").json()["documents"] == []
    assert client.delete("/documents/dhaka.txt").status_code == 404
