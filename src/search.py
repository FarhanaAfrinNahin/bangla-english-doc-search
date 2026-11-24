
import faiss

from src.embedder import embed_texts
from src.indexer import load_index

_cache = {}


def get_index():
    """Load the saved index once and reuse it."""
    if "index" not in _cache:
        _cache["index"], _cache["chunks"] = load_index()
    return _cache["index"], _cache["chunks"]


def search(query, top_k=3):
    """Find the passages that best match the meaning of the query."""
    index, chunks = get_index()

    vector = embed_texts([query]).astype("float32")
    faiss.normalize_L2(vector)

    scores, ids = index.search(vector, top_k)

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue

        chunk = chunks[i]
        results.append(
            {
                "score": round(float(score), 3),
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    return results
