
import faiss

from src.embedder import embed_texts
from src.indexer import load_index

_cache = {}


def get_index():
    """Load the saved index once and reuse it."""
    if "index" not in _cache:
        _cache["index"], _cache["chunks"] = load_index()
    return _cache["index"], _cache["chunks"]


def search(query, top_k=3, sources=None):
    """Find the passages that best match the meaning of the query.

    If sources is given, only passages from those files are returned.
    """
    index, chunks = get_index()

    if index.ntotal == 0:
        return []

    vector = embed_texts([query]).astype("float32")
    faiss.normalize_L2(vector)

    # Search everything when filtering, then keep selected documents.
    k = index.ntotal if sources else min(top_k, index.ntotal)
    scores, ids = index.search(vector, k)

    results = []

    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue

        chunk = chunks[i]

        if sources and chunk["source"] not in sources:
            continue

        results.append(
            {
                "score": round(float(score), 3),
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

        if len(results) == top_k:
            break

    return results
