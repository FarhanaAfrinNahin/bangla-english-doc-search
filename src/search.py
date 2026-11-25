from src.embedder import embed_texts
from src.indexer import load_index
from src.keyword import KeywordIndex

RRF_K = 60
RRF_DEPTH = 50

_cache = {}


def get_index():
    """Load the saved index once and reuse it."""
    if "index" not in _cache:
        _cache["index"], _cache["chunks"] = load_index()
    return _cache["index"], _cache["chunks"]


def reset_keyword_index():
    """Forget the keyword index (call after documents are added or removed)."""
    _cache.pop("keyword", None)
    _cache.pop("keyword_key", None)


def _get_keyword_index(chunks):
    key = (id(chunks), len(chunks))
    if _cache.get("keyword_key") != key:
        _cache["keyword"] = KeywordIndex([c["text"] for c in chunks])
        _cache["keyword_key"] = key
    return _cache["keyword"]


def search(query, top_k=3, sources=None, mode="semantic"):
    """Search the documents.

    mode: "semantic" (meaning), "keyword" (exact words), or "hybrid" (both).
    sources: optional list of file names to search in.
    """
    index, chunks = get_index()
    if index.ntotal == 0:
        return []

    allowed = {
        i for i, c in enumerate(chunks)
        if not sources or c["source"] in sources
    }
    if not allowed:
        return []

    vector = embed_texts([query]).astype("float32")
    all_scores, all_ids = index.search(vector, index.ntotal)

    semantic = [
        (int(i), float(s))
        for s, i in zip(all_scores[0], all_ids[0])
        if int(i) in allowed
    ]
    meaning_score = dict(semantic)

    if mode == "semantic":
        ranked = [i for i, _ in semantic]
    else:
        word_scores = _get_keyword_index(chunks).scores(query)
        keyword = sorted(
            ((i, word_scores[i]) for i in allowed if word_scores[i] > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )

        if mode == "keyword":
            ranked = [i for i, _ in keyword]
        elif mode == "hybrid":
            fused = {}
            for hits in (semantic, keyword):
                for rank, (i, _) in enumerate(hits[:RRF_DEPTH]):
                    fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank + 1)
            ranked = sorted(fused, key=fused.get, reverse=True)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

    results = []
    for i in ranked[:top_k]:
        chunk = chunks[i]
        results.append(
            {
                "score": round(meaning_score[i], 3),
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    return results