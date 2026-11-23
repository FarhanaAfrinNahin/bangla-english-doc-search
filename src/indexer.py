
import faiss

from src.embedder import embed_texts


def build_index(chunks):
    """Embed all chunks and put them into a FAISS index."""
    vectors = embed_texts([c["text"] for c in chunks]).astype("float32")

    # Normalize vectors so inner product gives cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index
