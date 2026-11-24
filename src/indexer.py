
import json
from pathlib import Path

import faiss

from src.embedder import embed_texts

INDEX_DIR = Path("data/processed")


def build_index(chunks):
    """Embed all chunks and put them into a FAISS index."""
    vectors = embed_texts([c["text"] for c in chunks]).astype("float32")

    # Normalize vectors so inner product gives cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, chunks, folder=INDEX_DIR):
    """Save the index and the chunk texts to disk."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(folder / "index.faiss"))

    (folder / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )


def load_index(folder=INDEX_DIR):
    """Load the saved index and chunk texts."""
    folder = Path(folder)

    index = faiss.read_index(str(folder / "index.faiss"))

    chunks = json.loads(
        (folder / "chunks.json").read_text(encoding="utf-8")
    )

    return index, chunks

def add_chunks(index, chunks, new_chunks):
    """Embed new chunks and add them to an existing index."""
    vectors = embed_texts([c["text"] for c in new_chunks]).astype("float32")

    # Normalize new vectors, just like the existing index vectors
    faiss.normalize_L2(vectors)

    index.add(vectors)
    chunks.extend(new_chunks)