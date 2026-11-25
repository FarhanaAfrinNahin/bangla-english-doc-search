from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_model():
    """Load the multilingual embedding model."""
    model = SentenceTransformer(MODEL_NAME)
    return model


def embed_texts(texts):
    """Convert a list of texts into numerical vectors."""
    model = load_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )
    return vectors


if __name__ == "__main__":
    model = load_model()
    print("Embedding model loaded successfully!")
    print("Model:", MODEL_NAME)
    print("Embedding dimensions:", model.get_embedding_dimension())