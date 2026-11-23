from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_model():
    """Load the multilingual embedding model."""
    model = SentenceTransformer(MODEL_NAME)
    return model


if __name__ == "__main__":
    model = load_model()
    print("Embedding model loaded successfully!")
    print("Model:", MODEL_NAME)
    print("Embedding dimensions:", model.get_embedding_dimension())