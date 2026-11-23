
def chunk_pages(pages, chunk_words=120, overlap=20):
    """Cut each page into small overlapping pieces, keeping source and page."""
    chunks = []
    step = chunk_words - overlap

    for page in pages:
        words = page["text"].split()

        if len(words) < 5:
            continue  # skip empty or nearly empty pages

        for part, start in enumerate(range(0, len(words), step)):
            piece = words[start:start + chunk_words]

            if start > 0 and len(piece) <= overlap:
                break  # this tail is already inside the previous chunk

            chunks.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_id": f"{page['source']}-p{page['page']}-c{part}",
                    "text": " ".join(piece),
                }
            )

    return chunks
