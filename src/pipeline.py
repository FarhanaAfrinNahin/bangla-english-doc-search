from pathlib import Path

from src.chunker import chunk_pages
from src.extractor import extract_file

SKIP_FILES = ("old_bangla_document.pdf",)  # OCR output is too noisy for search


def load_chunks(folder="data/raw", skip=SKIP_FILES):
    """Read every PDF/TXT in a folder and return all chunks."""
    chunks = []
    for file in sorted(Path(folder).iterdir()):
        suffix = file.suffix.lower()
        if suffix not in {".pdf", ".txt"} or file.name in skip:
            continue
        if suffix == ".pdf":
            pages = extract_file(file, use_ocr=False)
        else:
            pages = extract_file(file)
        chunks.extend(chunk_pages(pages))
    return chunks