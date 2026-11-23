
from pathlib import Path

import pymupdf


def extract_pdf(path):
    """Read a PDF and return a list of pages with their text."""
    path = Path(path)
    pages = []

    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            pages.append({
                "source": path.name,
                "page": number,
                "text": text
            })

    return pages


def extract_txt(path):
    """Read a plain text file as a single page."""
    path = Path(path)
    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    return [{
        "source": path.name,
        "page": 1,
        "text": text
    }]


def extract_file(path):
    """Pick the right reader based on the file type."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(path)

    if suffix == ".txt":
        return extract_txt(path)

    raise ValueError(f"Unsupported file type: {suffix}")


if __name__ == "__main__":
    # Quick test: read every file in data/raw and print a summary
    for file in sorted(Path("data/raw").iterdir()):
        if file.suffix.lower() not in {".pdf", ".txt"}:
            continue

        pages = extract_file(file)
        total_chars = sum(len(p["text"]) for p in pages)
        empty_pages = sum(1 for p in pages if len(p["text"]) < 20)

        print(
            f"{file.name}: {len(pages)} pages, "
            f"{total_chars} chars, {empty_pages} empty pages"
        )
