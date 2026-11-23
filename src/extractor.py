
import io
from pathlib import Path

import pymupdf

try:
    import pytesseract
    from PIL import Image

    TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if TESSERACT_PATH.exists():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

MIN_TEXT_CHARS = 20  # Pages with less text than this are treated as scanned


def ocr_page(page, dpi=100):
    """Turn a PDF page into an image and read it with Tesseract (Bangla + English)."""
    pix = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(image, lang="ben+eng").strip()


def extract_pdf(path, use_ocr=True, max_pages=None):
    """Read a PDF page by page. Uses OCR when a page has no normal text."""
    path = Path(path)
    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            if max_pages and number > max_pages:
                break
            text = page.get_text().strip()
            method = "text"
            if len(text) < MIN_TEXT_CHARS and use_ocr and OCR_AVAILABLE:
                text = ocr_page(page)
                method = "ocr"
            pages.append(
                {"source": path.name, "page": number, "text": text, "method": method}
            )
    return pages


def extract_txt(path):
    """Read a plain text file as a single page."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"source": path.name, "page": 1, "text": text, "method": "text"}]


def extract_file(path, **kwargs):
    """Pick the right reader based on the file type."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, **kwargs)
    if suffix == ".txt":
        return extract_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")


if __name__ == "__main__":
    # Quick summary (OCR off so it stays fast)
    for file in sorted(Path("data/raw").iterdir()):
        if file.suffix.lower() not in {".pdf", ".txt"}:
            continue
        pages = extract_file(file, use_ocr=False) if file.suffix.lower() == ".pdf" else extract_file(file)
        total_chars = sum(len(p["text"]) for p in pages)
        empty_pages = sum(1 for p in pages if len(p["text"]) < MIN_TEXT_CHARS)
        print(f"{file.name}: {len(pages)} pages, {total_chars} chars, {empty_pages} empty pages")
