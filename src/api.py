
import threading
from pathlib import Path

import numpy as np

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.chunker import chunk_pages
from src.extractor import extract_file
from src.indexer import add_chunks, save_index
from src.search import get_index, search
from fastapi.responses import FileResponse
RAW_DIR = Path("data/raw")
ALLOWED_TYPES = {".pdf", ".txt"}
MAX_MB = 20
lock = threading.Lock()

app = FastAPI(
    title="Bangla-English Document Search",
    description="Search Bangla and English documents by meaning.",
)


@app.get("/")
def health():
    """Check that the server is running."""
    return {"status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    """Upload a PDF or TXT file and add it to the search index."""
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED_TYPES:
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    limit = MAX_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")

    with lock:
        index, chunks = get_index()
        if any(c["source"] == name for c in chunks):
            raise HTTPException(409, "A document with this name already exists.")

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / name
        path.write_bytes(data)
        try:
            pages = extract_file(path, use_ocr=False)
            new_chunks = chunk_pages(pages)
        except Exception:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "Could not read this file.")

        if not new_chunks:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "No readable text found in this file.")

        add_chunks(index, chunks, new_chunks)
        save_index(index, chunks)

    return {
        "filename": name,
        "pages": len(pages),
        "chunks_added": len(new_chunks),
    }



@app.get("/search")
def search_documents(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(3, ge=1, le=10),
    source: list[str] | None = Query(None),
):
    """Search all documents by meaning, or only selected files."""
    return {"query": q, "results": search(q, top_k=top_k, sources=source)}

@app.get("/documents")
def list_documents():
    """List all indexed documents and how many chunks each has."""
    _, chunks = get_index()
    counts = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return {
        "documents": [
            {"filename": name, "chunks": n} for name, n in sorted(counts.items())
        ]
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document from the index and from disk."""
    name = Path(filename).name
    with lock:
        index, chunks = get_index()
        ids = [i for i, c in enumerate(chunks) if c["source"] == name]
        if not ids:
            raise HTTPException(404, "Document not found.")
        index.remove_ids(np.array(ids, dtype="int64"))
        chunks[:] = [c for c in chunks if c["source"] != name]
        save_index(index, chunks)
        (RAW_DIR / name).unlink(missing_ok=True)
        return {"deleted": name, "chunks_removed": len(ids)}


@app.get("/app", include_in_schema=False)
def web_app():
    """Serve the web page."""
    return FileResponse("static/index.html")

@app.get("/files/{filename}", include_in_schema=False)
def get_file(filename: str):
    """Open an uploaded document (used by the 'View page' link)."""
    name = Path(filename).name
    path = RAW_DIR / name

    if not path.is_file() or path.suffix.lower() not in ALLOWED_TYPES:
        raise HTTPException(404, "File not found.")

    media = (
        "application/pdf"
        if path.suffix.lower() == ".pdf"
        else "text/plain; charset=utf-8"
    )

    return FileResponse(path, media_type=media)
OVERLAP = 20  # must match the overlap used in chunk_pages


@app.get("/context")
def get_context(chunk_id: str = Query(..., max_length=300)):
    """Return the text just before and after a chunk (on the same page)."""
    _, chunks = get_index()

    for i, c in enumerate(chunks):
        if c["chunk_id"] != chunk_id:
            continue

        before, after = "", ""

        if (
            i > 0
            and chunks[i - 1]["source"] == c["source"]
            and chunks[i - 1]["page"] == c["page"]
        ):
            before = " ".join(chunks[i - 1]["text"].split()[:-OVERLAP])

        if (
            i + 1 < len(chunks)
            and chunks[i + 1]["source"] == c["source"]
            and chunks[i + 1]["page"] == c["page"]
        ):
            after = " ".join(chunks[i + 1]["text"].split()[OVERLAP:])

        return {"chunk_id": chunk_id, "before": before, "after": after}

    raise HTTPException(404, "Chunk not found.")