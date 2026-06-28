"""
darkclaw/core/doc_store.py

Document ingestion pipeline.

Upload → parse → chunk → ingest into Darkclaw memory as facts.
Once chunks are in memory, the orchestrator's existing memory-injection
path delivers them to agents automatically on every task submit.

Supported formats (graceful fallback if optional libs missing):
  .txt  .md  .py  .js  .json  .csv  .yaml  .toml  — built-in
  .pdf  — requires pypdf  (pip install pypdf)
  .docx — requires python-docx  (pip install python-docx)
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

DOCS_DIR = Path(os.environ.get("DARKCLAW_DOCS",
                               os.path.expanduser("~/.config/darkclaw/docs")))
META_FILE = DOCS_DIR / "index.json"

CHUNK_WORDS = 350     # target words per chunk
CHUNK_OVERLAP = 40    # words of overlap between chunks


# ── metadata persistence ───────────────────────────────────────────────

def _load_meta() -> dict:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if META_FILE.exists():
        try:
            return json.loads(META_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_meta(meta: dict):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2))


# ── text extraction ────────────────────────────────────────────────────

TEXT_EXTS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv",
             ".yaml", ".yml", ".toml", ".sh", ".html", ".xml", ".rst"}


def extract_text(filename: str, raw_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()

    if ext in TEXT_EXTS:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("utf-8", errors="replace")

    if ext == ".pdf":
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)
        except ImportError:
            return "[PDF support requires: pip install pypdf]"
        except Exception as e:
            return f"[PDF parse error: {e}]"

    if ext == ".docx":
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(raw_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return "[DOCX support requires: pip install python-docx]"
        except Exception as e:
            return f"[DOCX parse error: {e}]"

    # Unknown — try plain text
    try:
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return "[Binary file — cannot extract text]"


# ── chunking ───────────────────────────────────────────────────────────

def chunk_text(text: str, words_per_chunk: int = CHUNK_WORDS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = re.split(r"\s+", text.strip())
    if not words:
        return []
    chunks = []
    step = max(1, words_per_chunk - overlap)
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + words_per_chunk])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


# ── doc ID ────────────────────────────────────────────────────────────

def _doc_id(filename: str, content_hash: str) -> str:
    return hashlib.md5(f"{filename}:{content_hash}".encode()).hexdigest()[:12]


# ── public API ─────────────────────────────────────────────────────────

def ingest_document(filename: str, raw_bytes: bytes, memory) -> dict:
    """
    Parse, chunk, and ingest a document into Darkclaw memory.
    Returns metadata dict describing what was stored.
    """
    content_hash = hashlib.md5(raw_bytes).hexdigest()[:10]
    doc_id       = _doc_id(filename, content_hash)

    # Check for duplicate
    meta = _load_meta()
    if doc_id in meta:
        return {**meta[doc_id], "duplicate": True}

    text   = extract_text(filename, raw_bytes)
    chunks = chunk_text(text)

    # Ingest each chunk as a memory fact
    ingested = 0
    for i, chunk in enumerate(chunks):
        subject    = f"doc:{doc_id}:chunk_{i}"
        predicate  = "DOCUMENT_CHUNK"
        object_val = f"[{filename} §{i+1}] {chunk[:120]}"
        try:
            memory.ingest_fact(
                agent_id="docs",
                subject=subject,
                predicate=predicate,
                object_val=object_val,
                speaker="doc_store",
                text=chunk,              # full chunk goes into text search
            )
            ingested += 1
        except Exception:
            pass

    # Also ingest a summary fact so agents can find the doc by name
    memory.ingest_fact(
        agent_id="docs",
        subject=f"doc:{doc_id}",
        predicate="IS_DOCUMENT",
        object_val=filename,
        speaker="doc_store",
        text=f"Document: {filename} — {len(chunks)} chunks indexed",
    )

    record = {
        "doc_id":     doc_id,
        "filename":   filename,
        "size_bytes": len(raw_bytes),
        "chunks":     len(chunks),
        "ingested":   ingested,
        "hash":       content_hash,
        "uploaded_at": time.time(),
        "duplicate":  False,
    }
    meta[doc_id] = record
    _save_meta(meta)

    # Save raw file for reference
    dest = DOCS_DIR / f"{doc_id}_{filename}"
    dest.write_bytes(raw_bytes)

    return record


def list_documents() -> list[dict]:
    meta = _load_meta()
    return sorted(meta.values(), key=lambda d: d.get("uploaded_at", 0), reverse=True)


def delete_document(doc_id: str, memory) -> bool:
    meta = _load_meta()
    if doc_id not in meta:
        return False

    record = meta[doc_id]

    # Remove stored file
    for f in DOCS_DIR.glob(f"{doc_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass

    # Remove from memory (best-effort — memory doesn't support bulk delete yet)
    try:
        n_chunks = record.get("chunks", 0)
        for i in range(n_chunks):
            subject = f"doc:{doc_id}:chunk_{i}"
            memory.delete_subject(subject)
        memory.delete_subject(f"doc:{doc_id}")
    except Exception:
        pass

    del meta[doc_id]
    _save_meta(meta)
    return True
