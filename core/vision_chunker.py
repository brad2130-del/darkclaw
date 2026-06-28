"""
darkclaw/core/vision_chunker.py

Local Chunkr-equivalent: vision/layout-aware document chunking.

Replaces the naive word-count slicer in doc_store.py with semantic
chunking that understands document structure:

  Title           → stored as subject entity, seeds the knowledge graph
  NarrativeText   → paragraph chunks, full vector retrieval
  Table           → preserved as markdown table, tagged TABLE
  ListItem        → grouped into list chunks, tagged LIST
  Header          → section boundary markers
  Image/Figure    → noted with caption if present

Uses the `unstructured` library (local, CPU, no API needed).
Falls back to word-count chunking if unstructured is not available
or the file type isn't supported.

Why this matters for RAG:
  Naive chunking breaks tables across chunks and splits mid-sentence.
  Typed chunks let agents know WHAT they're reading:
    "The following is a TABLE from <doc>" vs "The following is a PARAGRAPH"
  Better chunk types → more precise memory queries → better answers.
"""
from __future__ import annotations
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Chunk type labels (mirrors Chunkr's taxonomy) ────────────────────

class ChunkType:
    TITLE       = "Title"
    TEXT        = "NarrativeText"
    TABLE       = "Table"
    LIST        = "List"
    HEADER      = "SectionHeader"
    CAPTION     = "FigureCaption"
    CODE        = "CodeSnippet"
    UNKNOWN     = "UncategorizedText"


@dataclass
class Chunk:
    text:       str
    type:       str  = ChunkType.TEXT
    page:       int  = 0
    section:    str  = ""        # nearest title above this chunk
    metadata:   dict = field(default_factory=dict)

    def to_ingest_text(self, filename: str, index: int) -> str:
        """Full text block stored in Darkclaw memory for retrieval."""
        prefix = f"[{filename} · {self.type}"
        if self.section:
            prefix += f" · §{self.section[:40]}"
        if self.page:
            prefix += f" · p{self.page}"
        prefix += f"] "
        return prefix + self.text

    def to_object_val(self) -> str:
        """Short summary for the graph edge (object field, ≤200 chars)."""
        return self.text[:180].replace("\n", " ")


# ── Main entry point ─────────────────────────────────────────────────

def chunk_document(filename: str, raw_bytes: bytes) -> list[Chunk]:
    """
    Parse and chunk a document using layout-aware analysis.
    Returns a list of Chunk objects ready for memory ingestion.
    Falls back to word-count chunking on any error.
    """
    try:
        return _unstructured_chunk(filename, raw_bytes)
    except Exception:
        # Graceful fallback — never crash the upload pipeline
        return _fallback_chunk(filename, raw_bytes)


# ── Unstructured-based chunker ────────────────────────────────────────

def _unstructured_chunk(filename: str, raw_bytes: bytes) -> list[Chunk]:
    from unstructured.documents.elements import (
        Title, NarrativeText, Table, ListItem, Header, FigureCaption,
        CodeSnippet, Text, Image,
    )

    ext = Path(filename).suffix.lower()
    elements = _partition(filename, ext, raw_bytes)

    chunks: list[Chunk] = []
    current_section = ""
    list_buffer: list[str] = []
    current_page = 0

    def flush_list():
        if list_buffer:
            chunks.append(Chunk(
                text="\n".join(f"• {item}" for item in list_buffer),
                type=ChunkType.LIST,
                page=current_page,
                section=current_section,
            ))
            list_buffer.clear()

    for el in elements:
        # Track page number
        page = getattr(getattr(el, "metadata", None), "page_number", None)
        if page:
            current_page = page

        text = str(el).strip()
        if not text:
            continue

        if isinstance(el, (Title, Header)):
            flush_list()
            current_section = text[:80]
            # Only add title as a chunk if it's substantial
            if len(text) > 3:
                chunks.append(Chunk(text=text, type=ChunkType.TITLE,
                                    page=current_page, section=current_section))

        elif isinstance(el, Table):
            flush_list()
            # Tables: try to preserve as markdown-ish grid
            table_text = _table_to_md(text)
            chunks.append(Chunk(text=table_text, type=ChunkType.TABLE,
                                page=current_page, section=current_section))

        elif isinstance(el, ListItem):
            list_buffer.append(text)

        elif isinstance(el, (FigureCaption, Image)):
            flush_list()
            if text and len(text) > 5:
                chunks.append(Chunk(text=text, type=ChunkType.CAPTION,
                                    page=current_page, section=current_section))

        elif isinstance(el, CodeSnippet):
            flush_list()
            chunks.append(Chunk(text=f"```\n{text}\n```", type=ChunkType.CODE,
                                page=current_page, section=current_section))

        elif isinstance(el, (NarrativeText, Text)):
            flush_list()
            # Split long paragraphs into ~350-word chunks with overlap
            for sub in _split_long(text, max_words=350, overlap=40):
                chunks.append(Chunk(text=sub, type=ChunkType.TEXT,
                                    page=current_page, section=current_section))
        else:
            # Catch-all for any other element type
            if len(text) > 20:
                flush_list()
                chunks.append(Chunk(text=text, type=ChunkType.UNKNOWN,
                                    page=current_page, section=current_section))

    flush_list()
    return chunks


# ── Helpers ───────────────────────────────────────────────────────────

def _partition(filename: str, ext: str, raw_bytes: bytes) -> list:
    """Use the best unstructured partitioner for each file type."""
    buf = io.BytesIO(raw_bytes)
    if ext == ".pdf":
        from unstructured.partition.pdf import partition_pdf
        return partition_pdf(file=buf, strategy="fast", include_page_breaks=True)
    if ext in (".docx", ".doc"):
        from unstructured.partition.docx import partition_docx
        return partition_docx(file=buf)
    if ext in (".md", ".markdown"):
        from unstructured.partition.md import partition_md
        return partition_md(file=buf)
    if ext in (".html", ".htm"):
        from unstructured.partition.html import partition_html
        return partition_html(file=buf)
    if ext == ".csv":
        from unstructured.partition.csv import partition_csv
        return partition_csv(file=buf)
    if ext in (".txt", ".rst", ".sh", ".py", ".js", ".ts", ".tsx",
               ".jsx", ".json", ".yaml", ".yml", ".toml", ".xml",
               ".go", ".rs", ".rb", ".kt"):
        from unstructured.partition.text import partition_text
        text = raw_bytes.decode("utf-8", errors="replace")
        return partition_text(text=text)
    # Fallback: generic auto-partitioner
    from unstructured.partition.auto import partition
    return partition(file=buf, content_type=_mime(ext), strategy="fast")


def _mime(ext: str) -> Optional[str]:
    return {
        ".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword", ".txt": "text/plain", ".md": "text/markdown",
        ".html": "text/html", ".htm": "text/html", ".xml": "text/xml",
        ".csv": "text/csv", ".tsv": "text/tsv",
        ".py": "text/plain", ".js": "text/plain", ".ts": "text/plain",
        ".tsx": "text/plain", ".json": "application/json",
    }.get(ext)


def _table_to_md(raw: str) -> str:
    """Convert unstructured's table text to a readable markdown-ish format."""
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return raw
    # Simple: join with pipe separators if looks like a grid
    if len(lines) > 1:
        header = "| " + " | ".join(lines[0].split()) + " |"
        sep    = "| " + " | ".join(["---"] * len(lines[0].split())) + " |"
        rows   = ["| " + " | ".join(l.split()) + " |" for l in lines[1:]]
        return "\n".join([header, sep] + rows)
    return raw


def _split_long(text: str, max_words: int = 350, overlap: int = 40) -> list[str]:
    words = re.split(r"\s+", text.strip())
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max(1, max_words - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ── Fallback: simple word-count chunker ──────────────────────────────

def _fallback_chunk(filename: str, raw_bytes: bytes) -> list[Chunk]:
    """Used when unstructured is unavailable or fails."""
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return []
    parts = _split_long(text, max_words=350, overlap=40)
    return [Chunk(text=p, type=ChunkType.TEXT) for p in parts]


# ── Summary stats (shown in upload response) ──────────────────────────

def chunk_summary(chunks: list[Chunk]) -> dict:
    from collections import Counter
    counts = Counter(c.type for c in chunks)
    return {
        "total": len(chunks),
        "by_type": dict(counts),
        "titles": [c.text[:60] for c in chunks if c.type == ChunkType.TITLE][:5],
    }
