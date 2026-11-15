"""Lazy corpus loading: one document in memory at a time."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str
    title: str = ""  # known for crawled pages (JSONL "title"); books get theirs from titles.json


def _read_text(path: Path) -> str:
    """Read a file as UTF-8; on bad bytes, log and fall back to replacement chars."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        log.warning("bad encoding in %s (%s); decoding with errors='replace'", path, exc)
        return path.read_text(encoding="utf-8", errors="replace")


def _iter_jsonl(path: Path) -> Iterator[Document]:
    """One JSON object per line: {"id": ..., "text": ...}. Streams line by line."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d: invalid JSON (%s); skipped", path, lineno, exc)
                continue
            doc_id = str(obj.get("id", f"{path.stem}:{lineno}"))
            yield Document(doc_id, path, str(obj.get("text", "")), str(obj.get("title", "")))


SUFFIXES = (".txt", ".jsonl")


def iter_paths(root: Path) -> Iterator[Path]:
    """Corpus files under `root` (or `root` itself if it is a file), in a fixed order.

    The order defines doc_ids, so both the serial and the parallel indexer use this.
    """
    root = Path(root)
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if path.suffix in SUFFIXES and path.is_file():
            yield path


def iter_file(path: Path) -> Iterator[Document]:
    """The documents of one corpus file: one for .txt, one per line for .jsonl."""
    if path.suffix == ".txt":
        yield Document(path.stem, path, _read_text(path))
    elif path.suffix == ".jsonl":
        yield from _iter_jsonl(path)


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents under `root` one at a time.

    `root` may be a directory (every *.txt and *.jsonl below it, found lazily
    with rglob) or a single .txt / .jsonl file.
    """
    for path in iter_paths(root):
        yield from iter_file(path)


def read_document(path: Path, doc_id: str) -> str:
    """The text of one document, e.g. for a snippet: the whole .txt file, or the one
    line of a .jsonl file whose id is `doc_id` (streamed, the file is not loaded)."""
    path = Path(path)
    if path.suffix != ".jsonl":
        return _read_text(path)
    return next((doc.text for doc in _iter_jsonl(path) if doc.doc_id == doc_id), "")
