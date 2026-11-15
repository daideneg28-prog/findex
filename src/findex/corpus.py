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
            yield Document(doc_id, path, str(obj.get("text", "")))


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield documents under `root` one at a time.

    `root` may be a directory (every *.txt and *.jsonl below it, found lazily
    with rglob) or a single .txt / .jsonl file.
    """
    root = Path(root)
    paths = [root] if root.is_file() else root.rglob("*")
    for path in paths:
        if path.suffix == ".txt":
            yield Document(path.stem, path, _read_text(path))
        elif path.suffix == ".jsonl":
            yield from _iter_jsonl(path)
