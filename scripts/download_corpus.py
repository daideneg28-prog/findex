"""Download a slice of Project Gutenberg into data/gutenberg/ as plain-text files.

Usage:
    uv run python scripts/download_corpus.py --count 150

Picks English-language texts from the official Gutenberg catalog (pg_catalog.csv),
downloads each book's UTF-8 plain text, strips the Gutenberg licence header/footer,
and writes data/gutenberg/<id>.txt. Already-downloaded books are skipped, and the
script sleeps between requests to be polite to gutenberg.org.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
USER_AGENT = "findex-course-corpus-downloader/1.0"

START_RE = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)
END_RE = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def english_text_ids(limit: int) -> list[int]:
    """Return the first `limit` English text ids from the catalog, lowest id first."""
    raw = fetch(CATALOG_URL).decode("utf-8")
    ids: list[int] = []
    for row in csv.DictReader(io.StringIO(raw)):
        if row.get("Type") == "Text" and row.get("Language") == "en":
            ids.append(int(row["Text#"]))
    ids.sort()
    return ids[:limit]


def strip_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    end = END_RE.search(text)
    lo = start.end() if start else 0
    hi = end.start() if end else len(text)
    return text[lo:hi].strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=150, help="number of books")
    parser.add_argument("--out", type=Path, default=Path("data/gutenberg"))
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    ids = english_text_ids(args.count * 2)  # some ids have no .txt; ask for spares
    saved = len(list(args.out.glob("*.txt")))
    for book_id in ids:
        if saved >= args.count:
            break
        target = args.out / f"{book_id}.txt"
        if target.exists():
            continue
        try:
            raw = fetch(BOOK_URL.format(id=book_id))
        except OSError as exc:
            print(f"skip {book_id}: {exc}", file=sys.stderr)
            continue
        text = strip_boilerplate(raw.decode("utf-8", errors="replace"))
        target.write_text(text, encoding="utf-8")
        saved += 1
        print(f"[{saved}/{args.count}] {book_id}.txt  {len(text) / 1e6:.2f} MB")
        time.sleep(args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
