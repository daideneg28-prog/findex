"""Split the books into ~N-word passages, one JSON line each (for Lab 8).

    python scripts/make_chunks.py data/gutenberg data/chunks/chunks.jsonl --words 300

Used twice: as a 50k-document corpus where postings lists are long enough for NumPy to
matter, and as the passages that get embedded for semantic search.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findex.corpus import iter_documents

WORD_RE = re.compile(r"\S+")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--words", type=int, default=300)
    args = parser.parse_args()
    titles_file = args.corpus / "titles.json"
    titles = json.loads(titles_file.read_text(encoding="utf-8")) if titles_file.exists() else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as out:
        for doc in iter_documents(args.corpus):
            words = WORD_RE.findall(doc.text)
            title = titles.get(doc.doc_id, doc.doc_id)
            for i in range(0, len(words), args.words):
                chunk = " ".join(words[i : i + args.words])
                part = i // args.words + 1
                record = {
                    "id": f"{doc.doc_id}#{part}",
                    "title": f"{title} (§{part})",
                    "text": chunk,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} passages -> {args.out}")


if __name__ == "__main__":
    main()
