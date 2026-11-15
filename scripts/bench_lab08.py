"""Lab 8 benchmarks on the full 150-book index.

python scripts/bench_lab08.py            # scorer table + snippet table, markdown
python scripts/bench_lab08.py --index data/chunks-index.pkl --no-snippets   # 48k passages
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findex import vector
from findex.corpus import read_document
from findex.rank import SCORERS, match, search, search_loop
from findex.snippets import MAX_MATCHES, _term_pattern, snippet
from findex.store import load

SCORER_QUERIES = [
    ("common term", "the"),
    ("frequent term", "whale OR ship"),
    ("rare term", "tarzan"),
    ("3-term query", "white AND whale AND ship"),
    ("3 common terms OR", "love OR war OR peace"),
]


def ms(f: Callable[[], object], repeat: int = 7) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def snippet_regex(text: str, terms: Iterable[str], width: int = 80) -> str:
    """The Lab 3-7 implementation, kept verbatim for comparison."""
    pattern = _term_pattern(terms)
    matches = list(islice(pattern.finditer(text), MAX_MATCHES)) if pattern else []
    if not matches:
        return " ".join(text[: 2 * width].split()) + "…"
    best_i, best_j, best_distinct = 0, 0, 0
    j = 0
    for i, m in enumerate(matches):
        j = max(j, i)
        while j + 1 < len(matches) and matches[j + 1].start() - m.start() <= 2 * width:
            j += 1
        distinct = len({x.group().casefold() for x in matches[i : j + 1]})
        if distinct > best_distinct:
            best_i, best_j, best_distinct = i, j, distinct
    center = (matches[best_i].start() + matches[best_j].end()) // 2
    lo, hi = max(0, center - width), min(len(text), center + width)
    window = " ".join(text[lo:hi].split())
    assert pattern is not None
    window = pattern.sub(lambda m: f"«{m.group()}»", window)
    return ("…" if lo > 0 else "") + window + ("…" if hi < len(text) else "")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.pkl")
    parser.add_argument("--no-snippets", action="store_true")
    args = parser.parse_args()
    index = load(args.index)
    bm25 = SCORERS["bm25"]
    print(f"{index!r}\n")

    print(
        "| Query | `q` | Docs matched | Postings scored | Lab 3 loop, ms | NumPy cold, ms | NumPy warm, ms | Speedup (warm) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for label, q in SCORER_QUERIES:
        matched, terms = match(index, q)  # warms the match cache for both
        postings = sum(index.df(t) for t in terms)
        loop_ms = ms(lambda q=q: search_loop(index, q, bm25, 10))
        vector._cache.clear()  # cold: postings -> arrays conversion included
        t0 = time.perf_counter()
        search(index, q, bm25, 10)
        cold = (time.perf_counter() - t0) * 1000
        warm = ms(lambda q=q: search(index, q, bm25, 10))
        assert search(index, q, bm25, 10) == search_loop(index, q, bm25, 10)
        print(
            f"| {label} | `{q}` | {len(matched)} | {postings:,} | {loop_ms:.2f} | {cold:.2f} | "
            f"{warm:.3f} | {loop_ms / warm:.0f}× |"
        )

    if args.no_snippets:
        return
    print(
        "\n| Query | Snippets | Regex (Lab 3), ms | str.find + cached lower(), ms | Speedup | Identical |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for q in ["white whale ship", "love war peace", "tarzan apes jungle", "the", "captain's ship"]:
        top = search(index, q, bm25, 10)
        _, terms = match(index, q)
        docs = []
        for r in top:
            meta = index.doc_meta[r.doc_id]
            text = read_document(Path(meta.path), meta.name)
            docs.append((text, text.lower()))
        old = ms(lambda d=docs, ts=terms: [snippet_regex(t, ts) for t, _ in d], repeat=3)
        new = ms(lambda d=docs, ts=terms: [snippet(t, ts, lowered=low) for t, low in d], repeat=3)
        same = sum(snippet_regex(t, terms) == snippet(t, terms, lowered=low) for t, low in docs)
        print(
            f"| `{q}` | {len(docs)} | {old:.1f} | {new:.1f} | {old / new:.0f}× | {same}/{len(docs)} |"
        )
    _ = re  # (kept for readers comparing with the regex version)


if __name__ == "__main__":
    main()
