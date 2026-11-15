"""Corpus statistics in a single streaming pass.

python -m findex.stats data/ [--limit N] [--top 50] [--mode lazy|eager]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    counts: Counter[str]

    @property
    def vocabulary(self) -> int:
        return len(self.counts)


def compute_stats(docs: Iterable[Document]) -> CorpusStats:
    """Lazy: documents and tokens stream through; only the Counter grows."""
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))
    return CorpusStats(n_docs, counts.total(), counts)


def compute_stats_eager(docs: Iterable[Document]) -> CorpusStats:
    """Deliberately eager: every stage is materialized as a list (for M4)."""
    all_docs = list(docs)
    all_tokens = [list(tokenize(doc.text)) for doc in all_docs]
    flat = [tok for doc_tokens in all_tokens for tok in doc_tokens]
    counts = Counter(flat)
    return CorpusStats(len(all_docs), len(flat), counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m findex.stats", description=__doc__)
    parser.add_argument("root", type=Path, help="corpus directory or .txt/.jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--top", type=int, default=50, help="how many top terms to print")
    parser.add_argument("--mode", choices=["lazy", "eager"], default="lazy")
    args = parser.parse_args(argv)

    docs = islice(iter_documents(args.root), args.limit)
    compute = compute_stats if args.mode == "lazy" else compute_stats_eager

    tracemalloc.start()
    t0 = time.perf_counter()
    stats = compute(docs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"mode:        {args.mode}")
    print(f"documents:   {stats.documents:,}")
    print(f"tokens:      {stats.tokens:,}")
    print(f"vocabulary:  {stats.vocabulary:,}")
    print(f"elapsed:     {elapsed:.2f} s")
    print(f"peak memory: {peak / 2**20:.1f} MiB")
    print(f"\ntop {args.top} terms:")
    for rank, (term, count) in enumerate(stats.counts.most_common(args.top), start=1):
        print(f"{rank:>3}. {term:<15} {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
