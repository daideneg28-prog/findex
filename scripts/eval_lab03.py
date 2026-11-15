"""Lab 3 evidence: three ranking sanity checks and precision@5 for TF-IDF vs BM25.

uv run python scripts/eval_lab03.py data/index.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

from findex.index import Index, Posting, open_index
from findex.rank import BM25, TfIdf, search

# 10 queries with hand-labeled relevant books (Gutenberg ids = DocMeta.name).
# "Relevant" = a book that is clearly *about* the query, judged from titles/contents.
QUERIES: list[tuple[str, set[str]]] = [
    ("tarzan apes jungle", {"78", "81", "85", "90", "92", "106"}),
    ("barsoom martian princess", {"62", "64", "68", "72"}),
    ("tom sawyer huck", {"74", "76", "91", "93"}),
    ("anne gilbert avonlea", {"45", "47", "51"}),
    ("internet email network", {"34", "39", "49", "56", "75", "80", "118"}),
    ("population gdp exports", {"14", "25", "48", "87"}),
    ("dorothy oz scarecrow", {"54", "55"}),
    ("inaugural address fellow citizens", {"3", "8", "9", "104"}),
    ("sara crewe minchin", {"137", "146"}),
    ("paradise satan eden", {"20", "26", "58"}),
]


def precision_at_5(index: Index, query: str, relevant: set[str], scorer) -> float:
    top = search(index, " OR ".join(query.split()), scorer, k=5)
    return sum(index.doc_meta[r.doc_id].name in relevant for r in top) / 5


def sanity_checks(index: Index) -> None:
    bm25, tfidf = BM25(), TfIdf()

    print("### 1. A rare query term outranks a common one\n")
    print("Query `the OR ahab` (BM25):\n")
    results = search(index, "the OR ahab", bm25, k=index.num_docs)
    for r in results[:3]:
        print(f"- {r.score:.3f} {r.title}")
    best_the_only = next(r for r in results if index.doc_meta[r.doc_id].name not in AHAB_DOCS)
    print(f"- … best document with only `the`: {best_the_only.score:.3f} {best_the_only.title}")
    print(f"\nidf(the) = {bm25.idf('the', index):.4f}, idf(ahab) = {bm25.idf('ahab', index):.4f}\n")

    print("### 2. BM25 saturates term frequency\n")
    moby = next(d for d, m in index.doc_meta.items() if m.name == "15")
    print("Score of `whale` for a document of average length, as tf grows:\n")
    print("| tf | BM25 | TF-IDF |\n|---:|---:|---:|")
    avg_doc = _avg_length_doc(index)
    for tf in (1, 2, 5, 10, 19, 20, 100):
        p = Posting(avg_doc, tf)
        print(
            f"| {tf} | {bm25.score('whale', p, index):.3f} | {tfidf.score('whale', p, index):.3f} |"
        )
    limit = bm25.idf("whale", index) * (bm25.k1 + 1)
    print(
        f"\nBM25 limit as tf→∞: idf·(k1+1) = {limit:.3f}. "
        f"(Moby-Dick really has tf(whale) = {_tf(index, 'whale', moby)}.)\n"
    )

    print("### 3. Length normalization: one match in a short vs a long document\n")
    term, short, long_ = _length_pair(index)
    for doc_id in (short, long_):
        p = Posting(doc_id, 1)
        print(
            f"- `{term}` once in *{index.doc_meta[doc_id].title}* "
            f"({index.doc_length(doc_id):,} tokens): BM25 = {bm25.score(term, p, index):.3f}, "
            f"TF-IDF = {tfidf.score(term, p, index):.3f}"
        )
    print()


AHAB_DOCS = {"15", "10", "30", "124", "17"}


def _avg_length_doc(index: Index) -> int:
    return min(index.doc_lengths, key=lambda d: abs(index.doc_lengths[d] - index.avg_doc_length))


def _tf(index: Index, term: str, doc_id: int) -> int:
    return next(p.tf for p in index[term] if p.doc_id == doc_id)


def _length_pair(index: Index) -> tuple[str, int, int]:
    """A term with tf=1 in exactly two documents whose lengths differ the most."""
    best = ("", 0, 0, 0.0)
    for term in index:
        postings = index[term]
        if len(postings) == 2 and all(p.tf == 1 for p in postings):
            a, b = sorted((p.doc_id for p in postings), key=index.doc_length)
            ratio = index.doc_length(b) / max(index.doc_length(a), 1)
            if ratio > best[3] and index.doc_length(a) > 1000:
                best = (term, a, b, ratio)
    return best[0], best[1], best[2]


def main() -> None:
    with open_index(Path(sys.argv[1])) as index:
        sanity_checks(index)
        print("### precision@5 (query terms joined with OR)\n")
        print("| Query | Relevant | TF-IDF P@5 | BM25 P@5 |\n|---|---:|---:|---:|")
        totals = [0.0, 0.0]
        for query, relevant in QUERIES:
            scores = [precision_at_5(index, query, relevant, s) for s in (TfIdf(), BM25())]
            totals = [t + s for t, s in zip(totals, scores, strict=True)]
            print(f"| `{query}` | {len(relevant)} | {scores[0]:.1f} | {scores[1]:.1f} |")
        n = len(QUERIES)
        print(f"| **mean** | | **{totals[0] / n:.2f}** | **{totals[1] / n:.2f}** |")


if __name__ == "__main__":
    main()
