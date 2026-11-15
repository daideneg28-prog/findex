"""Ranking: swappable scorers (TF-IDF, BM25) and top-k search."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from findex.index import Index, PostingLike
from findex.query import parse
from findex.timing import timed


class Scorer(Protocol):
    """Anything with this method is a scorer — no inheritance needed (structural typing)."""

    def score(self, term: str, posting: PostingLike, index: Index) -> float: ...


@dataclass(frozen=True)
class TfIdf:
    """(1 + log tf) · log(N / df). Sub-linear tf, but no saturation and no length norm."""

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        return (1 + math.log(posting.tf)) * idf


@dataclass(frozen=True)
class BM25:
    """Okapi BM25.

    k1 controls tf saturation: score(tf) → idf·(k1+1) as tf → ∞.
    b controls length normalization: b=0 ignores document length, b=1 normalizes fully.
    """

    k1: float = 1.5
    b: float = 0.75

    def idf(self, term: str, index: Index) -> float:
        n, df = index.num_docs, index.df(term)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))  # +1 inside keeps it positive

    def score(self, term: str, posting: PostingLike, index: Index) -> float:
        tf = posting.tf
        length_ratio = index.doc_length(posting.doc_id) / index.avg_doc_length
        norm = self.k1 * (1 - self.b + self.b * length_ratio)
        return self.idf(term, index) * tf * (self.k1 + 1) / (tf + norm)


SCORERS: dict[str, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}


@dataclass(frozen=True, order=True)
class SearchResult:
    """order=True: results compare by (score, doc_id), so sorted(results) works."""

    score: float
    doc_id: int
    title: str = field(compare=False)


@lru_cache(maxsize=256)
def _match_cached(index: Index, query: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    node = parse(query)
    terms = tuple(dict.fromkeys(node.terms()))  # unique, in query order
    return tuple(node.evaluate(index)), terms


@timed
def match(index: Index, query: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """(matching doc_ids, scoring terms) for a query; memoized per (index, query).

    lru_cache needs hashable arguments: the query is a str and Index hashes by identity.
    A repeated query skips parsing and evaluation entirely — the timing log shows it.
    """
    return _match_cached(index, query)


def cache_hits() -> int:
    return _match_cached.cache_info().hits


def cache_info() -> str:
    info = _match_cached.cache_info()
    return f"query cache: {info.hits} hits, {info.misses} misses, {info.currsize} cached"


@timed
def search(
    index: Index, query: str, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]:
    """Match with the query tree, then rank the matched documents and keep the top k.

    BM25 and TF-IDF are scored with NumPy array expressions (Lab 8, findex.vector);
    any other Scorer falls back to the per-posting loop of Lab 3.
    """
    if not query.strip():
        return []
    scorer = scorer or BM25()
    if not isinstance(scorer, BM25 | TfIdf):
        return search_loop(index, query, scorer, k)
    from findex import vector  # numpy is imported only when actually scoring

    matched, terms = match(index, query)
    present = tuple(t for t in terms if t in index)
    if isinstance(scorer, BM25):
        scores = vector.bm25_scores(index, present, scorer.k1, scorer.b)
    else:
        scores = vector.tfidf_scores(index, present)
    candidates = vector.np.fromiter(matched, dtype=vector.np.int32, count=len(matched))
    top = vector.top_k(scores, candidates, k)
    return [SearchResult(score, doc_id, index.doc_meta[doc_id].title) for doc_id, score in top]


def search_loop(
    index: Index, query: str, scorer: Scorer | None = None, k: int = 10
) -> list[SearchResult]:
    """The Lab 3 scorer: a Python loop over every posting of every query term."""
    if not query.strip():
        return []
    scorer = scorer or BM25()
    matched, terms = match(index, query)
    scores = dict.fromkeys(matched, 0.0)
    for term in terms:
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer.score(term, posting, index)
    # O(n log k) instead of sorting all n; ties broken by lower doc_id
    top = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
    return [SearchResult(score, doc_id, index.doc_meta[doc_id].title) for doc_id, score in top]
