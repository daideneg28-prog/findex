"""Lab 8: postings as NumPy arrays and BM25 / TF-IDF as array expressions.

The scores are *bit-identical* to the Lab 3 loop in rank.py, because every element
goes through the same IEEE operations in the same order:

    Lab 3:  idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len/avg))     # one doc at a time
    here:   idf * tfs * (k1 + 1) / (tfs + k1 * (1 - b + b * lens/avg))  # all docs at once

Two details keep it exact: idf is a scalar computed with math.log (np.log may differ in
the last bit), and TF-IDF's log(tf) is taken with math.log on the few distinct tf values
of a term, then spread with fancy indexing.
"""

from __future__ import annotations

import math
import weakref
from array import array
from typing import Any

import numpy as np
import numpy.typing as npt

from findex.index import Index

type IntArray = npt.NDArray[np.int32]
type FloatArray = npt.NDArray[np.float64]


class _Arrays:
    """Per-index cache: postings converted to int32 arrays on first use."""

    def __init__(self, index: Index) -> None:
        n = index.num_docs
        lengths = np.zeros(n, dtype=np.int32)
        for doc_id, length in index.doc_lengths.items():
            lengths[doc_id] = length
        self.doc_len: IntArray = lengths
        self.terms: dict[str, tuple[IntArray, IntArray]] = {}

    def postings(self, index: Index, term: str) -> tuple[IntArray, IntArray]:
        cached = self.terms.get(term)
        if cached is not None:
            return cached
        entry = index.postings.get(term)
        if entry is None:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty
        if isinstance(entry, tuple):  # "array" representation: zero-copy views of array('I')
            ids_buf, tfs_buf = entry
            ids = _view(ids_buf)
            tfs = _view(tfs_buf)
        else:
            n = len(entry)
            ids = np.fromiter((p.doc_id for p in entry), dtype=np.int32, count=n)
            tfs = np.fromiter((p.tf for p in entry), dtype=np.int32, count=n)
        self.terms[term] = (ids, tfs)
        return ids, tfs


def _view(buf: array[int]) -> IntArray:
    """np.frombuffer: a NumPy view of the array's memory — no copy (uint32 -> int32 is safe
    while ids and tfs stay below 2**31)."""
    raw: npt.NDArray[Any] = np.frombuffer(buf, dtype=np.uint32)
    return raw.view(np.int32)


_cache: weakref.WeakKeyDictionary[Index, _Arrays] = weakref.WeakKeyDictionary()


def arrays(index: Index) -> _Arrays:
    found = _cache.get(index)
    if found is None:
        found = _cache[index] = _Arrays(index)
    return found


def bm25_scores(
    index: Index, terms: tuple[str, ...], k1: float = 1.5, b: float = 0.75
) -> FloatArray:
    """BM25 of every document for `terms` (0 where no term occurs): one float64[N]."""
    arr = arrays(index)
    scores = np.zeros(index.num_docs, dtype=np.float64)
    n, avg = index.num_docs, index.avg_doc_length
    for term in terms:
        ids, tfs = arr.postings(index, term)
        if not len(ids):
            continue
        df = len(ids)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = tfs.astype(np.float64)
        length_ratio = arr.doc_len[ids] / avg  # fancy indexing: lengths of these docs
        norm = k1 * (1 - b + b * length_ratio)
        scores[ids] += idf * tf * (k1 + 1) / (tf + norm)  # ids are unique: plain += is safe
    return scores


def tfidf_scores(index: Index, terms: tuple[str, ...]) -> FloatArray:
    arr = arrays(index)
    scores = np.zeros(index.num_docs, dtype=np.float64)
    for term in terms:
        ids, tfs = arr.postings(index, term)
        if not len(ids):
            continue
        idf = math.log(index.num_docs / len(ids))
        distinct, inverse = np.unique(tfs, return_inverse=True)
        logs = np.array([math.log(int(t)) for t in distinct], dtype=np.float64)
        scores[ids] += (1 + logs[inverse]) * idf
    return scores


def top_k(scores: FloatArray, candidates: IntArray, k: int) -> list[tuple[int, float]]:
    """The k best candidates by (score desc, doc_id asc) — the same order heapq.nlargest
    with key (score, -doc_id) gives in rank.py.

    np.argpartition finds the k largest in O(n) instead of sorting all n (O(n log n));
    only those k are then sorted. Ties at the boundary are resolved by doc_id.
    """
    if k <= 0 or not len(candidates):
        return []
    s = scores[candidates]
    if len(candidates) > k:
        part = np.argpartition(-s, k - 1)[:k]
        threshold = s[part].min()
        above = candidates[s > threshold]
        tied = np.sort(candidates[s == threshold])[: k - len(above)]
        candidates = np.concatenate([above, tied])
        s = scores[candidates]
    order = np.lexsort((candidates, -s))  # last key is primary: -score, then doc_id
    return [(int(candidates[i]), float(s[i])) for i in order]
