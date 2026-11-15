"""Lab 8: performance tests.

`benchmark` tests time the hot paths with pytest-benchmark. They are disabled in normal
runs (`--benchmark-disable` in pyproject: each runs once, as a plain test) and enabled in
CI's benchmark step, which saves the numbers as a JSON artifact.

Absolute times differ between laptops and CI runners, so the test that *fails* on a
regression is relative: on long postings lists the NumPy scorer must stay well ahead of
the Lab 3 loop on the same machine, in the same run.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import Document
from findex.index import Index, build_index
from findex.rank import BM25, SearchResult, search, search_loop

pytestmark = pytest.mark.slow


def _synthetic_docs(n_docs: int, seed: int = 0) -> list[Document]:
    """n_docs documents of 50-250 words over a Zipf-distributed 2,000-word vocabulary."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(2000)]
    ranks = np.arange(1, len(vocab) + 1)
    p = (1 / ranks) / (1 / ranks).sum()
    docs: list[Document] = []
    for i in range(n_docs):
        words = rng.choice(vocab, size=int(rng.integers(50, 250)), p=p)
        docs.append(Document(str(i), Path(f"{i}.txt"), " ".join(words)))
    return docs


@pytest.fixture(scope="module")
def big_index() -> Index:
    return build_index(_synthetic_docs(5000))


def _best_of(f: Callable[[], object], n: int = 5) -> float:
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def test_numpy_scorer_beats_loop_on_long_postings(big_index: Index) -> None:
    query = "w0 OR w1 OR w2"  # the three most frequent words: ~5,000 postings each
    assert search(big_index, query, BM25(), 10) == search_loop(big_index, query, BM25(), 10)
    loop = _best_of(lambda: search_loop(big_index, query, BM25(), 10))
    fast = _best_of(lambda: search(big_index, query, BM25(), 10))
    assert loop / fast > 3, f"NumPy scorer only {loop / fast:.1f}x faster than the loop"


def test_bench_search_common_terms(benchmark: BenchmarkFixture, big_index: Index) -> None:
    results = cast(list[SearchResult], benchmark(search, big_index, "w0 OR w1 OR w2", BM25(), 10))
    assert len(results) == 10


def test_bench_search_rare_and(benchmark: BenchmarkFixture, big_index: Index) -> None:
    benchmark(search, big_index, "w500 AND w900", BM25(), 10)


def test_bench_build_index(benchmark: BenchmarkFixture) -> None:
    docs = _synthetic_docs(500, seed=1)
    index = cast(Index, benchmark(build_index, docs))
    assert index.num_docs == 500
