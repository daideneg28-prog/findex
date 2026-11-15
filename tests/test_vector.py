"""Lab 8: the NumPy scorer must give exactly the Lab 3 ranking (same scores, same order)."""

from pathlib import Path

import numpy as np
import pytest

from findex.corpus import Document, iter_documents
from findex.index import Index, PostingLike, Representation, build_index
from findex.rank import BM25, SCORERS, TfIdf, search, search_loop
from findex.vector import arrays, bm25_scores, top_k

QUERIES = [
    "the",
    "whale",
    "ship OR apes",
    "the AND old",
    '"the apes"',
    "captain's",
    "the NOT ship",
    "nothing",
    "zzz",
    "(ship OR whale) AND old",
]


def _bigger_corpus(corpus_dir: Path) -> Path:
    words = [
        "the",
        "ship",
        "whale",
        "apes",
        "old",
        "sea",
        "captain",
        "jungle",
        "tarzan",
        "white",
        "call",
    ]
    rng = np.random.default_rng(7)
    for i in range(5, 60):
        text = " ".join(rng.choice(words, size=int(rng.integers(3, 200))))
        (corpus_dir / f"{i}.txt").write_text(text, encoding="utf-8")
    return corpus_dir


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("representation", ["slots", "array"])
def test_identical_ranking(corpus_dir: Path, scorer: str, representation: Representation) -> None:
    index = build_index(
        iter_documents(_bigger_corpus(corpus_dir)), representation, representation != "array"
    )
    for query in QUERIES:
        if '"' in query and representation == "array":
            continue  # phrases need positions
        for k in (1, 3, 10, 100):
            fast = search(index, query, SCORERS[scorer], k)
            slow = search_loop(index, query, SCORERS[scorer], k)
            assert fast == slow, (query, k)  # SearchResult compares score and doc_id
            assert [r.score for r in fast] == [r.score for r in slow]  # bit-identical floats


def test_ties_resolved_by_doc_id() -> None:
    scores = np.array([1.0, 5.0, 5.0, 5.0, 2.0])
    candidates = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    assert top_k(scores, candidates, 2) == [(1, 5.0), (2, 5.0)]
    assert top_k(scores, candidates, 10)[-1] == (0, 1.0)
    assert top_k(scores, candidates[:0], 3) == []
    assert top_k(scores, candidates, 0) == []


def test_array_representation_is_zero_copy(tiny_docs: list[Document]) -> None:
    index: Index = build_index(tiny_docs, "array")
    ids, _ = arrays(index).postings(index, "the")
    assert not ids.flags.owndata  # a view of the array('I') buffer, not a copy
    assert arrays(index).postings(index, "missing")[0].size == 0


def test_unknown_scorer_falls_back_to_loop(tiny_index: Index) -> None:
    class Constant:
        def score(self, term: str, posting: PostingLike, index: Index) -> float:
            return 1.0

    results = search(tiny_index, "the", Constant(), 10)
    assert {r.score for r in results} == {1.0}


def test_bm25_scores_shape(tiny_index: Index) -> None:
    scores = bm25_scores(tiny_index, ("whale",))
    assert scores.shape == (tiny_index.num_docs,) and (scores > 0).sum() == 1
    assert isinstance(SCORERS["bm25"], BM25) and isinstance(SCORERS["tfidf"], TfIdf)
