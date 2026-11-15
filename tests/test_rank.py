import logging
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, Posting, build_index
from findex.rank import BM25, SearchResult, TfIdf, cache_hits, match, search
from findex.snippets import snippet
from findex.timing import timed

FILLER = " ".join(f"filler{i}" for i in range(200))
DOCS = [
    Document("short", Path("s.txt"), "the zebra"),
    Document("long", Path("l.txt"), f"the zebra {FILLER}"),
    Document("common", Path("c.txt"), "the the the the cat"),
    Document("rep", Path("r.txt"), "the dog " + "dog " * 19),
    Document("pad", Path("p.txt"), "the cat sat " * 5),
]


@pytest.fixture(scope="module")
def index() -> Index:
    return build_index(DOCS, positions=True)


def test_rare_term_outranks_common_one(index: Index) -> None:
    results = search(index, "the OR zebra", BM25())
    assert {r.doc_id for r in results[:2]} == {0, 1}  # zebra docs first
    assert results[0].score > max(r.score for r in results if r.doc_id == 2)


def test_bm25_saturates_tf(index: Index) -> None:
    bm25, tfidf = BM25(), TfIdf()
    gain = [bm25.score("dog", Posting(3, tf), index) for tf in (1, 2, 19, 20)]
    assert gain[1] - gain[0] > 10 * (gain[3] - gain[2])
    limit = bm25.idf("dog", index) * (bm25.k1 + 1)
    assert gain[3] < limit
    # TF-IDF keeps growing with log(tf)
    assert tfidf.score("dog", Posting(3, 20), index) > 3 * tfidf.score("dog", Posting(3, 1), index)


def test_bm25_prefers_short_document(index: Index) -> None:
    results = search(index, "zebra", BM25())
    assert [r.doc_id for r in results] == [0, 1]
    assert results[0].score > results[1].score
    no_norm = search(index, "zebra", BM25(b=0))
    assert no_norm[0].score == pytest.approx(no_norm[1].score)  # b=0: length ignored


def test_top_k_and_ordering(index: Index) -> None:
    results = search(index, "the", k=2)
    assert len(results) == 2
    assert sorted(results, reverse=True) == results
    assert SearchResult(1.0, 5, "x") < SearchResult(2.0, 0, "a")


def test_empty_and_negated_queries(index: Index) -> None:
    assert search(index, "") == []
    assert search(index, "unicorn") == []
    only_not = search(index, "NOT the")
    assert only_not == []  # every document contains "the"


def test_repeated_query_hits_cache(index: Index) -> None:
    match(index, "zebra OR cat")
    before = cache_hits()
    match(index, "zebra OR cat")
    assert cache_hits() == before + 1


def test_timed_preserves_metadata_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def double(x: int) -> int:
        """Docstring survives."""
        return 2 * x

    with caplog.at_level(logging.INFO, logger="findex.timing"):
        assert double(21) == 42
    assert double.__name__ == "double" and double.__doc__ == "Docstring survives."
    assert "double took" in caplog.text


def test_snippet_highlights_best_window() -> None:
    text = "whale " + "x " * 200 + "the white whale swam near the ship " + "y " * 200
    s = snippet(text, ["white", "whale", "ship"], width=40)
    assert "«white» «whale»" in s and "«ship»" in s
    assert s.startswith("…") and s.endswith("…")
    assert snippet("no match here", ["zebra"]).startswith("no match here")
    assert "«don’t»" in snippet("I don’t know", ["don't"])
