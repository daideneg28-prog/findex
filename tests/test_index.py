import subprocess
import sys
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import REPRESENTATIONS, DocMeta, Posting, build_index
from findex.search import difference, intersect, parse, search, union
from findex.store import load, save

DOCS = [
    Document("a", Path("a.txt"), "The whale and the ship."),
    Document("b", Path("b.txt"), "A ship, a captain, a sea."),
    Document("c", Path("c.txt"), "The whale! The whale!"),
    Document("d", Path("d.txt"), "Nothing to see here."),
]


@pytest.fixture(params=REPRESENTATIONS)
def index(request: pytest.FixtureRequest):
    return build_index(DOCS, request.param)


def test_postings_sorted_with_tf(index) -> None:
    assert list(index.doc_ids("whale")) == [0, 2]
    assert index.entries("whale") == [(0, 1), (2, 2)]
    assert index.doc_lengths == {0: 5, 1: 6, 2: 4, 3: 4}
    assert index.doc_meta[1].name == "b"


def test_dataclasses_are_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert hash(DocMeta(0, "a", "a.txt", "A")) == hash(DocMeta(0, "a", "a.txt", "A"))
    assert not hasattr(Posting(1, 2), "__dict__")


def test_positions() -> None:
    idx = build_index(DOCS, "slots", positions=True)
    assert idx.postings["whale"][1].positions == (1, 3)  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        build_index(DOCS, "array", positions=True)


@pytest.mark.parametrize(
    ("a", "b"),
    [([], []), ([1, 3, 5], [2, 3, 4, 5]), ([0, 1, 2], [7, 8]), ([4], [4])],
)
def test_merges_match_sets(a: list[int], b: list[int]) -> None:
    assert intersect(a, b) == sorted(set(a) & set(b))
    assert union(a, b) == sorted(set(a) | set(b))
    assert difference(a, b) == sorted(set(a) - set(b))


def test_parse() -> None:
    assert parse("Whale, ship OR sea NOT captain") == [
        ("AND", ["whale"]),
        ("AND", ["ship"]),
        ("OR", ["sea"]),
        ("NOT", ["captain"]),
    ]


@pytest.mark.parametrize("engine", ["merge", "set"])
def test_boolean_queries(index, engine) -> None:
    assert search(index, "whale", engine) == [0, 2]
    assert search(index, "whale ship", engine) == [0]
    assert search(index, "whale OR captain", engine) == [0, 1, 2]
    assert search(index, "ship NOT captain", engine) == [0]
    assert search(index, "NOT the", engine) == [1, 3]
    assert search(index, "unicorn", engine) == []
    assert search(index, "", engine) == []


def test_cli_pickle_is_loadable(tmp_path: Path) -> None:
    """Regression: `python -m findex.index` once pickled classes as `__main__.Posting`."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "1.txt").write_text("whale ship", encoding="utf-8")
    out = tmp_path / "index.pkl"
    subprocess.run(
        [sys.executable, "-m", "findex.index", str(corpus), "--out", str(out)],
        check=True,
        capture_output=True,
    )
    assert search(load(out), "whale") == [0]


@pytest.mark.parametrize("suffix", [".pkl", ".json"])
@pytest.mark.parametrize("positions", [False, True])
def test_save_load_roundtrip(tmp_path: Path, suffix: str, positions: bool) -> None:
    for representation in REPRESENTATIONS:
        if positions and representation == "array":
            continue
        idx = build_index(DOCS, representation, positions)
        path = tmp_path / f"{representation}{suffix}"
        save(idx, path)
        loaded = load(path)
        assert loaded.representation == representation
        assert loaded.doc_meta == idx.doc_meta
        assert loaded.doc_lengths == idx.doc_lengths
        for term in idx.postings:
            assert loaded.entries(term) == idx.entries(term)
        if positions:
            assert loaded.postings["whale"] == idx.postings["whale"]
