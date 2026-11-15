import subprocess
import sys
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import REPRESENTATIONS, DocMeta, Posting, build_index, open_index
from findex.merge import ENGINES, difference, intersect, union
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


def test_mapping_protocol(index) -> None:
    assert len(index) == len(index.postings)
    assert "whale" in index and "unicorn" not in index
    assert [(p.doc_id, p.tf) for p in index["whale"]] == [(0, 1), (2, 2)]
    with pytest.raises(KeyError):
        index["unicorn"]
    assert set(index) == set(index.postings)
    assert index.get("unicorn") is None  # inherited from Mapping
    assert repr(index).startswith("Index(4 docs, ")


def test_statistics(index) -> None:
    assert index.num_docs == 4
    assert index.avg_doc_length == pytest.approx(19 / 4)
    assert index.doc_length(1) == 6
    assert index.df("whale") == 2 and index.df("unicorn") == 0


def test_index_is_hashable_by_identity(index) -> None:
    assert len({index, index}) == 1
    assert index != build_index(DOCS, index.representation)


def test_dataclasses_are_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert hash(DocMeta(0, "a", "a.txt", "A")) == hash(DocMeta(0, "a", "a.txt", "A"))
    assert not hasattr(Posting(1, 2), "__dict__")


def test_positions() -> None:
    idx = build_index(DOCS, "slots", positions=True)
    assert idx["whale"][1].positions == (1, 3)
    with pytest.raises(ValueError):
        build_index(DOCS, "array", positions=True)


@pytest.mark.parametrize(
    ("a", "b"),
    [([], []), ([1, 3, 5], [2, 3, 4, 5]), ([0, 1, 2], [7, 8]), ([4], [4])],
)
def test_merges_match_sets(a: list[int], b: list[int]) -> None:
    assert intersect(a, b) == sorted(set(a) & set(b)) == ENGINES["set"]["AND"](a, b)
    assert union(a, b) == sorted(set(a) | set(b)) == ENGINES["set"]["OR"](a, b)
    assert difference(a, b) == sorted(set(a) - set(b)) == ENGINES["set"]["NOT"](a, b)


def test_open_index_closes_even_on_error(tmp_path: Path) -> None:
    path = tmp_path / "ix.pkl"
    save(build_index(DOCS), path)
    with open_index(path) as ix:
        assert "whale" in ix
    assert ix.closed and len(ix) == 0
    with pytest.raises(RuntimeError), open_index(path) as ix2:
        raise RuntimeError("boom")
    assert ix2.closed


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
    assert list(load(out).doc_ids("whale")) == [0]


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
