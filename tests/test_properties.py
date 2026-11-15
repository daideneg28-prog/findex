"""Property-based tests (Hypothesis): invariants that must hold for *any* input."""

import tempfile
import unicodedata
from pathlib import Path

import pytest

# Hypothesis ships a compiled extension; Windows Smart App Control on the author's laptop
# blocks it (ImportError: "An Application Control policy has blocked this file").
# Skip there with a clear reason — CI (Linux) always runs these properties.
pytest.importorskip(
    "hypothesis",
    reason="Hypothesis native extension cannot be loaded here",
    exc_type=ImportError,
)

from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import build_index
from findex.merge import difference, intersect, union
from findex.store import load, save
from findex.tokenize import tokenize

sorted_ids = st.lists(st.integers(0, 200), max_size=60, unique=True).map(sorted)


@given(sorted_ids, sorted_ids)
def test_merges_agree_with_sets(a: list[int], b: list[int]) -> None:
    assert intersect(a, b) == sorted(set(a) & set(b))
    assert union(a, b) == sorted(set(a) | set(b))
    assert difference(a, b) == sorted(set(a) - set(b))


@given(st.text())
def test_tokenize_is_idempotent(text: str) -> None:
    """Re-tokenizing the joined tokens gives the same tokens back."""
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


@given(st.text())
def test_tokens_are_normalized_and_clean(text: str) -> None:
    """Every token is non-empty, has no whitespace or underscore, and is already normalized."""
    for token in tokenize(text):
        assert token
        assert not any(ch.isspace() or ch == "_" for ch in token)
        assert unicodedata.normalize("NFC", token) == token
        assert token.casefold() == token


words = st.sampled_from(["whale", "ship", "sea", "apes", "jungle", "the", "a"])
documents = st.lists(st.lists(words, max_size=12).map(" ".join), min_size=1, max_size=8)


@given(documents)
def test_index_postings_match_the_documents(texts: list[str]) -> None:
    docs = [Document(str(i), Path(f"{i}.txt"), t) for i, t in enumerate(texts)]
    index = build_index(docs, positions=True)
    for term in index:
        expected = [i for i, t in enumerate(texts) if term in t.split()]
        assert list(index.doc_ids(term)) == expected  # sorted, complete, exact
        for p in index[term]:
            assert p.tf == texts[p.doc_id].split().count(term) == len(p.positions)
    assert sum(index.doc_lengths.values()) == sum(len(t.split()) for t in texts)


@settings(max_examples=25, deadline=None)
@given(documents)
def test_json_roundtrip_preserves_postings(texts: list[str]) -> None:
    # Hypothesis re-runs the body many times, so use a fresh temp dir per example
    # (pytest's function-scoped tmp_path fixture would be shared between examples).
    docs = [Document(str(i), Path(f"{i}.txt"), t) for i, t in enumerate(texts)]
    index = build_index(docs, positions=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ix.json"
        save(index, path)
        loaded = load(path)
    assert {t: list(loaded[t]) for t in loaded} == {t: list(index[t]) for t in index}
