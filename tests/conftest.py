"""Shared fixtures: a tiny on-disk corpus and indexes built from it."""

import json
from pathlib import Path

import pytest

from findex.corpus import Document, iter_documents
from findex.index import Index, build_index
from findex.store import save

TINY_CORPUS = {
    "1": ("Whale Book", "Call me Ishmael. The white whale swam near the old ship."),
    "2": ("Ship Book", "A ship, a captain and a sea. The captain's ship was old."),
    "3": ("Jungle Book", "Tarzan of the apes lived in the jungle with the apes."),
    "4": ("Empty-ish", "Nothing to see here."),
}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for name, (_, text) in TINY_CORPUS.items():
        (root / f"{name}.txt").write_text(text, encoding="utf-8")
    titles = {name: title for name, (title, _) in TINY_CORPUS.items()}
    (root / "titles.json").write_text(json.dumps(titles), encoding="utf-8")
    return root


@pytest.fixture
def tiny_docs(corpus_dir: Path) -> list[Document]:
    return sorted(iter_documents(corpus_dir), key=lambda d: d.doc_id)


@pytest.fixture
def tiny_index(tiny_docs: list[Document]) -> Index:
    return build_index(tiny_docs, positions=True)


@pytest.fixture
def index_file(tiny_index: Index, tmp_path: Path) -> Path:
    path = tmp_path / "tiny.pkl"
    save(tiny_index, path)
    return path
