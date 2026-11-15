"""Lab 8: semantic + hybrid search, with a tiny deterministic encoder (no model download)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from findex.index import Index
from findex.semantic import SemanticIndex, chunks, embed_index, keyword_ranking, ranked, rrf
from findex.web.app import create_app, index_state, semantic_state
from findex.web.settings import Settings

# three "concepts"; a passage's vector counts the words of each concept it contains
CONCEPTS = [
    {"whale", "ship", "sea", "captain", "ishmael", "ocean", "sailor"},
    {"tarzan", "apes", "jungle", "primates", "forest"},
    {"nothing", "see", "here"},
]


class FakeEncoder:
    def encode(self, sentences: list[str]) -> Any:
        out = np.zeros((len(sentences), len(CONCEPTS) + 1), dtype=np.float32)
        for i, s in enumerate(sentences):
            words = s.lower().replace(".", " ").replace(",", " ").split()
            for j, concept in enumerate(CONCEPTS):
                out[i, j] = sum(w in concept for w in words)
            out[i, -1] = 0.01  # never an all-zero vector
        return out


@pytest.fixture
def semantic(tiny_index: Index) -> SemanticIndex:
    return embed_index(tiny_index, FakeEncoder(), "fake")


def title(index: Index, doc_id: int) -> str:
    return index.doc_meta[doc_id].title


def test_chunks() -> None:
    assert list(chunks("a b c d e", words=2)) == ["a b", "c d", "e"]
    assert list(chunks("", words=2)) == []


def test_vectors_are_unit_length(semantic: SemanticIndex, tiny_index: Index) -> None:
    norms = np.linalg.norm(semantic.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert semantic.vectors.dtype == np.float32
    assert set(semantic.chunk_doc.tolist()) == set(tiny_index.doc_meta)


def test_semantic_finds_meaning_without_shared_words(
    semantic: SemanticIndex, tiny_index: Index
) -> None:
    # "primates forest" occurs in no document, but means the same as "apes jungle"
    [(best, score), *_] = semantic.search("primates forest", 4)
    assert title(tiny_index, best) == "Jungle Book" and score > 0.9
    assert keyword_ranking(tiny_index, "primates forest", 10) == []  # keywords find nothing


def test_save_load_roundtrip(semantic: SemanticIndex, tmp_path: Path) -> None:
    semantic.save(tmp_path / "e")
    loaded = SemanticIndex.load(tmp_path / "e", model=FakeEncoder())
    assert loaded.model_name == "fake" and loaded.n_docs == semantic.n_docs
    assert loaded.search("ocean sailor", 2) == semantic.search("ocean sailor", 2)


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [d for d, _ in fused] == [1, 3, 2]  # 1: 1/61+1/62, 3: 1/63+1/61, 2: 1/62
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_modes(semantic: SemanticIndex, tiny_index: Index) -> None:
    keyword = ranked(tiny_index, "whale", "keyword", None, 3)
    assert [title(tiny_index, d) for d, _ in keyword] == ["Whale Book"]
    hybrid = ranked(tiny_index, "whale ocean", "hybrid", semantic, 3)
    assert title(tiny_index, hybrid[0][0]) == "Whale Book"
    with pytest.raises(ValueError, match="embed"):
        ranked(tiny_index, "whale", "semantic", None, 3)


def test_keyword_ranking_falls_back_to_or(tiny_index: Index) -> None:
    # AND of both words matches nothing; the OR fallback still ranks something
    assert keyword_ranking(tiny_index, "whale tarzan", 5) != []
    assert keyword_ranking(tiny_index, "(whale", 5) != []  # syntax error -> words OR-ed


def test_web_modes(tiny_index: Index, semantic: SemanticIndex, tmp_path: Path) -> None:
    app = create_app(Settings(index_path=tmp_path / "none.pkl", log_json=False))
    app.dependency_overrides[index_state] = lambda: tiny_index
    app.dependency_overrides[semantic_state] = lambda: semantic
    with TestClient(app) as client:
        body = client.get("/search", params={"q": "primates forest", "mode": "semantic"}).json()
        assert body["mode"] == "semantic" and body["results"][0]["title"] == "Jungle Book"
        hybrid = client.get("/search", params={"q": "whale ocean", "mode": "hybrid"}).json()
        assert hybrid["results"][0]["title"] == "Whale Book"
        assert client.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422
        page = client.get("/", params={"q": "primates forest", "mode": "semantic"})
        assert "Jungle Book" in page.text and 'value="semantic" selected' in page.text
        app.dependency_overrides[semantic_state] = lambda: None
        off = client.get("/search", params={"q": "whale", "mode": "hybrid"})
        assert off.status_code == 503 and "not available" in off.json()["detail"]
        assert "disabled" in client.get("/").text  # the UI greys the modes out
