"""Lab 8: semantic and hybrid search.

* Every document is split into passages of ~CHUNK_WORDS words; each passage becomes one
  L2-normalized vector, so cosine similarity is a plain dot product.
* `E @ q` scores all passages at once (brute force — milliseconds for ~50k passages);
  a document's score is its best passage's score (np.maximum.at).
* Hybrid = Reciprocal Rank Fusion of the keyword (BM25) and the semantic ranking:
  score(d) = sum over lists of 1 / (60 + rank). No score scaling needed.

Model: model2vec's `minishlab/potion-base-8M`, a *static* embedding model distilled from
a sentence transformer: 256 dimensions, ~30 MB, pure NumPy at query time, no PyTorch.
It is weaker than all-MiniLM-L6-v2 but fits next to the index in a 512 MB container.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.corpus import read_document
from findex.index import Index
from findex.query import QuerySyntaxError
from findex.rank import SCORERS, search

DEFAULT_MODEL = "minishlab/potion-base-8M"
CHUNK_WORDS = 200
RRF_K = 60
Mode = Literal["keyword", "semantic", "hybrid"]
MODES: tuple[Mode, ...] = ("keyword", "semantic", "hybrid")

_WORDS = re.compile(r"\S+")
type Matrix = npt.NDArray[np.float32]


class Encoder(Protocol):
    def encode(self, sentences: list[str]) -> npt.NDArray[Any]: ...


def load_model(name: str = DEFAULT_MODEL) -> Encoder:
    from model2vec import StaticModel  # optional dependency: `uv sync --extra semantic`

    return StaticModel.from_pretrained(name)


def _normalize(m: npt.NDArray[Any]) -> Matrix:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


def chunks(text: str, words: int = CHUNK_WORDS) -> Iterator[str]:
    tokens = _WORDS.findall(text)
    for i in range(0, len(tokens), words):
        yield " ".join(tokens[i : i + words])


@dataclass
class SemanticIndex:
    """E: float32[n_chunks, d] (unit rows) and chunk_doc: int32[n_chunks] -> doc_id."""

    vectors: Matrix
    chunk_doc: npt.NDArray[np.int32]
    model_name: str
    n_docs: int
    _model: Encoder | None = field(default=None, repr=False)

    @property
    def model(self) -> Encoder:
        if self._model is None:
            self._model = load_model(self.model_name)
        return self._model

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        np.save(folder / "chunk_doc.npy", self.chunk_doc)
        meta = {"model": self.model_name, "n_docs": self.n_docs, "chunk_words": CHUNK_WORDS}
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, model: Encoder | None = None) -> SemanticIndex:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(
            vectors=np.load(folder / "vectors.npy", mmap_mode="r"),  # paged in on demand
            chunk_doc=np.load(folder / "chunk_doc.npy"),
            model_name=meta["model"],
            n_docs=meta["n_docs"],
            _model=model,
        )

    def doc_scores(self, query: str) -> npt.NDArray[np.float32]:
        """Best-passage cosine similarity of every document (-1 for docs without text)."""
        q = _normalize(self.model.encode([query]))[0]
        sims = np.asarray(self.vectors @ q, dtype=np.float32)  # one matrix-vector product
        best = np.full(self.n_docs, -1.0, dtype=np.float32)
        np.maximum.at(best, self.chunk_doc, sims)  # unbuffered: repeated doc ids are fine
        return best

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.doc_scores(query)
        k = min(k, int((scores > -1).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [(int(d), float(scores[d])) for d in top]


def embed_index(
    index: Index,
    model: Encoder,
    model_name: str = DEFAULT_MODEL,
    batch: int = 512,
    progress: Callable[[int], None] | None = None,
) -> SemanticIndex:
    """Chunk every indexed document, embed the chunks, normalize. Returns the matrix."""
    texts: list[str] = []
    owner: list[int] = []
    for doc_id, meta in sorted(index.doc_meta.items()):
        for passage in chunks(read_document(Path(meta.path), meta.name)):
            texts.append(passage)
            owner.append(doc_id)
    parts: list[Matrix] = []
    for start in range(0, len(texts), batch):
        parts.append(_normalize(model.encode(texts[start : start + batch])))
        if progress:
            progress(min(batch, len(texts) - start))
    dim = parts[0].shape[1] if parts else 0
    vectors = np.concatenate(parts) if parts else np.zeros((0, dim), dtype=np.float32)
    return SemanticIndex(
        vectors, np.array(owner, dtype=np.int32), model_name, index.num_docs, model
    )


def count_chunks(index: Index) -> int:
    return sum(
        sum(1 for _ in chunks(read_document(Path(m.path), m.name))) for m in index.doc_meta.values()
    )


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: sum of 1/(k + rank) over the lists a document appears in."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def keyword_ranking(index: Index, query: str, k: int, scorer: str = "bm25") -> list[int]:
    """BM25 over the query words OR-ed together (free text has no AND/NOT syntax)."""
    try:
        results = search(index, query, SCORERS[scorer], k)
    except QuerySyntaxError:
        results = []
    if not results:
        words = [w for w in _WORDS.findall(re.sub(r"[()\"]", " ", query)) if w.isalnum()]
        if words:
            results = search(index, " OR ".join(words), SCORERS[scorer], k)
    return [r.doc_id for r in results]


def ranked(
    index: Index,
    query: str,
    mode: Mode,
    semantic: SemanticIndex | None,
    k: int = 10,
    scorer: str = "bm25",
    depth: int = 100,
) -> list[tuple[int, float]]:
    """(doc_id, score) for any mode; `depth` = list length fed into RRF."""
    if mode == "keyword":
        results = search(index, query, SCORERS[scorer], k)
        return [(r.doc_id, r.score) for r in results]
    if semantic is None:
        raise ValueError("semantic/hybrid search needs embeddings: run `findex embed` first")
    if mode == "semantic":
        return semantic.search(query, k)
    lexical = keyword_ranking(index, query, depth, scorer)
    meaning = [d for d, _ in semantic.search(query, depth)]
    return rrf([lexical, meaning])[:k]
