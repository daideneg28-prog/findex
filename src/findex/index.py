"""Inverted index: term -> postings sorted by doc_id, built in one streaming pass.

python -m findex.index data/ --out index.pkl [--repr slots|plain|array] [--positions]
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

Representation = Literal["plain", "slots", "array"]
REPRESENTATIONS: tuple[Representation, ...] = ("plain", "slots", "array")


@dataclass(frozen=True, slots=True)
class Posting:
    """One document in a term's postings list. Frozen + slotted: hashable, no __dict__."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlainPosting:
    """Same fields as Posting but without slots: every instance carries a __dict__ (M4 only)."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    name: str
    path: str
    title: str


# "array" representation: two parallel buffers of raw uint32 — (doc_ids, tfs)
ArrayPostings = tuple[array, array]
PostingList = list[Posting] | list[PlainPosting] | ArrayPostings


@dataclass
class Index:
    representation: Representation
    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    has_positions: bool = False

    @property
    def n_docs(self) -> int:
        return len(self.doc_meta)

    @property
    def n_postings(self) -> int:
        return sum(len(self.doc_ids(term)) for term in self.postings)

    def doc_ids(self, term: str) -> Sequence[int]:
        """Sorted doc_ids that contain `term` (empty if the term is unknown)."""
        entry = self.postings.get(term)
        if entry is None:
            return ()
        if self.representation == "array":
            return entry[0]  # type: ignore[index]
        return [p.doc_id for p in entry]  # type: ignore[union-attr]

    def entries(self, term: str) -> list[tuple[int, int]]:
        """(doc_id, tf) pairs for `term`, whatever the representation."""
        entry = self.postings.get(term)
        if entry is None:
            return []
        if self.representation == "array":
            ids, tfs = entry  # type: ignore[misc]
            return list(zip(ids, tfs, strict=True))
        return [(p.doc_id, p.tf) for p in entry]  # type: ignore[union-attr]


class _Titles:
    """Looks up book titles in a titles.json next to the documents (written by the downloader)."""

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, str]] = {}

    def get(self, doc: Document) -> str:
        folder = doc.path.parent
        if folder not in self._cache:
            file = folder / "titles.json"
            self._cache[folder] = (
                json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
            )
        return self._cache[folder].get(doc.doc_id, doc.doc_id)


def build_index(
    docs: Iterable[Document],
    representation: Representation = "slots",
    positions: bool = False,
) -> Index:
    """Consume the lazy document stream once; only the index grows.

    Each document is tokenized into its own Counter (or position lists), which is
    flushed into the global postings when the document ends. doc_ids are assigned in
    stream order, so every postings list is already sorted by doc_id.
    """
    if positions and representation == "array":
        raise ValueError("positions are only supported for the plain/slots representations")

    postings: defaultdict[str, list] | defaultdict[str, ArrayPostings]
    if representation == "array":
        postings = defaultdict(lambda: (array("I"), array("I")))
    else:
        postings = defaultdict(list)
    cls = Posting if representation == "slots" else PlainPosting

    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    titles = _Titles()

    for doc_id, doc in enumerate(docs):
        if positions:
            offsets: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for length, token in enumerate(tokenize(doc.text), start=1):
                offsets[token].append(length - 1)
            for term, offs in offsets.items():
                postings[term].append(cls(doc_id, len(offs), tuple(offs)))
        else:
            counts = Counter(tokenize(doc.text))
            length = counts.total()
            for term, tf in counts.items():
                if representation == "array":
                    ids, tfs = postings[term]
                    ids.append(doc_id)
                    tfs.append(tf)
                else:
                    postings[term].append(cls(doc_id, tf))
        doc_lengths[doc_id] = length
        doc_meta[doc_id] = DocMeta(doc_id, doc.doc_id, str(doc.path), titles.get(doc))

    # plain dict: no default factory surprises at query time, and a lambda can't be pickled
    return Index(representation, dict(postings), doc_lengths, doc_meta, positions)


def main(argv: list[str] | None = None) -> int:
    from findex.store import save  # local import: store imports this module

    parser = argparse.ArgumentParser(prog="python -m findex.index", description=__doc__)
    parser.add_argument("root", type=Path, help="corpus directory or .txt/.jsonl file")
    parser.add_argument("--out", type=Path, required=True, help="index file (.pkl or .json)")
    parser.add_argument("--repr", choices=REPRESENTATIONS, default="slots", dest="representation")
    parser.add_argument("--positions", action="store_true", help="store token positions")
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(
        islice(iter_documents(args.root), args.limit), args.representation, args.positions
    )
    build_s = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    save(index, args.out)
    save_s = time.perf_counter() - t0

    print(f"documents:    {index.n_docs:,}")
    print(f"terms:        {len(index.postings):,}")
    print(f"postings:     {index.n_postings:,}")
    print(f"build:        {build_s:.2f} s")
    print(f"index in RAM: {current / 2**20:.1f} MiB (peak {peak / 2**20:.1f} MiB)")
    print(f"saved:        {args.out} ({args.out.stat().st_size / 2**20:.1f} MiB, {save_s:.2f} s)")
    return 0


if __name__ == "__main__":
    # Under `python -m findex.index` this file runs as `__main__`, so its classes would be
    # pickled as `__main__.Posting` and could not be unpickled by `findex.search`.
    # Run main() from the properly imported module so pickle records `findex.index.Posting`.
    from findex import index as _index_module  # noqa: PLW0406 — intentional, see above

    raise SystemExit(_index_module.main())
