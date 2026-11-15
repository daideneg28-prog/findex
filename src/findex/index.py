"""Inverted index: term -> postings sorted by doc_id, built in one streaming pass.

python -m findex.index data/ --out index.pkl [--repr slots|plain|array] [--no-positions]
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Literal, Protocol

from findex.corpus import Document, iter_documents
from findex.timing import timed
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


class PostingLike(Protocol):
    """What readers need from a posting; both Posting and PlainPosting satisfy it."""

    @property
    def doc_id(self) -> int: ...
    @property
    def tf(self) -> int: ...
    @property
    def positions(self) -> tuple[int, ...]: ...


# "array" representation: two parallel buffers of raw uint32 — (doc_ids, tfs)
type ArrayPostings = tuple[array[int], array[int]]
type PostingList = list[PostingLike] | ArrayPostings


def _new_arrays() -> ArrayPostings:
    return array("I"), array("I")


@dataclass(eq=False, repr=False)
class Index(Mapping[str, Sequence[PostingLike]]):
    """term -> postings. Behaves like a read-only dict thanks to the Mapping ABC.

    We implement the three abstract methods (__getitem__, __iter__, __len__); Mapping
    then supplies keys(), items(), values(), get() and __eq__ for free. __contains__ is
    overridden because Mapping's default calls __getitem__, which for the "array"
    representation would build a whole list just to answer yes/no.
    """

    representation: Representation
    postings: dict[str, PostingList]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]
    has_positions: bool = False
    closed: bool = field(default=False, compare=False)

    # Mapping defines __eq__, which makes Python set __hash__ = None (the hashing contract:
    # objects that compare equal must hash equal). An index is an identity object, so we
    # restore identity equality and hashing; that also lets lru_cache use it as a key.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return object.__hash__(self)

    # --- the Mapping protocol -----------------------------------------------------------

    def __getitem__(self, term: str) -> Sequence[PostingLike]:
        entry = self.postings[term]  # KeyError for unknown terms, as a dict would
        if isinstance(entry, tuple):  # "array" representation: build Postings on demand
            ids, tfs = entry
            return [Posting(d, tf) for d, tf in zip(ids, tfs, strict=True)]
        return entry

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self) -> int:
        """Number of distinct terms (so `len(index)` is the vocabulary size)."""
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __repr__(self) -> str:
        state = ", closed" if self.closed else ""
        return (
            f"Index({self.num_docs:,} docs, {len(self):,} terms, "
            f"representation={self.representation!r}, positions={self.has_positions}{state})"
        )

    # --- statistics used by the scorers -------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    n_docs = num_docs  # Lab 2 name

    @cached_property
    def avg_doc_length(self) -> float:
        """Computed once on first access, then stored in the instance __dict__."""
        return sum(self.doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Document frequency: in how many documents `term` occurs."""
        entry = self.postings.get(term)
        if entry is None:
            return 0
        return len(entry[0]) if isinstance(entry, tuple) else len(entry)

    def close(self) -> None:
        """Release the postings (the big part) and mark the index unusable."""
        self.postings = {}
        self.closed = True

    # --- Lab 2 helpers ------------------------------------------------------------------

    @property
    def n_postings(self) -> int:
        return sum(self.df(term) for term in self.postings)

    def doc_ids(self, term: str) -> Sequence[int]:
        """Sorted doc_ids that contain `term` (empty if the term is unknown)."""
        entry = self.postings.get(term)
        if entry is None:
            return ()
        if isinstance(entry, tuple):
            return entry[0]
        return [p.doc_id for p in entry]

    def entries(self, term: str) -> list[tuple[int, int]]:
        """(doc_id, tf) pairs for `term`, whatever the representation."""
        entry = self.postings.get(term)
        if entry is None:
            return []
        if isinstance(entry, tuple):
            ids, tfs = entry
            return list(zip(ids, tfs, strict=True))
        return [(p.doc_id, p.tf) for p in entry]


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


@contextmanager
def open_index(path: Path | str) -> Generator[Index]:
    """`with open_index("index.pkl") as ix:` — the index is closed even if the body raises.

    Everything before `yield` is __enter__, everything after (the finally) is __exit__.
    """
    from findex.store import load  # local import: store imports this module

    index = load(Path(path))
    try:
        yield index
    finally:
        index.close()


@timed
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

    lists: defaultdict[str, list[PostingLike]] = defaultdict(list)
    arrays: defaultdict[str, ArrayPostings] = defaultdict(_new_arrays)
    cls: type[Posting | PlainPosting] = Posting if representation == "slots" else PlainPosting

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
                lists[term].append(cls(doc_id, len(offs), tuple(offs)))
        else:
            counts = Counter(tokenize(doc.text))
            length = counts.total()
            for term, tf in counts.items():
                if representation == "array":
                    ids, tfs = arrays[term]
                    ids.append(doc_id)
                    tfs.append(tf)
                else:
                    lists[term].append(cls(doc_id, tf))
        doc_lengths[doc_id] = length
        doc_meta[doc_id] = DocMeta(doc_id, doc.doc_id, str(doc.path), titles.get(doc))

    # plain dict: no default factory surprises at query time
    postings: dict[str, PostingList] = {}
    postings.update(arrays if representation == "array" else lists)
    return Index(representation, postings, doc_lengths, doc_meta, positions)


def main(argv: list[str] | None = None) -> int:
    from findex.store import save  # local import: store imports this module

    parser = argparse.ArgumentParser(prog="python -m findex.index", description=__doc__)
    parser.add_argument("root", type=Path, help="corpus directory or .txt/.jsonl file")
    parser.add_argument("--out", type=Path, required=True, help="index file (.pkl or .json)")
    parser.add_argument("--repr", choices=REPRESENTATIONS, default="slots", dest="representation")
    parser.add_argument(
        "--positions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="store token positions (needed for phrase queries; default since Lab 3)",
    )
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
