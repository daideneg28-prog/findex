"""Boolean search over a saved index.

    python -m findex.search index.pkl "whale ship NOT captain" [--engine merge|set]

Query language (evaluated left to right, no precedence):
    a b          a AND b  (implicit AND)
    a OR b       union
    a NOT b      a AND NOT b
    NOT a        every document except those containing a
Query words go through the same tokenizer as documents, so "Whale," == "whale".
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from findex.index import Index
from findex.tokenize import tokenize

Op = Literal["AND", "OR", "NOT"]
Engine = Literal["merge", "set"]
SetOp = Callable[[Sequence[int], Sequence[int]], list[int]]


# --- two-pointer merges over sorted lists: O(len(a) + len(b)), no hashing -------------


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Elements of a that are not in b."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# --- the same three operations with Python sets (hashing + a final sort) --------------


def set_intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


ENGINES: dict[Engine, dict[Op, SetOp]] = {
    "merge": {"AND": intersect, "OR": union, "NOT": difference},
    "set": {"AND": set_intersect, "OR": set_union, "NOT": set_difference},
}


def parse(query: str) -> list[tuple[Op, list[str]]]:
    """Split a query into (operator, tokens) clauses. Operators must be UPPERCASE."""
    clauses: list[tuple[Op, list[str]]] = []
    op: Op = "AND"
    for word in query.split():
        if word in ("AND", "OR", "NOT"):
            op = word  # type: ignore[assignment]
            continue
        terms = list(tokenize(word))
        if terms:
            clauses.append((op, terms))
        op = "AND"
    return clauses


def search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Return sorted doc_ids matching the Boolean query."""
    ops = ENGINES[engine]
    result: list[int] | None = None
    for op, terms in parse(query):
        # one query word can yield several tokens ("state-of-the-art"): AND them
        ids: Sequence[int] = index.doc_ids(terms[0])
        for term in terms[1:]:
            ids = ops["AND"](ids, index.doc_ids(term))
        if result is None:
            result = ops["NOT"](range(index.n_docs), ids) if op == "NOT" else list(ids)
        else:
            result = ops[op](result, ids)
    return result or []


def main(argv: list[str] | None = None) -> int:
    from findex.store import load

    parser = argparse.ArgumentParser(prog="python -m findex.search", description=__doc__)
    parser.add_argument("index", type=Path, help="index file built by findex.index")
    parser.add_argument("query")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    parser.add_argument("--show", type=int, default=10, help="how many results to print")
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load(args.index)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hits = search(index, args.query, args.engine)
    search_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{len(hits)} documents match {args.query!r}  (engine={args.engine})")
    for doc_id in hits[: args.show]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id:>4}] {meta.title}  ({meta.name})")
    if len(hits) > args.show:
        print(f"  ... and {len(hits) - args.show} more")
    print(
        f"load: {load_s:.2f} s · search: {search_ms:.3f} ms · peak memory: {peak / 2**20:.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
