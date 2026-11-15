"""Lab 2 measurements: postings representations, file formats, merge vs set.

    uv run python scripts/bench_lab02.py data/ [--limit N]

Prints three Markdown tables for the README.
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
import timeit
import tracemalloc
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.search import ENGINES
from findex.store import load, save

MiB = 2**20


def build_measured(root: Path, limit: int | None, representation, positions=False):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(islice(iter_documents(root), limit), representation, positions)
    build_s = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return index, build_s, current, peak


def save_load(index: Index, path: Path) -> tuple[int, float, float]:
    t0 = time.perf_counter()
    save(index, path)
    save_s = time.perf_counter() - t0
    gc.collect()
    t0 = time.perf_counter()
    load(path)
    load_s = time.perf_counter() - t0
    return path.stat().st_size, save_s, load_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="findex-bench-"))

    # --- table 1: postings representations ------------------------------------------
    print(
        "| Postings representation | Index in RAM | Peak memory (build) | Build | "
        "Pickle size | Pickle load |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    variants = [
        ("`list[Posting]`, plain `@dataclass`", "plain", False),
        ("`list[Posting]`, `slots=True`", "slots", False),
        ("`array('I')` pairs (doc_ids, tfs)", "array", False),
        ("`list[Posting]`, `slots=True` + positions", "slots", True),
    ]
    slots_index = None
    for label, rep, positions in variants:
        index, build_s, current, peak = build_measured(args.root, args.limit, rep, positions)
        size, _, load_s = save_load(index, tmp / f"{rep}{int(positions)}.pkl")
        print(
            f"| {label} | {current / MiB:.1f} MiB | {peak / MiB:.1f} MiB | {build_s:.1f} s | "
            f"{size / MiB:.1f} MiB | {load_s:.2f} s |"
        )
        if rep == "slots" and not positions:
            slots_index = index
        else:
            del index
        gc.collect()
    assert slots_index is not None
    print(
        f"\n{slots_index.n_docs} documents, {len(slots_index.postings):,} terms, "
        f"{slots_index.n_postings:,} postings\n"
    )

    # --- table 2: file formats ----------------------------------------------------------
    print("| Format (slots index) | File size | Save | Load |")
    print("|---|---:|---:|---:|")
    for fmt in ("pkl", "json"):
        size, save_s, load_s = save_load(slots_index, tmp / f"fmt.{fmt}")
        print(f"| {fmt} | {size / MiB:.1f} MiB | {save_s:.2f} s | {load_s:.2f} s |")

    # --- table 3: merge vs set ------------------------------------------------------------
    by_df = sorted(slots_index.postings, key=lambda t: len(slots_index.postings[t]))
    common = by_df[-2:]
    rare = [
        by_df[0],
        next(
            t
            for t in by_df
            if slots_index.postings[t][0].doc_id != slots_index.postings[by_df[0]][0].doc_id
        ),
    ]
    print("\n| Query | Postings sizes | Engine | AND | OR |")
    print("|---|---|---|---:|---:|")
    for label, (t1, t2) in (("two most common", common), ("two rarest", rare)):
        a, b = list(slots_index.doc_ids(t1)), list(slots_index.doc_ids(t2))
        for engine, ops in ENGINES.items():
            n = 20_000
            and_op, or_op = ops["AND"], ops["OR"]
            and_us = timeit.timeit(lambda f=and_op, a=a, b=b: f(a, b), number=n) / n * 1e6
            or_us = timeit.timeit(lambda f=or_op, a=a, b=b: f(a, b), number=n) / n * 1e6
            print(
                f"| {label}: `{t1}` + `{t2}` | {len(a)} + {len(b)} | {engine} | "
                f"{and_us:.2f} µs | {or_us:.2f} µs |"
            )


if __name__ == "__main__":
    main()
