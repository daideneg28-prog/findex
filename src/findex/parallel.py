"""Parallel index construction as map-reduce.

map:    build_partial(paths)  -> PartialIndex   (in a thread or a worker process)
reduce: merge(partials)       -> Index          (always in the parent, sequentially)

The serial build is `merge([build_partial(all_paths)])` and gives exactly the index that
`build_index(iter_documents(root))` gives (tests compare the pickled bytes).

Design choices, each forced by how workers communicate:

* Chunks are lists of *paths*, not loaded documents: a path is ~100 bytes to pickle, a
  book is ~500 KB. Each worker reads its own files.
* A PartialIndex is *compact*: per term three `array('I')` buffers (local doc ids, tfs,
  flat positions). Arrays pickle as one raw byte string, not millions of int objects,
  so the result is cheap to send back from a process.
* Workers share *no* mutable state. Each builds its own dicts and returns them; only
  the parent writes the final index. That is why the free-threaded build (no GIL) gives
  the same, correct index — there is nothing for two threads to race on.
* doc_ids: a worker numbers its documents locally from 0; `merge` shifts chunk k by the
  number of documents in chunks 0..k-1. Chunks are contiguous slices of the path list,
  so the result is the same numbering the serial stream produces, and the ranges can't
  overlap. (We can't fix the ranges before the map step because a .jsonl file holds an
  unknown number of documents.)
"""

from __future__ import annotations

import logging
import os
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from concurrent import futures
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Literal, cast

from findex.corpus import iter_file, iter_paths
from findex.index import (
    ArrayPostings,
    DocMeta,
    Index,
    PlainPosting,
    Posting,
    PostingLike,
    PostingList,
    Representation,
    Titles,
)
from findex.sysinfo import process_cpu_seconds
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

Executor = Literal["serial", "threads", "processes"]
EXECUTORS: tuple[Executor, ...] = ("serial", "threads", "processes")

# term -> (local doc ids, tfs, positions of all those docs concatenated)
type PartialPostings = dict[str, tuple[array[int], array[int], array[int]]]


@dataclass(slots=True)
class PartialIndex:
    """What one worker sends back: postings for its chunk, with local doc ids 0..n-1."""

    docs: list[tuple[str, str, str, int]]  # (name, path, title, length) by local doc id
    postings: PartialPostings
    cpu_s: float = 0.0  # CPU time this task spent (time.thread_time: per thread)


@dataclass(slots=True)
class BuildStats:
    """Measurements filled in by build_parallel (Lab 5 benchmark)."""

    chunks: int = 0
    map_s: float = 0.0  # wall time until every partial index is back
    merge_s: float = 0.0  # wall time of the sequential reduce step
    task_cpu_s: float = 0.0  # sum of the tasks' own CPU time
    worker_cpu_s: float = 0.0  # processes: total CPU of the worker processes
    worker_pids: list[int] = field(default_factory=list[int])


def build_partial(paths: Sequence[Path], positions: bool = True) -> PartialIndex:
    """Index the documents in `paths`. Module-level so a process pool can pickle it."""
    t0 = time.thread_time()
    titles = Titles()
    docs: list[tuple[str, str, str, int]] = []
    postings: PartialPostings = {}
    for path in paths:
        for doc in iter_file(path):
            local_id = len(docs)
            if positions:
                offsets: defaultdict[str, list[int]] = defaultdict(list)
                length = 0
                for length, token in enumerate(tokenize(doc.text), start=1):
                    offsets[token].append(length - 1)
                items = ((term, len(offs), offs) for term, offs in offsets.items())
            else:
                counts = Counter(tokenize(doc.text))
                length = counts.total()
                items = ((term, tf, ()) for term, tf in counts.items())
            for term, tf, offs in items:
                entry = postings.get(term)
                if entry is None:
                    entry = postings[term] = (array("I"), array("I"), array("I"))
                entry[0].append(local_id)
                entry[1].append(tf)
                entry[2].extend(offs)
            docs.append((doc.doc_id, str(doc.path), titles.get(doc), length))
    return PartialIndex(docs, postings, time.thread_time() - t0)


def merge(
    partials: Iterable[PartialIndex],
    representation: Representation = "slots",
    positions: bool = True,
) -> Index:
    """Reduce: concatenate the partial postings in chunk order, shifting doc ids.

    Each partial covers a higher range of doc ids than the one before, so appending
    keeps every postings list sorted — the merge of sorted, non-overlapping runs is
    their concatenation. This loop is the sequential part Amdahl's law talks about.
    """
    if positions and representation == "array":
        raise ValueError("positions are only supported for the plain/slots representations")
    cls: type[Posting | PlainPosting] = Posting if representation == "slots" else PlainPosting
    lists: dict[str, list[PostingLike]] = {}
    arrays: dict[str, ArrayPostings] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    offset = 0
    for part in partials:
        for local_id, (name, path, title, length) in enumerate(part.docs):
            doc_id = offset + local_id
            doc_lengths[doc_id] = length
            doc_meta[doc_id] = DocMeta(doc_id, name, path, title)
        for term, (ids, tfs, pos) in part.postings.items():
            if representation == "array":
                target = arrays.get(term)
                if target is None:
                    target = arrays[term] = (array("I"), array("I"))
                target[0].extend([d + offset for d in ids] if offset else ids)
                target[1].extend(tfs)
                continue
            plist = lists.get(term)
            if plist is None:
                plist = lists[term] = []
            if positions:
                start = 0
                for d, tf in zip(ids, tfs, strict=True):
                    plist.append(cls(d + offset, tf, tuple(pos[start : start + tf])))
                    start += tf
            else:
                plist.extend(cls(d + offset, tf) for d, tf in zip(ids, tfs, strict=True))
        offset += len(part.docs)
    postings: dict[str, PostingList] = {}
    postings.update(arrays if representation == "array" else lists)
    return Index(representation, postings, doc_lengths, doc_meta, positions)


def chunk_paths(paths: Sequence[Path], n_chunks: int) -> list[list[Path]]:
    """Split into at most `n_chunks` contiguous chunks of roughly equal size in bytes.

    Books differ in size by 100x, so equal *counts* would leave one worker with all the
    big ones. More chunks than workers also evens out the load (a worker that finishes
    early takes the next chunk).
    """
    if not paths:
        return []
    sizes = [p.stat().st_size for p in paths]
    n_chunks = max(1, min(n_chunks, len(paths)))
    target = sum(sizes) / n_chunks
    chunks: list[list[Path]] = [[]]
    acc = 0
    for path, size in zip(paths, sizes, strict=True):
        if chunks[-1] and acc >= target * len(chunks) and len(chunks) < n_chunks:
            chunks.append([])
        chunks[-1].append(path)
        acc += size
    return chunks


def _make_pool(executor: Executor, workers: int) -> futures.Executor:
    match executor:
        case "threads":
            return futures.ThreadPoolExecutor(max_workers=workers)
        case "processes":
            return futures.ProcessPoolExecutor(max_workers=workers)
        case "serial":
            raise ValueError("the serial executor does not use a pool")


def _worker_cpu(pool: futures.Executor) -> tuple[list[int], float]:
    """Total CPU time of a process pool's workers, read while they are still alive."""
    processes: object = getattr(pool, "_processes", None)  # private: {pid: Process}
    keys: list[object] = list(cast(dict[object, object], processes)) if processes else []
    pids = [pid for pid in keys if isinstance(pid, int)]
    return pids, sum(process_cpu_seconds(pid) for pid in pids)


def _no_progress(_n: int) -> None:
    pass


def build_from_paths(
    paths: Sequence[Path],
    *,
    executor: Executor = "processes",
    workers: int | None = None,
    representation: Representation = "slots",
    positions: bool = True,
    chunks_per_worker: int = 4,
    stats: BuildStats | None = None,
    progress: Callable[[int], None] | None = None,
) -> Index:
    """Map `build_partial` over chunks of `paths` with the chosen executor, then merge.

    If any task raises, the remaining ones are cancelled and the original exception is
    re-raised here (for processes, its remote traceback is attached as __cause__), so a
    broken file fails the build loudly instead of hanging it.
    """
    if positions and representation == "array":
        raise ValueError("positions are only supported for the plain/slots representations")
    workers = 1 if executor == "serial" else (workers or os.cpu_count() or 1)
    chunks = chunk_paths(paths, workers * chunks_per_worker)
    stats = stats if stats is not None else BuildStats()
    stats.chunks = len(chunks)
    report = progress or _no_progress

    t0 = time.perf_counter()
    results: list[PartialIndex | None] = [None] * len(chunks)
    if executor == "serial":
        for i, chunk in enumerate(chunks):
            results[i] = build_partial(chunk, positions)
            report(len(chunk))
    else:
        pool = _make_pool(executor, workers)
        with pool:
            pending = {pool.submit(build_partial, c, positions): i for i, c in enumerate(chunks)}
            try:
                for future in futures.as_completed(pending):
                    i = pending[future]
                    results[i] = future.result()  # re-raises the worker's exception
                    report(len(chunks[i]))
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            if executor == "processes":
                stats.worker_pids, stats.worker_cpu_s = _worker_cpu(pool)
    stats.map_s = time.perf_counter() - t0

    partials = [r for r in results if r is not None]
    assert len(partials) == len(chunks)
    stats.task_cpu_s = sum(p.cpu_s for p in partials)
    t0 = time.perf_counter()
    index = merge(partials, representation, positions)
    stats.merge_s = time.perf_counter() - t0
    log.info(
        "%s x%d: %d chunks, map %.2f s, merge %.2f s",
        executor,
        workers,
        len(chunks),
        stats.map_s,
        stats.merge_s,
    )
    return index


def build_parallel(
    root: Path,
    *,
    limit: int | None = None,
    executor: Executor = "processes",
    workers: int | None = None,
    representation: Representation = "slots",
    positions: bool = True,
    stats: BuildStats | None = None,
) -> Index:
    """build_from_paths over the corpus files under `root` (the first `limit` files)."""
    paths = list(islice(iter_paths(root), limit))
    return build_from_paths(
        paths,
        executor=executor,
        workers=workers,
        representation=representation,
        positions=positions,
        stats=stats,
    )
