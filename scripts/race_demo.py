"""Lab 5: the race the GIL was hiding.

A tempting "simple" way to parallelize corpus statistics with threads is to let every
thread add into ONE shared Counter:

    totals[term] += 1          # read totals[term], add 1, store it back: 3 steps

Each step is atomic (the dict has its own lock), but the three together are not. Two
threads can both read 41, both store 42, and one occurrence is lost. Under the GIL a
thread is only interrupted every 5 ms (sys.getswitchinterval()), so the bad interleaving
is rare and the bug can go unnoticed for years. On the free-threaded build the threads
really run at the same moment and updates vanish by the thousand.

Fixes, both measured below:
  locked   - one threading.Lock around the read-modify-write (correct, but the
             threads now queue for the lock: hardly any parallelism left);
  private  - every thread counts into its own Counter, the parent adds them up at
             the end (no shared mutable state: the design findex.parallel uses).

    python scripts/race_demo.py [N_BOOKS] [THREADS]
"""

from __future__ import annotations

import dis
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findex.corpus import iter_paths
from findex.tokenize import tokenize


def load(n_books: int) -> list[list[str]]:
    return [
        list(tokenize(p.read_text(encoding="utf-8")))
        for p in islice(iter_paths(ROOT / "data" / "gutenberg"), n_books)
    ]


def unsafe(books: list[list[str]], totals: Counter[str]) -> None:
    for tokens in books:
        for term in tokens:
            totals[term] += 1  # the race


def locked(books: list[list[str]], totals: Counter[str], lock: threading.Lock) -> None:
    for tokens in books:
        for term in tokens:
            with lock:
                totals[term] += 1


def private(books: list[list[str]]) -> Counter[str]:
    mine: Counter[str] = Counter()
    for tokens in books:
        for term in tokens:
            mine[term] += 1
    return mine


def main() -> None:
    n_books = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    n_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL enabled: {gil}, {n_threads} threads")
    print("bytecode of `totals[term] += 1`:")
    for ins in dis.get_instructions("totals[term] += 1"):
        if ins.opname not in ("RESUME", "RETURN_CONST", "LOAD_CONST", "RETURN_VALUE"):
            print(f"    {ins.opname:<20} {ins.argrepr}")

    books = load(n_books)
    # every thread processes ALL books, so the exact answer is n_threads x the real counts
    expected = sum(len(t) for t in books) * n_threads

    for name in ("unsafe", "locked", "private"):
        totals: Counter[str] = Counter()
        lock = threading.Lock()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(n_threads) as pool:
            if name == "unsafe":
                list(pool.map(unsafe, [books] * n_threads, [totals] * n_threads))
            elif name == "locked":
                list(
                    pool.map(locked, [books] * n_threads, [totals] * n_threads, [lock] * n_threads)
                )
            else:
                for part in pool.map(private, [books] * n_threads):
                    totals.update(part)
        elapsed = time.perf_counter() - t0
        got = totals.total()
        lost = expected - got
        print(
            f"{name:<8} {elapsed:6.2f} s  counted {got:,} of {expected:,}"
            f"  -> lost {lost:,} ({lost / expected:.3%})"
        )


if __name__ == "__main__":
    main()
