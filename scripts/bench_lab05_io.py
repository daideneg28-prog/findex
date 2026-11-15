"""Lab 5 counter-benchmark: an I/O-bound job, where threads DO scale under the GIL.

Fetch the Project Gutenberg catalogue pages of the first N books of our corpus with
1, 4, 16 and 32 threads. While a thread waits for the network, socket.recv() has
released the GIL, so the other threads run.

    python scripts/bench_lab05_io.py [N_PAGES]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "findex-course-benchmark/0.5 (educational, low volume)"}


def fetch(url: str) -> int:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        return len(response.read())


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    titles = json.loads((ROOT / "data" / "gutenberg" / "titles.json").read_text(encoding="utf-8"))
    urls = [f"https://www.gutenberg.org/ebooks/{book_id}" for book_id in list(titles)[:n]]
    print(f"{len(urls)} pages, GIL enabled: {getattr(sys, '_is_gil_enabled', lambda: True)()}")
    print("| Threads | Wall, s (median of 3) | Speedup |")
    print("|---:|---:|---:|")
    base = 0.0
    for workers in (1, 4, 16, 32):
        walls: list[float] = []
        for _ in range(3):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(workers) as pool:
                total = sum(pool.map(fetch, urls))
            walls.append(time.perf_counter() - t0)
        wall = statistics.median(walls)
        base = base or wall
        print(f"| {workers} | {wall:.2f} | {base / wall:.1f}× |   ({total / 2**20:.1f} MiB)")
        time.sleep(2)  # be polite to the server between rounds


if __name__ == "__main__":
    main()
