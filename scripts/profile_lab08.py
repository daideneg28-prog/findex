"""Lab 8: the workloads we profile, isolated from start-up costs.

# search: what one /search request does (match + BM25 + 10 snippets), repeated
python -m cProfile -s cumtime scripts/profile_lab08.py search --repeat 20
py-spy record -o docs/lab08_search_flame.svg -- python scripts/profile_lab08.py search --repeat 200
python -m scalene --cli --cpu-only scripts/profile_lab08.py search --repeat 50

# index: the serial build of the full corpus
py-spy record -o docs/lab08_index_flame.svg -- python scripts/profile_lab08.py index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from findex.parallel import build_parallel
from findex.store import load
from findex.web.app import run_search

QUERIES = ["white whale ship", "love war peace", "tarzan apes jungle"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["search", "index"])
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.pkl")
    args = parser.parse_args()
    if args.what == "index":
        t0 = time.perf_counter()
        build_parallel(ROOT / "data" / "gutenberg", executor="serial")
        print(f"index: {time.perf_counter() - t0:.1f} s")
        return
    index = load(args.index)
    t0 = time.perf_counter()
    for i in range(args.repeat):
        q = QUERIES[i % len(QUERIES)]
        run_search(index, q, 10, 1, "bm25", ("«", "»"))
    elapsed = time.perf_counter() - t0
    print(f"{args.repeat} searches: {elapsed / args.repeat * 1000:.1f} ms each")


if __name__ == "__main__":
    main()
