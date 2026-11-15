"""Lab 6 benchmark: the same crawl at concurrency 1, 5 and 20.

Each run is a fresh interpreter (peak RSS is per process). Seeds are the documentation
sites of the Python Software Foundation; robots.txt allows them, the crawler obeys it,
and every host gets at most PER_HOST requests in flight with DELAY s between starts.

    python scripts/bench_lab06.py            # all three levels -> bench/lab06.json + table
    python scripts/bench_lab06.py one 20     # one run (what the driver executes)
    CRAWL_DELAY=0 python scripts/bench_lab06.py   # same, without the per-host delay
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import aclosing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [
    "https://docs.python.org/3/",
    "https://peps.python.org/",
    "https://devguide.python.org/",
    "https://packaging.python.org/en/latest/",
    "https://typing.python.org/en/latest/",
]
DOMAINS = [s.split("/")[2] for s in SEEDS]
MAX_PAGES = 200
PER_HOST = 4
DELAY = float(os.environ.get("CRAWL_DELAY", "0.25"))
RESULTS = ROOT / "bench" / f"lab06-delay{DELAY:g}.json"


async def one(concurrency: int) -> dict[str, float | int]:
    from findex.crawler import CrawlStats, crawl
    from findex.sysinfo import peak_rss_mib

    out_dir = ROOT / "data" / "crawl"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = CrawlStats()
    t0 = time.perf_counter()
    with (
        (out_dir / f"bench-c{concurrency}-d{DELAY:g}.jsonl").open("w", encoding="utf-8") as out,
        (out_dir / f"bench-c{concurrency}-d{DELAY:g}.log").open("w", encoding="utf-8") as log,
    ):
        pages = crawl(
            SEEDS,
            max_pages=MAX_PAGES,
            concurrency=concurrency,
            per_host=PER_HOST,
            min_delay=DELAY,
            allowed_domains=DOMAINS,
            stats=stats,
            log_file=log,
        )
        async with aclosing(pages):
            async for page in pages:
                out.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")
    wall = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "pages": stats.pages,
        "wall": wall,
        "pages_per_s": stats.pages / wall,
        "errors": stats.errors,
        "retries": stats.retries,
        "robots_blocked": stats.robots_blocked,
        "mib": stats.bytes / 2**20,
        "rss": peak_rss_mib(),
    }


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "one":
        print(json.dumps(asyncio.run(one(int(sys.argv[2])))))
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    rows = []
    for concurrency in (1, 5, 20):
        out = subprocess.run(
            [sys.executable, __file__, "one", str(concurrency)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
        print(rows[-1], flush=True)
        time.sleep(5)  # a pause between runs, for the servers' sake
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=1))
    print("| Concurrency | Pages | Wall, s | Pages/s | Errors | Retries | Peak RSS, MiB |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['concurrency']} | {r['pages']} | {r['wall']:.1f} | {r['pages_per_s']:.1f} | "
            f"{r['errors']} | {r['retries']} | {r['rss']:.0f} |"
        )


if __name__ == "__main__":
    main()
