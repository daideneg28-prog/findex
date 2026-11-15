"""Lab 7 load test: run `oha` against a running server and print a README table row.

    findex serve --index data/index.pkl                    # in another terminal
    python scripts/bench_lab07.py http://127.0.0.1:8000 --route /search -c 50 -z 30s

With --probe, /health is hammered by ONE client at the same time, which shows whether
the main load blocks the event loop (it does for the `async def` route).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

QUERIES = ["white whale", "tarzan", '"dejah thoris"', "love AND war", "ship NOT sea"]


def oha(url: str, concurrency: int, duration: str, exe: str) -> dict[str, Any]:
    out = subprocess.run(
        [exe, "-z", duration, "-c", str(concurrency), "--no-tui", "--output-format", "json", url],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def row(label: str, route: str, concurrency: int, result: dict[str, Any]) -> str:
    summary, pct = result["summary"], result["latencyPercentiles"]
    ok = summary["successRate"] * 100
    return (
        f"| {label} | `{route}` | {concurrency} | {summary['requestsPerSec']:.1f} | "
        f"{pct['p50'] * 1000:.0f} | {pct['p95'] * 1000:.0f} | {pct['p99'] * 1000:.0f} | {ok:.0f}% |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--route", default="/search")
    parser.add_argument("--query", default=QUERIES[0])
    parser.add_argument("-c", type=int, default=50)
    parser.add_argument("-z", default="30s")
    parser.add_argument("--label", default="local")
    parser.add_argument("--probe", action="store_true", help="also time /health meanwhile")
    parser.add_argument("--save", type=Path, help="append the raw JSON results to this file")
    args = parser.parse_args()
    exe = shutil.which("oha") or str(
        next(Path.home().glob("AppData/Local/Microsoft/WinGet/Packages/hatoo.oha*/oha.exe"))
    )
    url = f"{args.base}{args.route}?q={args.query.replace(' ', '%20').replace('"', '%22')}"

    probe: dict[str, Any] = {}
    thread = None
    if args.probe:
        thread = threading.Thread(
            target=lambda: probe.update(oha(f"{args.base}/health", 1, args.z, exe))
        )
        thread.start()
    result = oha(url, args.c, args.z, exe)
    if thread:
        thread.join()

    print(row(args.label, args.route, args.c, result))
    if probe:
        print(row(args.label + " — /health meanwhile", "/health", 1, probe))
    if args.save:
        entry = {"label": args.label, "route": args.route, "c": args.c, "result": result}
        if probe:
            entry["probe"] = probe
        with args.save.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
