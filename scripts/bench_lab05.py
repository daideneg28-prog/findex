"""Lab 5 benchmark: serial vs threads vs processes (and the free-threaded build).

Every measurement runs in a *fresh* interpreter (peak RSS is a per-process maximum, and
a warm pool would hide start-up costs):

    # all configurations, 3 runs each, with the normal (GIL) build
    python scripts/bench_lab05.py measure --runs 3
    # the free-threaded build, threads only (PYTHONPATH=src is set for the child)
    python scripts/bench_lab05.py measure --python <path to python3.14t> --label 3.14t \
        --configs serial:1 threads:1 threads:2 threads:4 threads:8 threads:12
    # table for the README + speedup plot (needs matplotlib)
    python scripts/bench_lab05.py report

One measurement (what the driver runs in the child):
    python scripts/bench_lab05.py one processes 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "gutenberg"
RESULTS = ROOT / "bench" / "lab05.json"
PLOT = ROOT / "docs" / "lab05_speedup.png"
CPUS = os.cpu_count() or 1
DEFAULT_CONFIGS = [
    "serial:1",
    *(f"threads:{w}" for w in (1, 2, 4, 8, CPUS)),
    *(f"processes:{w}" for w in (1, 2, 4, 8, CPUS)),
]


def one(executor: str, workers: int) -> dict[str, Any]:
    """Build the full positional index once and report what it cost."""
    from findex.parallel import BuildStats, build_parallel
    from findex.sysinfo import peak_rss_mib

    stats = BuildStats()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    index = build_parallel(CORPUS, executor=executor, workers=workers, stats=stats)
    wall = time.perf_counter() - wall0
    parent_cpu = time.process_time() - cpu0  # all threads of this process
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "executor": executor,
        "workers": workers,
        "wall": wall,
        "cpu": parent_cpu + stats.worker_cpu_s,
        "parent_cpu": parent_cpu,
        "worker_cpu": stats.worker_cpu_s,
        "task_cpu": stats.task_cpu_s,
        "map": stats.map_s,
        "merge": stats.merge_s,
        "rss": peak_rss_mib(),
        "chunks": stats.chunks,
        "docs": index.num_docs,
        "terms": len(index),
        "gil": gil,
        "python": sys.version.split()[0]
        + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""),
    }


def measure(python: str, label: str, configs: list[str], runs: int) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    data: dict[str, Any] = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    rows: dict[str, Any] = data.setdefault(label, {})
    script = str(Path(__file__).resolve())
    print(f"warm-up ({label}) ...", flush=True)  # fills the OS file cache, compiles .pyc
    subprocess.run([python, script, "one", "serial", "1"], env=env, check=True, capture_output=True)
    for config in configs:
        executor, workers = config.split(":")
        samples: list[dict[str, Any]] = []
        for run in range(runs):
            out = subprocess.run(
                [python, script, "one", executor, workers],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            sample = json.loads(out.strip().splitlines()[-1])
            samples.append(sample)
            print(f"  {label} {config} run {run + 1}: wall {sample['wall']:.2f} s", flush=True)
        rows[config] = samples
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(data, indent=1))


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return statistics.median(s[key] for s in samples)


def report() -> None:
    from findex.sysinfo import cpu_description

    data: dict[str, Any] = json.loads(RESULTS.read_text())
    base = _median(data["gil"]["serial:1"], "wall")
    print(f"Machine: {cpu_description()}\n")
    print(
        "| Python | Executor | Workers | Wall, s (median, min–max) | CPU, s | Peak RSS (parent), MiB | Merge, s | Speedup |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    curves: dict[str, list[tuple[int, float]]] = {}
    for rows in data.values():
        for config, samples in rows.items():
            executor, workers = config.split(":")
            wall = [s["wall"] for s in samples]
            med = statistics.median(wall)
            speedup = base / med
            print(
                f"| {samples[0]['python']} | {executor} | {workers} | "
                f"{med:.2f} ({min(wall):.2f}–{max(wall):.2f}) | {_median(samples, 'cpu'):.1f} | "
                f"{_median(samples, 'rss'):.0f} | {_median(samples, 'merge'):.2f} | {speedup:.2f}× |"
            )
            if executor != "serial":
                curves.setdefault(f"{executor} ({samples[0]['python']})", []).append(
                    (int(workers), speedup)
                )
    merge_share = _median(data["gil"]["serial:1"], "merge") / base
    print(f"\nserial merge share: {merge_share:.1%} -> Amdahl ceiling {1 / merge_share:.1f}x")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    xs = sorted({w for c in curves.values() for w, _ in c})
    ax.plot(xs, xs, "--", color="grey", label="ideal (linear)")
    for name, points in curves.items():
        points.sort()
        ax.plot([w for w, _ in points], [s for _, s in points], "o-", label=name)
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial (GIL build)")
    ax.set_title(f"findex: positional index of 150 books ({cpu_description()})", fontsize=9)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(4.0, max(s for c in curves.values() for _, s in c) + 0.5))
    ax.legend()
    ax.grid(alpha=0.3)
    PLOT.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT)
    print(f"plot saved to {PLOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_one = sub.add_parser("one")
    p_one.add_argument("executor")
    p_one.add_argument("workers", type=int)
    p_measure = sub.add_parser("measure")
    p_measure.add_argument("--python", default=sys.executable)
    p_measure.add_argument("--label", default="gil")
    p_measure.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    p_measure.add_argument("--runs", type=int, default=3)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "one":
        print(json.dumps(one(args.executor, args.workers)))
    elif args.cmd == "measure":
        measure(args.python, args.label, args.configs, args.runs)
    else:
        report()


if __name__ == "__main__":  # required: process pools re-import this file under `spawn`
    main()
