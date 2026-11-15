"""The `findex` command-line app (typer + rich).

    findex [-v|-vv] index  <corpus> --out index.pkl [--no-positions] [--limit N]
                           [--executor serial|threads|processes] [--workers N]
    findex [-v|-vv] search <index> "<query>" [-k 10] [--scorer bm25|tfidf] [--json]
    findex [-v|-vv] stats  <index> [--top 20]
    findex [-v|-vv] crawl  <seed-url>... [--max-pages 500] [--concurrency 10] [--per-host 2]
                           [--delay 0.5] [--domain d]... [--out data/crawl/crawl.jsonl]

Results go to stdout; every diagnostic (logs, progress, errors) goes to stderr, so
`findex search ix.pkl "q" --json | jq .title` stays clean.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from contextlib import aclosing
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.corpus import iter_paths, read_document
from findex.crawler import CrawlStats, crawl
from findex.index import REPRESENTATIONS, Representation, open_index
from findex.parallel import EXECUTORS, Executor, build_from_paths
from findex.query import QuerySyntaxError
from findex.rank import SCORERS, match, search
from findex.snippets import snippet
from findex.store import save

app = typer.Typer(
    name="findex",
    help="A small search engine over a plain-text corpus.",
    no_args_is_help=True,
    add_completion=False,
)
out = Console()  # stdout: results
err = Console(stderr=True)  # stderr: diagnostics
log = logging.getLogger("findex")

# private-use characters as highlight marks: they never occur in real text
_MARK = ("", "")
_MARK_RE = re.compile("(.*?)")


class ScorerName(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class Repr(StrEnum):
    slots = "slots"
    plain = "plain"
    array = "array"


class ExecutorName(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


# typer needs an Enum for choices; the library API takes the Literal type
_REPRS: dict[Repr, Representation] = {Repr.slots: "slots", Repr.plain: "plain", Repr.array: "array"}
assert set(_REPRS.values()) == set(REPRESENTATIONS)
_EXECUTORS: dict[ExecutorName, Executor] = {
    ExecutorName.serial: "serial",
    ExecutorName.threads: "threads",
    ExecutorName.processes: "processes",
}
assert set(_EXECUTORS.values()) == set(EXECUTORS)


def _configure_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=err, show_path=False, show_time=False)],
        force=True,
    )


def _version(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    verbose: Annotated[
        int, typer.Option("--verbose", "-v", count=True, help="-v: info, -vv: debug")
    ] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """A small search engine over a plain-text corpus."""
    _configure_logging(verbose)


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(exists=True, help="directory of .txt/.jsonl files")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="index file (.pkl or .json)")],
    positions: Annotated[bool, typer.Option(help="store token positions (phrases)")] = True,
    limit: Annotated[int | None, typer.Option(help="only the first N files")] = None,
    representation: Annotated[Repr, typer.Option("--repr")] = Repr.slots,
    executor: Annotated[
        ExecutorName, typer.Option(help="where build_partial runs (see README, Lab 5)")
    ] = ExecutorName.processes,
    workers: Annotated[
        int | None, typer.Option(min=1, help="pool size (default: os.cpu_count())")
    ] = None,
) -> None:
    """Build an index from a corpus and save it (map-reduce over chunks of files)."""
    rep = _REPRS[representation]
    paths = list(islice(iter_paths(corpus), limit))
    progress = Progress(
        TextColumn(f"[bold]indexing[/] ({executor.value})"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    )
    with progress:
        task = progress.add_task("files", total=len(paths))
        try:
            index = build_from_paths(
                paths,
                executor=_EXECUTORS[executor],
                workers=workers,
                representation=rep,
                positions=positions,
                progress=lambda n: progress.advance(task, n),
            )
        except ValueError as exc:
            err.print(f"[red]error:[/] {exc}")
            raise typer.Exit(2) from exc
    save(index, out_path)
    out.print(
        f"{index!r}\nsaved to [bold]{out_path}[/] ({out_path.stat().st_size / 2**20:.1f} MiB)"
    )


def _highlighted(s: str) -> Text:
    text = Text()
    pos = 0
    for m in _MARK_RE.finditer(s):
        text.append(s[pos : m.start()])
        text.append(m.group(1), style="bold yellow")
        pos = m.end()
    text.append(s[pos:])
    return text


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="index file")],
    query: Annotated[str, typer.Argument(help="e.g. 'tarzan \"the apes\" NOT jane'")],
    k: Annotated[int, typer.Option("-k", "--k", min=1, help="number of results")] = 10,
    scorer: Annotated[ScorerName, typer.Option(help="ranking function")] = ScorerName.bm25,
    as_json: Annotated[bool, typer.Option("--json", help="JSON lines on stdout")] = False,
    snippets: Annotated[bool, typer.Option(help="show text snippets")] = True,
) -> None:
    """Search an index; prints a ranked table (or JSON lines)."""
    with open_index(index_path) as index:
        try:
            results = search(index, query, SCORERS[scorer.value], k)
        except QuerySyntaxError as exc:
            err.print(f"[red]query error:[/] {exc}")
            raise typer.Exit(1) from exc
        matched, terms = match(index, query) if query.strip() else ((), ())
        log.info("%d documents match %r", len(matched), query)

        rows: list[tuple[int, float, str, str]] = []
        for rank, r in enumerate(results, start=1):
            snip = ""
            if snippets:
                meta = index.doc_meta[r.doc_id]
                path = Path(meta.path)
                try:
                    snip = snippet(read_document(path, meta.name), terms, mark=_MARK)
                except OSError:
                    log.warning("cannot read %s for a snippet", path)
            rows.append((rank, r.score, r.title, snip))

        if as_json:
            for rank, score, title, snip in rows:
                plain = snip.replace(_MARK[0], "").replace(_MARK[1], "")
                record = {"rank": rank, "score": round(score, 4), "title": title, "snippet": plain}
                typer.echo(json.dumps(record, ensure_ascii=False))
            return

        if not rows:
            err.print(f"no documents match {query!r}")
            raise typer.Exit(0)
        table = Table(title=f"{len(matched)} documents match {query!r} ({scorer.value})")
        table.add_column("#", justify="right")
        table.add_column("score", justify="right")
        table.add_column("title", style="bold")
        if snippets:
            table.add_column("snippet")
        for rank, score, title, snip in rows:
            cells: list[str | Text] = [str(rank), f"{score:.3f}", title]
            if snippets:
                cells.append(_highlighted(snip))
            table.add_row(*cells)
        out.print(table)


@app.command("stats")
def stats_cmd(
    index_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="index file")],
    top: Annotated[int, typer.Option(help="how many top terms")] = 20,
) -> None:
    """Corpus statistics (the Lab 1 numbers) from a saved index."""
    with open_index(index_path) as index:
        totals: Counter[str] = Counter()
        for term in index:
            totals[term] = sum(p.tf for p in index[term])
        summary = Table(show_header=False, title="corpus")
        summary.add_row("documents", f"{index.num_docs:,}")
        summary.add_row("tokens", f"{sum(index.doc_lengths.values()):,}")
        summary.add_row("vocabulary", f"{len(index):,}")
        summary.add_row("avg document length", f"{index.avg_doc_length:,.0f} tokens")
        summary.add_row("positions", str(index.has_positions))
        out.print(summary)
        terms = Table(title=f"top {top} terms")
        terms.add_column("#", justify="right")
        terms.add_column("term")
        terms.add_column("count", justify="right")
        terms.add_column("df", justify="right")
        for rank, (term, count) in enumerate(totals.most_common(top), start=1):
            terms.add_row(str(rank), term, f"{count:,}", str(index.df(term)))
        out.print(terms)


def _crawl_stats_table(stats: CrawlStats, max_pages: int) -> Table:
    table = Table(show_header=False, title="crawling", min_width=40)
    table.add_row("pages", f"{stats.pages} / {max_pages}")
    table.add_row("pages/s", f"{stats.pages_per_s:.1f}")
    table.add_row("in flight", str(stats.in_flight))
    table.add_row("queue", str(stats.queued))
    table.add_row("errors", str(stats.errors))
    table.add_row("retries", str(stats.retries))
    table.add_row("robots.txt blocked", str(stats.robots_blocked))
    table.add_row("downloaded", f"{stats.bytes / 2**20:.1f} MiB")
    return table


async def _crawl(
    seeds: list[str],
    out_path: Path,
    log_path: Path,
    stats: CrawlStats,
    *,
    max_pages: int,
    concurrency: int,
    per_host: int,
    min_delay: float,
    domains: list[str] | None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        out_path.open("w", encoding="utf-8") as out_file,
        log_path.open("w", encoding="utf-8") as log_file,
        Live(_crawl_stats_table(stats, max_pages), console=err, refresh_per_second=4) as live,
    ):

        async def refresh() -> None:
            while True:
                live.update(_crawl_stats_table(stats, max_pages))
                await asyncio.sleep(0.25)

        ticker = asyncio.create_task(refresh())
        try:
            pages = crawl(
                seeds,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                min_delay=min_delay,
                allowed_domains=domains,
                stats=stats,
                log_file=log_file,
            )
            async with aclosing(pages):
                async for page in pages:  # streamed: one line per page, as it arrives
                    out_file.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")
        finally:
            ticker.cancel()
            live.update(_crawl_stats_table(stats, max_pages))


@app.command("crawl")
def crawl_cmd(
    seeds: Annotated[list[str], typer.Argument(help="start URL(s)")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="JSON lines output")] = Path(
        "data/crawl/crawl.jsonl"
    ),
    max_pages: Annotated[int, typer.Option(min=1, help="page budget")] = 500,
    concurrency: Annotated[int, typer.Option(min=1, help="worker tasks")] = 10,
    per_host: Annotated[int, typer.Option(min=1, help="max requests in flight per host")] = 2,
    delay: Annotated[
        float, typer.Option(min=0, help="min seconds between requests to a host")
    ] = 0.5,
    domain: Annotated[
        list[str] | None, typer.Option(help="allowed domain (repeatable; default: the seeds')")
    ] = None,
    log_path: Annotated[Path | None, typer.Option("--log", help="per-URL log")] = None,
) -> None:
    """Crawl politely from seed URLs; write pages as JSON lines for `findex index`."""
    stats = CrawlStats()
    log_path = log_path or out_path.with_name("crawl.log")
    asyncio.run(
        _crawl(
            seeds,
            out_path,
            log_path,
            stats,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=delay,
            domains=domain,
        )
    )
    out.print(
        f"{stats.pages} pages in {stats.elapsed:.1f} s ({stats.pages_per_s:.1f} pages/s), "
        f"{stats.errors} errors -> [bold]{out_path}[/], log: {log_path}"
    )


if __name__ == "__main__":
    app()
