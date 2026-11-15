"""The `findex` command-line app (typer + rich).

    findex [-v|-vv] index  <corpus> --out index.pkl [--no-positions] [--limit N]
    findex [-v|-vv] search <index> "<query>" [-k 10] [--scorer bm25|tfidf] [--json]
    findex [-v|-vv] stats  <index> [--top 20]

Results go to stdout; every diagnostic (logs, progress, errors) goes to stderr, so
`findex search ix.pkl "q" --json | jq .title` stays clean.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from findex import __version__
from findex.corpus import iter_documents
from findex.index import REPRESENTATIONS, Representation, build_index, open_index
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


# typer needs an Enum for choices; the library API takes the Literal type
_REPRS: dict[Repr, Representation] = {Repr.slots: "slots", Repr.plain: "plain", Repr.array: "array"}
assert set(_REPRS.values()) == set(REPRESENTATIONS)


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
    limit: Annotated[int | None, typer.Option(help="only the first N documents")] = None,
    representation: Annotated[Repr, typer.Option("--repr")] = Repr.slots,
) -> None:
    """Build an index from a corpus and save it."""
    rep = _REPRS[representation]
    total = sum(1 for _ in corpus.rglob("*.txt")) if corpus.is_dir() else None
    if limit is not None and total is not None:
        total = min(total, limit)
    progress = Progress(
        TextColumn("[bold]indexing"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    )
    with progress:
        docs = progress.track(islice(iter_documents(corpus), limit), total=total)
        try:
            index = build_index(docs, rep, positions)
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
                path = Path(index.doc_meta[r.doc_id].path)
                try:
                    snip = snippet(path.read_text(encoding="utf-8"), terms, mark=_MARK)
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


if __name__ == "__main__":
    app()
