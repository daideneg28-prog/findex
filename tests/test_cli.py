import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex import __version__
from findex.cli import app
from findex.store import load

runner = CliRunner()


def test_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_index_command_builds_loadable_file(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.pkl"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)])
    assert result.exit_code == 0, result.output
    index = load(out)
    assert index.num_docs == 4 and index.has_positions
    assert "saved to" in result.stdout


def test_index_limit_and_json_format(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ix.json"
    result = runner.invoke(
        app, ["index", str(corpus_dir), "--out", str(out), "--limit", "2", "--no-positions"]
    )
    assert result.exit_code == 0, result.output
    assert load(out).num_docs == 2


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_index_executors_write_identical_files(
    corpus_dir: Path, tmp_path: Path, executor: str
) -> None:
    out = tmp_path / f"{executor}.pkl"
    args = ["index", str(corpus_dir), "-o", str(out), "--executor", executor, "--workers", "2"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    reference = tmp_path / "reference.pkl"
    runner.invoke(app, ["index", str(corpus_dir), "-o", str(reference), "--executor", "serial"])
    assert out.read_bytes() == reference.read_bytes()


def test_index_rejects_array_with_positions(corpus_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["index", str(corpus_dir), "--out", str(tmp_path / "x.pkl"), "--repr", "array"]
    )
    assert result.exit_code == 2
    assert "positions" in result.stderr


def test_search_table(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "whale", "--no-snippets"])
    assert result.exit_code == 0, result.output
    assert "Whale Book" in result.stdout


def test_search_json_lines(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "ship OR apes", "--json", "-k", "5"])
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r["rank"] for r in records] == list(range(1, len(records) + 1))
    assert {r["title"] for r in records} == {"Ship Book", "Whale Book", "Jungle Book"}
    assert all("snippet" in r for r in records)


def test_search_phrase_and_scorer(index_file: Path) -> None:
    result = runner.invoke(
        app, ["search", str(index_file), '"white whale"', "--scorer", "tfidf", "--json"]
    )
    assert result.exit_code == 0
    assert [json.loads(line)["title"] for line in result.stdout.splitlines()] == ["Whale Book"]


def test_search_no_results_is_not_an_error(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "unicorn"])
    assert result.exit_code == 0
    assert "no documents match" in result.stderr


def test_bad_query_exits_1(index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "(whale"])
    assert result.exit_code == 1
    assert "query error" in result.stderr


def test_missing_index_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "nope.pkl"), "x"])
    assert result.exit_code == 2


def test_verbose_logs_to_stderr(index_file: Path) -> None:
    result = runner.invoke(app, ["-v", "search", str(index_file), "whale", "--json"])
    assert result.exit_code == 0
    assert "took" in result.stderr  # @timed messages
    for line in result.stdout.splitlines():
        json.loads(line)  # stdout stays pure JSON


def test_stats(index_file: Path) -> None:
    result = runner.invoke(app, ["stats", str(index_file), "--top", "3"])
    assert result.exit_code == 0
    assert "documents" in result.stdout and "the" in result.stdout
