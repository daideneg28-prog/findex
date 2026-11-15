"""Lab 5: the parallel build must produce exactly the Lab 4 index, whatever the executor."""

import pickle
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from findex.corpus import iter_documents, iter_paths
from findex.index import Index, Representation, build_index
from findex.parallel import (
    BuildStats,
    Executor,
    build_from_paths,
    build_parallel,
    build_partial,
    chunk_paths,
    merge,
)


def _bytes(index: Index) -> bytes:
    return pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)


def _more_docs(corpus_dir: Path) -> Path:
    """Add a few more documents (and a .jsonl file) so there are several chunks."""
    for i in range(5, 12):
        text = f"document number {i} about whales and ships and apes, {'sea ' * i}"
        (corpus_dir / f"{i}.txt").write_text(text, encoding="utf-8")
    lines = ['{"id": "j1", "text": "a json whale"}', '{"id": "j2", "text": "a json ship"}']
    (corpus_dir / "extra.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return corpus_dir


@pytest.mark.parametrize("positions", [True, False])
@pytest.mark.parametrize("representation", ["slots", "plain", "array"])
def test_serial_merge_is_byte_identical_to_lab4(
    corpus_dir: Path, representation: Representation, positions: bool
) -> None:
    if positions and representation == "array":
        pytest.skip("array representation has no positions")
    root = _more_docs(corpus_dir)
    expected = build_index(iter_documents(root), representation, positions)
    actual = merge([build_partial(list(iter_paths(root)), positions)], representation, positions)
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_every_executor_gives_the_same_index(corpus_dir: Path, executor: Executor) -> None:
    root = _more_docs(corpus_dir)
    expected = build_index(iter_documents(root), positions=True)
    stats = BuildStats()
    actual = build_parallel(root, executor=executor, workers=3, stats=stats)
    assert stats.chunks > 1  # really split into several partial indexes
    assert _bytes(actual) == _bytes(expected)
    assert actual.doc_meta[0].title == expected.doc_meta[0].title


def test_processes_report_worker_cpu(corpus_dir: Path) -> None:
    stats = BuildStats()
    build_parallel(_more_docs(corpus_dir), executor="processes", workers=2, stats=stats)
    assert len(stats.worker_pids) == 2
    # Linux counts process CPU in 10 ms ticks: a tiny corpus can legitimately read as 0
    assert stats.worker_cpu_s >= 0 and stats.merge_s >= 0 and stats.map_s > 0


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_worker_exception_surfaces(corpus_dir: Path, executor: Executor) -> None:
    unreadable = corpus_dir / "not-a-file.txt"
    unreadable.mkdir()  # listed fine, but reading it fails inside the worker
    paths = [*iter_paths(corpus_dir), unreadable]
    with pytest.raises(OSError) as info:
        build_from_paths(paths, executor=executor, workers=2, chunks_per_worker=1)
    if executor == "processes":
        # the worker's own traceback travels back as the exception's cause
        assert "build_partial" in str(info.value.__cause__)
    assert not isinstance(info.value, BrokenProcessPool)


def test_chunks_are_contiguous_and_balanced(tmp_path: Path) -> None:
    paths: list[Path] = []
    for i, size in enumerate([1000, 10, 10, 10, 1000, 10, 10, 1000]):
        path = tmp_path / f"{i}.txt"
        path.write_text("x" * size)
        paths.append(path)
    chunks = chunk_paths(paths, 3)
    assert [p for c in chunks for p in c] == paths  # order kept, nothing lost
    assert len(chunks) == 3 and all(chunks)
    assert chunk_paths([], 4) == []
    assert len(chunk_paths(paths[:2], 8)) == 2  # never more chunks than files


def test_positions_with_array_rejected(corpus_dir: Path) -> None:
    with pytest.raises(ValueError, match="positions"):
        build_parallel(corpus_dir, executor="serial", representation="array", positions=True)
    with pytest.raises(ValueError, match="positions"):
        merge([], "array", positions=True)
