import inspect
import json
import logging
from pathlib import Path

import pytest

from findex.corpus import Document, iter_documents
from findex.stats import compute_stats, compute_stats_eager
from findex.stats import main as stats_main


def test_iter_documents_is_lazy(corpus_dir: Path) -> None:
    assert inspect.isgeneratorfunction(iter_documents)
    first = next(iter_documents(corpus_dir))
    assert isinstance(first, Document) and first.text


def test_jsonl_documents(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "docs.jsonl"
    lines = [json.dumps({"id": "x", "text": "hello"}), "", "{not json", json.dumps({"text": "hi"})]
    path.write_text("\n".join(lines), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        docs = list(iter_documents(path))
    assert [(d.doc_id, d.text) for d in docs] == [("x", "hello"), ("docs:4", "hi")]
    assert "invalid JSON" in caplog.text


def test_bad_encoding_is_logged_not_fatal(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "bad.txt").write_bytes(b"caf\xe9 ok")  # latin-1 byte, invalid UTF-8
    with caplog.at_level(logging.WARNING):
        [doc] = iter_documents(tmp_path)
    assert doc.text == "caf� ok"
    assert "bad encoding" in caplog.text


def test_lazy_and_eager_stats_agree(tiny_docs: list[Document]) -> None:
    lazy, eager = compute_stats(iter(tiny_docs)), compute_stats_eager(iter(tiny_docs))
    assert (lazy.documents, lazy.tokens, lazy.counts) == (
        eager.documents,
        eager.tokens,
        eager.counts,
    )
    assert lazy.documents == 4 and lazy.vocabulary == len(lazy.counts)
    assert lazy.counts.most_common(1)[0][0] == "the"


def test_stats_cli(corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert stats_main([str(corpus_dir), "--limit", "2", "--top", "3", "--mode", "eager"]) == 0
    output = capsys.readouterr().out
    assert "documents:   2" in output and "peak memory" in output
