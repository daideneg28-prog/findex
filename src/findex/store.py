"""Saving and loading an Index. The format is chosen by file extension.

* ``.pkl`` (pickle): fastest and smallest code. SECURITY WARNING: ``pickle.load``
  executes arbitrary code. A pickle file is a small program for the unpickler; it can
  call any importable function, e.g. ``os.system("...")``, through ``__reduce__``.
  Only ever load index files that *you* built on *your* machine. Never load a pickle
  downloaded from the internet or received over the network.
* ``.json``: safe (it can only produce dicts, lists, strings and numbers) and
  human-readable, but bigger and slower. Postings are stored column-wise
  (``[doc_ids, tfs]``) so we do not repeat key names millions of times.
"""

from __future__ import annotations

import json
import pickle
from array import array
from pathlib import Path
from typing import Any

from findex.index import DocMeta, Index, PlainPosting, Posting

JSON_FORMAT_VERSION = 1


def save(index: Index, path: Path) -> None:
    path = Path(path)
    if path.suffix == ".json":
        _save_json(index, path)
    else:
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> Index:
    path = Path(path)
    if path.suffix == ".json":
        return _load_json(path)
    with path.open("rb") as f:
        # Safe ONLY because we created this file ourselves — see module docstring.
        index = pickle.load(f)
    if not isinstance(index, Index):
        raise TypeError(f"{path} does not contain a findex Index")
    return index


def _save_json(index: Index, path: Path) -> None:
    postings: dict[str, list[Any]] = {}
    for term in index.postings:
        entry = index.postings[term]
        if index.representation == "array":
            ids, tfs = entry  # type: ignore[misc]
            postings[term] = [ids.tolist(), tfs.tolist()]
        else:
            col: list[Any] = [[p.doc_id for p in entry], [p.tf for p in entry]]  # type: ignore[union-attr]
            if index.has_positions:
                col.append([list(p.positions) for p in entry])  # type: ignore[union-attr]
            postings[term] = col
    payload = {
        "version": JSON_FORMAT_VERSION,
        "representation": index.representation,
        "has_positions": index.has_positions,
        "docs": [
            [m.doc_id, m.name, m.path, m.title, index.doc_lengths[m.doc_id]]
            for m in index.doc_meta.values()
        ],
        "postings": postings,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> Index:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != JSON_FORMAT_VERSION:
        raise ValueError(f"unsupported index format version in {path}")
    representation = payload["representation"]
    has_positions = payload["has_positions"]

    doc_meta: dict[int, DocMeta] = {}
    doc_lengths: dict[int, int] = {}
    for doc_id, name, doc_path, title, length in payload["docs"]:
        doc_meta[doc_id] = DocMeta(doc_id, name, doc_path, title)
        doc_lengths[doc_id] = length

    postings: dict[str, Any] = {}
    cls = Posting if representation == "slots" else PlainPosting
    for term, col in payload["postings"].items():
        if representation == "array":
            postings[term] = (array("I", col[0]), array("I", col[1]))
        elif has_positions:
            postings[term] = [
                cls(d, tf, tuple(pos)) for d, tf, pos in zip(col[0], col[1], col[2], strict=True)
            ]
        else:
            postings[term] = [cls(d, tf) for d, tf in zip(col[0], col[1], strict=True)]
    return Index(representation, postings, doc_lengths, doc_meta, has_positions)
