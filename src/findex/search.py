"""Ranked search over a saved index.

    python -m findex.search index.pkl 'tarzan "the apes" NOT jane' [more queries...]
        [--scorer bm25|tfidf] [-k 10] [--no-snippets]

Query syntax (see findex.query): implicit AND, OR, NOT, parentheses, "quoted phrases".
Several queries can be given; repeating one shows the lru_cache hit in the timing log.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from findex.index import open_index
from findex.query import QuerySyntaxError
from findex.rank import SCORERS, cache_info, match, search
from findex.snippets import snippet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m findex.search", description=__doc__)
    parser.add_argument("index", type=Path, help="index file built by findex.index")
    parser.add_argument("queries", nargs="+", metavar="query")
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    parser.add_argument("-k", type=int, default=10, help="how many results to show")
    parser.add_argument("--no-snippets", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="  [log] %(message)s", stream=sys.stderr)
    with open_index(args.index) as index:
        print(repr(index))
        for query in args.queries:
            print(f"\n=== {query!r} ({args.scorer})")
            try:
                results = search(index, query, SCORERS[args.scorer], args.k)
            except QuerySyntaxError as exc:
                print(f"  query error: {exc}")
                continue
            matched, terms = match(index, query)  # served from the cache filled by search()
            print(f"  {len(matched)} documents match; top {len(results)}:")
            for rank, r in enumerate(results, start=1):
                print(f"  {rank:>2}. {r.score:7.3f}  {r.title}")
                if not args.no_snippets:
                    text = Path(index.doc_meta[r.doc_id].path).read_text(encoding="utf-8")
                    print(f"      {snippet(text, terms)}")
        print(f"\n{cache_info()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
