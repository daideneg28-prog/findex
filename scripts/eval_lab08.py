"""Lab 8: precision@5 of keyword vs semantic vs hybrid on 10 + 5 labeled queries.

findex embed data/index.pkl --out data/index.emb
python scripts/eval_lab08.py data/index.pkl data/index.emb
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_lab03 import QUERIES

from findex.index import Index
from findex.semantic import SemanticIndex, keyword_ranking, rrf
from findex.store import load

# Paraphrases: the same information needs as above, in words the books barely use.
PARAPHRASES: list[tuple[str, set[str]]] = [
    ("boy raised by wild primates in africa", QUERIES[0][1]),
    ("warrior romance on the red planet", QUERIES[1][1]),
    ("mischievous kid on the mississippi riverbank", QUERIES[2][1]),
    ("orphan pupil mistreated at a boarding school", QUERIES[8][1]),
    ("fall of mankind and the rebel angels", QUERIES[9][1]),
]


def p_at_5(index: Index, ranking: list[int], relevant: set[str]) -> float:
    return sum(index.doc_meta[d].name in relevant for d in ranking[:5]) / 5


def main() -> None:
    index = load(Path(sys.argv[1]))
    semantic = SemanticIndex.load(Path(sys.argv[2]))
    print("| # | Query | Kind | Keyword (BM25, OR) | Semantic | Hybrid (RRF) |")
    print("|---:|---|---|---:|---:|---:|")
    totals = {kind: [0.0, 0.0, 0.0] for kind in ("original", "paraphrase")}
    rows = [(q, rel, "original") for q, rel in QUERIES]
    rows += [(q, rel, "paraphrase") for q, rel in PARAPHRASES]
    for i, (query, relevant, kind) in enumerate(rows, start=1):
        lexical = keyword_ranking(index, " OR ".join(query.split()), 100)
        meaning = [d for d, _ in semantic.search(query, 100)]
        hybrid = [d for d, _ in rrf([lexical, meaning])]
        scores = [p_at_5(index, r, relevant) for r in (lexical, meaning, hybrid)]
        totals[kind] = [t + s for t, s in zip(totals[kind], scores, strict=True)]
        cells = " | ".join(f"{s:.1f}" for s in scores)
        print(f"| {i} | `{query}` | {kind} | {cells} |")
    for kind, n in (("original", len(QUERIES)), ("paraphrase", len(PARAPHRASES))):
        cells = " | ".join(f"**{t / n:.2f}**" for t in totals[kind])
        print(f"| | **mean, {kind}** | | {cells} |")
    both = [a + b for a, b in zip(totals["original"], totals["paraphrase"], strict=True)]
    n = len(rows)
    print("| | **mean, all 15** | | " + " | ".join(f"**{t / n:.2f}**" for t in both) + " |")


if __name__ == "__main__":
    main()
