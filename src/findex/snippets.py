"""Result snippets: a ±80-character window around the best match, with terms highlighted."""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import islice

MAX_MATCHES = 5_000  # don't scan every "the" in the Bible


def _term_pattern(terms: Iterable[str]) -> re.Pattern[str] | None:
    alts = sorted({t for t in terms if t}, key=len, reverse=True)
    if not alts:
        return None
    # our tokens use ASCII ' but the text may use ’ or ʼ
    body = "|".join(re.escape(t).replace("'", "['’ʼ]") for t in alts)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def snippet(
    text: str, terms: Iterable[str], width: int = 80, mark: tuple[str, str] = ("«", "»")
) -> str:
    """Pick the window (±width chars) containing the most *distinct* query terms."""
    pattern = _term_pattern(terms)
    matches = list(islice(pattern.finditer(text), MAX_MATCHES)) if pattern else []
    if not matches:
        return " ".join(text[: 2 * width].split()) + "…"

    best_i, best_j, best_distinct = 0, 0, 0
    j = 0
    for i, m in enumerate(matches):  # sliding window over match start positions
        j = max(j, i)
        while j + 1 < len(matches) and matches[j + 1].start() - m.start() <= 2 * width:
            j += 1
        distinct = len({x.group().casefold() for x in matches[i : j + 1]})
        if distinct > best_distinct:
            best_i, best_j, best_distinct = i, j, distinct

    center = (matches[best_i].start() + matches[best_j].end()) // 2
    lo, hi = max(0, center - width), min(len(text), center + width)
    window = " ".join(text[lo:hi].split())
    assert pattern is not None
    window = pattern.sub(lambda m: f"{mark[0]}{m.group()}{mark[1]}", window)
    return ("…" if lo > 0 else "") + window + ("…" if hi < len(text) else "")
