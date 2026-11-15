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


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"  # what `\w` means for str patterns


def find_matches(text: str, terms: Iterable[str], lowered: str | None = None) -> list[Match]:
    """(start, end, term) of whole-word, case-insensitive occurrences, in text order.

    Lab 8: the old version ran one IGNORECASE regex with look-arounds over the whole
    book — 190 ms for the 5.5 MB Bible, 97% of a /search request. Here every term is
    found with str.find (a C memchr-style scan) in a lower-cased copy that the caller
    caches, and the word-boundary check runs only at the hits: ~14x faster.
    """
    low = lowered if lowered is not None and len(lowered) == len(text) else text.lower()
    if len(low) != len(text):  # rare characters whose lower() changes the length
        pattern = _term_pattern(terms)
        found = pattern.finditer(text) if pattern else iter(())
        return [(m.start(), m.end(), m.group().casefold()) for m in islice(found, MAX_MATCHES)]
    hits: list[Match] = []
    for term in {t for t in terms if t}:
        variants = {term, term.replace("'", "’"), term.replace("'", "ʼ")}
        for needle in {v.lower() for v in variants}:
            start, n = low.find(needle), len(needle)
            while start != -1 and len(hits) < 4 * MAX_MATCHES:
                end = start + n
                if (start == 0 or not _is_word(low[start - 1])) and (
                    end == len(low) or not _is_word(low[end])
                ):
                    hits.append((start, end, term))
                start = low.find(needle, start + 1)
    hits.sort()
    return hits[:MAX_MATCHES]


type Match = tuple[int, int, str]


def snippet(
    text: str,
    terms: Iterable[str],
    width: int = 80,
    mark: tuple[str, str] = ("«", "»"),
    lowered: str | None = None,
) -> str:
    """Pick the window (±width chars) containing the most *distinct* query terms.

    `lowered` is `text.lower()`, if the caller has it cached (the web app does).
    """
    terms = tuple(terms)
    matches = find_matches(text, terms, lowered)
    if not matches:
        return " ".join(text[: 2 * width].split()) + "…"

    best_i, best_j, best_distinct = 0, 0, 0
    j = 0
    for i, (start, _, _) in enumerate(matches):  # sliding window over match starts
        j = max(j, i)
        while j + 1 < len(matches) and matches[j + 1][0] - start <= 2 * width:
            j += 1
        distinct = len({term for _, _, term in matches[i : j + 1]})
        if distinct > best_distinct:
            best_i, best_j, best_distinct = i, j, distinct

    center = (matches[best_i][0] + matches[best_j][1]) // 2
    lo, hi = max(0, center - width), min(len(text), center + width)
    window = " ".join(text[lo:hi].split())
    pattern = _term_pattern(terms)  # the regex is fine on a 160-character window
    assert pattern is not None
    window = pattern.sub(lambda m: f"{mark[0]}{m.group()}{mark[1]}", window)
    return ("…" if lo > 0 else "") + window + ("…" if hi < len(text) else "")
