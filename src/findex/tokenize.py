"""Streaming tokenizer.

Policy (also documented in README):

1. Normalization: Unicode NFC, then str.casefold() ("Straße" -> "strasse").
2. A token is a run of letters/digits (Unicode-aware; underscore excluded).
   Combining marks left after NFC (e.g. a stress mark on Cyrillic "а́") stay
   inside the token instead of splitting the word.
3. Apostrophes: an apostrophe *between* letters keeps the word whole
   ("don't", "п'ять"). The variants ’ (U+2019) and ʼ (U+02BC) are mapped to
   the ASCII ' so all spellings of "п'ять" become one token. Leading/trailing
   apostrophes (quotes) are dropped.
4. Hyphens split words: "state-of-the-art" -> state, of, the, art. A query for
   "art" should find it.
5. Digits are tokens: "1984" -> "1984"; "3.14" -> "3", "14"; mixed runs
   such as "covid19" stay one token.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_CHAR = r"(?:[^\W_]|[̀-ͯ])"  # letter/digit or a combining mark
_APOSTROPHES = "'’ʼ"
TOKEN_RE = re.compile(rf"[^\W_]{_CHAR}*(?:[{_APOSTROPHES}]{_CHAR}+)*")
_APOS_TABLE = str.maketrans({"’": "'", "ʼ": "'"})


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens from `text` lazily (re.finditer, not findall)."""
    for match in TOKEN_RE.finditer(normalize(text)):
        yield match.group().translate(_APOS_TABLE)
