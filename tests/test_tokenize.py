import inspect

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_is_generator() -> None:
    assert inspect.isgeneratorfunction(tokenize)


def test_mixed_case_and_casefold() -> None:
    assert toks("Hello WORLD hello") == ["hello", "world", "hello"]
    assert toks("Straße STRASSE") == ["strasse", "strasse"]


def test_cyrillic() -> None:
    assert toks("Привіт, Світе! Їжак і Ґанок") == ["привіт", "світе", "їжак", "і", "ґанок"]


def test_combining_mark_is_normalized() -> None:
    # "café" written with a precomposed é vs. e + U+0301: same token after NFC
    assert toks("café") == toks("café") == ["café"]


def test_stress_mark_does_not_split_word() -> None:
    # Cyrillic а + U+0301 has no precomposed form; the word must stay whole
    assert toks("заслі́на") == ["заслі́на"]


def test_punctuation_is_dropped() -> None:
    assert toks("Wait... what?! (yes) -- no; ok.") == ["wait", "what", "yes", "no", "ok"]


def test_apostrophes() -> None:
    assert toks("don't п'ять п’ять пʼять") == ["don't", "п'ять", "п'ять", "п'ять"]
    assert toks("'quoted'") == ["quoted"]


def test_hyphens_split() -> None:
    assert toks("state-of-the-art") == ["state", "of", "the", "art"]


def test_digits() -> None:
    assert toks("In 1984 pi was 3.14, covid19") == ["in", "1984", "pi", "was", "3", "14", "covid19"]


def test_empty_and_whitespace() -> None:
    assert toks("") == []
    assert toks("   \n\t ") == []
    assert toks("___") == []
