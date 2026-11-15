from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import build_index
from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse

a, b, c = Term("a"), Term("b"), Term("c")


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", a),
        ("a b", And(a, b)),
        ("a AND b", And(a, b)),
        ("a OR b c", Or(a, And(b, c))),  # AND binds tighter than OR
        ("(a OR b) c", And(Or(a, b), c)),
        ("a NOT b", And(a, Not(b))),
        ("NOT NOT a", Not(Not(a))),
        ("a b c", And(And(a, b), c)),  # left-associative
        ('"a b" c', And(Phrase(("a", "b")), c)),
        ("A, b!", And(a, b)),  # words are normalized by the tokenizer
        ("a or b", And(And(a, Term("or")), b)),  # lowercase "or" is a word
        ("state-of-the-art", Phrase(("state", "of", "the", "art"))),
        ('"a"', a),
    ],
)
def test_parse_trees(query: str, tree) -> None:
    assert parse(query) == tree


def test_operator_overloading_builds_same_tree() -> None:
    assert (a & (b | c) & ~Term("d")) == parse("a (b OR c) NOT d")


@pytest.mark.parametrize("bad", ["", "   ", "!!!", "(a", "a )", "a OR", "NOT"])
def test_syntax_errors(bad: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(bad)


DOCS = [
    Document("0", Path("0.txt"), "the white whale and the old ship"),
    Document("1", Path("1.txt"), "a whale is white; the ship is old"),
    Document("2", Path("2.txt"), "captain of the ship"),
]


@pytest.fixture(scope="module")
def index():
    return build_index(DOCS, positions=True)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("whale", [0, 1]),
        ("whale ship", [0, 1]),
        ("whale OR captain", [0, 1, 2]),
        ("ship NOT whale", [2]),
        ("NOT whale", [2]),
        ("(whale OR captain) NOT old", [2]),
        ('"white whale"', [0]),
        ('"whale white"', []),
        ('"old ship"', [0]),
        ('"the ship"', [1, 2]),
        ("unicorn", []),
    ],
)
def test_evaluate(index, query: str, expected: list[int]) -> None:
    assert parse(query).evaluate(index) == expected


def test_phrase_without_positions_degrades_to_and() -> None:
    idx = build_index(DOCS, positions=False)
    assert parse('"whale white"').evaluate(idx) == [0, 1]


def test_scoring_terms_skip_negated() -> None:
    assert list(parse('a "b c" NOT d (e OR NOT f)').terms()) == ["a", "b", "c", "e"]
