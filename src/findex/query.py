"""Query language: a recursive-descent parser that builds a tree of nodes.

Grammar (lowest to highest precedence):

    or_expr  → and_expr ("OR" and_expr)*
    and_expr → not_expr (["AND"] not_expr)*        # AND is optional: "a b" == "a AND b"
    not_expr → "NOT" not_expr | atom
    atom     → WORD | "quoted phrase" | "(" or_expr ")"

Operators must be UPPERCASE; a lowercase "or" is just a word. Nodes overload
`&`, `|`, `~`, so a query can also be written in code:
    Term("python") & (Term("async") | Term("await")) & ~Term("java")
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass

from findex.index import Index
from findex.merge import difference, intersect, union
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


class QuerySyntaxError(ValueError):
    pass


class Node:
    """Base class: operator overloading shared by every node type."""

    __slots__ = ()

    def __and__(self, other: Node) -> And:
        return And(self, other)

    def __or__(self, other: Node) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)

    def evaluate(self, index: Index) -> list[int]:
        """Sorted doc_ids that match this (sub)query."""
        raise NotImplementedError

    def terms(self) -> Iterator[str]:
        """Terms that should contribute to the score (everything not under a NOT)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index) -> list[int]:
        return list(index.doc_ids(self.text))

    def terms(self) -> Iterator[str]:
        yield self.text


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> list[int]:
        candidates = list(index.doc_ids(self.words[0]))
        for word in self.words[1:]:
            candidates = intersect(candidates, index.doc_ids(word))
        if not index.has_positions:
            log.warning("index has no positions; phrase %r treated as AND", " ".join(self.words))
            return candidates
        return [doc_id for doc_id in candidates if self._occurs_in(index, doc_id)]

    def _occurs_in(self, index: Index, doc_id: int) -> bool:
        """Two-pointer merge on positions: keep positions where the phrase so far ends."""
        ends: list[int] = list(_positions(index, self.words[0], doc_id))
        for word in self.words[1:]:
            ends = intersect([p + 1 for p in ends], _positions(index, word, doc_id))
            if not ends:
                return False
        return True

    def terms(self) -> Iterator[str]:
        yield from self.words


def _positions(index: Index, term: str, doc_id: int) -> tuple[int, ...]:
    postings = index[term]
    i = bisect_left(postings, doc_id, key=lambda p: p.doc_id)  # postings are sorted by doc_id
    return postings[i].positions


@dataclass(frozen=True, slots=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        # "a NOT b" is And(a, Not(b)): a difference is cheaper than complementing b
        if isinstance(self.right, Not):
            return difference(self.left.evaluate(index), self.right.child.evaluate(index))
        if isinstance(self.left, Not):
            return difference(self.right.evaluate(index), self.left.child.evaluate(index))
        return intersect(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True, slots=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        return union(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True, slots=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> list[int]:
        return difference(range(index.num_docs), self.child.evaluate(index))

    def terms(self) -> Iterator[str]:
        return iter(())


# --- lexer -------------------------------------------------------------------------------

_LEX_RE = re.compile(r'\s*(?:(?P<lp>\()|(?P<rp>\))|"(?P<phrase>[^"]*)"?|(?P<word>[^\s()"]+))')
_OPERATORS = {"AND", "OR", "NOT"}


def _lex(query: str) -> list[tuple[str, object]]:
    """Turn the query string into (kind, value) tokens; words go through the tokenizer."""
    tokens: list[tuple[str, object]] = []
    for m in _LEX_RE.finditer(query):
        if m["lp"]:
            tokens.append(("(", None))
        elif m["rp"]:
            tokens.append((")", None))
        elif m["phrase"] is not None:
            words = tuple(tokenize(m["phrase"]))
            if words:
                tokens.append(("atom", Term(words[0]) if len(words) == 1 else Phrase(words)))
        elif m["word"]:
            raw = m["word"]
            if raw in _OPERATORS:
                tokens.append((raw, None))
                continue
            # "state-of-the-art" tokenizes to several words: treat it as a phrase
            words = tuple(tokenize(raw))
            if words:
                tokens.append(("atom", Term(words[0]) if len(words) == 1 else Phrase(words)))
    return tokens


# --- parser -------------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, object]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, object]:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() not in (None, ")", "OR"):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind = self.peek()
        if kind is None:
            raise QuerySyntaxError("unexpected end of query")
        if kind == "(":
            self.take()
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing ')'")
            self.take()
            return node
        if kind == "atom":
            return self.take()[1]  # type: ignore[return-value]
        raise QuerySyntaxError(f"unexpected {kind!r}")


def parse(query: str) -> Node:
    """Parse a query string into a node tree. Raises QuerySyntaxError on bad input."""
    parser = _Parser(_lex(query))
    if parser.peek() is None:
        raise QuerySyntaxError("empty query")
    node = parser.or_expr()
    if parser.peek() is not None:
        raise QuerySyntaxError(f"unexpected {parser.peek()!r}")
    return node
