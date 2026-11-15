"""Set operations on sorted doc_id lists.

Two implementations of the same three operations:
* two-pointer merges — O(len(a) + len(b)), streaming, no hashing (what real engines do);
* Python sets — hashing in C plus a final sort (the Lab 2 benchmark compares them).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

Op = Literal["AND", "OR", "NOT"]
Engine = Literal["merge", "set"]
SetOp = Callable[[Sequence[int], Sequence[int]], list[int]]


def intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    """Elements of a that are not in b."""
    out: list[int] = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def set_intersect(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_union(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_difference(a: Sequence[int], b: Sequence[int]) -> list[int]:
    return sorted(set(a) - set(b))


ENGINES: dict[Engine, dict[Op, SetOp]] = {
    "merge": {"AND": intersect, "OR": union, "NOT": difference},
    "set": {"AND": set_intersect, "OR": set_union, "NOT": set_difference},
}
