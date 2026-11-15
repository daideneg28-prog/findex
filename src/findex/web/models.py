"""Pydantic models: the validated boundary between HTTP and findex's own dataclasses.

Inside, findex keeps its Lab 2-3 dataclasses (Posting, DocMeta, SearchResult); these
models exist only at the edge, where untrusted JSON comes in or goes out.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ScorerName = Literal["bm25", "tfidf"]


class SearchHit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = Field(description="text around the best match; matched terms wrapped in «»")


class SearchResponse(BaseModel):
    query: str
    scorer: ScorerName
    total: int = Field(description="number of matching documents (all pages)")
    page: int
    k: int
    pages: int
    took_ms: float
    results: list[SearchHit]


class DocumentOut(BaseModel):
    doc_id: int
    name: str
    title: str
    length: int = Field(description="number of tokens")
    text: str


class StatsOut(BaseModel):
    documents: int
    terms: int
    tokens: int
    avg_doc_length: float
    positions: bool
    index_file_mib: float
    uptime_s: float
    version: str


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    documents: int = 0


class ErrorOut(BaseModel):
    detail: str
    request_id: str | None = None
