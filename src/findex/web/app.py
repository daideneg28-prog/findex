"""findex on the web: a FastAPI app (JSON API + server-rendered UI).

    INDEX_PATH=data/index.pkl uvicorn findex.web.app:app      # or: findex serve

API:  GET /search?q=&k=&page=&scorer=   GET /docs/{doc_id}   GET /stats   GET /health
      OpenAPI browser at /docs, schema at /openapi.json
UI:   GET /  (search page, htmx search-as-you-type)   GET /doc/{doc_id}
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape

from findex import __version__
from findex.corpus import read_document
from findex.index import Index
from findex.query import QuerySyntaxError
from findex.rank import SCORERS, match, search
from findex.snippets import snippet
from findex.store import load
from findex.web.models import (
    DocumentOut,
    ErrorOut,
    Health,
    ScorerName,
    SearchHit,
    SearchResponse,
    StatsOut,
)
from findex.web.settings import Settings, get_settings

HERE = Path(__file__).parent
log = logging.getLogger("findex.web")
request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

_UI_MARK = ("", "")  # private-use chars: survive HTML escaping, never in real text
_PREVIEW_CHARS = 200_000  # the document page shows at most this much text


# --- logging --------------------------------------------------------------------------------


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: what Render/Fly/any log platform can search and filter."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = getattr(record, "request_id", None)
        if rid:
            entry["request_id"] = rid
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


@contextmanager
def configured_logging(settings: Settings) -> Generator[None]:
    """JSON lines from every `findex.*` logger while the app runs; the old setup after."""
    handler = logging.StreamHandler()  # stderr; platforms collect stdout and stderr alike
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        JsonFormatter() if settings.log_json else logging.Formatter("%(levelname)s %(message)s")
    )
    logger = logging.getLogger("findex")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers[:] = [handler]
    logger.setLevel(settings.log_level)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers[:], logger.level, logger.propagate = saved


# --- index lifecycle and dependencies ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load the index once per process, in a background thread.

    The server starts answering at once (/health says 503 "unavailable" until the index is
    in memory), so the platform's health check sees a live process during a slow load.
    """
    settings: Settings = app.state.settings
    app.state.index = None
    app.state.started = time.monotonic()

    async def load_index() -> None:
        try:
            index = await asyncio.to_thread(load, settings.index_path)
        except (OSError, ValueError, TypeError) as exc:
            log.error("cannot load index %s: %s", settings.index_path, exc)
            return
        app.state.index = index
        log.info("index loaded: %r from %s", index, settings.index_path)

    with configured_logging(settings):
        loader = asyncio.create_task(load_index())
        try:
            yield
        finally:  # graceful shutdown (uvicorn handles SIGTERM, then runs this)
            loader.cancel()
            if app.state.index is not None:
                app.state.index.close()
            log.info("shut down")


async def index_state(request: Request) -> Index | None:
    """The loaded index, or None while loading / if loading failed. Override in tests.

    `async def` although it never awaits: FastAPI runs a plain `def` dependency in the
    thread pool, so /health would have to queue behind busy search requests to get it.
    """
    return request.app.state.index


async def get_index(index: Annotated[Index | None, Depends(index_state)]) -> Index:
    if index is None:
        raise HTTPException(503, "index not loaded yet")
    return index


IndexDep = Annotated[Index, Depends(get_index)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
Q = Annotated[str, Query(min_length=1, max_length=200, description="query, e.g. whale NOT ship")]
K = Annotated[int, Query(ge=1, le=100, description="results per page")]
PageQ = Annotated[int, Query(ge=1, le=1000, description="1-based page number")]


# --- the search itself (plain, blocking, CPU-bound) ---------------------------------------------


@lru_cache(maxsize=32)
def _document_text(path: str, name: str) -> str:
    """Book texts are read again and again for snippets; keep the most recent ones."""
    return read_document(Path(path), name)


def run_search(
    index: Index, q: str, k: int, page: int, scorer: ScorerName, mark: tuple[str, str]
) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        matched, terms = match(index, q)
    except QuerySyntaxError as exc:
        raise HTTPException(422, f"query error: {exc}") from exc
    total = len(matched)
    top = search(index, q, SCORERS[scorer], k * page)[k * (page - 1) :]
    hits: list[SearchHit] = []
    for r in top:
        meta = index.doc_meta[r.doc_id]
        try:
            snip = snippet(_document_text(meta.path, meta.name), terms, mark=mark)
        except OSError:
            snip = ""
        hits.append(
            SearchHit(doc_id=r.doc_id, title=r.title, score=round(r.score, 4), snippet=snip)
        )
    return SearchResponse(
        query=q,
        scorer=scorer,
        total=total,
        page=page,
        k=k,
        pages=max(1, math.ceil(total / k)),
        took_ms=round((time.perf_counter() - t0) * 1000, 2),
        results=hits,
    )


# --- the app ------------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="findex",
        version=__version__,
        summary="A small search engine over Project Gutenberg, built lab by lab.",
        lifespan=lifespan,
        responses={500: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    )
    app.state.settings = settings
    app.state.index = None
    app.state.started = time.monotonic()
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET"]
        )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Request id for every log line; unexpected errors become a clean JSON 500."""
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id.set(rid)
        t0 = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                log.exception("unhandled error on %s %s", request.method, request.url.path)
                response = JSONResponse(
                    {"detail": "internal server error", "request_id": rid}, status_code=500
                )
            response.headers["X-Request-ID"] = rid
            log.info(
                "%s %s %d %.1f ms",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - t0) * 1000,
            )
            return response
        finally:
            request_id.reset(token)

    # --- JSON API -------------------------------------------------------------------------

    @app.get("/search", response_model=SearchResponse, tags=["api"])
    def search_route(
        index: IndexDep, q: Q, k: K = 10, page: PageQ = 1, scorer: ScorerName = "bm25"
    ) -> SearchResponse:
        """Ranked, paginated search. `def`, not `async def`: scoring is CPU work, so FastAPI
        runs it in its thread pool instead of on the event loop (see README, Lab 7)."""
        return run_search(index, q, k, page, scorer, ("«", "»"))

    @app.get("/search-async", response_model=SearchResponse, include_in_schema=False)
    async def search_async_route(
        index: IndexDep, q: Q, k: K = 10, page: PageQ = 1, scorer: ScorerName = "bm25"
    ) -> SearchResponse:
        """The WRONG way, kept only for the benchmark: blocking work inside `async def`
        runs on the event loop and stalls every other request meanwhile."""
        return run_search(index, q, k, page, scorer, ("«", "»"))

    @app.get(
        "/docs/{doc_id}",
        response_model=DocumentOut,
        tags=["api"],
        responses={404: {"model": ErrorOut}},
    )
    def document_route(index: IndexDep, doc_id: int) -> DocumentOut:
        """A whole document by its doc_id."""
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(404, f"no document with doc_id {doc_id}")
        try:
            text = _document_text(meta.path, meta.name)
        except OSError as exc:
            raise HTTPException(404, f"document {doc_id} is indexed but its text is gone") from exc
        return DocumentOut(
            doc_id=doc_id,
            name=meta.name,
            title=meta.title,
            length=index.doc_length(doc_id),
            text=text,
        )

    @app.get("/stats", response_model=StatsOut, tags=["api"])
    def stats_route(index: IndexDep, request: Request) -> StatsOut:
        path: Path = request.app.state.settings.index_path
        size = path.stat().st_size / 2**20 if path.exists() else 0.0
        return StatsOut(
            documents=index.num_docs,
            terms=len(index),
            tokens=sum(index.doc_lengths.values()),
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.has_positions,
            index_file_mib=round(size, 1),
            uptime_s=round(time.monotonic() - request.app.state.started, 1),
            version=__version__,
        )

    @app.get("/health", response_model=Health, tags=["ops"], responses={503: {"model": Health}})
    async def health_route(index: Annotated[Index | None, Depends(index_state)]) -> Response:
        """200 when the index is loaded and usable, 503 otherwise (the platform's probe).

        `async def` on purpose (and so are its dependencies): it does no blocking work, so
        it runs right on the event loop and answers even while every thread-pool worker
        is busy with searches."""
        if index is None or index.closed:
            return JSONResponse(Health(status="unavailable").model_dump(), status_code=503)
        return JSONResponse(Health(status="ok", documents=index.num_docs).model_dump())

    # --- HTML UI --------------------------------------------------------------------------

    def highlight(snip: str) -> Markup:
        safe = str(escape(snip))
        return Markup(safe.replace(_UI_MARK[0], "<mark>").replace(_UI_MARK[1], "</mark>"))

    def ui_results(index: Index | None, q: str, page: int, scorer: ScorerName) -> dict[str, Any]:
        context: dict[str, Any] = {"q": q, "page": page, "scorer": scorer, "error": None}
        if index is None:
            context["error"] = "The index is still loading — try again in a few seconds."
        elif q.strip():
            try:
                found = run_search(index, q, settings.page_size, page, scorer, _UI_MARK)
            except HTTPException as exc:
                context["error"] = str(exc.detail)
            else:
                context["found"] = found
                context["hits"] = [(h, highlight(h.snippet)) for h in found.results]
        return context

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        index: Annotated[Index | None, Depends(index_state)],
        q: Annotated[str, Query(max_length=200)] = "",
        page: PageQ = 1,
        scorer: ScorerName = "bm25",
    ) -> HTMLResponse:
        context = ui_results(index, q, page, scorer)
        if index is not None:
            context["documents"] = index.num_docs
        return templates.TemplateResponse(request, "index.html", context)

    @app.get("/ui/results", response_class=HTMLResponse, include_in_schema=False)
    def results_fragment(
        request: Request,
        index: Annotated[Index | None, Depends(index_state)],
        q: Annotated[str, Query(max_length=200)] = "",
        page: PageQ = 1,
        scorer: ScorerName = "bm25",
    ) -> HTMLResponse:
        """Just the results list: htmx swaps it in while the user types."""
        return templates.TemplateResponse(
            request, "_results.html", ui_results(index, q, page, scorer)
        )

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def document_page(request: Request, index: IndexDep, doc_id: int) -> HTMLResponse:
        doc = document_route(index, doc_id)
        truncated = len(doc.text) > _PREVIEW_CHARS
        return templates.TemplateResponse(
            request,
            "doc.html",
            {"doc": doc, "text": doc.text[:_PREVIEW_CHARS], "truncated": truncated},
        )

    return app


app = create_app()
