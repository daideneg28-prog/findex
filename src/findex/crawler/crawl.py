"""The crawler: a Queue frontier, worker tasks in a TaskGroup, an async-generator API.

    async for page in crawl(["https://docs.python.org/3/"], max_pages=200):
        ...

Politeness is layered:
  * robots.txt (RobotsCache) before every request;
  * a per-host limiter: at most `per_host` requests in flight to one server, and request
    starts at least `min_delay` seconds apart (or the site's Crawl-delay if larger);
  * a global semaphore bounding all in-flight requests;
  * an allowlist of domains and a hard page budget.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import time
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import TextIO
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import certifi
import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)

_DEFAULT_PORTS = {"http": 80, "https": 443}
_SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".bz2", ".xz", ".tar", ".tgz", ".exe", ".msi", ".dmg", ".whl", ".epub", ".mp3",
    ".mp4", ".css", ".js", ".json", ".xml", ".txt", ".woff", ".woff2", ".ttf",
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str  # ISO 8601, UTC

    def to_json(self) -> dict[str, str]:
        """One JSON line in the format corpus.iter_documents reads."""
        return {
            "id": self.url,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "fetched_at": self.fetched_at,
        }


@dataclass(slots=True)
class CrawlStats:
    """Live counters (read by the CLI's progress display)."""

    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queued: int = 0
    robots_blocked: int = 0
    retries: int = 0
    bytes: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def pages_per_s(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Absolute, normalized URL; None for anything that is not an http(s) page.

    scheme and host lower-cased, default port dropped, fragment removed, empty path -> "/",
    query parameters sorted (`?b=2&a=1` and `?a=1&b=2` are the same page).
    """
    url = urljoin(base, url.strip()) if base else url.strip()
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, parts.path or "/", query, ""))


def seen_key(url: str) -> str:
    """Dedup key: `/library` and `/library/` are one page (we still fetch the URL as found)."""
    return url[:-1] if url.endswith("/") and url.count("/") > 3 else url


class _Extractor(HTMLParser):
    """Title, visible text and links of an HTML page (stdlib parser: no native code)."""

    _SKIP = frozenset({"script", "style", "noscript", "template", "svg", "nav", "footer"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.links: list[str] = []
        self._text: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title" and not self._skip_depth and not self.title:
            self._in_title = True  # the page's first <title>, not an <svg><title> tooltip
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth and data.strip():
            self._text.append(data.strip())

    @property
    def text(self) -> str:
        return " ".join(self._text)


def extract(html: str) -> tuple[str, str, list[str]]:
    """(title, text, raw links) of an HTML page."""
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.text, parser.links


def parse_page(html: str, base: str) -> tuple[str, str, list[str]]:
    """extract() plus canonical, crawlable links: ALL the per-page CPU work in one call.

    Takes 50-150 ms on a big documentation page (800 KB of HTML, 1400 links), so the
    crawler runs it with asyncio.to_thread: the event loop keeps serving sockets meanwhile.
    """
    title, text, raw = extract(html)
    links: list[str] = []
    for link in raw:
        if (absolute := canonicalize(link, base)) and not absolute.endswith(_SKIP_SUFFIXES):
            links.append(absolute)
    return title, text, links


class HostLimiter:
    """Per-host concurrency (a Semaphore per host) plus a minimum gap between starts."""

    def __init__(self, per_host: int, min_delay: float) -> None:
        self.per_host = per_host
        self.min_delay = min_delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._next_start: dict[str, float] = {}

    @asynccontextmanager
    async def slot(self, host: str, delay: float | None = None) -> AsyncGenerator[None]:
        sem = self._sems.setdefault(host, asyncio.Semaphore(self.per_host))
        async with sem:
            loop = asyncio.get_running_loop()
            now = loop.time()
            # read-modify-write with no await in between: atomic in asyncio, no lock needed
            start = max(now, self._next_start.get(host, now))
            self._next_start[host] = start + max(self.min_delay, delay or 0.0)
            if start > now:
                await asyncio.sleep(start - now)
            yield


def _allowed_domain(url: str, domains: frozenset[str]) -> bool:
    host = urlsplit(url).hostname or ""
    return any(host == d or host.endswith("." + d) for d in domains)


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.5,
    allowed_domains: Iterable[str] | None = None,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
    log_file: TextIO | None = None,
) -> AsyncGenerator[Page]:
    """Yield pages as they are fetched (an async generator over the frontier).

    The workers run in a background task that owns the TaskGroup; pages reach this
    generator through a bounded queue, so a slow consumer slows the crawl down
    (back-pressure) instead of piling pages up in memory.
    """
    seeds = [s for s in (canonicalize(u) for u in seeds) if s]
    domains = frozenset(
        d.lower() for d in (allowed_domains or {urlsplit(s).hostname or "" for s in seeds})
    )
    stats = stats if stats is not None else CrawlStats()
    out: asyncio.Queue[Page | None] = asyncio.Queue(maxsize=concurrency * 2)
    runner = asyncio.create_task(
        _run(seeds, domains, max_pages, concurrency, per_host, min_delay, timeout,
             client, stats, log_file, out)
    )  # fmt: skip
    try:
        while (page := await out.get()) is not None:
            yield page
        await runner  # re-raises if the crawl itself failed
    finally:  # consumer stopped early (break, exception, Ctrl-C): stop the workers
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner


async def _run(
    seeds: list[str],
    domains: frozenset[str],
    max_pages: int,
    concurrency: int,
    per_host: int,
    min_delay: float,
    timeout: float,
    client: httpx.AsyncClient | None,
    stats: CrawlStats,
    log_file: TextIO | None,
    out: asyncio.Queue[Page | None],
) -> None:
    own_client = client is None
    if client is None:
        # Building the TLS context reads ~300 KB of CA certificates from disk: ~0.3 s that
        # PYTHONASYNCIODEBUG flagged as blocking the loop. Done in a thread instead.
        tls = await asyncio.to_thread(ssl.create_default_context, cafile=certifi.where())
        client = httpx.AsyncClient(
            verify=tls,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        )
    frontier: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()
    robots = RobotsCache(client)
    hosts = HostLimiter(per_host, min_delay)
    in_flight = asyncio.Semaphore(concurrency)  # global bound on open requests
    done = asyncio.Event()
    reserved = 0  # pages started or finished; never exceeds max_pages

    def enqueue(url: str) -> None:
        key = seen_key(url)
        if key not in seen and _allowed_domain(url, domains):
            seen.add(key)
            frontier.put_nowait(url)
            stats.queued = frontier.qsize()

    def record(result: FetchResult) -> None:
        if log_file is not None:
            status = result.status if result.status is not None else "ERR"
            log_file.write(
                f"{datetime.now(UTC).isoformat(timespec='seconds')}\t{status}\t{result.size}\t"
                f"{result.elapsed:.3f}\t{result.attempts}\t{result.url}\t{result.error or ''}\n"
            )
            log_file.flush()

    async def visit(url: str) -> None:
        nonlocal reserved
        if not await robots.allowed(url):
            stats.robots_blocked += 1
            log.info("robots.txt disallows %s", url)
            return
        if reserved >= max_pages:
            return
        reserved += 1  # claim a slot of the budget before awaiting the network
        host = urlsplit(url).hostname or ""
        delay = await robots.crawl_delay(url)
        async with hosts.slot(host, delay), in_flight:
            stats.in_flight += 1
            try:
                result = await fetch(client, url, timeout=timeout)
            finally:
                stats.in_flight -= 1
        record(result)
        stats.retries += result.attempts - 1
        stats.bytes += result.size
        final = canonicalize(result.final_url)
        if not result.is_html or not final or not _allowed_domain(final, domains):
            reserved -= 1  # nothing indexable: give the slot back
            if not result.ok or result.error:
                stats.errors += 1
            return
        seen.add(seen_key(final))  # a redirect target is visited too
        # HTML parsing is CPU work: run it in a worker thread so the loop keeps serving I/O
        title, text, links = await asyncio.to_thread(parse_page, result.text, final)
        for link in links:
            enqueue(link)
        page = Page(final, title or final, text, datetime.now(UTC).isoformat(timespec="seconds"))
        await out.put(page)
        stats.pages += 1
        if stats.pages >= max_pages:
            done.set()

    async def worker() -> None:
        while True:
            url = await frontier.get()
            stats.queued = frontier.qsize()
            try:
                await visit(url)
            except Exception:  # one bad page must not kill the crawl (CancelledError passes)
                stats.errors += 1
                log.exception("unexpected error on %s", url)
            finally:
                frontier.task_done()

    try:
        for seed in seeds:
            enqueue(seed)
        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(worker()) for _ in range(concurrency)]
            finished = tg.create_task(frontier.join())  # frontier exhausted
            budget = tg.create_task(done.wait())  # page budget reached
            await asyncio.wait([finished, budget], return_when=asyncio.FIRST_COMPLETED)
            for task in (*workers, finished, budget):
                task.cancel()
    except BaseException:
        # cancelled by the consumer, or a real failure: wake the consumer if it is
        # waiting (it then awaits this task and sees the exception) and get out
        with suppress(asyncio.QueueFull):
            out.put_nowait(None)
        raise
    finally:
        if own_client:
            await client.aclose()
    await out.put(None)  # normal end of the crawl
