"""One polite HTTP GET: timeout, bounded retries with backoff + jitter, Retry-After.

Never raises for network trouble: the outcome (status or error) is a FetchResult, so a
crawl keeps going and every URL ends up in the crawl log.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

log = logging.getLogger(__name__)

USER_AGENT = (
    "findex-crawler/0.6 (student search-engine project; +https://github.com/daideneg28-prog/findex)"
)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 60.0  # never let a server park us for longer than this
MAX_BYTES = 5 * 2**20  # pages bigger than this are not documents we want

type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str  # what we asked for
    final_url: str  # after redirects
    status: int | None  # None: no HTTP response at all (timeout, DNS, reset, ...)
    content_type: str
    text: str
    size: int
    elapsed: float  # seconds, all attempts and back-off included
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    @property
    def is_html(self) -> bool:
        return self.ok and self.content_type.startswith(("text/html", "application/xhtml"))


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    """Exponential back-off with jitter: min(cap, base * 2**attempt) + U(0, 1)."""
    return min(cap, base * 2**attempt) + random.uniform(0, 1)


def retry_after(response: httpx.Response) -> float | None:
    """Seconds to wait according to a Retry-After header (delta-seconds or HTTP date)."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return min(float(value), MAX_RETRY_AFTER)
    try:
        when = parsedate_to_datetime(value)
    except TypeError, ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return min(max(0.0, (when - datetime.now(UTC)).total_seconds()), MAX_RETRY_AFTER)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    attempts: int = 3,
    sleep: Sleep = asyncio.sleep,
) -> FetchResult:
    """GET `url` with at most `attempts` tries.

    Retried: 429 and 5xx (honouring Retry-After), timeouts and transport errors.
    Not retried: any other status (a 404 will still be a 404 in a second), redirect loops.
    """
    t0 = time.perf_counter()
    error: str | None = None
    response: httpx.Response | None = None
    attempt = 0
    for attempt in range(1, attempts + 1):
        response, error = None, None
        try:
            async with asyncio.timeout(timeout):
                response = await client.get(url, follow_redirects=True)
        except TimeoutError:
            error = f"timeout after {timeout:g} s"
        except httpx.TooManyRedirects:
            error = "too many redirects"
            break
        except httpx.TransportError as exc:
            error = f"{type(exc).__name__}: {exc}"
        if response is not None and response.status_code not in RETRY_STATUSES:
            break
        if attempt == attempts:
            break
        wait = (retry_after(response) if response is not None else None) or backoff_delay(attempt)
        log.info("%s: %s, retry %d in %.1f s", url, error or response, attempt, wait)
        await sleep(wait)

    elapsed = time.perf_counter() - t0
    if response is None:
        return FetchResult(url, url, None, "", "", 0, elapsed, attempt, error)
    content_type = response.headers.get("content-type", "").lower()
    body = response.content
    text = ""
    if len(body) > MAX_BYTES:
        error = f"body too large ({len(body)} bytes)"
    elif content_type.startswith(("text/", "application/xhtml")):
        text = response.text
    return FetchResult(
        url,
        str(response.url),
        response.status_code,
        content_type,
        text,
        len(body),
        elapsed,
        attempt,
        error,
    )
