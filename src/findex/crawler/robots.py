"""robots.txt: fetched once per host, cached, consulted before every request."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT, fetch

log = logging.getLogger(__name__)


class RobotsCache:
    """`await robots.allowed(url)` -> may our user agent fetch this URL?

    Rules follow Google's documented behaviour:
      * 2xx  -> obey the file;
      * 4xx  -> no robots.txt: everything is allowed;
      * 5xx or no response -> assume the server is in trouble: nothing is allowed.
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str = USER_AGENT) -> None:
        self._client = client
        self._agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._parsers:
            return self._parsers[origin]
        # One fetch per host even if 20 workers ask at once: the first one fetches while
        # the others wait on the lock (an await sits between the check and the store,
        # so unlike plain asyncio code this *does* need a lock).
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._parsers:
                self._parsers[origin] = await self._load(origin)
        return self._parsers[origin]

    async def _load(self, origin: str) -> RobotFileParser:
        parser = RobotFileParser(origin + "/robots.txt")
        result = await fetch(self._client, origin + "/robots.txt")
        if result.status is not None and 200 <= result.status < 300:
            parser.parse(result.text.splitlines())
        elif result.status is not None and 400 <= result.status < 500:
            parser.parse([])  # no rules: everything allowed
        else:
            parser.parse(["User-agent: *", "Disallow: /"])
        log.info("robots.txt for %s: %s", origin, result.status or result.error)
        return parser

    async def allowed(self, url: str) -> bool:
        return (await self._parser(url)).can_fetch(self._agent, url)

    async def crawl_delay(self, url: str) -> float:
        """The Crawl-delay the site asks for (0 if none)."""
        delay = (await self._parser(url)).crawl_delay(self._agent)
        return float(delay) if delay else 0.0
