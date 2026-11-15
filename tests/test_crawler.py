"""Lab 6: the crawler, tested against httpx.MockTransport (no network in tests)."""

import asyncio
import io
import itertools
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler import CrawlStats, Page, RobotsCache, canonicalize, crawl, fetch
from findex.crawler.crawl import HostLimiter, extract, seen_key
from findex.crawler.fetch import backoff_delay, retry_after
from findex.index import build_index

type Handler = Callable[[httpx.Request], httpx.Response]


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def no_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


# --- fetch -------------------------------------------------------------------------------


def test_fetch_ok() -> None:
    async def main() -> None:
        async with client_for(lambda r: httpx.Response(200, html="<p>hi</p>")) as client:
            result = await fetch(client, "https://a.test/")
        assert result.ok and result.is_html and result.attempts == 1 and result.text == "<p>hi</p>"

    asyncio.run(main(), debug=True)


def test_fetch_retries_5xx_then_succeeds() -> None:
    calls = Counter[str]()
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="ok")

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    async def main() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "https://a.test/", sleep=record_sleep)
        assert result.status == 200 and result.attempts == 3
        assert (
            len(waits) == 2 and waits[1] >= 1.0
        )  # backoff grows: 0.5*2^1 + jitter, 0.5*2^2 + jitter

    asyncio.run(main(), debug=True)


def test_fetch_gives_up_after_three_attempts() -> None:
    async def main() -> None:
        async with client_for(lambda r: httpx.Response(500)) as client:
            result = await fetch(client, "https://a.test/", sleep=no_sleep)
        assert result.status == 500 and result.attempts == 3 and not result.ok

    asyncio.run(main(), debug=True)


def test_fetch_honours_retry_after() -> None:
    waits: list[float] = []
    responses = iter([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])

    async def record_sleep(seconds: float) -> None:
        waits.append(seconds)

    async def main() -> None:
        async with client_for(lambda r: next(responses)) as client:
            result = await fetch(client, "https://a.test/", sleep=record_sleep)
        assert result.status == 200 and waits == [7.0]

    asyncio.run(main(), debug=True)


def test_fetch_does_not_retry_404() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(404)

    async def main() -> None:
        async with client_for(handler) as client:
            result = await fetch(client, "https://a.test/missing", sleep=no_sleep)
        assert result.status == 404 and result.attempts == 1 and len(calls) == 1

    asyncio.run(main(), debug=True)


def test_fetch_timeout_and_transport_errors_are_results_not_exceptions() -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async def slow_transport(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    async def main() -> None:
        async with client_for(broken) as client:
            result = await fetch(client, "https://a.test/", sleep=no_sleep)
        assert (
            result.status is None
            and result.attempts == 3
            and "ConnectError" in (result.error or "")
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow_transport)) as client:
            result = await fetch(
                client, "https://a.test/", timeout=0.05, attempts=2, sleep=no_sleep
            )
        assert result.status is None and result.error == "timeout after 0.05 s"

    asyncio.run(main())


def test_redirect_loop_is_not_retried() -> None:
    def loop(request: httpx.Request) -> httpx.Response:
        target = "/b" if request.url.path == "/a" else "/a"
        return httpx.Response(302, headers={"Location": target})

    async def main() -> None:
        async with client_for(loop) as client:
            result = await fetch(client, "https://a.test/a", sleep=no_sleep)
        assert result.error == "too many redirects" and result.attempts == 1

    asyncio.run(main(), debug=True)


def test_backoff_and_retry_after_parsing() -> None:
    assert 0.5 * 2 <= backoff_delay(1) <= 0.5 * 2 + 1
    assert 10 <= backoff_delay(20) <= 11  # capped
    assert retry_after(httpx.Response(429, headers={"Retry-After": "3600"})) == 60.0
    date = "Wed, 21 Oct 2015 07:28:00 GMT"  # in the past
    assert retry_after(httpx.Response(429, headers={"Retry-After": date})) == 0.0
    assert retry_after(httpx.Response(429, headers={"Retry-After": "soon"})) is None
    assert retry_after(httpx.Response(429)) is None


# --- robots.txt ------------------------------------------------------------------------


def test_robots_rules_and_single_fetch() -> None:
    fetched: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        if request.url.host == "a.test":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\nCrawl-delay: 2\n")
        if request.url.host == "b.test":
            return httpx.Response(404)
        return httpx.Response(503)

    async def main() -> None:
        async with client_for(handler) as client:
            robots = RobotsCache(client)
            answers = await asyncio.gather(
                *(robots.allowed("https://a.test/public") for _ in range(10))
            )
            assert all(answers)
            assert not await robots.allowed("https://a.test/private/x")
            assert await robots.crawl_delay("https://a.test/") == 2.0
            assert await robots.allowed("https://b.test/anything")  # 404: allow all
            assert not await robots.allowed("https://c.test/")  # 5xx: allow nothing

    asyncio.run(main(), debug=True)
    assert fetched.count("https://a.test/robots.txt") == 1  # 11 questions, one fetch


# --- URLs and HTML ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("HTTPS://Example.COM", "https://example.com/"),
        ("https://example.com:443/a#frag", "https://example.com/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("https://example.com/s?b=2&a=1", "https://example.com/s?a=1&b=2"),
        ("mailto:me@example.com", None),
        ("javascript:void(0)", None),
        ("https://[::1", None),
    ],
)
def test_canonicalize(url: str, expected: str | None) -> None:
    assert canonicalize(url) == expected


def test_canonicalize_relative_links_and_seen_key() -> None:
    assert canonicalize("../b.html", "https://x.test/a/c/") == "https://x.test/a/b.html"
    assert seen_key("https://x.test/lib/") == seen_key("https://x.test/lib")
    assert seen_key("https://x.test/") == "https://x.test/"


def test_extract_title_text_links() -> None:
    html = (
        "<html><head><title> My  Page </title><style>p{}</style></head><body>"
        "<nav><a href='/menu'>menu</a></nav><p>Hello <b>world</b></p>"
        "<script>var x = 1;</script><a href='/next'>next</a><a>no href</a></body></html>"
    )
    title, text, links = extract(html)
    assert title == "My Page"
    assert text == "Hello world next no href"  # no script/style/nav text
    assert links == ["/menu", "/next"]


def test_svg_title_is_not_the_page_title() -> None:
    # regression: peps.python.org has <svg><title>Following system colour scheme</title>
    html = "<title>Typing PEPs</title><svg><title>colour scheme</title></svg><p>x</p>"
    assert extract(html)[0] == "Typing PEPs"


def test_snippet_text_of_a_jsonl_document(tmp_path: Path) -> None:
    from findex.corpus import read_document

    path = tmp_path / "c.jsonl"
    rows = [{"id": "a", "text": "first page"}, {"id": "b", "text": "second page"}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert read_document(path, "b") == "second page"
    assert read_document(path, "zzz") == ""


def test_host_limiter_spaces_out_requests() -> None:
    async def main() -> list[float]:
        limiter = HostLimiter(per_host=5, min_delay=0.05)
        loop = asyncio.get_running_loop()
        starts: list[float] = []

        async def one() -> None:
            async with limiter.slot("h"):
                starts.append(loop.time())

        await asyncio.gather(*(one() for _ in range(4)))
        return starts

    starts = sorted(asyncio.run(main()))
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert all(g >= 0.04 for g in gaps)


# --- the whole crawl ------------------------------------------------------------------------


def site(n_pages: int, *, flaky: bool = False) -> tuple[Handler, Counter[str]]:
    """A fake site: /p0 .. /pN, each linking to the next two pages and to an outside host."""
    hits = Counter[str]()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        hits[path] += 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /secret\n")
        if path == "/secret":
            raise AssertionError("robots.txt was ignored")
        if flaky and path == "/p1" and hits[path] == 1:
            return httpx.Response(503)
        if path == "/missing":
            return httpx.Response(404)
        if path.startswith("/p"):
            i = int(path[2:])
            links = "".join(f'<a href="/p{j}">p{j}</a>' for j in (i + 1, i + 2) if j < n_pages)
            links += '<a href="/secret">s</a><a href="https://elsewhere.test/">x</a>'
            links += f'<a href="/p{i}#top">self</a><a href="/missing">m</a>'
            return httpx.Response(
                200, html=f"<title>Page {i}</title><p>text of page {i}</p>{links}"
            )
        return httpx.Response(404)

    return handler, hits


def collect(
    handler: Handler, *, max_pages: int, concurrency: int
) -> tuple[list[Page], CrawlStats, str]:
    log = io.StringIO()
    stats = CrawlStats()

    async def main() -> list[Page]:
        async with client_for(handler) as client:
            return [
                p
                async for p in crawl(
                    ["https://site.test/p0"],
                    client=client,
                    stats=stats,
                    log_file=log,
                    min_delay=0,
                    max_pages=max_pages,
                    concurrency=concurrency,
                )
            ]

    return asyncio.run(main(), debug=True), stats, log.getvalue()


def test_crawl_visits_each_page_once_and_obeys_robots() -> None:
    handler, hits = site(8, flaky=True)
    pages, stats, log = collect(handler, max_pages=50, concurrency=4)
    assert sorted(p.url for p in pages) == sorted(f"https://site.test/p{i}" for i in range(8))
    assert hits["/p1"] == 2  # the 503 was retried once
    assert all(hits[f"/p{i}"] == 1 for i in (0, 2, 3, 4, 5, 6, 7))  # no duplicates
    assert hits["/secret"] == 0 and stats.robots_blocked >= 1
    assert "elsewhere" not in log  # outside the allowlist: never requested
    assert "\t404\t" in log and stats.errors == 1  # /missing logged as an error
    assert stats.pages == 8 and stats.retries == 1
    assert pages[0].title == "Page 0" and "text of page 0" in pages[0].text


def test_crawl_stops_at_max_pages() -> None:
    handler, hits = site(100)
    pages, stats, _ = collect(handler, max_pages=5, concurrency=3)
    assert len(pages) == 5 and stats.pages == 5
    fetched_pages = sum(n for path, n in hits.items() if path.startswith("/p"))
    assert fetched_pages == 5  # the budget is claimed before fetching: no wasted requests


def test_consumer_can_stop_early() -> None:
    async def main() -> int:
        async with client_for(site(100)[0]) as client:
            count = 0
            async for _ in crawl(["https://site.test/p0"], client=client, min_delay=0):
                count += 1
                if count == 3:
                    break
            return count

    assert asyncio.run(main(), debug=True) == 3


def test_crawled_jsonl_is_indexable(tmp_path: Path) -> None:
    pages, _, _ = collect(site(4)[0], max_pages=10, concurrency=2)
    out = tmp_path / "crawl.jsonl"
    out.write_text("\n".join(json.dumps(p.to_json()) for p in pages), encoding="utf-8")
    index = build_index(iter_documents(tmp_path))
    assert index.num_docs == 4
    assert {m.title for m in index.doc_meta.values()} == {f"Page {i}" for i in range(4)}
    assert index.df("text") == 4
