"""Lab 7: every route through TestClient, with the tiny fixture index injected."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex.index import Index
from findex.web.app import JsonFormatter, create_app, index_state
from findex.web.settings import Settings


@pytest.fixture
def app(tiny_index: Index, tmp_path: Path) -> FastAPI:
    application = create_app(Settings(index_path=tmp_path / "missing.pkl", log_json=False))
    application.dependency_overrides[index_state] = lambda: tiny_index
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:  # `with` runs the lifespan
        yield test_client


def test_search_ok(client: TestClient) -> None:
    r = client.get("/search", params={"q": "whale", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1 and body["pages"] == 1 and body["scorer"] == "bm25"
    [hit] = body["results"]
    assert hit["title"] == "Whale Book" and "«whale»" in hit["snippet"]
    assert r.headers["x-request-id"]


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "the", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "the", "k": 1, "page": 2}).json()
    assert first["total"] == second["total"] == 3 and first["pages"] == 3
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    beyond = client.get("/search", params={"q": "the", "k": 1, "page": 9}).json()
    assert beyond["results"] == []


def test_phrase_and_tfidf(client: TestClient) -> None:
    body = client.get("/search", params={"q": '"white whale"', "scorer": "tfidf"}).json()
    assert [h["title"] for h in body["results"]] == ["Whale Book"]


@pytest.mark.parametrize(
    "params",
    [
        {"q": "whale", "k": "abc"},
        {"q": "whale", "k": 0},
        {"q": "whale", "k": 101},
        {"q": ""},
        {"q": "x" * 201},
        {"q": "whale", "scorer": "pagerank"},
        {"q": "whale", "page": 0},
        {},
    ],
)
def test_search_validation_422(client: TestClient, params: dict[str, str | int]) -> None:
    r = client.get("/search", params=params)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_query_syntax_error_is_422(client: TestClient) -> None:
    r = client.get("/search", params={"q": "(whale"})
    assert r.status_code == 422 and "query error" in r.json()["detail"]


def test_document_and_404(client: TestClient) -> None:
    r = client.get("/docs/0")
    assert r.status_code == 200 and r.json()["text"].startswith("Call me Ishmael")
    missing = client.get("/docs/999")
    assert missing.status_code == 404 and missing.json()["detail"] == "no document with doc_id 999"
    assert client.get("/docs/abc").status_code == 422


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == 4 and body["positions"] is True and body["terms"] > 10


def test_health_ok_and_503_when_unloaded(app: FastAPI, client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "documents": 4}
    app.dependency_overrides[index_state] = lambda: None
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "unavailable"
    assert client.get("/search", params={"q": "whale"}).status_code == 503


def test_real_lifespan_loads_index(index_file: Path) -> None:
    application = create_app(Settings(index_path=index_file, log_json=False))
    with TestClient(application) as c:
        for _ in range(200):  # the index loads in a background thread
            if c.get("/health").status_code == 200:
                break
        assert c.get("/search", params={"q": "tarzan"}).json()["total"] == 1


def test_missing_index_file_keeps_serving_503(tmp_path: Path) -> None:
    application = create_app(Settings(index_path=tmp_path / "nope.pkl", log_json=False))
    with TestClient(application) as c:
        assert c.get("/health").status_code == 503


def test_unexpected_error_is_clean_500(app: FastAPI, caplog: pytest.LogCaptureFixture) -> None:
    def broken() -> Index:
        raise RuntimeError("boom")

    app.dependency_overrides[index_state] = broken
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/stats", headers={"X-Request-ID": "req-42"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error", "request_id": "req-42"}


def test_openapi_lists_api_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/docs/{doc_id}", "/stats", "/health"} <= set(paths)
    assert "/search-async" not in paths
    assert client.get("/docs").status_code == 200  # Swagger UI


def test_ui_pages(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200 and "<form" in home.text
    page = client.get("/", params={"q": "white whale"})
    assert "<mark>whale</mark>" in page.text and "Whale Book" in page.text
    fragment = client.get("/ui/results", params={"q": "<script>"})
    assert "<script>" not in fragment.text  # escaped
    doc = client.get("/doc/0")
    assert doc.status_code == 200 and "Call me Ishmael" in doc.text
    assert client.get("/static/style.css").status_code == 200
    assert "error" in client.get("/ui/results", params={"q": "(broken"}).text


def test_async_route_gives_same_answer(client: TestClient) -> None:
    a = client.get("/search", params={"q": "ship"}).json()
    b = client.get("/search-async", params={"q": "ship"}).json()
    assert a["results"] == b["results"]


def test_json_log_format() -> None:
    record = logging.LogRecord("findex.web", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    record.request_id = "abc"
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "hello x" and entry["request_id"] == "abc"


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_PATH", "/srv/ix.pkl")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.index_path == Path("/srv/ix.pkl") and settings.log_level == "DEBUG"
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError, match="log_level"):
        Settings()
