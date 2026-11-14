import io
import json
from email.message import Message

from online_resource_adapters import fetch_web_page, search_datasets, search_web


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_general_web_adapter_marks_results_as_unverified_snippets(monkeypatch):
    html = b'''<div class="result">
      <a class="result__a" href="https://example.org/page">Technical standard overview</a>
      <a class="result__snippet">Summary visible in search results</a>
    </div>'''
    monkeypatch.setattr("online_resource_adapters.urlopen", lambda *args, **kwargs: _Response(html))
    monkeypatch.setenv("WEB_PAGE_FETCH_LIMIT", "0")
    results = search_web("technical standard", limit=4)
    assert results[0]["title"] == "Technical standard overview"
    assert results[0]["evidence_level"] == "search_snippet"
    assert results[0]["url"] == "https://example.org/page"


def test_dataset_adapter_returns_repository_metadata_only(monkeypatch):
    payload = [{"id": "org/sample-data", "downloads": 20, "cardData": {"description": "A sample dataset"}}]
    monkeypatch.setattr(
        "online_resource_adapters.urlopen",
        lambda *args, **kwargs: _Response(json.dumps(payload).encode()),
    )
    result = search_datasets("sample data")[0]
    assert result["title"] == "org/sample-data"
    assert result["evidence_level"] == "repository_metadata"
    assert "datasets/org/sample-data" in result["url"]


def test_public_page_fetch_marks_inspected_text_and_rejects_local_addresses(monkeypatch):
    class Response(_Response):
        headers = Message()
        def geturl(self):
            return "https://example.org/spec"

    Response.headers.add_header("Content-Type", "text/html; charset=utf-8")
    html = b"<html><title>Spec title</title><body><nav>navigation</nav><main><h1>Requirement</h1><p>Temperature range is specified here.</p></main><script>ignored()</script></body></html>"
    monkeypatch.setattr("online_resource_adapters.urlopen", lambda *args, **kwargs: Response(html))
    result = fetch_web_page("https://example.org/spec")
    assert result["title"] == "Spec title"
    assert "Temperature range" in result["summary"]
    assert "navigation" not in result["summary"]
    assert result["evidence_level"] == "page_text"
    try:
        fetch_web_page("http://127.0.0.1/private")
    except ValueError:
        pass
    else:
        raise AssertionError("local/private URLs must not be fetched")
