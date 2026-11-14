import io
import json
from email.message import Message

from online_resource_adapters import (
    fetch_arxiv_html_sections,
    fetch_web_page,
    search_datasets,
    search_web,
)


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


def test_search_web_prefers_tavily_and_tags_web_commentary(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    seen_urls = []

    def fake_urlopen(request, timeout=10):
        seen_urls.append(request.full_url)
        if "api.tavily.com/search" in request.full_url:
            body = json.dumps({
                "results": [
                    {
                        "title": "FlashAttention architecture intuition",
                        "url": "https://example.com/flashattention-blog",
                        "content": "Explains IO-aware tiling and SRAM usage.",
                    }
                ]
            }).encode("utf-8")
            return _Response(body)
        raise RuntimeError("fallback should not fail test")

    monkeypatch.setattr("online_resource_adapters.urlopen", fake_urlopen)
    results = search_web("FlashAttention", limit=1)
    assert seen_urls == ["https://api.tavily.com/search"]
    assert len(results) == 1
    assert results[0]["provider"] == "tavily"
    assert results[0]["evidence_level"] == "web_commentary"


def test_search_web_uses_searxng_when_configured(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_URL", "https://searx.example.org")

    def fake_urlopen(request, timeout=10):
        if "searx.example.org/search" in request.full_url:
            return _Response(json.dumps({
                "results": [
                    {
                        "title": "Mamba engineering comparison",
                        "url": "https://example.org/mamba-notes",
                        "content": "Compares selective SSM vs Transformer KV cache.",
                    }
                ]
            }).encode("utf-8"))
        raise RuntimeError("unused")

    monkeypatch.setattr("online_resource_adapters.urlopen", fake_urlopen)
    results = search_web("Mamba", limit=1)
    assert len(results) == 1
    assert results[0]["provider"] == "searxng"
    assert results[0]["evidence_level"] == "web_commentary"


def test_search_web_huggingface_papers_and_isolates_ddg_failure(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)

    def fake_urlopen(request, timeout=10):
        if "huggingface.co/api/papers/search" in request.full_url:
            payload = [
                {
                    "title": "Longformer: The Long-Document Transformer",
                    "numComments": 5,
                    "paper": {
                        "id": "2004.05150",
                        "upvotes": 42,
                        "ai_summary": "Combines local windowed attention with global attention.",
                        "ai_keywords": ["sparse attention", "long context"],
                    },
                }
            ]
            return _Response(json.dumps(payload).encode("utf-8"))
        if "html.duckduckgo.com" in request.full_url:
            raise TimeoutError("DuckDuckGo blocked")
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    monkeypatch.setattr("online_resource_adapters.urlopen", fake_urlopen)
    results = search_web("Longformer", limit=3)
    assert len(results) == 1
    assert results[0]["provider"] == "huggingface_papers"
    assert results[0]["evidence_level"] == "web_commentary"
    assert results[0]["arxiv_id"] == "2004.05150"
    assert "upvotes=42" in results[0]["summary"]
    assert "comments=5" in results[0]["summary"]


def test_fetch_arxiv_html_sections_extracts_method_and_experiment_only(monkeypatch):
    class Response(_Response):
        headers = Message()

    Response.headers.add_header("Content-Type", "text/html; charset=utf-8")
    html = (
        "<html><body>"
        "<h2>Abstract</h2><p>Abstract text that should be skipped because it is already available elsewhere.</p>"
        "<h2>2 Methodology</h2><p>" + ("We propose a sparse routing mechanism with O(N) complexity. " * 4) + "</p>"
        "<h2>4 Experiments</h2><p>" + ("On Wikitext-103 our model achieves 18.2 perplexity across benchmarks. " * 4) + "</p>"
        "<h2>References</h2><p>" + ("Author et al. 2023. " * 10) + "</p>"
        "</body></html>"
    ).encode("utf-8")
    monkeypatch.setattr("online_resource_adapters.urlopen", lambda *args, **kwargs: Response(html))
    res = fetch_arxiv_html_sections("arXiv:2401.12345v1")
    assert res["arxiv_id"] == "2401.12345v1"
    assert res["evidence_level"] == "full_text_excerpt"
    assert "2 Methodology" in res["sections"] and "4 Experiments" in res["sections"]
    assert "sparse routing mechanism" in res["text"]
    assert "18.2 perplexity" in res["text"]
    assert "Abstract text" not in res["text"]
    assert "Author et al." not in res["text"]


def test_fetch_arxiv_html_sections_rejects_invalid_id():
    assert fetch_arxiv_html_sections("http://127.0.0.1/private") == {}
    assert fetch_arxiv_html_sections("not-an-arxiv-id") == {}


def test_enrich_top_arxiv_html_upgrades_top_two_cards_only(monkeypatch):
    from cross_source_hybrid_retriever import CrossSourceHybridRetriever, UnifiedEvidenceCard

    retriever = CrossSourceHybridRetriever.__new__(CrossSourceHybridRetriever)
    cards = [
        UnifiedEvidenceCard(doc_name="Paper A", doc_type="arxiv", text="abs A", score=0.95, arxiv_id="2401.00001", evidence_level="abstract"),
        UnifiedEvidenceCard(doc_name="Paper B", doc_type="arxiv", text="abs B", score=0.88, arxiv_id="2401.00002", evidence_level="abstract"),
        UnifiedEvidenceCard(doc_name="Paper C", doc_type="arxiv", text="abs C", score=0.70, arxiv_id="2401.00003", evidence_level="abstract"),
    ]
    monkeypatch.setattr(
        "online_resource_adapters.fetch_arxiv_html_sections",
        lambda aid: {"text": f"Method details for {aid}", "evidence_level": "full_text_excerpt", "content_scope": "arXiv HTML"},
    )
    retriever._enrich_top_arxiv_html(cards, max_papers=2)
    assert cards[0].evidence_level == "full_text_excerpt" and "Method details for 2401.00001" in cards[0].text
    assert cards[1].evidence_level == "full_text_excerpt" and "Method details for 2401.00002" in cards[1].text
    assert cards[2].evidence_level == "abstract" and "Method details" not in cards[2].text


def test_search_web_multidisciplinary_sources_quanta_wiki_epmc(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)

    def fake_urlopen(request, timeout=10):
        url = request.full_url
        if "quantamagazine.org" in url:
            return _Response(json.dumps([
                {
                    "title": {"rendered": "Quantum Theorem &#8217;s Test"},
                    "link": "https://www.quantamagazine.org/quantum-test/",
                    "excerpt": {"rendered": "<p>Physicists test Bell bounds.</p>"},
                    "date": "2026-11-06T10:00:00",
                }
            ]).encode("utf-8"))
        if "en.wikipedia.org" in url:
            return _Response(json.dumps({
                "query": {
                    "search": [
                        {
                            "title": "Quantum foundations",
                            "snippet": "<span class=\"searchmatch\">Quantum</span> foundations studies axioms.",
                        }
                    ]
                }
            }).encode("utf-8"))
        if "ebi.ac.uk/europepmc" in url:
            return _Response(json.dumps({
                "resultList": {
                    "result": [
                        {
                            "title": "Quantum biology review",
                            "pmcid": "PMC123456",
                            "journalTitle": "Nature Reviews",
                            "pubYear": "2025",
                        }
                    ]
                }
            }).encode("utf-8"))
        if "inspirehep.net" in url:
            return _Response(json.dumps({
                "hits": {
                    "hits": [
                        {
                            "id": "812316",
                            "metadata": {
                                "titles": [{"title": "Quantum Gravity Foundations"}],
                                "abstracts": [{"value": "Review of spacetime quantization."}],
                                "citation_count": 42,
                                "arxiv_eprints": [{"value": "0902.0190"}],
                            },
                        }
                    ]
                }
            }).encode("utf-8"))
        if "api.zbmath.org" in url:
            return _Response(json.dumps({
                "result": [
                    {
                        "id": 2192486,
                        "title": {"title": "Mathematical Quantum Gravity"},
                        "year": "2024",
                        "editorial_contributions": [{"text": "Rigorous operator algebra review."}],
                    }
                ]
            }).encode("utf-8"))
        if "huggingface.co" in url:
            return _Response(b"[]")
        return _Response(b"")

    monkeypatch.setattr("online_resource_adapters.urlopen", fake_urlopen)
    results = search_web("quantum mechanics", limit=8)
    providers = {r["provider"] for r in results}
    assert {"quanta_magazine", "wikipedia", "europe_pmc", "inspire_hep", "zbmath"} <= providers
    quanta_item = next(r for r in results if r["provider"] == "quanta_magazine")
    assert quanta_item["evidence_level"] == "web_commentary"
    assert "<p>" not in quanta_item["summary"]


def test_search_web_respects_time_constraint_and_keeps_undated_encyclopedia(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    seen_urls = []

    def fake_urlopen(request, timeout=10):
        url = request.full_url
        seen_urls.append(url)
        if "inspirehep.net" in url:
            return _Response(json.dumps({
                "hits": {
                    "hits": [
                        {
                            "id": "1997001",
                            "metadata": {
                                "titles": [{"title": "Old 1997 Proceedings"}],
                                "earliest_date": "1997-07-09",
                            },
                        },
                        {
                            "id": "2025001",
                            "metadata": {
                                "titles": [{"title": "Recent 2025 Quantum Study"}],
                                "earliest_date": "2025-05-01",
                            },
                        },
                    ]
                }
            }).encode("utf-8"))
        if "en.wikipedia.org" in url:
            return _Response(json.dumps({
                "query": {"search": [{"title": "Quantum mechanics", "snippet": "Core theory."}]}
            }).encode("utf-8"))
        return _Response(b"[]")

    monkeypatch.setattr("online_resource_adapters.urlopen", fake_urlopen)
    results = search_web(
        "quantum mechanics",
        limit=6,
        time_constraint={"start_date": "2024-01-01", "end_date": "2026-11-14"},
    )
    titles = [r["title"] for r in results]
    assert "Old 1997 Proceedings" not in titles
    assert "Recent 2025 Quantum Study" in titles
    assert "Quantum mechanics" in titles
    assert any("inspirehep.net" in u and "sort=mostrecent" in u for u in seen_urls)
    assert any("quantamagazine.org" in u and "after=2024-01-01T00:00:00" in u for u in seen_urls)
