import urllib.parse
from unittest.mock import MagicMock

import online_paper_search_agent as opsa
from online_paper_search_agent import (
    AcademicResource,
    OnlinePaperSearchAgent,
    _build_arxiv_fallback_expressions,
    extract_prf_terms_from_papers,
)
from cross_source_hybrid_retriever import CrossSourceHybridRetriever


def test_arxiv_fallback_expressions_relax_overconstrained_and_query():
    phrases = ["sequential recommendation large language model diffusion", "generative recommendation"]
    primary = "all:(sequential AND recommendation AND large AND language AND model AND diffusion) OR all:(generative AND recommendation)"
    fallbacks = _build_arxiv_fallback_expressions(phrases, primary)
    assert fallbacks == [
        "all:(sequential AND recommendation AND large AND language AND model AND diffusion)",
        "all:(sequential AND recommendation)",
    ]


def test_extract_academic_keywords_preserves_ascii_terms_on_llm_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = OnlinePaperSearchAgent()
    agent.llm_client = MagicMock()
    agent.llm_client.chat.completions.create.side_effect = TimeoutError("relay timeout")

    result = agent.extract_academic_keywords("推荐系统中的 LLM 和 Diffusion 最新研究")
    assert result == "LLM Diffusion"


def test_search_arxiv_relaxes_to_core_anchor_when_initial_hits_zero(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    opsa._ARXIV_SCHEDULER.reset_cooldown()
    with opsa._ARXIV_CACHE_LOCK:
        opsa._ARXIV_CACHE.clear()

    empty_feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    hit_feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2501.00001v1</id>
        <published>2025-01-01T00:00:00Z</published>
        <title>Sequential Recommendation with LLM</title>
        <summary>Test abstract</summary>
        <author><name>Alice</name></author>
      </entry>
    </feed>"""

    seen_queries = []

    def fake_execute(req, timeout=8, is_background=False):
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(req.full_url).query)
        q = qs["search_query"][0]
        seen_queries.append(q)
        if len(seen_queries) == 1:
            return empty_feed
        return hit_feed

    monkeypatch.setattr(opsa._ARXIV_SCHEDULER, "execute", fake_execute)
    agent = OnlinePaperSearchAgent()
    papers = agent.search_arxiv("sequential recommendation large language model")

    assert len(papers) == 1
    assert papers[0].paper_id == "2501.00001v1"
    assert len(seen_queries) == 2
    assert "all:(sequential AND recommendation)" in seen_queries[1]


def test_cross_source_retriever_ingests_semanticscholar_and_dblp():
    fake_agent = MagicMock()
    fake_agent.search_multisource.return_value = {
        "arxiv_papers": [],
        "openalex_papers": [],
        "crossref_papers": [],
        "semanticscholar_papers": [
            {
                "id": "s2-1",
                "title": "S2 Recommender Paper",
                "authors": ["Bob"],
                "year": 2025,
                "source": "semanticscholar",
                "summary": "Abstract body",
                "url": "https://www.semanticscholar.org/paper/s2-1",
                "evidence_level": "abstract",
            }
        ],
        "dblp_papers": [
            {
                "id": "dblp-1",
                "title": "DBLP RecSys Paper",
                "authors": ["Carol"],
                "year": 2024,
                "source": "dblp",
                "summary": "DBLP venue",
                "url": "https://dblp.org/rec/1",
                "evidence_level": "bibliographic_metadata",
            }
        ],
        "github_repos": [],
        "web_results": [],
        "datasets": [],
        "source_errors": {},
    }

    retriever = CrossSourceHybridRetriever(rag_engine=None, paper_agent=fake_agent)
    cards = retriever._fetch_online_evidence("recommender systems")
    doc_types = {c.doc_type for c in cards}
    assert "semanticscholar" in doc_types
    assert "dblp" in doc_types
    assert retriever.last_source_counts["semanticscholar_papers"] == 1
    assert retriever.last_source_counts["dblp_papers"] == 1


def test_extract_prf_terms_from_papers_learns_real_subfield_terms():
    records = [
        {
            "title": "Recommender Systems with Generative Retrieval",
            "summary": "We propose RQ-VAE and Semantic ID representations for Generative Retrieval.",
        },
        {
            "title": "Semantic ID Scaling for Generative Retrieval",
            "summary": "Experiments show RQ-VAE improves cold-start generalization.",
        },
    ]
    terms = extract_prf_terms_from_papers("recommendation systems", records, max_terms=3)
    assert "Generative Retrieval" in terms
    assert "Semantic ID" in terms or "RQ-VAE" in terms


def test_search_citation_snowball_fetches_citations_and_references(monkeypatch):
    import json
    import io

    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = OnlinePaperSearchAgent()

    def fake_urlopen(req, timeout=6):
        url = req.full_url
        if "/citations" in url:
            payload = {
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "cite-1",
                            "title": "Follow-up Generative RecSys",
                            "year": 2025,
                            "abstract": "Builds on the seed paper.",
                            "citationCount": 12,
                            "venue": "SIGIR",
                            "authors": [{"name": "Dan"}],
                            "externalIds": {"DOI": "10.1145/1234"},
                        }
                    }
                ]
            }
        else:
            payload = {
                "data": [
                    {
                        "citedPaper": {
                            "paperId": "ref-1",
                            "title": "Foundational Sequence Model",
                            "year": 2020,
                            "abstract": "Classic baseline.",
                            "citationCount": 500,
                            "venue": "NeurIPS",
                            "authors": [{"name": "Eve"}],
                            "externalIds": {},
                        }
                    }
                ]
            }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: False
        return resp

    monkeypatch.setattr(opsa.urllib.request, "urlopen", fake_urlopen)
    res = agent.search_citation_snowball(
        [{"title": "Seed Paper", "arxiv_id": "2401.00001v1"}],
        max_per_seed=2,
        time_constraint={"has_time_constraint": True, "start_year": 2024, "end_year": 2026, "basis": "explicit_user_constraint"},
    )
    titles = {r.title for r in res}
    assert "Follow-up Generative RecSys" in titles
    assert "Foundational Sequence Model" in titles
    assert all(r.source == "citation_snowball" for r in res)


def test_hybrid_search_triggers_prf_and_snowball_when_initial_recall_is_thin(monkeypatch):
    monkeypatch.setenv("ENABLE_CITATION_SNOWBALL", "true")
    fake_agent = MagicMock()
    fake_agent.arxiv_cooldown_remaining.return_value = 0.0
    fake_agent.search_multisource.return_value = {
        "arxiv_papers": [
            {
                "paper_id": "2401.00001",
                "title": "Seed Paper on Generative Retrieval",
                "authors": ["Alice"],
                "year": 2024,
                "summary": "We introduce Semantic ID and RQ-VAE for recommendation.",
                "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
            }
        ],
        "openalex_papers": [],
        "crossref_papers": [],
        "semanticscholar_papers": [],
        "dblp_papers": [],
        "github_repos": [],
        "web_results": [],
        "datasets": [],
        "source_errors": {},
    }
    fake_agent.search_semanticscholar.return_value = [
        AcademicResource(
            id="prf-s2-1",
            title="PRF Discovered Semantic ID Paper",
            authors=["Bob"],
            year=2025,
            source="semanticscholar",
            summary="Uses Semantic ID.",
            url="https://example.org/prf",
            evidence_level="abstract",
        )
    ]
    fake_agent.search_arxiv.return_value = []
    fake_agent.search_citation_snowball.return_value = [
        AcademicResource(
            id="sb-1",
            title="Snowballed Citing Paper",
            authors=["Carol"],
            year=2025,
            source="citation_snowball",
            summary="Cites Seed Paper.",
            url="https://example.org/sb",
            evidence_level="abstract",
        )
    ]

    retriever = CrossSourceHybridRetriever(rag_engine=None, paper_agent=fake_agent)
    results = retriever.hybrid_search_and_deduplicate(
        query="推荐系统中的大模型检索",
        top_k=6,
        include_local=False,
        include_online=True,
    )
    titles = {r["doc_name"] for r in results}
    assert "Seed Paper on Generative Retrieval" in titles
    assert "PRF Discovered Semantic ID Paper" in titles
    assert "Snowballed Citing Paper" in titles
    fake_agent.search_semanticscholar.assert_called_once()
    fake_agent.search_citation_snowball.assert_called_once()
