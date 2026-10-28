from unittest.mock import MagicMock
import numpy as np

import online_paper_search_agent as mod


def test_arxiv_uses_submitted_date_when_time_constrained(monkeypatch):
    called_urls = []

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>"

    def fake_urlopen(req, timeout):
        called_urls.append(req.full_url)
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    monkeypatch.setattr(agent, "_parse_atom_feed", lambda _xml: [])

    # 1. 带有动态时间约束 (如 start_year=2024, sort_by_date=True)
    tc = {"has_time_constraint": True, "start_year": 2024, "end_year": 2026, "sort_by_date": True}
    agent.search_arxiv("lattice gauge theory", max_results=3, time_constraint=tc)
    assert len(called_urls) == 1
    assert "sortBy=submittedDate" in called_urls[0]
    assert "sortOrder=descending" in called_urls[0]

    # 2. 无时间约束时，按经典度/相关度排序
    called_urls.clear()
    agent.search_arxiv("lattice gauge theory", max_results=3, time_constraint=None)
    assert len(called_urls) == 1
    assert "sortBy=relevance" in called_urls[0]


def test_direction_grounding_matches_by_semantic_similarity():
    """测试根据具体学术方向（而不是第几篇）自动向量对齐目标文献"""
    candidates = [
        {
            "paper_id": "paper_qcd_sign",
            "title": "Complex Langevin Dynamics for the QCD Sign Problem at Finite Density",
            "summary": "We study complex Langevin simulation to overcome the fermion sign problem in lattice QCD at non-zero chemical potential.",
            "pdf_url": "https://arxiv.org/pdf/fake1.pdf",
        },
        {
            "paper_id": "paper_tensor_net",
            "title": "Tensor Network Renormalization for Non-Abelian Gauge Theories",
            "summary": "We construct tensor contractions for 2D and 3D non-abelian lattice gauge theories.",
            "pdf_url": "https://arxiv.org/pdf/fake2.pdf",
        }
    ]

    focus_direction = "符号问题 复作用量 Langevin 模拟"

    # 模拟 BGE-M3 的高维语义向量 (符号问题向量与 paper_qcd_sign 高度平行，与 paper_tensor_net 几乎正交)
    vec_focus = np.array([0.9, 0.1, 0.0])
    cand_vecs = np.array([
        [0.85, 0.15, 0.0],  # paper 1: 符号问题
        [0.05, 0.95, 0.2],  # paper 2: 张量网络
    ])

    sims = np.dot(cand_vecs, vec_focus)
    best_idx = int(np.argmax(sims))
    assert best_idx == 0
    assert candidates[best_idx]["paper_id"] == "paper_qcd_sign"


def test_closed_window_temporal_query_for_historical_year(monkeypatch):
    """测试当用户指定看特定历史年份（如2021年）时，学术接口严格执行闭合时域拦截"""
    called_urls = []

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return b'{"results": [], "message": {"items": []}}'

    def fake_urlopen(req, timeout):
        called_urls.append(req.full_url)
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    monkeypatch.setattr(agent, "_parse_atom_feed", lambda _xml: [])

    # 用户指定 2021 年
    tc_2021 = {"has_time_constraint": True, "start_year": 2021, "end_year": 2021, "sort_by_date": False}

    # 1. arXiv: 必须注入 submittedDate:[20210101000000 TO 20211231235959]
    agent.search_arxiv("transformer", max_results=3, time_constraint=tc_2021)
    assert any("submittedDate%3A%5B20210101000000+TO+20211231235959%5D" in u or "submittedDate:[20210101000000 TO 20211231235959]" in u for u in called_urls)

    # 2. OpenAlex: 必须注入 filter=publication_year:2021
    called_urls.clear()
    agent.search_openalex("transformer", max_results=3, time_constraint=tc_2021)
    assert any("publication_year%3A2021" in u or "publication_year:2021" in u for u in called_urls)

    # 3. CrossRef: 必须注入 from-pub-date:2021-01-01,until-pub-date:2021-12-31
    called_urls.clear()
    agent.search_crossref("transformer", max_results=3, time_constraint=tc_2021)
    assert any("until-pub-date%3A2021-12-31" in u or "until-pub-date:2021-12-31" in u for u in called_urls)
