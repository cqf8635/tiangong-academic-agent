import pytest
from unittest.mock import MagicMock
from master_agent_workflow import (
    MasterAgentState,
    intent_routing_node,
    online_paper_search_node,
    MasterEngineRegistry,
)


def test_main_task_retrieval_shared_with_subtask_and_no_duplicate_search(monkeypatch):
    """测试：主任务检索结果直接注入共享池，子任务消费共享池且不再发起重复多源检索"""
    mock_paper_agent = MagicMock()
    mock_search_multisource = MagicMock()
    mock_paper_agent.search_multisource = mock_search_multisource

    # 模拟主任务检索返回候选（格点规范理论 + 量子模拟双领域）
    mock_hybrid = MagicMock()
    mock_hybrid.hybrid_search_and_deduplicate.return_value = [
        {
            "doc_name": "Lattice Gauge Theory on Quantum Computers",
            "doc_type": "arxiv",
            "location_label": "arXiv:2501.12345 (2025年)",
            "score": 0.95,
            "text": "Lattice gauge simulation with error mitigation.",
            "summary": "We study Hamiltonian lattice gauge simulation.",
            "arxiv_id": "2501.12345",
            "pdf_url": "https://arxiv.org/pdf/2501.12345.pdf",
            "authors": ["Alice Quantum", "Bob Lattice"],
            "page_or_slide": 2025,
        },
        {
            "doc_name": "Quantum Simulation of Non-Abelian Gauge Theory",
            "doc_type": "crossref",
            "location_label": "DOI: 10.1103/PhysRevD.110.014501",
            "score": 0.92,
            "text": "Non-Abelian gauge theory quantum simulation.",
            "summary": "Published in Phys. Rev. D 110.",
            "doi": "10.1103/PhysRevD.110.014501",
            "pdf_url": "https://doi.org/10.1103/PhysRevD.110.014501",
            "authors": ["Carol Field"],
            "page_or_slide": 2024,
        }
    ]

    mock_intent_agent = MagicMock()
    mock_decision = MagicMock()
    mock_decision.model_dump.return_value = {
        "intent": "arxiv_paper_search",
        "target_agent": "OnlinePaperSearchAgent",
        "domain": "量子场论与量子模拟",
        "need_online_paper_search": True,
        "is_cross_disciplinary": True,
        "sub_queries": [
            {"discipline": "格点规范理论", "query": "格点规范理论非微扰方法"},
            {"discipline": "量子模拟", "query": "量子计算机格点规范模拟"}
        ],
        "search_query": "非微扰格点规范理论与量子模拟",
        "online_search_query": "lattice gauge theory quantum simulation",
        "time_constraint": {"has_time_constraint": True, "start_year": 2024, "end_year": 2026},
        "negative_constraints": [],
        "requires_long_form": False,
        "long_form_type": "concise_analysis",
    }
    mock_decision.intent = "arxiv_paper_search"
    mock_decision.target_agent = "OnlinePaperSearchAgent"
    mock_decision.domain = "量子场论与量子模拟"
    mock_decision.need_online_paper_search = True
    mock_decision.is_cross_disciplinary = True
    mock_decision.sub_queries = [
        {"discipline": "格点规范理论", "query": "格点规范理论非微扰方法"},
        {"discipline": "量子模拟", "query": "量子计算机格点规范模拟"}
    ]
    mock_decision.search_query = "非微扰格点规范理论与量子模拟"
    mock_decision.online_search_query = "lattice gauge theory quantum simulation"
    mock_decision.time_constraint = {"has_time_constraint": True, "start_year": 2024, "end_year": 2026}
    mock_decision.negative_constraints = []
    mock_decision.target_topic = ""
    mock_decision.subject_action = "INIT"
    mock_decision.subject_entity = "量子场论与量子模拟"
    mock_decision.requires_long_form = False
    mock_decision.long_form_type = "concise_analysis"
    mock_intent_agent.parse_intent.return_value = mock_decision

    mock_rag = MagicMock()
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: mock_hybrid)
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", lambda: mock_paper_agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_intent_agent", lambda: mock_intent_agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_rag_engine", lambda: mock_rag)

    # 1. 运行主任务路由节点
    initial_state: MasterAgentState = {
        "user_query": "请检索近两年非微扰格点规范理论与量子模拟代表性论文",
        "active_subject": None,
        "sub_subject": None,
        "subject_evolution_log": [],
        "topic_state": None,
        "history_summary": "",
        "active_doc": None,
        "recent_paper_candidates": [],
        "retrieved_evidence": [],
        "intent_decision": None,
        "target_agent": "",
        "final_answer": "",
        "status": "init",
    }

    router_output = intent_routing_node(initial_state)

    # 验证主任务已经完成检索并将结果沉淀入共享池
    cands = router_output.get("recent_paper_candidates", [])
    evs = router_output.get("retrieved_evidence", [])
    assert len(cands) == 2
    assert len(evs) == 2
    assert cands[0]["title"] == "Lattice Gauge Theory on Quantum Computers"
    assert cands[0]["source"] == "arxiv"
    assert cands[1]["title"] == "Quantum Simulation of Non-Abelian Gauge Theory"
    assert cands[1]["source"] == "crossref"

    # 2. 状态透传给子任务 online_paper_search_node
    subtask_state = dict(initial_state)
    subtask_state.update(router_output)

    mock_paper_agent.generate_search_brief.return_value = "学术综述简报正文..."

    subtask_output = online_paper_search_node(subtask_state)

    # 核心验证：子任务绝不重复调用 search_multisource！
    assert mock_search_multisource.call_count == 0, "子任务重复调用了 search_multisource，未能共享主任务成果！"

    # 验证子任务正确消费了共享候选池
    assert subtask_output["final_answer"] == "学术综述简报正文..."
    assert len(subtask_output["recent_paper_candidates"]) == 2
    assert len(subtask_output["retrieved_evidence"]) == 2

    # 验证 generate_search_brief 被传入了从共享候选重构的 paper 对象
    brief_call = mock_paper_agent.generate_search_brief.call_args
    assert len(brief_call.kwargs["papers"]) == 1
    assert brief_call.kwargs["papers"][0].paper_id == "2501.12345"
    assert len(brief_call.kwargs["crossref_papers"]) == 1
    assert brief_call.kwargs["crossref_papers"][0].title == "Quantum Simulation of Non-Abelian Gauge Theory"
