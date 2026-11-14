import pytest
from unittest.mock import MagicMock
from master_agent_workflow import (
    MasterAgentState,
    intent_routing_node,
    online_paper_search_node,
    MasterEngineRegistry,
)
from retrieval_planner import RetrievalCoverage, RetrievalPlan
from intent_recognition_agent import IntentDecision


@pytest.fixture
def coverage_router(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_GAP_ROUNDS", raising=False)
    decision = IntentDecision(domain="推荐系统", intent="比较模型并验证条件", target_agent="ComparisonAgent",
                              search_query="baseline", subject_action="INIT", subject_entity="主题")
    intent_agent, planner, retriever = MagicMock(), MagicMock(), MagicMock()
    intent_agent.parse_intent.return_value = decision
    planner.plan.return_value = RetrievalPlan(online_search=True)
    calls = []
    def search(**kwargs):
        calls.append(kwargs)
        return [{"doc_name": f"paper {len(calls)}", "doc_type": "arxiv", "text": "observed facts"}]
    retriever.hybrid_search_and_deduplicate.side_effect = search
    monkeypatch.setattr(MasterEngineRegistry, "get_intent_agent", lambda: intent_agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    return planner, retriever, calls, lambda: intent_routing_node({"user_query": "比较模型并验证条件"})


def test_router_continues_until_required_facts_are_supported(coverage_router):
    planner, retriever, calls, run = coverage_router
    assessments = []
    for index, (status, facts) in enumerate([
        ("missing", []), ("partial", ["identity verified"]),
        ("partial", ["identity verified", "date verified"]),
    ]):
        assessments.append(RetrievalCoverage(task_checks=[{
            "requirement": "model evolution", "status": status, "evidence_ids": [1] if facts else [],
            "supported_facts": facts,
        }], gaps=[{"query": f"targeted lookup {index}", "requirement": "model evolution",
                   "strategy": f"inspect source {index}", "rationale": "verify remaining facts"}]))
    assessments.append(RetrievalCoverage(sufficient=True, task_checks=[{
        "requirement": "model evolution", "status": "supported", "evidence_ids": [2, 3, 4],
    }]))
    planner.assess_coverage.side_effect = assessments
    result = run()
    assert len(calls) == 4  # Initial search and three justified follow-ups.
    assert result["retrieval_coverage"]["status"] == "sufficient"
    assert result["retrieval_coverage"]["stop_reason"] == "covered"


def test_empty_lookup_can_use_another_strategy(coverage_router):
    planner, retriever, calls, run = coverage_router
    original = retriever.hybrid_search_and_deduplicate.side_effect
    def search(**kwargs):
        if kwargs["query"] == "broad lookup":
            calls.append(kwargs)
            return []
        return original(**kwargs)
    retriever.hybrid_search_and_deduplicate.side_effect = search
    planner.assess_coverage.side_effect = [
        RetrievalCoverage(gaps=[{"query": "broad lookup", "requirement": "identity", "strategy": "title search"}]),
        RetrievalCoverage(gaps=[{"query": "exact author identifier", "requirement": "identity", "strategy": "author lookup"}]),
        RetrievalCoverage(sufficient=True, task_checks=[{"requirement": "identity", "status": "supported", "evidence_ids": [2]}]),
    ]
    result = run()
    assert len(calls) == 3
    assert result["retrieval_coverage"]["status"] == "sufficient"
    assert result["retrieval_coverage"]["search_history"][1]["result"] == "no_new_evidence"


def test_independent_requirements_are_not_limited_to_three_queries(coverage_router):
    planner, retriever, calls, run = coverage_router
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[{"query": f"requirement {i}"} for i in range(6)])
    planner.assess_coverage.return_value = RetrievalCoverage(sufficient=True, task_checks=[{
        "requirement": f"requirement {i}", "status": "supported", "evidence_ids": [i + 1],
    } for i in range(6)])
    result = run()
    assert len(calls) == 6
    assert len(result["retrieval_queries"]) == 6
    assert result["retrieval_coverage"]["status"] == "sufficient"


def test_safety_budget_never_counts_as_sufficient(coverage_router, monkeypatch):
    planner, retriever, calls, run = coverage_router
    monkeypatch.setenv("RETRIEVAL_GAP_ROUNDS", "2")
    planner.assess_coverage.side_effect = [RetrievalCoverage(gaps=[{
        "query": f"lookup {i}", "requirement": "still missing", "strategy": f"source {i}",
    }]) for i in range(3)]
    result = run()
    assert len(calls) == 3
    assert result["retrieval_coverage"]["status"] == "budget_exhausted"
    assert result["retrieval_coverage"]["unresolved"]


def test_default_allows_more_than_four_progressive_searches(coverage_router):
    planner, retriever, calls, run = coverage_router
    planner.assess_coverage.side_effect = [RetrievalCoverage(task_checks=[{
        "requirement": "necessary facts", "status": "partial", "evidence_ids": [1],
        "supported_facts": [f"verified fact {j}" for j in range(i + 1)],
    }], gaps=[{"query": f"lookup needed fact {i}", "requirement": "necessary facts", "strategy": f"inspect reference {i}"}])
        for i in range(5)] + [RetrievalCoverage(sufficient=True, task_checks=[{
            "requirement": "necessary facts", "status": "supported", "evidence_ids": [1, 2, 3, 4, 5, 6],
        }])]
    result = run()
    assert len(calls) == 6
    assert result["retrieval_coverage"]["status"] == "sufficient"


def test_retrieving_more_irrelevant_papers_without_new_strategy_stops(coverage_router):
    planner, retriever, calls, run = coverage_router
    planner.assess_coverage.side_effect = [RetrievalCoverage(task_checks=[{
        "requirement": "identity", "status": "missing",
    }], gaps=[{"query": f"unrelated search {i}", "rationale": f"invented gap {i}"}]) for i in range(3)]
    result = run()
    assert len(calls) == 3
    assert result["retrieval_coverage"]["stop_reason"] == "stalled"


def test_same_source_additional_content_is_merged(coverage_router):
    planner, retriever, calls, run = coverage_router
    values = iter(["initial abstract", "later verified experimental detail"])
    def search(**kwargs):
        calls.append(kwargs)
        return [{"doc_name": "same paper", "doc_type": "arxiv", "text": next(values)}]
    retriever.hybrid_search_and_deduplicate.side_effect = search
    planner.assess_coverage.side_effect = [
        RetrievalCoverage(gaps=[{"query": "targeted detail", "strategy": "inspect details"}]),
        RetrievalCoverage(sufficient=True, task_checks=[{"requirement": "details", "status": "supported", "evidence_ids": [1]}]),
    ]
    result = run()
    assert result["retrieved_evidence"][0]["text"] == "initial abstract\nlater verified experimental detail"
    assert len(result["retrieved_evidence"]) == 1


def test_new_papers_do_not_justify_repeating_same_claim_strategy(coverage_router):
    planner, retriever, calls, run = coverage_router
    planner.assess_coverage.side_effect = [RetrievalCoverage(task_checks=[{
        "requirement": "missing fact", "status": "missing",
    }], gaps=[{"query": q, "requirement": "missing fact", "strategy": "broad survey"}])
        for q in ("some broad query", "synonymous broad query")]
    result = run()
    assert len(calls) == 2
    assert result["retrieval_coverage"]["status"] == "unresolved"
    assert result["retrieval_coverage"]["stop_reason"] == "no_new_strategy"


@pytest.mark.parametrize("mode, expected_searches, expected_assessments", [
    ("many_gaps", 9, 2),
    ("no_new_evidence", 9, 2),
    ("already_searched", 1, 1),
    ("sufficient", 1, 1),
])
def test_router_handles_all_distinct_gaps_without_repeating_them(monkeypatch, mode, expected_searches, expected_assessments):
    monkeypatch.delenv("RETRIEVAL_GAP_ROUNDS", raising=False)
    decision = IntentDecision(
        domain="推荐系统", intent="比较后续模型", target_agent="ComparisonAgent",
        search_query="OneTrans follow-up models", online_search_query="OneTrans follow-up models",
        subject_action="INIT", subject_entity="OneTrans",
        task_segments=[{"retrieval_query": f"same task view {i}"} for i in range(6)],
        sub_queries=[{"query": f"other task view {i}"} for i in range(3)],
    )
    intent_agent, planner, retriever = MagicMock(), MagicMock(), MagicMock()
    intent_agent.parse_intent.return_value = decision
    planner.plan.return_value = RetrievalPlan(online_search=True)
    gaps = [{"query": f"specific missing fact {i}", "rationale": f"verify requirement {i}"}
            for i in range(8)]
    if mode == "already_searched":
        gaps = [{"query": "models OneTrans follow-up", "rationale": "same search"}]
    assessment = RetrievalCoverage(sufficient=mode == "sufficient", gaps=[] if mode == "sufficient" else gaps,
                                   task_checks=[{"requirement": "用户任务", "status": "supported" if mode == "sufficient" else "missing",
                                                 "evidence_ids": [1] if mode == "sufficient" else []}], summary="coverage checked")
    planner.assess_coverage.return_value = assessment
    calls = []

    def search(**kwargs):
        calls.append(kwargs)
        return [{"doc_name": "same paper" if mode == "no_new_evidence" else f"paper {len(calls)}",
                 "doc_type": "arxiv", "location_label": "abstract", "text": "Inspected abstract",
                 "url": "https://example.org/paper"}]

    retriever.hybrid_search_and_deduplicate.side_effect = search
    monkeypatch.setattr(MasterEngineRegistry, "get_intent_agent", lambda: intent_agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = intent_routing_node({"user_query": "OneTrans 之后的推荐模型还有什么迭代？"})
    assert len(calls) == expected_searches
    assert planner.assess_coverage.call_count == expected_assessments
    assert len(result["retrieval_queries"]) == 1
    assert len(result["intent_decision"]["task_segments"]) == 6
    assert len(result["intent_decision"]["sub_queries"]) == 3
    if mode != "sufficient":
        assert result["retrieval_coverage"]["unresolved"]
    if mode == "many_gaps":
        assert calls[1]["query"] == "specific missing fact 0"
        assert len(result["retrieval_coverage"]["search_history"]) == 9


def test_router_gap_cannot_expand_to_unplanned_sources(monkeypatch):
    decision = IntentDecision(domain="领域", intent="查本地依据", target_agent="GeneralTutorAgent",
                              search_query="local fact", subject_entity="主题", subject_action="INIT")
    intent_agent, planner, retriever = MagicMock(), MagicMock(), MagicMock()
    intent_agent.parse_intent.return_value = decision
    planner.plan.return_value = RetrievalPlan(local_search=True)
    planner.assess_coverage.return_value = RetrievalCoverage(gaps=[{
        "query": "online missing fact", "sources": ["academic", "web", "github"], "rationale": "still missing",
    }])
    retriever.hybrid_search_and_deduplicate.return_value = []
    monkeypatch.setattr(MasterEngineRegistry, "get_intent_agent", lambda: intent_agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = intent_routing_node({"user_query": "只查本地依据"})
    assert retriever.hybrid_search_and_deduplicate.call_count == 1
    assert result["retrieval_coverage"]["status"] == "unresolved"


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
    mock_retrieval_planner = MagicMock()
    mock_retrieval_planner.plan.return_value = RetrievalPlan(
        local_search=False,
        online_search=True,
        include_github=False,
        parallel_search=False,
        reasoning="在线检索学术论文",
    )
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: mock_retrieval_planner)

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
