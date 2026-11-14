"""Regressions for planned lookups and evidence continuity across answer routes."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from master_agent_workflow import (
    MasterEngineRegistry, _build_deterministic_evidence_slate,
    dialectical_inquiry_node, evidence_synthesis_node, online_paper_search_node,
    reflective_revision_node, review_gap_retrieval_node,
)
from online_paper_search_agent import AcademicResource, ArxivPaper, OnlinePaperSearchAgent
from prompts import format_evidence_for_prompt
from retrieval_planner import (
    RetrievalPlan, RetrievalPlannerAgent, apply_evidence_relevance, filter_irrelevant_candidates,
    select_retrieval_probes,
)


@pytest.mark.parametrize("raw", [
    "[confirmed OneTrans DOI or full title] recommendation benchmark ablation",
    "[确认后的 OneTrans DOI 或完整标题] recommendation architecture",
    "请提供 OneTrans 推荐模型的完整论文标题、作者、DOI、链接或摘要片段",
])
def test_invalid_review_query_cannot_be_replaced_by_a_broader_planner_survey(monkeypatch, raw):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[
        {"query": "OneTrans 正式论文", "english_query": '"OneTrans" recommender system'},
        {"query": "OneTrans 后续模型", "english_query": '"OneTrans" follow-up recommendation'},
    ])
    retriever.hybrid_search_and_deduplicate.side_effect = [
        [{"doc_name": "Anchor", "doi": "10.test/anchor", "text": "Identity"}],
        [{"doc_name": "Anchor", "doi": "10.test/anchor", "text": "Additional mechanism"},
         {"doc_name": "Follow-up", "doi": "10.test/later", "text": "Reported improvement"}],
    ]
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    state = {"user_query": "比较后续模型", "review_verdict": {"evidence_requests": [{"query": raw}]}}
    result = review_gap_retrieval_node(state)
    retriever.hybrid_search_and_deduplicate.assert_not_called()
    assert result["review_retrieval_log"][-1]["result"] == "invalid_executable_query"
    assert "retrieved_evidence" not in result


@pytest.mark.parametrize("query,english", [
    ("[confirmed title] recommendation", ""),
    ("请提供论文标题或链接", ""),
    ("OneTrans recommendation", "after [confirmed title]"),
    ("Please provide the paper title and DOI", ""),
])
def test_invalid_plans_never_reach_sources_or_fall_back_to_other_task_views(monkeypatch, query, english):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[{
        "query": query, "english_query": english,
    }])
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = review_gap_retrieval_node({
        "user_query": "比较后续模型",
        "review_verdict": {"evidence_requests": [{"query": "verify missing result"}]},
    })
    retriever.hybrid_search_and_deduplicate.assert_not_called()
    assert result["review_retrieval_status"] == "unresolved"
    assert result["review_retrieval_log"][-1]["result"] == "invalid_executable_query"
    assert select_retrieval_probes("other task", {}, planner.plan.return_value) == []


def test_coverage_cannot_use_rejected_evidence_as_verified_support():
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test"
    response = {"sufficient": True, "task_checks": [
        {"requirement": "模型机制", "status": "supported", "evidence_ids": [1], "supported_facts": ["invalid support"]},
        {"requirement": "相关基线", "status": "supported", "evidence_ids": [2], "supported_facts": ["baseline"]},
    ], "irrelevant_evidence": [{"evidence_id": 1, "reason": "研究的是流体而非推荐模型", "basis": [
        {"source": "request", "quote": "推荐模型比较"},
        {"source": "evidence", "evidence_id": 1, "quote": "Flow simulation"}]}]}
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]))))
    evidence = [{"doc_name": "Aerodynamics", "text": "Flow simulation"},
                {"doc_name": "Related baseline", "text": "Sequential recommendation baseline"}]
    assessment = planner.assess_coverage("推荐模型比较", {}, evidence)
    assert assessment.sufficient is False
    assert assessment.task_checks[0].status == "missing"
    assert assessment.task_checks[0].supported_facts == []
    assert assessment.task_checks[1].status == "supported"
    filtered = apply_evidence_relevance(evidence, assessment.irrelevant_evidence, "推荐模型比较")
    assert "Aerodynamics" not in format_evidence_for_prompt(filtered)
    assert "[2] Related baseline" in format_evidence_for_prompt(filtered)
    assert filtered[0]["text"] == "Flow simulation"  # Audit trail remains intact.
    assert "excluded_as_irrelevant" not in evidence[0]  # No shared-object mutation.


def test_one_failed_planned_lookup_does_not_discard_other_lookup_results(monkeypatch):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[
        {"query": "missing identity"}, {"query": "useful method"},
    ])
    retriever.hybrid_search_and_deduplicate.side_effect = [
        TimeoutError(), [{"doc_name": "Preserved source", "text": "Useful reported method"}],
    ]
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = review_gap_retrieval_node({
        "user_query": "比较模型", "review_verdict": {"evidence_requests": [
            {"query": "missing identity"}, {"query": "useful method"}]},
    })
    assert result["review_retrieval_status"] == "retrieved"
    assert result["retrieved_evidence"][0]["doc_name"] == "Preserved source"
    assert result["review_retrieval_log"][0]["failed_queries"][0]["error"] == "TimeoutError"
    assert result["review_retrieval_log"][1]["executed_queries"][0]["query"] == "useful method"


def test_online_route_review_and_revision_share_full_evidence_and_citation_ids(monkeypatch):
    evidence = [{"doc_name": f"Source {i}", "text": f"Fact {i}", "doc_type": "crossref"}
                for i in range(1, 57)]
    evidence[1]["excluded_as_irrelevant"] = "Clearly different research task"
    evidence[50].update({"doc_type": "arxiv", "arxiv_id": "test-51",
                         "text": "Mechanism. " * 180 + "FULL_ABSTRACT_TAIL",
                         "summary": "Supplementary reported measurement"})
    agent = OnlinePaperSearchAgent.__new__(OnlinePaperSearchAgent)
    agent.llm_client = MagicMock()
    agent.search_multisource = MagicMock(side_effect=AssertionError("Do not repeat a completed search"))
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", lambda: agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    formatted = format_evidence_for_prompt(evidence)
    messages = []
    draft = "完整定义、公式、比较及适用条件。" * 300

    def complete(**kwargs):
        prompt = "\n".join(m["content"] for m in kwargs["messages"])
        messages.append(prompt)
        if "你是交付前审核者" in prompt:
            return json.dumps({"verdict": "revise", "evidence_status": "supported", "task_checks": [],
                               "findings": [{"severity": "blocking", "location": "模型比较段",
                                             "reason": "缺少测量条件", "suggestion": "补充摘要中的测量条件"}]})
        return draft

    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    # Include a shared candidate in a different order: the state evidence remains authoritative.
    state = {"user_query": "比较模型机制和条件", "intent_decision": {}, "retrieved_evidence": evidence,
             "recent_paper_candidates": [{"source": "arxiv", "title": "Source 51", "paper_id": "test-51"}]}
    online = online_paper_search_node(state)
    reviewed = dialectical_inquiry_node({**state, **online})
    revised = reflective_revision_node({**state, **online, **reviewed, "pending_evidence_additions": [
        {**ev, "evidence_id": i} for i, ev in enumerate(evidence, 1)
    ]})
    assert len(messages) == 3
    for prompt in messages:
        assert formatted in prompt
        assert "FULL_ABSTRACT_TAIL" in prompt
        assert "[51] Source 51" in prompt and "[56] Source 56" in prompt
        assert "[2] Source 2 " not in prompt
    assert revised["final_answer"] == draft
    agent.search_multisource.assert_not_called()


def test_standalone_brief_uses_one_numbering_across_sources_and_complete_abstracts(monkeypatch):
    agent = OnlinePaperSearchAgent.__new__(OnlinePaperSearchAgent)
    agent.llm_client = MagicMock()
    complete = MagicMock(return_value="Answer")
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    agent.generate_search_brief(
        "compare models", papers=[ArxivPaper(paper_id="test-id", title="Preprint",
                                             summary="Methods. " * 180 + "TAIL", pdf_url="https://example.org/preprint")],
        openalex_papers=[AcademicResource(id="oa", title="Published", source="openalex", summary="Published abstract")],
        crossref_papers=[AcademicResource(id="doi", title="DOI record", source="crossref", summary="Bibliographic information")],
        github_repos=[AcademicResource(id="repo", title="Repository", source="github", version_ref="commit-123")],
    )
    prompt = complete.call_args.kwargs["messages"][0]["content"]
    for snippet in ("[1] Published", "[2] Preprint", "[3] DOI record", "[4] Repository",
                    "Methods. " * 180 + "TAIL", "commit-123", "test-id"):
        assert snippet in prompt


def test_candidate_only_online_route_retains_identical_evidence_for_review(monkeypatch):
    agent = MagicMock()
    agent.generate_search_brief.return_value = "Answer"
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", lambda: agent)
    candidates = [{"source": "arxiv", "title": "Shared paper", "paper_id": "test-id",
                   "summary": "Full abstract. " * 100 + "END", "pdf_url": "https://example.org/paper"}]
    result = online_paper_search_node({"user_query": "比较模型", "recent_paper_candidates": candidates})
    assert result["retrieved_evidence"] == agent.generate_search_brief.call_args.kwargs["evidence"]
    assert "END" in format_evidence_for_prompt(result["retrieved_evidence"])
    agent.search_multisource.assert_not_called()


def test_off_topic_page_does_not_remove_other_useful_pages_or_the_document_candidate():
    evidence = [{"doc_name": "Textbook", "text": "Different chapter", "excluded_as_irrelevant": "Off topic"},
                {"doc_name": "Textbook", "text": "Relevant derivation"}]
    candidates = [{"title": "Textbook", "summary": "Full document metadata"}]
    assert filter_irrelevant_candidates(candidates, evidence) == candidates
    assert "[2] Textbook" in format_evidence_for_prompt(evidence)


def test_reviewer_excludes_only_explicit_mismatches_and_preserves_other_ids(monkeypatch):
    evidence = [{"doc_name": "Noise", "text": "Unrelated flow benchmark"},
                {"doc_name": "Related theory", "text": "Useful local theoretical premise"},
                {"doc_name": "Uncertain lead", "text": "Possibly related; needs verification"}]
    response = {"verdict": "pass", "evidence_status": "limited", "task_checks": [
        {"requirement": "限定证据范围", "result": "met"}],
        "irrelevant_evidence": [{"evidence_id": 1, "reason": "研究对象为流体", "basis": [
            {"source": "request", "quote": "比较推荐模型"},
            {"source": "evidence", "evidence_id": 1, "quote": "Unrelated flow benchmark"}]},
                                {"evidence_id": 2, "reason": ""}, {"evidence_id": 999, "reason": "invalid id"}]}
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(response))
    result = dialectical_inquiry_node({
        "user_query": "比较推荐模型", "final_answer": "仅说明资料所支持的理论，后续模型待确认。",
        "retrieved_evidence": evidence,
        "recent_paper_candidates": [{"title": e["doc_name"]} for e in evidence],
    })
    formatted = format_evidence_for_prompt(result["retrieved_evidence"])
    assert "[1] Noise" not in formatted
    assert "[2] Related theory" in formatted and "[3] Uncertain lead" in formatted
    assert [c["title"] for c in result["recent_paper_candidates"]] == ["Related theory", "Uncertain lead"]


def test_synthesis_cannot_reintroduce_rejected_sources_or_invent_relevance(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_EVIDENCE_SYNTHESIS", "true")
    evidence = [{"doc_name": "Noise", "text": "BAD_SOURCE", "excluded_as_irrelevant": "Off topic"},
                {"doc_name": "Useful source", "text": "Full mechanism. " * 80 + "END_DETAIL"}]
    candidates = [{"title": "Noise", "summary": "BAD_SOURCE"},
                  {"title": "Useful source", "summary": "Relevant supporting premise"}]
    complete = MagicMock(return_value="Navigation only")
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    evidence_synthesis_node({"user_query": "推荐模型", "retrieved_evidence": evidence,
                             "recent_paper_candidates": candidates})
    prompt = complete.call_args.kwargs["messages"][0]["content"]
    assert "BAD_SOURCE" not in prompt and "END_DETAIL" in prompt
    slate = _build_deterministic_evidence_slate("推荐模型", candidates, evidence)
    assert "Useful source" in slate  # Different language is not a rejection criterion.
    assert "BAD_SOURCE" not in slate
    assert "均紧密支撑" not in slate
    empty_slate = _build_deterministic_evidence_slate("推荐模型", [], [])
    assert "0 项" in empty_slate and "独立来源" in empty_slate
