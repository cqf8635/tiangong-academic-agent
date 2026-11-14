"""Concrete regressions for inflated review requirements and reopened surveys."""
import json
import socket
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as w
from retrieval_planner import RetrievalPlan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network or real model calls in review scope tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr(w, "safe_chat_completion", forbidden)


def review(monkeypatch, payload, query="量子力学有什么最新研究？", answer="量子基础与计算的近期成果。[1]", **extra):
    complete = MagicMock(return_value=json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(w, "safe_chat_completion", complete)
    result = w.dialectical_inquiry_node({"user_query": query, "final_answer": answer,
        "retrieved_evidence": [{"doc_name": "研究摘要", "text": "量子研究结果"}], **extra})
    complete.assert_called_once()
    assert result["final_answer"] == answer
    return result


def test_completed_overview_does_not_turn_optional_metrics_into_automatic_search(monkeypatch):
    payload = {"verdict": "pass", "evidence_status": "limited",
        "task_checks": [{"requirement": "介绍近期方向与代表成果", "result": "met"}],
        "findings": [{"kind": "completeness", "severity": "material", "check_ids": [1],
            "reason": "还应逐项提供灵敏度、码距、实验误差条，并均衡覆盖整个学科", "basis": [
                {"source": "request", "quote": "量子力学有什么最新研究？"}]}],
        "evidence_requests": [{"query": "quantum experimental metrics complete survey",
            "purpose": "missing_task", "finding_ids": [1]}]}
    result = review(monkeypatch, payload)
    verdict = result["review_verdict"]
    assert verdict["verdict"] == "pass"
    assert verdict["findings"][0]["kind"] == "advisory"
    assert verdict["findings"][0]["reported_kind"] == "completeness"
    assert not verdict["evidence_requests"]
    assert verdict["deferred_evidence_requests"][0]["query"] == payload["evidence_requests"][0]["query"]
    assert w.reviewer_decision_routing(result) == "tts_synthesis"


@pytest.mark.parametrize("query,missing", [
    ("介绍量子纠错，并比较两种实验平台", "比较两种实验平台"),
    ("说明材料老化证据，并给出验证方案", "给出验证方案"),
    ("比较两种历史解释，并说明史料分歧", "说明史料分歧"),
])
def test_real_omission_remains_searchable_across_fields(monkeypatch, query, missing):
    payload = {"verdict": "inconclusive", "evidence_status": "limited",
        "task_checks": [{"requirement": "已答部分", "result": "met"}, {"requirement": missing, "result": "missing"}],
        "findings": [{"kind": "completeness", "severity": "material", "check_ids": [2],
            "reason": "原话要求的第二项未回答", "basis": [{"source": "request", "quote": missing}]}],
        "evidence_requests": [{"query": missing, "purpose": "missing_task", "finding_ids": [1]}]}
    result = review(monkeypatch, payload, query=query)
    assert result["review_verdict"]["evidence_requests"][0]["retrieval_decision"] == "actionable"
    assert w.reviewer_decision_routing(result) == "review_gap_retrieval"


def test_met_task_can_still_verify_a_specific_unsupported_answer_claim(monkeypatch):
    payload = {"verdict": "inconclusive", "evidence_status": "limited",
        "task_checks": [{"requirement": "介绍方法", "result": "met"}],
        "findings": [{"kind": "evidence", "severity": "material", "reason": "该数值尚无资料支持",
            "basis": [{"source": "answer", "quote": "灵敏度提升100倍"}]}],
        "evidence_requests": [{"query": "method measured sensitivity gain 100", "purpose": "verify_claim", "finding_ids": [1]}]}
    result = review(monkeypatch, payload, answer="该方法灵敏度提升100倍。")
    assert result["review_verdict"]["findings"][0]["basis_status"] == "unverified"
    assert len(result["review_verdict"]["evidence_requests"]) == 1
    assert w.reviewer_decision_routing(result) == "review_gap_retrieval"


@pytest.mark.parametrize("proposal", [
    {"query": "broader survey"},
    {"query": "broader survey", "purpose": "optional", "finding_ids": [1]},
    {"query": "broader survey", "purpose": "verify_claim", "finding_ids": [999]},
    {"query": "broader survey", "purpose": "verify_claim", "finding_ids": [True]},
    {"query": "broader survey", "purpose": "verify_claim", "finding_ids": "1"},
    {"query": "broader survey", "purpose": "missing_task", "finding_ids": [1]},
])
def test_unlinked_or_optional_requests_are_recorded_without_extra_calls(monkeypatch, proposal):
    payload = {"verdict": "pass", "evidence_status": "supported", "findings": [
        {"kind": "advisory", "reason": "未来可以更深入"}],
        "task_checks": [{"requirement": "介绍方向", "result": "met"}], "evidence_requests": [proposal]}
    result = review(monkeypatch, payload)
    assert not result["review_verdict"]["evidence_requests"]
    assert len(result["review_verdict"]["deferred_evidence_requests"]) == 1
    assert result["review_verdict"]["verdict"] == "pass"
    assert w.reviewer_decision_routing(result) == "tts_synthesis"


def test_fake_or_absent_request_quote_cannot_promote_new_requirement(monkeypatch):
    payload = {"verdict": "pass", "evidence_status": "supported",
        "task_checks": [{"requirement": "给出完整实验误差条", "result": "missing"}],
        "findings": [{"kind": "completeness", "check_ids": [1], "reason": "需要误差条", "basis": [
            {"source": "request", "quote": "给出完整实验误差条"}]}],
        "evidence_requests": [{"query": "complete error bars", "purpose": "missing_task", "finding_ids": [1]}]}
    result = review(monkeypatch, payload)
    assert result["review_verdict"]["findings"][0]["scope_status"] == "unverified_requirement"
    assert not result["review_verdict"]["evidence_requests"]


def test_unmatched_new_check_cannot_become_an_original_task_omission(monkeypatch):
    payload = {"verdict": "inconclusive", "evidence_status": "limited",
        "task_checks": [{"task_ids": [1], "requirement": "介绍方向", "result": "met"},
                        {"task_ids": [9], "requirement": "核实全部误差条", "result": "missing"}],
        "findings": [{"kind": "completeness", "check_ids": [2], "reason": "需要完整指标", "basis": [
            {"source": "request", "quote": "量子力学有什么最新研究？"}]}],
        "evidence_requests": [{"query": "all error bars", "purpose": "missing_task", "finding_ids": [1]}]}
    result = review(monkeypatch, payload, intent_decision={"task_segments": [{"task": "介绍方向"}]})
    assert not result["review_verdict"]["evidence_requests"]
    assert result["review_verdict"]["findings"][0]["scope_status"] == "unverified_requirement"


def test_nonexistent_answer_quote_does_not_authorize_claim_search(monkeypatch):
    payload = {"verdict": "inconclusive", "evidence_status": "limited", "task_checks": [],
        "findings": [{"kind": "evidence", "reason": "需要核实", "basis": [
            {"source": "answer", "quote": "已经实现通用容错"}]}],
        "evidence_requests": [{"query": "verify universal fault tolerance", "purpose": "verify_claim", "finding_ids": [1]}]}
    result = review(monkeypatch, payload)
    assert not result["review_verdict"]["evidence_requests"]


def test_filtered_malformed_checks_do_not_shift_original_review_links(monkeypatch):
    query = "介绍成果并说明条件"
    payload = {"verdict": "inconclusive", "evidence_status": "limited",
        "task_checks": [None, {"requirement": "介绍成果", "result": "met"},
                        {"requirement": "说明条件", "result": "missing"}],
        "findings": [None, {"kind": "completeness", "check_ids": [3], "reason": "条件缺失", "basis": [
            {"source": "request", "quote": "说明条件"}]}],
        "evidence_requests": [{"query": "applicability conditions", "purpose": "missing_task", "finding_ids": [2]}]}
    result = review(monkeypatch, payload, query=query)
    assert [check["check_id"] for check in result["review_verdict"]["task_checks"]] == [2, 3]
    assert result["review_verdict"]["findings"][0]["scope_status"] == "missing_task"
    assert result["review_verdict"]["findings"][0]["finding_id"] == 2
    assert w.reviewer_decision_routing(result) == "review_gap_retrieval"


def test_gap_planner_cannot_expand_one_lookup_into_six_subject_surveys(monkeypatch):
    original = "量子力学有什么最新研究？"
    request = {"query": "量子材料代表性实验", "english_query": "quantum materials representative experiments"}
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, include_web=True, queries=[
        {"query": name} for name in ["材料", "纠错", "算法", "网络", "基础", "冷原子"]])
    retriever.hybrid_search_and_deduplicate.return_value = [{"doc_name": "材料实验", "text": "观测结果"}]
    monkeypatch.setattr(w.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    state = {"user_query": original, "intent_decision": {"intent": "全领域综述", "is_cross_disciplinary": True,
        "sub_queries": [{"query": "算法"}], "time_constraint": {"start_year": 2024, "end_year": 2025}},
        "final_answer": "完整原回答", "review_verdict": {"evidence_requests": [request]}}
    before = deepcopy(state)
    result = w.review_gap_retrieval_node(state)
    call = retriever.hybrid_search_and_deduplicate.call_args.kwargs
    retriever.hybrid_search_and_deduplicate.assert_called_once()
    assert (call["query"], call["english_query"]) == (request["query"], request["english_query"])
    context = planner.plan.call_args.kwargs
    assert context["user_query"] == request["query"]
    assert context["state_context"]["shared_task"]["original_user_query"] == original
    assert context["intent"]["sub_queries"] == []
    assert call["time_constraint"] == state["intent_decision"]["time_constraint"]
    assert len(result["review_retrieval_log"][0]["proposed_queries"]) == 6
    assert len(result["review_retrieval_log"][0]["executed_queries"]) == 1
    assert result["pending_evidence_additions"][0]["text"] == "观测结果"
    assert state == before


def test_optional_review_does_not_reach_search_in_real_graph(monkeypatch):
    original = "已有研究与机制。" * 200
    monkeypatch.setattr(w, "intent_routing_node", lambda state: {"target_agent": "GeneralTutorAgent"})
    author = MagicMock(return_value={"final_answer": original, "retrieved_evidence": [{"text": "来源报告"}]})
    monkeypatch.setattr(w, "general_tutor_node", author)
    monkeypatch.setattr(w, "evidence_synthesis_node", lambda state: {})
    monkeypatch.setattr(w, "tts_synthesis_node", lambda state: {})
    never = MagicMock(side_effect=AssertionError("Optional suggestions must not cause planning or retrieval"))
    monkeypatch.setattr(w.MasterEngineRegistry, "get_retrieval_planner", never)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_hybrid_retriever", never)
    completion = MagicMock(return_value=json.dumps({"verdict": "pass", "evidence_status": "limited",
        "task_checks": [{"requirement": "介绍近期研究", "result": "met"}],
        "findings": [{"kind": "completeness", "severity": "material", "check_ids": [1],
            "reason": "还可以补充更多平台", "basis": [{"source": "request", "quote": "介绍近期研究"}]}],
        "evidence_requests": [{"query": "more platforms", "purpose": "missing_task", "finding_ids": [1]}]}))
    monkeypatch.setattr(w, "safe_chat_completion", completion)
    result = w.build_master_workflow(checkpointer=None).invoke({"user_query": "介绍近期研究"})
    assert result["final_answer"] == original
    never.assert_not_called()
    completion.assert_called_once()
    author.assert_called_once()
