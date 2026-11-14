"""Malformed numeric review fields must not discard an otherwise usable review."""

import json
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as workflow
from retrieval_planner import retrieval_query_key


@pytest.mark.parametrize("facts,expected", [
    ([1, 2, "已有机制", None, {"evidence_id": 3}], ["已有机制"]),
    (7, []),
    ("已有机制", ["已有机制"]),
    (None, []),
])
def test_numeric_review_facts_survive_postprocessing_without_retry(monkeypatch, facts, expected):
    check = {"requirement": "模型机制", "result": "unclear", "supported_facts": facts}
    payload = {"verdict": "inconclusive", "evidence_status": "limited", "task_checks": [check],
               "evidence_requests": [{"query": "mechanism verification", "requirement": 1,
                                      "strategy": 2, "reason": 3}]}
    complete = MagicMock(return_value=json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(workflow, "safe_chat_completion", complete)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    original = "完整回答。\n\n补充内容。[2]"
    result = workflow.dialectical_inquiry_node({
        "user_query": "说明模型机制", "final_answer": original, "evidence_append_count": 1,
        "retrieved_evidence": [{"doc_name": "Paper", "text": "已有机制"}],
        "review_retrieval_rounds": 2,
        "review_retrieval_log": [
            {"result": "evidence_added", "task_checks": [check]},
            {"result": "evidence_added", "task_checks": [check]},
        ],
    })
    assert result["status"] != "dialectical_inquiry_fallback"
    assert result["review_verdict"]["verdict"] != "unavailable"
    assert result["review_checks"][0]["supported_facts"] == expected
    assert not result["review_verdict"]["evidence_requests"]
    request = result["review_verdict"]["deferred_evidence_requests"][0]
    assert request["query"] == "mechanism verification"
    assert (request["requirement"], request["strategy"], request["reason"]) == ("", "", "")
    assert result["review_retrieval_status"] is None
    assert result["final_answer"] == original
    assert json.loads(result["dialectical_notes"])["task_checks"][0]["supported_facts"] == expected
    complete.assert_called_once()


def test_numeric_evidence_ids_are_not_new_verified_facts():
    state = {"review_retrieval_log": [
        {"result": "evidence_added", "task_checks": [
            {"requirement": "机制", "result": "unclear", "supported_facts": [1]}]},
        {"result": "evidence_added", "task_checks": [
            {"requirement": "机制", "result": "unclear", "supported_facts": [2]}]},
    ]}
    assert workflow._review_search_stalled(state, [
        {"requirement": "机制", "result": "unclear", "supported_facts": [3]}])
    assert not workflow._review_search_stalled(state, [
        {"requirement": "机制", "result": "unclear", "supported_facts": [3, "新验证的交互机制"]}])


@pytest.mark.parametrize("value", [1, 0, None, [1], {"evidence_id": 1}])
def test_query_key_does_not_treat_nontext_values_as_queries(value):
    assert retrieval_query_key(value) == ""
