"""Supplemental retrieval must preserve the existing answer and stable citations."""

import json
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as workflow
from retrieval_planner import RetrievalPlan


@pytest.mark.parametrize("rounds", [1, 2])
def test_graph_appends_only_new_sources_and_never_reauthors(monkeypatch, rounds):
    original = "  原回答与公式：\n\n  x = a + b  \n完整讨论。[1]\n\n"
    sources = [{"doc_name": name, "doi": f"10.test/{name}", "text": f"NEW_FACT_{name}"}
               for name in ("A", "B", "C")]
    author = MagicMock(return_value={"final_answer": original})
    monkeypatch.setattr(workflow, "intent_routing_node", lambda state: {
        "target_agent": "GeneralTutorAgent", "retrieved_evidence": [sources[0]],
    })
    monkeypatch.setattr(workflow, "general_tutor_node", author)
    monkeypatch.setattr(workflow, "evidence_synthesis_node", lambda state: {})
    monkeypatch.setattr(workflow, "tts_synthesis_node", lambda state: {})
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.side_effect = [RetrievalPlan(online_search=True, queries=[{"query": f"lookup {i}"}])
                               for i in range(rounds)]
    retriever.hybrid_search_and_deduplicate.side_effect = [sources[:i + 2] for i in range(rounds)]
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    reviews, additions = [], []

    def complete(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "你是交付前审核者" in prompt:
            i = len(reviews)
            reviews.append(prompt)
            return json.dumps({
                "verdict": "pass" if i == rounds else "revise", "needs_revision": True,
                "evidence_status": "supported",
                "findings": [{"kind": "completeness", "check_ids": [1], "severity": "blocking", "location": "正文", "reason": "机制说明缺项",
                    "basis": [{"source": "request", "quote": "比较相关模型机制"}]}],
                "task_checks": [{"requirement": "相关模型机制", "result": "met" if i == rounds else "missing",
                                 "supported_facts": [f"fact{j}" for j in range(i + 1)]}],
                "evidence_requests": [] if i == rounds else [{
                    "query": f"gap {i}", "purpose": "missing_task", "finding_ids": [1],
                    "requirement": "相关模型机制", "strategy": f"strategy {i}"}],
            }, ensure_ascii=False)
        i = len(additions)
        additions.append(prompt)
        assert original in prompt
        delta = prompt.split("本次补查新增资料：\n", 1)[1]
        assert f"[{i + 2}] {sources[i + 1]['doc_name']}" in delta
        assert sources[i + 1]["text"] in delta
        for old in sources[:i + 1]:
            assert old["text"] not in delta
        return f"新增机制{i + 1}。[{i + 2}]"

    monkeypatch.setattr(workflow, "safe_chat_completion", complete)
    app = workflow.build_master_workflow(checkpointer=None)
    result = app.invoke({"user_query": "比较相关模型机制"}, config={"recursion_limit": 30})
    expected = original
    for i in range(rounds):
        expected += f"\n\n---\n\n### 补充内容\n\n新增机制{i + 1}。[{i + 2}]"
    assert result["final_answer"] == expected
    assert result["final_answer"].encode().startswith(original.encode())
    assert result["retrieved_evidence"] == sources[:rounds + 1]
    assert result["pending_evidence_additions"] == []
    assert result["evidence_append_count"] == rounds
    assert len(reviews) == rounds + 1
    assert len(additions) == rounds
    author.assert_called_once()
    assert "reflective_revision" not in app.get_graph().nodes


@pytest.mark.parametrize("incoming,expected", [
    ("已知机制", ""),
    ("已知机制\n新增测量条件", "新增测量条件"),
])
def test_same_source_reuses_id_and_only_passes_new_text(monkeypatch, incoming, expected):
    source = {"doc_name": "A", "doi": "10.test/A", "text": "已知机制"}
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[{"query": "model mechanism"}])
    retriever.hybrid_search_and_deduplicate.return_value = [{**source, "text": incoming}]
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = workflow.review_gap_retrieval_node({
        "user_query": "解释模型", "retrieved_evidence": [source],
        "review_verdict": {"evidence_requests": [{"query": "missing conditions"}]},
    })
    assert len(result["retrieved_evidence"]) == 1
    assert result["retrieved_evidence"][0]["text"] == incoming
    assert source["text"] == "已知机制"
    pending = result["pending_evidence_additions"]
    if expected:
        assert len(pending) == 1
        assert pending[0]["evidence_id"] == 1
        assert pending[0]["text"] == expected
    else:
        assert pending == []
        complete = MagicMock(side_effect=AssertionError("Duplicate evidence needs no generation"))
        monkeypatch.setattr(workflow, "safe_chat_completion", complete)
        appended = workflow.evidence_append_node({**result, "final_answer": "原文"})
        assert appended["final_answer"] == "原文"
        complete.assert_not_called()
