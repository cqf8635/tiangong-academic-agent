"""Defect reports need locatable grounds, and supplements receive the actual gap."""

import json
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as workflow
from retrieval_planner import RetrievalPlan


@pytest.mark.parametrize("ref,kind,located", [
    ({"source": "evidence", "evidence_id": 1, "quote": "仅适用于有限规模"}, "factual", True),
    ({"source": "evidence", "evidence_id": 1, "quote": "额外前提：独立采样"}, "factual", True),
    ({"source": "request", "quote": "说明适用条件"}, "completeness", True),
    ({"source": "answer", "quote": "对任何规模都成立"}, "factual", False),
    ({"source": "evidence", "evidence_id": 1, "quote": "不存在的来源原句"}, "factual", False),
    ({"source": "evidence", "evidence_id": 99, "quote": "仅适用于有限规模"}, "factual", False),
    ({"source": "evidence", "evidence_id": True, "quote": "仅适用于有限规模"}, "factual", False),
    ({"source": "evidence", "evidence_id": 2, "quote": "已排除资料的文字"}, "factual", False),
    ({"source": "request", "quote": "必须给出直接继承关系"}, "completeness", False),
    ({"source": "model_memory", "quote": "理论常识"}, "factual", False),
])
def test_review_basis_must_exist_in_the_named_input(monkeypatch, ref, kind, located):
    response = {"verdict": "revise", "evidence_status": "limited",
                "findings": [{"kind": kind, "location": "结论", "reason": "适用条件需补证",
                              "basis": [ref], "basis_status": "located"}],
                "task_checks": [{"requirement": "说明适用条件", "result": "unclear"}]}
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr(workflow, "safe_chat_completion", lambda **kwargs: json.dumps(response, ensure_ascii=False))
    original = "结论：对任何规模都成立。"
    result = workflow.dialectical_inquiry_node({
        "user_query": "说明适用条件", "final_answer": original,
        "retrieved_evidence": [
            {"doc_name": "理论依据", "text": "仅适用于有限规模", "summary": "额外前提：独立采样"},
            {"doc_name": "无关资料", "text": "已排除资料的文字", "excluded_as_irrelevant": "无关"},
        ],
    })
    finding = result["review_verdict"]["findings"][0]
    assert finding["basis_status"] == ("located" if located else "unverified")
    if located:
        assert finding["basis"] == [ref]
    else:
        assert "待核实" in result["review_notice"]
    assert result["final_answer"] == original


def test_retrieval_appends_evidence_for_the_reported_gap_without_rewriting(monkeypatch):
    original = "完整推导。\n结论：对任何规模都成立。\n"
    known = {"doc_name": "已读论文", "doi": "10.test/known", "text": "有限规模下已验证"}
    new = {"doc_name": "补查定理", "doi": "10.test/new", "text": "该定理仅适用于有限规模，不能推出无限规模结论"}
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[{"query": "theorem finite size conditions"}])
    retriever.hybrid_search_and_deduplicate.return_value = [known, new]
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    state = {"user_query": "完整推导并说明适用条件", "final_answer": original, "retrieved_evidence": [known],
             "review_verdict": {
                 "findings": [{"kind": "factual", "location": "结论", "reason": "适用规模需要核实",
                               "suggestion": "删除全文重新写", "basis": [
                                   {"source": "answer", "quote": "对任何规模都成立"},
                                   {"source": "evidence", "evidence_id": 1, "quote": "有限规模下已验证"},
                               ]}],
                 "evidence_requests": [{"query": "theorem applicability", "requirement": "适用规模",
                                        "reason": "补充规模条件的来源"}],
             }}
    retrieved = workflow.review_gap_retrieval_node(state)
    assert workflow.review_gap_result_routing(retrieved) == "evidence_append"

    def append(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        context = prompt.split("本次待补证问题", 1)[1].split("本次补查新增资料", 1)[0]
        assert "适用规模需要核实" in context
        assert "补充规模条件的来源" in context
        assert "有限规模下已验证" in context
        assert '"evidence_id": 1' in context
        assert "删除全文重新写" not in prompt
        assert "[2] 补查定理" in prompt
        assert new["text"] in prompt
        return "适用条件补证：该定理限于有限规模；原文的无限规模推广尚无依据。[2]"

    monkeypatch.setattr(workflow, "safe_chat_completion", append)
    result = workflow.evidence_append_node({**state, **retrieved})
    assert result["final_answer"].startswith(original)
    assert "适用条件补证" in result["final_answer"][len(original):]
    assert result["pending_evidence_additions"] == []


def test_unverified_material_finding_cannot_be_reported_as_a_completed_review(monkeypatch):
    response = {"verdict": "pass", "evidence_status": "supported",
                "task_checks": [{"requirement": "说明方法", "result": "met"}],
                "findings": [{"kind": "factual", "severity": "material", "location": "正文",
                              "reason": "模型认为公式有误", "basis": "invalid response shape"}]}
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr(workflow, "safe_chat_completion", lambda **kwargs: json.dumps(response, ensure_ascii=False))
    result = workflow.dialectical_inquiry_node({
        "user_query": "说明方法", "final_answer": "原有论述与公式。",
        "retrieved_evidence": [{"doc_name": "来源", "text": "已读内容"}],
    })
    assert result["review_verdict"]["findings"][0]["basis_status"] == "unverified"
    assert result["review_verdict"]["verdict"] == "inconclusive"
    assert result["final_answer"] == "原有论述与公式。"
