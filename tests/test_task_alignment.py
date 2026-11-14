"""Compare proposed subjects/tasks with the original request before retrieval."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as workflow
from intent_recognition_agent import IntentDecision
from retrieval_planner import RetrievalPlan, RetrievalPlannerAgent


def planner_returning(payload):
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test"
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]),
    ))))
    return planner


@pytest.mark.parametrize("action", ["REPLACE", "APPEND"])
def test_original_comparison_corrects_task_before_search_and_commits_subject_once(monkeypatch, action):
    original = "OneTrans之后有哪些推荐模型，重点说明序列和非序列交互，不需要代码"
    wrong = IntentDecision(domain="推荐系统", intent="证明OneTrans的直接继承关系并编写实现",
                           target_agent="GeneralTutorAgent", subject_action=action, subject_entity="错误主题",
                           search_query="OneTrans direct inheritance implementation",
                           task_segments=[{"action": "编写实现"}], negative_constraints=[])
    corrected = {
        "intent": "了解后续推荐模型及序列与非序列交互机制", "subject_entity": "推荐模型迭代",
        "task_segments": [{"action": "说明", "object": "推荐模型的交互机制"}],
        "search_query": "OneTrans recommendation interaction models", "negative_constraints": ["代码"],
        "extracted_entities": {},
    }
    planner = planner_returning({
        "requires_evidence": True, "online_search": True, "is_resolved": True,
        "queries": [{"query": corrected["search_query"]}],
        "task_alignment": {"status": "corrected", "issues": ["额外添加直接继承证明和编程要求"],
                           "corrected_fields": corrected},
    })
    planner.assess_coverage = MagicMock(side_effect=RuntimeError("Offline test: stop after initial lookup"))
    agent, retriever = MagicMock(), MagicMock()
    agent.parse_intent.return_value = wrong
    observed = []

    def search(**kwargs):
        observed.append(kwargs)
        return [{"doc_name": "Paper", "doc_type": "arxiv", "text": "Reported mechanism"}]

    retriever.hybrid_search_and_deduplicate.side_effect = search
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: agent)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = workflow.intent_routing_node({
        "user_query": original, "active_subject": "旧主题",
        "topic_state": {"current_topic": "旧主题", "status": "LOCKED", "evolution_path": ["旧主题"]},
    })
    create = planner.client.chat.completions.create
    create.assert_called_once()
    context = json.loads(create.call_args.kwargs["messages"][1]["content"])
    assert context["verify_task_against_original"] is True
    assert context["user_query"] == original
    assert context["proposed_intent"]["intent"] == wrong.intent
    assert context["previous_subject"] == "旧主题"
    assert observed and {call["query"] for call in observed} == {corrected["search_query"]}
    assert all(call["negative_keywords"] == ["代码"] for call in observed)
    assert result["intent_decision"]["intent"] == corrected["intent"]
    assert result["intent_decision"]["task_segments"] == corrected["task_segments"]
    expected = "推荐模型迭代" if action == "REPLACE" else "旧主题 & 推荐模型迭代"
    assert result["active_subject"] == expected
    assert "错误主题" not in result["topic_state"]["evolution_path"]
    assert len(result["subject_evolution_log"]) == 1
    assert result["retrieval_plan"]["task_alignment"]["status"] == "corrected"


@pytest.mark.parametrize("alignment", [
    None,
    {"status": "unresolved", "issues": ["任务含义尚不明确"]},
    {"status": "corrected", "corrected_fields": {}},
    {"status": "matched", "corrected_fields": {"intent": "悄悄修改任务"}},
    {"status": "corrected", "corrected_fields": {"task_segments": "invalid schema"}},
    {"status": "corrected", "corrected_fields": {"unknown_field": "invented"}},
])
def test_failed_comparison_preserves_subject_and_prevents_search_and_generation(monkeypatch, alignment):
    payload = {"requires_evidence": True, "online_search": True, "local_search": True,
               "is_resolved": True, "queries": [{"query": "wrong task search"}]}
    if alignment is not None:
        payload["task_alignment"] = alignment
    planner = planner_returning(payload)
    agent = MagicMock()
    agent.parse_intent.return_value = IntentDecision(
        domain="领域", intent="暂定任务", target_agent="CrossModalRAGAgent",
        subject_action="REPLACE", subject_entity="暂定主题", search_query="wrong task search")
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: agent)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    forbidden = MagicMock(side_effect=AssertionError("No source or answer call before task alignment"))
    for method in ("get_hybrid_retriever", "get_rag_engine", "get_paper_search_agent", "get_llm_client"):
        monkeypatch.setattr(workflow.MasterEngineRegistry, method, forbidden)
    monkeypatch.setattr(workflow, "safe_chat_completion", forbidden)
    state = {"user_query": "先明确需求", "active_subject": "原主题",
             "topic_state": {"current_topic": "原主题", "status": "LOCKED"}}
    routed = workflow.intent_routing_node(state)
    assert routed["active_subject"] == "原主题"
    assert routed["subject_evolution_log"] == []
    assert routed["retrieved_evidence"] == []
    assert routed["intent_decision"]["is_resolved"] is False
    assert not routed["retrieval_plan"]["online_search"]
    assert not routed["retrieval_plan"]["local_search"]
    assert workflow.route_selector(routed) == "general_tutor_pipeline"
    answered = workflow.general_tutor_node({**state, **routed})
    synthesized = workflow.evidence_synthesis_node({**state, **routed, **answered})
    reviewed = workflow.dialectical_inquiry_node({**state, **routed, **answered, **synthesized})
    assert reviewed["status"] == "dialectical_inquiry_skipped"
    forbidden.assert_not_called()
    planner.client.chat.completions.create.assert_called_once()


def test_matched_comparison_keeps_task_unchanged():
    planner = planner_returning({"requires_evidence": False, "is_resolved": True,
                                "task_alignment": {"status": "matched"}})
    intent = {"intent": "整理用户提供的文字", "subject_entity": "用户笔记"}
    plan = planner.plan("整理这些笔记", intent, {"verify_task_against_original": True})
    assert plan.is_resolved
    assert plan.task_alignment.status == "matched"
    assert plan.task_alignment.corrected_fields == {}
    assert intent == {"intent": "整理用户提供的文字", "subject_entity": "用户笔记"}
