"""Keep the user's task intact when handing it to retrieval and answer generation."""

from unittest.mock import MagicMock

import pytest

import master_agent_workflow as workflow
from intent_recognition_agent import IntentDecision
from retrieval_planner import RetrievalPlan


@pytest.mark.parametrize("action,entity,expected", [
    ("REPLACE", "推荐模型迭代", "推荐模型迭代"),
    ("INIT", "推荐模型迭代", "推荐模型迭代"),
    ("INHERIT", "", "原有主题"),
])
def test_planning_uses_current_subject_and_original_task(monkeypatch, action, entity, expected):
    query = "了解推荐场景中序列与非序列交互的模型迭代，不需要实现代码"
    decision = IntentDecision(domain="推荐系统", intent="了解模型的机制变化",
                              target_agent="GeneralTutorAgent", search_query="recommendation models",
                              subject_action=action, subject_entity=entity)
    agent, planner = MagicMock(), MagicMock()
    agent.parse_intent.return_value = decision
    planner.plan.return_value = RetrievalPlan(requires_evidence=False)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: agent)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    source = MagicMock(side_effect=AssertionError("A no-search plan must not call sources"))
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", source)
    result = workflow.intent_routing_node({
        "user_query": query, "active_subject": "原有主题",
        "topic_state": {"current_topic": "原有主题", "status": "LOCKED"},
    })
    call = planner.plan.call_args.kwargs
    assert call["user_query"] == query
    assert call["intent"]["intent"] == decision.intent
    assert call["state_context"]["active_subject"] == expected
    assert result["active_subject"] == expected
    planner.plan.assert_called_once()
    source.assert_not_called()


def test_online_answer_receives_original_task_instead_of_search_expression(monkeypatch):
    query = "推荐场景中，OneTrans之后的模型如何处理序列和非序列交互？说明适用场景，不需要代码。"
    agent = MagicMock()
    agent.generate_search_brief.return_value = "模型、机制与适用场景。"
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_paper_search_agent", lambda: agent)
    evidence = [{"doc_name": "Source", "text": "Reported mechanism"}]
    result = workflow.online_paper_search_node({
        "user_query": query, "intent_decision": {"search_query": "OneTrans recommendation follow-up"},
        "retrieval_queries": ["sequential recommendation", "non-sequential feature fusion"],
        "retrieved_evidence": evidence,
    })
    assert agent.generate_search_brief.call_args.kwargs["query"] == query
    assert agent.generate_search_brief.call_args.kwargs["evidence"] == evidence
    assert result["final_answer"] == "模型、机制与适用场景。"
    agent.generate_search_brief.assert_called_once()
    agent.search_multisource.assert_not_called()


def test_generated_entity_hints_do_not_override_the_original_request(monkeypatch):
    query = "解释这个模型的交互机制，不需要证明或实验设计。"
    completion = MagicMock(return_value="依据资料解释交互机制。[1]")
    monkeypatch.setattr(workflow, "safe_chat_completion", completion)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    workflow.general_tutor_node({
        "user_query": query, "retrieved_evidence": [{"doc_name": "Source", "text": "交互机制"}],
        "intent_decision": {"extracted_entities": {"模型推断的设问": "证明收敛性并设计消融实验"}},
    })
    prompt = completion.call_args.kwargs["messages"][1]["content"]
    assert query in prompt
    assert "模型推断，仅供理解原话，不增加用户任务" in prompt
    assert "最高优先级必答" not in prompt
    assert "请务必以直接、严密解答与分步证明" not in prompt
    completion.assert_called_once()
