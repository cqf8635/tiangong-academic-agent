import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from intent_recognition_agent import IntentRecognitionAgent


def _response(payload):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))])


def test_multiturn_intent_reads_state_without_using_fixed_source_flags(monkeypatch):
    """Intent follows the current task and state; retrieval source choice stays separate."""
    agent = IntentRecognitionAgent()
    outputs = [
        {
            "domain": "量子场论",
            "intent": "检索并比较近期非微扰研究",
            "task_segments": [{"action": "检索并比较", "object": "非微扰研究", "retrieval_query": "非微扰量子场论格点规范理论近期研究"}],
            "target_agent": "GeneralTutorAgent",
            "subject_action": "INIT",
            "subject_entity": "量子场论非微扰与格点规范理论",
            "need_online_paper_search": False,
        },
        {
            "domain": "量子场论",
            "intent": "从已有资料筛选并区分出版类型",
            "task_segments": [{"action": "筛选", "object": "已有候选资料", "retrieval_query": "候选资料的研究对象和出版类型"}],
            "target_agent": "GeneralTutorAgent",
            "subject_action": "INHERIT",
            "need_online_paper_search": True,
        },
        {
            "domain": "量子场论",
            "intent": "推导Wilson项抑制费米子倍增的机制",
            "task_segments": [{"action": "推导", "object": "Wilson项消除虚假极点"}],
            "target_agent": "GeneralTutorAgent",
            "subject_action": "INHERIT",
            "need_online_paper_search": True,
        },
    ]
    create = MagicMock(side_effect=[_response(item) for item in outputs])
    agent.client.chat.completions.create = create

    first_state = {
        "user_query": "检索近两年量子场论非微扰方法，说明主要问题、方法和贡献",
        "active_subject": None,
        "topic_state": {"current_topic": "", "status": "INIT"},
        "recent_paper_candidates": [],
    }
    first = agent.parse_intent(state=first_state)
    assert first.is_resolved
    assert first.subject_action == "INIT"
    assert first.intent == "检索并比较近期非微扰研究"
    assert first.need_online_paper_search is False  # ignored by the separate retrieval planner

    second_state = {
        "user_query": "从刚才的资料筛选强耦合规范场论，并区分预印本和正式论文",
        "active_subject": first.subject_entity,
        "topic_state": {"current_topic": first.subject_entity, "status": "LOCKED"},
        "recent_paper_candidates": [{"title": "A previous paper", "source": "arxiv"}],
    }
    second = agent.parse_intent(state=second_state)
    assert second.subject_action == "INHERIT"
    assert second.intent != first.intent
    assert second.need_online_paper_search is True  # does not itself force an online search

    third = agent.parse_intent(state={
        **second_state,
        "user_query": "进一步推导Wilson项如何消除虚假极点",
        "active_subject": first.subject_entity,
    })
    assert third.subject_action == "INHERIT"
    assert "推导" in third.intent
    assert create.call_count == 3
