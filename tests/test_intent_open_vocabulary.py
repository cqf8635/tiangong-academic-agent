import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from intent_recognition_agent import IntentParsingError, IntentRecognitionAgent


def fake_response(payload):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))],
    )


def base_payload(**overrides):
    payload = {
        "domain": "general",
        "intent": "research_design",
        "confidence": 0.9,
        "target_agent": "DocumentSearchAgent",
        "required_data_sources": ["general_science_knowledge"],
        "search_query": "跨学科研究方案",
        "subject_action": "INIT",
        "subject_entity": "研究主题",
        "negative_constraints": [],
        "need_online_paper_search": False,
        "online_search_query": "",
        "topic_action": "INIT_LOCK",
        "sub_topic": "",
        "is_cross_disciplinary": True,
        "disciplines": ["materials_science", "machine_learning"],
        "sub_queries": [],
        "extracted_entities": {},
        "reasoning": "测试",
    }
    payload.update(overrides)
    return payload


def test_open_domain_normalization_and_research_design_routing():
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: fake_response(base_payload())))
    )

    decision = agent.parse_intent("设计一个跨学科研究方案")

    assert decision.domain == "materials_science"
    assert decision.disciplines == ["materials_science", "machine_learning"]
    assert decision.intent == "research_design"
    assert decision.target_agent == "StudyPlannerAgent"


def test_general_domain_is_preserved_without_specific_disciplines():
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: fake_response(base_payload(intent="general_chitchat", disciplines=["general"]))
            )
        )
    )

    decision = agent.parse_intent("你好")

    assert decision.domain == "general"
    assert decision.disciplines == ["general"]


def test_explicit_data_scope_is_not_expanded():
    for sources in ([], ["general_science_knowledge"]):
        agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
        agent.model = "test"
        agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_: fake_response(base_payload(required_data_sources=sources))
        )))

        decision = agent.parse_intent("仅用题目给出的条件设计验证步骤，不查其他材料")

        assert decision.required_data_sources == sources


def test_missing_data_scope_does_not_invent_sources():
    payload = base_payload()
    del payload["required_data_sources"]
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: fake_response(payload)
    )))

    decision = agent.parse_intent("设计验证步骤")

    assert decision.required_data_sources == []


def test_information_extraction_preserves_relations_tasks_and_unspecified_method():
    extraction = {
        "objects": ["光照", "幼苗生长"],
        "goal": "比较影响并说明可能原因",
        "actions": ["比较两组生长情况", "说明可能原因"],
        "relations": ["光照变化与幼苗生长的关系"],
        "conditions": ["仅使用给出的两组观测"],
        "method": None,
        "time_expression": None,
    }
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: fake_response(base_payload(extracted_entities=extraction))
    )))

    decision = agent.parse_intent("仅使用给出的两组观测，比较光照变化下幼苗的生长情况，并说明可能原因")

    assert decision.extracted_entities == extraction
    assert decision.time_constraint is None


def test_api_failure_raises_without_inheriting_topic_or_switching_model(monkeypatch):
    monkeypatch.setattr("intent_recognition_agent.time.sleep", lambda _: None)
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "unrequested-model")
    create = Mock(side_effect=ConnectionError("unavailable"))
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with pytest.raises(IntentParsingError) as caught:
        agent.parse_intent(state={"user_query": "分析一个新的问题", "active_subject": "旧主题"})

    assert isinstance(caught.value.__cause__, ConnectionError)
    assert create.call_count == 3
    assert {call.kwargs["model"] for call in create.call_args_list} == {"test"}


@pytest.mark.parametrize("payload", [
    base_payload(search_query="", required_data_sources=["uploaded_pdf_docs"]),
    base_payload(search_query=None, need_online_paper_search=True),
    base_payload(intent="unsupported_intent"),
])
def test_invalid_intent_result_is_not_replaced_with_a_default(payload, monkeypatch):
    monkeypatch.setattr("intent_recognition_agent.time.sleep", lambda _: None)
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: fake_response(payload)
    )))

    with pytest.raises(IntentParsingError):
        agent.parse_intent("不能把这段完整原话自动塞进检索词")


def test_non_search_task_can_keep_empty_search_query():
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: fake_response(base_payload(search_query="", required_data_sources=[]))
    )))

    decision = agent.parse_intent("只分析给定条件，不查资料")

    assert decision.search_query == ""
    assert decision.required_data_sources == []
