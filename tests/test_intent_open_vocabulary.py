import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from intent_recognition_agent import IntentRecognitionAgent


def fake_response(payload):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))],
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


def test_open_domain_normalization_preserves_recommended_capability():
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: fake_response(base_payload())))
    )

    decision = agent.parse_intent("设计一个跨学科研究方案")

    assert decision.domain == "materials_science"
    assert decision.disciplines == ["materials_science", "machine_learning"]
    assert decision.intent == "research_design"
    assert decision.target_agent == "DocumentSearchAgent"


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


def test_malformed_optional_object_fields_are_normalized_without_retry():
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    relative_time = "重点关注 OneTrans 之后三年的研究"
    create = MagicMock(return_value=fake_response(base_payload(
        framework_reference="OneTrans",
        time_constraint=relative_time,
    )))
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    decision = agent.parse_intent("推荐场景用户序列和非序列交互，OneTrans之后推荐模型还有什么迭代？")

    assert create.call_count == 1
    assert decision.framework_reference is None
    assert decision.time_constraint is None
    assert "OneTrans之后推荐模型还有什么迭代" in decision.time_constraint_note
    assert relative_time not in decision.time_constraint_note  # The model invented a three-year limit.
    assert "未转换为公历日期" in decision.reasoning


def test_explicit_date_constraint_dict_is_preserved():
    agent = IntentRecognitionAgent.__new__(IntentRecognitionAgent)
    agent.model = "test"
    constraint = {"has_time_constraint": True, "start_year": 2022, "end_year": 2024}
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: fake_response(base_payload(time_constraint=constraint))
    )))
    decision = agent.parse_intent("查看2022到2024年的研究")
    assert decision.time_constraint == constraint
    assert not decision.time_constraint_note
