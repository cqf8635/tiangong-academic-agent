import json
from types import SimpleNamespace

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
