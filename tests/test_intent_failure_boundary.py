"""No API/model loading: verify failed intent parsing cannot enter later stages."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from intent_recognition_agent import IntentDecision, IntentParsingError
import master_agent_workflow as workflow


def test_workflow_stops_before_retrieval_generation_or_topic_changes(monkeypatch):
    parser = Mock()
    parser.parse_intent.side_effect = IntentParsingError("解析失败")
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: parser)
    downstream = Mock(side_effect=AssertionError("解析失败后不应执行下游"))
    for name in ("get_rag_engine", "get_hybrid_retriever", "get_paper_search_agent", "get_llm_client"):
        monkeypatch.setattr(workflow.MasterEngineRegistry, name, downstream)
    state = {
        "user_query": "一个独立的新问题",
        "active_subject": "原主题",
        "topic_state": {"current_topic": "原主题", "status": "LOCKED"},
        "recent_paper_candidates": [],
        "retrieved_evidence": [],
    }
    before = copy.deepcopy(state)

    with pytest.raises(IntentParsingError):
        workflow.build_master_workflow(checkpointer=None).invoke(state)

    downstream.assert_not_called()
    assert state == before


def test_online_node_does_not_replace_empty_query_with_history(monkeypatch):
    downstream = Mock(side_effect=AssertionError("空查询不能触发在线任务"))
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_paper_search_agent", downstream)

    with pytest.raises(IntentParsingError):
        workflow.online_paper_search_node({
            "user_query": "请继续找",
            "topic_state": {"current_topic": "原主题"},
            "intent_decision": {"search_query": "", "subject_action": "INHERIT"},
        })

    downstream.assert_not_called()


def test_document_node_does_not_search_with_full_user_input(monkeypatch):
    downstream = Mock(side_effect=AssertionError("空查询不能自动改用完整用户原话"))
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_rag_engine", downstream)

    with pytest.raises(IntentParsingError):
        workflow.document_rag_node({
            "user_query": "很长的完整问题和全部条件",
            "intent_decision": {"search_query": ""},
        })

    downstream.assert_not_called()


def test_explicit_no_local_source_does_not_trigger_router_retrieval(monkeypatch):
    parser = Mock()
    parser.parse_intent.return_value = IntentDecision(
        domain="统计", intent="stepwise_derivation", target_agent="DerivationAgent",
        subject_action="INIT", subject_entity="给定样本", required_data_sources=[], search_query="",
    )
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: parser)
    downstream = Mock(side_effect=AssertionError("不应补足本地检索"))
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", downstream)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_rag_engine", downstream)

    result = workflow.intent_routing_node({"user_query": "只用题目条件推导", "recent_paper_candidates": []})

    assert result["retrieved_evidence"] == []
    downstream.assert_not_called()


def test_chat_saves_failure_without_archiving_topic_or_creating_mindmap():
    # Execute the actual handler body without importing server startup/background jobs.
    path = Path(__file__).resolve().parents[1] / "backend_api.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_execute_chat_workflow")
    manager = Mock()
    manager.get_session.return_value = SimpleNamespace(topic_catalog=[])
    manager.get_topic_state.return_value = {"current_topic": "原主题", "status": "LOCKED"}
    manager.get_recent_candidates.return_value = []
    app = Mock()
    app.invoke.side_effect = IntentParsingError("解析失败")
    namespace = {
        "ChatRequest": object, "Dict": dict, "Any": object,
        "SessionManager": manager, "ChatMessage": lambda **kwargs: SimpleNamespace(**kwargs),
        "MasterAgentState": dict, "master_app": app, "IntentParsingError": IntentParsingError,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)

    result = namespace["_execute_chat_workflow"](SimpleNamespace(
        session_id="test-session", query="新的问题", image_data=None, voice=None,
    ))

    assert result["status"] == "intent_parsing_failed"
    assert result["intent_decision"] is None
    assert result["stage_statuses"] == {"intent_parsing": "failed", "generation": "skipped"}
    assert result["mindmap_markdown"] is None
    assert result["retrieved_evidence"] == []
    assert manager.add_message.call_count == 2  # User input and honest failure notice.
    manager.record_turn_topic.assert_not_called()
    manager.update_topic_state.assert_not_called()
    manager.update_recent_candidates.assert_not_called()
    manager.save_session.assert_not_called()
