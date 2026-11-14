"""Remaining audit regressions. No network, GPU loading or real control actions."""
import importlib.util
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as w
from retrieval_planner import apply_evidence_relevance
from session_manager import ChatMessage, SessionManager
from system_controls import execute_system_control


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected network, engine loading or model call")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_llm_client", forbidden)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_rag_engine", forbidden)
    monkeypatch.setattr(w, "safe_chat_completion", forbidden)


@pytest.mark.parametrize("basis", [None, "reason only", [42],
    [{"source": "request", "quote": "推荐模型"}],
    [{"source": "request", "quote": "推荐模型"}, {"source": "evidence", "evidence_id": 1, "quote": "虚构内容"}],
    [{"source": "request", "quote": "推荐模型"}, {"source": "evidence", "evidence_id": True, "quote": "流体"}],
])
def test_unlocated_exclusions_keep_readable_source_and_ids(basis):
    evidence = [{"doc_name": "报告", "text": "流体", "summary": "摘要"}, {"text": "相关理论"}]
    result = apply_evidence_relevance(evidence, [{"evidence_id": 1, "reason": "模型认为无关", "basis": basis}], "推荐模型")
    assert len(result) == 2 and result[0]["text"] == "流体"
    assert "excluded_as_irrelevant" not in result[0]
    assert result[0]["relevance_review"]["basis_status"] == "unverified"
    assert "relevance_review" not in evidence[0]


@pytest.mark.parametrize("source", ["request", "visual_context", "attachment"])
def test_exclusion_requires_locatable_task_and_target_source(source):
    result = apply_evidence_relevance([{"doc_name": "Flow simulation", "text": "完整保留"}], [
        {"evidence_id": 1, "reason": "研究对象不匹配", "basis": [
            {"source": source, "quote": "推荐模型"},
            {"source": "evidence", "evidence_id": 1, "quote": "Flow simulation"}]}],
        "推荐模型", "推荐模型")
    assert result[0]["excluded_as_irrelevant"] == "研究对象不匹配"
    assert result[0]["text"] == "完整保留"


def test_unverified_reviewer_exclusion_does_not_hide_evidence_or_change_answer(monkeypatch):
    monkeypatch.setattr(w.MasterEngineRegistry, "get_llm_client", MagicMock())
    monkeypatch.setattr(w, "safe_chat_completion", lambda **kwargs: json.dumps({
        "verdict": "pass", "evidence_status": "supported", "findings": [],
        "task_checks": [{"requirement": "解释理论", "result": "met"}],
        "irrelevant_evidence": [{"evidence_id": 1, "reason": "没有证明是直接后续"}]}))
    result = w.dialectical_inquiry_node({"user_query": "解释理论", "final_answer": "已有解释。[1]",
        "retrieved_evidence": [{"doc_name": "相关背景", "text": "理论依据"}],
        "recent_paper_candidates": [{"title": "相关背景"}]})
    assert result["recent_paper_candidates"] == [{"title": "相关背景"}]
    assert "excluded_as_irrelevant" not in result["retrieved_evidence"][0]
    assert result["final_answer"] == "已有解释。[1]"
    assert w.reviewer_decision_routing(result) == "tts_synthesis"


def test_control_executor_context_is_isolated_and_restored():
    barrier = Barrier(2)
    def run(label):
        with w.system_control_executor(lambda action, params: {"status": "completed", "message": label}):
            barrier.wait(timeout=5)
            return w.system_control_node({"intent_decision": {"control_action": "change_voice"}})
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, label) for label in ("session A", "session B")]
        assert [future.result()["final_answer"] for future in futures] == ["session A", "session B"]
    result = w.system_control_node({"intent_decision": {"control_action": "change_voice"}})
    assert result["control_result"]["status"] == "not_executed"


def test_control_executor_reaches_real_graph_node_without_being_checkpointed(monkeypatch):
    monkeypatch.setattr(w, "intent_routing_node", lambda state: {
        "target_agent": "SystemControlAgent", "intent_decision": {
            "target_agent": "SystemControlAgent", "control_action": "stop_recording"}})
    monkeypatch.setattr(w, "tts_synthesis_node", lambda state: {"tts_audio_file": None})
    graph = w.build_master_workflow(checkpointer=None)
    actual = MagicMock(return_value={"status": "completed", "message": "已停止", "is_recording": False})
    with w.system_control_executor(actual):
        result = graph.invoke({"user_query": "停止录音"})
    actual.assert_called_once_with("stop_recording", {})
    assert result["control_result"]["status"] == "completed"
    assert not any(callable(value) for value in result.values())


def test_controller_executes_specific_controls_and_keeps_failures_honest():
    voice = MagicMock(event_loop=NS(is_closed=lambda: False))
    voice.start_voice_agent.return_value = {"status": "starting", "message": "初始化", "is_recording": True}
    voice.stop_voice_agent.return_value = {"status": "completed", "message": "停止", "is_recording": False}
    sessions = MagicMock()
    call = lambda action, params={}: execute_system_control(action, params, session_id="s", voice_manager=voice, session_manager=sessions)
    assert call("start_recording", {"mode": "system"})["status"] == "starting"
    voice.start_voice_agent.assert_called_once_with(voice.event_loop, mode="system")
    assert call("pause_recording")["is_recording"] is False
    assert call("change_voice", {"voice": "晓晓"})["tts_voice"] == "zh-CN-XiaoxiaoNeural"
    sessions.save_session.assert_called_once()
    assert call("change_voice")["status"] == "not_executed"
    assert call("other_action")["status"] == "not_executed"
    assert call("clear_session")["status"] == "pending"
    sessions.clear_session_messages.assert_not_called()
    voice.event_loop = None
    assert call("start_recording")["status"] == "not_executed"


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Import API with its eager engines replaced before module execution."""
    import document_rag_engine
    import meeting_summarizer
    import online_paper_search_agent
    import radar_sync_engine
    import session_manager
    for module, name in [(document_rag_engine, "UniversalDocumentRAGEngine"),
        (meeting_summarizer, "MeetingSummarizer"), (online_paper_search_agent, "OnlinePaperSearchAgent"),
        (radar_sync_engine, "RadarSyncEngine")]:
        monkeypatch.setattr(module, name, MagicMock())
    for attr in ("rag_engine", "meeting_summarizer", "paper_search_agent"):
        monkeypatch.setattr(w.MasterEngineRegistry, attr, getattr(w.MasterEngineRegistry, attr))
    monkeypatch.setattr(session_manager, "SESSIONS_DIR", tmp_path)
    spec = importlib.util.spec_from_file_location("offline_backend_api", Path(w.__file__).with_name("backend_api.py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_api_clear_finishes_after_graph_and_does_not_republish_old_data(api, monkeypatch):
    session = SessionManager.create_session("旧话题")
    SessionManager.add_message(session.session_id, ChatMessage(sender="agent", content="旧答案"))
    checkpointer = MagicMock()
    monkeypatch.setattr(api, "GLOBAL_MEMORY_SAVER", checkpointer)
    def invoke(state, config):
        result = w.system_control_node({**state, "intent_decision": {"control_action": "clear_session"}})
        assert result["control_result"]["status"] == "pending"
        assert SessionManager.get_session(session.session_id).messages
        checkpointer.delete_thread.assert_not_called()
        return {**state, **result}
    monkeypatch.setattr(api.master_app, "invoke", invoke)
    result = api._execute_chat_workflow(api.ChatRequest(query="清空当前会话", session_id=session.session_id))
    assert result["control_result"]["status"] == "completed"
    cleared = SessionManager.get_session(session.session_id)
    assert not cleared.messages and not cleared.shared_pool and not cleared.shared_resources
    checkpointer.delete_thread.assert_called_once_with(session.session_id)


def test_api_voice_change_is_persisted_and_used_next_turn(api, monkeypatch):
    session = SessionManager.create_session("会话")
    def invoke(state, config):
        return {**state, **w.system_control_node({**state, "intent_decision": {
            "control_action": "change_voice", "control_parameters": {"voice": "晓晓"}}})}
    monkeypatch.setattr(api.master_app, "invoke", invoke)
    result = api._execute_chat_workflow(api.ChatRequest(query="换成晓晓", session_id=session.session_id))
    assert result["control_result"]["tts_voice"] == "zh-CN-XiaoxiaoNeural"
    assert SessionManager.get_session(session.session_id).tts_voice == "zh-CN-XiaoxiaoNeural"
    def next_turn(state, config):
        assert state["tts_voice"] == "zh-CN-XiaoxiaoNeural"
        assert state["control_result"] is None
        return {**state, "final_answer": "继续回答"}
    monkeypatch.setattr(api.master_app, "invoke", next_turn)
    api._execute_chat_workflow(api.ChatRequest(query="继续", session_id=session.session_id))


def test_failed_session_clear_does_not_claim_success_or_publish_old_turn(api, monkeypatch):
    session = SessionManager.create_session("会话")
    monkeypatch.setattr(api.master_app, "invoke", lambda state, config: {**state,
        "control_result": {"action": "clear_session", "status": "pending"}})
    monkeypatch.setattr(SessionManager, "clear_session_messages", MagicMock(side_effect=OSError("write failed")))
    publish = MagicMock()
    monkeypatch.setattr(SessionManager, "publish_turn_to_shared_pool", publish)
    result = api._execute_chat_workflow(api.ChatRequest(query="清空", session_id=session.session_id))
    assert result["control_result"]["status"] == "failed"
    publish.assert_not_called()


def test_voice_api_does_not_claim_started_while_previous_worker_is_stopping(api):
    manager = api.VoiceSessionManager()
    manager.agent_thread = NS(is_alive=lambda: True)
    loop = NS(is_closed=lambda: False)
    result = manager.start_voice_agent(loop)
    assert result["status"] == "stopping" and not result["is_recording"]


def test_long_text_is_fully_sent_to_tts_and_partial_audio_is_not_cached(monkeypatch, tmp_path):
    import edge_tts
    text = "开头。" + "完整长文。" * 2500 + "末尾哨兵。"
    captured = []
    class Communicate:
        def __init__(self, spoken, **kwargs):
            captured.append(spoken)
        async def save(self, path):
            Path(path).write_bytes(b"audio" * 300)
    monkeypatch.setattr(edge_tts, "Communicate", Communicate)
    monkeypatch.setattr(w, "BASE_DIR", tmp_path)
    result = w.tts_synthesis_node({"final_answer": text})
    assert captured == [text]
    assert Path(result["tts_audio_file"]).stat().st_size == 1500
    assert w.clean_text_for_speech(text).endswith("末尾哨兵。")
    class Failed(Communicate):
        async def save(self, path):
            Path(path).write_bytes(b"partial")
            raise RuntimeError("interrupted")
    monkeypatch.setattr(edge_tts, "Communicate", Failed)
    assert w.tts_synthesis_node({"final_answer": text + "新增"})["tts_audio_file"] is None
    assert len(list((tmp_path / "audio_cache").iterdir())) == 1


def test_mcp_speech_passes_full_text_and_writes_real_wav(monkeypatch, tmp_path):
    import mcp_server
    import subprocess
    audio = tmp_path / "source.mp3"
    audio.write_bytes(b"MP3")
    captured = []
    def synthesize(state):
        captured.append(state["final_answer"])
        return {"tts_audio_file": str(audio)}
    def convert(cmd, **kwargs):
        assert cmd[0] == "ffmpeg" and str(audio) in cmd
        Path(cmd[-1]).write_bytes(b"RIFF_WAV")
    monkeypatch.setattr(w, "tts_synthesis_node", synthesize)
    monkeypatch.setattr(subprocess, "run", convert)
    text = "完整内容" * 3000 + "最后一段"
    target = tmp_path / "output.wav"
    assert "成功" in mcp_server.text_to_speech(text, str(target))
    assert captured == [text] and target.read_bytes() == b"RIFF_WAV"


@pytest.mark.parametrize("query,tasks,source", [
    ("梳理推荐模型进展，并提出后续验证方案，不要代码", ["已有成果", "验证方案"], "模型研究摘要"),
    ("分析材料老化的已有证据与后续实验，说明条件", ["老化证据", "实验方案", "适用条件"], "湿度与温度实验记录"),
    ("比较两种历史解释的依据，并提出进一步研究路径", ["已有解释比较", "后续研究"], "史料中的记载与分歧"),
])
def test_multidisciplinary_author_and_reviewer_keep_raw_task_and_full_data(monkeypatch, query, tasks, source):
    """Contract checks only: this intentionally makes no semantic-quality claim."""
    decision = {"task_segments": [{"task": task} for task in tasks]}
    evidence = [{"doc_name": "来源", "text": source + "完整原文" * 1000 + "资料末尾"}]
    answer = "已有成果与后续建议。" * 1000 + "回答末尾。[1]"
    calls = []
    def complete(**kwargs):
        calls.append(str(kwargs["messages"]))
        assert query in calls[-1] and "资料末尾" in calls[-1]
        assert all(task in calls[-1] for task in tasks)
        if len(calls) == 1:
            return answer
        assert "回答末尾" in calls[-1]
        return json.dumps({"verdict": "pass", "evidence_status": "supported", "task_checks": [
            {"task_ids": list(range(1, len(tasks) + 1)), "requirement": query, "result": "met"}]})
    monkeypatch.setattr(w.MasterEngineRegistry, "get_llm_client", MagicMock())
    monkeypatch.setattr(w, "safe_chat_completion", complete)
    state = {"user_query": query, "intent_decision": decision, "retrieved_evidence": evidence,
             "retrieval_plan": {"local_search": False, "online_search": False}}
    authored = w.general_tutor_node(state)
    reviewed = w.dialectical_inquiry_node({**state, **authored})
    assert len(calls) == 2 and reviewed["final_answer"] == answer
    assert reviewed["review_verdict"]["verdict"] == "pass"
    assert reviewed["retrieved_evidence"] == evidence
