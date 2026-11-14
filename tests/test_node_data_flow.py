"""Regression cases for audited cross-node data loss; all engines and model calls are offline."""
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as w
import online_paper_search_agent as online
from cross_source_hybrid_retriever import CrossSourceHybridRetriever, UnifiedEvidenceCard
from intent_recognition_agent import IntentDecision
from meeting_summarizer import MeetingSummaryReport
from retrieval_planner import RetrievalPlan
from workflow_contracts import merge_evidence, task_payload


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External calls are forbidden in node data-flow tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr(w.time, "sleep", lambda _: None)


def test_cross_modal_reuses_all_sources_and_leaves_them_for_review(monkeypatch):
    sources = [{"doc_name": "Slides", "doc_type": "pptx", "page_or_slide": i,
                "location_label": f"slide {i}", "text": f"PAGE_{i}_DETAIL"} for i in range(1, 4)]
    forbidden = MagicMock(side_effect=AssertionError("No repeated lookup"))
    monkeypatch.setattr(w.MasterEngineRegistry, "get_rag_engine", forbidden)
    complete = MagicMock(return_value="解释和页码 [1][2][3]")
    monkeypatch.setattr(w, "safe_chat_completion", complete)
    result = w.cross_modal_node({"user_query": "详细解释这些课件", "retrieved_evidence": sources,
                                "retrieval_plan": {"local_search": False}})
    prompt = str(complete.call_args.kwargs["messages"])
    assert all(source["text"] in prompt for source in sources)
    assert result["retrieved_evidence"] == sources
    assert "简要说明该页" not in prompt
    forbidden.assert_not_called()


def test_meeting_keeps_report_fields_and_actual_selected_transcript(monkeypatch, tmp_path):
    records = tmp_path / "records"; records.mkdir()
    selected = records / "record_selected.jsonl"
    selected.write_text('{"text":"SELECTED_TRANSCRIPT"}\n')
    (records / "record_latest.jsonl").write_text('{"text":"OTHER_SESSION"}\n')
    summarizer = MagicMock()
    summarizer.summarize_session.return_value = MeetingSummaryReport(
        title="会议", overview="概览", key_decisions=["决议"],
        discussion_topics=[{"topic": "讨论", "summary": "FULL_DISCUSSION"}],
        action_items=[dict(task="待办", assignee="甲", deadline="明天", priority="HIGH_PRIORITY")])
    monkeypatch.setattr(w, "BASE_DIR", tmp_path)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_meeting_summarizer", lambda: summarizer)
    result = w.meeting_summary_node({"user_query": "保留议题及待办", "meeting_record_path": str(selected)})
    assert "FULL_DISCUSSION" in result["final_answer"]
    assert "HIGH_PRIORITY" in result["final_answer"]
    assert result["meeting_record_path"] == str(selected)
    assert "SELECTED_TRANSCRIPT" in result["retrieved_evidence"][0]["text"]
    assert "OTHER_SESSION" not in str(result)


def test_all_specialists_receive_fallback_evidence_and_failures_remain_failures(monkeypatch):
    rag = MagicMock()
    rag.search.return_value = [{"doc_name": "Study", "text": "ACTUAL_PREMISE", "year": 2025}]
    monkeypatch.setattr(w.MasterEngineRegistry, "get_rag_engine", lambda: rag)
    calls = []
    def complete(**kwargs):
        prompt = str(kwargs["messages"]); calls.append(prompt)
        if "特派专家子代理" in prompt:
            assert "ACTUAL_PREMISE" in prompt
            raise RuntimeError("simulated provider failure")
        assert "未完成" in prompt
        assert "已确立基本理论支点" not in prompt
        return "主作者依据现有资料继续回答。"
    monkeypatch.setattr(w, "safe_chat_completion", complete)
    result = w.general_tutor_node({"user_query": "四领域任务", "retrieval_plan": {"local_search": True},
        "intent_decision": {"is_cross_disciplinary": True,
            "sub_queries": [{"discipline": f"d{i}", "query": f"task{i}"} for i in range(4)]}})
    assert len(calls) == 5
    assert len(result["specialist_results"]) == 4
    assert all(item["status"] == "failed" and not item["analysis"] for item in result["specialist_results"])
    assert result["final_answer"]


@pytest.mark.parametrize("fault", ["grounded_error", "unchecked_task"])
def test_inconsistent_review_cannot_claim_pass_but_does_not_block_delivery(monkeypatch, fault):
    payload = {"verdict": "pass", "evidence_status": "supported",
        "task_checks": [{"task_ids": [1, 2] if fault == "grounded_error" else [1],
                         "requirement": "实际要求", "result": "met"}], "findings": []}
    if fault == "grounded_error":
        payload["findings"] = [{"kind": "factual", "severity": "blocking", "reason": "适用范围冲突",
            "basis": [{"source": "evidence", "evidence_id": 1, "quote": "仅限有限规模"}]}]
    monkeypatch.setattr(w, "safe_chat_completion", lambda **kwargs: json.dumps(payload))
    original = "完整原回答。"
    result = w.dialectical_inquiry_node({"user_query": "完成甲乙", "final_answer": original,
        "intent_decision": {"task_segments": [{"task": "甲"}, {"task": "乙"}]},
        "retrieved_evidence": [{"doc_name": "来源", "text": "仅限有限规模"}]})
    assert result["review_verdict"]["verdict"] == "inconclusive"
    assert result["final_answer"] == original
    assert w.reviewer_decision_routing(result) == "tts_synthesis"
    assert not result["review_verdict"]["evidence_requests"]


def test_shared_only_supplement_enters_append_without_external_lookup(monkeypatch):
    source = {"doc_name": "共享来源", "doi": "10.test/shared", "text": "NEW_SHARED_DETAIL"}
    planner = MagicMock(); planner.plan.return_value = RetrievalPlan(selected_shared_item_ids=["shared-1"])
    monkeypatch.setattr(w.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    forbidden = MagicMock(side_effect=AssertionError("No external lookup"))
    monkeypatch.setattr(w.MasterEngineRegistry, "get_hybrid_retriever", forbidden)
    state = {"user_query": "说明机制", "final_answer": "原回答。", "retrieved_evidence": [],
        "shared_pool": [{"kind": "evidence", "item_id": "shared-1", "resource": source}],
        "review_verdict": {"evidence_requests": [{"query": "mechanism", "requirement": "机制"}]}}
    retrieved = w.review_gap_retrieval_node(state)
    assert w.review_gap_result_routing(retrieved) == "evidence_append"
    assert retrieved["pending_evidence_additions"][0]["text"] == source["text"]
    monkeypatch.setattr(w, "safe_chat_completion", lambda **kw: "新增机制说明。[1]")
    result = w.evidence_append_node({**state, **retrieved})
    assert result["final_answer"].startswith(state["final_answer"])
    assert result["final_answer"].endswith("新增机制说明。[1]")
    forbidden.assert_not_called()


def test_source_aliases_merge_new_information_without_losing_pages_or_exclusions():
    prior = [{"doc_name": "Paper", "doi": "10.TEST/paper", "text": "FACT_A", "evidence_level": "metadata", "excluded_as_irrelevant": "旧任务不相关"}]
    merged, delta = merge_evidence(prior, [{"doc_name": "Paper", "url": "https://doi.org/10.test/paper",
        "location_label": "different index", "text": "FACT_A\nFACT_B", "evidence_level": "abstract"}])
    assert len(merged) == 1 and merged[0]["text"] == "FACT_A\nFACT_B"
    assert delta[0]["evidence_id"] == 1 and delta[0]["text"] == "FACT_B"
    assert "excluded_as_irrelevant" not in merged[0]
    assert merged[0]["evidence_level"] == "abstract"
    assert prior[0]["text"] == "FACT_A"  # callers' prior state is not mutated
    pages, _ = merge_evidence([], [{"doc_name": "Book", "doc_type": "pdf", "page_or_slide": i,
        "text": f"page-{i}"} for i in [1, 2]])
    assert len(pages) == 2
    same, delta = merge_evidence(merged, merged)
    assert same == merged and delta == []


def test_fusion_retains_both_online_abstracts_and_prioritizes_local_page():
    base = UnifiedEvidenceCard("Study", text="INDEX_DETAIL", summary="SUMMARY_A", evidence_level="bibliographic_metadata")
    incoming = UnifiedEvidenceCard("Study", text="ABSTRACT_DETAIL", summary="SUMMARY_B", evidence_level="abstract")
    CrossSourceHybridRetriever._merge_two_cards(None, base, incoming)
    assert "INDEX_DETAIL" in base.text and "ABSTRACT_DETAIL" in base.text
    assert "SUMMARY_A" in base.summary and "SUMMARY_B" in base.summary
    assert base.evidence_level == "abstract"
    local = UnifiedEvidenceCard("Study", text="LOCAL_PAGE", location_label="page 9", page_or_slide=9, is_local_indexed=True)
    CrossSourceHybridRetriever._merge_two_cards(None, base, local)
    assert base.text.startswith("LOCAL_PAGE") and "ABSTRACT_DETAIL" in base.text
    assert base.location_label == "page 9" and base.page_or_slide == 9


def test_attachment_grounds_remain_locatable():
    finding = {"kind": "completeness", "reason": "解释图中条件", "basis": [{"source": "visual_context", "quote": "图中条件"}]}
    out = w._locate_review_basis([finding], "请解释图片", "已有回答", [], "图中条件：有限样本")
    assert out[0]["basis_status"] == "located"


def test_truncation_preserves_text_and_reports_incompleteness(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = iter([
        NS(choices=[NS(delta=NS(content="部分内容"), finish_reason=None)]),
        NS(choices=[NS(delta=NS(content=None), finish_reason="length")])])
    text = w.safe_chat_completion(client, "offline", [], max_retries=1)
    assert text == "部分内容" and text.finish_reason == "length"
    monkeypatch.setattr(w, "safe_chat_completion", lambda **kwargs: text)
    result = w.cross_modal_node({"user_query": "解释", "retrieved_evidence": [{"doc_name": "来源", "text": "依据"}]})
    assert result["final_answer"] == text
    assert result["delivery_checks"]["model_completion"] == {"finish_reason": "length", "complete": False}


def test_default_synthesis_uses_production_branch_when_model_is_mocked(monkeypatch):
    monkeypatch.delenv("ENABLE_LLM_EVIDENCE_SYNTHESIS", raising=False)
    forbidden = MagicMock(side_effect=AssertionError("No synthesis model call by default"))
    monkeypatch.setattr(w, "safe_chat_completion", forbidden)
    result = w.evidence_synthesis_node({"user_query": "解释", "final_answer": "原文", "retrieved_evidence": []})
    assert result["synthesized_evidence_slate"]
    forbidden.assert_not_called()


def test_local_fallback_propagates_constraints_without_repeating_initial_lookup(monkeypatch):
    rag = MagicMock(); rag.search.return_value = [
        {"doc_name": "unknown date", "text": "UNDATED"},
        {"doc_name": "old", "year": 2010, "text": "OLD"},
        {"doc_name": "current", "year": 2025, "text": "CURRENT"}]
    state = {"user_query": "解释", "active_doc": "指定教材", "retrieval_plan": {"local_search": True},
        "intent_decision": {"domain": "科学", "negative_constraints": ["排除甲"],
            "time_constraint": {"has_time_constraint": True, "start_year": 2024, "end_year": 2026}}}
    result = w._local_fallback_evidence(state, rag)
    assert [item["text"] for item in result] == ["UNDATED", "CURRENT"]
    assert result[0]["temporal_match"] == "requires_date_verification"
    assert rag.search.call_args.kwargs["doc_filter"] == "指定教材"
    assert rag.search.call_args.kwargs["negative_keywords"] == ["排除甲"]
    assert w._local_fallback_evidence({**state, "retrieval_coverage": {"search_history": [{"query": "已查"}]}}, rag) == []
    rag.search.assert_called_once()


class Response:
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return json.dumps(self.data).encode()


def test_crossref_retains_abstract_and_uses_publication_year(monkeypatch):
    data = {"message": {"items": [{"title": ["Study"], "DOI": "10.test/paper",
        "created": {"date-parts": [[2026]]}, "published": {"date-parts": [[2024, 3, 2]]},
        "abstract": "<jats:p>ACTUAL &amp; ABSTRACT</jats:p>"}]}}
    monkeypatch.setattr(online.urllib.request, "urlopen", lambda *a, **kw: Response(data))
    agent = online.OnlinePaperSearchAgent.__new__(online.OnlinePaperSearchAgent)
    source = agent.search_crossref("exact study")[0]
    assert source.year == 2024 and "ACTUAL & ABSTRACT" in source.summary
    assert source.evidence_level == "abstract"


def test_openalex_reconstructs_abstract_without_claiming_publication_status(monkeypatch):
    monkeypatch.setattr(online, "_OPENALEX_COOLDOWN_UNTIL", 0)
    data = {"results": [{"title": "Study", "id": "test", "publication_year": 2025,
        "abstract_inverted_index": {"reported": [1], "Mechanism": [0], "here": [2]}}]}
    monkeypatch.setattr(online.urllib.request, "urlopen", lambda *a, **kw: Response(data))
    agent = online.OnlinePaperSearchAgent.__new__(online.OnlinePaperSearchAgent)
    source = agent.search_openalex("exact study")[0]
    assert "Mechanism reported here" in source.summary
    assert "权威录用" not in source.summary and source.evidence_level == "abstract"


def test_failed_source_survives_multisource_and_node_state(monkeypatch):
    agent = online.OnlinePaperSearchAgent.__new__(online.OnlinePaperSearchAgent)
    monkeypatch.setattr(agent, "search_arxiv", lambda *a: [])
    monkeypatch.setattr(agent, "search_openalex", lambda *a: [])
    monkeypatch.setattr(online.urllib.request, "urlopen", MagicMock(side_effect=TimeoutError("offline")))
    bundle = agent.search_multisource("study", english_query="study", include_github=False)
    assert bundle["source_errors"]["crossref"] == "TimeoutError"
    monkeypatch.setattr(agent, "search_multisource", lambda *a, **kw: bundle)
    monkeypatch.setattr(agent, "generate_search_brief", lambda **kw: "现有来源不足")
    monkeypatch.setattr(w.MasterEngineRegistry, "get_paper_search_agent", lambda: agent)
    result = w.online_paper_search_node({"user_query": "检索", "retrieval_plan": {"online_search": True}})
    assert result["source_diagnostics"]["errors"]["crossref"] == "TimeoutError"
    assert "source_errors" not in result


def test_stream_callbacks_are_isolated_between_concurrent_requests():
    ready = Barrier(2)
    finished = Barrier(2)
    def request(label):
        received = []
        w.register_stream_callback(received.append)
        try:
            ready.wait(timeout=5)
            w.get_stream_callback()(label)
            finished.wait(timeout=5)
            return received
        finally:
            w.register_stream_callback(None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(request, ["request-A", "request-B"])) == [["request-A"], ["request-B"]]


def test_downloaded_subject_is_consistent_with_shared_task(monkeypatch):
    intent = IntentDecision(domain="研究", intent="下载并分析", target_agent="DocumentSearchAgent",
        subject_action="INIT", subject_entity="初始主题", search_query="Study", need_download_fulltext=True)
    planner = MagicMock(); planner.plan.return_value = RetrievalPlan(local_search=True)
    rag = MagicMock(); rag.embedding_model.encode.return_value = w.np.array([[1.0]])
    rag.search.return_value = [{"doc_name": "study.pdf", "text": "FULL_TEXT", "doc_type": "pdf", "page_or_slide": 1}]
    paper = MagicMock(); paper.download_and_ingest.return_value = {"status": "success", "file_name": "study.pdf"}
    agent = MagicMock(); agent.parse_intent.return_value = intent
    for method, value in [("get_intent_agent", agent), ("get_retrieval_planner", planner),
                          ("get_rag_engine", rag), ("get_paper_search_agent", paper)]:
        monkeypatch.setattr(w.MasterEngineRegistry, method, lambda value=value: value)
    result = w.intent_routing_node({"user_query": "下载并分析这篇论文", "recent_paper_candidates": [
        {"title": "Study", "paper_id": "2501.12345", "summary": "已知摘要", "source": "arxiv"}]})
    assert result["active_subject"] == "《Study》"
    assert result["topic_state"]["current_topic"] == result["active_subject"]
    assert task_payload(result)["active_subject"] == result["active_subject"]
    assert result["task_snapshot"]["original_user_query"] == "下载并分析这篇论文"


def test_missing_selected_meeting_does_not_switch_to_another_record(monkeypatch, tmp_path):
    records = tmp_path / "records"; records.mkdir()
    (records / "record_other.jsonl").write_text('{"text":"OTHER_SESSION"}\n')
    summarizer = MagicMock()
    monkeypatch.setattr(w, "BASE_DIR", tmp_path)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_meeting_summarizer", lambda: summarizer)
    out = w.meeting_summary_node({"user_query": "总结指定会议", "meeting_record_path": str(records / "missing.jsonl")})
    assert out["retrieved_evidence"] == []
    summarizer.summarize_session.assert_not_called()


def test_shared_storage_keeps_shorter_new_passages_and_hydrates_turn_sources(monkeypatch, tmp_path):
    from session_manager import SessionManager, ChatMessage
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="audit-flow")
    source = {"doc_name": "来源", "doi": "10.test/storage", "text": "LONG_PRIOR_PASSAGE" * 20}
    SessionManager.update_shared_resources(session.session_id, [source])
    SessionManager.update_shared_resources(session.session_id, [{**source, "text": "SHORT_NEW_FACT"}])
    resources = SessionManager.get_session(session.session_id).shared_resources
    assert "SHORT_NEW_FACT" in resources[0]["text"] and source["text"] in resources[0]["text"]
    message = ChatMessage(sender="agent", content="GENERATED_PROSE_NOT_A_SOURCE", retrieved_evidence=[source],
                          specialist_results=[{"status": "failed", "sub_query": "task"}], meeting_record_path="record.jsonl")
    SessionManager.add_message(session.session_id, message)
    SessionManager.publish_turn_to_shared_pool(session.session_id, 1, "原始问题", message)
    record = next(item for item in SessionManager.get_shared_pool(session.session_id) if item["kind"] == "turn_record")
    hydrated = next(item for item in SessionManager.get_shared_pool_items(session.session_id, [record["item_id"]]) if item["kind"] == "turn_record")
    assert hydrated["state"]["specialist_results"][0]["status"] == "failed"
    assert hydrated["state"]["meeting_record_path"] == "record.jsonl"
    sources = w._selected_shared_evidence({"session_id": session.session_id}, [record["item_id"]])
    assert sources and all("GENERATED_PROSE_NOT_A_SOURCE" not in item["text"] for item in sources)


@pytest.mark.parametrize("action,expected", [("start_recording", "completed"), ("change_voice", "completed"),
                                            (None, "completed"), ("stop_recording", "shutdown"), ("pause_recording", "shutdown")])
def test_voice_only_stops_for_explicit_stop_actions(monkeypatch, action, expected):
    from voice_langgraph_agent import clean_and_format_node
    monkeypatch.setattr("intent_recognition_agent.IntentRecognitionAgent.parse_intent", lambda *a, **kw:
        NS(target_agent="SystemControlAgent", intent="voice_system_control", control_action=action))
    out = clean_and_format_node({"raw_transcription": "这是系统控制请求"})
    assert out["status"] == expected and out["final_text"]


def test_arxiv_identifier_does_not_turn_request_into_external_evidence(monkeypatch):
    intent_agent, planner, rag = MagicMock(), MagicMock(), MagicMock()
    intent_agent.parse_intent.return_value = IntentDecision(domain="研究", intent="分析论文", target_agent="GeneralTutorAgent")
    planner.plan.return_value = RetrievalPlan(requires_evidence=False)
    rag.search.return_value = []
    monkeypatch.setattr(w.MasterEngineRegistry, "get_intent_agent", lambda: intent_agent)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_rag_engine", lambda: rag)
    query = "分析《Study》(arXiv:2501.12345) 中模型的性能"
    result = w.intent_routing_node({"user_query": query})
    assert result["retrieved_evidence"][0]["text"] == ""
    assert result["retrieved_evidence"][0]["evidence_level"] == "metadata"
    assert result["task_snapshot"]["original_user_query"] == query


def test_partial_specialists_keep_their_content_for_author(monkeypatch):
    def complete(**kwargs):
        prompt = str(kwargs["messages"])
        if "特派专家子代理" in prompt:
            return w.CompletionText("PARTIAL_ANALYSIS", "length")
        assert "PARTIAL_ANALYSIS" in prompt and "未完整结束" in prompt
        return "主作者继续交付。"
    monkeypatch.setattr(w, "safe_chat_completion", complete)
    result = w.general_tutor_node({"user_query": "跨学科分析", "retrieved_evidence": [{"doc_name": "来源", "text": "实际依据"}],
        "intent_decision": {"is_cross_disciplinary": True,
                            "sub_queries": [{"discipline": "甲", "query": "任务甲"}, {"discipline": "乙", "query": "任务乙"}]}})
    assert all(item["status"] == "partial" for item in result["specialist_results"])
    assert result["final_answer"] == "主作者继续交付。"


@pytest.mark.parametrize("reason,cached", [("length", False), ("stop", True)])
def test_vlm_keeps_partial_text_without_persisting_complete_page(monkeypatch, tmp_path, reason, cached):
    import document_rag_engine as rag_module
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://offline.invalid")
    monkeypatch.setattr(rag_module, "EXTRACTED_TEXTS_DIR", tmp_path)
    client = MagicMock()
    client.chat.completions.create.return_value = NS(choices=[NS(
        message=NS(content="EXTRACTED_PAGE_TEXT"), finish_reason=reason)])
    monkeypatch.setattr(rag_module, "OpenAI", lambda **kwargs: client)
    image = tmp_path / "page.png"; image.write_bytes(b"fake image for offline client")
    assert rag_module.UniversalDocumentParser._extract_page_vlm(image, "doc", 1) == "EXTRACTED_PAGE_TEXT"
    assert (tmp_path / "doc_p1.txt").exists() is cached


def test_formula_tool_accepts_provided_premises_without_forcing_search(monkeypatch):
    import mcp_server
    forbidden = MagicMock(side_effect=AssertionError("Provided premises need no fresh lookup"))
    monkeypatch.setattr(w.MasterEngineRegistry, "get_rag_engine", forbidden)
    complete = MagicMock(return_value="依据给定方程的条件推导。[1]")
    monkeypatch.setattr(w, "safe_chat_completion", complete)
    tool = getattr(mcp_server.derive_formula_stepwise, "fn", mcp_server.derive_formula_stepwise)
    result = tool("推导给定方程", reference_text="EXPLICIT_PREMISE")
    assert result and "EXPLICIT_PREMISE" in str(complete.call_args.kwargs["messages"])
    forbidden.assert_not_called()


def test_meeting_failure_keeps_transcript_and_returns_an_honest_status(monkeypatch, tmp_path):
    records = tmp_path / "records"; records.mkdir()
    record = records / "record_selected.jsonl"; record.write_text('{"text":"PRESERVED_TRANSCRIPT"}\n')
    summarizer = MagicMock(); summarizer.summarize_session.side_effect = TimeoutError("offline")
    monkeypatch.setattr(w, "BASE_DIR", tmp_path)
    monkeypatch.setattr(w.MasterEngineRegistry, "get_meeting_summarizer", lambda: summarizer)
    result = w.meeting_summary_node({"user_query": "总结会议", "meeting_record_path": str(record)})
    assert result["status"] == "meeting_summary_unavailable"
    assert "PRESERVED_TRANSCRIPT" in result["retrieved_evidence"][0]["text"]
    assert "未完成" in result["final_answer"]
