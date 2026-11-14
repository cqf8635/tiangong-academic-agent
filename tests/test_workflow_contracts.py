"""Workflow-owned invariants across interpretation, authors, review and persistence."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import master_agent_workflow as workflow
from intent_recognition_agent import IntentDecision
from meeting_summarizer import MeetingSummarizer
from online_paper_search_agent import OnlinePaperSearchAgent
from retrieval_planner import RetrievalPlan, RetrievalPlannerAgent
from session_manager import ChatMessage, SessionManager
from workflow_contracts import build_task_snapshot, citation_integrity, task_decision, task_payload


def make_state():
    original = "分析材料老化的已有研究，并讨论下一步怎么验证，不需要代码。"
    accepted = {"domain": "材料学", "intent": "研究进展与验证方案",
                "task_segments": [{"task": "梳理研究"}, {"task": "讨论验证方案"}],
                "negative_constraints": ["代码"]}
    return {"user_query": original, "task_snapshot": build_task_snapshot(
        original, "", {**accepted, "intent": "编写程序"}, accepted, "材料老化",
        {"status": "corrected", "issues": ["原话排除代码，同时要求验证方案"]}),
        "intent_decision": {"intent": "后续节点错误重建的任务"},
        "retrieved_evidence": [{"doc_name": "实验报告", "doi": "10.test/aging", "text": "湿度影响老化速率"}]}


def test_task_handoff_copies_and_change_grounds_are_not_promoted_to_verified_truth():
    state = make_state()
    before = deepcopy(state)
    worker_copy = task_decision(state)
    worker_copy["task_segments"][0]["task"] = "改写需求"
    assert state == before
    assert task_payload(state)["interpreted_task"]["intent"] == "研究进展与验证方案"
    change = state["task_snapshot"]["interpretation_log"][1]
    assert change["changes"]["intent"] == {"before": "编写程序", "after": "研究进展与验证方案"}
    assert change["grounds_type"] == "model_assessment"
    assert change["semantic_correctness_verified"] is False


def test_image_transcription_does_not_mutate_literal_request_or_previous_turn(monkeypatch):
    parser, planner = MagicMock(), MagicMock()
    parser.parse_intent.return_value = IntentDecision(
        domain="材料学", intent="解释图中的结果", target_agent="GeneralTutorAgent")
    planner.plan.return_value = RetrievalPlan(requires_evidence=False)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: parser)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr(workflow, "extract_visual_content_from_image", lambda *args: "图中文字与测量数据")
    state = {"user_query": "  解释此图\n", "user_image": "data:image/png;base64,example",
             "task_snapshot": {"original_user_query": "上轮问题"}}
    before = deepcopy(state)
    result = workflow.intent_routing_node(state)
    assert state == before
    assert parser.parse_intent.call_args.kwargs["state"]["user_query"].endswith("图中文字与测量数据")
    assert result["task_snapshot"]["original_user_query"] == "  解释此图\n"
    assert result["task_snapshot"]["visual_context"] == "图中文字与测量数据"
    assert "上轮问题" not in json.dumps(result["task_snapshot"], ensure_ascii=False)


def test_calendar_policy_survives_alignment_and_is_logged_separately(monkeypatch):
    parser, planner = MagicMock(), MagicMock()
    parser.parse_intent.return_value = IntentDecision(
        domain="材料学", intent="梳理进展", target_agent="GeneralTutorAgent",
        time_constraint={"basis": "default_three_years"})
    planner.plan.return_value = RetrievalPlan(requires_evidence=False, task_alignment={
        "status": "corrected", "issues": ["未明确时间，不应添加模型推断年份"],
        "corrected_fields": {"time_constraint": None}})
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_intent_agent", lambda: parser)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    result = workflow.intent_routing_node({"user_query": "梳理材料老化的研究进展"})
    assert result["intent_decision"]["time_constraint"]["basis"] == "default_three_years"
    log = result["task_snapshot"]["interpretation_log"]
    assert log[1]["changes"]["time_constraint"]["after"] is None
    assert log[2]["grounds_type"] == "system_policy"
    assert log[2]["changes"]["time_constraint"]["after"]["basis"] == "default_three_years"


@pytest.mark.parametrize("node", [workflow.general_tutor_node, workflow.document_rag_node])
def test_local_authors_receive_same_accepted_task_and_complete_evidence(monkeypatch, node):
    completion = MagicMock(return_value="研究依据与验证思路。[1]")
    monkeypatch.setattr(workflow, "safe_chat_completion", completion)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_rag_engine", lambda: MagicMock())
    state = make_state()
    node(state)
    sent = "\n".join(m["content"] for m in completion.call_args.kwargs["messages"])
    assert state["user_query"] in sent
    assert "研究进展与验证方案" in sent
    assert "后续节点错误重建的任务" not in sent
    assert "[1] 实验报告" in sent and "湿度影响老化速率" in sent
    completion.assert_called_once()


def test_review_reports_program_checks_without_rewriting_or_automatic_search(monkeypatch):
    state = {**make_state(), "final_answer": "研究讨论。[99] 还需验证。",
             "delivery_checks": {"original_answer_preserved": True}}
    response = {"verdict": "pass", "needs_revision": True, "evidence_status": "supported",
                "task_checks": [{"requirement": "研究与验证", "result": "met"}]}
    completion = MagicMock(return_value=json.dumps(response))
    monkeypatch.setattr(workflow, "safe_chat_completion", completion)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    result = workflow.dialectical_inquiry_node(state)
    sent = completion.call_args.kwargs["messages"][0]["content"]
    assert "研究进展与验证方案" in sent
    assert '"unknown_ids": [99]' in sent
    assert result["delivery_checks"]["citation_integrity"]["unknown_ids"] == [99]
    assert result["delivery_checks"]["original_answer_preserved"] is True
    assert result["final_answer"] == state["final_answer"]
    assert result["review_verdict"]["needs_revision"] is False
    assert result["review_verdict"]["evidence_requests"] == []
    completion.assert_called_once()


def test_supplement_planning_is_subordinate_and_cannot_change_main_task(monkeypatch):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, queries=[{"query": "humidity aging test"}],
        task_alignment={"status": "corrected", "corrected_fields": {"intent": "错误的新任务"}})
    retriever.hybrid_search_and_deduplicate.return_value = []
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(workflow.MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    state = {**make_state(), "review_verdict": {"evidence_requests": [{
        "query": "humidity aging test", "requirement": "验证湿度影响"}]}}
    before = deepcopy(state)
    result = workflow.review_gap_retrieval_node(state)
    context = planner.plan.call_args.kwargs["state_context"]
    assert context["shared_task"] == task_payload(state)
    assert context["evidence_gap"]["requirement"] == "验证湿度影响"
    assert state == before
    assert "task_snapshot" not in result and "intent_decision" not in result


def test_citation_checks_distinguish_ids_read_scope_and_exclusions():
    checks = citation_integrity(
        "结论。[1, 2–4] 不存在。[9] `a[88]` $x[77]$ \\(x[66]\\) "
        "[55](https://example.com)\n```python\na[44]\n```",
        [{"text": "已读"}, {"text": "无关", "excluded_as_irrelevant": "不匹配"},
         {"doc_name": "仅有标识"}, {"summary": "已读摘要"}],
    )
    assert checks["unknown_ids"] == [9]
    assert checks["excluded_ids"] == [2]
    assert checks["unread_ids"] == [3]
    assert checks["requires_semantic_review"] is True


def test_workflow_state_roundtrips_as_historical_context_only(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="workflow")
    state = make_state()
    message = ChatMessage(sender="agent", content="完整讨论", task_snapshot=state["task_snapshot"],
                          delivery_checks={"original_answer_preserved": True})
    SessionManager.add_message(session.session_id, message)
    pool = SessionManager.publish_turn_to_shared_pool(session.session_id, 1, state["user_query"], message)
    turn_id = next(item["item_id"] for item in pool if item["kind"] == "turn_record")
    historical = next(item for item in SessionManager.get_shared_pool_items(session.session_id, [turn_id])
                      if item["kind"] == "turn_record")
    assert historical["state"]["task_snapshot"] == state["task_snapshot"]
    assert historical["state"]["delivery_checks"] == message.delivery_checks
    assert all("state" not in item for item in SessionManager.get_shared_pool(session.session_id, include_content=False))
    assert ChatMessage(sender="agent", content="旧记录").task_snapshot is None


def test_planner_removes_invented_shared_ids_without_an_extra_call():
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test"
    create = MagicMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
        "requires_evidence": False, "selected_shared_item_ids": ["known", "invented", "known"]}))) ]))
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    plan = planner.plan("整理笔记", {}, {"shared_pool": [{"item_id": "known", "kind": "turn_record"}]})
    assert plan.selected_shared_item_ids == ["known"]
    create.assert_called_once()


def test_online_author_uses_same_task_policy_without_research_template_inflation(monkeypatch):
    state = make_state()
    agent = OnlinePaperSearchAgent.__new__(OnlinePaperSearchAgent)
    agent.llm_client = MagicMock()
    completion = MagicMock(return_value="已有成果与验证方案。[1]")
    monkeypatch.setattr(workflow, "safe_chat_completion", completion)
    agent.generate_search_brief(query=state["user_query"], papers=[], evidence=state["retrieved_evidence"],
                                shared_task=task_payload(state))
    sent = completion.call_args.kwargs["messages"][0]["content"]
    assert "研究进展与验证方案" in sent
    assert "[1] 实验报告" in sent
    assert "湿度影响老化速率" in sent
    assert "万字级" not in sent and "必须深入展开" not in sent
    completion.assert_called_once()


def test_meeting_map_and_reduce_share_task_without_extra_calls(tmp_path):
    summarizer = MeetingSummarizer.__new__(MeetingSummarizer)
    summarizer.model = "test"
    records = [{"text": "湿度实验方案", "timestamp": "00:01"},
               {"text": "下一步测量方案", "timestamp": "00:02"}]
    summarizer._read_records = lambda path: records
    def response(content):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    create = MagicMock(side_effect=[response('{"summary":"实验"}'), response('{"summary":"测量"}'),
        response(json.dumps({"title": "验证讨论", "overview": "实验与测量", "key_decisions": [],
                             "discussion_topics": [], "action_items": []}))])
    summarizer.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    summarizer.summarize_session(tmp_path / "records.jsonl", chunk_sentences=1,
                                task_context="只提取验证方案，保留测量条件。")
    assert create.call_count == 3  # Two existing map calls and one reduce call.
    assert all("只提取验证方案，保留测量条件。" in call.kwargs["messages"][-1]["content"]
               for call in create.call_args_list)
