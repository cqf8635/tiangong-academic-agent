import json
from unittest.mock import MagicMock

import pytest

from master_agent_workflow import (
    MasterEngineRegistry,
    dialectical_inquiry_node,
    general_tutor_node,
    review_gap_retrieval_node,
    review_gap_result_routing,
    reflective_revision_node,
    reviewer_decision_routing,
)
from retrieval_planner import RetrievalPlan
from prompts import format_evidence_for_prompt


@pytest.mark.parametrize("stage", ["author", "specialist", "reviewer", "revision"])
def test_all_stages_receive_late_evidence_with_complete_text_and_stable_ids(monkeypatch, stage):
    evidence = [{"doc_name": f"Source {i}", "text": f"Inspected fact {i}",
                 "doc_type": "arxiv", "evidence_level": "abstract",
                 "url": f"https://example.org/paper/{i}"} for i in range(1, 57)]
    evidence[50]["text"] = "Full abstract: " + "Mechanism and conditions. " * 80 + "END: measured latency decreases."
    evidence[50]["summary"] = "Supplementary reported result: reward-model training."
    evidence[50]["arxiv_id"] = "test-source-51"
    evidence[51]["text"] = "Full abstract: " + "Joint training and serving. " * 80 + "END: throughput increases."
    formatted = format_evidence_for_prompt(evidence)
    calls = []
    draft = "保留必要定义、公式、完整推导和讨论。" * 300

    def complete(**kwargs):
        calls.append(kwargs["messages"])
        if stage == "reviewer":
            return json.dumps({
                "verdict": "pass", "evidence_status": "supported", "findings": [],
                "task_checks": [{"requirement": "比较后续机制", "result": "met"}],
            })
        return draft

    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    network = MagicMock(side_effect=AssertionError("Existing evidence must be reused"))
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", network)
    state = {
        "user_query": "比较后续模型机制与报告结果，保留详细推导",
        "target_agent": "GeneralTutorAgent", "intent_decision": {},
        "retrieval_plan": {"requires_evidence": True, "online_search": True},
        "retrieved_evidence": evidence, "final_answer": draft,
    }
    if stage == "specialist":
        monkeypatch.setattr("master_agent_workflow.time.sleep", lambda _: None)
        state["intent_decision"] = {"is_cross_disciplinary": True, "sub_queries": [
            {"discipline": "modeling", "query": "compare mechanisms"},
            {"discipline": "evaluation", "query": "compare measurements"},
        ]}
    if stage == "revision":
        state["review_verdict"] = {"needs_revision": True, "findings": [{
            "severity": "blocking", "location": "后续模型比较段",
            "reason": "遗漏证据[51]和[52]明确报告的机制与效果",
            "suggestion": "依照原始摘要补充，保留其余推导",
        }]}
        state["pending_evidence_additions"] = [{**ev, "evidence_id": i} for i, ev in enumerate(evidence, 1)]
        result = reflective_revision_node(state)
    elif stage == "reviewer":
        result = dialectical_inquiry_node(state)
    else:
        result = general_tutor_node(state)

    assert len(calls) == (3 if stage == "specialist" else 1)
    for messages in calls:
        prompt = "\n".join(message["content"] for message in messages)
        assert formatted in prompt
        assert "[51] Source 51" in prompt and "[52] Source 52" in prompt and "[56] Source 56" in prompt
        assert evidence[50]["text"] in prompt
        assert evidence[50]["summary"] in prompt
        assert evidence[51]["text"] in prompt
    assert result["final_answer"] == draft
    assert result["retrieved_evidence"] == evidence
    network.assert_not_called()


def test_tutor_online_fallback_preserves_all_returned_unique_abstracts(monkeypatch):
    papers = [{"title": f"Study {chr(65 + i // 26)}{chr(65 + i % 26)}", "doi": f"10.test/{i}",
               "summary": "Reported mechanism. " * 80 + f"MEASUREMENT-{i}", "source": "crossref"}
              for i in range(30)]
    # Distinct DOIs establish identity; disable fuzzy matching of synthetic titles.
    monkeypatch.setattr("cross_source_hybrid_retriever.calculate_title_similarity", lambda *_: 0)
    paper_agent = MagicMock()
    paper_agent.search_multisource.return_value = {"crossref_papers": papers}
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", lambda: paper_agent)
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    complete = MagicMock(return_value="根据资料比较。")
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    result = general_tutor_node({
        "user_query": "比较检索到的模型", "intent_decision": {},
        "retrieval_plan": {"online_search": True},
    })
    prompt = complete.call_args.kwargs["messages"][-1]["content"]
    assert len(result["retrieved_evidence"]) == len(papers)
    for paper in papers:
        assert paper["summary"] in prompt
    paper_agent.search_multisource.assert_called_once()
    paper_agent.search_arxiv.assert_not_called()


def test_actionable_review_reports_findings_without_rewriting(monkeypatch):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    draft = "方法甲的完整论述。" * 300
    payload = {"verdict": "revise", "needs_revision": True, "evidence_status": "limited",
               "findings": [{"severity": "blocking", "location": "比较段", "reason": "第二种方法尚未展开"}],
               "task_checks": [{"requirement": "比较甲乙", "result": "missing"}]}
    complete = MagicMock(return_value=json.dumps(payload))
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    state = {"user_query": "比较甲乙", "final_answer": draft}
    reviewed = dialectical_inquiry_node(state)
    assert reviewed["review_verdict"]["needs_revision"] is False
    assert reviewed["review_verdict"]["findings"]
    assert reviewer_decision_routing(reviewed) == "tts_synthesis"
    preserved = reflective_revision_node({**state, **reviewed})
    assert preserved["final_answer"] == draft
    assert complete.call_count == 1



def test_review_pass_routes_to_delivery(monkeypatch):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    payload = {
        "verdict": "pass", "needs_revision": False, "findings": [],
        "evidence_status": "supported",
        "task_checks": [{"requirement": "解释定义", "result": "met", "note": "直接回答了用户问题"}],
        "evidence_requests": [], "summary": "覆盖了本轮要求。",
    }
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(payload, ensure_ascii=False))
    result = dialectical_inquiry_node({
        "user_query": "解释这个定义", "target_agent": "StudyPlannerAgent",
        "final_answer": "这个定义表示……", "retrieved_evidence": [{"doc_name": "教材", "location_label": "第1页", "text": "这个定义表示……"}], "agent_dialogue_history": [],
    })
    assert result["review_verdict"]["verdict"] == "pass"
    assert reviewer_decision_routing({"review_verdict": result["review_verdict"]}) == "tts_synthesis"


def test_review_stops_after_one_revision_and_surfaces_unresolved_finding(monkeypatch):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    payload = {
        "verdict": "revise", "needs_revision": True,
        "findings": [{"severity": "blocking", "location": "结论", "reason": "关键条件仍不明确", "suggestion": "补充条件"}],
        "dimensions": {}, "summary": "仍有一项未解决。",
    }
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(payload, ensure_ascii=False))
    at_revision_limit = {"review_verdict": {"stage": "final", "needs_revision": True, "findings": [{}]}, "revision_count": 1}
    assert reviewer_decision_routing(at_revision_limit) == "tts_synthesis"

    terminal = dialectical_inquiry_node({
        "user_query": "说明该方法的边界", "target_agent": "GeneralTutorAgent",
        "final_answer": "方法适用于该情况。", "retrieved_evidence": [],
        "agent_dialogue_history": [], "revision_count": 1,
    })
    assert terminal["status"] == "review_inconclusive"
    assert "关键条件仍不明确" in terminal["review_notice"]
    assert terminal["final_answer"] == "方法适用于该情况。"


def test_audit_pass_requires_task_checks_and_minor_findings_do_not_force_revision(monkeypatch):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    payload = {
        "verdict": "pass", "needs_revision": False,
        "findings": [{"severity": "minor", "location": "开头", "reason": "可更简洁", "suggestion": "可选优化"}],
        "task_checks": [{"requirement": "比较两种方法", "result": "missing", "note": "只说了第一种"}],
        "evidence_requests": [], "summary": "存在漏答。",
    }
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(payload, ensure_ascii=False))
    result = dialectical_inquiry_node({
        "user_query": "比较方法甲和乙", "target_agent": "GeneralTutorAgent",
        "final_answer": "方法甲……", "retrieved_evidence": [],
        "agent_dialogue_history": [], "revision_count": 0,
    })
    assert result["review_verdict"]["verdict"] == "inconclusive"
    assert result["review_verdict"]["task_checks"][0]["note"] == "只说了第一种"
    assert reviewer_decision_routing({"review_verdict": result["review_verdict"], "revision_count": 0}) == "tts_synthesis"

    minor_only = {
        "verdict": "revise", "needs_revision": True,
        "evidence_status": "supported",
        "findings": [{"severity": "minor", "location": "第一段", "reason": "表达可以更简洁", "suggestion": "可选调整"}],
        "task_checks": [{"requirement": "解释概念", "result": "met", "note": "已解释"}],
        "evidence_requests": [], "summary": "只有风格建议。",
    }
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(minor_only, ensure_ascii=False))
    minor_result = dialectical_inquiry_node({
        "user_query": "解释概念", "target_agent": "GeneralTutorAgent",
        "final_answer": "概念的意思是……", "retrieved_evidence": [{"doc_name": "教材", "text": "概念的意思是……"}],
        "agent_dialogue_history": [], "revision_count": 0,
    })
    assert minor_result["review_verdict"]["verdict"] == "inconclusive"
    assert minor_result["review_verdict"]["needs_revision"] is False
    assert reviewer_decision_routing({"review_verdict": minor_result["review_verdict"], "revision_count": 0}) == "tts_synthesis"


def test_reviewer_requests_evidence_then_planner_selects_and_runs_source(monkeypatch):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    payload = {
        "verdict": "revise", "needs_revision": True,
        "findings": [{"kind": "evidence", "severity": "material", "location": "稳定性结论", "reason": "缺少验证依据", "suggestion": "限定表述",
                      "basis": [{"source": "answer", "quote": "方法甲在该规模下稳定"}]}],
        "task_checks": [{"requirement": "说明适用条件", "result": "unclear", "note": "尚无来源支持"}],
        "evidence_requests": [{"query": "方法甲在该数据规模下的稳定性实证结果", "purpose": "verify_claim", "finding_ids": [1], "reason": "现有资料没有验证该结论", "evidence_type": "empirical result"}],
        "summary": "需要核实一项经验性主张。",
    }
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(payload, ensure_ascii=False))
    initial = dialectical_inquiry_node({
        "user_query": "说明方法甲在该数据规模下的稳定性", "target_agent": "GeneralTutorAgent",
        "final_answer": "方法甲在该规模下稳定。", "retrieved_evidence": [],
        "agent_dialogue_history": [], "revision_count": 0, "review_retrieval_rounds": 0,
    })
    state = {"review_retrieval_rounds": 0, "review_retrieval_status": "ready", **initial}
    assert reviewer_decision_routing(state) == "review_gap_retrieval"

    planner = MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True, reasoning="外部实证资料缺口")
    retriever = MagicMock()
    retriever.hybrid_search_and_deduplicate.return_value = [{
        "doc_name": "Validation study", "doc_type": "crossref", "text": "Results under matching conditions",
        "url": "https://example.org/study", "evidence_level": "abstract",
        "location_label": "CrossRef",
    }]
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = review_gap_retrieval_node({
        "user_query": "说明方法甲在该数据规模下的稳定性",
        "intent_decision": {"intent": "evaluate stability", "task_segments": []},
        "review_verdict": initial["review_verdict"],
        "review_retrieval_rounds": 0,
        "retrieved_evidence": [],
        "agent_dialogue_history": [],
        "review_retrieval_log": [],
    })
    assert result["review_retrieval_status"] == "retrieved"
    assert result["review_retrieval_rounds"] == 1
    assert result["retrieved_evidence"][0]["doc_name"] == "Validation study"
    assert retriever.hybrid_search_and_deduplicate.call_args.kwargs["include_online"] is True
    assert review_gap_result_routing(result) == "evidence_append"


def _run_evidence_review(monkeypatch, payload, **state):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda **kwargs: json.dumps(payload, ensure_ascii=False))
    return dialectical_inquiry_node({
        "user_query": "说明方法甲的适用条件", "target_agent": "GeneralTutorAgent",
        "final_answer": "方法甲适用于所有规模。", "retrieved_evidence": [],
        "agent_dialogue_history": [], "revision_count": 0,
        **state,
    })


def test_model_agreement_without_sources_cannot_pass_professional_review(monkeypatch):
    result = _run_evidence_review(monkeypatch, {
        "verdict": "pass", "evidence_status": "supported", "findings": [],
        "task_checks": [{"requirement": "说明适用条件", "result": "met", "note": "符合模型知识"}],
    })
    assert result["review_verdict"]["evidence_status"] == "unsupported"
    assert result["review_verdict"]["needs_revision"] is False
    assert reviewer_decision_routing(result) == "tts_synthesis"


def test_missing_source_assessment_cannot_pass_professional_review(monkeypatch):
    result = _run_evidence_review(monkeypatch, {
        "verdict": "pass", "findings": [],
        "task_checks": [{"requirement": "说明适用条件", "result": "met"}],
    }, retrieved_evidence=[{"doc_name": "教材", "text": "仅在有限规模成立"}])
    assert result["review_verdict"]["verdict"] == "inconclusive"


def test_honest_evidence_limit_and_ordinary_conversation_can_be_delivered(monkeypatch):
    for status, plan, answer in [
        ("limited", {"requires_evidence": True}, "尚无资料核实适用条件，不能确认其在所有规模下成立。"),
        ("not_required", {"requires_evidence": False}, "你好。"),
    ]:
        result = _run_evidence_review(monkeypatch, {
            "verdict": "pass", "evidence_status": status, "findings": [],
            "task_checks": [{"requirement": "按实际可支持范围回应", "result": "met"}],
        }, retrieval_plan=plan, final_answer=answer)
        assert result["review_verdict"]["verdict"] == "pass"
        assert result["review_verdict"]["needs_revision"] is False


def test_evidence_gap_at_limit_is_unresolved_even_if_model_says_pass(monkeypatch):
    monkeypatch.setenv("REVIEW_EVIDENCE_GAP_ROUNDS", "1")
    result = _run_evidence_review(monkeypatch, {
        "verdict": "pass", "evidence_status": "limited", "findings": [{"kind": "evidence", "reason": "规模主张待核实",
            "basis": [{"source": "answer", "quote": "适用于所有规模"}]}],
        "task_checks": [{"requirement": "说明适用条件", "result": "met"}],
        "evidence_requests": [{"query": "方法甲的适用规模", "purpose": "verify_claim", "finding_ids": [1]}],
    }, review_retrieval_rounds=1)
    assert result["status"] == "review_unresolved"
    assert result["review_verdict"]["verdict"] != "pass"
    assert "证据缺口" in result["review_notice"]


def test_review_gap_preserves_time_and_exclusions(monkeypatch):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(local_search=True)
    retriever.hybrid_search_and_deduplicate.return_value = []
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    time_range = {"has_time_constraint": True, "start_year": 2024, "end_year": 2025}
    review_gap_retrieval_node({
        "user_query": "只看2024至2025年的正式论文", "review_verdict": {"evidence_requests": [{"query": "验证结果"}]},
        "intent_decision": {"time_constraint": time_range, "negative_constraints": ["预印本"]},
        "shared_pool": [{"item_id": "prior"}],
    })
    kwargs = retriever.hybrid_search_and_deduplicate.call_args.kwargs
    assert kwargs["time_constraint"] == time_range
    assert kwargs["negative_keywords"] == ["预印本"]
    assert planner.plan.call_args.kwargs["state_context"]["shared_pool"] == [{"item_id": "prior"}]


def test_review_does_not_repeat_router_gap_search(monkeypatch):
    retriever, planner = MagicMock(), MagicMock()
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    result = review_gap_retrieval_node({
        "user_query": "核实后续推荐模型",
        "review_verdict": {"evidence_requests": [{"query": "citation OneTrans"}]},
        "retrieval_coverage": {"search_history": [{"query": "OneTrans citation", "stage": "gap"}]},
    })
    assert result["review_retrieval_status"] == "unresolved"
    assert result["review_retrieval_log"][-1]["result"] == "already_searched"
    assert reviewer_decision_routing(result) == "tts_synthesis"
    planner.plan.assert_not_called()
    retriever.hybrid_search_and_deduplicate.assert_not_called()


def test_review_empty_search_reassesses_instead_of_ending(monkeypatch):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True)
    retriever.hybrid_search_and_deduplicate.return_value = []
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = review_gap_retrieval_node({
        "user_query": "验证事实", "review_verdict": {"evidence_requests": [{"query": "initial lookup"}]},
    })
    assert result["review_retrieval_status"] == "reassess"
    assert review_gap_result_routing(result) == "dialectical_inquiry"


def test_review_checks_other_requirements_when_first_lookup_is_empty(monkeypatch):
    planner, retriever = MagicMock(), MagicMock()
    planner.plan.return_value = RetrievalPlan(online_search=True)
    retriever.hybrid_search_and_deduplicate.side_effect = [[], [{"doc_name": "evidence", "text": "fact"}]]
    monkeypatch.setattr(MasterEngineRegistry, "get_retrieval_planner", lambda: planner)
    monkeypatch.setattr(MasterEngineRegistry, "get_hybrid_retriever", lambda: retriever)
    result = review_gap_retrieval_node({
        "user_query": "核实两项要求", "review_verdict": {"evidence_requests": [
            {"query": "first requirement"}, {"query": "second requirement"},
        ]},
    })
    assert retriever.hybrid_search_and_deduplicate.call_count == 2
    assert result["review_retrieval_status"] == "retrieved"
    assert len(result["review_retrieval_log"]) == 2
    assert result["review_retrieval_rounds"] == 1


def test_review_does_not_hide_later_retrieved_evidence(monkeypatch):
    captured = {}
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    def complete(**kwargs):
        captured.update(kwargs)
        return json.dumps({"verdict": "pass", "evidence_status": "supported", "task_checks": [
            {"requirement": "verify fact", "result": "met"},
        ]})
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    dialectical_inquiry_node({
        "user_query": "核实事实", "final_answer": "结果", "retrieved_evidence": [
            {"doc_name": f"paper {i}", "text": "necessary context " * 90} for i in range(12)
        ] + [{"doc_name": "late decisive source", "text": "DECISIVE_VERIFIED_FACT"}],
    })
    assert "DECISIVE_VERIFIED_FACT" in captured["messages"][0]["content"]


def test_review_progress_can_continue_beyond_one_round_and_stop_when_stalled(monkeypatch):
    monkeypatch.delenv("REVIEW_EVIDENCE_GAP_ROUNDS", raising=False)
    check = {"requirement": "verify fact", "result": "unclear"}
    state = {
        "review_verdict": {"evidence_requests": [{"query": "new fact", "requirement": "verify fact", "strategy": "exact citation lookup"}],
                           "task_checks": [check]},
        "review_retrieval_status": "ready", "review_retrieval_rounds": 3,
        "review_retrieval_log": [
            {"result": "evidence_added", "requirement": "verify fact", "strategy": "broad lookup", "task_checks": [check]},
            {"result": "evidence_added", "requirement": "verify fact", "strategy": "author lookup", "task_checks": [check]},
        ],
    }
    assert reviewer_decision_routing(state) == "review_gap_retrieval"
    state["review_verdict"]["evidence_requests"][0]["strategy"] = "author lookup"
    assert reviewer_decision_routing(state) == "tts_synthesis"


@pytest.mark.parametrize("answer", ["完整回答。", "必要的推导与假设说明。" * 400])
def test_advisory_review_preserves_exact_answer_without_second_pass(monkeypatch, answer):
    monkeypatch.setenv("ENABLE_SECOND_PASS_REVISION", "true")
    llm_call = MagicMock(side_effect=AssertionError("风格建议不应调用重写"))
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", llm_call)
    state = {
        "final_answer": answer, "revision_count": 0,
        "dialectical_notes": "建议全文压缩并改写得更优雅。",
        "review_verdict": {"needs_revision": True, "findings": [{
            "severity": "minor", "location": "全文", "reason": "可以更简洁", "suggestion": "压缩全文",
        }]},
    }
    assert reviewer_decision_routing(state) == "tts_synthesis"
    result = reflective_revision_node(state)
    assert result["status"] == "evidence_append_skipped"
    assert result["final_answer"] == answer
    assert result.get("revision_count", 0) == 0
    llm_call.assert_not_called()


def test_new_evidence_only_adds_material_without_rewriting_the_draft(monkeypatch):
    draft = "定义与完整证明。" * 500 + "末尾的反例与边界必须保留。"
    source = {"doc_name": "New evidence", "text": "Reported finite-scale condition"}
    complete = MagicMock(return_value="补充：该结果适用于有限规模。[1]")
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    result = reflective_revision_node({
        "user_query": "保留完整证明", "final_answer": draft, "retrieved_evidence": [source],
        "pending_evidence_additions": [{**source, "evidence_id": 1}],
        "review_verdict": {"needs_revision": True, "findings": [{"suggestion": "删除反例并压缩全文"}]},
    })
    prompt = complete.call_args.kwargs["messages"][0]["content"]
    assert draft in prompt and source["text"] in prompt
    assert "删除反例并压缩全文" not in prompt
    assert result["final_answer"].startswith(draft)
    assert result["final_answer"][len(draft):].endswith("补充：该结果适用于有限规模。[1]")
    assert result["pending_evidence_additions"] == []



def test_review_receives_current_task_context_and_complete_answer(monkeypatch):
    draft = "必要推导。" * 800 + "最终边界结论。"
    captured = {}

    def review(**kwargs):
        captured.update(kwargs)
        return json.dumps({
            "verdict": "pass", "evidence_status": "supported", "findings": [],
            "task_checks": [{"task_ids": [1], "requirement": "完整推导并说明边界", "result": "met", "note": "教材第3节支持"}],
        })

    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", review)
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    result = dialectical_inquiry_node({
        "user_query": "比较乙与丙并完整推导", "active_subject": "早先的甲方法",
        "active_doc": "指定教材.pdf", "history_summary": "上一轮用甲作基准。",
        "intent_decision": {"task_segments": [{"task": "比较乙与丙"}], "negative_constraints": ["不引入丁方法"]},
        "final_answer": draft, "retrieved_evidence": [{"doc_name": "指定教材", "text": "前提与公式", "location_label": "第3节"}],
    })
    prompt = captured["messages"][0]["content"]
    for content in (draft, "比较乙与丙", "不引入丁方法", "指定教材.pdf", "上一轮用甲作基准。"):
        assert content in prompt
    assert result["final_answer"] == draft
    assert result["review_verdict"]["verdict"] == "pass"
    assert reviewer_decision_routing(result) == "tts_synthesis"


@pytest.mark.parametrize("location,reason", [
    ("任务比较段", "用户要求比较甲乙，正文却只讨论丙丁"),
    ("后半段扩展", "擅自增加硬件采购任务并代替用户要求的推导"),
    ("公式推导第3步", "由有限规模定理直接推出无限规模结论，缺失极限条件"),
])
def test_concrete_task_scope_and_theory_problems_never_rewrite(monkeypatch, location, reason):
    result = _run_evidence_review(monkeypatch, {
        "verdict": "revise", "evidence_status": "limited",
        "task_checks": [{"requirement": "完成指定比较与推导", "result": "met"}],
        "findings": [{"severity": "blocking", "location": location, "reason": reason, "suggestion": "修正该处并保留其余内容"}],
    })
    assert result["review_verdict"]["needs_revision"] is False
    assert reviewer_decision_routing(result) == "tts_synthesis"


@pytest.mark.parametrize("response", [RuntimeError("模拟失败"), "", "【搭档提示】模型服务繁忙"])
def test_append_failure_or_empty_result_preserves_the_draft(monkeypatch, response):
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: MagicMock())
    complete = MagicMock(side_effect=response) if isinstance(response, Exception) else MagicMock(return_value=response)
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", complete)
    source = {"doc_name": "New source", "text": "New fact"}
    state = {"final_answer": "保留完整初稿与限定条件。", "retrieved_evidence": [source],
             "pending_evidence_additions": [{**source, "evidence_id": 1}]}
    result = reflective_revision_node(state)
    assert result["final_answer"] == state["final_answer"]
    assert result.get("revision_count", 0) == 0
    assert result["status"] in {"evidence_append_failed", "evidence_append_skipped"}

