import json
import time
from types import SimpleNamespace

from retrieval_planner import RetrievalPlan, RetrievalPlannerAgent, build_retrieval_probes, build_retrieval_queries, select_retrieval_probes
from online_paper_search_agent import (
    _build_arxiv_search_expression,
    _build_github_repository_query,
    _extract_academic_phrases,
)


def test_query_views_keep_raw_text_and_every_semantic_segment():
    raw = "分析RAG在多跳问答里的失败原因给出评测指标再说怎么改进"
    intent = {
        "search_query": "RAG多跳问答失败、指标和改进",
        "task_segments": [
            {"retrieval_query": "RAG multi-hop failure modes"},
            {"retrieval_query": "multi-hop QA evaluation metrics"},
            {"retrieval_query": "RAG multi-hop improvements"},
        ],
        "sub_queries": [
            {"query": "RAG retrieval recall in multi-hop QA", "english_query": "RAG multi-hop retrieval recall"},
            {"query": "RAG generation faithfulness in multi-hop QA", "english_query": "RAG multi-hop generation faithfulness"},
        ],
    }
    queries = build_retrieval_queries(raw, intent)
    assert raw in queries
    assert intent["search_query"] in queries
    assert [s["retrieval_query"] for s in intent["task_segments"]] == queries[:3]
    probes = build_retrieval_probes(raw, intent)
    assert probes[3]["query"] == intent["sub_queries"][0]["query"]
    assert probes[3]["english_query"] == intent["sub_queries"][0]["english_query"]
    assert probes[4]["query"] == intent["sub_queries"][1]["query"]


def test_execution_uses_merged_plan_instead_of_all_task_views():
    raw = "比较 OneTrans 之后的推荐模型"
    intent = {
        "search_query": "推荐模型演进",
        "task_segments": [{"retrieval_query": f"任务不同表达 {i}"} for i in range(5)],
        "sub_queries": [{"query": f"推荐子问题 {i}"} for i in range(4)],
    }
    plan = RetrievalPlan(online_search=True, queries=[{
        "query": "OneTrans 后续推荐模型", "english_query": "OneTrans recommendation follow-up models",
    }])
    assert len(build_retrieval_probes(raw, intent)) == 11  # All requirements remain visible.
    assert select_retrieval_probes(raw, intent, plan) == [{
        "query": "OneTrans 后续推荐模型", "english_query": "OneTrans recommendation follow-up models",
    }]
    # Old responses without executable queries still do one summary search.
    assert select_retrieval_probes(raw, intent, RetrievalPlan())[0]["query"] == "推荐模型演进"
    assert len(select_retrieval_probes(raw, intent, RetrievalPlan())) == 1


def test_executable_plan_deduplicates_without_discarding_independent_queries():
    plan = RetrievalPlan(queries=[{"query": q} for q in [
        "OneTrans recommendation", 'recommendation "OneTrans"', "sequence models",
        "set-based recommendation", "fourth query",
    ]])
    assert [p["query"] for p in select_retrieval_probes("原话", {}, plan)] == [
        "OneTrans recommendation", "sequence models", "set-based recommendation", "fourth query",
    ]


def test_coverage_assessment_receives_attempted_searches():
    captured = {}
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test"
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"sufficient":false}'))])
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    history = [{"query": "OneTrans citation", "stage": "gap"}]
    planner.assess_coverage("后续模型", {}, [], search_history=history)
    payload = json.loads(captured["messages"][1]["content"])
    assert payload["search_history"] == history
    assert "EVERY explicit task requirement" in payload["instructions"]
    assert "materially different strategy" in payload["instructions"]


def test_coverage_does_not_hide_evidence_after_first_forty_or_accept_missing_support():
    captured = {}
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test"
    payload = {"sufficient": True, "task_checks": [
        {"requirement": "first", "status": "supported", "evidence_ids": [41]},
        {"requirement": "second", "status": "supported", "evidence_ids": [999]},
    ]}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assessment = planner.assess_coverage("two requirements", {}, [
        {"doc_name": f"paper {i}", "text": f"fact {i}"} for i in range(45)
    ])
    sent = json.loads(captured["messages"][1]["content"])
    assert len(sent["evidence"]) == 45
    assert assessment.task_checks[0].status == "supported"
    assert assessment.task_checks[1].status == "missing"
    assert assessment.sufficient is False


def test_coverage_cannot_pass_by_omitting_a_user_task():
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test"
    result = {"sufficient": True, "task_checks": [
        {"requirement": "first", "task_ids": [1], "status": "supported", "evidence_ids": [1]},
    ]}
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))]))))
    assessment = planner.assess_coverage("two tasks", {"task_segments": [{"task": "first"}, {"task": "second"}]},
                                         [{"text": "verified first fact"}])
    assert assessment.sufficient is False
    assert "任务 2" in assessment.unresolved[0]
    assert assessment.gaps[0].query == "second"


def test_retrieval_planner_does_not_route_from_intent_source_flags():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "local_search": False,
            "online_search": True,
            "include_github": True,
            "parallel_search": False,
            "reasoning": "需要当前外部证据和实现仓库",
            "is_resolved": True,
        })))] )

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    planner.model = "test-model"
    plan = planner.plan(
        "找最新的多跳RAG实现并比较仓库",
        {
            "intent": "检索并比较实现",
            "task_segments": [{"task": "找最新实现"}, {"task": "比较仓库"}],
            "required_data_sources": ["uploaded_pdf_docs"],
            "need_online_paper_search": False,
            "target_agent": "DocumentSearchAgent",
        },
    )

    sent_context = json.loads(captured["messages"][1]["content"])
    assert "required_data_sources" not in sent_context
    assert "need_online_paper_search" not in sent_context
    assert "target_agent" not in sent_context
    assert plan.online_search is True
    assert plan.local_search is False
    assert plan.include_github is True
    assert plan.parallel_search is False


def test_online_query_phrases_are_not_cut_to_two_terms_or_two_phrases():
    expression = "RAG multi-hop failure analysis; multi-hop evaluation metrics; retrieval evidence coverage"
    phrases = _extract_academic_phrases(expression)
    assert len(phrases) == 3
    assert _build_arxiv_search_expression(phrases).count("all:") == 3
    github_query = _build_github_repository_query(expression)
    assert "failure analysis" in github_query
    assert "evaluation metrics" in github_query
    assert "evidence coverage" in github_query
    assert " OR " in github_query


def test_planner_receives_last_fetch_age_for_explicit_24_hour_refresh():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "local_search": False, "online_search": False, "is_resolved": True,
        })))] )

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    planner.model = "test-model"
    planner.plan("请刷新超过24小时的资料", {"intent": "refresh"}, {
        "shared_pool": [{
            "item_id": "paper:stale", "kind": "evidence", "title": "stale", "source": "web",
            "last_fetched_at": time.time() - 30 * 3600,
            "updated_at": time.time(),
        }],
    })
    sent_context = json.loads(captured["messages"][1]["content"])
    assert sent_context["available_shared_resources"][0]["age_hours_since_fetch"] >= 29.9
    assert "current_utc" in sent_context


def test_recent_candidates_are_visible_even_when_shared_pool_has_items():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "local_search": False,
            "online_search": False,
            "selected_shared_item_ids": ["candidate:c1"],
            "is_resolved": True,
        })))] )

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    planner.model = "test-model"
    plan = planner.plan("继续看候选文献", {"intent": "选择文献"}, {
        "shared_pool": [{"item_id": "evidence:e1", "kind": "evidence", "title": "existing"}],
        "recent_paper_candidates": [{"shared_item_id": "candidate:c1", "title": "candidate"}],
    })
    context = json.loads(captured["messages"][1]["content"])
    ids = {item["item_id"] for item in context["available_shared_resources"]}
    assert {"evidence:e1", "candidate:c1"} <= ids
    assert plan.selected_shared_item_ids == ["candidate:c1"]


def test_planner_can_select_task_relevant_shared_pool_records():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "local_search": False,
            "online_search": False,
            "selected_shared_item_ids": ["turn:3:abc"],
            "is_resolved": True,
        })))] )

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    planner.model = "test-model"
    plan = planner.plan("继续上一轮的结论", {"intent": "续接论证"}, {
        "local_material_available": True,
        "shared_pool": [{"item_id": "turn:3:abc", "kind": "turn_record", "title": "论证", "task": "分析前提"}],
    })
    sent_context = json.loads(captured["messages"][1]["content"])
    assert sent_context["available_shared_resources"][0]["item_id"] == "turn:3:abc"
    assert plan.selected_shared_item_ids == ["turn:3:abc"]
    assert plan.local_search is True  # Prior generated conclusions do not supply professional evidence.


def test_professional_no_search_plan_uses_local_or_reports_missing_basis():
    for local_available in (True, False):
        planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
        planner.model = "test-model"
        planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "requires_evidence": True, "local_search": False, "online_search": False,
                "reasoning": "模型已经知道定义", "selected_shared_item_ids": ["invented-id"],
            })))]),
        )))
        plan = planner.plan("解释一个专业概念", {}, {"local_material_available": local_available})
        assert plan.local_search is local_available
        assert plan.is_resolved is local_available
        assert plan.online_search is False  # Do not invent online permission from an invalid plan.


def test_shared_sources_can_cover_task_without_search_and_conversation_needs_none():
    for requires_evidence, resource in [
        (True, {"item_id": "textbook:1", "kind": "evidence", "title": "教材定义", "evidence_level": "local_snippet"}),
        (False, None),
    ]:
        planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
        planner.model = "test-model"
        selected_ids = [resource["item_id"]] if resource else []
        planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "requires_evidence": requires_evidence, "selected_shared_item_ids": selected_ids,
                "local_search": False, "online_search": False,
            })))]),
        )))
        plan = planner.plan("解释定义" if resource else "你好", {}, {"shared_pool": [resource] if resource else []})
        assert plan.is_resolved is True
        assert plan.local_search is False
        assert plan.online_search is False
        assert plan.selected_shared_item_ids == selected_ids
