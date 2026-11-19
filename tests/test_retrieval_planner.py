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


def test_retrieval_planner_parses_think_tags_prose_and_trailing_commas():
    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "primary-model"
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=(
                "<think>分析检索需求</think>\n"
                "规划结果如下：\n"
                "```json\n"
                '{"requires_evidence": true, "online_search": true, "is_resolved": true, '
                '"queries": [{"query": "OneTrans 推荐模型迭代", "english_query": "OneTrans sequential recommendation"}],}\n'
                "```\n"
                "以上为规划。"
            )
        ))])
    )))
    plan = planner.plan("OneTrans之后还有什么迭代?", {"intent": "调研模型迭代"})
    assert plan.is_resolved is True
    assert plan.online_search is True
    assert len(plan.queries) == 1
    assert plan.queries[0].english_query == "OneTrans sequential recommendation"


def test_retrieval_planner_retries_with_fallback_model_on_empty_or_invalid_json(monkeypatch):
    monkeypatch.setenv("OPENAI_FALLBACK_MODEL", "fallback-model")
    monkeypatch.setattr("retrieval_planner.time.sleep", lambda _s: None)
    called_models = []

    def fake_create(**kwargs):
        called_models.append(kwargs["model"])
        if len(called_models) == 1:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "requires_evidence": True,
            "online_search": True,
            "is_resolved": True,
            "queries": [{"query": "OneTrans 迭代", "english_query": "OneTrans recommendation"}],
        })))])

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "primary-model"
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    plan = planner.plan("OneTrans之后还有什么迭代?", {"intent": "调研模型迭代"})
    assert called_models == ["primary-model", "fallback-model"]
    assert plan.is_resolved is True
    assert plan.online_search is True


def test_retrieval_planner_deduplicates_paired_pool_items_and_skips_excluded():
    from retrieval_planner import finalize_answer_markdown

    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "requires_evidence": True, "online_search": True, "is_resolved": True,
        })))])

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test-model"
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    planner.plan("量子力学最新研究", {"intent": "调研进展"}, {
        "shared_pool": [
            {"item_id": "abc", "kind": "evidence", "title": "Floquet order", "source": "arxiv"},
            {"item_id": "candidate:abc", "kind": "candidate", "title": "Floquet order", "year": 2026},
            {"item_id": "bad", "kind": "evidence", "title": "大语言模型", "excluded_as_irrelevant": "跨学科无关"},
        ]
    })
    sent = json.loads(captured["messages"][1]["content"])
    resources = sent["available_shared_resources"]
    assert len(resources) == 1
    assert resources[0]["item_id"] == "abc"
    assert resources[0]["year"] == 2026


def test_finalize_answer_markdown_fixes_glued_headings_and_appends_reference_links():
    from retrieval_planner import finalize_answer_markdown

    raw = "我会依据资料介绍。## 一、概念与研究主体\n\n核心结论见 [1] 与 [2]。"
    evidence = [
        {"title": "Floquet ground state order", "url": "https://arxiv.org/pdf/2601.00001", "year": 2026},
        {"title": "大语言模型", "url": "https://example.com/llm.pdf", "excluded_as_irrelevant": "无关"},
    ]
    out = finalize_answer_markdown(raw, evidence)
    assert "我会依据资料介绍。\n\n## 一、概念与研究主体" in out
    assert "### 参考资料链接" in out
    assert "**[1]** [Floquet ground state order](https://arxiv.org/pdf/2601.00001) (2026)" in out
    assert "大语言模型" not in out


def test_cross_disciplinary_evidence_is_retained_and_prompt_guards_against_misfiltering():
    captured = {}

    def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "sufficient": True,
            "task_checks": [
                {"requirement": "结合大语言模型与量子计算", "task_ids": [1], "status": "supported", "evidence_ids": [1, 2], "supported_facts": ["双学科均有支撑"]},
            ],
            "irrelevant_evidence": [],
            "gaps": [],
            "unresolved": [],
            "summary": "跨学科资料均相关且已保留。",
        })))])

    planner = RetrievalPlannerAgent.__new__(RetrievalPlannerAgent)
    planner.model = "test-model"
    planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    evidence = [
        {"doc_name": "大语言模型", "text": "大语言模型结构与推理能力"},
        {"doc_name": "Quantum Circuit Optimization", "text": "Variational quantum circuit design"},
    ]
    coverage = planner.assess_coverage(
        "大语言模型如何用于量子计算线路优化？",
        {"task_segments": ["大语言模型用于量子计算线路优化"], "is_cross_disciplinary": True},
        evidence,
    )
    sent_payload = json.loads(captured["messages"][1]["content"])
    assert "不得仅因属于另一学科而排除" in sent_payload["instructions"]
    assert coverage.sufficient is True
    assert coverage.irrelevant_evidence == []
    assert coverage.task_checks[0].evidence_ids == [1, 2]


def test_ingest_online_evidence_filters_noise_prioritizes_cited_and_tags_metadata():
    from document_rag_engine import UniversalDocumentRAGEngine

    upserted = {}

    class FakeCollection:
        def upsert(self, *, ids, embeddings, documents, metadatas):
            upserted["ids"] = ids
            upserted["embeddings"] = embeddings
            upserted["documents"] = documents
            upserted["metadatas"] = metadatas

    class FakeEmbedder:
        def encode(self, texts, **_kwargs):
            return [[0.1, 0.2] for _ in texts]

    engine = UniversalDocumentRAGEngine.__new__(UniversalDocumentRAGEngine)
    engine.collection = FakeCollection()
    engine.embedding_model = FakeEmbedder()

    long_abstract = "Unified transformer architecture for sequential and non-sequential recommendation modeling. " * 3
    evidence_items = [
        {"doc_name": "LocalBook.pdf", "doc_type": "local_pdf", "text": long_abstract},
        {"doc_name": "FusedLocalBook.pdf", "doc_type": "fused_academic", "evidence_level": "local_snippet", "text": long_abstract},
        {"doc_name": "WebPage", "doc_type": "web", "summary": long_abstract},
        {"doc_name": "ShortMeta", "doc_type": "arxiv", "summary": "too short"},
        {"doc_name": "NoisePaper", "doc_type": "crossref", "summary": long_abstract, "excluded_as_irrelevant": "无关"},
        {"doc_name": "UncitedPaper", "doc_type": "arxiv", "arxiv_id": "2601.11111", "summary": long_abstract, "year": 2026, "rerank_score": 0.91},
        {"doc_name": "OneTrans-V2", "doc_type": "fused_academic", "arxiv_id": "2609.28589", "summary": long_abstract, "year": 2026, "url": "https://arxiv.org/pdf/2609.28589"},
    ]

    count = engine.ingest_online_evidence(evidence_items, cited_indices=[7], max_items=2)
    assert count == 2
    assert len(upserted["ids"]) == 2
    assert upserted["metadatas"][0]["doc_name"] == "OneTrans-V2"
    assert upserted["metadatas"][0]["doc_type"] == "online_synced_paper"
    assert upserted["metadatas"][0]["arxiv_id"] == "2609.28589"
    assert upserted["metadatas"][1]["doc_name"] == "UncitedPaper"


def test_filter_probes_by_semantic_alignment_reranker_and_fallback():
    from retrieval_planner import filter_probes_by_semantic_alignment

    class FakeReranker:
        def predict(self, pairs):
            # First probe matches task (high logit), second drifts off-topic (low negative logit).
            return [3.8 if "OneTrans" in pt else -5.2 for _task, pt in pairs]

    scorer = SimpleNamespace(reranker=FakeReranker(), embedding_model=None)
    probes = [
        {"query": "OneTrans 推荐模型迭代", "english_query": "OneTrans sequential recommendation"},
        {"query": "旅游景点路径规划综述", "english_query": "tourism destination recommendation"},
    ]
    kept, dropped = filter_probes_by_semantic_alignment(
        probes,
        "推荐场景用户序列和非序列交互，OneTrans之后推荐模型还有什么迭代?",
        {"intent": "调研OneTrans后续推荐模型迭代"},
        scorer=scorer,
    )
    assert len(kept) == 1
    assert kept[0]["query"] == "OneTrans 推荐模型迭代"
    assert len(dropped) == 1
    assert dropped[0]["query"] == "旅游景点路径规划综述"

    # Single off-topic gap query with allow_fallback=False must be stripped cleanly.
    gap_kept, gap_dropped = filter_probes_by_semantic_alignment(
        [{"query": "旅游景点路径规划综述", "english_query": ""}],
        "OneTrans之后推荐模型还有什么迭代?",
        scorer=scorer,
        allow_fallback=False,
    )
    assert gap_kept == []
    assert len(gap_dropped) == 1

    # All-drifted initial probes with allow_fallback=True fall back to user_query/search_query.
    fb_kept, fb_dropped = filter_probes_by_semantic_alignment(
        [{"query": "完全跑偏的词", "english_query": ""}],
        "OneTrans之后推荐模型还有什么迭代?",
        {"search_query": "OneTrans 推荐模型演进"},
        scorer=lambda _task, pts: [0.05 for _ in pts],
        allow_fallback=True,
    )
    assert len(fb_dropped) == 1
    assert fb_kept == [{"query": "OneTrans 推荐模型演进", "english_query": ""}]


def test_apply_evidence_relevance_tolerates_punctuation_and_case_differences():
    from retrieval_planner import apply_evidence_relevance

    evidence = [
        {"doc_name": "PEFT for Photovoltaic Cell Classification", "text": "We compare LoRA, QLoRA on ConvNeXt-Tiny."},
    ]
    exclusions = [
        {
            "evidence_id": 1,
            "reason": "光伏电池缺陷分类属于工业视觉，与大语言模型微调无关",
            "basis": [
                {"source": "request", "quote": "介绍大模型的微调"},
                {"source": "evidence", "evidence_id": 1, "quote": "peft for photovoltaic cell classification we compare lora qlora"},
            ],
        }
    ]
    out = apply_evidence_relevance(evidence, exclusions, user_query="你好，介绍大模型的微调！")
    assert out[0].get("excluded_as_irrelevant")
    assert out[0]["relevance_review"]["basis_status"] == "located"


def test_prune_list_and_clear_online_synced_papers():
    import time
    from document_rag_engine import UniversalDocumentRAGEngine

    now = time.time()
    store = {
        "old_1": {"doc_name": "Old Paper", "doc_type": "online_synced_paper", "synced_at": now - 40 * 86400, "year": "2024", "source": "arxiv", "url": "https://arxiv.org/pdf/2401.00001"},
        "fresh_1": {"doc_name": "Fresh Paper 1", "doc_type": "online_synced_paper", "synced_at": now - 100, "year": "2026", "source": "arxiv", "url": "https://arxiv.org/pdf/2601.00001", "synced_date": "2026-11-16"},
        "fresh_2": {"doc_name": "Fresh Paper 2", "doc_type": "online_synced_paper", "synced_at": now - 10, "year": "2026", "source": "openalex", "url": "https://doi.org/10.1000/2", "synced_date": "2026-11-16"},
    }

    class FakeCollection:
        def get(self, where=None, include=None):
            if where and "$and" in where:
                target = where["$and"][1]["doc_name"]
                ids = [k for k, v in store.items() if v["doc_name"] == target]
            else:
                ids = list(store.keys())
            return {"ids": ids, "metadatas": [store[k] for k in ids]}

        def delete(self, ids=None):
            for k in (ids or []):
                store.pop(k, None)

    eng = UniversalDocumentRAGEngine.__new__(UniversalDocumentRAGEngine)
    eng.collection = FakeCollection()

    pruned = eng.prune_online_synced_papers(ttl_days=30, max_papers=10)
    assert pruned == 1 and "old_1" not in store

    listed = eng.list_online_synced_papers()
    assert [x["doc_name"] for x in listed] == ["Fresh Paper 2", "Fresh Paper 1"]

    assert eng.clear_online_synced_papers("Fresh Paper 1") == 1
    assert "fresh_1" not in store
    assert eng.clear_online_synced_papers() == 1
    assert store == {}


def test_renumber_citations_and_evidence_produces_contiguous_order():
    from retrieval_planner import finalize_answer_markdown, renumber_citations_and_evidence

    evidence = [
        {"title": "Paper 1", "url": "https://arxiv.org/pdf/2601.00001"},
        {"title": "Paper 2", "url": "https://arxiv.org/pdf/2601.00002"},
        {"title": "Paper 3", "url": "https://arxiv.org/pdf/2601.00003", "year": 2026},
        {"title": "Paper 4", "url": "https://arxiv.org/pdf/2601.00004"},
        {"title": "Paper 5", "url": "https://arxiv.org/pdf/2601.00005", "year": 2026},
    ]
    raw = "核心结论来自 [3] 与 [5]，进一步验证见 [3]。"
    renum_text, ordered_ev = renumber_citations_and_evidence(raw, evidence)
    assert renum_text == "核心结论来自 [1] 与 [2]，进一步验证见 [1]。"
    assert [e["title"] for e in ordered_ev[:2]] == ["Paper 3", "Paper 5"]
    assert [e["title"] for e in ordered_ev[2:]] == ["Paper 1", "Paper 2", "Paper 4"]

    final = finalize_answer_markdown(renum_text, ordered_ev)
    assert "**[1]** [Paper 3](https://arxiv.org/pdf/2601.00003) (2026)" in final
    assert "**[2]** [Paper 5](https://arxiv.org/pdf/2601.00005) (2026)" in final
    assert "[3]" not in final


def test_renumber_citations_repairs_drifted_framework_indices_and_preserves_math():
    from retrieval_planner import renumber_citations_and_evidence

    evidence = [
        {"title": "Space-Filling Curves for Quantum Computing", "summary": "Hilbert and Morton curves.", "url": "https://example.com/1"},
        {"title": "Design and Simulation of Quantum Error Correction Codes", "summary": "Proposes Hybrid QECC and Hybrid Code in Qiskit.", "url": "https://example.com/2"},
        {"title": "From Nanotechnology to Topological Quantum Computers", "summary": "Majorana fermions.", "url": "https://example.com/3"},
        {"title": "The Coherence-Decoherence Boundary: A Dual Domain Framework for Quantum Foundations", "summary": "Dual-Domain Framework with -0 and +1.", "url": "https://example.com/4"},
    ]
    raw = (
        "1. **Hybrid QECC** [2] 在去极化噪声下表现优异；概率位于 $[0, 1]$ 区间。\n"
        "2. **Dual Domain Framework** [1, 3] 提出了相干-退相干边界。\n"
        "3. 后续小标题：**Hybrid QECC** 基准测试 [3] 与 **Dual Domain Framework** [4]。"
    )
    renum_text, ordered_ev = renumber_citations_and_evidence(raw, evidence)
    # Hybrid QECC -> [1], Dual Domain Framework -> [2] consistently everywhere; $[0, 1]$ untouched
    assert "Hybrid QECC** [1]" in renum_text
    assert "Hybrid QECC** 基准测试 [1]" in renum_text
    assert "Dual Domain Framework** [2]" in renum_text
    assert "$[0, 1]$" in renum_text
    assert "[3]" not in renum_text
    assert [e["title"] for e in ordered_ev[:2]] == [
        "Design and Simulation of Quantum Error Correction Codes",
        "The Coherence-Decoherence Boundary: A Dual Domain Framework for Quantum Foundations",
    ]
