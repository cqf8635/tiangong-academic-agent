"""Master LangGraph Multi-Agent Orchestrator.

Orchestrates:
1. Intent Recognition Gateway (intent_recognition_agent.py)
2. Route A: Meeting & Long-Stream Processing (meeting_summarizer.py)
3. Route B: Multi-format Document RAG with Page Attribution (document_rag_engine.py)
4. Route C: Cross-modal Audio-Slide Alignment
5. Route D: Deep Academic Tutoring & Stepwise Derivation
6. TTS Female Voice Output
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from document_rag_engine import UniversalDocumentRAGEngine
from intent_recognition_agent import IntentDecision, IntentRecognitionAgent
from retrieval_planner import (
    RetrievalPlan, RetrievalQuery, RetrievalPlannerAgent, TaskAlignment, retrieval_query_key, select_retrieval_probes,
    apply_evidence_relevance, filter_irrelevant_candidates, is_executable_retrieval_query,
)
from meeting_summarizer import MeetingSummarizer
from online_paper_search_agent import OnlinePaperSearchAgent
from subject_memory import transition_subject_state
from workflow_contracts import (build_task_snapshot, task_decision, task_payload, task_prompt,
                                citation_integrity, append_answer, merge_evidence, merge_text, completion_checks)

from config_utils import get_default_llm_model, get_llm_timeout, resolve_retrieval_time_constraint
import prompts
from prompts import (
    build_lead_tutor_system_prompt,
    build_evidence_synthesis_prompt,
    build_dialectical_inquiry_prompt,
    build_deterministic_dialectical_inquiry,
    build_boundary_memo,
    build_vlm_transcription_prompt,
    get_assistant_name,
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


# ==========================================
# 1. 大脑全局主状态 (Master State)
# ==========================================
class MasterAgentState(TypedDict):
    session_id: Optional[str]
    user_query: str                          # 用户原始输入 (文字/语音转录，100% 保持纯净未篡改)
    task_snapshot: Optional[Dict[str, Any]]  # 本轮原始需求、统一解释和修正记录
    extracted_visual_context: Optional[str]  # 图片转录与原话分开保存
    delivery_checks: Optional[Dict[str, Any]] # 程序可核对的引用与正文完整性
    active_subject: Optional[str]            # LangGraph 原生记忆主体 (无偏见确定性状态通道)
    sub_subject: Optional[str]               # 当前深入的子分支技术点
    subject_evolution_log: List[str]         # 主体生命周期演进轨迹
    intent_decision: Optional[Dict[str, Any]]# 意图识别结果结构体
    retrieval_plan: Optional[Dict[str, Any]] # 独立于意图识别的资料检索计划
    retrieval_queries: Optional[List[str]]   # 原话和语义片段形成的互补查询表达
    retrieval_probes: Optional[List[Dict[str, str]]] # 含任务关联英文检索表达的逐项探针
    retrieval_coverage: Optional[Dict[str, Any]]     # 基于实际召回证据的覆盖评估及未解缺口
    source_diagnostics: Optional[Dict[str, Any]]     # 来源级调用、命中与错误记录
    retrieved_evidence: List[Dict[str, Any]] # 召回的知识切片 (含页码/来源)
    final_answer: str                        # 大脑生成的最终回复内容
    target_agent: str                        # 当前执行的智能体
    tts_audio_file: Optional[str]            # 生成的语音回复文件路径
    tts_voice: Optional[str]                 # 导师音色
    history_summary: Optional[str]           # 长期研讨结构化记忆摘要 (全生命周期高密压缩记忆，0 轮堆叠)
    active_doc: Optional[str]                # 当前会话正在聚焦研讨的核心文献/课件名称 (避免跨文献混杂)
    topic_state: Optional[Dict[str, Any]]    # 会话级主题状态机
    recent_paper_candidates: Optional[List[Dict[str, Any]]] # 前序在线搜索出的候选文献元数据池
    shared_resources: Optional[List[Dict[str, Any]]] # 本轮已选择并加载的共享资料
    shared_pool: Optional[List[Dict[str, Any]]] # 状态、成果、审核意见与资料的统一共享池
    topic_catalog: Optional[List[Dict[str, Any]]] # 会话级结构化主题档案库 (按主题组织历史，拒绝全局粗暴扫描)
    matched_topic: Optional[Dict[str, Any]]       # 本轮外科手术式精准锁定的最佳前序历史主题定论
    pending_action: Optional[Dict[str, Any]]      # 人机协同挂起确认动作 (HITL Pending Action Proposal)
    synthesized_evidence_slate: Optional[str]     # 学术证据与多源资料整理归纳备忘录 (Evidence Slate)
    dialectical_notes: Optional[str]              # 学术博弈质询与去冗评注 (Dialectical Inquiry Memorandum)
    reflection_notes: Optional[str]               # 主研自我内省与证据反思记录 (Cognitive Reflection Log)
    peer_review_notes: Optional[str]              # 质询备忘 (向后兼容)
    review_verdict: Optional[Dict[str, Any]]       # 审核结论及可执行问题
    review_checks: Optional[List[Dict[str, Any]]]  # 按当前任务动态生成的审核点
    review_scores_round1: Optional[Dict[str, Any]] # 审核维度结果（非固定评分门槛）
    review_notice: Optional[str]                  # 多次修订后仍未解决的问题提示
    review_retrieval_rounds: int                  # 本轮审核触发的定向补查次数
    review_retrieval_status: Optional[str]        # 补查结果状态
    review_retrieval_log: Optional[List[Dict[str, Any]]] # 审核缺口与补查记录
    revision_count: int                           # 兼容旧会话；新流程不重写正文
    pending_evidence_additions: Optional[List[Dict[str, Any]]]
    evidence_append_count: Optional[int]
    agent_dialogue_history: Optional[List[Dict[str, Any]]] # 多智能体点对点研讨对话总线日志 (Agent Dialogue Bus)
    user_image: Optional[str]                     # 用户附带的学术题目或图表截图 (Base64)
    specialist_results: Optional[List[Dict[str, Any]]]
    meeting_record_path: Optional[str]
    control_result: Optional[Dict[str, Any]]
    status: str                              # 运行状态


# ==========================================
# 2. 单例引擎初始化
# ==========================================
class MasterEngineRegistry:
    intent_agent = None
    retrieval_planner = None
    rag_engine = None
    meeting_summarizer: Optional[Any] = None
    paper_search_agent: Optional[Any] = None
    hybrid_retriever: Optional[Any] = None
    llm_client: Optional[Any] = None

    @classmethod
    def get_hybrid_retriever(cls):
        if cls.hybrid_retriever is None:
            from cross_source_hybrid_retriever import CrossSourceHybridRetriever
            cls.hybrid_retriever = CrossSourceHybridRetriever(
                rag_engine=cls.get_rag_engine(),
                paper_agent=cls.get_paper_search_agent()
            )
        return cls.hybrid_retriever

    @classmethod
    def get_intent_agent(cls):
        if cls.intent_agent is None:
            cls.intent_agent = IntentRecognitionAgent()
        return cls.intent_agent

    @classmethod
    def get_retrieval_planner(cls):
        if cls.retrieval_planner is None:
            cls.retrieval_planner = RetrievalPlannerAgent()
        return cls.retrieval_planner

    @classmethod
    def get_rag_engine(cls):
        if cls.rag_engine is None:
            cls.rag_engine = UniversalDocumentRAGEngine()
        return cls.rag_engine

    @classmethod
    def get_meeting_summarizer(cls):
        if cls.meeting_summarizer is None:
            cls.meeting_summarizer = MeetingSummarizer()
        return cls.meeting_summarizer

    @classmethod
    def get_paper_search_agent(cls):
        if cls.paper_search_agent is None:
            cls.paper_search_agent = OnlinePaperSearchAgent()
        return cls.paper_search_agent

    @classmethod
    def get_online_paper_agent(cls):
        return cls.get_paper_search_agent()

    @classmethod
    def get_llm_client(cls):
        if cls.llm_client is None:
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
            if not base_url or not api_key:
                raise ValueError("[错误] 未在 .env 中正确配置 OPENAI_BASE_URL 或 OPENAI_API_KEY！")
            cls.llm_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return cls.llm_client


def extract_visual_content_from_image(image_data: str, client: Any) -> str:
    """使用多模态视觉模型忠实、客观、高保真提取图片中的题目、公式（LaTeX）、图表事实 (零过度推演)"""
    if not image_data or not isinstance(image_data, str) or not image_data.startswith("data:image"):
        return ""
    try:
        model = get_default_llm_model()
        print(f"📷 [Vision] 正在通过多模态视觉大模型 ({model}) 客观提取图片/截图内容...")
        prompt = build_vlm_transcription_prompt()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }
        ]
        res = safe_chat_completion(client=client, model=model, messages=messages, temperature=0.1, max_tokens=2048, timeout=get_llm_timeout(90.0), broadcast=False)
        print(f"✅ [Vision] 图片内容客观识别提取完成 (字数: {len(res)})")
        return res
    except Exception as e:
        print(f"⚠️ [Vision Error]: 图片多模态识别失败: {e}")
        return ""


# ==========================================
# 3. LangGraph 节点函数实现
# ==========================================
def intent_routing_node(state: MasterAgentState) -> Dict[str, Any]:
    """节点 1: 意图分析、LangGraph 原生记忆主体流转与上下文理解"""
    query = state["user_query"]
    history_summary = state.get("history_summary", "")
    topic_state = state.get("topic_state") or {"current_topic": "", "sub_topic": "", "status": "INIT", "evolution_path": []}
    agent = MasterEngineRegistry.get_intent_agent()
    rag_engine = None

    # 多模态图片内容前置识别 (仅客观转录图片事实，不预设任何求解偏见)
    user_img = state.get("user_image")
    visual_context = ""
    raw_user_text = query.strip()
    if user_img:
        visual_context = extract_visual_content_from_image(user_img, MasterEngineRegistry.get_llm_client())
        if visual_context:
            if raw_user_text:
                query = f"{raw_user_text}\n\n[附带图片内容]\n{visual_context}"
            else:
                query = visual_context

    # 终端日志展示处理：只展示学者文字提问，绝不把冗长的图片转录内容打印到控制台刷屏
    display_query = (raw_user_text or "附带图片分析") if user_img else query[:80]
    if user_img:
        display_query = f"{display_query} [已附带图片]"

    print(f"\n🎯 [MasterRouter] 正在解析意图 (当前锚定主体: 【{state.get('active_subject') or topic_state.get('current_topic') or '无'}】): \"{display_query}\" ...")
    # The parser sees the transcription as context; the stored request stays literal.
    decision: IntentDecision = agent.parse_intent(state={**state, "user_query": query})
    proposed_decision = decision.model_dump()

    # 意图解析失败时不根据原话长度或旧主题伪造任务/主体变化。
    current_subject = state.get("active_subject") or topic_state.get("current_topic") or ""
    current_sub_subject = state.get("sub_subject") or topic_state.get("sub_topic") or ""
    evo_log = list(state.get("subject_evolution_log") or [])
    previous_memory = {
        "subject": current_subject, "sub_subject": current_sub_subject,
        "status": topic_state.get("status", "INIT"),
        "evolution_path": topic_state.get("evolution_path", []),
    }

    def propose_subject(parsed):
        return transition_subject_state(
            previous_memory, parsed.subject_action or "INHERIT", query,
            subject_entity=(parsed.subject_entity or parsed.target_topic or "").strip(),
            sub_subject=(parsed.sub_topic or "").strip(), intent=parsed.intent,
            evidence=[], constraints=parsed.negative_constraints or [],
        )

    memory = propose_subject(decision) if decision.is_resolved else None
    active_subject = memory["subject"] if memory else current_subject
    sub_subject = memory["sub_subject"] if memory else current_sub_subject

    # 先确定本轮主体，再独立规划检索，避免旧主题覆盖当前任务。
    retrieval_plan = RetrievalPlan(is_resolved=False, reasoning="意图未能可靠解析")
    if decision.is_resolved:
        try:
            retrieval_plan = MasterEngineRegistry.get_retrieval_planner().plan(
                user_query=query,
                intent=decision.model_dump(),
                state_context={
                    "active_subject": active_subject,
                    "previous_subject": current_subject,
                    "history_context": history_summary,
                    "verify_task_against_original": True,
                    "active_doc": state.get("active_doc"),
                    "local_material_available": bool(
                        state.get("active_doc") or state.get("retrieved_evidence")
                        or any((BASE_DIR / "uploads").glob("*.pdf"))
                        or (BASE_DIR / "knowledge_db" / "chroma.sqlite3").exists()
                    ),
                    "recent_paper_candidates": state.get("recent_paper_candidates") or [],
                    "shared_resources": state.get("shared_resources") or [],
                    "shared_pool": state.get("shared_pool") or [],
                },
            )
        except Exception as exc:
            print(f"⚠️ [RetrievalPlanner] 规划失败，本轮跳过自动检索以免错误触发来源: {type(exc).__name__}")
            retrieval_plan = RetrievalPlan(is_resolved=False, reasoning="检索规划失败；保留原话，未自动触发检索",
                task_alignment=TaskAlignment(status="unresolved", issues=["检索前原话对照未能完成"]))

    alignment = retrieval_plan.task_alignment
    if alignment is not None:
        if alignment.status == "matched":
            print("🧭 [MasterRouter] 已对照用户原话核对主体、任务与限制条件。")
        if alignment.status == "corrected":
            allowed = {
                "domain", "intent", "task_segments", "subject_action", "subject_entity", "target_topic",
                "sub_topic", "topic_action", "negative_constraints", "search_query", "online_search_query",
                "time_constraint", "time_constraint_note", "extracted_entities", "is_cross_disciplinary",
                "disciplines", "sub_queries", "requires_long_form", "long_form_type", "framework_reference",
                "need_download_fulltext", "is_candidate_selection", "control_action", "control_parameters",
            }
            try:
                if not alignment.corrected_fields or set(alignment.corrected_fields) - allowed:
                    raise ValueError("任务修正字段无效")
                decision = IntentDecision(**{**decision.model_dump(), **alignment.corrected_fields})
                print("🧭 [MasterRouter] 已对照用户原话修正主体或任务，检索使用修正后的需求。")
            except Exception:
                alignment.status = "unresolved"
        if alignment.status == "unresolved":
            decision = decision.model_copy(update={"is_resolved": False})
            retrieval_plan = RetrievalPlan(is_resolved=False, task_alignment=alignment,
                                          reasoning="主体与任务尚未与原始内容对齐，未执行检索。")

    # Calendar defaults are code policy even after semantic alignment corrections.
    aligned_decision = decision.model_dump()
    decision.time_constraint = resolve_retrieval_time_constraint(decision.time_constraint)

    # 对照完成后才提交主体；修正不得在原有主题上重复 APPEND。
    if decision.is_resolved:
        memory = propose_subject(decision)
        active_subject, sub_subject = memory["subject"], memory["sub_subject"]
        normalized_action = memory["action"]
        subj_entity = (decision.subject_entity or decision.target_topic or "").strip()
        subj_branch = (decision.sub_topic or "").strip()
        evo_msg = (f"APPEND -> 【{subj_entity}】 (Now: {active_subject})" if normalized_action == "APPEND"
                   else f"{normalized_action} -> 【{active_subject}】")
        topic_state = {"current_topic": active_subject, "sub_topic": sub_subject,
                       "status": memory["status"], "evolution_path": memory["evolution_path"]}
        evo_log.append(evo_msg)
        print(f"🧠 [LangGraph Memory] 主体状态通道更新: {evo_msg}")
    else:
        active_subject, sub_subject = current_subject, current_sub_subject
        print("⚠️ [MasterRouter] 当前轮任务尚未可靠确认，保留原主体状态。")

    task_snapshot = build_task_snapshot(state["user_query"], visual_context, proposed_decision,
        decision.model_dump(), active_subject, alignment.model_dump() if alignment else None,
        aligned=aligned_decision)
    task_state = {**state, "task_snapshot": task_snapshot}

    retrieval_probes = select_retrieval_probes(query, decision.model_dump(), retrieval_plan)
    retrieval_queries = [probe["query"] for probe in retrieval_probes]

    # 主查询保持模型对当前任务的概括；原话与各语义片段作为互补查询表达保留。
    search_query = (decision.search_query or "").strip() or query
    if not retrieval_queries:
        retrieval_queries = [search_query]

    print(f"👉 [MasterRouter] 识别结果: 领域=[{decision.domain}] | 意图=[{decision.intent}] | 派发=[{decision.target_agent}]")
    print(f"🔍 [MasterRouter] 正向检索词: \"{search_query}\"")

    evidence_chunks = []
    shared_resources = state.get("shared_resources") or []
    selected_ids = set(retrieval_plan.selected_shared_item_ids or [])
    shared_pool = state.get("shared_pool") or []
    if state.get("session_id") and selected_ids:
        try:
            from session_manager import SessionManager
            selected_pool_items = SessionManager.get_shared_pool_items(state["session_id"], list(selected_ids))
        except Exception as exc:
            print(f"⚠️ [SharedPool] 选择项读取失败：{type(exc).__name__}")
            selected_pool_items = []
    else:
        selected_pool_items = [item for item in shared_pool if isinstance(item, dict) and item.get("item_id") in selected_ids]
    if not shared_pool and not selected_pool_items:
        selected_pool_items = [
            {**item, "item_id": item.get("resource_id"), "kind": "evidence", "resource_id": item.get("resource_id")}
            for item in shared_resources if isinstance(item, dict) and item.get("resource_id") in selected_ids
        ]
    reused_turns = [item for item in selected_pool_items if item.get("kind") == "turn_record"]
    shared_resources = [
        item["resource"] for item in selected_pool_items
        if item.get("kind") == "evidence" and isinstance(item.get("resource"), dict)
    ] or shared_resources
    if reused_turns:
        reusable_context = []
        for item in reused_turns:
            prior_state = item.get("state") or {}
            useful_state = {
                key: prior_state.get(key)
                for key in ("topic_state", "active_doc", "active_subject", "sub_subject", "retrieval_coverage", "review_verdict", "review_checks")
                if prior_state.get(key) is not None
            }
            reusable_context.append(
                f"【复用的前序成果｜第 {item.get('turn_idx', '?')} 轮｜{item.get('title') or ''}】\n"
                f"任务：{item.get('task') or ''}\n成果：{item.get('content') or ''}\n"
                f"相关状态与审核记录：{json.dumps(useful_state, ensure_ascii=False)}"
            )
        history_summary = "\n\n".join(part for part in [state.get("history_summary") or "", *reusable_context] if part)
    if decision.is_resolved:
        resource_by_id = {
            str(resource.get("resource_id") or ""): resource
            for resource in shared_resources if isinstance(resource, dict)
        }
        for item in selected_pool_items:
            resource = item.get("resource") or resource_by_id.get(str(item.get("resource_id") or item.get("item_id") or ""))
            if item.get("kind") == "evidence" and isinstance(resource, dict) and (resource.get("text") or resource.get("summary")):
                evidence_chunks.append(dict(resource))
        for ev in evidence_chunks:
            # Relevance exclusions belong to the previous task, not the source itself.
            ev.pop("excluded_as_irrelevant", None)
            ev["reused_from_shared_pool"] = True
        if evidence_chunks:
            print(f"♻️ [SharedPool] 按本轮任务复用 {len(evidence_chunks)} 项既有证据")
    retrieval_coverage = {"status": "not_assessed", "rounds": 0, "unresolved": []}
    source_diagnostics: Dict[str, Any] = {"counts": {}, "errors": {}, "lookups": []}
    def lookup(**kwargs):
        try:
            return hybrid_retriever.hybrid_search_and_deduplicate(**kwargs)
        finally:
            counts = dict(getattr(hybrid_retriever, "last_source_counts", {}) or {})
            errors = dict(getattr(hybrid_retriever, "last_source_errors", {}) or {})
            source_diagnostics["lookups"].append({"query": kwargs.get("query"), "counts": counts, "errors": errors})
            for source, count in counts.items():
                if type(count) is int:
                    source_diagnostics["counts"][source] = source_diagnostics["counts"].get(source, 0) + count
            source_diagnostics["errors"].update(errors)
    neg_kw = decision.negative_constraints or []
    active_doc = state.get("active_doc")

    # 0. 若提问中明确指定了单篇 arXiv 论文标识 (如天工雷达定向研读《Title》(arXiv:xxxx.xxxxx))，优先锁定本地已入库全文或摘要
    m_target_arxiv = re.search(r"《([^》]+)》\s*\(arXiv:([0-9]+\.[0-9]+(?:v\d+)?)", query)
    if decision.is_resolved and m_target_arxiv:
        target_title = m_target_arxiv.group(1).strip()
        target_pid = m_target_arxiv.group(2).strip()
        clean_pid = re.sub(r"[^A-Za-z0-9._-]+", "_", target_pid).strip("._")
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", target_title).strip("._")[:60]
        possible_name = f"{clean_pid}_{safe_title}.pdf"
        uploads_dir = BASE_DIR / "uploads"
        if not (uploads_dir / possible_name).exists() and uploads_dir.exists():
            matched_files = list(uploads_dir.glob(f"{clean_pid}_*.pdf"))
            if matched_files:
                possible_name = matched_files[0].name
        if not rag_engine:
            rag_engine = MasterEngineRegistry.get_rag_engine()
        local_chunks = rag_engine.search(search_query or query, top_k=6, doc_filter=possible_name)
        if not local_chunks:
            local_chunks = rag_engine.search("", top_k=6, doc_filter=possible_name)
        if local_chunks:
            evidence_chunks = local_chunks
            active_doc = possible_name
            print(f"📄 [MasterRouter] 已直接锁定目标论文全文切片: {possible_name} ({len(evidence_chunks)} 处切片)，0 外部冗余检索！")
        else:
            evidence_chunks = [{
                "doc_name": target_title,
                "location_label": f"arXiv:{target_pid}",
                "text": "",
                "evidence_level": "metadata",
                "content_scope": "用户提供的文献标题与标识；尚未读取摘要或正文",
                "score": 0.98,
                "url": f"https://arxiv.org/pdf/{target_pid}.pdf",
                "doc_type": "arxiv",
            }]
            print(f"📋 [MasterRouter] 已记录目标论文标识（尚未读取摘要）: 《{target_title}》 (arXiv:{target_pid})，0 外部冗余检索！")

    # 1. 依据具体研讨学术方向与前序就绪文献进行自然处理 (0 硬编码，0 意图枚举写死)
    candidates = state.get("recent_paper_candidates") or []
    known_candidate_ids = {item.get("shared_item_id") for item in candidates if isinstance(item, dict)}
    for pooled_item in selected_pool_items:
        candidate = pooled_item.get("candidate")
        if pooled_item.get("kind") == "candidate" and isinstance(candidate, dict):
            candidate_id = candidate.get("shared_item_id") or pooled_item.get("item_id")
            if candidate_id not in known_candidate_ids:
                candidates.append(candidate)
                known_candidate_ids.add(candidate_id)
    matched_candidate = None
    time_constraint = getattr(decision, "time_constraint", None)

    is_multi_paper_query = bool(getattr(decision, "is_candidate_selection", False))
    is_explicit_download = getattr(decision, "need_download_fulltext", False)

    # 若当前无需全网重新检索且会话已有就绪文献：
    # 默认 100% 不触发网络下载，全部基于已有文献摘要进行轻量高密推演；
    # 仅当大模型语义判定 need_download_fulltext 为 True 时，才针对锁定单篇执行真实下载与向量建库。
    if decision.is_resolved and candidates and (retrieval_plan.local_search or is_multi_paper_query) and not retrieval_plan.online_search and decision.target_agent not in ["GeneralChatAgent", "SystemControlAgent"]:
        focus_direction = f"{subj_entity} {subj_branch} {search_query}".strip() or query
        cand_texts = [f"{c.get('title', '')} {c.get('summary', '')[:250]}" for c in candidates]

        try:
            if not rag_engine:
                rag_engine = MasterEngineRegistry.get_rag_engine()
            query_emb = rag_engine.embedding_model.encode([focus_direction], normalize_embeddings=True)
            cand_embs = rag_engine.embedding_model.encode(cand_texts, normalize_embeddings=True)
            sims = np.dot(cand_embs, query_emb.T).reshape(-1)
            if len(sims) == len(candidates):
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                # 非多篇对比场景下，若提问与单篇文献高度共振，锁定该单篇
                if not is_multi_paper_query and best_sim >= 0.68 and 0 <= best_idx < len(candidates):
                    matched_candidate = candidates[best_idx]
                    print(f"🎯 [DirectionGrounding] 依据具体学术方向 【{focus_direction}】 锁定文献: 《{matched_candidate.get('title')}》 (相似度: {best_sim:.4f})")
        except Exception as e:
            print(f"⚠️ [DirectionGrounding Note]: {e}")

        # 分支 A: 用户指令经语义裁定明确要求下载深入分析该论文
        if matched_candidate and is_explicit_download:
            paper_agent = MasterEngineRegistry.get_paper_search_agent()
            print(f"📥 [MasterRouter] 用户指令明确要求深入分析，执行真实下载: 《{matched_candidate.get('title')}》...")
            ingest_res = paper_agent.download_and_ingest(matched_candidate, rag_engine=rag_engine)
            if ingest_res.get("status") == "success":
                doc_file_name = ingest_res.get("file_name")
                active_doc = doc_file_name
                active_subject = f"《{matched_candidate.get('title')}》"
                found_chunks = rag_engine.search(query, top_k=5, doc_filter=doc_file_name)
                if not found_chunks:
                    found_chunks = rag_engine.search(search_query, top_k=5, doc_filter=doc_file_name)
                if not found_chunks:
                    found_chunks = rag_engine.search("", top_k=5, doc_filter=doc_file_name)
                evidence_chunks = found_chunks
                print(f"✅ [MasterRouter] 目标论文全文就绪: {doc_file_name}，命中 {len(evidence_chunks)} 处切片！")

        # 分支 B: 默认情况 (未要求下载) —— 坚决不执行网络下载与切片建库！
        elif matched_candidate and not is_explicit_download:
            # 优先检查本地磁盘此前是否已经下载过该论文缓存
            clean_pid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(matched_candidate.get("paper_id") or "")).strip("._")
            safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", str(matched_candidate.get("title") or "")).strip("._")[:60]
            possible_name = f"{clean_pid}_{safe_title}.pdf"
            local_chunks = rag_engine.search(query, top_k=5, doc_filter=possible_name)
            if local_chunks:
                evidence_chunks = local_chunks
                active_doc = possible_name
                print(f"📄 [MasterRouter] 复用本地已有文献缓存: 《{matched_candidate.get('title')}》，0 网络下载！")
            else:
                # 本地无缓存：直接使用已有的权威 Abstract 摘要进行高密研讨，0 磁盘写入！
                evidence_chunks = [{
                    "doc_name": matched_candidate.get("title", "学术文献"),
                    "evidence_level": matched_candidate.get("evidence_level") or "metadata_or_abstract",
                    "doi": matched_candidate.get("doi"), "arxiv_id": matched_candidate.get("arxiv_id"),
                    "location_label": f"文献权威摘要 ({matched_candidate.get('source', 'online')})",
                    "text": f"【论文题目】: {matched_candidate.get('title')}\n【主要作者】: {', '.join(matched_candidate.get('authors', []))}\n【发表年份】: {matched_candidate.get('year')}\n【核心摘要/方案】: {matched_candidate.get('summary')}\n【文献链接/DOI】: {matched_candidate.get('paper_id') or matched_candidate.get('pdf_url')}",
                    "score": 0.95,
                    "url": matched_candidate.get("pdf_url") or matched_candidate.get("paper_id"),
                    "doc_type": matched_candidate.get("source", "online"),
                }]
                print(f"📋 [MasterRouter] 用户未要求下载，仅使用文献权威摘要进行研讨: 《{matched_candidate.get('title')}》，0 磁盘写入！")

        # 分支 C: 多篇横向筛选/比较/全量问答场景，装配全集文献资产
        if not evidence_chunks and candidates and is_multi_paper_query:
            evidence_chunks = []
            for i, c in enumerate(candidates, 1):
                evidence_chunks.append({
                    "doc_name": c.get("title", f"候选文献 {i}"),
                    "evidence_level": c.get("evidence_level") or "metadata_or_abstract",
                    "doi": c.get("doi"), "arxiv_id": c.get("arxiv_id"),
                    "location_label": f"学术文献 ({c.get('source', 'online')})",
                    "text": f"【文献序号】: {i}\n【论文题目】: {c.get('title')}\n【作者团队】: {', '.join(c.get('authors', []))}\n【发表年份】: {c.get('year')}\n【收录来源/类型】: {c.get('source')}\n【核心摘要/方案】: {c.get('summary')}\n【链接/DOI】: {c.get('paper_id') or c.get('pdf_url')}",
                    "score": 0.95,
                    "url": c.get("pdf_url") or c.get("paper_id"),
                    "doc_type": c.get("source", "online"),
                })
            print(f"📋 [MasterRouter] 会话已有 {len(evidence_chunks)} 篇就绪文献，全量装配至研讨证据链，0 网络下载！")

    if task_snapshot["active_subject"] != active_subject:
        task_snapshot["interpretation_log"].append({"stage": "resource_focus", "before": task_snapshot["active_subject"],
            "after": active_subject, "active_doc": active_doc, "grounds_type": "retrieved_resource"})
        task_snapshot["active_subject"] = active_subject
        topic_state = {**topic_state, "current_topic": active_subject}
        evo_log.append(f"RESOURCE_FOCUS -> 【{active_subject}】")
    task_snapshot["active_doc"] = active_doc

    need_online = decision.is_resolved and retrieval_plan.online_search
    need_local_search = decision.is_resolved and retrieval_plan.local_search
    need_online_search = need_online
    probe_by_query = {p["query"]: p.get("english_query", "") for p in retrieval_probes}

    if (not evidence_chunks or need_online_search) and (need_local_search or need_online_search):
        is_cross = getattr(decision, "is_cross_disciplinary", False)
        sub_qs = getattr(decision, "sub_queries", []) or []
        disc_list = getattr(decision, "disciplines", [decision.domain]) or [decision.domain]

        if is_cross and sub_qs:
            # 1. 跨学科主次分层（纯语义大模型决策，零硬编码）：
            # 过滤由大模型语义裁定为上位宽泛通识的宏观学科 (is_umbrella_generic)
            filtered_sub_qs = [sq for sq in sub_qs if not sq.get("is_umbrella_generic", False)]
            if not filtered_sub_qs:
                filtered_sub_qs = sub_qs

            print(f"🔀 [MasterRouter] 检测到多领域交叉课题，纯语义主次分层生效 (专精领域: {[s.get('discipline') for s in filtered_sub_qs]})...")
            
            # 2. 任务级去重与检索下沉（纯语义大模型决策，零硬编码）：
            # 读取大模型在语义理解阶段裁定的核心理论冲突瓶颈子任务 (is_core_bottleneck 或 retrieval_strategy == 'online_probe')
        else:
            filtered_sub_qs = []
            print(f"🌐 [MasterRouter] 执行领域定向混合检索 (领域: 【{decision.domain}】 | 负向抑制: {neg_kw})...")

        try:
            hybrid_retriever = MasterEngineRegistry.get_hybrid_retriever()
            if is_cross and filtered_sub_qs:
                all_raw_chunks = []
                main_en = (getattr(decision, "online_search_query", "") or "").strip()
                probe_online_enabled = need_online_search and (not need_local_search or retrieval_plan.parallel_search)
                probe_tasks = [
                    (p["query"], 6 if p["query"] == search_query else 4,
                     p.get("english_query") or (main_en if p["query"] == search_query else ""),
                     "任务语义探针", bool(probe_online_enabled))
                    for p in retrieval_probes
                ]
                print(f"🎯 [MasterRouter] 按任务语义构造 {len(probe_tasks)} 路检索探针")

                # 🚀 全域多探针并发检索 (彻底消除串行累加等待，耗时缩减 70%+)
                print(f"⚡ [MasterRouter] 启动全域 {len(probe_tasks)} 路学术探针并发召回...")
                def _execute_single_probe(task_info):
                    q, k, en_q, label, probe_online = task_info
                    return lookup(
                        query=q,
                        negative_keywords=neg_kw,
                        top_k=k,
                        include_local=need_local_search,
                        include_online=probe_online,
                        include_github=retrieval_plan.include_github,
                        include_web=retrieval_plan.include_web,
                        include_datasets=retrieval_plan.include_datasets,
                        time_constraint=time_constraint,
                        english_query=en_q or None,
                    )

                probe_results = [_execute_single_probe(task_info) for task_info in probe_tasks]

                for chunks in probe_results:
                    all_raw_chunks.extend(chunks)

                # Preserve newly inspected content when repeated sources meet across probes.
                evidence_chunks, _ = merge_evidence(evidence_chunks, all_raw_chunks)
                if need_local_search and need_online_search and not retrieval_plan.parallel_search:
                    sequential_tasks = [
                        (q, k, en_q, label, True)
                        for q, k, en_q, label, _ in probe_tasks
                    ]
                    def _execute_online_only(task_info):
                        q, k, en_q, label, _ = task_info
                        return lookup(
                            query=q,
                            negative_keywords=neg_kw,
                            top_k=k,
                            include_local=False,
                            include_online=True,
                            include_github=retrieval_plan.include_github,
                            include_web=retrieval_plan.include_web,
                            include_datasets=retrieval_plan.include_datasets,
                            time_constraint=time_constraint,
                            english_query=en_q or None,
                        )
                    online_results = [_execute_online_only(task_info) for task_info in sequential_tasks]
                    for chunks in online_results:
                        all_raw_chunks.extend(chunks)
                    for chunks in online_results:
                        evidence_chunks, _ = merge_evidence(evidence_chunks, chunks)

            else:
                main_en = (getattr(decision, "online_search_query", "") or "").strip()
                def _run_query_views(include_local: bool, include_online: bool):
                    if not retrieval_queries:
                        return []
                    groups = [
                        lookup(
                            query=q,
                            negative_keywords=neg_kw,
                            top_k=10,
                            include_local=include_local,
                            include_online=include_online,
                            include_github=retrieval_plan.include_github,
                            include_web=retrieval_plan.include_web,
                            include_datasets=retrieval_plan.include_datasets,
                            time_constraint=time_constraint,
                            english_query=probe_by_query.get(q) or ((main_en or None) if q == search_query else None),
                        ) for q in retrieval_queries
                    ]
                    merged = []
                    for group in groups:
                        merged, _ = merge_evidence(merged, group)
                    return merged

                if need_local_search and need_online_search and not retrieval_plan.parallel_search:
                    local_chunks = _run_query_views(include_local=True, include_online=False)
                    online_chunks = _run_query_views(include_local=False, include_online=True)
                    evidence_chunks, _ = merge_evidence(evidence_chunks, [*local_chunks, *online_chunks])
                else:
                    evidence_chunks, _ = merge_evidence(evidence_chunks, _run_query_views(need_local_search, need_online_search))
            print(f"📚 [MasterRouter] 初查完成: {len(evidence_chunks)} 项候选证据，转入回答")
        except Exception as e:
            print(f"⚠️ [MasterRouter Hybrid Retrieval Note]: {e}")
            if need_local_search:
                evidence_chunks, _ = merge_evidence(evidence_chunks, _local_fallback_evidence(
                    {**task_state, "retrieval_plan": retrieval_plan.model_dump(), "retrieval_queries": [search_query],
                     "retrieval_coverage": {}, "active_doc": active_doc}, top_k=6))

        # Retain evidence-driven supplementary retrieval before drafting.
        if retrieval_plan.is_resolved and (need_local_search or need_online_search):
            initial_sources = (["local"] if need_local_search else []) + (["academic"] if need_online_search else [])
            search_history = [{"query": q, "stage": "initial", "sources": initial_sources}
                              for q in retrieval_queries]
            attempted_queries, attempted_strategies = set(), set()
            stagnant_rounds, previous_coverage, last_added = 0, None, None
            max_gap_rounds = _evidence_retrieval_budget(
                {"intent_decision": decision.model_dump()}, "RETRIEVAL_GAP_ROUNDS")
            try:
                planner = MasterEngineRegistry.get_retrieval_planner()
                for gap_round in range(max_gap_rounds + 1):
                    assessment = planner.assess_coverage(query, {**decision.model_dump(), "shared_task": task_payload(task_state)}, evidence_chunks,
                                                         search_history=search_history)
                    evidence_chunks = apply_evidence_relevance(evidence_chunks, assessment.irrelevant_evidence,
                        raw_user_text, visual_context)
                    checks = [check.model_dump() for check in assessment.task_checks]
                    covered = bool(assessment.sufficient and assessment.task_checks and not assessment.gaps
                                   and not assessment.unresolved and all(c.status == "supported" for c in assessment.task_checks))
                    signature = {retrieval_query_key(c.requirement): (
                        {"missing": 0, "partial": 1, "supported": 2}[c.status],
                        {retrieval_query_key(f) for f in c.supported_facts}) for c in assessment.task_checks}
                    if last_added is not None:
                        unchanged = bool(signature and previous_coverage is not None and not any(
                            rank > previous_coverage.get(key, (0, set()))[0]
                            or facts - previous_coverage.get(key, (0, set()))[1]
                            for key, (rank, facts) in signature.items()))
                        stagnant_rounds = stagnant_rounds + 1 if last_added == 0 or unchanged else 0
                    previous_coverage = signature
                    unresolved = list(assessment.unresolved)
                    unresolved.extend(c.note or c.requirement for c in assessment.task_checks if c.status != "supported")
                    retrieval_coverage = {
                        "status": "sufficient" if covered else "gaps_found", "rounds": gap_round,
                        "summary": assessment.summary, "task_checks": checks,
                        "irrelevant_evidence": [item.model_dump() for item in assessment.irrelevant_evidence],
                        "unresolved": list(dict.fromkeys(unresolved)), "search_history": search_history,
                    }
                    new_strategy = any(g.strategy and (
                        retrieval_query_key(g.requirement or g.rationale or g.query), retrieval_query_key(g.strategy))
                        not in {(claim, strategy) for claim, strategy, _ in attempted_strategies} for g in assessment.gaps)
                    stop_reason = ("covered" if covered else "safety_limit" if gap_round >= max_gap_rounds else
                                   "stalled" if stagnant_rounds >= 2 and not new_strategy else
                                   "no_actionable_gap" if not assessment.gaps else "")
                    if stop_reason:
                        retrieval_coverage["stop_reason"] = stop_reason
                        if not covered:
                            retrieval_coverage["status"] = "budget_exhausted" if stop_reason == "safety_limit" else "unresolved"
                            retrieval_coverage["unresolved"].extend(g.rationale or g.query for g in assessment.gaps)
                        break
                    executed, last_added = 0, 0
                    for gap in assessment.gaps:
                        if not is_executable_retrieval_query(gap.query):
                            continue
                        sources = {source.casefold() for source in gap.sources}
                        local = need_local_search and (not sources or "local" in sources)
                        online = need_online_search and (not sources or bool(sources & {"academic", "web", "github", "datasets"}))
                        if not local and not online:
                            continue
                        channels = (local, online, retrieval_plan.include_github, retrieval_plan.include_web, retrieval_plan.include_datasets)
                        query_key = (retrieval_query_key(gap.query), channels)
                        strategy_key = (retrieval_query_key(gap.requirement or gap.rationale or gap.query),
                                        retrieval_query_key(gap.strategy), channels)
                        initial_repeat = (channels[:2] == (need_local_search, need_online_search)
                                          and query_key[0] in {retrieval_query_key(q) for q in retrieval_queries})
                        if not query_key[0] or query_key in attempted_queries or strategy_key in attempted_strategies or initial_repeat:
                            continue
                        attempted_queries.add(query_key)
                        attempted_strategies.add(strategy_key)
                        record = {"query": gap.query, "reason": gap.rationale, "requirement": gap.requirement,
                                  "strategy": gap.strategy, "sources": list(sources) or initial_sources, "stage": "gap"}
                        search_history.append(record)
                        executed += 1
                        found = lookup(
                            query=gap.query, negative_keywords=neg_kw, top_k=6,
                            include_local=local, include_online=online, include_github=retrieval_plan.include_github,
                            include_web=retrieval_plan.include_web, include_datasets=retrieval_plan.include_datasets,
                            time_constraint=time_constraint)
                        evidence_chunks, delta = merge_evidence(evidence_chunks, found)
                        added = len(delta)
                        record.update(new_or_updated_evidence=added, result="evidence_added" if added else "no_new_evidence")
                        last_added += added
                    if not executed:
                        retrieval_coverage.update(status="unresolved", stop_reason="no_new_strategy")
                        retrieval_coverage["unresolved"].extend(g.rationale or g.query for g in assessment.gaps)
                        break
            except Exception as exc:
                retrieval_coverage = {"status": "assessment_unavailable",
                    "rounds": len([r for r in search_history if r["stage"] == "gap"]),
                    "unresolved": [f"检索覆盖度评估失败：{type(exc).__name__}"], "search_history": search_history}

        print(f"✅ [MasterRouter] 检索阶段结束: {len(evidence_chunks)} 项证据；覆盖状态={retrieval_coverage['status']}，转入回答")

        # Publish new source assets even when the session already has candidates.
        incoming_candidates = []
        for c in evidence_chunks:
            kind = c.get("doc_type", "")
            if c.get("excluded_as_irrelevant") or kind not in {"arxiv", "openalex", "crossref", "github", "web", "dataset", "fused_academic"}:
                continue
            sources = c.get("sources_merged") or []
            source = next((v for v in sources if v in {"arxiv", "openalex", "crossref", "github"}), kind) if kind == "fused_academic" else kind
            incoming_candidates.append({"paper_id": str(c.get("arxiv_id") or c.get("doi") or c.get("doc_name") or ""),
                "title": c.get("doc_name") or "未命名文献", "authors": c.get("authors") or [],
                "year": c.get("year") or (c.get("provenance") or {}).get("year") or c.get("page_or_slide"),
                "summary": c.get("summary") or c.get("text") or "", "source": source,
                "doi": c.get("doi"), "arxiv_id": c.get("arxiv_id"),
                "url": c.get("url") or c.get("pdf_url") or "", "pdf_url": c.get("pdf_url") or c.get("url") or "",
                "evidence_level": c.get("evidence_level"), "provenance": c.get("provenance") or {},
                "stars": c.get("stars"), "version_ref": c.get("version_ref")})
        candidates, _ = merge_evidence(candidates, incoming_candidates)

    candidates = filter_irrelevant_candidates(candidates, evidence_chunks)
    for ev in evidence_chunks:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("doc_type") or "unknown").casefold()
        ev.setdefault(
            "evidence_level",
            "local_snippet" if kind in {"local_pdf", "local", "document"} else
            "abstract" if kind == "arxiv" else
            "repository_search_metadata" if kind == "github" else
            "dataset_repository_metadata" if kind == "dataset" else
            "web_search_snippet" if kind == "web" else
            "bibliographic_metadata_or_abstract" if kind in {"openalex", "crossref", "fused_academic"} else
            "unknown",
        )
        ev.setdefault("provenance", {
            "source": kind,
            "location": ev.get("location_label"),
            "url": ev.get("url") or ev.get("pdf_url") or ev.get("github_url"),
            "year": ev.get("year") or ev.get("page_or_slide"),
            "content_scope": ev.get("evidence_level"),
            "repository_code_inspected": False if kind == "github" else None,
        })

    return {
        "task_snapshot": task_snapshot,
        "extracted_visual_context": visual_context,
        "intent_decision": decision.model_dump(),
        "retrieval_plan": retrieval_plan.model_dump(),
        "retrieval_queries": retrieval_queries,
        "retrieval_probes": retrieval_probes,
        "retrieval_coverage": retrieval_coverage,
        "source_diagnostics": source_diagnostics,
        "target_agent": decision.target_agent,
        "active_subject": active_subject,
        "sub_subject": sub_subject,
        "subject_evolution_log": evo_log,
        "topic_state": topic_state,
        "retrieved_evidence": evidence_chunks,
        "active_doc": active_doc,
        "recent_paper_candidates": candidates,
        "history_summary": history_summary,
        "status": "intent_routed",
    }


AGENT_TO_PIPELINE = {
    "SystemControlAgent": "system_control_pipeline",
    "OnlinePaperSearchAgent": "online_paper_search_pipeline",
    "CrossModalRAGAgent": "cross_modal_pipeline",
    "SummarizerAgent": "meeting_summary_pipeline",
    "DocumentSearchAgent": "document_rag_pipeline",
}


def route_selector(state: MasterAgentState) -> str:
    """条件路由分支判断：严格基于语义意图推荐智能体派发执行管线 (0 硬编码、字典映射)"""
    decision = task_decision(state)
    target_agent = decision.get("target_agent", "") if isinstance(decision, dict) else getattr(decision, "target_agent", "")
    retrieval_plan = state.get("retrieval_plan") or {}
    if (retrieval_plan.get("task_alignment") or {}).get("status") == "unresolved":
        return "general_tutor_pipeline"
    if target_agent == "OnlinePaperSearchAgent" and not retrieval_plan.get("online_search", False):
        return "general_tutor_pipeline"
    if target_agent == "DocumentSearchAgent" and not retrieval_plan.get("local_search", False):
        return "general_tutor_pipeline"
    return AGENT_TO_PIPELINE.get(target_agent, "general_tutor_pipeline")


def online_paper_search_node(state: MasterAgentState) -> Dict[str, Any]:
    """节点: 多源在线学术文献与开源生态检索智能体 (arXiv + CrossRef + GitHub)"""
    decision = task_decision(state)
    retrieval_plan = state.get("retrieval_plan") or {}
    raw_query = state.get("user_query", "").strip()
    topic_state = state.get("topic_state") or {}
    active_topic = topic_state.get("current_topic") or (decision.get("target_topic") if isinstance(decision, dict) else getattr(decision, "target_topic", "")) or ""
    history_summary = state.get("history_summary") or ""

    # 优先采用经过主题状态机语境化补全的检索词
    search_query = (decision.get("search_query") if isinstance(decision, dict) else getattr(decision, "search_query", "")) or ""
    search_query = search_query.strip()
    topic_action = decision.get("topic_action") if isinstance(decision, dict) else getattr(decision, "topic_action", "STAY_LOCKED")
    subject_action = decision.get("subject_action") if isinstance(decision, dict) else getattr(decision, "subject_action", "INHERIT")

    if not search_query:
        if active_topic and subject_action not in ["REPLACE", "INIT"] and topic_action != "RESET":
            search_query = f"{active_topic} {raw_query}".strip()
        else:
            search_query = raw_query
    # 🤝 主任务检索成果共享：优先消费主任务沉淀的文献资产，严禁子任务发起重复冗余检索
    shared_cands = filter_irrelevant_candidates([
        c for c in (state.get("recent_paper_candidates") or [])
        if c.get("source") in ["arxiv", "openalex", "crossref", "github"]
    ], state.get("retrieved_evidence") or [])
    reuse_evidence = bool(shared_cands or state.get("retrieved_evidence")
                          or (state.get("retrieval_coverage") or {}).get("search_history"))
    online_query_views = state.get("retrieval_queries") or [search_query]
    if not shared_cands and len(online_query_views) > 1:
        search_query = "; ".join(online_query_views)
    agent = MasterEngineRegistry.get_paper_search_agent()
    from online_paper_search_agent import ArxivPaper, AcademicResource

    if reuse_evidence:
        print(f"\n🔄 [OnlinePaperSearchAgent] 检测到主任务已完成多源检索并沉淀共享候选池 ({len(shared_cands)} 篇)，直接复用证据链，避免重复外部调用...")
        arxiv_papers = [
            ArxivPaper(
                paper_id=str(c.get("paper_id") or ""),
                title=str(c.get("title") or "未命名论文"),
                authors=c.get("authors") or [],
                summary=str(c.get("summary") or ""),
                pdf_url=str(c.get("pdf_url") or c.get("url") or ""),
                year=c.get("year"),
            )
            for c in shared_cands if c.get("source") == "arxiv"
        ]
        openalex_resources = [
            AcademicResource(
                id=str(c.get("paper_id") or ""),
                title=str(c.get("title") or "未命名文献"),
                authors=c.get("authors") or [],
                summary=str(c.get("summary") or ""),
                url=str(c.get("pdf_url") or c.get("url") or ""),
                year=c.get("year"),
                source="openalex",
            )
            for c in shared_cands if c.get("source") == "openalex"
        ]
        crossref_resources = [
            AcademicResource(
                id=str(c.get("paper_id") or ""),
                title=str(c.get("title") or "未命名文献"),
                authors=c.get("authors") or [],
                summary=str(c.get("summary") or ""),
                url=str(c.get("pdf_url") or c.get("url") or ""),
                doi=str(c.get("paper_id") or ""),
                year=c.get("year"),
                source="crossref",
            )
            for c in shared_cands if c.get("source") == "crossref"
        ]
        github_resources = [
            AcademicResource(
                id=str(c.get("paper_id") or ""),
                title=str(c.get("title") or "未命名仓库"),
                summary=str(c.get("summary") or ""),
                url=str(c.get("pdf_url") or c.get("url") or ""),
                stars=c.get("stars") or 0,
                source="github",
            )
            for c in shared_cands if c.get("source") == "github"
        ]
        evidence = list(state.get("retrieved_evidence") or [])
        candidates_list = shared_cands
    else:
        # 回退分支：仅当主任务未就绪候选时才由子任务补充执行多源检索
        print(f"\n🌐 [ArxivSearchAgent] 主任务未就绪候选，正在全网多源检索学术文献与开源生态 (主题: 【{active_topic or '通用'}】 | 检索词: \"{search_query}\") ...")
        bundle = agent.search_multisource(
            search_query,
            include_github=bool(retrieval_plan.get("include_github", False)),
            include_web=bool(retrieval_plan.get("include_web", False)),
            include_datasets=bool(retrieval_plan.get("include_datasets", False)),
            time_constraint=decision.get("time_constraint"),
            english_query=decision.get("online_search_query") if len(online_query_views) == 1 else None,
        )
        arxiv_raw = bundle.get("arxiv_papers", [])
        openalex_raw = bundle.get("openalex_papers", [])
        crossref_raw = bundle.get("crossref_papers", [])
        github_raw = bundle.get("github_repos", [])
        web_raw = bundle.get("web_results", [])
        dataset_raw = bundle.get("datasets", [])

        arxiv_papers = [ArxivPaper(**p) for p in arxiv_raw]
        openalex_resources = [AcademicResource(**op) for op in openalex_raw]
        crossref_resources = [AcademicResource(**cp) for cp in crossref_raw]
        github_resources = [AcademicResource(**gh) for gh in github_raw]

        evidence = []
        for op in openalex_resources:
            evidence.append({
                "doc_name": op.title,
                "location_label": f"OpenAlex ({op.citation_count}引用)",
                "evidence_level": op.evidence_level or "bibliographic_metadata",
                "content_scope": op.content_scope,
                "text": f"来源: {op.summary}\nDOI: {op.doi}\nURL: {op.url}",
                "score": 0.98,
                "url": op.url,
                "doi": op.doi,
                "doc_type": "openalex",
                "summary": op.summary,
                "evidence_level": op.evidence_level or "index_metadata_or_abstract",
            })
        for p in arxiv_papers:
            evidence.append({
                "doc_name": p.title,
                "location_label": f"arXiv:{p.paper_id} ({p.year or 'Recent'})",
                "text": f"作者: {', '.join(p.authors)}\nPDF: {p.pdf_url}\n摘要: {p.summary}",
                "score": 0.95,
                "pdf_url": p.pdf_url,
                "paper_id": p.paper_id,
                "arxiv_id": p.paper_id,
                "doc_type": "arxiv",
                "summary": p.summary,
                "evidence_level": "abstract",
            })
        for cp in crossref_resources:
            evidence.append({
                "doc_name": cp.title,
                "evidence_level": cp.evidence_level or "bibliographic_metadata",
                "content_scope": cp.content_scope,
                "location_label": f"CrossRef DOI ({cp.year or 'Recent'})",
                "text": f"DOI: {cp.doi}\n出处: {cp.summary}\nURL: {cp.url}",
                "score": 0.92,
                "url": cp.url,
                "doi": cp.doi,
                "doc_type": "crossref",
                "summary": cp.summary,
                "evidence_level": cp.evidence_level or "index_metadata_or_abstract",
            })
        for gh in github_resources:
            evidence.append({
                "doc_name": gh.title,
                "location_label": f"GitHub ({gh.stars}⭐)",
                "text": f"Repo: {gh.title}\nStars: {gh.stars}⭐\nURL: {gh.url}\n简介: {gh.summary}",
                "score": 0.90,
                "url": gh.url,
                "doc_type": "github",
                "summary": gh.summary,
                "version_ref": gh.version_ref,
                "version_ref_type": gh.version_ref_type,
                "evidence_level": gh.evidence_level or "repository_metadata",
                "provenance": {"source": "github", "url": gh.url, "version_ref": gh.version_ref,
                               "version_ref_type": gh.version_ref_type,
                               "content_scope": gh.content_scope or "repository metadata; source files not inspected"},
            })
        for item in web_raw + dataset_raw:
            evidence.append({
                "doc_name": item.get("title", "External resource"),
                "location_label": "Web result snippet" if item.get("source") == "web" else "Dataset repository metadata",
                "text": item.get("summary", ""),
                "score": 0.72,
                "url": item.get("url"),
                "doc_type": item.get("source", "web"),
                "evidence_level": item.get("evidence_level", "search_snippet"),
                "provenance": {"source": item.get("source"), "url": item.get("url"), "content_scope": item.get("content_scope") or item.get("evidence_level")},
                "summary": item.get("summary", ""),
            })

        candidates_list = []
        for p in arxiv_papers:
            candidates_list.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "summary": p.summary,
                "pdf_url": p.pdf_url,
                "source": "arxiv",
            })
        for op in openalex_resources:
            candidates_list.append({
                "paper_id": op.id,
                "title": op.title,
                "authors": op.authors,
                "year": op.year,
                "summary": op.summary,
                "pdf_url": op.url,
                "source": "openalex",
            })
        for cp in crossref_resources:
            candidates_list.append({
                "paper_id": cp.id,
                "title": cp.title,
                "authors": cp.authors,
                "year": cp.year,
                "summary": cp.summary,
                "pdf_url": cp.url,
                "source": "crossref",
            })
        for gh in github_resources:
            candidates_list.append({
                "paper_id": gh.id,
                "title": gh.title,
                "summary": gh.summary,
                "pdf_url": gh.url,
                "source": "github",
                "stars": gh.stars,
                "version_ref": gh.version_ref,
                "version_ref_type": gh.version_ref_type,
                "evidence_level": gh.evidence_level or "repository_metadata",
                "content_scope": gh.content_scope,
            })
        candidates_list.extend([
            {
                "paper_id": str(item.get("url") or item.get("title") or ""),
                "title": item.get("title", "External resource"),
                "summary": item.get("summary", ""),
                "pdf_url": item.get("url", ""),
                "url": item.get("url", ""),
                "source": item.get("source", "web"),
                "evidence_level": item.get("evidence_level", "search_snippet"),
            }
            for item in web_raw + dataset_raw
        ])
        fetched_at = time.time()
        for candidate in candidates_list:
            candidate.setdefault("last_fetched_at", fetched_at)

    if not evidence and candidates_list:
        evidence = [{
            **candidate,
            "doc_name": candidate.get("title") or "未命名资料",
            "doc_type": candidate.get("source") or "unknown",
            "text": candidate.get("summary") or "",
            "url": candidate.get("url") or candidate.get("pdf_url") or "",
            "arxiv_id": candidate.get("paper_id") if candidate.get("source") == "arxiv" else None,
            "doi": candidate.get("doi") or (candidate.get("paper_id") if candidate.get("source") == "crossref" else None),
            "evidence_level": candidate.get("evidence_level") or "metadata_or_abstract",
        } for candidate in candidates_list]

    requires_long_form = decision.get("requires_long_form", False)
    long_form_type = decision.get("long_form_type", "concise_analysis")

    brief = agent.generate_search_brief(
        query=raw_query or search_query,
        shared_task=task_payload(state),
        papers=arxiv_papers,
        openalex_papers=openalex_resources,
        crossref_papers=crossref_resources,
        github_repos=github_resources,
        history_context=history_summary,
        active_topic=active_topic,
        requires_long_form=requires_long_form,
        long_form_type=long_form_type,
        evidence=evidence if evidence else None,
    )

    # 构造人机协同确认提议 (HITL Pending Action Proposal)
    top_candidate = candidates_list[0] if candidates_list else None
    pending_action = None
    if top_candidate:
        top_title = top_candidate.get("title", "")
        pending_action = {
            "action_type": "confirm_download_paper",
            "paper_id": top_candidate.get("paper_id"),
            "title": top_title,
            "prompt": f"检测到与研讨课题契合度最高的代表作《{top_title}》。",
        }

    return {
        "final_answer": brief,
        "retrieved_evidence": evidence,
        "source_diagnostics": {**(state.get("source_diagnostics") or {}),
            "errors": {**((state.get("source_diagnostics") or {}).get("errors") or {}),
                       **(bundle.get("source_errors", {}) if not reuse_evidence else {})}},
        "delivery_checks": {**(state.get("delivery_checks") or {}), **completion_checks(brief)},
        "recent_paper_candidates": candidates_list,
        "pending_action": pending_action,
        "status": "online_paper_search_completed",
        "target_agent": "OnlinePaperSearchAgent",
    }


DEFAULT_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "100.0"))

# 🛡️ 全局大模型并发流控信号量 (防止瞬时并发长连接压垮中继代理，彻底消除 503 Overloaded 与 Cloudflare 502 Bad Gateway)
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "2"))
GLOBAL_LLM_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_LLM)


_STREAM_CALLBACK = ContextVar("workflow_stream_callback", default=None)


def register_stream_callback(callback):
    _STREAM_CALLBACK.set(callback)


def get_stream_callback():
    return _STREAM_CALLBACK.get()


class CompletionText(str):
    def __new__(cls, content, finish_reason=None):
        result = super().__new__(cls, content)
        result.finish_reason = finish_reason
        return result


def safe_chat_completion(
    client,
    model: str,
    messages: list,
    temperature: float = 0.3,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = 12288,
    broadcast: bool = True,
) -> str:
    """带全局并发流控、智能错峰退避、实时 SSE 广播与精准分类熔断的 LLM 安全调用封装"""
    eff_timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT
    prompt_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
    fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "").strip()
    last_err = None
    for attempt in range(1, max_retries + 1):
        active_model = (fallback_model if (attempt >= 2 and fallback_model and fallback_model != model) else model)
        attempt_started = time.monotonic()
        try:
            with GLOBAL_LLM_SEMAPHORE:
                print(f"🌐 [LLM-API] 正在调用大模型 ({active_model} | 尝试 {attempt}/{max_retries} | 单次时限: {eff_timeout}s)...")
                kwargs = {
                    "model": active_model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": eff_timeout,
                    "stream": True,
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens

                stream = client.chat.completions.create(**kwargs)
                collected = []
                cb = get_stream_callback() if broadcast else None
                finish_reason = None
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            collected.append(delta.content)
                            if cb:
                                try:
                                    cb(delta.content)
                                except Exception:
                                    pass
                res_str = "".join(collected).strip()
                if res_str:
                    print(f"✅ [LLM-API] 收到大模型回复 (总字数: {len(res_str)}, finish_reason={finish_reason or 'unknown'})")
                    print(
                        f"📊 [LLM-UsageEstimate] model={active_model} elapsed_s={time.monotonic() - attempt_started:.2f} "
                        f"prompt_chars={prompt_chars} output_chars={len(res_str)} "
                        f"approx_tokens={(prompt_chars + len(res_str)) // 4} estimate_only=true"
                    )
                    return CompletionText(res_str, finish_reason)
                raise RuntimeError(f"上游接口返回空流式响应 (Empty stream from {active_model})")
        except Exception as e:
            last_err = e
            print(
                f"📊 [LLM-CallFailure] model={active_model} elapsed_s={time.monotonic() - attempt_started:.2f} "
                f"prompt_chars={prompt_chars} error={type(e).__name__}"
            )
            err_str = str(e).lower()
            if "timeout" in err_str or "timed out" in err_str or "524" in err_str:
                print(f"⚠️ [LLM 响应超时或网关截断 (HTTP 524) | 尝试 {attempt}/{max_retries}]: {e}")
            elif any(kw in err_str for kw in ["502", "503", "overloaded", "bad gateway"]):
                print(f"⚠️ [LLM 上游代理过载 (HTTP 502/503) | 尝试 {attempt}/{max_retries}]: {e}")
            elif "429" in err_str or "rate limit" in err_str:
                print(f"⚠️ [LLM 调用频次受限 (HTTP 429) | 尝试 {attempt}/{max_retries}]: {e}")
            else:
                print(f"⚠️ [LLM 调用异常 | 尝试 {attempt}/{max_retries}]: {e}")

            if attempt < max_retries:
                # 针对服务器过载或网关超时 (502/503/524/Overloaded)，给予充足的错峰退避恢复窗口 (4.0s~9.0s)
                if any(kw in err_str for kw in ["502", "503", "524", "overloaded", "bad gateway", "429", "rate limit"]):
                    wait_time = min(4.0 * (1.6 ** (attempt - 1)), 10.0) + random.uniform(0.5, 1.5)
                    print(f"⏳ [LLM-Backoff] 上游代理服务过载/限流/超时，错峰退避 {wait_time:.1f}s 后进行第 {attempt + 1} 次重试...")
                else:
                    wait_time = min(2.0 * (1.5 ** (attempt - 1)), 6.0) + random.uniform(0.3, 0.8)
                    print(f"⏳ [LLM-Retry] 等待 {wait_time:.1f}s 后进行第 {attempt + 1} 次重试...")
                time.sleep(wait_time)

    err_str = str(last_err)
    if any(kw in err_str.lower() for kw in ["503", "502", "524", "bad gateway", "overloaded"]):
        return "⚠️ **【搭档提示】上游大模型 API 处于繁忙排队或瞬时过载/网关超时中 (Server Overloaded / HTTP 502/503/524)**\n\n上游中继算力正在释放，请稍候重新发送问题即可继续研讨。"
    elif "429" in err_str or "Rate limit" in err_str:
        return "⚠️ **【搭档提示】模型调用频次已达限制 (HTTP 429 Rate Limit)**\n\n请稍候 3~5 秒后再次提问。"
    else:
        return f"⚠️ **【搭档提示】大模型接口调用发生异常**: {err_str}\n\n建议稍后重新发送提问。"


_ORIGINAL_SAFE_CHAT_COMPLETION = safe_chat_completion


def _build_deterministic_evidence_slate(
    query: str,
    candidates: list,
    evidence: list,
    final_ans: str = ""
) -> str:
    """纯代码高性能多源学术资产与证据归纳提炼引擎 (0 API 消耗，0ms 耗时，杜绝中继过载)"""
    lines = [
        "### 📑 《多源学术资产与证据归纳备忘录》\n",
        "#### 1. 🔍 跨学科噪声过滤与有效性审查 (Noise Filtering)",
    ]

    valid_papers = []
    noise_papers = [{"title": ev.get("doc_name") or ev.get("title") or "资料"}
                    for ev in (evidence or []) if ev.get("excluded_as_irrelevant")]
    candidates = filter_irrelevant_candidates(candidates or [], evidence or [])

    # 处理候选文献
    for c in (candidates or []):
        title = (c.get("title") or "未命名文献").strip()
        summary = (c.get("summary") or "").strip()
        src = c.get("source", "online")
        authors = ", ".join((c.get("authors") or [])[:3])
        year = c.get("year") or "Recent"
        p_id = c.get("paper_id") or c.get("doi") or c.get("id") or ""

        item = {
            "title": title,
            "summary": summary,
            "src": src,
            "authors": authors,
            "year": year,
            "id": p_id,
        }
        valid_papers.append(item)

    # 处理主证据列表（保留原始 [i] 编号与证据深度，对齐 Consensus / Elicit 结构化核验）
    local_evidence = []
    depth_counts = {"full": 0, "abstract": 0, "commentary": 0, "metadata": 0}
    for idx, ev in enumerate(evidence or [], 1):
        if ev.get("excluded_as_irrelevant"):
            continue
        doc = ev.get("doc_name") or ev.get("title") or "资料切片"
        loc = ev.get("location_label") or ""
        txt = (ev.get("text") or ev.get("summary") or "").strip()
        level_raw = str(ev.get("evidence_level") or (ev.get("provenance") or {}).get("content_scope") or "").lower()
        src_type = str(ev.get("doc_type") or ev.get("source") or "").lower()
        year = ev.get("year") or (ev.get("provenance") or {}).get("year") or "-"
        if any(k in level_raw for k in ("web_commentary", "search_snippet")):
            depth_label = "网页/社区解读"
            depth_counts["commentary"] += 1
        elif any(k in level_raw for k in ("full", "local", "chunk", "transcript", "text")) or src_type in ("local", "pdf", "meeting_transcript"):
            depth_label = "全文/本地原文"
            depth_counts["full"] += 1
        elif "abstract" in level_raw or (txt and "metadata" not in level_raw):
            depth_label = "仅摘要"
            depth_counts["abstract"] += 1
        else:
            depth_label = "题录/元数据"
            depth_counts["metadata"] += 1
        local_evidence.append({
            "idx": idx,
            "doc": doc,
            "loc": loc,
            "year": year,
            "src": src_type or "evidence",
            "depth": depth_label,
            "text": txt[:300],
        })

    if noise_papers:
        noise_titles = "、".join([f"《{p['title']}》" for p in noise_papers[:2]])
        lines.append(f"- 剔除噪声：{noise_titles} 已被覆盖检查或审核明确判为与本轮任务无关，不作为回答依据；")

    total_valid = len(valid_papers) + len(local_evidence)
    lines.append(f"- 待核验资料记录：{total_valid} 项（候选与证据可能重叠）。本备忘录仅作导航，记录数量不代表任务覆盖或事实已核实。")
    if local_evidence:
        commentary_part = f" | 网页/社区解读 {depth_counts['commentary']} 项" if depth_counts["commentary"] else ""
        lines.append(
            f"- 证据深度分布（Consensus 核验标尺）：全文/本地原文 {depth_counts['full']} 项"
            f" | 仅摘要 {depth_counts['abstract']} 项{commentary_part} | 题录/元数据 {depth_counts['metadata']} 项"
        )
    lines.append("")

    lines.append("#### 2. 📚 资料层级分类 (Taxonomy & Categorization)")
    arxiv_list = [p for p in valid_papers if p["src"] == "arxiv"]
    journal_list = [p for p in valid_papers if p["src"] in ["openalex", "crossref"]]
    code_list = [p for p in valid_papers if p["src"] == "github"]

    if arxiv_list:
        lines.append("- 【前沿预印本 (arXiv)】：" + "；".join([f"{p['authors']}《{p['title']}》({p['year']})" for p in arxiv_list[:3]]))
    if journal_list:
        lines.append("- 【正式期刊/国际学术文献】：" + "；".join([f"{p['authors']}《{p['title']}》({p['year']})" for p in journal_list[:3]]))
    if local_evidence:
        lines.append("- 【主证据列表（含编号与深度）】：" + "；".join([f"[{ev['idx']}]《{ev['doc']}》({ev['depth']})" for ev in local_evidence[:5]]))
    if code_list:
        lines.append("- 【代码仓库元数据（尚未核验实现与基准）】：" + "；".join([f"《{p['title']}》" for p in code_list[:2]]))
    if not (arxiv_list or journal_list or local_evidence or code_list):
        lines.append("- 当前没有独立来源资料，不能用模型记忆补作专业事实依据。")
    lines.append("")

    lines.append("#### 3. 🔬 核心机理、方程与边界条件骨架 (Core Mechanisms & Boundary Slate)")
    if local_evidence:
        lines.append("| 编号 | 资料/文献 | 年份 | 来源与证据深度 | 核心内容与可核验依据 |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for ev in local_evidence:
            safe_doc = ev["doc"].replace("|", "｜").replace("\n", " ")[:60]
            safe_txt = (ev["text"][:160].replace("|", "｜").replace("\n", " ") or "仅有标识，暂无正文/摘要")
            loc_suffix = f" ({ev['loc']})" if ev["loc"] else ""
            lines.append(f"| [{ev['idx']}] | 《{safe_doc}》{loc_suffix} | {ev['year']} | {ev['src']} · {ev['depth']} | {safe_txt} |")
    for p in valid_papers[:4]:
        lines.append(f"- **《{p['title']}》** ({p['year']}):")
        if p['summary']:
            lines.append(f"  * 核心机制与发现: {p['summary'][:180]}...")

    if not valid_papers and not local_evidence and final_ans:
        lines.append("- 主研初稿是生成内容，只能辅助定位待核查主张，不能作为独立验证。")

    return "\n".join(lines)


def _build_deterministic_dialectical_inquiry(
    query: str,
    target_agent: str,
    final_ans: str = "",
    evidence_slate: str = ""
) -> str:
    """纯代码高性能学术博弈质询与去冗备忘生成 (委托至 prompts.build_deterministic_dialectical_inquiry)"""
    return prompts.build_deterministic_dialectical_inquiry(
        query=query,
        target_agent=target_agent,
        final_ans=final_ans,
        evidence_slate=evidence_slate,
    )


def normalize_academic_numbering(content: str) -> str:
    """确定性学术章节层级与序号自洽性校验器 (杜绝 '三、 2.2' 与 '3.2.1' 杂交错乱编号)"""
    if not content:
        return ""
    lines = content.split('\n')
    fixed_lines = []
    current_parent_section = None  # 如 '2.2'

    for line in lines:
        stripped = line.strip()
        # 1. 修复大纲杂交: '三、 2.2 扩展法则' -> '### 2.2 扩展法则' 或 '2.2 扩展法则'
        m_hybrid = re.match(r'^(#{1,4}\s*)?[一二三四五六七八九十]+、\s*(\d+\.\d+)(.*)$', stripped)
        if m_hybrid:
            hashes = m_hybrid.group(1) or '### '
            sec_num = m_hybrid.group(2)
            rest = m_hybrid.group(3)
            current_parent_section = sec_num
            fixed_lines.append(f'{hashes.strip()} {sec_num}{rest}'.strip())
            continue

        # 跟踪标准二级标题: '### 2.2 ...' 或 '2.2 ...'
        m_sec = re.match(r'^(#{1,4}\s*)?(\d+\.\d+)\s+([^\d].*)$', stripped)
        if m_sec:
            current_parent_section = m_sec.group(2)

        # 2. 修复子小节编号错乱 (如在 2.2 节下出现 3.2.1、4.2.1 或粗体 **3.2.1**)
        if current_parent_section:
            parent_parts = current_parent_section.split('.')
            if len(parent_parts) == 2:
                # 匹配普通文本或标题: '3.2.1 标题'
                m_sub = re.match(r'^(#{1,4}\s*)?(\d+)\.(\d+)\.(\d+)(\s+.*)$', stripped)
                if m_sub:
                    hashes = m_sub.group(1) or ''
                    p1, p2, p3 = m_sub.group(2), m_sub.group(3), m_sub.group(4)
                    rest = m_sub.group(5)
                    if (p1 != parent_parts[0] or p2 != parent_parts[1]) and p2 == parent_parts[1]:
                        correct_sec = f'{parent_parts[0]}.{parent_parts[1]}.{p3}'
                        fixed_lines.append(f'{hashes}{correct_sec}{rest}'.strip())
                        continue

                # 匹配粗体格式: '**3.2.1 标题**'
                m_bold = re.match(r'^(\*\*)(\d+)\.(\d+)\.(\d+)(\s+[^*]+\*\*.*)$', stripped)
                if m_bold:
                    p1, p2, p3 = m_bold.group(2), m_bold.group(3), m_bold.group(4)
                    rest = m_bold.group(5)
                    if (p1 != parent_parts[0] or p2 != parent_parts[1]) and p2 == parent_parts[1]:
                        correct_sec = f'{parent_parts[0]}.{parent_parts[1]}.{p3}'
                        fixed_lines.append(f'**{correct_sec}{rest}'.strip())
                        continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def _local_fallback_evidence(state, rag_engine=None, top_k=6, candidate_k=20):
    """Only perform a planned lookup that the initial retrieval has not executed."""
    if not (state.get("retrieval_plan") or {}).get("local_search") or (
            state.get("retrieval_coverage") or {}).get("search_history"):
        return []
    from cross_source_hybrid_retriever import CrossSourceHybridRetriever
    rag = rag_engine or MasterEngineRegistry.get_rag_engine()
    decision = task_decision(state)
    chunks = []
    for query in state.get("retrieval_queries") or [decision.get("search_query") or state["user_query"]]:
        try:
            found = rag.search(query, top_k=top_k, candidate_k=candidate_k,
                doc_filter=state.get("active_doc"),
                negative_keywords=decision.get("negative_constraints") or [])
            # Reuse the main retrieval's date interpretation; missing dates stay visible as uncertain.
            found = CrossSourceHybridRetriever.filter_local_dates(found, decision.get("time_constraint"))
            chunks, _ = merge_evidence(chunks, found)
        except Exception as exc:
            print(f"[LocalFallback] {type(exc).__name__}")
    return chunks


def _selected_shared_evidence(state, selected_ids):
    """Hydrate selected source records; historical generated prose is never a source."""
    selected_ids = set(selected_ids or [])
    pool = [item for item in state.get("shared_pool") or [] if isinstance(item, dict)
            and item.get("item_id") in selected_ids]
    if state.get("session_id") and selected_ids:
        try:
            from session_manager import SessionManager
            loaded = SessionManager.get_shared_pool_items(state["session_id"], list(selected_ids))
            pool = loaded or pool
        except Exception as exc:
            print(f"[SharedPool] {type(exc).__name__}; using available in-memory records")
    result = []
    for item in pool:
        if item.get("kind") == "turn_record":
            result.extend((item.get("state") or {}).get("retrieved_evidence") or [])
        elif item.get("kind") == "candidate":
            candidate = item.get("candidate") or {}
            if candidate:
                result.append({**candidate, "doc_name": candidate.get("title"),
                    "doc_type": candidate.get("source"), "text": candidate.get("summary") or ""})
        elif item.get("kind", "evidence") == "evidence":
            result.append(dict(item.get("resource") or item))
    result.extend(dict(item) for item in state.get("shared_resources") or []
        if isinstance(item, dict) and (item.get("resource_id") in selected_ids or item.get("item_id") in selected_ids))
    result.extend({**item, "doc_name": item.get("title"), "doc_type": item.get("source"),
                   "text": item.get("summary") or ""}
        for item in state.get("recent_paper_candidates") or [] if isinstance(item, dict)
        and (item.get("shared_item_id") in selected_ids or item.get("resource_id") in selected_ids))
    result = [dict(item) for item in result if isinstance(item, dict)]
    for item in result:
        item.pop("excluded_as_irrelevant", None)
        item["reused_from_shared_pool"] = True
    return merge_evidence([], result)[0]


def document_rag_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 B: 全格式资料知识库检索与页码证据问答 (结合高密度结构化记忆摘要，0 轮堆叠)"""
    query = state["user_query"]
    history_summary = (state.get("history_summary") or "").strip()
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()

    evidence_chunks = list(state.get("retrieved_evidence") or [])
    if not evidence_chunks:
        evidence_chunks = _local_fallback_evidence(state, top_k=6, candidate_k=20)

    if not evidence_chunks:
        context_str = "本地知识库中未找到直接相关的段落。请说明资料缺口，不凭模型记忆生成专业结论；仅可依据用户明确提供的前提推导并说明范围。"
    else:
        print(f"🔍 [DocumentRAG] 命中 {len(evidence_chunks)} 个高相关度段落：")
        context_str = prompts.format_evidence_for_prompt(evidence_chunks)

    decision = task_decision(state)
    requires_long_form = decision.get("requires_long_form", False)
    long_form_type = decision.get("long_form_type")
    framework_ref = decision.get("framework_reference")
    doc_name = framework_ref.get("doc_name", "参考文档") if (framework_ref and framework_ref.get("usage") == "structure_only") else None
    target_topic = decision.get("target_topic")
    negative_constraints = decision.get("negative_constraints") or []
    active_subject = state.get("active_subject")

    system_prompt = build_lead_tutor_system_prompt(
        assistant_name=get_assistant_name(),
        requires_long_form=requires_long_form,
        long_form_type=long_form_type,
        doc_name=doc_name,
        target_topic=target_topic or query,
        negative_constraints=negative_constraints,
        active_subject=active_subject,
        history_summary=history_summary,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"参考资料证据：\n{context_str}\n\n" + task_prompt(state)}
    ]
    answer = safe_chat_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=0.2,
        timeout=get_llm_timeout(),
    )
    
    answer_checks = completion_checks(answer)
    answer = normalize_academic_numbering(answer)

    return {
        "retrieved_evidence": evidence_chunks,
        "final_answer": answer,
        "delivery_checks": {**(state.get("delivery_checks") or {}), **answer_checks},
        "status": "answered_by_doc_rag",
    }


def meeting_summary_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 A: 会议流水处理与待办决议提炼"""
    query = state["user_query"]
    summarizer = MasterEngineRegistry.get_meeting_summarizer()

    records_dir = BASE_DIR / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    latest_records = list(records_dir.glob("record_*.jsonl"))

    requested = state.get("meeting_record_path")
    if requested:
        latest_records = [Path(requested)] if Path(requested).is_file() else []
    evidence = list(state.get("retrieved_evidence") or [])
    status = "answered_by_meeting_summarizer"
    if latest_records:
        latest_records.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        target_record = latest_records[0]
        print(f"📋 [MeetingAgent] 正在对最新会话流水执行归纳: {target_record.name}")
        record_text = target_record.read_text(encoding="utf-8")
        evidence, _ = merge_evidence(evidence, [{"doc_name": target_record.name,
            "doc_type": "meeting_transcript", "location_label": str(target_record),
            "text": record_text, "evidence_level": "transcript",
            "provenance": {"source": "recorded_transcript", "path": str(target_record)}}])
        try:
            report = summarizer.summarize_session(target_record, task_context=task_prompt(state))
            answer = MeetingSummarizer._render_markdown_report(summarizer, report)
        except Exception as exc:
            answer = f"本轮会议归纳未完成（{type(exc).__name__}），原始记录已保留。"
            status = "meeting_summary_unavailable"
    else:
        answer = "当前没有正在进行的会议录音流水。您可以通过 `python voice_langgraph_agent.py` 开启录制。"

    return {
        "final_answer": answer,
        "retrieved_evidence": evidence,
        "meeting_record_path": str(target_record) if latest_records else requested,
        "status": status,
    }


def cross_modal_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 C: 跨模态音视频讲解与课件 PPT 对照"""
    query = state["user_query"]
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()

    print(f"🔄 [CrossModal] 执行音视频与课件对齐检索 (查询词: \"{query}\")...")
    evidence_chunks = list(state.get("retrieved_evidence") or [])
    if not evidence_chunks:
        evidence_chunks = _local_fallback_evidence(state, top_k=2)
    context_str = prompts.format_evidence_for_prompt(evidence_chunks) or "尚未取得课件内容，请说明可定位的信息与资料缺口。"
    system_prompt = build_lead_tutor_system_prompt(assistant_name=get_assistant_name(),
        history_summary=state.get("history_summary"), active_subject=state.get("active_subject"))
    system_prompt += "\n结合本轮任务解释讲解与课件的关系，按实际资料标明页码、位置与来源。"
    answer = safe_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"检索到的课件信息：\n{context_str}\n\n" + task_prompt(state)},
        ],
        temperature=0.1,
        timeout=get_llm_timeout(),
    )
    return {
        "retrieved_evidence": evidence_chunks,
        "final_answer": answer,
        "delivery_checks": {**(state.get("delivery_checks") or {}), **completion_checks(answer)},
        "status": "answered_by_cross_modal",
    }


def general_tutor_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 D: 通用学科辅导、公式推导与概念拆解 (结合高密度结构化记忆摘要，0 轮堆叠)"""
    query = state["user_query"]
    alignment = (state.get("retrieval_plan") or {}).get("task_alignment") or {}
    if alignment.get("status") == "unresolved":
        return {"final_answer": "本轮主体与任务尚未能和原始内容可靠对齐，未执行检索或生成长篇回答。",
                "retrieved_evidence": [], "status": "task_alignment_unresolved"}
    history_summary = (state.get("history_summary") or "").strip()
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()
    decision = task_decision(state)
    intent = decision.get("intent", "concept_simplify")
    target_agent_name = (
        state.get("target_agent")
        or (decision.get("target_agent") if isinstance(decision, dict) else getattr(decision, "target_agent", "StudyPlannerAgent"))
        or "StudyPlannerAgent"
    )

    display_tutor_q = (query.split("\n\n[附带图片内容]")[0] + " [已附带图片]") if "\n\n[附带图片内容]" in query else query[:80]
    print(f"🎓 [TutorAgent] 启动学术辅导与推导引擎 (席位: {target_agent_name} | 意图: {intent} | 查询: \"{display_tutor_q}\")...")

    evidence_chunks = list(state.get("retrieved_evidence") or [])
    dialogue_log = list(state.get("agent_dialogue_history") or [])

    # 🔀 动态跨学科特派子代理并发派生与合流 (Dynamic Cross-Disciplinary Specialist Spawning & Convergence)
    is_cross = bool(decision.get("is_cross_disciplinary") if isinstance(decision, dict) else getattr(decision, "is_cross_disciplinary", False))
    sub_qs = (decision.get("sub_queries") if isinstance(decision, dict) else getattr(decision, "sub_queries", [])) or []
    disciplines = (decision.get("disciplines") if isinstance(decision, dict) else getattr(decision, "disciplines", [])) or []

    retrieval_plan = state.get("retrieval_plan") or {}
    tutor_query = (decision.get("search_query") if isinstance(decision, dict) else getattr(decision, "search_query", "")) or query
    tutor_queries = state.get("retrieval_queries") or [tutor_query]
    need_local_search = bool(retrieval_plan.get("local_search", False))
    need_online_search = bool(retrieval_plan.get("online_search", False))

    initial_search_done = bool((state.get("retrieval_coverage") or {}).get("search_history"))
    if not evidence_chunks and need_local_search and not initial_search_done:
        evidence_chunks = _local_fallback_evidence(state, top_k=4, candidate_k=15)

    recent_candidates = list(state.get("recent_paper_candidates") or [])

    # 🤝 多智能体点对点协作总线 (Agent Collaboration Bus):
    # 只有独立检索计划决定需要在线资料且当前没有可复用证据时，才委托在线补查。
    if not evidence_chunks and need_online_search and not initial_search_done:
        try:
            print("🤝 [Multi-Agent Bus] 主研学者向在线文献专员发起点对点前沿基线探查请求...")
            paper_agent = MasterEngineRegistry.get_paper_search_agent()
            eng_query = (decision.get("online_search_query") or "").strip()
            chinese_query = "; ".join(str(q).strip() for q in tutor_queries if str(q).strip()) or str(tutor_query).strip()
            sub_qs = decision.get("sub_queries") or []
            if sub_qs:
                for sq in sub_qs:
                    if sq.get("is_core_bottleneck", False) or sq.get("retrieval_strategy") == "online_probe":
                        chinese_query = sq.get("query")
                        break

            raw_papers = []
            if hasattr(paper_agent, "search_multisource"):
                bundle = paper_agent.search_multisource(
                    query=chinese_query,
                    include_github=bool(retrieval_plan.get("include_github", False)),
                    include_web=bool(retrieval_plan.get("include_web", False)),
                    include_datasets=bool(retrieval_plan.get("include_datasets", False)),
                    time_constraint=decision.get("time_constraint"),
                    english_query=eng_query if len(tutor_queries) == 1 else None,
                )
                if isinstance(bundle, dict):
                    raw_papers.extend(bundle.get("arxiv_papers", []))
                    raw_papers.extend(bundle.get("openalex_papers", []))
                    raw_papers.extend(bundle.get("crossref_papers", []))
                    raw_papers.extend(bundle.get("github_repos", []))
                    raw_papers.extend(bundle.get("web_results", []))
                    raw_papers.extend(bundle.get("datasets", []))

            # 兼容仅 mock 了 search_arxiv 的单元测试或降级调用
            if not raw_papers and hasattr(paper_agent, "search_arxiv"):
                arxiv_res = paper_agent.search_arxiv(eng_query or chinese_query, max_results=3)
                if isinstance(arxiv_res, list):
                    for item in arxiv_res:
                        raw_papers.append(item.model_dump() if hasattr(item, "model_dump") else item)

            # 强化多源去重与元数据属性融合机制 (基于规范化指纹与模糊标题相似度)
            from cross_source_hybrid_retriever import calculate_title_similarity
            unique_papers = []
            for p in raw_papers:
                p_dict = p if isinstance(p, dict) else p.model_dump() if hasattr(p, "model_dump") else vars(p)
                title = (p_dict.get("title") or "").strip()
                if not title:
                    continue
                p_id = str(p_dict.get("paper_id") or p_dict.get("doi") or p_dict.get("id") or "").strip()

                is_duplicate = False
                for existing in unique_papers:
                    ex_id = str(existing.get("paper_id") or existing.get("doi") or existing.get("id") or "").strip()
                    if p_id and ex_id and p_id == ex_id:
                        is_duplicate = True
                        if not existing.get("doi") and p_dict.get("doi"):
                            existing["doi"] = p_dict.get("doi")
                        if not existing.get("venue") and p_dict.get("venue"):
                            existing["venue"] = p_dict.get("venue")
                        break
                    sim = calculate_title_similarity(title, existing.get("title", ""))
                    if sim >= 0.85:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_papers.append(p_dict)

            if unique_papers:
                # 候选库完整保留所有去重后的前沿文献，供用户在界面浏览与点播下载
                recent_candidates.extend(unique_papers)

                dialogue_log.append({
                    "sender": target_agent_name,
                    "receiver": "OnlinePaperSearchAgent",
                    "round": 0,
                    "action": "REQUEST_BASELINE_PAPERS",
                    "summary": f"🤖 {target_agent_name} 向文献发现专员发起前沿基线求助，调取与课题相关的权威实证锚点。"
                })
                # Preserve every returned unique source for author and reviewer alike.
                for p_dict in unique_papers:
                    title = p_dict.get("title", "学术文献")
                    p_id = p_dict.get("paper_id") or p_dict.get("doi") or p_dict.get("id") or "online"
                    authors = p_dict.get("authors") or []
                    year = p_dict.get("year") or "Recent"
                    summary = p_dict.get("summary") or ""
                    url = p_dict.get("url") or p_dict.get("pdf_url") or ""
                    src = p_dict.get("source", "online")

                    if src == "arxiv":
                        loc_label = f"在线预印本文献 (arXiv:{p_id})"
                    elif src == "openalex":
                        loc_label = f"OpenAlex 权威文献 ({p_dict.get('venue') or '顶刊'})"
                    elif src == "crossref":
                        loc_label = f"CrossRef 权威收录 (DOI: {p_dict.get('doi') or p_id})"
                    else:
                        loc_label = f"在线学术文献 ({src})"

                    chunk = {
                        "doc_name": title,
                        "location_label": loc_label,
                        "text": f"【论文题目】: 《{title}》\n【主要作者】: {', '.join(authors[:3])} ({year})\n【核心摘要/方案】: {summary}\n【在线链接/DOI】: {url}",
                        "score": 0.95,
                        "url": url,
                        "doc_type": src,
                        "evidence_level": p_dict.get("evidence_level") or ("web_search_snippet" if src == "web" else "dataset_repository_metadata" if src == "huggingface_dataset" else "abstract_or_bibliographic_metadata"),
                        "version_ref": p_dict.get("version_ref"),
                        "version_ref_type": p_dict.get("version_ref_type"),
                        "provenance": {"source": src, "url": url, "content_scope": p_dict.get("content_scope") or p_dict.get("evidence_level") or "search summary"},
                    }
                    evidence_chunks.append(chunk)
                dialogue_log.append({
                    "sender": "OnlinePaperSearchAgent",
                    "receiver": target_agent_name,
                    "round": 0,
                    "action": "DELIVER_BASELINE_EVIDENCE",
                    "summary": f"🌐 文献专员成功为研究方案交接了 {len(unique_papers)} 篇多源在线权威基线文献支撑。"
                })
                print(f"✅ [Multi-Agent Bus] 文献专员已成功为主研方案注入 {len(unique_papers)} 篇权威基线！")
        except Exception as e:
            print(f"⚠️ [Multi-Agent Bus Delegation Note]: {e}")

    specialist_insights = []
    specialist_results = []
    if is_cross and len(sub_qs) >= 2:
        from concurrent.futures import ThreadPoolExecutor
        print(f"🧬 [GeneralTutor] 检测到多学科交叉课题 (涉及学科: {disciplines})，正在动态派生 {len(sub_qs)} 路领域特派专家子代理并行推演...")

        def _run_specialist(sq):
            disc = sq.get("discipline", "交叉学科")
            sub_q = sq.get("query", query)
            agent_name = f"Specialist{re.sub(r'[^A-Za-z0-9_]+', '', disc)}Agent"
            if agent_name == "SpecialistAgent":
                agent_name = "SpecialistSpecialistAgent"
            spec_length = (
                "按所需长篇体裁充分展开该领域的方法、依据与边界条件。"
                if decision.get("requires_long_form", False)
                else "不设固定字数，充分解释该领域的关键机制、方法与推导，按实际需要使用公式。"
            )
            spec_prompt = (
                f"你是一名专注于【{disc}】领域的资深理论科学家与特派专家子代理。\n"
                + task_prompt(state) + "\n"
                + f"你受托专攻的子命题：{sub_q}\n"
                + prompts.EVIDENCE_GUIDANCE + "\n"
                + f"本轮检索资料：\n{prompts.format_evidence_for_prompt(evidence_chunks)}\n"
                f"{spec_length}"
            )
            try:
                res = safe_chat_completion(
                    client=client,
                    model=model,
                    messages=[{"role": "user", "content": spec_prompt}],
                    temperature=0.2,
                    max_tokens=6144,
                    broadcast=False,
                    timeout=get_llm_timeout(),
                )
                if not res.strip() or "【搭档提示】" in res:
                    raise RuntimeError("specialist_response_unavailable")
                return {
                    "discipline": disc,
                    "agent_name": agent_name,
                    "sub_query": sub_q,
                    "analysis": res.strip(), "status": "partial" if getattr(res, "finish_reason", None) in {"length", "content_filter"} else "completed", **completion_checks(res)
                }
            except Exception as e:
                print(f"⚠️ [Specialist Subagent Error - {disc}]: {e}")
                return {
                    "discipline": disc,
                    "agent_name": agent_name,
                    "sub_query": sub_q,
                    "analysis": "", "status": "failed", "error": type(e).__name__
                }

        with ThreadPoolExecutor(max_workers=min(len(sub_qs), 2)) as pool:
            futures = []
            for sq in sub_qs:
                futures.append(pool.submit(_run_specialist, sq))
                time.sleep(0.6)  # 阶梯错峰，配合 GLOBAL_LLM_SEMAPHORE 彻底消除瞬时并发洪峰
            specialist_results = [f.result() for f in futures]

        for s_res in specialist_results:
            agent_name = s_res["agent_name"]
            disc = s_res["discipline"]
            analysis = s_res["analysis"]
            specialist_insights.append(f"【特派专家席位：{agent_name} ({disc})】\n" +
                (analysis + ("\n该专家输出未完整结束，保留已收到内容。" if s_res["status"] == "partial" else "") if analysis else f"子任务 {s_res['sub_query']} 未完成：{s_res.get('error', 'empty_response')}；由主作者结合现有资料继续处理。"))
            dialogue_log.append({
                "sender": agent_name,
                "receiver": target_agent_name,
                "round": 0,
                "action": "DELIVER_DISCIPLINARY_FOUNDATION" if s_res["status"] != "failed" else "SPECIALIST_FAILED",
                "status": s_res["status"],
                "summary": f"{agent_name} 子任务：{s_res['sub_query']}；状态：{s_res['status']}。"
            })
            print(f"[Specialist Subagent] {agent_name}: {s_res['status']}")


    decision = task_decision(state)
    requires_long_form = decision.get("requires_long_form", False)
    long_form_type = decision.get("long_form_type")
    framework_ref = decision.get("framework_reference")
    doc_name = framework_ref.get("doc_name", "参考文档") if (framework_ref and framework_ref.get("usage") == "structure_only") else None
    target_topic = decision.get("target_topic")
    negative_constraints = decision.get("negative_constraints") or []
    active_subject = state.get("active_subject")

    system_prompt = build_lead_tutor_system_prompt(
        assistant_name=get_assistant_name(),
        requires_long_form=requires_long_form,
        long_form_type=long_form_type,
        doc_name=doc_name,
        target_topic=target_topic or query,
        negative_constraints=negative_constraints,
        active_subject=active_subject,
        history_summary=history_summary,
    )

    user_prompt = task_prompt(state)
    extracted_entities = decision.get("extracted_entities") if isinstance(decision, dict) else getattr(decision, "extracted_entities", {})
    if extracted_entities:
        task_items = []
        for k, v in extracted_entities.items():
            if isinstance(v, list):
                val_s = "；".join(str(x) for x in v)
            elif isinstance(v, dict):
                val_s = json.dumps(v, ensure_ascii=False)
            else:
                val_s = str(v)
            task_items.append(f"- 【{k}】：{val_s}")
        user_prompt += "\n\n意图解析提取的实体线索（模型推断，仅供理解原话，不增加用户任务）：\n" + "\n".join(task_items)

    if specialist_insights:
        spec_text = "\n\n".join(specialist_insights)
        user_prompt = f"【多学科特派专家解耦推演输入】：\n{spec_text}\n\n" + user_prompt
    rag_context_str = prompts.format_evidence_for_prompt(evidence_chunks)
    if rag_context_str:
        user_prompt = (
            f"【本轮已检索资料（请将检索资料与学科公认基础理论有机结合，按资料编号 [1]、[2] 规范标注具体文献论断，无需逐条宣称未查看全文）】：\n{rag_context_str}\n\n"
            + user_prompt
        )
    elif retrieval_plan.get("requires_evidence", True):
        user_prompt += "\n\n【证据状态说明】：本轮未检出直接匹配的外部文献。请基于本学科公认的基础理论、数学公式推导与经典方法直接作答；若涉及需要特定最新文献支撑的具体实验数值，客观说明检索边界即可，切勿堆砌免责声明。"
    coverage = state.get("retrieval_coverage") or {}
    if coverage.get("status") in {"unresolved", "gaps_found", "budget_exhausted", "assessment_unavailable"} or coverage.get("unresolved"):
        gaps = coverage.get("unresolved") or [coverage.get("summary", "部分子项外部文献未完全覆盖")]
        user_prompt += "\n\n【检索覆盖提示（仅供把握论述边界，切勿在正文中反复自证缺陷或刷屏免责）】：\n- " + "\n- ".join(str(x) for x in gaps)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    answer = safe_chat_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=0.3,
        timeout=get_llm_timeout(),
    )
    answer_checks = completion_checks(answer)
    answer = normalize_academic_numbering(answer)

    dialogue_log.append({
        "sender": target_agent_name,
        "receiver": "DialecticalInterrogator",
        "round": 0,
        "action": "SUBMIT_INITIAL_DRAFT",
        "summary": f"🤖 {target_agent_name} 完成理论推演/方案初稿生成，提交学术博弈质询席位进行证据链审视与去冗研讨。"
    })

    return {
        "retrieved_evidence": evidence_chunks,
        "recent_paper_candidates": recent_candidates,
        "final_answer": answer,
        "dialectical_notes": None,
        "reflection_notes": None,
        "agent_dialogue_history": dialogue_log,
        "delivery_checks": {**(state.get("delivery_checks") or {}), **answer_checks},
        "specialist_results": specialist_results,
        "status": "answered_by_tutor",
    }


def evidence_synthesis_node(state: MasterAgentState) -> Dict[str, Any]:
    """学术证据与多源资料整理归纳席位 (Evidence & Multi-Source Material Synthesis Node)
    
    核心职责：
    1. 跨学科噪声清洗：审查检索或抽取的学术论文、专著书籍、教材讲义、技术规范等素材，过滤领域漂移与误检内容；
    2. 多源资料分级分类：自动划分为经典专著/权威教材、同行评审正刊、最新预印本、技术规范与综述；
    3. 核心机制骨架提炼：抽取关键物理方程、数学定理、实验参数与理论冲突点，构建高信噪比证据基石。
    """
    query = state.get("user_query", "")
    decision = task_decision(state)
    intent = decision.get("intent", "")
    evidence_list = state.get("retrieved_evidence") or []
    candidates_list = state.get("recent_paper_candidates") or []
    final_ans = state.get("final_answer", "")
    target_agent = state.get("target_agent") or ""
    dialogue_log = list(state.get("agent_dialogue_history") or [])

    # 若为纯系统控制指令或完全没有任何多源检索素材与内容，则跳过归纳
    if (target_agent == "SystemControlAgent"
            or ((state.get("retrieval_plan") or {}).get("task_alignment") or {}).get("status") == "unresolved"
            or (not evidence_list and not candidates_list and not final_ans)):
        return {
            "synthesized_evidence_slate": None,
            "status": "evidence_synthesis_skipped",
        }

    # Keep the authoritative evidence order and full inspected content.
    candidates_list = filter_irrelevant_candidates(candidates_list, evidence_list)
    raw_materials_text = prompts.format_evidence_for_prompt(evidence_list or candidates_list)
    if not raw_materials_text and final_ans:
        raw_materials_text = f"主研初稿（生成内容，仅供导航，不是独立证据）：\n{final_ans}"

    enable_llm_synthesis = os.getenv("ENABLE_LLM_EVIDENCE_SYNTHESIS", "false").lower() == "true"
    if not enable_llm_synthesis:
        clean_slate = _build_deterministic_evidence_slate(
            query=query,
            candidates=candidates_list,
            evidence=evidence_list,
            final_ans=final_ans,
        )
        print(f"📑 [EvidenceSynthesis] 纯代码高性能证据归纳引擎完成资料去噪与结构化提炼 (0 API 消耗，0ms 耗时)")
        dialogue_log.append({
            "sender": "EvidenceSynthesisAgent",
            "receiver": "DialecticalInterrogator",
            "round": 1,
            "action": "EVIDENCE_CLEANSING_AND_SYNTHESIS",
            "summary": "整理归纳智能体完成多源资料去噪、分级分类及核心机制骨架提炼，沉淀高信噪比证据备忘录。"
        })
        return {
            "synthesized_evidence_slate": clean_slate,
            "agent_dialogue_history": dialogue_log,
            "status": "evidence_synthesis_completed",
        }

    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()

    print(f"📑 [EvidenceSynthesis] 启动多源证据整理归纳智能体：正在清洗跨学科噪声、分类打标并提炼机理骨架...")

    prompt = prompts.build_evidence_synthesis_prompt(task_prompt(state), raw_materials_text)

    try:
        slate_raw = safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            broadcast=False,
            max_tokens=600,
            timeout=get_llm_timeout(),
        )
        clean_slate = slate_raw.strip()

        dialogue_log.append({
            "sender": "EvidenceSynthesisAgent",
            "receiver": "DialecticalInterrogator",
            "round": 1,
            "action": "EVIDENCE_CLEANSING_AND_SYNTHESIS",
            "summary": "整理归纳智能体完成多源资料去噪、分级分类及核心机制骨架提炼，沉淀高信噪比证据备忘录。"
        })

        return {
            "synthesized_evidence_slate": clean_slate,
            "agent_dialogue_history": dialogue_log,
            "status": "evidence_synthesis_completed",
        }
    except Exception as e:
        print(f"⚠️ [EvidenceSynthesis Note]: {e}")
        return {
            "synthesized_evidence_slate": None,
            "agent_dialogue_history": dialogue_log,
            "status": "evidence_synthesis_fallback",
        }


def _blocking_review_findings(findings: Any) -> List[Dict[str, Any]]:
    """Return located review findings for reporting, without authorizing a rewrite."""
    return [
        finding for finding in findings if isinstance(finding, dict)
        and str(finding.get("severity", "")).casefold() == "blocking"
        and str(finding.get("location") or "").strip()
        and str(finding.get("reason") or "").strip()
    ] if isinstance(findings, list) else []


def _locate_review_basis(findings: Any, query: str, answer: str,
                         evidence: List[Dict[str, Any]], visual_context: str = "") -> List[Dict[str, Any]]:
    """Locate quoted review grounds; locating a quote does not prove the review's interpretation."""
    def normalize(value):
        return re.sub(r"\s+", " ", str(value or "")).strip()

    located = []
    for finding_id, finding in enumerate(findings if isinstance(findings, list) else [], 1):
        if not isinstance(finding, dict):
            continue
        basis = []
        refs = finding.get("basis")
        for ref in refs if isinstance(refs, list) else []:
            if not isinstance(ref, dict):
                continue
            source, quote = ref.get("source"), str(ref.get("quote") or "").strip()
            if source in ("request", "answer", "visual_context", "attachment"):
                content = (query + "\n" + visual_context if source == "request" else
                           answer if source == "answer" else visual_context)
                item = {"source": source, "quote": quote}
            elif source == "evidence":
                idx = ref.get("evidence_id")
                if type(idx) is not int or not 1 <= idx <= len(evidence):
                    continue
                record = evidence[idx - 1]
                if record.get("excluded_as_irrelevant"):
                    continue
                content = "\n".join(str(record.get(field) or "") for field in ("text", "summary"))
                item = {"source": source, "evidence_id": idx, "quote": quote}
            else:
                continue
            if quote and normalize(quote) in normalize(content) and item not in basis:
                basis.append(item)
        needs_source = finding.get("kind") in ("factual", "evidence")
        has_basis = bool(basis) and (not needs_source or any(ref["source"] == "evidence" for ref in basis))
        located.append({**finding, "finding_id": finding_id, "basis": basis,
                        "basis_status": "located" if has_basis else "unverified"})
    return located


def _evidence_retrieval_budget(state: Dict[str, Any], env_key: str) -> int:
    configured = os.getenv(env_key)
    if configured is not None:
        try:
            return max(0, min(int(configured), 20))
        except ValueError:
            pass
    return max(8, min(20, len(task_decision(state).get("task_segments") or []) * 2))


def _review_search_stalled(state: Dict[str, Any], checks: List[Dict[str, Any]]) -> bool:
    attempts = [item for item in (state.get("review_retrieval_log") or [])
                if item.get("result") in {"no_new_evidence", "evidence_added"}]
    if len(attempts) < 2:
        return False
    for request in (state.get("review_verdict") or {}).get("evidence_requests") or []:
        if request.get("requirement") and not any(
            retrieval_query_key(str(p.get("requirement") or "")) == retrieval_query_key(request["requirement"])
            for p in attempts):
            return False
        if request.get("strategy") and not any(
            retrieval_query_key(str(p.get("strategy") or "")) == retrieval_query_key(request["strategy"])
            and retrieval_query_key(str(p.get("requirement") or "")) == retrieval_query_key(str(request.get("requirement") or ""))
            for p in attempts):
            return False
    if all(p.get("result") == "no_new_evidence" for p in attempts[-2:]):
        return True
    def signature(items):
        return {retrieval_query_key(str(c.get("requirement") or "")): (
            {"missing": 0, "unclear": 1, "met": 2}.get(c.get("result"), 0),
            {retrieval_query_key(f) for f in (
                [c["supported_facts"]] if isinstance(c.get("supported_facts"), str)
                else c.get("supported_facts") if isinstance(c.get("supported_facts"), list) else [])
             if isinstance(f, str) and f.strip()})
            for c in items if isinstance(c, dict) and c.get("requirement")}
    current = signature(checks)
    return bool(current and all(not any(
        rank > signature(p.get("task_checks") or []).get(key, (0, set()))[0]
        or facts - signature(p.get("task_checks") or []).get(key, (0, set()))[1]
        for key, (rank, facts) in current.items()) for p in attempts[-2:]))


def _qualify_review_retrieval(findings, checks, requests, required_task_count=0):
    """Keep observations, but execute only linked omissions or specific claims.

    This validates the review's internal consistency and quoted inputs, not the
    semantic truth of its conclusions. Optional depth is never a delivery gate.
    """
    def ids(value, limit):
        values = value if isinstance(value, list) else [value]
        return list(dict.fromkeys(i for i in values if type(i) is int and 1 <= i <= limit))

    findings = [dict(f) for f in findings]
    checks_by_id = {check.get("check_id", index): check for index, check in enumerate(checks, 1)}
    for finding in findings:
        check_ids = ids(finding.get("check_ids"), max(checks_by_id, default=0))
        check_ids = [idx for idx in check_ids if idx in checks_by_id]
        finding["check_ids"] = check_ids
        if finding.get("kind") == "completeness":
            linked = [checks_by_id[i] for i in check_ids]
            if linked and all(c.get("result") == "met" for c in linked):
                finding.update(reported_kind="completeness", reported_severity=finding.get("severity"),
                    kind="advisory", severity="minor", scope_status="optional_extension")
            elif (linked and any(c.get("result") == "missing" for c in linked)
                  and (not required_task_count or any(c.get("task_ids") for c in linked if c.get("result") == "missing"))
                  and finding.get("basis_status") == "located"
                  and any(b.get("source") in {"request", "visual_context", "attachment"}
                          for b in finding.get("basis") or [])):
                finding["scope_status"] = "missing_task"
            else:
                finding["scope_status"] = "unverified_requirement"
    accepted, deferred = [], []
    findings_by_id = {f["finding_id"]: f for f in findings}
    for item in requests:
        request = dict(item)
        linked_ids = ids(request.get("finding_ids"), max(findings_by_id, default=0))
        linked_ids = [idx for idx in linked_ids if idx in findings_by_id]
        linked = [findings_by_id[i] for i in linked_ids]
        purpose = request.get("purpose")
        if purpose == "missing_task":
            valid = any(f.get("scope_status") == "missing_task" for f in linked)
        elif purpose == "verify_claim":
            # A concrete answer claim can need verification even when the task
            # itself is covered and no contradictory source is available yet.
            valid = any(f.get("kind") in {"factual", "evidence"}
                        and any(b.get("source") == "answer" for b in f.get("basis") or []) for f in linked)
        else:
            valid = False
        request["finding_ids"] = linked_ids
        request["retrieval_decision"] = "actionable" if valid else "record_only"
        if valid:
            accepted.append(request)
        else:
            request["decision_reason"] = "未关联已定位的回答主张或原任务缺项；保留为审核记录，不自动补查。"
            deferred.append(request)
    return findings, accepted, deferred


def dialectical_inquiry_node(state: MasterAgentState) -> Dict[str, Any]:
    """审核记录具体问题，可提出补查；永不要求重写已有正文。"""
    final_ans = state.get("final_answer", "")
    decision = task_decision(state)
    intent = decision.get("intent", "")
    query = state.get("user_query", "")
    target_agent = state.get("target_agent") or "StudyPlannerAgent"
    evidence_list = state.get("retrieved_evidence", [])

    # 仅跳过无答案、错误回退或系统控制输出；普通任务都由审核者按任务复杂度调整检查深度。
    is_error_ans = any(kw in final_ans for kw in ["【搭档提示】", "Server Overloaded", "HTTP 502", "HTTP 503", "HTTP 524"])
    if (target_agent == "SystemControlAgent"
            or ((state.get("retrieval_plan") or {}).get("task_alignment") or {}).get("status") == "unresolved"
            or not final_ans or not final_ans.strip() or is_error_ans):
        return {
            "status": "dialectical_inquiry_skipped",
            "dialectical_notes": None,
            "peer_review_notes": None,
            "agent_dialogue_history": state.get("agent_dialogue_history", []),
            "retrieved_evidence": evidence_list,
        }

    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()
    dialogue_log = list(state.get("agent_dialogue_history") or [])

    print(f"🧐 [DialecticalInquiry] 审核 {target_agent} 的任务匹配、展开范围、理论依据与完整性...")

    # 共享池与已检索证据作为唯一权威证据源：对候选资源做 ID/标题去重差集，不再重复拼接衍生 slate
    evidence_blocks = [prompts.format_evidence_for_prompt(evidence_list)] if evidence_list else []
    candidates = filter_irrelevant_candidates(state.get("recent_paper_candidates") or [], evidence_list)
    seen_evidence_keys = set()
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        for k in ("arxiv_id", "paper_id", "doi", "url", "pdf_url"):
            val = str(ev.get(k) or "").strip().casefold()
            if val:
                seen_evidence_keys.add(val)
        title_val = re.sub(r"\s+", " ", str(ev.get("doc_name") or ev.get("title") or "")).strip().casefold()
        if title_val:
            seen_evidence_keys.add(title_val)
    extra_candidates = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        item_keys = [
            str(item.get(k) or "").strip().casefold()
            for k in ("arxiv_id", "paper_id", "doi", "url", "pdf_url")
            if item.get(k)
        ]
        item_title = re.sub(r"\s+", " ", str(item.get("title") or item.get("doc_name") or "")).strip().casefold()
        if item_title:
            item_keys.append(item_title)
        if not any(k in seen_evidence_keys for k in item_keys):
            extra_candidates.append(item)
    if extra_candidates:
        evidence_blocks.append("未入主证据列表的补充候选（仅作线索参考）：\n" + "\n".join(
            f"- {item.get('title')} | source={item.get('source')} | level={item.get('evidence_level', 'metadata/abstract unknown')} | url={item.get('url') or item.get('pdf_url') or ''} | {(item.get('summary') or '')[:220]}"
            for item in extra_candidates
        ))
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "本轮没有可用的独立来源证据。"

    # This is an actual review gate: findings must be grounded in this request,
    # draft and the available evidence. Generic checklist advice is not actionable.
    stage = "supplemented" if int(state.get("evidence_append_count") or 0) > 0 else "initial"
    requires_evidence = (state.get("retrieval_plan") or {}).get("requires_evidence", True)
    task_context = {
        "shared_task": task_payload(state), "active_doc": state.get("active_doc"),
        "required_tasks": [{"task_id": idx, "task": item} for idx, item in enumerate(decision.get("task_segments") or [], 1)],
        "history_context": state.get("history_summary") or "",
        "retrieval_coverage": state.get("retrieval_coverage") or {},
        "review_retrieval_log": state.get("review_retrieval_log") or [],
    }
    delivery_checks = {**(state.get("delivery_checks") or {}),
                       "citation_integrity": citation_integrity(final_ans, evidence_list)}
    task_context["program_checks"] = delivery_checks
    prompt = (
        prompts.TASK_COVERAGE_GUIDANCE + "\n" +
        "你是交付前审核者。判断回答是否切合用户原话、解释充分、理论与事实有依据，并保留有用的深入研讨；所有说明字段（reason、suggestion、note、summary、requirement、strategy）统一使用中文。"
        "请围绕【主体是什么 -> 任务是什么 -> 要做什么 -> 检索到什么 -> 怎么利用】及学术研究全链条（研究目的与痛点、理论依据、怎么研究与研究方法、怎么实验与怎么分析、什么结论与边界、有什么扩展点），并结合【要验证什么、实证分析、得到什么结论】评估回答质量。"
        "任务解释和历史审核仅作参考，不增加额外验收条件；学科基础理论推导、数学公式展开与合理学术扩展均属高价值内容，不需要外部论文逐句背书，绝不判为缺陷。"
        "研究方向不必构成直接继承、基线胜出或单线演进；检索资料为论文摘要（abstract）是正常的学术检索形态，绝不因为“仅见摘要未见全文”而判定证据不足或发起补查。"
        + prompts.EVIDENCE_GUIDANCE +
        "只报告具体缺陷；每项 basis 引用 request、visual_context、answer 或 evidence 的原句，资料附 evidence_id。"
        "事实与具体论文结论冲突需资料依据；区分具体文献事实、学科基础理论推导和未来研究建议。"
        "程序检查只核对编号与原句存在，不能证明语义正确，请判断其实际含义。"
        "按原话要求的深度审核；只要核心任务已回答，即应判定 result='met'、verdict='pass' 且不发起补查（evidence_requests=[]）。可选深化记 advisory。"
        "task_checks 按返回顺序从1编号；completeness 用 check_ids 关联未完成项，已 met 的任务不能又作为漏答触发补查。"
        "evidence_requests 用 finding_ids 关联 findings 的顺序编号，purpose 为 missing_task 或 verify_claim。"
        "missing_task 需引用需求原句并关联 missing 检查；verify_claim 仅用于回答中捏造了具体论文数值或违背已知事实的主张（切勿对常规理论推导或摘要范围发起 verify_claim）。"
        "只对真正遗漏的用户必答子任务给可执行查询、理由和有别于既有查询的策略；可选扩展不请求自动检索。"
        "已有内容覆盖则不补查；缺少身份、全文或继承关系本身不强制补查；无可行新策略时说明限制。"
        "明确无关资料可列入 irrelevant_evidence，给具体中文理由及需求、目标资料两侧的 basis 原句（资料可引标题）；保留相关背景、理论和局部支持，相关性不确定不排除。"
        "按 required_tasks 的 task_id 标注 task_checks.task_ids，一条检查可覆盖多个任务；缺少检查不等于回答缺内容。"
        "交付动作由程序负责，本轮只报告审核判断与补证请求。"
        "返回 JSON：{\"verdict\":\"pass|inconclusive\","
        "\"evidence_status\":\"supported|limited|unsupported|not_required\","
        "\"findings\":[{\"kind\":\"severe_topic_deviation|factual|evidence|completeness|advisory\","
        "\"severity\":\"blocking|material|minor\",\"check_ids\":[1],\"location\":\"具体位置\",\"reason\":\"具体依据\",\"suggestion\":\"说明\","
        "\"basis\":[{\"source\":\"request|visual_context|answer|evidence\",\"evidence_id\":1,\"quote\":\"依据原句\"}]}],"
        "\"task_checks\":[{\"task_ids\":[1],\"requirement\":\"用户实际要求\",\"result\":\"met|missing|unclear\",\"supported_facts\":[],\"note\":\"说明\"}],"
        "\"irrelevant_evidence\":[{\"evidence_id\":1,\"reason\":\"具体的主题不匹配依据\",\"basis\":[{\"source\":\"request|visual_context|evidence\",\"quote\":\"依据原句\",\"evidence_id\":1}]}],\"evidence_requests\":[{\"query\":\"仅针对该缺口的可执行中文查询\",\"english_query\":\"2-5个词的精炼英文学术关键词\",\"purpose\":\"missing_task|verify_claim\",\"finding_ids\":[1],\"reason\":\"具体信息缺口\",\"requirement\":\"用户实际需要的内容\",\"strategy\":\"与既有查询不同的策略\"}],\"summary\":\"中文简述\"}\n\n"
        f"审核阶段：{stage}\n本轮是否需要专业资料依据：{requires_evidence}\n"
        f"任务与承接上下文（辅助理解，不是独立证据或额外验收要求）：\n{json.dumps(task_context, ensure_ascii=False)}\n\n"
        f"待审核回答（完整内容）：\n{final_ans}\n\n可用证据：\n{evidence_text}"
    )

    try:
        inquiry_raw = safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            broadcast=False,
            max_tokens=4096,
            timeout=get_llm_timeout(),
        )

        raw = inquiry_raw.strip()
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            review = json.loads(match.group(0)) if match else None
        except Exception:
            review = None
        if not isinstance(review, dict):
            review = {"verdict": "inconclusive", "needs_revision": False,
                      "findings": [], "dimensions": {},
                      "summary": "审核输出无法解析；本轮未能确认审核结论。"}
        evidence_list = apply_evidence_relevance(evidence_list, review.get("irrelevant_evidence"),
            query, task_payload(state).get("visual_context", ""))
        candidates = filter_irrelevant_candidates(candidates, evidence_list)
        delivery_checks["citation_integrity"] = citation_integrity(final_ans, evidence_list)
        findings = review.get("findings") if isinstance(review.get("findings"), list) else []
        evidence_requests = [item for item in review.get("evidence_requests") or []
                             if isinstance(item, dict) and isinstance(item.get("query"), str) and item["query"].strip()]
        evidence_requests = [{**item, **{key: item.get(key) if isinstance(item.get(key), str) else ""
                                        for key in ("requirement", "strategy", "reason", "english_query")}}
                             for item in evidence_requests]
        evidence_status = str(review.get("evidence_status") or "unclear").casefold()
        if evidence_status not in {"supported", "limited", "unsupported", "not_required"}:
            evidence_status = "unclear"
        if requires_evidence and evidence_status == "not_required":
            evidence_status = "unclear"
        if requires_evidence and evidence_status == "supported" and not (
                any(not e.get("excluded_as_irrelevant") for e in evidence_list) or candidates):
            evidence_status = "unsupported"
        if evidence_status == "unsupported":
            findings.append({
                "severity": "blocking", "location": "关键专业主张的资料依据",
                "reason": "关键专业主张的资料依据尚未核实，不能以模型记忆作为核验。",
                "suggestion": "该主张尚无可定位的资料支持，记录为未核实。",
            })
        findings = _locate_review_basis(findings, query, final_ans, evidence_list, task_payload(state).get("visual_context", ""))
        task_checks = review.get("task_checks") if isinstance(review.get("task_checks"), list) else []
        task_checks = [
            {**check, "check_id": index} for index, check in enumerate(task_checks, 1)
            if isinstance(check, dict) and str(check.get("requirement") or "").strip()
            and check.get("result") in {"met", "missing", "unclear"}
        ]
        task_checks = [{**check, "requirement": str(check["requirement"]), "supported_facts": [fact.strip() for fact in (
            [check["supported_facts"]] if isinstance(check.get("supported_facts"), str)
            else check.get("supported_facts") if isinstance(check.get("supported_facts"), list) else [])
            if isinstance(fact, str) and fact.strip()]} for check in task_checks]
        required_tasks = task_context["required_tasks"]
        checked_ids = set()
        for check in task_checks:
            ids = check.get("task_ids") or []
            ids = ids if isinstance(ids, list) else [ids]
            if type(check.get("task_id")) is int:
                ids = [*ids, check["task_id"]]
            # Support previous reviewer outputs without inventing semantic matches.
            for task in required_tasks:
                detail = task["task"]
                label = (detail.get("task") or detail.get("action") or "") if isinstance(detail, dict) else str(detail)
                if label and check["requirement"].strip() == label.strip():
                    ids = [*ids, task["task_id"]]
            check["task_ids"] = sorted({idx for idx in ids if type(idx) is int and 1 <= idx <= len(required_tasks)})
            checked_ids.update(check["task_ids"])
        unchecked_ids = sorted(set(range(1, len(required_tasks) + 1)) - checked_ids)
        delivery_checks["review_task_coverage"] = {"unchecked_task_ids": unchecked_ids,
            "note": "未记录检查不等于回答缺失；保留正文，不自动补查。"}
        incomplete_checks = [
            item for item in task_checks
            if isinstance(item, dict) and str(item.get("result", "")).casefold() == "missing"
        ]
        unclear_checks = [
            item for item in task_checks
            if isinstance(item, dict) and str(item.get("result", "")).casefold() == "unclear"
        ]
        findings, evidence_requests, deferred_requests = _qualify_review_retrieval(
            findings, task_checks, evidence_requests, len(required_tasks))
        delivery_checks["review_retrieval_eligibility"] = {
            "actionable_count": len(evidence_requests), "record_only_count": len(deferred_requests),
            "checked": "linked_review_claim_or_missing_task_only", "semantic_correctness_verified": False}
        needs_revision = False  # Existing prose is immutable; new evidence is appended separately.
        verdict = str(review.get("verdict", "inconclusive")).casefold()
        if (verdict != "pass" or not task_checks or unchecked_ids or incomplete_checks or unclear_checks
              or evidence_requests or (requires_evidence and evidence_status in {"unclear", "unsupported"})
              or any(f.get("severity") in ("blocking", "material") and f.get("kind") != "advisory" for f in findings)):
            verdict = "inconclusive"
        review["verdict"] = verdict
        review["needs_revision"] = needs_revision
        review["findings"] = findings
        review["task_checks"] = task_checks
        review["evidence_requests"] = evidence_requests
        review["deferred_evidence_requests"] = deferred_requests
        review["evidence_status"] = evidence_status
        clean_inquiry = json.dumps(review, ensure_ascii=False, indent=2) if match else raw
        search_stalled = _review_search_stalled({**state, "review_verdict": review}, task_checks)
        retrieval_can_continue = (bool(evidence_requests) and not search_stalled and
            int(state.get("review_retrieval_rounds") or 0) < _evidence_retrieval_budget(state, "REVIEW_EVIDENCE_GAP_ROUNDS"))
        exhausted_with_findings = bool(evidence_requests) and not retrieval_can_continue
        details = [("依据可定位：" if f["basis_status"] == "located" else "待核实：") + str(f["reason"])
                   for f in findings if f.get("reason")]
        if exhausted_with_findings:
            details.append("证据缺口仍未核实，保留原回答与现有补充。")
        review_notice = "审核记录：" + "；".join(details[:3]) if details else None
        # Keep the answer body clean. Findings and status are returned separately
        # for the review panel and must not be presented as part of the answer.
        final_answer = final_ans

        dialogue_log.append({
            "sender": "DialecticalInterrogator",
            "receiver": target_agent,
            "round": int(state.get("review_retrieval_rounds") or 0) + int(state.get("revision_count") or 0) + 1,
            "action": "DIALECTICAL_INQUIRY_AND_PRUNING",
            "summary": str(review.get("summary") or "完成本轮回答审核")
        })

        return {
            "dialectical_notes": clean_inquiry,
            "peer_review_notes": clean_inquiry,
            "review_verdict": {"stage": stage, "verdict": review.get("verdict", "inconclusive"),
                               "needs_revision": needs_revision, "findings": findings,
                               "task_checks": task_checks,
                               "evidence_status": evidence_status,
                               "evidence_requests": evidence_requests,
                               "deferred_evidence_requests": deferred_requests,
                               "irrelevant_evidence": review.get("irrelevant_evidence") or [],
                               "summary": review.get("summary", "")},
            "delivery_checks": delivery_checks,
            "review_checks": task_checks,
            "review_scores_round1": {},
            "review_notice": review_notice,
            "review_retrieval_status": "ready" if retrieval_can_continue else "stalled" if evidence_requests else None,
            "final_answer": final_answer,
            "agent_dialogue_history": dialogue_log,
            "status": "review_unresolved" if exhausted_with_findings else "review_inconclusive" if verdict == "inconclusive" else "review_completed",
            "retrieved_evidence": evidence_list,
            "recent_paper_candidates": candidates,
        }
    except Exception as e:
        print(f"⚠️ [DialecticalInquiry Note]: {e}")
        review_notice = f"审核服务本轮未能完成（{type(e).__name__}）；回答尚未通过审核。"
        return {
            "delivery_checks": delivery_checks,
            "dialectical_notes": None,
            "peer_review_notes": None,
            "review_verdict": {"stage": stage, "verdict": "unavailable", "needs_revision": False,
                               "findings": [], "summary": f"审核调用失败：{type(e).__name__}"},
            "review_notice": review_notice,
            "final_answer": final_ans,
            "status": "dialectical_inquiry_fallback",
            "retrieved_evidence": evidence_list,
        }


def evidence_append_node(state: MasterAgentState) -> Dict[str, Any]:
    """Generate only new material; concatenate in code so the existing draft cannot change."""
    original = state.get("final_answer", "")
    evidence = state.get("retrieved_evidence") or []
    pending = state.get("pending_evidence_additions") or []
    base = {"final_answer": original, "retrieved_evidence": evidence,
            "agent_dialogue_history": list(state.get("agent_dialogue_history") or [])}
    blocks = []
    for item in pending:
        idx = item.get("evidence_id")
        if type(idx) is not int or not 1 <= idx <= len(evidence) or evidence[idx - 1].get("excluded_as_irrelevant"):
            continue
        block = prompts.format_evidence_for_prompt([item])
        if block:
            blocks.append(block.replace("[1]", f"[{idx}]", 1))
    if not blocks:
        return {**base, "pending_evidence_additions": [], "status": "evidence_append_skipped"}
    verdict = state.get("review_verdict") or {}
    gaps = {
        "findings": [{key: f.get(key) for key in ("location", "reason", "basis", "basis_status")}
                     for f in _locate_review_basis(verdict.get("findings"), state.get("user_query", ""), original, evidence, task_payload(state).get("visual_context", ""))],
        "evidence_requests": [{key: r.get(key) for key in ("query", "reason", "requirement")}
                              for r in verdict.get("evidence_requests") or [] if isinstance(r, dict)],
    }
    prompt = (
        "根据本次补查新增的资料，为用户已有回答撰写补充内容。只输出要追加的新内容，不输出或复述原回答。"
        "只补充与用户问题有关且原文尚未覆盖的信息；新资料与原文有实质矛盾时，在补充中指明具体事实及来源。"
        "缺少身份、引用关系或全文本身不是反复声明限制的理由。没有有用的新内容时输出空字符串。"
        "专业事实限于新增资料支持的范围，区分摘要报告、推导和假设，严格沿用给定的原始引用编号。"
        "\n" + task_prompt(state) + f"\n\n原回答（只读，防止重复）：\n{original}"
        + "\n\n本次待补证问题（审核线索，不是已确认结论；按资料核实后补充证据或澄清）：\n"
        + json.dumps(gaps, ensure_ascii=False)
        + "\n\n本次补查新增资料：\n" + "\n\n".join(blocks)
    )
    try:
        addition = safe_chat_completion(client=MasterEngineRegistry.get_llm_client(), model=get_default_llm_model(),
            messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=6144, timeout=get_llm_timeout())
        append_checks = {"supplement_completion": completion_checks(addition)["model_completion"]} if completion_checks(addition) else {}
        addition = (addition or "").strip()
        if "【搭档提示】" in addition:
            raise ValueError("补充内容生成失败")
        answer, appended = append_answer(original, addition)
        history = base["agent_dialogue_history"]
        history.append({"sender": state.get("target_agent") or "StudyPlannerAgent", "receiver": "DialecticalInterrogator",
            "round": int(state.get("review_retrieval_rounds") or 0), "action": "APPEND_RETRIEVED_EVIDENCE",
            "summary": "将补查得到的新信息追加到回答，原正文完整保留。"})
        return {**base, "final_answer": answer, "pending_evidence_additions": [],
                "evidence_append_count": int(state.get("evidence_append_count") or 0) + int(appended),
                "delivery_checks": {**(state.get("delivery_checks") or {}), **append_checks,
                                    "original_answer_preserved": answer.startswith(original),
                                    "citation_integrity": citation_integrity(answer, evidence)},
                "status": "evidence_append_completed" if appended else "evidence_append_skipped"}
    except Exception as exc:
        return {**base, "status": "evidence_append_failed",
                "review_notice": f"新增资料未能生成补充说明（{type(exc).__name__}），原回答已保留。"}


def reflective_revision_node(state: MasterAgentState) -> Dict[str, Any]:
    """Legacy entry point now appends evidence; it never rewrites the draft."""
    return evidence_append_node(state)


academic_reviewer_node = dialectical_inquiry_node
author_revision_node = reflective_revision_node


def _review_gap_retrieval_single(state: MasterAgentState) -> Dict[str, Any]:
    verdict = state.get("review_verdict") or {}
    requests = verdict.get("evidence_requests") or []
    evidence = [dict(e) for e in state.get("retrieved_evidence") or []]
    log = list(state.get("review_retrieval_log") or [])
    pending = list(state.get("pending_evidence_additions") or [])
    prior_searches = list((state.get("retrieval_coverage") or {}).get("search_history") or []) + log
    round_no = int(state.get("review_retrieval_rounds") or 0) + 1
    def already_searched(request):
        for prior in prior_searches:
            same_query = retrieval_query_key(str(prior.get("query") or "")) == retrieval_query_key(str(request.get("query") or ""))
            same_claim = bool(request.get("requirement") and prior.get("requirement") and
                retrieval_query_key(request["requirement"]) == retrieval_query_key(prior["requirement"]))
            same_strategy = retrieval_query_key(str(prior.get("strategy") or "")) == retrieval_query_key(str(request.get("strategy") or ""))
            if (same_query and (same_strategy or not request.get("strategy"))) or (same_claim and same_strategy):
                return True
        return False
    valid = [r for r in requests if isinstance(r, dict) and str(r.get("query") or "").strip()]
    request = next((r for r in valid if not already_searched(r)), None)
    if not request:
        if valid:
            log.append({"query": valid[0]["query"], "result": "already_searched", "round": round_no})
        return {"review_retrieval_rounds": round_no, "review_retrieval_status": "unresolved",
                "review_retrieval_log": log, "pending_evidence_additions": pending}
    query = str(request["query"]).strip()
    decision = task_decision(state)
    record = {"query": query, "round": round_no, "reason": request.get("reason", ""),
              "requirement": request.get("requirement", ""), "strategy": request.get("strategy", ""),
              "task_checks": verdict.get("task_checks") or []}
    try:
        planner = MasterEngineRegistry.get_retrieval_planner()
        gap_task = {**decision, "intent": "核实当前具体证据缺口", "search_query": query,
            "online_search_query": request.get("english_query") or "", "sub_queries": [], "is_cross_disciplinary": False,
            "task_segments": [{"action": "核实审核指出的证据缺口",
            "object": request.get("evidence_type") or query, "retrieval_query": query}]}
        plan = planner.plan(user_query=query, intent=gap_task,
            state_context={
                "active_subject": state.get("active_subject"), "active_doc": state.get("active_doc"),
                "local_material_available": bool(state.get("shared_resources") or state.get("active_doc")
                    or any((BASE_DIR / "uploads").glob("*.pdf")) or (BASE_DIR / "knowledge_db" / "chroma.sqlite3").exists()),
                "shared_pool": state.get("shared_pool") or [], "shared_resources": state.get("shared_resources") or [],
                "recent_paper_candidates": state.get("recent_paper_candidates") or [],
                "shared_task": task_payload(state), "evidence_gap": dict(request),
                "search_history": prior_searches, "retrieved_evidence": evidence})
        record["plan"] = plan.model_dump()
        record["proposed_queries"] = [item.model_dump() for item in plan.queries]
        # The review owns the concrete lookup target. This existing planning
        # call chooses sources/reuse; it cannot reopen the original survey.
        scoped_queries = [{"query": query, "english_query": request.get("english_query") or ""}]
        executable_plan = plan.model_copy(update={"queries": [RetrievalQuery(**item) for item in scoped_queries]})
        found = _selected_shared_evidence(state, plan.selected_shared_item_ids)
        record["selected_shared_item_ids"] = list(plan.selected_shared_item_ids)
        has_search = plan.is_resolved and (plan.local_search or plan.online_search)
        if not has_search and not found:
            record["result"] = "planner_selected_no_search_or_unresolved"
            log.append(record)
            return {"review_retrieval_rounds": round_no, "review_retrieval_status": "unresolved",
                    "review_retrieval_log": log, "pending_evidence_additions": pending}
        valid_proposal = not plan.queries or bool(select_retrieval_probes(query, gap_task, plan))
        probes = select_retrieval_probes(query, gap_task, executable_plan) if has_search and valid_proposal else []
        if has_search and not probes and not found:
            record["result"] = "invalid_executable_query"
            log.append(record)
            return {"review_retrieval_rounds": round_no, "review_retrieval_status": "unresolved",
                    "review_retrieval_log": log, "pending_evidence_additions": pending}
        def lookup_key(item):
            p = item.get("plan") or item
            sources = set(item.get("sources") or [])
            return (retrieval_query_key(str(item.get("english_query") or item.get("query") or "")),
                    bool(p.get("local_search") or "local" in sources),
                    bool(p.get("online_search") or sources & {"academic", "web", "github", "datasets"}),
                    *(bool(p.get(f)) for f in ("include_github", "include_web", "include_datasets")))
        previous = {lookup_key(lookup) for p in prior_searches for lookup in (p.get("executed_queries") or [p])
                    if p.get("result") not in {"invalid_executable_query", "planner_selected_no_search_or_unresolved"}}
        retriever = MasterEngineRegistry.get_hybrid_retriever() if probes else None
        executed, failed = [], []
        for probe in probes:
            lookup = {**probe, **{f: getattr(plan, f) for f in (
                "local_search", "online_search", "include_github", "include_web", "include_datasets")}}
            key = lookup_key(lookup)
            if key in previous:
                continue
            previous.add(key)
            try:
                found.extend(retriever.hybrid_search_and_deduplicate(
                    query=probe["query"], english_query=probe.get("english_query") or None, top_k=8,
                    include_local=plan.local_search, include_online=plan.online_search,
                    include_github=plan.include_github, include_web=plan.include_web, include_datasets=plan.include_datasets,
                    time_constraint=decision.get("time_constraint"), negative_keywords=decision.get("negative_constraints") or []))
            except Exception as exc:
                failed.append({**lookup, "error": type(exc).__name__})
                continue
            executed.append(lookup)
        record.update(executed_queries=executed, failed_queries=failed)
        if not executed and not found:
            record["result"] = "error" if failed else "already_searched"
            log.append(record)
            return {"review_retrieval_rounds": round_no, "review_retrieval_status": "reassess",
                    "review_retrieval_log": log, "pending_evidence_additions": pending}
        previous_count = len(evidence)
        evidence, delta = merge_evidence(evidence, found)
        pending.extend(delta)
        added = len(evidence) - previous_count
        updated = len(delta) - added
        record.update(result="evidence_added" if added or updated else "no_new_evidence",
                      added_count=added, updated_count=updated)
        log.append(record)
        return {"retrieved_evidence": evidence, "pending_evidence_additions": pending,
                "review_retrieval_rounds": round_no, "review_retrieval_status": "retrieved" if added or updated else "reassess",
                "review_retrieval_log": log,
                "source_diagnostics": {"counts": {**((state.get("source_diagnostics") or {}).get("counts") or {}),
                    **dict(getattr(retriever, "last_source_counts", {}) or {})},
                    "errors": {**((state.get("source_diagnostics") or {}).get("errors") or {}),
                    **dict(getattr(retriever, "last_source_errors", {}) or {})}}}
    except Exception as exc:
        record.update(result="error", error=type(exc).__name__)
        log.append(record)
        return {"review_retrieval_rounds": round_no, "review_retrieval_status": "unresolved",
                "review_retrieval_log": log, "pending_evidence_additions": pending}


def review_gap_retrieval_node(state: MasterAgentState) -> Dict[str, Any]:
    """Retrieve each independent gap, keeping a delta for append-only answer updates."""
    requests = [r for r in (state.get("review_verdict") or {}).get("evidence_requests") or []
                if isinstance(r, dict) and str(r.get("query") or "").strip()]
    if not requests:
        return {"review_retrieval_status": "no_request", "review_retrieval_log": state.get("review_retrieval_log") or []}
    working, combined = dict(state), {}
    retrieved, reassess = False, False
    round_no = int(state.get("review_retrieval_rounds") or 0) + 1
    for request in requests:
        working["review_verdict"] = {**(state.get("review_verdict") or {}), "evidence_requests": [request]}
        working["review_retrieval_rounds"] = round_no - 1
        result = _review_gap_retrieval_single(working)
        retrieved = retrieved or result.get("review_retrieval_status") == "retrieved"
        reassess = reassess or result.get("review_retrieval_status") == "reassess"
        working.update(result)
        combined.update(result)
    combined.update(review_retrieval_rounds=round_no,
                    review_retrieval_status="retrieved" if retrieved else "reassess" if reassess else "unresolved")
    return combined


_CONTROL_EXECUTOR = ContextVar("system_control_executor", default=None)


@contextmanager
def system_control_executor(executor):
    """Bind actual application controls to this request, outside persisted state."""
    token = _CONTROL_EXECUTOR.set(executor)
    try:
        yield
    finally:
        _CONTROL_EXECUTOR.reset(token)


def system_control_node(state: MasterAgentState) -> Dict[str, Any]:
    decision = task_decision(state)
    action = decision.get("control_action")
    result = {"action": action, "status": "not_executed",
              "message": "已收到控制请求。本入口没有可用执行器，未执行系统操作。"}
    executor = _CONTROL_EXECUTOR.get()
    if executor and action:
        try:
            actual = executor(action, decision.get("control_parameters") or {})
            if isinstance(actual, dict) and actual.get("status") and actual.get("message"):
                result = {**actual, "action": action}
        except Exception as exc:
            result.update(status="failed", message=f"系统操作未完成：{exc}")
    elif not action:
        result["message"] = "尚未识别出具体控制动作，本轮未执行系统操作。"
    return {"final_answer": result["message"], "retrieved_evidence": [],
            "control_result": result, "tts_voice": result.get("tts_voice") or state.get("tts_voice"),
            "status": "system_control_" + result["status"]}


def clean_text_for_speech(text: str, max_chars: Optional[int] = None) -> str:
    """智能将 Markdown、数学公式和学术 LaTeX 转换为流畅自然的口语播报文本 (精准支持负号、分式、微积分与希腊字母)"""
    import re
    if not text:
        return ""

    c = text

    # 0. 彻底剔除所有 Emoji 图标、智能体前缀与意图标签，杜绝 Edge-TTS 念出“机器人/靶心”
    c = re.sub(r'[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]', '', c)
    c = re.sub(r'🤖\s*[a-zA-Z0-9_]*', '', c)
    c = re.sub(r'🎯\s*意图[：:]\s*[a-zA-Z0-9_]*', '', c)
    c = re.sub(r'AI\s*回复', '', c)

    # 1. 过滤长代码块与行内代码
    c = re.sub(r'```[\s\S]*?```', '，具体代码实现请参阅屏幕面板，', c)
    c = re.sub(r'`([^`]+)`', r'\1', c)

    # 2. 摄氏度与角度
    c = re.sub(r'\^\{\s*\\circ\s*\}\s*\\?(?:text|mathrm)?\{?C\}?', '摄氏度', c)
    c = re.sub(r'\^\{\s*\\circ\s*\}|\^\\circ', '度', c)

    # 3. LaTeX 宏包裹与字体清理，保留内部文本
    c = re.sub(r'\\(mathbf|mathrm|text|mathbb|mathcal|mathit|bm|boldsymbol|operatorname)\{([^}]+)\}', r'\2', c)
    c = re.sub(r'\\hat\{([^}]+)\}', r'\1帽', c)
    c = re.sub(r'\\overline\{([^}]+)\}', r'\1平均', c)
    c = re.sub(r'\\vec\{([^}]+)\}', r'\1向量', c)
    c = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'\1次根号\2', c)
    c = re.sub(r'\\sqrt\{([^}]+)\}', r'根号\1', c)

    # 4. 分数匹配 \frac{A}{B} -> B分之A
    def _frac_sub(m):
        num = m.group(1).strip()
        den = m.group(2).strip()
        return f"{den}分之{num}"
    c = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', _frac_sub, c)
    c = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', _frac_sub, c)

    # 5. 指数与幂次
    c = re.sub(r'\^\{\s*-\s*1\s*\}', '的倒数', c)
    c = re.sub(r'\^\{\s*-\s*([^{}]+)\}', r'的负\1次方', c)
    c = re.sub(r'\^-\s*([0-9a-zA-Z]+)', r'的负\1次方', c)
    c = re.sub(r'\^\{\s*2\s*\}', '的平方', c)
    c = re.sub(r'\^2\b', '的平方', c)
    c = re.sub(r'\^\{\s*3\s*\}', '的立方', c)
    c = re.sub(r'\^3\b', '的立方', c)
    c = re.sub(r'\^\{\s*([^{}]+)\}', r'的\1次方', c)
    c = re.sub(r'\^([0-9a-zA-Z])', r'的\1次方', c)

    # 6. 学术函数与微积分符号
    funcs = [
        (r'\\operatorname\{Tr\}|\\text\{Tr\}|\\mathrm\{Tr\}|\bTr\b|\btr\b', '迹'),
        (r'\\operatorname\{diag\}|\\mathrm\{diag\}|\bdiag\b', '对角矩阵'),
        (r'\\operatorname\{rank\}|\\mathrm\{rank\}|\brank\b', '秩'),
        (r'\\operatorname\{dim\}|\bdim\b', '维度'),
        (r'\\log_2', '以2为底的对数'),
        (r'\\log_e|\\ln', '自然对数'),
        (r'\\log_\{([^{}]+)\}', r'以\1为底的对数'),
        (r'\\log', '对数'),
        (r'\\exp', '指数'),
        (r'\\arg\s*\\max', '自变量最大化'),
        (r'\\arg\s*\\min', '自变量最小化'),
        (r'\\lim', '极限'),
        (r'\\det', '行列式'),
    ]

    for pat, rep in funcs:
        c = re.sub(pat, f' {rep} ', c)

    # 7. 先替换长符号，再替换短符号 (防止 \\in 破坏 \\infty)
    long_math_symbols = [
        (r'\\subseteq', '包含于'), (r'\\supseteq', '包含'),
        (r'\\subset', '真包含于'), (r'\\supset', '真包含'),
        (r'\\notin', '不属于'), (r'\\infty', '无穷大'), (r'\\in\b|\\in(?=[^a-zA-Z])', '属于'),
        (r'\\forall', '任意'), (r'\\exists', '存在'),
        (r'\\leftrightarrow|\\iff|\\Leftrightarrow', '当且仅当'),
        (r'\\rightarrow|\\to|\\longrightarrow', '推导出'),
        (r'\\leftarrow|\\longleftarrow', '来自'),
        (r'\\cup', '并'), (r'\\cap', '交'),
        (r'\\cdot', '点乘'), (r'\\times', '乘以'), (r'\\div', '除以'),
        (r'\\pm', '正负'), (r'\\mp', '负正'),
        (r'\\approx', '约等于'), (r'\\neq', '不等于'),
        (r'\\leq|\\le', '小于等于'), (r'\\geq|\\ge', '大于等于'),
        (r'\\ll', '远小于'), (r'\\gg', '远大于'),
        (r'\\sim', '服从'), (r'\\propto', '正比于'),
        (r'\\nabla', '梯度'), (r'\\partial', '偏导'),
        (r'\\sum_\{([^{}]+)\}\^\{([^{}]+)\}', r'从\1到\2求和'),
        (r'\\sum_\{([^{}]+)\}', r'对\1求和'),
        (r'\\sum_([0-9a-zA-Z])', r'对\1求和'),
        (r'\\sum', '求和'),
        (r'\\prod_\{([^{}]+)\}\^\{([^{}]+)\}', r'从\1到\2连乘'),
        (r'\\prod_\{([^{}]+)\}', r'对\1连乘'),
        (r'\\prod', '连乘'),
        (r'\\int_\{([^{}]+)\}\^\{([^{}]+)\}', r'在区间\1到\2上的积分'),
        (r'\\int', '积分'),
        (r'\\dag', '共轭转置'), (r'\\ddag', '双剑号'), (r'\\top', '转置'),
        (r'\\ldots|\\cdots|\\dots|\\vdots', '等'),
    ]
    for pat, rep in long_math_symbols:
        c = re.sub(pat, f' {rep} ', c)

    # 8. 常见希腊字母
    greeks = {
        r'\\alpha': '阿尔法', r'\\beta': '贝塔', r'\\gamma': '伽马', r'\\delta': '德尔塔',
        r'\\epsilon': '艾普西隆', r'\\varepsilon': '艾普西隆', r'\\zeta': '泽塔', r'\\eta': '伊塔',
        r'\\theta': '西塔', r'\\vartheta': '西塔', r'\\iota': '约塔', r'\\kappa': '卡帕',
        r'\\lambda': '兰姆达', r'\\mu': '缪', r'\\nu': '纽', r'\\xi': '克西',
        r'\\pi': '派', r'\\rho': '柔', r'\\sigma': '西格玛', r'\\tau': '陶',
        r'\\upsilon': '宇普西隆', r'\\phi': '斐', r'\\varphi': '斐', r'\\chi': '凯',
        r'\\psi': '普赛', r'\\omega': '欧米伽',
        r'\\Gamma': '大伽马', r'\\Delta': '大德尔塔', r'\\Theta': '大西塔', r'\\Lambda': '大兰姆达',
        r'\\Xi': '大克西', r'\\Pi': '大派', r'\\Sigma': '大西格玛', r'\\Upsilon': '大宇普西隆',
        r'\\Phi': '大斐', r'\\Psi': '大普赛', r'\\Omega': '大欧米伽',
    }
    for pat, sp in greeks.items():
        c = re.sub(pat, f' {sp} ', c)

    # 9. 负号与减号的精准口语化 (彻底解决 Edge-TTS 漏读负号/减号问题)
    # 9.1 等号后跟负号: "= -" -> "等于负 "
    c = re.sub(r'=\s*-\s*', ' 等于负 ', c)
    # 9.2 括号或开头单目负号: "(-" -> "(负 "
    c = re.sub(r'(?<=[(\[（【])\s*-\s*', '负 ', c)
    # 9.3 二元减法：两个算式或数字相减 "A - B" -> "A 减 B"
    c = re.sub(r'(?<=[0-9a-zA-Z\u4e00-\u9fa5\)])\s+-\s+(?=[0-9a-zA-Z\u4e00-\u9fa5\(])', ' 减 ', c)
    # 9.4 单独的前缀负号 "-A" -> "负 A"
    c = re.sub(r'(?:^|(?<=[\s,，:：]))-\s*([0-9a-zA-Z\u4e00-\u9fa5\\])', r'负 \1', c)

    # 10. 公式界定符转换
    c = re.sub(r'\\\[(.*?)\\\]', r'，公式为：\1，', c, flags=re.DOTALL)
    c = re.sub(r'\$\$(.*?)\$\$', r'，公式为：\1，', c, flags=re.DOTALL)
    c = re.sub(r'\\\((.*?)\\\)', r' \1 ', c, flags=re.DOTALL)
    c = re.sub(r'\$([^\$]+)\$', r' \1 ', c)

    # 11. 等号自然口语化
    c = re.sub(r'(?<=[0-9a-zA-Z\u4e00-\u9fa5\)])\s*=\s*(?=[0-9a-zA-Z\u4e00-\u9fa5\(])', ' 等于 ', c)

    # 12. 清理残余定界符与宏 (确保绝对 0 处反斜杠)
    c = re.sub(r'\\(quad|qquad|hspace|vspace|left|right|big|Big|bigg|Bigg|label|ref|cite|tag)(\{[^}]*\})?', ' ', c)
    c = re.sub(r'\\[a-zA-Z]+', ' ', c)
    c = re.sub(r'\\', ' ', c)

    # 13. 学术缩写口语化
    abbrs = {
        r'\bLLMs?\b': '大语言模型', r'\bRAG\b': '检索增强', r'\bNLP\b': '自然语言处理',
        r'\bCV\b': '计算机视觉', r'\bSOTA\b': '最前沿基准', r'\bHGNN\b': '超图神经网络',
        r'\bvs\.?\b': '对比', r'\be\.g\.?\b': '例如', r'\bi\.e\.?\b': '也就是',
        r'\bFig\.?\s*(\d+)': r'图\1', r'\bEq\.?\s*(\d+)': r'公式\1', r'\bSec\.?\s*(\d+)': r'第\1节',
    }
    for pat, sp in abbrs.items():
        c = re.sub(pat, sp, c, flags=re.IGNORECASE)

    # 14. 格式与标点平滑
    c = re.sub(r'#{1,6}\s*', '，', c)
    c = re.sub(r'\*\*(.*?)\*\*', r'\1', c)
    c = re.sub(r'\*(.*?)\*', r'\1', c)
    c = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', c)
    c = re.sub(r'https?://\S+', '相关学术链接', c)
    c = re.sub(r'[`_~|>]', ' ', c)
    c = re.sub(r'(\d+)\.\s+', r'第\1，', c)
    c = re.sub(r'\n+', '，', c)
    c = re.sub(r'[，,]{2,}', '，', c)
    c = re.sub(r'[。\.]{2,}', '。', c)
    c = re.sub(r'\s+', ' ', c).strip(' ，、')

    # Only explicitly requested previews may truncate the spoken text.
    if max_chars and len(c) > max_chars:
        snippet = c[:max_chars]
        last_punct = max(snippet.rfind('。'), snippet.rfind('！'), snippet.rfind('？'), snippet.rfind('；'))
        if last_punct > max_chars * 0.8:
            return snippet[:last_punct + 1]
        return snippet + "。"

    return c


def tts_synthesis_node(state: MasterAgentState) -> Dict[str, Any]:
    """节点: TTS 女声音色合成 (生动灵动自然学术语音播报，独立哈希缓存)"""
    import asyncio
    import hashlib
    import shutil
    import tempfile
    import edge_tts

    if (state.get("control_result") or {}).get("status") == "pending":
        return {"tts_audio_file": None}

    answer_text = state.get("final_answer", "").strip()
    if not answer_text:
        return {"tts_audio_file": None}

    # 清洗全文为口语文本；服务按需分段，不截取长文的前缀。
    spoken_text = clean_text_for_speech(answer_text)
    if not spoken_text:
        return {"tts_audio_file": None}

    # 优先使用富有情感起伏、生动鲜活的神经语音音色
    voice_id = state.get("tts_voice") or "zh-CN-XiaoyiNeural"  # 晓伊 · 生动灵动 (摆脱机械感)

    # 基于文本内容与音色生成唯一 MD5 缓存哈希
    cache_key = f"{voice_id}_{spoken_text}"
    text_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
    cache_dir = BASE_DIR / "audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_mp3 = cache_dir / f"tts_{text_hash}.mp3"
    latest_mp3 = BASE_DIR / "tts_response.mp3"

    # 若已命中缓存且文件完整，直接复用
    if out_mp3.exists() and out_mp3.stat().st_size > 1024:
        try:
            shutil.copyfile(out_mp3, latest_mp3)
        except Exception:
            pass
        return {"tts_audio_file": str(out_mp3)}
    temporary_audio = None

    async def _generate_audio():
        nonlocal temporary_audio
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".mp3", delete=False) as target:
            temporary_audio = Path(target.name)
        communicate = edge_tts.Communicate(spoken_text, voice=voice_id, rate="+4%", pitch="+0Hz")
        await communicate.save(str(temporary_audio))
        if not temporary_audio.stat().st_size:
            raise RuntimeError("语音服务返回空音频")
        temporary_audio.replace(out_mp3)

    try:
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(lambda: asyncio.run(_generate_audio())).result()
        else:
            asyncio.run(_generate_audio())

        try:
            shutil.copyfile(out_mp3, latest_mp3)
        except Exception:
            pass

        return {"tts_audio_file": str(out_mp3)}
    except Exception as e:
        print(f"❌ [TTS] 语音合成异常: {e}")
        return {"tts_audio_file": None}
    finally:
        if temporary_audio is not None:
            temporary_audio.unlink(missing_ok=True)


# ==========================================
# 4. 构建 LangGraph 总调度状态图
# ==========================================
from langgraph.checkpoint.memory import MemorySaver

# 全局单例 LangGraph 内存检查点记忆器
GLOBAL_MEMORY_SAVER = MemorySaver()


def reviewer_decision_routing(state: MasterAgentState) -> str:
    verdict = state.get("review_verdict") or {}
    if (verdict.get("evidence_requests") and state.get("review_retrieval_status") == "ready"
            and not _review_search_stalled(state, verdict.get("task_checks") or [])
            and int(state.get("review_retrieval_rounds") or 0) < _evidence_retrieval_budget(state, "REVIEW_EVIDENCE_GAP_ROUNDS")):
        return "review_gap_retrieval"
    return "tts_synthesis"


def review_gap_result_routing(state: MasterAgentState) -> str:
    if state.get("review_retrieval_status") == "retrieved":
        return "evidence_append"
    if state.get("review_retrieval_status") == "reassess":
        return "dialectical_inquiry"
    return "tts_synthesis"


def evidence_append_result_routing(state: MasterAgentState) -> str:
    return "dialectical_inquiry" if state.get("status") == "evidence_append_completed" else "tts_synthesis"


def build_master_workflow(checkpointer: Optional[Any] = GLOBAL_MEMORY_SAVER):
    workflow = StateGraph(MasterAgentState)

    # 注册回答、证据整理、审核、补查与追加内容节点。
    workflow.add_node("intent_router", intent_routing_node)
    workflow.add_node("online_paper_search_pipeline", online_paper_search_node)
    workflow.add_node("document_rag_pipeline", document_rag_node)
    workflow.add_node("meeting_summary_pipeline", meeting_summary_node)
    workflow.add_node("cross_modal_pipeline", cross_modal_node)
    workflow.add_node("general_tutor_pipeline", general_tutor_node)
    workflow.add_node("evidence_synthesis", evidence_synthesis_node)
    workflow.add_node("dialectical_inquiry", dialectical_inquiry_node)
    workflow.add_node("review_gap_retrieval", review_gap_retrieval_node)
    workflow.add_node("evidence_append", evidence_append_node)
    workflow.add_node("system_control_pipeline", system_control_node)
    workflow.add_node("tts_synthesis", tts_synthesis_node)

    # 连边与路由
    workflow.add_edge(START, "intent_router")
    workflow.add_conditional_edges(
        "intent_router",
        route_selector,
        {
            "online_paper_search_pipeline": "online_paper_search_pipeline",
            "document_rag_pipeline": "document_rag_pipeline",
            "meeting_summary_pipeline": "meeting_summary_pipeline",
            "cross_modal_pipeline": "cross_modal_pipeline",
            "general_tutor_pipeline": "general_tutor_pipeline",
            "system_control_pipeline": "system_control_pipeline",
        },
    )

    # 所有学术研讨与证据抽取流程汇流至整理归纳智能体
    workflow.add_edge("online_paper_search_pipeline", "evidence_synthesis")
    workflow.add_edge("document_rag_pipeline", "evidence_synthesis")
    workflow.add_edge("meeting_summary_pipeline", "evidence_synthesis")
    workflow.add_edge("cross_modal_pipeline", "evidence_synthesis")
    workflow.add_edge("general_tutor_pipeline", "evidence_synthesis")

    # 补查后只生成追加内容，原回答由代码原样保留。
    workflow.add_edge("evidence_synthesis", "dialectical_inquiry")
    workflow.add_conditional_edges(
        "dialectical_inquiry", reviewer_decision_routing,
        {
            "review_gap_retrieval": "review_gap_retrieval",
            "tts_synthesis": "tts_synthesis",
        },
    )
    workflow.add_conditional_edges("review_gap_retrieval", review_gap_result_routing, {
        "evidence_append": "evidence_append", "dialectical_inquiry": "dialectical_inquiry", "tts_synthesis": "tts_synthesis",
    })
    workflow.add_conditional_edges("evidence_append", evidence_append_result_routing, {
        "dialectical_inquiry": "dialectical_inquiry", "tts_synthesis": "tts_synthesis",
    })

    # 系统指令直连 TTS
    workflow.add_edge("system_control_pipeline", "tts_synthesis")
    workflow.add_edge("tts_synthesis", END)

    return workflow.compile(checkpointer=checkpointer)


# ==========================================
# 5. 主执行入口与联调测试
# ==========================================
def run_end_to_end_test():
    print("=" * 75)
    print("🚀 【LangGraph 多智能体主中枢】端到端全链路联调启动")
    print("=" * 75 + "\n")

    app = build_master_workflow()

    test_queries = [
        # 测试 1: 课件 PPT 页码溯源 (跨模态)
        "老师刚才在课件里讲的缩放点积注意力公式为什么要除以根号 dk？在第几张幻灯片？",
        # 测试 2: 物理公式分步推导 (导师辅导)
        "一维含时薛定谔方程中，算符 iħ ∂/∂t 对应什么物理量？请简要推导。",
        # 测试 3: 通俗大白话解释 (大模型)
        "用大白话给我解释一下多智能体系统中的 LangGraph 状态机是怎么工作的？",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n" + "━" * 75)
        print(f"💬 【测试场景 {i}】 用户输入: \"{q}\"")
        print("━" * 75)

        initial_state: MasterAgentState = {
            "user_query": q,
            "intent_decision": None,
            "retrieval_plan": None,
            "retrieval_queries": [],
            "retrieval_probes": [],
            "retrieved_evidence": [],
            "final_answer": "",
            "target_agent": "",
            "tts_audio_file": None,
            "status": "init",
        }

        start_t = time.time()
        final_state = app.invoke(
            initial_state,
            config={"recursion_limit": 100, "configurable": {"thread_id": f"e2e_test_{hash(q) & 0xffff}"}},
        )
        elapsed = time.time() - start_t

        print(f"\n🤖 【最终回复】 (耗时: {elapsed:.2f}s | 状态: {final_state['status']}):")
        print(final_state["final_answer"])
        if final_state["tts_audio_file"]:
            print(f"\n🔊 【女声语音已就绪】: {final_state['tts_audio_file']}")

    print("\n" + "=" * 75)
    print("🎉 【LangGraph 多智能体全链路联调圆满成功！】")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph 多智能体主中枢系统")
    parser.add_argument("--query", type=str, default=None, help="单次测试提问")
    args = parser.parse_args()

    if args.query:
        app = build_master_workflow()
        res = app.invoke(
            {"user_query": args.query, "intent_decision": None, "retrieval_plan": None, "retrieval_queries": [], "retrieval_probes": [], "retrieved_evidence": [], "final_answer": "", "target_agent": "", "tts_audio_file": None, "status": "init"},
            config={"recursion_limit": 100, "configurable": {"thread_id": "cli_query"}},
        )
        print("\n" + res["final_answer"])
    else:
        run_end_to_end_test()
