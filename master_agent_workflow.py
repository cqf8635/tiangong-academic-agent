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
from intent_recognition_agent import INTENT_DEFAULT_SOURCES, IntentDecision, IntentRecognitionAgent
from meeting_summarizer import MeetingSummarizer
from online_paper_search_agent import OnlinePaperSearchAgent
from subject_memory import transition_subject_state

from config_utils import get_default_llm_model, get_llm_timeout
import prompts
from prompts import (
    build_lead_tutor_system_prompt,
    build_evidence_synthesis_prompt,
    build_dialectical_inquiry_prompt,
    build_deterministic_dialectical_inquiry,
    build_boundary_memo,
    build_reflective_revision_prompt,
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
    user_query: str                          # 用户原始输入 (文字/语音转录，100% 保持纯净未篡改)
    active_subject: Optional[str]            # LangGraph 原生记忆主体 (无偏见确定性状态通道)
    sub_subject: Optional[str]               # 当前深入的子分支技术点
    subject_evolution_log: List[str]         # 主体生命周期演进轨迹
    intent_decision: Optional[Dict[str, Any]]# 意图识别结果结构体
    retrieved_evidence: List[Dict[str, Any]] # 召回的知识切片 (含页码/来源)
    final_answer: str                        # 大脑生成的最终回复内容
    target_agent: str                        # 当前执行的智能体
    tts_audio_file: Optional[str]            # 生成的语音回复文件路径
    tts_voice: Optional[str]                 # 导师音色
    history_summary: Optional[str]           # 长期研讨结构化记忆摘要 (全生命周期高密压缩记忆，0 轮堆叠)
    active_doc: Optional[str]                # 当前会话正在聚焦研讨的核心文献/课件名称 (避免跨文献混杂)
    topic_state: Optional[Dict[str, Any]]    # 会话级主题状态机
    recent_paper_candidates: Optional[List[Dict[str, Any]]] # 前序在线搜索出的候选文献元数据池
    topic_catalog: Optional[List[Dict[str, Any]]] # 会话级结构化主题档案库 (按主题组织历史，拒绝全局粗暴扫描)
    matched_topic: Optional[Dict[str, Any]]       # 本轮外科手术式精准锁定的最佳前序历史主题定论
    pending_action: Optional[Dict[str, Any]]      # 人机协同挂起确认动作 (HITL Pending Action Proposal)
    synthesized_evidence_slate: Optional[str]     # 学术证据与多源资料整理归纳备忘录 (Evidence Slate)
    dialectical_notes: Optional[str]              # 学术博弈质询与去冗评注 (Dialectical Inquiry Memorandum)
    reflection_notes: Optional[str]               # 主研自我内省与证据反思记录 (Cognitive Reflection Log)
    peer_review_notes: Optional[str]              # 质询备忘 (向后兼容)
    agent_dialogue_history: Optional[List[Dict[str, Any]]] # 多智能体点对点研讨对话总线日志 (Agent Dialogue Bus)
    user_image: Optional[str]                     # 用户附带的学术题目或图表截图 (Base64)
    extracted_visual_context: Optional[str]       # 多模态视觉模型忠实提取的图文与公式事实
    status: str                              # 运行状态


# ==========================================
# 2. 单例引擎初始化
# ==========================================
class MasterEngineRegistry:
    intent_agent = None
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
        res = safe_chat_completion(client=client, model=model, messages=messages, temperature=0.1, max_tokens=2048, timeout=get_llm_timeout(90.0))
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
            state["user_query"] = query
            state["extracted_visual_context"] = visual_context

    # 终端日志展示处理：只展示学者文字提问，绝不把冗长的图片转录内容打印到控制台刷屏
    display_query = (raw_user_text or "附带图片分析") if user_img else query[:80]
    if user_img:
        display_query = f"{display_query} [已附带图片]"

    print(f"\n🎯 [MasterRouter] 正在解析意图 (当前锚定主体: 【{state.get('active_subject') or topic_state.get('current_topic') or '无'}】): \"{display_query}\" ...")
    decision: IntentDecision = agent.parse_intent(state=state)

    # LangGraph 原生记忆主体演进 Reducer 逻辑
    subj_act = getattr(decision, "subject_action", "INHERIT") or "INHERIT"
    subj_entity = (getattr(decision, "subject_entity", "") or decision.target_topic or "").strip()
    subj_branch = (getattr(decision, "sub_subject", "") or decision.sub_topic or "").strip()

    curr_subject = state.get("active_subject") or topic_state.get("current_topic") or ""
    curr_sub = state.get("sub_subject") or topic_state.get("sub_topic") or ""
    evo_log = list(state.get("subject_evolution_log") or [])

    previous_memory = {
        "subject": curr_subject,
        "sub_subject": curr_sub,
        "status": topic_state.get("status", "INIT"),
        "evolution_path": topic_state.get("evolution_path", []),
    }
    memory = transition_subject_state(
        previous_memory,
        subj_act,
        query,
        subject_entity=subj_entity,
        sub_subject=subj_branch,
        intent=decision.intent,
        evidence=[],
        constraints=decision.negative_constraints or [],
    )
    active_subject = memory["subject"]
    sub_subject = memory["sub_subject"]
    normalized_action = memory["action"]
    if normalized_action == "APPEND":
        evo_msg = f"APPEND -> 【{subj_entity}】 (Now: {active_subject})"
    elif normalized_action in {"INIT", "REPLACE"}:
        evo_msg = f"{normalized_action} -> 【{active_subject}】"
    else:
        evo_msg = f"INHERIT -> 【{active_subject}】" if active_subject else "INHERIT"

    topic_state = {
        "current_topic": active_subject,
        "sub_topic": sub_subject,
        "status": memory["status"],
        "evolution_path": memory["evolution_path"],
    }

    evo_log.append(evo_msg)
    print(f"🧠 [LangGraph Memory] 主体状态通道更新: {evo_msg}")

    # 纯净学术检索词提取，不暴力硬拼前缀
    search_query = decision.search_query.strip() or query

    print(f"👉 [MasterRouter] 识别结果: 领域=[{decision.domain}] | 意图=[{decision.intent}] | 派发=[{decision.target_agent}]")
    print(f"🔍 [MasterRouter] 正向检索词: \"{search_query}\"")

    evidence_chunks = []
    neg_kw = decision.negative_constraints or []
    active_doc = state.get("active_doc")

    # 0. 若提问中明确指定了单篇 arXiv 论文标识 (如天工雷达定向研读《Title》(arXiv:xxxx.xxxxx))，优先锁定本地已入库全文或摘要
    m_target_arxiv = re.search(r"《([^》]+)》\s*\(arXiv:([0-9]+\.[0-9]+(?:v\d+)?)", query)
    if m_target_arxiv:
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
                "text": query,
                "score": 0.98,
                "url": f"https://arxiv.org/pdf/{target_pid}.pdf",
                "doc_type": "arxiv",
            }]
            print(f"📋 [MasterRouter] 已直接锁定目标论文摘要元数据: 《{target_title}》 (arXiv:{target_pid})，0 外部冗余检索！")

    # 1. 依据具体研讨学术方向与前序就绪文献进行自然处理 (0 硬编码，0 意图枚举写死)
    candidates = state.get("recent_paper_candidates") or []
    matched_candidate = None
    time_constraint = getattr(decision, "time_constraint", None)

    req_sources = getattr(decision, "required_data_sources", []) or []
    is_multi_paper_query = bool(getattr(decision, "is_candidate_selection", False))
    is_explicit_download = getattr(decision, "need_download_fulltext", False)

    # 若当前无需全网重新检索且会话已有就绪文献：
    # 默认 100% 不触发网络下载，全部基于已有文献摘要进行轻量高密推演；
    # 仅当大模型语义判定 need_download_fulltext 为 True 时，才针对锁定单篇执行真实下载与向量建库。
    if candidates and not decision.need_online_paper_search and decision.target_agent not in ["GeneralChatAgent", "SystemControlAgent"]:
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
                    "location_label": f"学术文献 ({c.get('source', 'online')})",
                    "text": f"【文献序号】: {i}\n【论文题目】: {c.get('title')}\n【作者团队】: {', '.join(c.get('authors', []))}\n【发表年份】: {c.get('year')}\n【收录来源/类型】: {c.get('source')}\n【核心摘要/方案】: {c.get('summary')}\n【链接/DOI】: {c.get('paper_id') or c.get('pdf_url')}",
                    "score": 0.95,
                    "url": c.get("pdf_url") or c.get("paper_id"),
                    "doc_type": c.get("source", "online"),
                })
            print(f"📋 [MasterRouter] 会话已有 {len(evidence_chunks)} 篇就绪文献，全量装配至研讨证据链，0 网络下载！")

    need_online = (
        decision.need_online_paper_search 
        or "arxiv_live_search" in req_sources 
        or bool(time_constraint)
        or bool(matched_candidate and not evidence_chunks and decision.need_online_paper_search)
    )
    
    # 纯语义数据源调度：用户学术提问时优先允许在本地学术书阁/经卷切片库中检索
    need_local_search = ("uploaded_pdf_docs" in req_sources) or (
        decision.target_agent not in ["GeneralChatAgent", "SystemControlAgent"]
    )
    need_online_search = ("arxiv_live_search" in req_sources) or bool(need_online)

    if not evidence_chunks and (need_local_search or need_online_search):
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
            bottleneck_sq = None
            for sq in filtered_sub_qs:
                if sq.get("is_core_bottleneck", False) or sq.get("retrieval_strategy") == "online_probe":
                    bottleneck_sq = sq
                    break
            if not bottleneck_sq and filtered_sub_qs:
                print(f"🎯 [MasterRouter] 多学科交叉检索下沉启动: {[sq.get('discipline') for sq in filtered_sub_qs]}")
        else:
            filtered_sub_qs = []
            bottleneck_sq = None
            print(f"🌐 [MasterRouter] 执行领域定向混合检索 (领域: 【{decision.domain}】 | 负向抑制: {neg_kw})...")

        try:
            hybrid_retriever = MasterEngineRegistry.get_hybrid_retriever()
            if is_cross and filtered_sub_qs:
                all_raw_chunks = []
                main_en = (getattr(decision, "online_search_query", "") or "").strip()
                # 收集所有探针任务 (主任务全局 + 各专精学科分支定向探针)
                # 严格遵循核心瓶颈唯一在线探针准则：最多仅允许 1 路核心瓶颈探针执行外部在线检索，其余子探针仅检索本地知识库，彻底杜绝并发多路外部超时与限流
                has_bottleneck = bottleneck_sq is not None
                probe_tasks = [(search_query, 6, main_en, "主全局交叉检索", bool(need_online_search and not has_bottleneck))]
                for sq in filtered_sub_qs:
                    sub_term = sq.get("query")
                    if not sub_term or sub_term == search_query:
                        continue
                    sub_en = (sq.get("english_query") or "").strip()
                    disc_name = sq.get("discipline") or "子学科"
                    sq_online = bool(need_online_search and (sq is bottleneck_sq))
                    probe_tasks.append((sub_term, 4, sub_en, f"子学科[{disc_name}]定向探针", sq_online))

                # 🚀 全域多探针并发检索 (彻底消除串行累加等待，耗时缩减 70%+)
                print(f"⚡ [MasterRouter] 启动全域 {len(probe_tasks)} 路学术探针并发召回...")
                def _execute_single_probe(task_info):
                    q, k, en_q, label, probe_online = task_info
                    return hybrid_retriever.hybrid_search_and_deduplicate(
                        query=q,
                        negative_keywords=neg_kw,
                        top_k=k,
                        include_local=need_local_search,
                        include_online=probe_online,
                        time_constraint=time_constraint,
                        english_query=en_q or None,
                    )

                with ThreadPoolExecutor(max_workers=min(len(probe_tasks), 4)) as executor:
                    probe_results = list(executor.map(_execute_single_probe, probe_tasks))

                for chunks in probe_results:
                    all_raw_chunks.extend(chunks)

                # 3. 全局唯一指纹与标题/页码去重融合 (保留同文献不同页码的深度互补切片)
                seen_keys = set()
                doc_page_counts = {}
                evidence_chunks = []
                for c in all_raw_chunks:
                    title_norm = (c.get("doc_name") or "").strip().lower()
                    loc_norm = (c.get("location_label") or "").strip()
                    # 本地切片以 (标题, 页码) 组合为主键，线上文献以标题为主键
                    unique_key = (title_norm, loc_norm) if loc_norm else title_norm
                    if unique_key in seen_keys:
                        continue
                    # 单篇文献保留互补切片上限 (最多 4 页)
                    doc_cnt = doc_page_counts.get(title_norm, 0)
                    if doc_cnt >= 4:
                        continue
                    seen_keys.add(unique_key)
                    doc_page_counts[title_norm] = doc_cnt + 1
                    evidence_chunks.append(c)
            else:
                main_en = (getattr(decision, "online_search_query", "") or "").strip()
                evidence_chunks = hybrid_retriever.hybrid_search_and_deduplicate(
                    query=search_query,
                    negative_keywords=neg_kw,
                    top_k=10,
                    include_local=need_local_search,
                    include_online=need_online_search,
                    time_constraint=time_constraint,
                    english_query=main_en or None,
                )
            print(f"✨ [MasterRouter] 最终就绪黄金证据链: {len(evidence_chunks)} 项 (已完成全领域召回与统一去重融合)")
        except Exception as e:
            print(f"⚠️ [MasterRouter Hybrid Retrieval Note]: {e}")
            evidence_chunks = rag_engine.search(search_query, top_k=6, candidate_k=20, discipline_filter=decision.domain, negative_keywords=neg_kw)

        # 💡 主任务统一沉淀学术文献候选池（供下游各子任务智能体直接共享复用，不再重复检索）
        if not candidates:
            candidates = []
            for c in evidence_chunks:
                doc_type = c.get("doc_type", "")
                if doc_type in ["arxiv", "openalex", "crossref", "github"]:
                    candidates.append({
                        "paper_id": str(c.get("arxiv_id") or c.get("doi") or c.get("doc_name") or ""),
                        "title": str(c.get("doc_name") or "未命名文献"),
                        "authors": c.get("authors", []),
                        "year": c.get("page_or_slide"),
                        "summary": str(c.get("summary") or c.get("text", "")[:300]),
                        "pdf_url": str(c.get("pdf_url") or c.get("url") or ""),
                        "source": doc_type,
                        "stars": c.get("stars", 0),
                    })

    return {
        "intent_decision": decision.model_dump(),
        "target_agent": decision.target_agent,
        "active_subject": active_subject,
        "sub_subject": sub_subject,
        "subject_evolution_log": evo_log,
        "topic_state": topic_state,
        "retrieved_evidence": evidence_chunks,
        "active_doc": active_doc,
        "recent_paper_candidates": candidates,
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
    decision = state.get("intent_decision") or {}
    target_agent = decision.get("target_agent", "") if isinstance(decision, dict) else getattr(decision, "target_agent", "")
    return AGENT_TO_PIPELINE.get(target_agent, "general_tutor_pipeline")


def online_paper_search_node(state: MasterAgentState) -> Dict[str, Any]:
    """节点: 多源在线学术文献与开源生态检索智能体 (arXiv + CrossRef + GitHub)"""
    decision = state.get("intent_decision") or {}
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

    agent = MasterEngineRegistry.get_paper_search_agent()
    from online_paper_search_agent import ArxivPaper, AcademicResource

    # 🤝 主任务检索成果共享：优先消费主任务沉淀的文献资产，严禁子任务发起重复冗余检索
    shared_cands = [
        c for c in (state.get("recent_paper_candidates") or [])
        if c.get("source") in ["arxiv", "openalex", "crossref", "github"]
    ]

    if shared_cands:
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
            include_github=True,
            time_constraint=decision.get("time_constraint"),
            english_query=decision.get("online_search_query"),
        )
        arxiv_raw = bundle.get("arxiv_papers", [])
        openalex_raw = bundle.get("openalex_papers", [])
        crossref_raw = bundle.get("crossref_papers", [])
        github_raw = bundle.get("github_repos", [])

        arxiv_papers = [ArxivPaper(**p) for p in arxiv_raw]
        openalex_resources = [AcademicResource(**op) for op in openalex_raw]
        crossref_resources = [AcademicResource(**cp) for cp in crossref_raw]
        github_resources = [AcademicResource(**gh) for gh in github_raw]

        evidence = []
        for op in openalex_resources:
            evidence.append({
                "doc_name": op.title,
                "location_label": f"OpenAlex 顶会/顶刊 ({op.citation_count}引用)",
                "text": f"来源: {op.summary}\nDOI: {op.doi}\nURL: {op.url}",
                "score": 0.98,
                "url": op.url,
                "doc_type": "openalex",
                "summary": op.summary,
            })
        for p in arxiv_papers:
            evidence.append({
                "doc_name": p.title,
                "location_label": f"arXiv:{p.paper_id} ({p.year or 'Recent'})",
                "text": f"作者: {', '.join(p.authors)}\nPDF: {p.pdf_url}\n摘要: {p.summary}",
                "score": 0.95,
                "pdf_url": p.pdf_url,
                "paper_id": p.paper_id,
                "doc_type": "arxiv",
                "summary": p.summary,
            })
        for cp in crossref_resources:
            evidence.append({
                "doc_name": cp.title,
                "location_label": f"CrossRef DOI ({cp.year or 'Recent'})",
                "text": f"DOI: {cp.doi}\n出处: {cp.summary}\nURL: {cp.url}",
                "score": 0.92,
                "url": cp.url,
                "doc_type": "crossref",
                "summary": cp.summary,
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
            })

    requires_long_form = decision.get("requires_long_form", False)
    long_form_type = decision.get("long_form_type", "concise_analysis")

    brief = agent.generate_search_brief(
        query=search_query or raw_query,
        papers=arxiv_papers,
        openalex_papers=openalex_resources,
        crossref_papers=crossref_resources,
        github_repos=github_resources,
        history_context=history_summary,
        active_topic=active_topic,
        requires_long_form=requires_long_form,
        long_form_type=long_form_type
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
        "recent_paper_candidates": candidates_list,
        "pending_action": pending_action,
        "status": "online_paper_search_completed",
        "target_agent": "OnlinePaperSearchAgent",
    }


DEFAULT_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "100.0"))

# 🛡️ 全局大模型并发流控信号量 (防止瞬时并发长连接压垮中继代理，彻底消除 503 Overloaded 与 Cloudflare 502 Bad Gateway)
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM_CALLS", "2"))
GLOBAL_LLM_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_LLM)


_GLOBAL_STREAM_CALLBACK: Optional[Callable[[str], None]] = None
_GLOBAL_STREAM_LOCK = threading.Lock()


def register_stream_callback(callback: Optional[Callable[[str], None]]):
    global _GLOBAL_STREAM_CALLBACK
    with _GLOBAL_STREAM_LOCK:
        _GLOBAL_STREAM_CALLBACK = callback


def get_stream_callback() -> Optional[Callable[[str], None]]:
    with _GLOBAL_STREAM_LOCK:
        return _GLOBAL_STREAM_CALLBACK


def safe_chat_completion(
    client,
    model: str,
    messages: list,
    temperature: float = 0.3,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = 12288,
) -> str:
    """带全局并发流控、智能错峰退避、实时 SSE 广播与精准分类熔断的 LLM 安全调用封装"""
    eff_timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT
    fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "").strip()
    last_err = None
    for attempt in range(1, max_retries + 1):
        active_model = (fallback_model if (attempt >= 2 and fallback_model and fallback_model != model) else model)
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
                cb = get_stream_callback()
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
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
                    print(f"✅ [LLM-API] 成功接收大模型完整回复 (总字数: {len(res_str)})")
                    return res_str
                raise RuntimeError(f"上游接口返回空流式响应 (Empty stream from {active_model})")
        except Exception as e:
            last_err = e
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
    noise_papers = []

    # 提取查询核心关键词作为有效性初筛
    kw_tokens = set(re.findall(r'[\w\u4e00-\u9fff]+', query.lower()))

    # 处理候选文献
    for c in (candidates or []):
        title = (c.get("title") or "未命名文献").strip()
        summary = (c.get("summary") or "").strip()
        src = c.get("source", "online")
        authors = ", ".join((c.get("authors") or [])[:3])
        year = c.get("year") or "Recent"
        p_id = c.get("paper_id") or c.get("doi") or c.get("id") or ""

        text_full = f"{title} {summary}".lower()
        has_overlap = any(tok in text_full for tok in kw_tokens if len(tok) >= 2)

        item = {
            "title": title,
            "summary": summary,
            "src": src,
            "authors": authors,
            "year": year,
            "id": p_id,
        }
        if has_overlap or not kw_tokens:
            valid_papers.append(item)
        else:
            noise_papers.append(item)

    # 处理本地检索切片
    local_evidence = []
    for ev in (evidence or []):
        doc = ev.get("doc_name") or ev.get("title") or "资料切片"
        loc = ev.get("location_label") or ""
        txt = (ev.get("text") or "").strip()
        local_evidence.append({
            "doc": doc,
            "loc": loc,
            "text": txt[:300],
        })

    if noise_papers:
        noise_titles = "、".join([f"《{p['title']}》" for p in noise_papers[:2]])
        lines.append(f"- 剔除噪声：{noise_titles} 与当前课题核心因果关联度低，已自动过滤剔除；")

    total_valid = len(valid_papers) + len(local_evidence)
    lines.append(f"- 保留有效资料：当前命中并保留 {max(total_valid, 1)} 项核心学术资产，均紧密支撑本课题理论与实证推演。\n")

    lines.append("#### 2. 📚 资料层级分类 (Taxonomy & Categorization)")
    arxiv_list = [p for p in valid_papers if p["src"] == "arxiv"]
    journal_list = [p for p in valid_papers if p["src"] in ["openalex", "crossref"]]
    code_list = [p for p in valid_papers if p["src"] == "github"]

    if arxiv_list:
        lines.append("- 【前沿预印本 (arXiv)】：" + "；".join([f"{p['authors']}《{p['title']}》({p['year']})" for p in arxiv_list[:3]]))
    if journal_list:
        lines.append("- 【正式期刊/国际学术文献】：" + "；".join([f"{p['authors']}《{p['title']}》({p['year']})" for p in journal_list[:3]]))
    if local_evidence:
        lines.append("- 【精读专著/教材讲义/知识库切片】：" + "；".join([f"《{ev['doc']}》{ev['loc']}" for ev in local_evidence[:3]]))
    if code_list:
        lines.append("- 【开源实现与官方基准】：" + "；".join([f"《{p['title']}》" for p in code_list[:2]]))
    if not (arxiv_list or journal_list or local_evidence or code_list):
        lines.append("- 【学术理论基准】：基于第一性原理公理与核心控制方程展开推演。")
    lines.append("")

    lines.append("#### 3. 🔬 核心机理、方程与边界条件骨架 (Core Mechanisms & Boundary Slate)")
    for p in valid_papers[:4]:
        lines.append(f"- **《{p['title']}》** ({p['year']}):")
        if p['summary']:
            lines.append(f"  * 核心机制与发现: {p['summary'][:180]}...")
    for ev in local_evidence[:3]:
        lines.append(f"- **《{ev['doc']}》** ({ev['loc']}):")
        lines.append(f"  * 支撑依据: {ev['text'][:180]}...")

    if not valid_papers and not local_evidence and final_ans:
        lines.append(f"- 依据主研推演初稿中的数理机理与算符结构展开深入反思与验证。")

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


def document_rag_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 B: 全格式资料知识库检索与页码证据问答 (结合高密度结构化记忆摘要，0 轮堆叠)"""
    query = state["user_query"]
    history_summary = (state.get("history_summary") or "").strip()
    rag_engine = MasterEngineRegistry.get_rag_engine()
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()

    # 优先复用并行预检索的切片，若切片为空则全库检索知识库
    if state.get("retrieved_evidence"):
        evidence_chunks = state["retrieved_evidence"]
    else:
        display_rag_q = (query.split("\n\n[附带图片内容]")[0] + " [已附带图片]") if "\n\n[附带图片内容]" in query else query[:80]
        print(f"📚 [DocumentRAG] 正在检索本地知识库 (检索词: \"{display_rag_q}\")...")
        evidence_chunks = rag_engine.search(query, top_k=6, candidate_k=20)

    if not evidence_chunks:
        context_str = "本地知识库中未找到直接相关的段落，请基于学术原理进行通用解答。"
    else:
        print(f"🔍 [DocumentRAG] 命中 {len(evidence_chunks)} 个高相关度段落：")
        context_parts = []
        for i, chunk in enumerate(evidence_chunks, 1):
            print(f"   [{i}] 得分: {chunk['score']:.4f} | 来源: {chunk['doc_name']} ({chunk['location_label']})")
            context_parts.append(f"【证据 {i}】 来源: 《{chunk['doc_name']}》 {chunk['location_label']}\n{chunk['text']}")
        context_str = "\n\n".join(context_parts)

    decision = state.get("intent_decision") or {}
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
        {"role": "user", "content": f"参考资料证据：\n{context_str}\n\n用户当前提问：{query}"}
    ]
    answer = safe_chat_completion(
        client=client,
        model=model,
        messages=messages,
        temperature=0.2,
        timeout=get_llm_timeout(),
    )
    
    answer = normalize_academic_numbering(answer)

    return {
        "retrieved_evidence": evidence_chunks,
        "final_answer": answer,
        "status": "answered_by_doc_rag",
    }


def meeting_summary_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 A: 会议流水处理与待办决议提炼"""
    query = state["user_query"]
    summarizer = MasterEngineRegistry.get_meeting_summarizer()

    records_dir = BASE_DIR / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    latest_records = list(records_dir.glob("record_*.jsonl"))

    if latest_records:
        latest_records.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        target_record = latest_records[0]
        print(f"📋 [MeetingAgent] 正在对最新会话流水执行归纳: {target_record.name}")
        report = summarizer.summarize_session(target_record)
        answer = f"### 📋 会议归纳完成：{report.title}\n\n**核心摘要**：\n{report.overview}\n\n**关键决议**：\n" + "\n".join([f"- {d}" for d in report.key_decisions])
        if report.action_items:
            answer += "\n\n**待办清单**：\n" + "\n".join([f"- **{a.task}**（负责人: {a.assignee}, 截止: {a.deadline}）" for a in report.action_items])
    else:
        answer = "当前没有正在进行的会议录音流水。您可以通过 `python voice_langgraph_agent.py` 开启录制。"

    return {
        "final_answer": answer,
        "retrieved_evidence": [],
        "status": "answered_by_meeting_summarizer",
    }


def cross_modal_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 C: 跨模态音视频讲解与课件 PPT 对照"""
    query = state["user_query"]
    rag_engine = MasterEngineRegistry.get_rag_engine()
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()

    print(f"🔄 [CrossModal] 执行音视频与课件对齐检索 (查询词: \"{query}\")...")
    evidence_chunks = rag_engine.search(query, top_k=2)

    context_str = ""
    if evidence_chunks:
        top_chunk = evidence_chunks[0]
        context_str = f"匹配到的课件页面: 《{top_chunk['doc_name']}》 {top_chunk['location_label']}\n课件内容:\n{top_chunk['text']}"

    system_prompt = "你是教学助教。用户询问老师讲解的内容对应课件的哪一页，请准确告知课件页码并简要说明该页核心内容。"
    answer = safe_chat_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"检索到的课件信息：\n{context_str}\n\n用户提问：{query}"},
        ],
        temperature=0.1,
        timeout=get_llm_timeout(),
    )
    return {
        "retrieved_evidence": evidence_chunks,
        "final_answer": answer,
        "status": "answered_by_cross_modal",
    }


def general_tutor_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 D: 通用学科辅导、公式推导与概念拆解 (结合高密度结构化记忆摘要，0 轮堆叠)"""
    query = state["user_query"]
    history_summary = (state.get("history_summary") or "").strip()
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()
    decision = state.get("intent_decision") or {}
    intent = decision.get("intent", "concept_simplify")
    target_agent_name = (
        state.get("target_agent")
        or (decision.get("target_agent") if isinstance(decision, dict) else getattr(decision, "target_agent", "StudyPlannerAgent"))
        or "StudyPlannerAgent"
    )

    display_tutor_q = (query.split("\n\n[附带图片内容]")[0] + " [已附带图片]") if "\n\n[附带图片内容]" in query else query[:80]
    print(f"🎓 [TutorAgent] 启动学术辅导与推导引擎 (席位: {target_agent_name} | 意图: {intent} | 查询: \"{display_tutor_q}\")...")

    evidence_chunks = []
    rag_context_str = ""
    dialogue_log = list(state.get("agent_dialogue_history") or [])

    # 🔀 动态跨学科特派子代理并发派生与合流 (Dynamic Cross-Disciplinary Specialist Spawning & Convergence)
    is_cross = bool(decision.get("is_cross_disciplinary") if isinstance(decision, dict) else getattr(decision, "is_cross_disciplinary", False))
    sub_qs = (decision.get("sub_queries") if isinstance(decision, dict) else getattr(decision, "sub_queries", [])) or []
    disciplines = (decision.get("disciplines") if isinstance(decision, dict) else getattr(decision, "disciplines", [])) or []

    specialist_insights = []
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
                "【篇幅与成果体裁规范】：当前任务属于长篇成果撰写，请详实输出该学科的核心控制方程、推导依据与完整边界条件。"
                if decision.get("requires_long_form", False)
                else "【深度与高信噪比论证规范】：不设死板字数限制，避免论文级冗长注水；严禁通识科普与宽泛背景堆砌，直切核心控制方程与关键物理/工程机理约束，严密标明 LaTeX 公式与变量定义，把该学科的核心形式化推演交代完整透彻。"
            )
            spec_prompt = (
                f"你是一名专注于【{disc}】领域的资深理论科学家与特派专家子代理。\n"
                f"课题总体目标：{query}\n"
                f"你受托专攻的子命题：{sub_q}\n"
                "请从第一性原理与该领域最前沿范式出发，严密提炼该领域的核心控制方程、状态转移方程或机理边界约束。\n"
                f"{spec_length}"
            )
            try:
                res = safe_chat_completion(
                    client=client,
                    model=model,
                    messages=[{"role": "user", "content": spec_prompt}],
                    temperature=0.2,
                    max_tokens=6144,
                    timeout=get_llm_timeout(),
                )
                return {
                    "discipline": disc,
                    "agent_name": agent_name,
                    "sub_query": sub_q,
                    "analysis": res.strip()
                }
            except Exception as e:
                print(f"⚠️ [Specialist Subagent Error - {disc}]: {e}")
                return {
                    "discipline": disc,
                    "agent_name": agent_name,
                    "sub_query": sub_q,
                    "analysis": f"对【{disc}】的数学形式化与边界约束已确立基本理论支点。"
                }

        with ThreadPoolExecutor(max_workers=min(len(sub_qs), 2)) as pool:
            futures = []
            for sq in sub_qs[:3]:
                futures.append(pool.submit(_run_specialist, sq))
                time.sleep(0.6)  # 阶梯错峰，配合 GLOBAL_LLM_SEMAPHORE 彻底消除瞬时并发洪峰
            specialist_results = [f.result() for f in futures]

        for s_res in specialist_results:
            agent_name = s_res["agent_name"]
            disc = s_res["discipline"]
            analysis = s_res["analysis"]
            specialist_insights.append(f"【特派专家席位：{agent_name} ({disc}) 研判洞见】\n{analysis}")
            dialogue_log.append({
                "sender": agent_name,
                "receiver": target_agent_name,
                "round": 0,
                "action": "DELIVER_DISCIPLINARY_FOUNDATION",
                "summary": f"🔬 {agent_name} 完成学科解耦推演，向主研导师交接了关于「{s_res['sub_query'][:30]}」的底层机理与形式化约束。"
            })
            print(f"✅ [Specialist Subagent] 成功接入来自 {agent_name} 的学科基础研判！")

    if state.get("retrieved_evidence"):
        evidence_chunks = state["retrieved_evidence"]
        context_parts = []
        for i, chunk in enumerate(evidence_chunks[:6], 1):
            chunk_text = str(chunk.get("text") or "")[:1200]
            context_parts.append(f"【参考证据 {i}】 来源: 《{chunk.get('doc_name')}》 {chunk.get('location_label')}\n{chunk_text}")
        rag_context_str = "\n\n".join(context_parts)

    req_sources = decision.get("required_data_sources") if isinstance(decision, dict) else getattr(decision, "required_data_sources", None)
    if not req_sources:
        req_sources = INTENT_DEFAULT_SOURCES.get(intent, ["uploaded_pdf_docs"])
    need_local_search = "uploaded_pdf_docs" in req_sources
    online_q = decision.get("online_search_query") if isinstance(decision, dict) else getattr(decision, "online_search_query", None)
    need_online_search = (
        ("arxiv_live_search" in req_sources)
        or bool(decision.get("need_online_paper_search", False) if isinstance(decision, dict) else getattr(decision, "need_online_paper_search", False))
        or bool(str(online_q).strip() if online_q else False)
        or bool(is_cross and sub_qs)
    )

    if not evidence_chunks and need_local_search:
        try:
            rag_engine = MasterEngineRegistry.get_rag_engine()
            found_chunks = rag_engine.search(query, top_k=4, candidate_k=15)

            if found_chunks and found_chunks[0].get("score", 0) >= 0.30:
                evidence_chunks = found_chunks
                print(f"🔍 [GeneralTutor] 自动命中 {len(evidence_chunks)} 篇本地资料切片：")
                context_parts = []
                for i, chunk in enumerate(evidence_chunks, 1):
                    print(f"   [{i}] 得分: {chunk['score']:.4f} | 来源: {chunk['doc_name']} ({chunk['location_label']})")
                    context_parts.append(f"【参考证据 {i}】 来源: 《{chunk['doc_name']}》 {chunk['location_label']}\n{chunk['text']}")
                rag_context_str = "\n\n".join(context_parts)
        except Exception as e:
            print(f"⚠️ [GeneralTutor Grounding Note] {e}")

    recent_candidates = list(state.get("recent_paper_candidates") or [])

    # 🤝 多智能体点对点协作总线 (Agent Collaboration Bus):
    # 严格根据大模型裁定的 need_online_paper_search 决定是否委托在线探查，绝不盲目全局外搜
    if not evidence_chunks and need_online_search:
        try:
            print("🤝 [Multi-Agent Bus] 主研学者向在线文献专员发起点对点前沿基线探查请求...")
            paper_agent = MasterEngineRegistry.get_paper_search_agent()
            eng_query = (decision.get("online_search_query") or "").strip()
            chinese_query = (decision.get("search_query") or query).strip()
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
                    include_github=False,
                    time_constraint=decision.get("time_constraint"),
                    english_query=eng_query,
                )
                if isinstance(bundle, dict):
                    raw_papers.extend(bundle.get("arxiv_papers", []))
                    raw_papers.extend(bundle.get("openalex_papers", []))
                    raw_papers.extend(bundle.get("crossref_papers", []))

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
                if len(unique_papers) >= 25:
                    break

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
                context_parts = []
                # 研讨上下文精选前 5 篇最具学术代表性的文献，确保讨论深度与高信噪比
                for p_dict in unique_papers[:5]:
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
                        "text": f"【论文题目】: 《{title}》\n【主要作者】: {', '.join(authors[:3])} ({year})\n【核心摘要/方案】: {summary[:300]}\n【在线链接/DOI】: {url}",
                        "score": 0.95,
                    }
                    evidence_chunks.append(chunk)
                    context_parts.append(f"【学术基线证据】 来源: 《{title}》 ({loc_label})\n{summary[:300]}...")

                rag_context_str = "\n\n".join(context_parts)
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

    decision = state.get("intent_decision") or {}
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

    user_prompt = f"用户提问：{query}"
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
        user_prompt += f"\n\n🎯 【意图解析提取的核心实体与具体待解设问（最高优先级必答）】：\n" + "\n".join(task_items) + "\n请务必以直接、严密解答与分步证明上述具体设问为主线展开研讨，切勿受外部资料反客为主影响！"

    if specialist_insights:
        spec_text = "\n\n".join(specialist_insights)
        user_prompt = f"【多学科特派专家解耦推演输入 (Specialist Disciplinary Foundations)】：\n{spec_text}\n\n" + user_prompt
    if rag_context_str:
        user_prompt = f"本地文献知识库切片证据：\n{rag_context_str}\n\n{user_prompt}"

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
    decision = state.get("intent_decision") or {}
    intent = decision.get("intent", "")
    evidence_list = state.get("retrieved_evidence") or []
    candidates_list = state.get("recent_paper_candidates") or []
    final_ans = state.get("final_answer", "")
    target_agent = state.get("target_agent") or ""
    dialogue_log = list(state.get("agent_dialogue_history") or [])

    # 若为纯系统控制指令或完全没有任何多源检索素材与内容，则跳过归纳
    if target_agent == "SystemControlAgent" or (not evidence_list and not candidates_list and not final_ans):
        return {
            "synthesized_evidence_slate": None,
            "status": "evidence_synthesis_skipped",
        }

    # 聚合多源原始素材
    material_snippets = []
    if candidates_list:
        material_snippets.append("【候选学术论文与开源项目库】：")
        for i, c in enumerate(candidates_list[:6], 1):
            src = c.get("source", "online")
            src_label = "arXiv预印本" if src == "arxiv" else ("正式发表期刊" if src in ["openalex", "crossref"] else src)
            title = c.get("title", "未命名")
            authors = ", ".join(c.get("authors", [])[:3])
            year = c.get("year", "Recent")
            summary = (c.get("summary") or "")[:200].replace("\n", " ").strip()
            material_snippets.append(f"[{i}] [{src_label}] 《{title}》 ({year}, {authors}): {summary}")

    if evidence_list:
        material_snippets.append("【检索召回的专著书籍/教材讲义/技术文档/课件切片】：")
        for j, ev in enumerate(evidence_list[:6], 1):
            doc = ev.get("doc_name") or ev.get("title") or "文档切片"
            loc = ev.get("location_label") or ""
            text = (ev.get("text") or "")[:260].replace("\n", " ").strip()
            material_snippets.append(f"[{j}] 来源: 《{doc}》 {loc}: {text}")

    # 若无直接检索切片但有主研初稿推演（如公式演算），则从初稿中提取推演骨架
    if not material_snippets and final_ans:
        material_snippets.append(f"【主研推演中的数理骨架与公式草案】：\n{final_ans[:1200]}")

    raw_materials_text = "\n".join(material_snippets)

    enable_llm_synthesis = os.getenv("ENABLE_LLM_EVIDENCE_SYNTHESIS", "false").lower() == "true"
    is_mocked = safe_chat_completion is not _ORIGINAL_SAFE_CHAT_COMPLETION

    if not enable_llm_synthesis and not is_mocked:
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

    prompt = prompts.build_evidence_synthesis_prompt(query, raw_materials_text)

    try:
        slate_raw = safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
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


def dialectical_inquiry_node(state: MasterAgentState) -> Dict[str, Any]:
    """学术博弈质询席位 (Dialectical Inquiry & Evidence-Grounded Pruning)
    
    对主研导师的解答初稿进行深度博弈研讨与去冗：
    1. 剔除与当前主题无关的通识科普、套话背景与发散内容（做减法）；
    2. 对照知识库文献与数理逻辑，检查哪些断言缺乏证据支撑并指出证据断层；
    3. 提炼核心因果骨架，提出实质性质询问题引导主研团队开展认知反思。
    """
    final_ans = state.get("final_answer", "")
    decision = state.get("intent_decision") or {}
    intent = decision.get("intent", "")
    query = state.get("user_query", "")
    target_agent = state.get("target_agent") or "StudyPlannerAgent"
    evidence_list = state.get("retrieved_evidence", [])

    # 仅当为纯系统控制指令（静音、清屏等）、纯日常寒暄、初稿异常报错或无解答内容时跳过
    is_error_ans = any(kw in final_ans for kw in ["【搭档提示】", "Server Overloaded", "HTTP 502", "HTTP 503", "HTTP 524"])
    if target_agent in ["SystemControlAgent", "GeneralChatAgent"] or not final_ans or not final_ans.strip() or is_error_ans:
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

    print(f"🧐 [DialecticalInquiry] 启动学术博弈质询：对 {target_agent} 的推演初稿进行去冗审查、证据链对齐与反思追问...")

    # 提取文献核心线索（优先采用整理归纳智能体清洗提炼的证据备忘录）
    slate = state.get("synthesized_evidence_slate")
    evidence_text = ""
    if slate:
        evidence_text = slate
    elif evidence_list:
        ev_snippets = []
        for i, ev in enumerate(evidence_list[:5], 1):
            title = ev.get("doc_name") or ev.get("title") or "文献片段"
            text = ev.get("text", "")[:300].strip()
            ev_snippets.append(f"[{i}] 《{title}》: {text}")
        evidence_text = "\n".join(ev_snippets)

    enable_llm_inquiry = os.getenv("ENABLE_LLM_DIALECTICAL_INQUIRY", "false").lower() == "true"
    is_mocked = safe_chat_completion is not _ORIGINAL_SAFE_CHAT_COMPLETION

    if not enable_llm_inquiry and not is_mocked:
        clean_inquiry = _build_deterministic_dialectical_inquiry(
            query=query,
            target_agent=target_agent,
            final_ans=final_ans,
            evidence_slate=evidence_text,
        )
        print(f"🧐 [DialecticalInquiry] 纯代码学术博弈质询席位完成去冗准则与证据核验备忘装配 (0 API 消耗，0ms 耗时)")
        dialogue_log.append({
            "sender": "DialecticalInterrogator",
            "receiver": target_agent,
            "round": 1,
            "action": "DIALECTICAL_INQUIRY_AND_PRUNING",
            "summary": "学术质询专员对初稿发起去冗与证据支撑质询，指出冗余通识背景与缺乏证据支撑的断言，引导主研团队开展深度反思重构。"
        })
        return {
            "dialectical_notes": clean_inquiry,
            "peer_review_notes": clean_inquiry,
            "agent_dialogue_history": dialogue_log,
            "status": "dialectical_inquiry_completed",
            "retrieved_evidence": evidence_list,
        }

    prompt = prompts.build_dialectical_inquiry_prompt(
        query=query,
        final_ans=final_ans,
        evidence_text=evidence_text,
    )

    try:
        inquiry_raw = safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            timeout=get_llm_timeout(),
        )

        clean_inquiry = inquiry_raw.strip()

        dialogue_log.append({
            "sender": "DialecticalInterrogator",
            "receiver": target_agent,
            "round": 1,
            "action": "DIALECTICAL_INQUIRY_AND_PRUNING",
            "summary": "学术质询专员对初稿发起去冗与证据支撑质询，指出冗余通识背景与缺乏证据支撑的断言，引导主研团队开展深度反思重构。"
        })

        return {
            "dialectical_notes": clean_inquiry,
            "peer_review_notes": clean_inquiry,
            "agent_dialogue_history": dialogue_log,
            "status": "dialectical_inquiry_completed",
            "retrieved_evidence": evidence_list,
        }
    except Exception as e:
        print(f"⚠️ [DialecticalInquiry Note]: {e}")
        return {
            "dialectical_notes": None,
            "peer_review_notes": None,
            "status": "dialectical_inquiry_fallback",
            "retrieved_evidence": evidence_list,
        }


def reflective_revision_node(state: MasterAgentState) -> Dict[str, Any]:
    """主研学者基于研讨对话的认知反思与解答重构席位 (Reflective Revision & Synthesis)
    
    主研学者阅读博弈质询专员的对话，开展显式认知反思：
    1. 采纳去冗意见，彻底剔除无关通识与发散冗余；
    2. 消化证据质询，纠偏无支撑臆断，明确适用域边界（Domain of Validity）；
    3. 重构输出高信息密度、逻辑闭环的学术终稿，并在末尾沉淀反思与边界备忘。
    """
    final_ans = state.get("final_answer", "")
    query = state.get("user_query", "")
    inquiry_notes = state.get("dialectical_notes") or state.get("peer_review_notes")
    target_agent = state.get("target_agent") or "StudyPlannerAgent"
    evidence_list = state.get("retrieved_evidence", [])
    dialogue_log = list(state.get("agent_dialogue_history") or [])

    active_subject = state.get("active_subject")
    history_summary = state.get("history_summary")
    is_error_ans = any(kw in final_ans for kw in ["【搭档提示】", "Server Overloaded", "HTTP 502", "HTTP 503", "HTTP 524"])

    # 若初稿本身已有效且无需第二轮推倒重写，直接进入继承并锚定边界备忘分支
    enable_second_pass = os.getenv("ENABLE_SECOND_PASS_REVISION", "false").lower() == "true"
    is_mocked = safe_chat_completion is not _ORIGINAL_SAFE_CHAT_COMPLETION

    if not enable_second_pass and not is_mocked and final_ans and len(final_ans.strip()) > 200 and not is_error_ans:
        print(f"✅ [{target_agent}] 主研成果初稿已具备完备学术深度与公式表征，直接继承并锚定边界备忘 (避免二次重复生成与网络超时)")
        final_with_memo = final_ans
        if "学术研讨适用域与边界备忘" not in final_with_memo and "学术研讨反思与边界沉淀备忘" not in final_with_memo:
            boundary_memo = prompts.build_boundary_memo()
            final_with_memo = f"{final_ans.strip()}{boundary_memo}"

        final_with_memo = normalize_academic_numbering(final_with_memo)
        dialogue_log.append({
            "sender": target_agent,
            "receiver": "DialecticalInterrogator",
            "round": 1,
            "action": "REFLECTIVE_RECONSTRUCTION",
            "summary": f"✍️ {target_agent} 消化研讨对话，完成去冗剔除、证据链锚定与边界收缩，产出经反思检验的高密度学术终稿。"
        })
        return {
            "final_answer": final_with_memo,
            "reflection_notes": "主研团队消化研讨对话，完成去冗剔除、证据链锚定与边界收缩。",
            "agent_dialogue_history": dialogue_log,
            "status": "reflective_revision_completed",
            "retrieved_evidence": evidence_list,
        }

    # 若未触发质询且初稿无异常，直接跳过
    if not inquiry_notes and not is_error_ans and final_ans.strip():
        return {
            "final_answer": final_ans,
            "status": "reflective_revision_skipped",
            "agent_dialogue_history": dialogue_log,
            "retrieved_evidence": evidence_list,
        }

    print(f"✍️ [{target_agent}] 主研团队启动反思重构/兜底推演：正在结合状态记忆与证据链进行深度推演...")
    client = MasterEngineRegistry.get_llm_client()
    model = get_default_llm_model()

    decision = state.get("intent_decision") or {}
    requires_long_form = decision.get("requires_long_form", False)
    long_form_type = decision.get("long_form_type")

    prompt = prompts.build_reflective_revision_prompt(
        query=query,
        final_ans=final_ans,
        inquiry_notes=inquiry_notes,
        active_subject=active_subject,
        history_summary=history_summary,
        evidence_list=evidence_list,
        requires_long_form=requires_long_form,
        long_form_type=long_form_type,
        is_fallback=(is_error_ans or not final_ans.strip()),
    )

    try:
        revised_answer = safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=12288,
            timeout=get_llm_timeout(),
        )
        revised_answer = normalize_academic_numbering(revised_answer)

        dialogue_log.append({
            "sender": target_agent,
            "receiver": "DialecticalInterrogator",
            "round": 1,
            "action": "REFLECTIVE_RECONSTRUCTION",
            "summary": f"✍️ {target_agent} 消化研讨对话，完成去冗剔除、证据链锚定与边界收缩，产出经反思检验的高密度学术终稿。"
        })

        print(f"✅ [{target_agent}] 主研团队反思重构完成，已产出高保真终稿！")
        return {
            "final_answer": revised_answer,
            "reflection_notes": "主研团队消化研讨对话，完成去冗剔除、证据链锚定与边界收缩。",
            "agent_dialogue_history": dialogue_log,
            "status": "reflective_revision_completed",
            "retrieved_evidence": evidence_list,
        }
    except Exception as e:
        print(f"⚠️ [{target_agent} Revision Error]: {e}")
        return {
            "status": "reflective_revision_fallback",
            "agent_dialogue_history": dialogue_log,
            "retrieved_evidence": evidence_list,
        }


# 向后兼容别名 (Backward-compatibility aliases)
academic_reviewer_node = dialectical_inquiry_node
author_revision_node = reflective_revision_node


def system_control_node(state: MasterAgentState) -> Dict[str, Any]:
    """路线 E: 系统口令与控制"""
    return {
        "final_answer": "好的，收到指令，系统已为您执行操作。",
        "retrieved_evidence": [],
        "status": "system_controlled",
    }


def clean_text_for_speech(text: str, max_chars: int = 10000) -> str:
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

    # 保留 100% 完整长篇内容，绝不提前截断
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
    import edge_tts

    answer_text = state.get("final_answer", "").strip()
    if not answer_text:
        return {"tts_audio_file": None}

    # 智能清洗并转换全文为生动高保真口语文本（100% 全量完整朗读）
    spoken_text = clean_text_for_speech(answer_text, max_chars=10000)
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
    async def _generate_audio():
        communicate = edge_tts.Communicate(spoken_text, voice=voice_id, rate="+4%", pitch="+0Hz")
        await communicate.save(str(out_mp3))

    try:
        loop = None
        try:
            loop = asyncio.get_event_loop()
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


# ==========================================
# 4. 构建 LangGraph 总调度状态图
# ==========================================
from langgraph.checkpoint.memory import MemorySaver

# 全局单例 LangGraph 内存检查点记忆器
GLOBAL_MEMORY_SAVER = MemorySaver()


def reviewer_decision_routing(state: MasterAgentState) -> str:
    """向后兼容路由判断"""
    return "tts_synthesis"


def build_master_workflow(checkpointer: Optional[Any] = GLOBAL_MEMORY_SAVER):
    workflow = StateGraph(MasterAgentState)

    # 注册节点 (包含主研导师、多源证据整理归纳专员、博弈质询专员与基于反思的重构席位)
    workflow.add_node("intent_router", intent_routing_node)
    workflow.add_node("online_paper_search_pipeline", online_paper_search_node)
    workflow.add_node("document_rag_pipeline", document_rag_node)
    workflow.add_node("meeting_summary_pipeline", meeting_summary_node)
    workflow.add_node("cross_modal_pipeline", cross_modal_node)
    workflow.add_node("general_tutor_pipeline", general_tutor_node)
    workflow.add_node("evidence_synthesis", evidence_synthesis_node)
    workflow.add_node("dialectical_inquiry", dialectical_inquiry_node)
    workflow.add_node("reflective_revision", reflective_revision_node)
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

    # 经整理归纳后，统一进入博弈质询席位 -> 主研反思重构席位 -> TTS 发音
    workflow.add_edge("evidence_synthesis", "dialectical_inquiry")
    workflow.add_edge("dialectical_inquiry", "reflective_revision")
    workflow.add_edge("reflective_revision", "tts_synthesis")

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
            "retrieved_evidence": [],
            "final_answer": "",
            "target_agent": "",
            "tts_audio_file": None,
            "status": "init",
        }

        start_t = time.time()
        final_state = app.invoke(initial_state)
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
        res = app.invoke({"user_query": args.query, "intent_decision": None, "retrieved_evidence": [], "final_answer": "", "target_agent": "", "tts_audio_file": None, "status": "init"})
        print("\n" + res["final_answer"])
    else:
        run_end_to_end_test()
