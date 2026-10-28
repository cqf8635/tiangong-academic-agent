"""Dynamic User-Defined Session and Multi-Domain Conversation Store.

Features:
1. No rigid hardcoded channels: Users freely create, rename, and delete sessions.
2. Auto-naming: First user query can dynamically name the conversation.
3. Completely isolated conversation histories per session.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from config_utils import get_default_llm_model, get_llm_timeout
from prompts import build_session_memory_prompt, get_assistant_name

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str  # "user" | "agent"
    content: str
    intent_decision: Optional[Dict[str, Any]] = None
    target_agent: Optional[str] = None
    retrieved_evidence: List[Dict[str, Any]] = []
    mindmap_markdown: Optional[str] = None
    audio_url: Optional[str] = None
    agent_dialogue_history: Optional[List[Dict[str, Any]]] = None
    synthesized_evidence_slate: Optional[str] = None
    dialectical_notes: Optional[str] = None
    image_url: Optional[str] = None  # 用户附带的截图或图片 (URL 或 Base64)
    timestamp: str = Field(default_factory=lambda: time.strftime("%H:%M:%S"))


class ConversationSession(BaseModel):
    session_id: str
    title: str
    domain: str = "custom"
    active_doc: Optional[str] = None      # 当前会话正在聚焦研讨的核心文献/课件名称
    topic_state: Dict[str, Any] = Field(  # 会话级主题生命周期状态机 (INIT -> LOCKED -> EVOLVING -> RESET)
        default_factory=lambda: {
            "current_topic": "",
            "sub_topic": "",
            "status": "INIT",             # INIT | LOCKED | EVOLVING | RESET
            "evolution_path": [],
        }
    )
    history_summary: Optional[str] = ""  # 早期超长历史对话的结构化压缩摘要
    last_summarized_index: int = 0      # 已被压缩合并的历史消息索引游标
    recent_paper_candidates: List[Dict[str, Any]] = Field(default_factory=list)  # 前文在线学术检索出的候选文献池
    topic_catalog: List[Dict[str, Any]] = Field(default_factory=list)  # 按主题索引的结构化历史档案库 (拒绝全局粗暴扫描)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: List[ChatMessage] = []


class SessionManager:
    @staticmethod
    def _file_path(session_id: str) -> Path:
        return SESSIONS_DIR / f"{session_id}.json"

    @classmethod
    def list_sessions(cls) -> List[Dict[str, Any]]:
        sessions = []
        empty_default_sessions = []
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                msg_count = len(data.get("messages", []))
                sid = data.get("session_id")
                title = data.get("title", "学术研讨会话")
                
                # 记录未使用的默认空会话
                if msg_count == 0 and "新建" in title:
                    empty_default_sessions.append((f, data.get("updated_at", 0)))

                sessions.append({
                    "session_id": sid,
                    "title": title,
                    "domain": data.get("domain", "custom"),
                    "message_count": msg_count,
                    "updated_at": data.get("updated_at", 0),
                })
            except Exception:
                pass

        # 若存在多个未发过消息的重复空会话，自动清理多余的幽灵会话文件
        if len(empty_default_sessions) > 1:
            empty_default_sessions.sort(key=lambda x: x[1], reverse=True)
            for stale_file, _ in empty_default_sessions[1:]:
                try:
                    stale_file.unlink(missing_ok=True)
                except Exception:
                    pass
            # 重新过滤
            sessions = [s for s in sessions if not (s["message_count"] == 0 and "新建" in s["title"]) or s["session_id"] == empty_default_sessions[0][0].stem]

        if not sessions:
            init_s = cls.create_session(title="新建学术研讨会话")
            sessions.append({
                "session_id": init_s.session_id,
                "title": init_s.title,
                "domain": init_s.domain,
                "message_count": 0,
                "updated_at": init_s.updated_at,
            })

        sessions.sort(key=lambda x: x["updated_at"], reverse=True)
        return sessions

    @classmethod
    def create_session(cls, title: str = "新建研讨会话", domain: str = "custom", session_id: Optional[str] = None) -> ConversationSession:
        # 如果是新建默认空会话，且当前已有一个未使用的空会话，则直接复用，避免产生冗余空会话
        if not session_id and ("新建" in title):
            for f in SESSIONS_DIR.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if len(data.get("messages", [])) == 0 and "新建" in data.get("title", ""):
                        return ConversationSession(**data)
                except Exception:
                    pass

        sid = session_id or f"session_{str(uuid.uuid4())[:8]}"
        session = ConversationSession(
            session_id=sid,
            title=title,
            domain=domain,
            messages=[],
        )
        cls.save_session(session)
        return session

    @classmethod
    def get_session(cls, session_id: str) -> Optional[ConversationSession]:
        p = cls._file_path(session_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return ConversationSession(**data)
        except Exception:
            return None

    @classmethod
    def save_session(cls, session: ConversationSession):
        session.updated_at = time.time()
        p = cls._file_path(session.session_id)
        p.write_text(json.dumps(session.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def rename_session(cls, session_id: str, new_title: str) -> Optional[ConversationSession]:
        session = cls.get_session(session_id)
        if session:
            session.title = new_title.strip()
            cls.save_session(session)
            return session
        return None

    @classmethod
    def get_active_doc(cls, session_id: str) -> Optional[str]:
        """获取当前会话正在聚焦讨论的文献名称 (优先当前标记，其次回溯最近证据)"""
        session = cls.get_session(session_id)
        if not session:
            return None
        if session.active_doc:
            return session.active_doc
        # 回溯查找最近一次命中证据的文献
        for m in reversed(session.messages):
            if m.retrieved_evidence:
                first_ev = m.retrieved_evidence[0]
                doc_name = first_ev.get("doc_name")
                if doc_name:
                    session.active_doc = doc_name
                    cls.save_session(session)
                    return doc_name
        return None

    @classmethod
    def get_topic_state(cls, session_id: str) -> Dict[str, Any]:
        """获取当前会话的主题生命周期状态机"""
        session = cls.get_session(session_id)
        if not session or not session.topic_state:
            return {"current_topic": "", "sub_topic": "", "status": "INIT", "evolution_path": []}
        return session.topic_state

    @classmethod
    def update_topic_state(cls, session_id: str, new_state: Dict[str, Any]) -> None:
        """更新并持久化保存会话的主题生命周期状态机"""
        session = cls.get_session(session_id)
        if session and new_state:
            session.topic_state = new_state
            cls.save_session(session)

    @classmethod
    def add_message(cls, session_id: str, message: ChatMessage) -> ConversationSession:
        session = cls.get_session(session_id)
        if not session:
            session = cls.create_session(title="新建研讨会话", session_id=session_id)

        # 如果是新会话的第一条用户发言，自动提炼为会话标题
        if len(session.messages) == 0 and message.sender == "user":
            snippet = message.content.strip().replace("\n", " ")[:18]
            session.title = snippet

        # 如果智能体回复中命中了特定文献证据，自动将该文献绑定为会话聚焦文献
        if message.sender == "agent" and message.retrieved_evidence:
            first_doc = message.retrieved_evidence[0].get("doc_name")
            if first_doc:
                session.active_doc = first_doc

        session.messages.append(message)
        cls.save_session(session)
        return session

    @classmethod
    def update_recent_candidates(cls, session_id: str, candidates: List[Dict[str, Any]]) -> Optional[ConversationSession]:
        session = cls.get_session(session_id)
        if not session:
            return None
        session.recent_paper_candidates = list(candidates)
        cls.save_session(session)
        return session

    @classmethod
    def get_recent_candidates(cls, session_id: str) -> List[Dict[str, Any]]:
        session = cls.get_session(session_id)
        if not session:
            return []
        return list(session.recent_paper_candidates or [])

    @classmethod
    def record_turn_topic(
        cls,
        session_id: str,
        turn_idx: int,
        query: str,
        agent_msg: ChatMessage,
    ) -> Optional[Dict[str, Any]]:
        """在每轮对话结束时，自动归档本轮主题档案 (绝无暴力截断，提炼核心结论与实体)"""
        session = cls.get_session(session_id)
        if not session:
            return None

        try:
            intent_dec = agent_msg.intent_decision or {}
            topic_name = intent_dec.get("subject_entity") or intent_dec.get("target_topic") or session.topic_state.get("current_topic") or query[:30]
            action_type = intent_dec.get("intent") or "academic_discussion"
            disciplines = intent_dec.get("disciplines") or []
            
            # 开放领域多学科实体自适应提取 (严格基于意图解析与导图骨架，100% 拒绝任何硬编码)
            entities = list(disciplines)
            if intent_dec.get("subject_entity") and intent_dec["subject_entity"] not in entities:
                entities.append(intent_dec["subject_entity"])
            if intent_dec.get("target_topic") and intent_dec["target_topic"] not in entities:
                entities.append(intent_dec["target_topic"])
            for c in intent_dec.get("core_concepts", []):
                if c and c not in entities:
                    entities.append(c)

            # 自动提取导图中的高维结构化标题作为本轮概念锚点 (完全由模型生成内容决定，跨学科自适应)
            if agent_msg.mindmap_markdown:
                for mm_line in agent_msg.mindmap_markdown.split("\n"):
                    if mm_line.startswith("## ") or mm_line.startswith("### "):
                        concept = mm_line.lstrip("# -*").strip()
                        if 2 <= len(concept) <= 30 and concept not in entities:
                            entities.append(concept)

            # 优先从已有的结构化导图脉络中无损提炼核心定论分支 (绝无字符串硬截断)
            findings = ""
            if agent_msg.mindmap_markdown:
                lines = [l.strip() for l in agent_msg.mindmap_markdown.split("\n") if l.strip() and not l.strip().startswith("# 🧠")]
                # 取前 10 个高维骨架分支
                findings = "\n".join(lines[:10])
            
            if not findings and agent_msg.content:
                # 备用：按自然段落提取核心论断段落
                paras = [p.strip() for p in agent_msg.content.split("\n\n") if p.strip()]
                findings = "\n".join(paras[:2])

            topic_node = {
                "topic_id": f"T{len(session.topic_catalog) + 1}",
                "topic_name": topic_name,
                "action_type": action_type,
                "entities": entities,
                "key_findings": findings or "完成本轮数理研讨与推演",
                "turn_idx": turn_idx,
                "created_at": time.time(),
            }
            session.topic_catalog.append(topic_node)
            cls.save_session(session)
            print(f"🗂️ [SessionManager] 已为会话 {session_id} 沉淀主题档案 [{topic_node['topic_id']}]: 《{topic_name}》 (类型: {action_type})")
            return topic_node
        except Exception as e:
            print(f"⚠️ [SessionManager Topic Record Note]: {e}")
            return None

    @classmethod
    def match_best_historical_topic(
        cls,
        session_id: str,
        query: str,
        current_intent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """在历史中按语义实体与意图对齐锁定最合适的主题档案 (0 硬编码，纯语义与实体驱动)"""
        session = cls.get_session(session_id)
        if not session or not session.topic_catalog:
            return None

        q_lower = query.lower()
        catalog = session.topic_catalog
        best_topic = None
        best_score = 0.0

        for topic in reversed(catalog):
            score = 0.0
            act = topic.get("action_type", "")
            t_name = topic.get("topic_name", "")
            t_entities = topic.get("entities", [])

            # 意图动作对齐 (纯语义意图类型匹配，杜绝死词表猜词)
            if current_intent and act == current_intent:
                score += 0.5

            # 动态学术实体概念重合度
            for ent in t_entities:
                if ent and ent.lower() in q_lower:
                    score += 0.35

            if t_name and t_name.lower() in q_lower:
                score += 0.4

            # 临近惯性加分 (最近一轮微幅加分，便于顺流递进)
            if topic == catalog[-1]:
                score += 0.15

            if score > best_score:
                best_score = score
                best_topic = topic

        if best_topic and best_score >= 0.3:
            return best_topic
        return None

    @classmethod
    def get_scoped_thematic_context(
        cls,
        session_id: str,
        query: str,
        current_intent: Optional[str] = None,
    ) -> str:
        """外科手术式按意图定向调取最匹配的历史主题定论 (0 硬编码)"""
        session = cls.get_session(session_id)
        if not session:
            return ""

        catalog = session.topic_catalog or []
        if not catalog:
            return session.history_summary or ""

        best = cls.match_best_historical_topic(session_id, query, current_intent=current_intent)
        if best:
            print(f"🎯 [SessionManager] 外科手术式锁定最佳匹配主题: [{best.get('topic_id')}] 《{best.get('topic_name')}》")
            return (
                f"【与当前提问最匹配的前序研讨主题定论 (锁定主题: 《{best.get('topic_name')}》)】:\n"
                f"- 主题类型: {best.get('action_type')}\n"
                f"- 核心理论定论与基线依据:\n{best.get('key_findings')}\n"
            )

        # 兜底：若是普通自然追问且只有少数轮次，取紧邻上一轮定论
        if catalog:
            last = catalog[-1]
            return (
                f"【紧邻上一轮研讨定论 (主题: 《{last.get('topic_name')}》)】:\n"
                f"{last.get('key_findings')}\n"
            )

        return session.history_summary or ""

    @classmethod
    def compress_and_update_summary(
        cls,
        session_id: str,
        client: Any,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """对全部历史对话进行增量结构化压缩，直接提炼为唯一的长期研讨记忆卡片（无原始轮堆叠）"""
        session = cls.get_session(session_id)
        if not session or not session.messages:
            return ""

        effective_model = model or get_default_llm_model()
        total_msgs = len(session.messages)
        start_idx = session.last_summarized_index
        end_idx = total_msgs

        # 如果没有未压缩的新消息，直接返回已有摘要
        if end_idx <= start_idx:
            return session.history_summary or ""

        uncompressed_slice = session.messages[start_idx:end_idx]
        formatted_dialogue = []
        partner_name = get_assistant_name()
        for m in uncompressed_slice:
            role = "用户" if m.sender == "user" else partner_name
            content = (m.content or "").strip()
            if content:
                formatted_dialogue.append(f"{role}: {content[:3000]}")

        if not formatted_dialogue:
            return session.history_summary or ""

        dialogue_text = "\n".join(formatted_dialogue)
        existing_summary = session.history_summary or "暂无前序摘要（这是初始研讨阶段）"

        # 挂载当前会话已捕获的结构化学术文献资产 (杜绝文献在长期记忆中丢失)
        papers_context = ""
        if session.recent_paper_candidates:
            cand_lines = []
            for c in session.recent_paper_candidates[:8]:
                src = c.get("source", "online")
                src_type = "arXiv预印本" if src == "arxiv" else ("正式发表期刊" if src in ["openalex", "crossref"] else src)
                cand_lines.append(f"- [{src_type}] 《{c.get('title')}》 ({c.get('year')}, DOI/ID: {c.get('paper_id')}) | 主题/摘要: {c.get('summary', '')[:120]}")
            papers_context = "\n【当前会话中已就绪/讨论的学术文献清单】:\n" + "\n".join(cand_lines) + "\n"

        prompt = build_session_memory_prompt(existing_summary, papers_context, dialogue_text)

        try:
            resp = client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": "你负责提炼多轮学术对话长期全息记忆，保持高信息密度，完整保留文献与公式要点。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=get_llm_timeout(60.0),
            )
            new_summary = resp.choices[0].message.content.strip()
            session.history_summary = new_summary
            session.last_summarized_index = end_idx
            cls.save_session(session)
            print(f"🧠 [SessionManager] 已为会话 {session_id} 成功增量压缩历史（索引 {start_idx} -> {end_idx}）")
            return new_summary
        except Exception as e:
            print(f"⚠️ [SessionManager] 历史对话压缩摘要生成异常: {e}")
            return session.history_summary or ""

    @classmethod
    def clear_session_messages(cls, session_id: str) -> Optional[ConversationSession]:
        """清空指定会话的全部消息记录与摘要，但保留会话本身"""
        session = cls.get_session(session_id)
        if session:
            session.messages = []
            session.history_summary = ""
            session.last_summarized_index = 0
            cls.save_session(session)
            return session
        return None

    @classmethod
    def clear_all_sessions(cls):
        """清空所有历史会话并新建一个初始空白研讨空间"""
        for f in SESSIONS_DIR.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        return cls.create_session(title="新建学术研讨会话")

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        p = cls._file_path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False
