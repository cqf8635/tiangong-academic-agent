"""Dynamic User-Defined Session and Multi-Domain Conversation Store.

Features:
1. No rigid hardcoded channels: Users freely create, rename, and delete sessions.
2. Auto-naming: First user query can dynamically name the conversation.
3. Completely isolated conversation histories per session.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from weakref import WeakValueDictionary
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from config_utils import get_default_llm_model, get_llm_timeout
from prompts import build_session_memory_prompt, get_assistant_name

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_LOCKS: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
_SESSION_LOCKS_GUARD = threading.Lock()


def _serialized_session_access(method):
    @wraps(method)
    def wrapper(cls, session_or_id, *args, **kwargs):
        session_id = session_or_id if isinstance(session_or_id, str) else session_or_id.session_id
        with _SESSION_LOCKS_GUARD:
            lock = _SESSION_LOCKS.setdefault(session_id, threading.RLock())
        with lock:
            return method(cls, session_or_id, *args, **kwargs)
    return wrapper


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str  # "user" | "agent"
    content: str
    intent_decision: Optional[Dict[str, Any]] = None
    task_snapshot: Optional[Dict[str, Any]] = None
    delivery_checks: Optional[Dict[str, Any]] = None
    specialist_results: List[Dict[str, Any]] = Field(default_factory=list)
    meeting_record_path: Optional[str] = None
    control_result: Optional[Dict[str, Any]] = None
    target_agent: Optional[str] = None
    retrieved_evidence: List[Dict[str, Any]] = []
    mindmap_markdown: Optional[str] = None
    audio_url: Optional[str] = None
    agent_dialogue_history: Optional[List[Dict[str, Any]]] = None
    synthesized_evidence_slate: Optional[str] = None
    dialectical_notes: Optional[str] = None
    review_verdict: Optional[Dict[str, Any]] = None
    review_checks: List[Dict[str, Any]] = Field(default_factory=list)
    review_notice: Optional[str] = None
    review_retrieval_log: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_plan: Optional[Dict[str, Any]] = None
    retrieval_coverage: Optional[Dict[str, Any]] = None
    source_diagnostics: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None  # 用户附带的截图或图片 (URL 或 Base64)
    timestamp: str = Field(default_factory=lambda: time.strftime("%H:%M:%S"))


class ConversationSession(BaseModel):
    session_id: str
    title: str
    domain: str = "custom"
    active_doc: Optional[str] = None      # 当前会话正在聚焦研讨的核心文献/课件名称
    tts_voice: str = "zh-CN-XiaoyiNeural"
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
    shared_resources: List[Dict[str, Any]] = Field(default_factory=list)  # 会话隔离的持久共享证据池
    shared_pool: List[Dict[str, Any]] = Field(default_factory=list)  # 跨轮统一共享信息池：状态、成果、意见与证据
    shared_pool_migrated: bool = False
    topic_catalog: List[Dict[str, Any]] = Field(default_factory=list)  # 按主题索引的结构化历史档案库 (拒绝全局粗暴扫描)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    messages: List[ChatMessage] = []


class SessionManager:
    @classmethod
    @contextmanager
    def session_lock(cls, session_id: str):
        with _SESSION_LOCKS_GUARD:
            lock = _SESSION_LOCKS.setdefault(session_id, threading.RLock())
        with lock:
            yield

    @staticmethod
    def _prune_shared_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Bound the per-session reusable index; full conversation records remain in messages."""
        deduped = {}
        for item in items:
            item_id = item.get("item_id")
            if item_id:
                deduped[item_id] = item
        newest = sorted(
            deduped.values(),
            key=lambda item: float(item.get("last_accessed_at") or item.get("updated_at") or 0),
            reverse=True,
        )
        max_items = max(1, int(os.getenv("SESSION_SHARED_POOL_MAX_ITEMS", "1000")))
        max_bytes = max(1024, int(os.getenv("SESSION_SHARED_POOL_MAX_BYTES", str(32 * 1024 * 1024))))
        kept = []
        used_bytes = 0
        for item in newest:
            size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if len(kept) >= max_items or used_bytes + size > max_bytes:
                continue
            kept.append(item)
            used_bytes += size
        return kept

    @staticmethod
    def _prune_shared_resources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        newest = sorted(
            items,
            key=lambda item: float(item.get("last_seen_at") or item.get("last_fetched_at") or item.get("updated_at") or 0),
            reverse=True,
        )
        max_items = max(1, int(os.getenv("SESSION_SHARED_RESOURCE_LIMIT", "500")))
        max_bytes = max(1024, int(os.getenv("SESSION_SHARED_RESOURCE_MAX_BYTES", str(64 * 1024 * 1024))))
        kept = []
        used_bytes = 0
        for item in newest:
            size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if len(kept) >= max_items or used_bytes + size > max_bytes:
                continue
            kept.append(item)
            used_bytes += size
        return kept

    @classmethod
    def _sync_resources_with_pool(cls, session: ConversationSession) -> None:
        retained_ids = {
            str(item.get("resource_id") or item.get("item_id"))
            for item in session.shared_pool if item.get("kind") == "evidence"
        }
        session.shared_resources = cls._prune_shared_resources([
            resource for resource in session.shared_resources
            if str(resource.get("resource_id") or cls._resource_identity(resource)) in retained_ids
        ])

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
            resources = {}
            for resource in data.get("shared_resources", []):
                if isinstance(resource, dict):
                    rid = str(resource.get("resource_id") or cls._resource_identity(resource))
                    resources[rid] = resource
            for message in data.get("messages", []):
                expanded = []
                for evidence in message.get("retrieved_evidence", []):
                    if isinstance(evidence, dict) and set(evidence) == {"resource_id"}:
                        resource = resources.get(str(evidence["resource_id"]))
                        if resource:
                            expanded.append(dict(resource))
                            continue
                    expanded.append(evidence)
                message["retrieved_evidence"] = expanded
            return ConversationSession(**data)
        except Exception:
            return None

    @classmethod
    @_serialized_session_access
    def save_session(cls, session: ConversationSession):
        session.updated_at = time.time()
        p = cls._file_path(session.session_id)
        data = session.model_dump()
        resources = {
            str(resource.get("resource_id") or cls._resource_identity(resource)): resource
            for resource in session.shared_resources
        }
        for message in data.get("messages", []):
            compacted = []
            for evidence in message.get("retrieved_evidence", []):
                if not isinstance(evidence, dict):
                    compacted.append(evidence)
                    continue
                rid = str(evidence.get("resource_id") or cls._resource_identity(evidence))
                resource = resources.get(rid)
                same_content = bool(
                    resource
                    and evidence.get("text")
                    and evidence.get("text") == resource.get("text")
                    and (evidence.get("location_label") or "") == (resource.get("location_label") or "")
                )
                if same_content:
                    compacted.append({"resource_id": rid})
                else:
                    compacted.append(evidence)
            message["retrieved_evidence"] = compacted
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, delete=False) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        temp_path.replace(p)

    @classmethod
    @_serialized_session_access
    def get_shared_pool(cls, session_id: str, include_content: bool = True) -> List[Dict[str, Any]]:
        """Return the unified pool, migrating legacy resource/topic records in place."""
        session = cls.get_session(session_id)
        if not session:
            return []
        indexed = {item.get("item_id"): item for item in session.shared_pool if item.get("item_id")}
        needs_migration = not session.shared_pool_migrated
        changed = False
        messages_by_id = {message.id: message for message in session.messages}
        for item in indexed.values():
            if item.get("kind") == "evidence" and isinstance(item.get("resource"), dict):
                resource = item.pop("resource")
                item["resource_id"] = item.get("resource_id") or resource.get("resource_id") or item.get("item_id")
                item["source"] = item.get("source") or resource.get("source") or resource.get("doc_type")
                item["evidence_level"] = item.get("evidence_level") or resource.get("evidence_level")
                item["title"] = item.get("title") or resource.get("title") or resource.get("doc_name")
                item.pop("content", None)
                changed = True
            if item.get("kind") == "turn_record" and item.get("content") and not item.get("message_id"):
                candidate_id = str(item.get("item_id", "")).rsplit(":", 1)[-1]
                if candidate_id in messages_by_id:
                    item["message_id"] = candidate_id
                    item.pop("content", None)
                    changed = True
        if needs_migration:
            for resource in session.shared_resources:
                rid = resource.get("resource_id") or cls._resource_identity(resource)
                if rid and rid not in indexed:
                    indexed[rid] = {
                        "item_id": rid, "kind": "evidence", "turn_idx": resource.get("turn_idx"),
                        "title": resource.get("title") or resource.get("doc_name"),
                        "task": resource.get("task") or resource.get("query"),
                        "source": resource.get("source") or resource.get("doc_type"),
                        "evidence_level": resource.get("evidence_level"),
                        "superseded_by": resource.get("superseded_by"),
                        "resource_id": rid, "updated_at": resource.get("updated_at", 0),
                    }
            for topic in session.topic_catalog:
                item_id = f"topic:{topic.get('topic_id')}"
                if item_id not in indexed:
                    indexed[item_id] = {
                        "item_id": item_id, "kind": "turn_record", "turn_idx": topic.get("turn_idx"),
                        "title": topic.get("topic_name"), "task": topic.get("task"),
                        "content": topic.get("key_findings", ""), "state": {},
                        "entities": topic.get("entities", []), "updated_at": topic.get("created_at", 0),
                    }
            # The migrated shared-pool record becomes the canonical topic index.
            session.topic_catalog = []
            session.shared_pool_migrated = True
        pruned = cls._prune_shared_pool(list(indexed.values()))
        if pruned != session.shared_pool or needs_migration or changed:
            session.shared_pool = pruned
            cls._sync_resources_with_pool(session)
            cls.save_session(session)
        if not include_content:
            return [
                {key: value for key, value in item.items()
                 if key not in {"state", "review", "evidence_item_ids", "content", "candidate"}}
                for item in session.shared_pool
            ]
        message_by_id = {message.id: message for message in session.messages}
        return [
            ({**item, "content": message_by_id[item["message_id"]].content}
             if item.get("kind") == "turn_record" and item.get("message_id") in message_by_id and not item.get("content")
             else item)
            for item in session.shared_pool
        ]

    @classmethod
    @_serialized_session_access
    def get_shared_pool_items(cls, session_id: str, item_ids: List[str]) -> List[Dict[str, Any]]:
        """Hydrate only selected records and the evidence explicitly linked to them."""
        if not item_ids:
            return []
        if not cls.get_session(session_id):
            return []
        cls.get_shared_pool(session_id, include_content=False)
        session = cls.get_session(session_id)
        if not session:
            return []
        by_id = {item.get("item_id"): item for item in session.shared_pool}
        message_by_id = {message.id: message for message in session.messages}
        selected_ids = set(item_ids)
        for item_id in list(selected_ids):
            record = by_id.get(item_id)
            if record and record.get("kind") == "turn_record":
                message = next((m for m in session.messages if m.id == record.get("message_id")), None)
                if message:
                    selected_ids.update(
                        str(e.get("resource_id") or cls._resource_identity(e))
                        for e in message.retrieved_evidence if isinstance(e, dict)
                    )
        resources = {
            str(resource.get("resource_id") or cls._resource_identity(resource)): resource
            for resource in session.shared_resources
        }
        evidence_by_id = {}
        for message in session.messages:
            for evidence in message.retrieved_evidence:
                if isinstance(evidence, dict):
                    evidence_by_id[cls._resource_identity(evidence)] = evidence
        candidates = {
            str(candidate.get("shared_item_id") or ""): candidate
            for candidate in session.recent_paper_candidates
        }
        hydrated = []
        for item_id in selected_ids:
            record = by_id.get(item_id)
            if not record:
                continue
            item = dict(record)
            if item.get("kind") == "turn_record" and item.get("message_id") in message_by_id:
                message = message_by_id[item["message_id"]]
                item["content"] = message.content
                item["review"] = message.review_verdict
                item["state"] = {
                    **(item.get("state") or {}),
                    "intent_decision": message.intent_decision,
                    "task_snapshot": message.task_snapshot,
                    "delivery_checks": message.delivery_checks,
                    "specialist_results": message.specialist_results,
                    "meeting_record_path": message.meeting_record_path,
                    "retrieved_evidence": message.retrieved_evidence,
                    "retrieval_plan": message.retrieval_plan,
                    "retrieval_coverage": message.retrieval_coverage,
                    "review_verdict": message.review_verdict,
                    "review_checks": message.review_checks,
                    "review_notice": message.review_notice,
                    "synthesized_evidence_slate": message.synthesized_evidence_slate,
                    "dialectical_notes": message.dialectical_notes,
                }
            if item.get("kind") == "evidence":
                resource_id = str(item.get("resource_id") or item_id)
                cached_resource = resources.get(resource_id)
                archived_resource = evidence_by_id.get(resource_id)
                item["resource"] = (
                    cached_resource if cached_resource and (cached_resource.get("text") or cached_resource.get("summary"))
                    else archived_resource or cached_resource
                )
            elif item.get("kind") == "candidate":
                item["candidate"] = item.get("candidate") or candidates.get(str(item_id))
            hydrated.append(item)
        return hydrated

    @classmethod
    @_serialized_session_access
    def update_shared_resources(cls, session_id: str, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist useful evidence within this session; never cross session boundaries."""
        session = cls.get_session(session_id)
        if not session:
            return []
        indexed = {}
        for old in session.shared_resources:
            key = cls._resource_identity(old)
            if key:
                indexed[key] = old
        for incoming in resources or []:
            if not isinstance(incoming, dict) or not (incoming.get("doc_name") or incoming.get("title")):
                continue
            item = dict(incoming)
            key = cls._resource_identity(item)
            if not key:
                continue
            existing = indexed.get(key)
            if existing:
                provenance = list(existing.get("provenance_records") or [])
                current_provenance = dict(item.get("provenance") or {})
                current_provenance.setdefault("source", item.get("source") or item.get("doc_type"))
                current_provenance.setdefault("url", item.get("url"))
                current_provenance.setdefault("version", item.get("version") or item.get("revision") or item.get("commit_sha"))
                current_provenance.setdefault("content_scope", item.get("evidence_level"))
                provenance_added = current_provenance not in provenance
                if current_provenance not in provenance:
                    provenance.append(current_provenance)
                # A shorter passage can still contain complementary information.
                from workflow_contracts import merge_text
                richer_content = False
                for field in ("text", "summary"):
                    merged, delta = merge_text(existing.get(field), item.get(field))
                    if delta:
                        existing[field] = merged
                        richer_content = True
                if richer_content:
                    if item.get("evidence_level") not in {None, "metadata", "bibliographic_metadata"}:
                        existing["evidence_level"] = item.get("evidence_level")
                    for field, value in item.items():
                        if not existing.get(field) and value:
                            existing[field] = value
                existing["provenance_records"] = provenance
                now = time.time()
                existing["last_seen_at"] = now
                if provenance_added or richer_content:
                    existing["updated_at"] = now
                if not item.get("reused_from_shared_pool"):
                    existing["last_fetched_at"] = now
            else:
                item["resource_id"] = key
                item["created_at"] = time.time()
                item["updated_at"] = item["created_at"]
                item["last_seen_at"] = item["created_at"]
                if not item.get("reused_from_shared_pool"):
                    item["last_fetched_at"] = item["created_at"]
                provenance = dict(item.get("provenance") or {})
                provenance.setdefault("source", item.get("source") or item.get("doc_type"))
                provenance.setdefault("url", item.get("url"))
                provenance.setdefault("version", item.get("version") or item.get("revision") or item.get("commit_sha"))
                provenance.setdefault("content_scope", item.get("evidence_level"))
                item["provenance_records"] = [provenance]
                indexed[key] = item
        # When a newer explicit version arrives, keep the old turn's evidence in
        # message history but remove its duplicate full-text cache copy.
        archived_resource_ids = {
            cls._resource_identity(evidence)
            for message in session.messages
            for evidence in message.retrieved_evidence
            if isinstance(evidence, dict)
        }
        for resource in indexed.values():
            version = cls._resource_version(resource)
            if not version:
                continue
            base_id = cls._resource_base_identity(resource)
            for other in indexed.values():
                other_version = cls._resource_version(other)
                if other is resource or not other_version or other_version == version:
                    continue
                if cls._resource_base_identity(other) == base_id:
                    other["superseded_by"] = resource.get("resource_id") or cls._resource_identity(resource)
                    other["superseded_at"] = time.time()
                    old_id = other.get("resource_id") or cls._resource_identity(other)
                    if old_id in archived_resource_ids:
                        other.pop("text", None)
                        other.pop("summary", None)
        session.shared_resources = cls._prune_shared_resources(list(indexed.values()))
        # Keep the unified pool as the canonical cross-turn view while retaining
        # shared_resources for compatibility with existing retrieval paths.
        pool = {item.get("item_id"): item for item in session.shared_pool if item.get("item_id")}
        for resource in session.shared_resources:
            rid = resource.get("resource_id") or cls._resource_identity(resource)
            pool[rid] = {
                "item_id": rid,
                "kind": "evidence",
                "turn_idx": resource.get("turn_idx"),
                "title": resource.get("title") or resource.get("doc_name"),
                "task": resource.get("task") or resource.get("query"),
                "resource_id": rid,
                "source": resource.get("source") or resource.get("doc_type"),
                "evidence_level": resource.get("evidence_level"),
                "last_fetched_at": resource.get("last_fetched_at"),
                "superseded_by": resource.get("superseded_by"),
                "updated_at": resource.get("updated_at", time.time()),
            }
        session.shared_pool = cls._prune_shared_pool(list(pool.values()))
        cls._sync_resources_with_pool(session)
        cls.save_session(session)
        return session.shared_resources

    @classmethod
    @_serialized_session_access
    def publish_turn_to_shared_pool(
        cls, session_id: str, turn_idx: int, query: str, message: ChatMessage,
        turn_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Publish this turn's task, state, answer, review and evidence as selectable records."""
        session = cls.get_session(session_id)
        if not session:
            return []
        now = time.time()
        pool = {item.get("item_id"): item for item in session.shared_pool if item.get("item_id")}
        item_id = f"turn:{turn_idx}:{message.id}"
        intent = message.intent_decision or {}
        def as_list(value):
            return value if isinstance(value, list) else ([value] if value else [])

        pool[item_id] = {
            "item_id": item_id,
            "kind": "turn_record",
            "turn_idx": turn_idx,
            "title": intent.get("subject_entity") or intent.get("target_topic") or query[:100],
            "task": query,
            "message_id": message.id,
            "state": turn_state or {},
            "action_type": intent.get("intent"),
            "entities": list(dict.fromkeys(
                as_list(intent.get("disciplines"))
                + as_list(intent.get("core_concepts"))
                + [value for value in (intent.get("subject_entity"), intent.get("target_topic")) if value]
            )),
            "updated_at": now,
        }
        for evidence in message.retrieved_evidence:
            if not isinstance(evidence, dict):
                continue
            rid = evidence.get("resource_id") or cls._resource_identity(evidence)
            if not rid:
                continue
            resource = dict(evidence)
            resource.setdefault("resource_id", rid)
            resource.setdefault("turn_idx", turn_idx)
            resource.setdefault("task", query)
            existing = pool.get(rid, {})
            pool[rid] = {
                "item_id": rid, "kind": "evidence", "turn_idx": turn_idx,
                "title": resource.get("title") or resource.get("doc_name"),
                "task": query, "resource_id": rid,
                "source": resource.get("source") or resource.get("doc_type"),
                "evidence_level": resource.get("evidence_level"),
                "last_fetched_at": resource.get("last_fetched_at") or existing.get("last_fetched_at"),
                "last_accessed_at": existing.get("last_accessed_at"),
                "superseded_by": resource.get("superseded_by") or existing.get("superseded_by"),
                "updated_at": now,
            }
        session.shared_pool = cls._prune_shared_pool(list(pool.values()))
        cls._sync_resources_with_pool(session)
        session.shared_pool_migrated = True
        cls.save_session(session)
        return session.shared_pool

    @classmethod
    @_serialized_session_access
    def touch_shared_pool_items(cls, session_id: str, item_ids: List[str]) -> None:
        """Refresh recency for records actually selected by a task."""
        if not item_ids:
            return
        session = cls.get_session(session_id)
        if not session:
            return
        selected = set(item_ids)
        now = time.time()
        changed = False
        for item in session.shared_pool:
            if item.get("item_id") in selected:
                item["last_accessed_at"] = now
                changed = True
        if changed:
            session.shared_pool = cls._prune_shared_pool(session.shared_pool)
            cls.save_session(session)

    @staticmethod
    def _resource_identity(item: Dict[str, Any]) -> str:
        import hashlib
        stable = SessionManager._resource_base_identity(item)
        version = SessionManager._resource_version(item)
        if version:
            stable = f"{stable}|version:{version}"
        return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24] if stable else ""

    @staticmethod
    def _resource_base_identity(item: Dict[str, Any]) -> str:
        return str(item.get("doi") or item.get("arxiv_id") or item.get("paper_id") or item.get("url") or item.get("doc_name") or item.get("title") or "").strip().casefold()

    @staticmethod
    def _resource_version(item: Dict[str, Any]) -> str:
        return str(item.get("version") or item.get("revision") or item.get("commit_sha") or "").strip().casefold()

    @classmethod
    @_serialized_session_access
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
    @_serialized_session_access
    def update_topic_state(cls, session_id: str, new_state: Dict[str, Any]) -> None:
        """更新并持久化保存会话的主题生命周期状态机"""
        session = cls.get_session(session_id)
        if session and new_state:
            session.topic_state = new_state
            cls.save_session(session)

    @classmethod
    @_serialized_session_access
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
    @_serialized_session_access
    def update_recent_candidates(cls, session_id: str, candidates: List[Dict[str, Any]]) -> Optional[ConversationSession]:
        session = cls.get_session(session_id)
        if not session:
            return None
        normalized = []
        pool = {item.get("item_id"): item for item in session.shared_pool if item.get("item_id")}
        previous_candidate_records = {item_id: item for item_id, item in pool.items() if item.get("kind") == "candidate"}
        pool = {item_id: item for item_id, item in pool.items() if item.get("kind") != "candidate"}
        previous_candidates = {
            str(candidate.get("shared_item_id") or f"candidate:{cls._resource_identity(candidate)}"): candidate
            for candidate in session.recent_paper_candidates if isinstance(candidate, dict)
        }
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            item = dict(candidate)
            rid = cls._resource_identity(item)
            if not rid:
                continue
            item_id = f"candidate:{rid}"
            item["shared_item_id"] = item_id
            previous = previous_candidates.get(item_id, {})
            item["last_fetched_at"] = item.get("last_fetched_at") or previous.get("last_fetched_at") or time.time()
            normalized.append(item)
            pool[item_id] = {
                "item_id": item_id,
                "kind": "candidate",
                "title": item.get("title") or item.get("doc_name"),
                "task": item.get("summary") or item.get("query"),
                "source": item.get("source") or item.get("doc_type"),
                "year": item.get("year"),
                "last_fetched_at": item.get("last_fetched_at"),
                "last_accessed_at": previous_candidate_records.get(item_id, {}).get("last_accessed_at"),
                "updated_at": item.get("last_fetched_at") or previous_candidate_records.get(item_id, {}).get("updated_at") or time.time(),
            }
        session.recent_paper_candidates = normalized
        session.shared_pool = cls._prune_shared_pool(list(pool.values()))
        session.shared_pool_migrated = True
        cls._sync_resources_with_pool(session)
        cls.save_session(session)
        return session

    @classmethod
    def get_recent_candidates(cls, session_id: str) -> List[Dict[str, Any]]:
        session = cls.get_session(session_id)
        if not session:
            return []
        return list(session.recent_paper_candidates or [])

    @classmethod
    @_serialized_session_access
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
                "task": query,
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
    def _historical_turn_records(cls, session: ConversationSession) -> List[Dict[str, Any]]:
        """Build a compact history index from the shared pool, with legacy topic fallback."""
        records = {
            item.get("turn_idx"): item
            for item in session.shared_pool
            if item.get("kind") == "turn_record" and item.get("turn_idx") is not None
        }
        for topic in session.topic_catalog:
            turn_idx = topic.get("turn_idx")
            if turn_idx not in records:
                records[turn_idx] = topic
        return sorted(records.values(), key=lambda item: item.get("turn_idx") or 0)

    @classmethod
    def match_best_historical_topic(
        cls,
        session_id: str,
        query: str,
        current_intent: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """在历史中按语义实体与意图对齐锁定最合适的主题档案 (0 硬编码，纯语义与实体驱动)"""
        session = cls.get_session(session_id)
        if not session:
            return None

        cls.get_shared_pool(session_id, include_content=False)
        session = cls.get_session(session_id)
        if not session:
            return None

        q_lower = query.lower()
        catalog = cls._historical_turn_records(session)
        best_topic = None
        best_score = 0.0

        for topic in reversed(catalog):
            score = 0.0
            act = topic.get("action_type", "")
            t_name = topic.get("topic_name") or topic.get("title", "")
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

        cls.get_shared_pool(session_id, include_content=False)
        session = cls.get_session(session_id)
        if not session:
            return ""
        catalog = cls._historical_turn_records(session)
        if not catalog:
            return session.history_summary or ""

        best = cls.match_best_historical_topic(session_id, query, current_intent=current_intent)
        if best:
            findings = cls._historical_record_findings(session, best)
            best_title = best.get("topic_name") or best.get("title") or ""
            print(f"🎯 [SessionManager] 外科手术式锁定最佳匹配主题: [{best.get('topic_id') or best.get('turn_idx')}] 《{best_title}》")
            return (
                f"【与当前提问最匹配的前序研讨主题定论 (锁定主题: 《{best_title}》)】:\n"
                f"- 主题类型: {best.get('action_type')}\n"
                f"- 核心理论定论与基线依据:\n{findings}\n"
            )

        # 兜底：若是普通自然追问且只有少数轮次，取紧邻上一轮定论
        if catalog:
            last = catalog[-1]
            return (
                f"【紧邻上一轮研讨定论 (主题: 《{last.get('topic_name') or last.get('title') or ''}》)】:\n"
                f"{cls._historical_record_findings(session, last)}\n"
            )

        return session.history_summary or ""

    @staticmethod
    def _historical_record_findings(session: ConversationSession, record: Dict[str, Any]) -> str:
        if record.get("key_findings"):
            return str(record["key_findings"])
        message_id = record.get("message_id")
        message = next((item for item in session.messages if item.id == message_id), None)
        if message:
            if message.mindmap_markdown:
                return message.mindmap_markdown
            paragraphs = [part.strip() for part in message.content.split("\n\n") if part.strip()]
            return "\n".join(paragraphs[:2])
        return str(record.get("content") or "")

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
    @_serialized_session_access
    def clear_session_messages(cls, session_id: str) -> Optional[ConversationSession]:
        """清空指定会话的对话及其状态、候选文献和共享资源，但保留会话本身。"""
        session = cls.get_session(session_id)
        if session:
            session.messages = []
            session.history_summary = ""
            session.last_summarized_index = 0
            session.active_doc = None
            session.topic_state = {"current_topic": "", "sub_topic": "", "status": "INIT", "evolution_path": []}
            session.recent_paper_candidates = []
            session.shared_resources = []
            session.shared_pool = []
            session.shared_pool_migrated = True
            session.topic_catalog = []
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
    @_serialized_session_access
    def delete_session(cls, session_id: str) -> bool:
        p = cls._file_path(session_id)
        if p.exists():
            p.unlink()
            return True
        return False
