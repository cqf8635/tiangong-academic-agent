"""Task-driven retrieval planning, separate from intent classification."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from config_utils import get_default_llm_model, get_llm_timeout, resolve_retrieval_time_constraint
from prompts import TASK_COVERAGE_GUIDANCE, EVIDENCE_GUIDANCE


class RetrievalQuery(BaseModel):
    query: str
    english_query: str = ""


class TaskAlignment(BaseModel):
    status: Literal["matched", "corrected", "unresolved"]
    issues: List[str] = Field(default_factory=list)
    corrected_fields: Dict[str, Any] = Field(default_factory=dict)


class RetrievalPlan(BaseModel):
    requires_evidence: bool = Field(default=True, description="当前任务是否需要超出用户明确前提的可追溯专业资料依据")
    local_search: bool = Field(default=False, description="是否需要检索本地文件或已索引知识库")
    online_search: bool = Field(default=False, description="是否需要检索外部或最新在线资料")
    include_github: bool = Field(default=False, description="是否需要检索代码仓库与工程实现")
    include_web: bool = Field(default=False, description="是否需要检索通用网页、机构网站、技术标准或官方文档")
    include_datasets: bool = Field(default=False, description="是否需要检索公开数据集与基准测试")
    parallel_search: bool = Field(default=False, description="本地检索与在线检索是否互补且适合并行执行")
    selected_shared_item_ids: List[str] = Field(default_factory=list, description="共享池中可直接服务于当前任务的资料记录 ID 列表")
    queries: List[RetrievalQuery] = Field(default_factory=list, description="精简互补的可执行检索词列表；合并同义任务视角")
    reasoning: str = Field(default="", description="检索渠道与执行策略的简要中文说明")
    is_resolved: bool = Field(default=True, description="检索需求与渠道选择是否已明确解析")
    task_alignment: TaskAlignment | None = None


class RetrievalGap(BaseModel):
    query: str
    rationale: str = ""
    sources: List[str] = Field(default_factory=list)
    requirement: str = ""
    strategy: str = ""


class CoverageCheck(BaseModel):
    requirement: str
    task_ids: List[int] = Field(default_factory=list)
    status: Literal["supported", "partial", "missing"] = "missing"
    evidence_ids: List[int] = Field(default_factory=list)
    note: str = ""
    supported_facts: List[str] = Field(default_factory=list)


class IrrelevantEvidence(BaseModel):
    evidence_id: int
    reason: str
    basis: Any = Field(default_factory=list)


class RetrievalCoverage(BaseModel):
    sufficient: bool = False
    gaps: List[RetrievalGap] = Field(default_factory=list)
    unresolved: List[str] = Field(default_factory=list)
    summary: str = ""
    task_checks: List[CoverageCheck] = Field(default_factory=list)
    irrelevant_evidence: List[IrrelevantEvidence] = Field(default_factory=list)


def apply_evidence_relevance(evidence, exclusions, user_query="", visual_context=""):
    """Locate both sides of a proposed mismatch; uncertain exclusions retain data.

    Quotation checks establish provenance, not semantic correctness. No extra
    model calls or delivery gates are introduced.
    """
    decisions = {}
    normalize = lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
    for item in exclusions or []:
        item = item.model_dump() if isinstance(item, BaseModel) else item
        if not isinstance(item, dict):
            continue
        idx, reason = item.get("evidence_id"), str(item.get("reason") or "").strip()
        if type(idx) is not int or not 1 <= idx <= len(evidence) or not reason:
            continue
        record, basis = evidence[idx - 1], []
        for ref in item.get("basis") if isinstance(item.get("basis"), list) else []:
            if not isinstance(ref, dict):
                continue
            source, quote = ref.get("source"), normalize(ref.get("quote"))
            if source == "request":
                content = user_query
            elif source in {"visual_context", "attachment"}:
                content = visual_context
            elif source == "evidence" and type(ref.get("evidence_id")) is int and ref["evidence_id"] == idx:
                content = "\n".join(str(record.get(k) or "") for k in ("doc_name", "title", "text", "summary"))
            else:
                continue
            if quote and quote in normalize(content):
                basis.append(dict(ref))
        located = (any(ref["source"] in {"request", "visual_context", "attachment"} for ref in basis)
                   and any(ref["source"] == "evidence" for ref in basis))
        decisions[idx] = {"reason": reason, "basis": basis,
                          "basis_status": "located" if located else "unverified"}
    result = []
    for idx, item in enumerate(evidence, 1):
        record = dict(item)
        if idx in decisions:
            record["relevance_review"] = decisions[idx]
            if decisions[idx]["basis_status"] == "located":
                record["excluded_as_irrelevant"] = decisions[idx]["reason"]
        result.append(record)
    return result


def filter_irrelevant_candidates(candidates, evidence):
    rejected_titles = {str(e.get("doc_name") or e.get("title") or "").strip().casefold()
                       for e in evidence if e.get("excluded_as_irrelevant")}
    rejected_titles.discard("")
    # An irrelevant page does not disqualify the other useful pages of a document.
    rejected_titles -= {str(e.get("doc_name") or e.get("title") or "").strip().casefold()
                        for e in evidence if not e.get("excluded_as_irrelevant")}
    return [c for c in candidates
            if str(c.get("title") or "").strip().casefold() not in rejected_titles]


def is_executable_retrieval_query(query: str) -> bool:
    """Reject unresolved templates and clarification requests before source calls."""
    query = str(query or "").strip()
    if not query:
        return False
    if re.search(r"[\[【<{][^\]】>}]*(?:confirmed|待确认|确认后|待填|placeholder|占位|full\s+title|paper\s+title|论文标题|完整标题)[^\]】>}]*[\]】>}]", query, re.I):
        return False
    return not re.search(
        r"^(?:(?:请|麻烦)(?:(?:用户|您|你)\s*)?(?:提供|补充|告知)|please\s+(?:provide|supply|clarify)|ask\s+(?:the\s+)?user\b)",
        query, re.I,
    )


def build_retrieval_probes(user_query: str, intent: Dict[str, Any]) -> List[Dict[str, str]]:
    """Keep raw and parsed query views as task-linked probes; none replaces the user's wording."""
    variants: List[Dict[str, str]] = []
    for segment in intent.get("task_segments") or []:
        if isinstance(segment, dict):
            q = str(segment.get("retrieval_query") or "").strip()
            if q:
                variants.append({"query": q, "english_query": ""})

    for subquery in intent.get("sub_queries") or []:
        if isinstance(subquery, dict):
            q = str(subquery.get("query") or "").strip()
            if q:
                variants.append({"query": q, "english_query": str(subquery.get("english_query") or "").strip()})

    summary_query = str(intent.get("search_query") or "").strip()
    if summary_query:
        variants.append({"query": summary_query, "english_query": str(intent.get("online_search_query") or "").strip()})
    raw_query = str(user_query or "").strip()
    if raw_query:
        variants.append({"query": raw_query, "english_query": ""})

    deduped: List[Dict[str, str]] = []
    seen: Dict[str, Dict[str, str]] = {}
    for item in variants:
        q = item["query"]
        key = re.sub(r"\s+", " ", q).casefold()
        if key in seen:
            if item["english_query"] and not seen[key]["english_query"]:
                seen[key]["english_query"] = item["english_query"]
        else:
            seen[key] = item
            deduped.append(item)
    return deduped


def build_retrieval_queries(user_query: str, intent: Dict[str, Any]) -> List[str]:
    return [probe["query"] for probe in build_retrieval_probes(user_query, intent)]


def retrieval_query_key(query: str) -> str:
    if not isinstance(query, str):
        return ""
    return " ".join(sorted(re.findall(r"[\w]+", query.casefold())))


def select_retrieval_probes(user_query: str, intent: Dict[str, Any], plan: RetrievalPlan) -> List[Dict[str, str]]:
    """Execute a small plan, not every alternative wording produced by intent parsing."""
    probes = [item.model_dump() for item in plan.queries if item.query.strip()]
    if not probes and not plan.queries:
        # Legacy plans use one task summary; genuine cross-domain tasks may use
        # their distinct subqueries. Raw wording remains in the full intent context.
        subqueries = intent.get("sub_queries") or []
        if intent.get("is_cross_disciplinary") and subqueries:
            probes = [{"query": str(item.get("query") or ""),
                       "english_query": str(item.get("english_query") or "")}
                      for item in subqueries if isinstance(item, dict) and not item.get("is_umbrella_generic")]
        if not probes:
            probes = [{"query": str(intent.get("search_query") or user_query),
                       "english_query": str(intent.get("online_search_query") or "")}]
    selected = []
    seen = set()
    for probe in probes:
        key = retrieval_query_key(probe["query"])
        if (key and key not in seen and is_executable_retrieval_query(probe["query"])
                and (not probe["english_query"] or is_executable_retrieval_query(probe["english_query"]))):
            seen.add(key)
            selected.append({k: probe[k].strip() for k in ("query", "english_query")})
    return selected


class RetrievalPlannerAgent:
    """围绕当前任务独立规划检索需求，不依赖意图分类器的固定渠道标签。"""

    SYSTEM_PROMPT = """请围绕【主体是什么 -> 任务是什么 -> 要做什么】独立规划本轮任务的资料检索方案，不受分类器标签限制，所有说明字段（reasoning、issues、summary、note 等）统一使用中文。
1. 忠实原话与共享任务：优先依据用户原话和 shared_task；若存在 evidence_gap（补证缺口），仅针对该具体缺口选择检索渠道与可复用记录，绝不替换原任务或重新发起宽泛普查。历史记录仅用于辅助理解上下文，不得覆盖当前请求；已被替代的旧记录（superseded_by）仅在用户明确要求历史对比时选用。
2. 原话对齐核验（极重要）：当 verify_task_against_original=true 时，检查 proposed_intent 是否符合用户原话的核心意图：
   - 【默认判为 matched】：只要 proposed_intent 的学科方向、核心主体与用户原话一致（即使 proposed_intent 对宽泛课题合理拆解了细分子方向或补充了常规检索词），必须返回 task_alignment: {"status": "matched", "issues": [], "corrected_fields": {}}！切勿吹毛求疵将正常的子方向展开判为偏差。
   - 【仅在严重跑偏时判为 corrected】：仅当 proposed_intent 完全认错学科主体、捏造了用户未要求的编程/证明前置条件、或遗漏了原话明确排除项时，才返回 status="corrected"。此时 corrected_fields 的 JSON 键名必须严格使用 proposed_intent 原有的英文属性名（仅限 domain, intent, subject_entity, target_topic, sub_topic, search_query, online_search_query, negative_constraints, task_segments，严禁自造“主体”“任务动作”等中文键名）。
3. 渠道能力按需选择：local_search（检索本地文档/知识库）、online_search（外部学术检索）、include_github（代码仓库与工程实现）、include_web（机构官网/技术规范/通用网页）、include_datasets（数据集与基准）。通过 selected_shared_item_ids 独立勾选可直接复用的共享池记录；当本地与在线检索互补时可开启 parallel_search。
4. 专业任务全面检索原则：凡属学术研究、领域探讨、理论推导、方法剖析、方案对比或实证分析等专业任务，一律设 requires_evidence=true、is_resolved=true。只要用户未明确限制“仅使用本地文档”且共享池现有证据尚未完全覆盖当前问题，均应开启 online_search=true 检索外部学术文献（不得仅限含“最新/近期”字眼才检索；若 local_material_available=true，可同时开启 local_search=true 与 parallel_search=true 实现本地与在线互补检索，切勿因 local_material_available=true 而关闭 online_search）。仅对纯日常寒暄、纯文本改写或完全基于用户给定前提的推演，设 requires_evidence=false。
5. 优先复用与遵循约束：优先勾选可服务于当前任务的共享池证据（selected_shared_item_ids），当共享池证据不足以完整支撑本轮验证命题与实证分析时，主动通过在线与本地渠道检索补充。严格遵守用户指定的来源、文档、时间或排除限制；若已查看证据已充分覆盖任务，不再重复检索。
6. 精简检索词（极重要）：合并同义子任务视角，仅生成 1～2 条精炼互补的可执行查询（绝不把同一主题拆成 5～6 条冗余查询）。query 使用中文核心学术词；english_query 仅供底层国际论文 API（arXiv/OpenAlex）检索使用，必须是 2～5 个词的精炼英文学术短语（严禁包含 site: 语法、布尔 OR 长链、自然语言时间状语或超过 6 个词的英文长句）。绝不使用未解析的标题/DOI 占位符。
7. 严格返回 JSON：requires_evidence, local_search, online_search, include_github, include_web, include_datasets, parallel_search, selected_shared_item_ids, queries [{query,english_query}], reasoning（中文）, is_resolved，以及按需返回 task_alignment。
""" + TASK_COVERAGE_GUIDANCE + "\n" + EVIDENCE_GUIDANCE

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("Missing model API configuration for retrieval planning")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = get_default_llm_model()

    def plan(
        self,
        user_query: str,
        intent: Dict[str, Any],
        state_context: Dict[str, Any] | None = None,
    ) -> RetrievalPlan:
        # Deliberately exclude required_data_sources, need_online_paper_search,
        # target_agent and intent category fields from this independent decision.
        now = time.time()
        shared_resources = []
        shared_pool = (state_context or {}).get("shared_pool") or []
        recent_candidates = (state_context or {}).get("recent_paper_candidates") or []
        pool_ids = {item.get("item_id") for item in shared_pool if isinstance(item, dict)}
        all_resources = list(shared_pool)
        all_resources.extend(
            candidate for candidate in recent_candidates
            if isinstance(candidate, dict) and candidate.get("shared_item_id") not in pool_ids
        )
        if not all_resources:
            all_resources = (state_context or {}).get("shared_resources") or []
        for item in all_resources:
            if not isinstance(item, dict):
                continue
            fetched_at = item.get("last_fetched_at") or item.get("updated_at")
            try:
                age_hours = max(0.0, (now - float(fetched_at)) / 3600.0) if fetched_at else None
            except (TypeError, ValueError):
                age_hours = None
            shared_resources.append({
                "item_id": item.get("item_id") or item.get("shared_item_id") or item.get("resource_id"),
                "kind": item.get("kind", "evidence"),
                "title": item.get("title") or item.get("doc_name"),
                "task": item.get("task"),
                "source": item.get("source") or item.get("doc_type"),
                "year": item.get("year"),
                "age_hours_since_fetch": round(age_hours, 1) if age_hours is not None else None,
                "evidence_level": item.get("evidence_level"),
                "superseded_by": item.get("superseded_by"),
                "entities": item.get("entities", []),
            })
        task_context = {
            "shared_task": (state_context or {}).get("shared_task"),
            "evidence_gap": (state_context or {}).get("evidence_gap"),
            "current_utc": datetime.now(timezone.utc).isoformat(),
            "user_query": user_query,
            "task_description": intent.get("intent", ""),
            "task_segments": intent.get("task_segments", []),
            "query_views": build_retrieval_probes(user_query, intent),
            "constraints": intent.get("negative_constraints", []),
            "time_requirement": resolve_retrieval_time_constraint(intent.get("time_constraint")),
            "time_requirement_note": intent.get("time_constraint_note"),
            "active_subject": (state_context or {}).get("active_subject"),
            "active_doc": (state_context or {}).get("active_doc"),
            "local_material_available": bool((state_context or {}).get("local_material_available", False)),
            "available_shared_resources": shared_resources,
            "search_history": (state_context or {}).get("search_history") or [],
            "retrieved_evidence": (state_context or {}).get("retrieved_evidence") or [],
        }
        verify_task = bool((state_context or {}).get("verify_task_against_original"))
        if verify_task:
            task_context.update(
                verify_task_against_original=True,
                proposed_intent={key: value for key, value in intent.items()
                                 if key not in {"required_data_sources", "need_online_paper_search", "target_agent"}},
                previous_subject=(state_context or {}).get("previous_subject"),
                history_context=(state_context or {}).get("history_context") or "",
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(task_context, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4096,
            timeout=min(get_llm_timeout(60), 60),
        )
        content = (response.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
        data = json.loads(content)
        plan = RetrievalPlan(**data)
        if verify_task:
            alignment = plan.task_alignment
            if alignment is not None and isinstance(alignment.corrected_fields, dict) and alignment.corrected_fields:
                zh_rubric_keys = {"主体", "任务", "任务动作", "要做什么", "研究对象", "期望产出", "限定条件", "要验证什么", "实证分析", "得到什么结论"}
                if alignment.status in {"corrected", "matched"} and set(alignment.corrected_fields).issubset(zh_rubric_keys):
                    # 模型将中文核对维度名称当成了 corrected_fields 键名，说明并非实质 schema 字段冲突，自动归一为 matched
                    alignment = TaskAlignment(status="matched", issues=[], corrected_fields={})
                    plan.task_alignment = alignment
                    plan.is_resolved = True
            if alignment is None or alignment.status == "unresolved" or (
                    alignment.status == "corrected" and not alignment.corrected_fields) or (
                    alignment.status == "matched" and alignment.corrected_fields):
                plan.task_alignment = TaskAlignment(status="unresolved", issues=(alignment.issues if alignment else [])
                                                   or ["未返回有效的原话对照结果"])
                plan.is_resolved = False
                plan.reasoning = "主体与任务对照原始内容尚未完成，未执行检索。"
        available_record_ids = {
            str(item.get("item_id") or item.get("shared_item_id") or item.get("resource_id"))
            for item in all_resources if isinstance(item, dict)
            and (item.get("item_id") or item.get("shared_item_id") or item.get("resource_id"))
        }
        plan.selected_shared_item_ids = list(dict.fromkeys(
            item_id for item_id in plan.selected_shared_item_ids if item_id in available_record_ids))
        available_evidence_ids = {
            str(item.get("item_id") or item.get("shared_item_id") or item.get("resource_id"))
            for item in all_resources
            if isinstance(item, dict) and item.get("kind", "evidence") in {"evidence", "candidate"}
            and (item.get("item_id") or item.get("shared_item_id") or item.get("resource_id"))
        }
        if (
            plan.is_resolved and plan.requires_evidence
            and not (plan.local_search or plan.online_search)
            and not available_evidence_ids.intersection(plan.selected_shared_item_ids)
        ):
            # Do not accept model confidence or a previous answer as a source.
            if task_context["local_material_available"]:
                plan.local_search = True
            else:
                plan.is_resolved = False
                plan.reasoning = "专业任务尚无可追溯资料依据；未选择可用证据或允许的检索渠道。"
        if not plan.is_resolved:
            plan.local_search = False
            plan.online_search = False
            plan.include_github = False
            plan.include_web = False
            plan.include_datasets = False
            plan.parallel_search = False
            plan.selected_shared_item_ids = []
            plan.queries = []
        if not plan.online_search:
            plan.include_github = False
            plan.include_web = False
            plan.include_datasets = False
        if not (plan.local_search and plan.online_search):
            plan.parallel_search = False
        return plan

    def assess_coverage(self, user_query: str, intent: Dict[str, Any], evidence: List[Dict[str, Any]],
                        search_history: List[Dict[str, Any]] | None = None) -> RetrievalCoverage:
        """Assess actual returned evidence and propose only concrete missing pieces."""
        request_text = (intent.get("shared_task") or {}).get("original_user_query", user_query)
        payload = {
            "shared_task": intent.get("shared_task"),
            "user_query": request_text,
            "task_segments": intent.get("task_segments") or [],
            "required_tasks": [{"task_id": idx, "task": task}
                               for idx, task in enumerate(intent.get("task_segments") or [], 1)],
            "constraints": intent.get("negative_constraints") or [],
            "time_requirement": intent.get("time_constraint"),
            "time_requirement_note": intent.get("time_constraint_note"),
            "search_history": search_history or [],
            "evidence": [
                {"evidence_id": idx, **{k: item.get(k) for k in ("doc_name", "doc_type", "location_label", "text", "summary", "url", "evidence_level", "provenance")}}
                for idx, item in enumerate(evidence, 1) if isinstance(item, dict)
            ],
            "instructions": (
                "请对照已检索资料，逐项评估用户的每项明确任务要求（Evaluate EVERY explicit task requirement），包括任务范围、来源限制与时间要求；所有说明字段（summary、note、rationale、strategy、reason）统一使用中文。"
                "任务解析仅供理解原话参考，不得额外增加用户未提出的验收条件。"
                "返回与 required_tasks 通过 task_ids 关联的 task_checks，保持 requirement 表述稳定，标注 status (supported|partial|missing)、evidence_ids、简明已核实的 supported_facts 及具体中文说明 note。"
                "【核心准则】：本学科公认的基础理论、数学公式推导与标准算法机理不需要外部论文逐句背书；只要相应任务已有相关论文摘要或本地教材片段支撑，即应判定为 status='supported'，并返回 sufficient=true、gaps=[]。"
                "先检查 search_history；仅当用户明确要求的子任务完全缺乏相关资料，且存在实质不同的检索策略（materially different strategy）时，才提出补充检索。"
                "绝不要仅仅为了给已有摘要索取全文、或为了穷尽整个学科综述而追加 gap 查询。"
                "对于 irrelevant_evidence（明确无关资料），必须给出具体的主题不匹配中文原因，同时在 basis 中分别引用 request/visual_context 原句与目标资料（标题或正文）原句；保留相关背景、基础理论与局部支撑资料，相关性不确定或缺少全文不作为排除理由。"
                "返回 JSON：sufficient, task_checks [{requirement,task_ids,status,evidence_ids,supported_facts,note}], "
                "irrelevant_evidence [{evidence_id,reason,basis:[{source,quote,evidence_id}]}], gaps [{query,rationale,sources,requirement,strategy}], unresolved, summary。"
                "sources 可选用 local,academic,web,github,datasets 或留空。"
                + TASK_COVERAGE_GUIDANCE + "\n" + EVIDENCE_GUIDANCE
            ),
        }
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是学术检索证据覆盖度评估专家。请严格基于已检出资料与学科理论边界进行评估，统一使用中文说明，绝不捏造事实。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=4096,
            timeout=min(get_llm_timeout(60), 60),
        )
        content = (response.choices[0].message.content or "{}").strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()
        coverage = RetrievalCoverage(**json.loads(content))
        inspected_evidence = apply_evidence_relevance(evidence, coverage.irrelevant_evidence,
            request_text, (intent.get("shared_task") or {}).get("visual_context", ""))
        available_ids = {idx for idx, item in enumerate(evidence, 1) if isinstance(item, dict)
                         and (item.get("text") or item.get("summary"))
                         and not inspected_evidence[idx - 1].get("excluded_as_irrelevant")}
        for check in coverage.task_checks:
            check.evidence_ids = [idx for idx in check.evidence_ids if idx in available_ids]
            if not check.evidence_ids:
                check.supported_facts = []
            if check.status != "missing" and not check.evidence_ids:
                check.status = "missing"
                check.note = check.note or "未定位到实际资料支持"
        checked_tasks = {task_id for check in coverage.task_checks for task_id in check.task_ids}
        missing_tasks = {task["task_id"] for task in payload["required_tasks"]} - checked_tasks
        coverage.unresolved.extend(f"任务 {task_id} 尚未完成证据覆盖核验" for task_id in sorted(missing_tasks))
        for task in payload["required_tasks"]:
            if task["task_id"] not in missing_tasks:
                continue
            detail = task["task"]
            query = str(detail.get("retrieval_query") or detail.get("query") or detail.get("task")
                        or detail.get("object") or "") if isinstance(detail, dict) else str(detail)
            if query.strip() and not any(retrieval_query_key(gap.query) == retrieval_query_key(query) for gap in coverage.gaps):
                coverage.gaps.append(RetrievalGap(
                    query=query, requirement=f"任务 {task['task_id']}",
                    rationale="用户明确要求的子任务尚未核验资料覆盖",
                    strategy="检索遗漏子任务的直接依据",
                ))
        if (not coverage.task_checks or coverage.gaps or coverage.unresolved
                or any(check.status != "supported" for check in coverage.task_checks)):
            coverage.sufficient = False
        return coverage
