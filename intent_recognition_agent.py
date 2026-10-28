"""Domain-Agnostic Intent Recognition Agent.

Supports open-ended learning and research across arbitrary domains and meetings.
Integrates Fast-Path rules + LLM structured JSON schema reasoning via .env API.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from config_utils import get_default_llm_model, get_llm_timeout
from prompts import build_intent_system_prompt

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)


# ==========================================
# 1. 结构化意图输出定义 (Pydantic Schema)
# ==========================================
class IntentDecision(BaseModel):
    domain: str = Field(
        description=(
            "当前问题的主处理研究领域或专业方向。允许使用任意自然语言领域名称，"
            "不得受固定枚举限制，由模型根据问题实际内容自由提炼与泛化。"
        )
    )
    intent: str = Field(
        description=(
            "跨学科通用任务类型标识，从支持的意图定义全集中选取。"
            "应按用户任务选择，不按具体学科创建专用意图。"
        )
    )
    confidence: float = Field(default=0.9, description="意图识别置信度 (0.0 ~ 1.0)")
    target_agent: str = Field(description="推荐路由的下游智能体名称")
    required_data_sources: List[str] = Field(
        default_factory=list,
        description="该意图所需的数据源列表，如 ['uploaded_pdf_docs', 'general_science_knowledge', 'arxiv_live_search']"
    )
    search_query: Optional[str] = Field(
        default="",
        description="纯正向学术检索词（从当前提问提取，去除否定词、排除词及无关口头语）"
    )
    target_topic: Optional[str] = Field(
        default="",
        description="识别出的学术主题（若继承则同上轮，若重置则为新主题）"
    )
    subject_action: str = Field(
        default="INHERIT",
        description="LangGraph 记忆主体生命周期动作: INIT (初次确立) | INHERIT (隐式继承) | APPEND (追加对比) | REPLACE (显式重置)"
    )
    subject_entity: Optional[str] = Field(
        default="",
        description="本次提取或变更的核心学术概念或研究对象实体 (INHERIT 时可为空)"
    )
    framework_reference: Optional[Dict[str, Any]] = Field(
        default=None,
        description="若用户提问仅要求参考某已有文档的分析框架/结构，填入 {'doc_name': '...', 'usage': 'structure_only'}；否则为 None"
    )
    negative_constraints: List[str] = Field(
        default_factory=list,
        description="用户明确指定排除/舍弃的负向主题或禁区关键词列表"
    )
    need_online_paper_search: bool = Field(
        default=False,
        description="是否需要调用在线 arXiv 检索前沿论文（如用户要求最新学术成果查重、前沿去重或调研新工作）"
    )
    online_search_query: Optional[str] = Field(
        default="",
        description="用于 arXiv 联网搜索的高精度英文学术关键词（跨学科通用）"
    )
    topic_action: str = Field(
        default="STAY_LOCKED",
        description="主题生命周期动作: INIT_LOCK (首轮确立并锁定) | STAY_LOCKED (多轮默认持续锁定当前主题) | EVOLVE (在原主题下深入子分支) | RESET (用户明确指示切换/重置为新主题)"
    )
    sub_topic: Optional[str] = Field(
        default="",
        description="若处于 EVOLVE 状态，记录用户深化的具体子分支技术点"
    )
    is_cross_disciplinary: bool = Field(
        default=False,
        description="是否属于多领域/多学科交叉课题。"
    )
    disciplines: List[str] = Field(
        default_factory=lambda: ["general"],
        description=(
            "涉及的一个或多个学科、技术方向或交叉领域。允许模型自由生成，"
            "不得局限于预设学科列表；单学科问题也应保留一个明确领域标签。"
        )
    )
    sub_queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "若为跨领域交叉课题，拆解出的各领域子检索/推演问题字典列表。"
            "格式: [{'discipline': str, 'query': str, 'is_umbrella_generic': bool, 'is_core_bottleneck': bool, 'retrieval_strategy': 'online_probe'|'local_only'|'llm_prior'}]。"
            "严禁在代码中写死任何学科词，全部由大模型自主推断学科层级、任务重合度与核心理论瓶颈。"
        )
    )
    time_constraint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="若用户提问涉及时间、年份或时效要求，结合当前真实日期动态计算的时间约束，格式: {'has_time_constraint': True, 'start_year': int, 'end_year': int, 'sort_by_date': bool}；若无时间要求则必须为 null。"
    )
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="抽取的关键学术概念或实体")
    requires_long_form: bool = Field(
        default=False,
        description=(
            "根据用户完整描述与任务本质，智能判定是否属于需要突破日常研讨篇幅的超长篇完整成果撰写任务"
            "（如起草整篇长篇学术论文/期刊正文、万字专著书籍章节、完整基金/重大课题申报书正文、完整技术白皮书、全书纲要等）。"
            "若属于日常学术探究、概念辨析、单点或分步数理推导、文献/书籍/课件/技术文档的局部考据与对比、深入机理剖析（即使包含'详细解释'、'深入剖析'、'展开讲讲'），一律为 False。"
        ),
    )
    long_form_type: Optional[str] = Field(
        default=None,
        description="若 requires_long_form 为 True，标明长篇成果的体裁；否则为 None。"
    )
    need_download_fulltext: bool = Field(
        default=False,
        description="若用户明确指示下载/获取指定在线候选论文的完整全文并建库精读，设为 True；否则为 False"
    )
    is_candidate_selection: bool = Field(
        default=False,
        description="若用户的提问是针对近期已就绪文献候选池进行横向对比、多篇筛选、总结或全量问答，设为 True；否则为 False"
    )
    reasoning: Optional[str] = Field(default="", description="做出该意图判断的简明逻辑依据")


# ==========================================
# 2. 意图知识库元数据定义 (Domain-Agnostic)
# ==========================================
INTENT_DEFINITIONS = [
    {
        "intent": "concept_simplify",
        "name": "通俗概念原理解释",
        "description": "请求用通俗语言、直观比喻或大白话解释某个学术概念、物理现象、语法规则或模型机制",
        "agent": "ConceptTutorAgent",
        "sources": ["general_science_knowledge", "uploaded_pdf_docs"],
    },
    {
        "intent": "stepwise_derivation",
        "name": "公式与算法分步推导",
        "description": "请求对数学公式、物理定律方程、算法每一步的演化推导或代码流程进行逐步展开解析",
        "agent": "DerivationAgent",
        "sources": ["uploaded_pdf_docs", "general_science_knowledge"],
    },
    {
        "intent": "cross_modal_locate",
        "name": "音视频与课件教材对照",
        "description": "询问视频/讲课当前所讲内容对应在课件、PPT或教材文档的哪一页/哪个位置",
        "agent": "CrossModalRAGAgent",
        "sources": ["live_audio_records", "uploaded_pdf_docs"],
    },
    {
        "intent": "doc_evidence_query",
        "name": "学术论文/专著书籍/教材讲义/技术文档全景解读与细节考据",
        "description": "在上传的 PDF 论文、学术专著书籍、教材讲义、技术报告、课件PPT，或当前会话已检索就绪的文献资产池（recent_paper_candidates）中查找证据、筛选分类、提取章节段落，或者请求对某篇论文/专著书籍/课件进行全文概述、背景介绍、方法拆解、方案构思或核心贡献总结",
        "agent": "DocumentSearchAgent",
        "sources": ["uploaded_pdf_docs", "recent_paper_candidates"],
    },
    {
        "intent": "content_summarize",
        "name": "实时语音会议/课堂流水要点提炼",
        "description": "专门针对刚才开启的实时麦克风/声卡语音录音流水（即会议记录、课堂口头讲解）进行要点总结与纪要生成。注意：如果是针对上传的论文/PDF/文档做概述，绝对不能选这项，必须选 doc_evidence_query！",
        "agent": "SummarizerAgent",
        "sources": ["live_audio_records"],
    },
    {
        "intent": "action_homework_extract",
        "name": "作业与待办事项提取",
        "description": "专门提取老师布置的课后作业、思考题、或会议达成的待办事项与截止时间",
        "agent": "SummarizerAgent",
        "sources": ["live_audio_records"],
    },
    {
        "intent": "compare_contrast",
        "name": "多方案/概念对比分析",
        "description": "请求对比两个或多个理论、方法、模型、技术路线或语言现象的区别与优缺点",
        "agent": "ComparisonAgent",
        "sources": ["general_science_knowledge", "uploaded_pdf_docs"],
    },
    {
        "intent": "quiz_generate",
        "name": "考题与随堂测验生成",
        "description": "请求出题考查用户对指定知识点、理论、方法、技术或语言内容的掌握情况",
        "agent": "QuizAgent",
        "sources": ["general_science_knowledge", "uploaded_pdf_docs"],
    },
    {
        "intent": "study_path_plan",
        "name": "学科与技能学习路径规划",
        "description": "请求针对任意学科、技术方向或跨学科主题制定系统化阶段学习计划或前置知识图谱",
        "agent": "StudyPlannerAgent",
        "sources": ["user_study_profile", "general_science_knowledge"],
    },
    {
        "intent": "research_design",
        "name": "研究方案与验证路线设计",
        "description": (
            "请求设计、构建或规划研究方案、理论框架、实验流程、算法路线、"
            "跨学科技术路线或验证闭环。重点是明确研究问题、方法、变量、"
            "数据/材料、评价指标、对照实验和可证伪性；不是单纯解释概念、"
            "总结文献或安排学习先后顺序。适用于任意学科。"
        ),
        "agent": "StudyPlannerAgent",
        "sources": ["general_science_knowledge", "uploaded_pdf_docs"],
    },
    {
        "intent": "voice_system_control",
        "name": "系统语音口令与功能控制",
        "description": "用户的控制指令，如开启/关闭录音、暂停、切换模型音色、清空会话等系统级操作",
        "agent": "SystemControlAgent",
        "sources": [],
    },
    {
        "intent": "arxiv_paper_search",
        "name": "ArXiv 线上最新学术文献检索与前沿综述",
        "description": "请求在线实时检索 arXiv、SOTA 前沿论文库中最新的研究论文、论文作者、发表年份及 Abstract 摘要并整理学术综述",
        "agent": "OnlinePaperSearchAgent",
        "sources": ["arxiv_live_search"],
    },
    {
        "intent": "general_chitchat",
        "name": "通用日常与学术问候",
        "description": "打招呼、日常寒暄、感谢、身份询问或与学术无关的闲聊",
        "agent": "GeneralChatAgent",
        "sources": [],
    },
]

INTENT_TO_AGENT = {d["intent"]: d["agent"] for d in INTENT_DEFINITIONS}
INTENT_DEFAULT_SOURCES = {d["intent"]: list(d.get("sources", [])) for d in INTENT_DEFINITIONS}


# ==========================================
# 3. 意图识别 Agent 实现
# ==========================================
class IntentRecognitionAgent:
    def __init__(self):
        self.model = get_default_llm_model()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")

        if not self.api_key or not self.base_url:
            raise ValueError(f"[错误] 未在 {ENV_PATH} 中正确找到 OPENAI_API_KEY 或 OPENAI_BASE_URL 配置！")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_system_prompt(self, current_date: Optional[str] = None) -> str:
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")

        intents_desc = "\n".join(
            [f"- `{item['intent']}`: {item['name']} - {item['description']} (推荐智能体: {item['agent']}, 关联数据源: {item.get('sources', [])})" for item in INTENT_DEFINITIONS]
        )

        return build_intent_system_prompt(intents_desc=intents_desc, current_date=current_date)

    def parse_intent(
        self,
        state: Optional[Union[Dict[str, Any], str]] = None,
        query: Optional[str] = None,
        history_summary: Optional[str] = None,
        topic_state: Optional[Dict[str, Any]] = None,
        recent_candidates: Optional[List[Dict[str, Any]]] = None,
        active_doc: Optional[str] = None,
    ) -> IntentDecision:
        """基于 LangGraph 运行时全息状态 (MasterAgentState) 动态感知研讨现场并做高维意图决策"""
        if isinstance(state, str):
            query = state
            state = None

        if state is not None and isinstance(state, dict):
            query = state.get("user_query") or query or ""
            history_summary = state.get("history_summary") if state.get("history_summary") is not None else history_summary
            topic_state = state.get("topic_state") if state.get("topic_state") is not None else topic_state
            recent_candidates = state.get("recent_paper_candidates") if state.get("recent_paper_candidates") is not None else recent_candidates
            active_doc = state.get("active_doc") if state.get("active_doc") is not None else active_doc
            active_subject = state.get("active_subject") or (topic_state or {}).get("current_topic")
            sub_subject = state.get("sub_subject") or (topic_state or {}).get("sub_topic")
        else:
            query = query or ""
            active_subject = (topic_state or {}).get("current_topic")
            sub_subject = (topic_state or {}).get("sub_topic")

        query_clean = query.strip()
        if not query_clean:
            return IntentDecision(
                domain="general",
                intent="general_chitchat",
                confidence=1.0,
                target_agent="GeneralChatAgent",
                required_data_sources=[],
                search_query="",
                extracted_entities={},
                reasoning="空输入默认问候",
            )

        # 直接从 LangGraph 状态字段中投射会话研讨现场态势
        context_parts = []
        if active_subject or ((topic_state or {}).get("current_topic")):
            curr_t = active_subject or (topic_state or {}).get("current_topic")
            sub_t = sub_subject or (topic_state or {}).get("sub_topic")
            status = (topic_state or {}).get("status", "LOCKED")
            context_parts.append(
                f"【LangGraph 会话研讨现场状态】：\n- 核心研究主体 (active_subject): {curr_t}\n- 当前子分支 (sub_subject): {sub_t or '无'}\n- 状态: {status}（默认无偏见继承此主题，除非明确要求切换/重置）"
            )

        if active_doc:
            context_parts.append(f"【当前聚焦精读文献/课件 (active_doc)】：{active_doc}")

        if recent_candidates:
            cand_lines = []
            for i, c in enumerate(recent_candidates[:8], 1):
                src = c.get("source", "online")
                src_label = "arXiv预印本" if src == "arxiv" else ("正式发表期刊" if src in ["openalex", "crossref"] else src)
                cand_lines.append(f"{i}. [{src_label}] 《{c.get('title')}》 ({c.get('year', 'Recent')}, 出处/DOI: {c.get('paper_id', '')})")
            context_parts.append(f"【会话当前已就绪文献资产池 (recent_paper_candidates, 共 {len(recent_candidates)} 篇)】：\n" + "\n".join(cand_lines))

        if isinstance(state, dict) and state.get("matched_topic"):
            m_top = state["matched_topic"]
            context_parts.append(
                f"【外科手术式锁定的前序主题档案 (matched_topic)】：\n- 主题: 《{m_top.get('topic_name')}》 | 动作类型: {m_top.get('action_type')}\n- 核心理论定论与基线依据:\n{m_top.get('key_findings')}"
            )

        if history_summary and history_summary.strip():
            context_parts.append(f"【前期研讨结构化全息记忆 (history_summary)】：\n{history_summary.strip()}")

        context_str = "\n\n".join(context_parts)
        if context_str:
            user_content = f"{context_str}\n\n【用户当前提问】：{query_clean}"
        else:
            user_content = f"【用户当前提问】：{query_clean}"

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        max_retries = 3
        fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "").strip()
        last_err = None
        for attempt in range(1, max_retries + 1):
            active_model = (fallback_model if (attempt >= 2 and fallback_model and fallback_model != self.model) else self.model)
            try:
                resp = self.client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {"role": "system", "content": self._build_system_prompt(current_date=current_date_str)},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=2048,
                    timeout=get_llm_timeout(100.0),
                )
                raw_json = (resp.choices[0].message.content or "").strip()
                if raw_json.startswith("```"):
                    raw_json = re.sub(r"^```[a-zA-Z]*\n?", "", raw_json)
                    raw_json = re.sub(r"\n?```$", "", raw_json).strip()
                parsed_dict = json.loads(raw_json)
                if not parsed_dict.get("search_query"):
                    parsed_dict["search_query"] = query_clean

                # Keep the domain vocabulary open while preserving the legacy
                # singular `domain` field used by the router and RAG fallback.
                raw_disciplines = parsed_dict.get("disciplines") or []
                if isinstance(raw_disciplines, str):
                    raw_disciplines = [raw_disciplines]
                parsed_dict["disciplines"] = [
                    str(item).strip()
                    for item in raw_disciplines
                    if str(item).strip()
                ] or ["general"]

                # If the model uses `general` only as a placeholder, expose the
                # first real discipline through the backward-compatible domain
                # field. No discipline names are hardcoded here.
                domain = str(parsed_dict.get("domain") or "").strip()
                if domain in {"", "general", "综合科学"} and parsed_dict["disciplines"] != ["general"]:
                    parsed_dict["domain"] = parsed_dict["disciplines"][0]

                intent_key = parsed_dict.get("intent", "")
                expected_agent = INTENT_TO_AGENT.get(intent_key)
                if expected_agent:
                    parsed_dict["target_agent"] = expected_agent
                elif not parsed_dict.get("target_agent"):
                    parsed_dict["target_agent"] = "GeneralTutorAgent"

                # 数据源对齐：结合意图元数据与模型判定，确保学术意图天然具备的文献考据数据源不被遗漏
                intent_key = parsed_dict.get("intent", "")
                default_sources = INTENT_DEFAULT_SOURCES.get(intent_key, [])
                raw_sources = parsed_dict.get("required_data_sources") or []
                if not raw_sources:
                    raw_sources = list(default_sources)
                elif "uploaded_pdf_docs" in default_sources and "uploaded_pdf_docs" not in raw_sources and "attached_image_content" not in raw_sources:
                    raw_sources.append("uploaded_pdf_docs")
                parsed_dict["required_data_sources"] = raw_sources

                # 规范化长篇大论字段类型
                if "requires_long_form" in parsed_dict:
                    if isinstance(parsed_dict["requires_long_form"], str):
                        parsed_dict["requires_long_form"] = parsed_dict["requires_long_form"].lower() in ["true", "1", "yes"]
                    else:
                        parsed_dict["requires_long_form"] = bool(parsed_dict["requires_long_form"])
                else:
                    parsed_dict["requires_long_form"] = False

                return IntentDecision(**parsed_dict)

            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    print(f"⚠️ [IntentRouter Note] 意图解析请求异常 ({e})，正在重试 ({attempt}/{max_retries})...")
                    time.sleep(2.0)

        # 智能学术降级兜底：无死规则硬编码，依据前序上下文主体状态自洽继承
        if active_subject:
            fallback_domain = active_subject
            fallback_intent = "concept_simplify"
            fallback_agent = "GeneralTutorAgent"
            subject_action = "INHERIT"
        elif len(query_clean) <= 6:
            fallback_domain = "general"
            fallback_intent = "general_chitchat"
            fallback_agent = "GeneralChatAgent"
            subject_action = "INIT"
        else:
            fallback_domain = "general"
            fallback_intent = "concept_simplify"
            fallback_agent = "GeneralTutorAgent"
            subject_action = "INIT"

        return IntentDecision(
            domain=fallback_domain,
            intent=fallback_intent,
            confidence=0.5,
            target_agent=fallback_agent,
            subject_action=subject_action,
            subject_entity=active_subject or query_clean,
            required_data_sources=["general_science_knowledge"],
            search_query=query_clean,
            extracted_entities={"raw_query": query_clean},
            reasoning=f"LLM 解析异常降级: {last_err}",
        )


# ==========================================
# 4. 多领域通用性测试套件 (Physics / English / Algorithm / LLM)
# ==========================================
TEST_CASES = [
    # --- 1. 物理领域 (Physics) ---
    {
        "category": "物理 (Physics)",
        "query": "老师刚才讲的薛定谔方程波函数塌缩，在课件 PPT 第几页能找到？",
        "expected_intent": "cross_modal_locate",
    },
    {
        "category": "物理 (Physics)",
        "query": "从麦克斯韦方程组怎么一步步推导出电磁波波动方程？能详细写下推导过程吗？",
        "expected_intent": "stepwise_derivation",
    },
    {
        "category": "物理 (Physics)",
        "query": "用大白话给我讲讲广义相对论的时空弯曲到底是什么意思？",
        "expected_intent": "concept_simplify",
    },

    # --- 2. 英语语言学与写作 (English) ---
    {
        "category": "英语 (English)",
        "query": "根据刚才讲的科技论文写作规范，出两道题考考我关于动词时态的选择。",
        "expected_intent": "quiz_generate",
    },
    {
        "category": "英语 (English)",
        "query": "英语中虚拟语气中 should + 动词原形 和 直接用过去式的区别与适用场景是什么？",
        "expected_intent": "compare_contrast",
    },
    {
        "category": "英语 (English)",
        "query": "刚才课上老师留的学术论文摘要重构作业要求是什么？什么时候截止？",
        "expected_intent": "action_homework_extract",
    },

    # --- 3. 算法与数据结构 (Algorithm) ---
    {
        "category": "算法 (Algorithm)",
        "query": "Dijkstra 算法和 A* 启发式搜索算法的核心区别是什么？各自的时间复杂度如何？",
        "expected_intent": "compare_contrast",
    },
    {
        "category": "算法 (Algorithm)",
        "query": "红黑树插入节点后发生双红冲突时的旋转与变色步骤是怎么推演的？请一步步展开。",
        "expected_intent": "stepwise_derivation",
    },
    {
        "category": "算法 (Algorithm)",
        "query": "如果要彻底掌握动态规划，我需要先补哪些前置算法基础？推荐个学习路线。",
        "expected_intent": "study_path_plan",
    },

    # --- 4. 大模型与前沿 AI (LLM / Multi-Agent) ---
    {
        "category": "大模型 (LLM / AI)",
        "query": "总结一下今天这篇关于 MoE 混合专家模型架构讲座的核心重点和结论。",
        "expected_intent": "content_summarize",
    },
    {
        "category": "大模型 (LLM / AI)",
        "query": "在上传的 Attention Is All You Need 论文中，第 3 页提到的缩放点积注意力公式为什么除以根号 dk？",
        "expected_intent": "doc_evidence_query",
    },
    {
        "category": "大模型 (LLM / AI)",
        "query": "用通俗的比喻给我解释一下 LangGraph 中的条件边 Conditional Edge 是怎么工作的？",
        "expected_intent": "concept_simplify",
    },

    # --- 5. 系统控制指令 (Control) ---
    {
        "category": "系统控制 (Control)",
        "query": "好的，今天就学到这里，录音结束",
        "expected_intent": "voice_system_control",
        "expected_long_form": False,
    },

    # --- 6. 专著书籍与技术文档全景考据 (Books, Monographs & Tech Docs) ---
    {
        "category": "学术专著/书籍 (Book/Monograph)",
        "query": "在上传的专著《凝聚态场论》第四章中，关于贝里曲率在动量空间的积分如何得到陈数？请结合书中的公式考据其数学推导。",
        "expected_intent": "doc_evidence_query",
        "expected_long_form": False,
    },
    {
        "category": "技术文档与规范 (Tech Doc)",
        "query": "查阅上传的系统微服务架构设计规范与接口文档，第 3 节关于服务降级与熔断的超时阈值配置标准是什么？",
        "expected_intent": "doc_evidence_query",
        "expected_long_form": False,
    },

    # --- 7. 深水区机理单点推演 (包含深入/详细，但非长篇大论 -> requires_long_form=False) ---
    {
        "category": "深水区单点推演 (Deep Drill-down)",
        "query": "在刚才讨论的基础上深入解析瞬子作用量积分的高斯涨落展开，展开详细推导演算，给出每一个鞍点修正项，不要跳步。",
        "expected_intent": "stepwise_derivation",
        "expected_long_form": False,
    },

    # --- 8. 超长篇完整成果撰写 (立项申报书 / 专著全章节 -> requires_long_form=True) ---
    {
        "category": "基金立项申请书 (Grant Proposal)",
        "query": "请为我起草一份关于拓扑量子物态与格点规范场论的国家自然科学基金重大研究计划立项申请书正文，包括立项依据、关键科学问题与研究方案。",
        "expected_intent": "research_design",
        "expected_long_form": True,
    },
    {
        "category": "学术专著章节草案 (Monograph Chapter)",
        "query": "针对正在编撰的《分布式多智能体系统协同控制》专著，撰写第六章完整章节草案，系统阐述基于合同网与一致性协议的任务协同分配算法与收敛性证明。",
        "expected_intent": "research_design",
        "expected_long_form": True,
    },
]


def run_evaluation():
    print("=" * 75)
    print("🚀 【跨领域通用意图识别智能体】开始多学科与多载体测试评估")
    print(f"📡 大模型后端: {os.getenv('OPENAI_BASE_URL')} | 模型: {os.getenv('OPENAI_MODEL')}")
    print("=" * 75 + "\n")

    agent = IntentRecognitionAgent()
    passed = 0
    total = len(TEST_CASES)

    for i, item in enumerate(TEST_CASES, start=1):
        cat = item["category"]
        q = item["query"]
        expected = item["expected_intent"]
        expected_long = item.get("expected_long_form", False)

        start_t = time.time()
        res = agent.parse_intent(q)
        elapsed = int((time.time() - start_t) * 1000)

        intent_match = (res.intent == expected)
        long_form_match = (res.requires_long_form == expected_long)
        is_all_match = intent_match and long_form_match

        if is_all_match:
            passed += 1
            status_tag = "✅ PASS"
        else:
            diff_reasons = []
            if not intent_match:
                diff_reasons.append(f"意图期望: {expected}, 实际: {res.intent}")
            if not long_form_match:
                diff_reasons.append(f"长篇期望: {expected_long}, 实际: {res.requires_long_form}")
            status_tag = f"❌ MISMATCH ({'; '.join(diff_reasons)})"

        print(f"[{i}/{total}] [{cat}] {status_tag} ({elapsed}ms)")
        print(f"   👤 输入: \"{q}\"")
        print(f"   🎯 识别: 学科=[{res.domain}] | 意图=[{res.intent}] | 目标智能体=[{res.target_agent}]")
        print(f"   📏 长篇大论: requires_long_form={res.requires_long_form} | 体裁={res.long_form_type}")
        print(f"   📦 所需数据源: {res.required_data_sources}")
        print(f"   💡 依据: {res.reasoning}\n")

    acc = (passed / total) * 100
    print("=" * 75)
    print(f"🏁 【测试汇总结果】")
    print(f"📊 总测试用例: {total} 条 | 全指标命中通过: {passed} 条 | 综合准确率: {acc:.1f}%")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_evaluation()
