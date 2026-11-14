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

from config_utils import get_default_llm_model, get_llm_timeout, resolve_retrieval_time_constraint
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
            "用简短自然语言概括用户本轮实际要做的事。可以组合多个动作；"
            "不要求从固定任务类别中选择。"
        )
    )
    task_segments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "按语义关系识别本轮任务片段，保留每项动作、对象、条件及必要的检索表达；"
            "不以标点作为唯一切分依据，也不为凑数量拆分。"
        ),
    )
    confidence: float = Field(default=0.9, description="意图识别置信度 (0.0 ~ 1.0)")
    target_agent: str = Field(description="推荐处理本轮任务的已实现能力名称；不限制或改写任务意图")
    control_action: Optional[str] = Field(default=None, description="仅系统控制任务填写具体动作，如 stop_recording、pause_recording、start_recording、change_voice、clear_session；不确定留空")
    control_parameters: Dict[str, Any] = Field(default_factory=dict, description="控制动作的明确参数：voice 为音色名称或 ID，mode 为 mic 或 system；未指定不填")
    required_data_sources: List[str] = Field(
        default_factory=list,
        description="当前任务明确需要读取的资料类型或用户指定资源；不因学术问题而自动添加资料源"
    )
    search_query: Optional[str] = Field(
        default="",
        description="概括当前任务对象与目标的检索表达；保留影响任务的条件和限定"
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
        description="本轮任务是否需要当前外部学术信息；不能仅因主题学术或生成了检索表达就设为 true"
    )
    online_search_query: Optional[str] = Field(
        default="",
        description="确有在线信息需求时提供有助检索的英文表达；不要求固定数量或纯关键词格式"
    )
    topic_action: str = Field(
        default="STAY_LOCKED",
        description="本轮主题状态动作: INIT_LOCK | STAY_LOCKED | EVOLVE | RESET；按本轮与当前状态的关系判断"
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
            "任务包含相互独立的动作或领域、且拆分有助于执行时，记录相关子任务；"
            "不要求固定数量，也不只用于跨学科问题。"
        )
    )
    time_constraint: Optional[Dict[str, Any]] = Field(
        default=None,
        description="检索时间范围：用户明确要求优先，否则默认当前日期向前推三年；时间范围本身不决定是否检索。"
    )
    time_constraint_note: Optional[str] = Field(
        default="",
        description="用户使用相对事件锚点表达的时间条件原文；需先核实锚点，不直接当作公历年份过滤。"
    )
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="抽取的关键学术概念或实体")
    requires_long_form: bool = Field(
        default=False,
        description="用户是否要求完整长篇成果；详细程度本身不决定该字段",
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
    is_resolved: bool = Field(default=True, description="本轮任务是否已能从输入和上下文中可靠理解")


# ==========================================
# 2. 可调用能力目录（仅供路由能力推荐，不作为任务类别）
# ==========================================
AGENT_CAPABILITIES = [
    {
        "name": "通俗概念原理解释",
        "description": "请求用通俗语言、直观比喻或大白话解释某个学术概念、物理现象、语法规则或模型机制",
        "agent": "ConceptTutorAgent",
    },
    {
        "name": "公式与算法分步推导",
        "description": "请求对数学公式、物理定律方程、算法每一步的演化推导或代码流程进行逐步展开解析",
        "agent": "DerivationAgent",
    },
    {
        "name": "音视频与课件教材对照",
        "description": "询问视频/讲课当前所讲内容对应在课件、PPT或教材文档的哪一页/哪个位置",
        "agent": "CrossModalRAGAgent",
    },
    {
        "name": "学术论文/专著书籍/教材讲义/技术文档全景解读与细节考据",
        "description": "在上传的 PDF 论文、学术专著书籍、教材讲义、技术报告、课件PPT，或当前会话已检索就绪的文献资产池（recent_paper_candidates）中查找证据、筛选分类、提取章节段落，或者请求对某篇论文/专著书籍/课件进行全文概述、背景介绍、方法拆解、方案构思或核心贡献总结",
        "agent": "DocumentSearchAgent",
    },
    {
        "name": "实时语音会议/课堂流水要点提炼",
        "description": "针对刚才开启的实时麦克风/声卡语音录音流水（会议记录、课堂口头讲解）提炼要点与纪要；上传文档的概述应使用文档处理能力。",
        "agent": "SummarizerAgent",
    },
    {
        "name": "作业与待办事项提取",
        "description": "专门提取老师布置的课后作业、思考题、或会议达成的待办事项与截止时间",
        "agent": "SummarizerAgent",
    },
    {
        "name": "多方案/概念对比分析",
        "description": "请求对比两个或多个理论、方法、模型、技术路线或语言现象的区别与优缺点",
        "agent": "ComparisonAgent",
    },
    {
        "name": "考题与随堂测验生成",
        "description": "请求出题考查用户对指定知识点、理论、方法、技术或语言内容的掌握情况",
        "agent": "QuizAgent",
    },
    {
        "name": "学科与技能学习路径规划",
        "description": "请求针对任意学科、技术方向或跨学科主题制定系统化阶段学习计划或前置知识图谱",
        "agent": "StudyPlannerAgent",
    },
    {
        "name": "研究方案与验证路线设计",
        "description": (
            "请求设计、构建或规划研究方案、理论框架、实验流程、算法路线、"
            "跨学科技术路线或验证闭环。重点是明确研究问题、方法、变量、"
            "数据/材料、评价指标、对照实验和可证伪性；不是单纯解释概念、"
            "总结文献或安排学习先后顺序。适用于任意学科。"
        ),
        "agent": "StudyPlannerAgent",
    },
    {
        "name": "系统语音口令与功能控制",
        "description": "用户的控制指令，如开启/关闭录音、暂停、切换模型音色、清空会话等系统级操作",
        "agent": "SystemControlAgent",
    },
    {
        "name": "ArXiv 线上最新学术文献检索与前沿综述",
        "description": "请求在线实时检索 arXiv、SOTA 前沿论文库中最新的研究论文、论文作者、发表年份及 Abstract 摘要并整理学术综述",
        "agent": "OnlinePaperSearchAgent",
    },
    {
        "name": "通用日常与学术问候",
        "description": "打招呼、日常寒暄、感谢、身份询问或与学术无关的闲聊",
        "agent": "GeneralChatAgent",
    },
]

AVAILABLE_AGENTS = {d["agent"] for d in AGENT_CAPABILITIES} | {"GeneralTutorAgent"}


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

        capability_desc = "\n".join(
            [f"- {item['agent']}: {item['name']} — {item['description']}" for item in AGENT_CAPABILITIES]
        )

        return build_intent_system_prompt(intents_desc=capability_desc, current_date=current_date)

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
                f"【会话已有主体状态（仅作上下文）】：\n- 主体: {curr_t}\n- 子分支: {sub_t or '无'}\n- 状态: {status}\n本轮任务可以延续、改变或替换；不要仅凭旧状态推定本轮任务。"
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
                f"【相关历史主题记录（仅供参考）】：\n- 主题: 《{m_top.get('topic_name')}》 | 动作类型: {m_top.get('action_type')}\n- 既有结论与依据:\n{m_top.get('key_findings')}"
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
                    max_tokens=8192,
                    timeout=get_llm_timeout(100.0),
                )
                raw_json = (resp.choices[0].message.content or "").strip()
                if not raw_json:
                    raise RuntimeError(f"模型 {active_model} 返回空内容 (finish_reason={resp.choices[0].finish_reason})")
                if raw_json.startswith("```"):
                    raw_json = re.sub(r"^```[a-zA-Z]*\n?", "", raw_json)
                    raw_json = re.sub(r"\n?```$", "", raw_json).strip()
                parsed_dict = json.loads(raw_json)
                if not parsed_dict.get("search_query"):
                    parsed_dict["search_query"] = query_clean

                # Keep malformed optional objects from forcing the entire intent
                # request to retry. Preserve textual time conditions for repair
                # rather than discarding an existing requirement.
                schema_notes = []
                framework_ref = parsed_dict.get("framework_reference")
                if not isinstance(parsed_dict.get("control_parameters"), dict):
                    parsed_dict["control_parameters"] = {}
                if framework_ref is not None and not isinstance(framework_ref, dict):
                    parsed_dict["framework_reference"] = None
                    schema_notes.append("framework_reference 格式无效，已忽略")
                time_constraint = parsed_dict.get("time_constraint")
                if time_constraint is not None and not isinstance(time_constraint, dict):
                    parsed_dict["time_constraint"] = None
                    if isinstance(time_constraint, str) and time_constraint.strip():
                        if not parsed_dict.get("time_constraint_note"):
                            parsed_dict["time_constraint_note"] = query_clean
                        schema_notes.append("time_constraint 为相对文字条件，未转换为公历日期")
                    else:
                        schema_notes.append("time_constraint 格式无效，采用系统默认检索时间")
                elif isinstance(time_constraint, dict):
                    parsed_dict["time_constraint"] = resolve_retrieval_time_constraint(time_constraint)
                    for k in list(parsed_dict["time_constraint"].keys()):
                        if k not in time_constraint and parsed_dict["time_constraint"][k] in (None, False, "explicit_user_constraint"):
                            parsed_dict["time_constraint"].pop(k, None)
                if schema_notes:
                    previous_reasoning = str(parsed_dict.get("reasoning") or "").strip()
                    parsed_dict["reasoning"] = "；".join(filter(None, [previous_reasoning, *schema_notes]))

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

                suggested_agent = str(parsed_dict.get("target_agent") or "").strip()
                if suggested_agent not in AVAILABLE_AGENTS:
                    parsed_dict["target_agent"] = "GeneralTutorAgent"

                raw_sources = parsed_dict.get("required_data_sources") or []
                if isinstance(raw_sources, str):
                    raw_sources = [raw_sources]
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

        return IntentDecision(
            domain="",
            intent="unresolved",
            confidence=0.0,
            target_agent="GeneralTutorAgent",
            subject_action="INHERIT",
            subject_entity="",
            required_data_sources=[],
            search_query=query_clean,
            extracted_entities={"raw_query": query_clean},
            reasoning=f"本轮任务未能可靠解析；保留用户原话，不推断任务或资料需求。解析错误: {last_err}",
            is_resolved=False,
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
