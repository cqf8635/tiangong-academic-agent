"""Official Model Context Protocol (MCP) Server for the Multi-Agent System.

Standardized MCP Interfaces for Claude Desktop, Cursor, and IDEs:
1. MCP Tools:
   - search_arxiv_papers: Online arXiv academic literature discovery & abstract ranking.
   - download_arxiv_paper: Automated PDF downloading & BGE-M3 vector indexing.
   - ingest_document: Ingests PDF, Word (.docx), PPT (.pptx), MD into local BGE-M3 + Reranker vector DB.
   - search_knowledge_base: Retrieves relevant document chunks with exact Page/Slide citations.
   - derive_formula_stepwise: Step-by-step LaTeX formula derivation and academic reasoning.
   - recognize_intent: Cross-domain intent parsing across Physics, English, Algorithms, AI, Control.
   - summarize_meeting: Generates Map-Reduce meeting minutes & Action items from records.
   - text_to_speech: Synthesizes text into high-quality female voice (WAV).

2. MCP Resources:
   - records://latest: Real-time meeting stream transcript.
   - knowledge://manifest: List of ingested documents, page counts, and chunks.

3. MCP Prompts:
   - paper_review_prompt: Template for deep academic paper critique and analysis.
   - formula_derivation_prompt: Template for mathematical physics equation derivation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from document_rag_engine import UniversalDocumentRAGEngine
from intent_recognition_agent import IntentRecognitionAgent
from meeting_summarizer import MeetingSummarizer
from online_paper_search_agent import OnlinePaperSearchAgent

from config_utils import get_default_llm_model, get_llm_timeout
from prompts import get_assistant_name

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

# ==========================================
# 1. 初始化 FastMCP 服务端实例
# ==========================================
mcp = FastMCP("ResearchAgentMCP")

# 懒加载单例引擎
_rag_engine: Optional[UniversalDocumentRAGEngine] = None
_intent_agent: Optional[IntentRecognitionAgent] = None
_meeting_summarizer: Optional[MeetingSummarizer] = None
_paper_search_agent: Optional[OnlinePaperSearchAgent] = None


def get_rag():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = UniversalDocumentRAGEngine()
    return _rag_engine


def get_intent():
    global _intent_agent
    if _intent_agent is None:
        _intent_agent = IntentRecognitionAgent()
    return _intent_agent


def get_summarizer():
    global _meeting_summarizer
    if _meeting_summarizer is None:
        _meeting_summarizer = MeetingSummarizer()
    return _meeting_summarizer


def get_paper_agent():
    global _paper_search_agent
    if _paper_search_agent is None:
        _paper_search_agent = OnlinePaperSearchAgent()
    return _paper_search_agent


# ==========================================
# 2. 暴露 MCP Tools (标准工具接口)
# ==========================================
@mcp.tool()
def search_arxiv_papers(query: str, max_results: int = 5) -> str:
    """
    在线检索 arXiv 最新学术论文预印本文献 (MCP 标准接口)。
    返回论文标题、作者团队、发表年份、核心 Abstract 摘要与 PDF 官方直达链接。

    Args:
        query: 学术检索关键词或专业研究领域
        max_results: 返回结果篇数 (默认 5)
    """
    agent = get_paper_agent()
    papers = agent.search_arxiv(query=query, max_results=max_results)
    if not papers:
        return f"未在 arXiv 上找到与「{query}」匹配的论文。"
    return json.dumps([p.model_dump() for p in papers], ensure_ascii=False, indent=2)


@mcp.tool()
def search_openalex_papers(query: str, max_results: int = 4) -> str:
    """
    在线检索 OpenAlex 全球学术知识图谱 (2.5 亿篇全学科文献、顶会顶刊录用与高被引引用量统计)。
    适用于医学、物理、经济金融、工程材料、计算机与人文社科等全行业。

    Args:
        query: 学术检索关键词或专业研究领域
        max_results: 返回结果篇数 (默认 4)
    """
    agent = get_paper_agent()
    resources = agent.search_openalex(query=query, max_results=max_results)
    if not resources:
        return f"未在 OpenAlex 上找到与「{query}」匹配的学术文献。"
    return json.dumps([r.model_dump() for r in resources], ensure_ascii=False, indent=2)


@mcp.tool()
def search_biomedical_pubmed(query: str, max_results: int = 4) -> str:
    """
    在线检索 Europe PMC / PubMed 全球生物医学、生命科学与临床健康顶刊 (Lancet, NEJM, Nature Medicine, Cell 等)。
    100% 免登录、权威公立学术数据库。

    Args:
        query: 生物医学、疾病诊断、靶向药研发或基因组学检索词
        max_results: 返回结果篇数 (默认 4)
    """
    agent = get_paper_agent()
    resources = agent.search_europe_pmc(query=query, max_results=max_results)
    if not resources:
        return f"未在 Europe PMC/PubMed 上找到与「{query}」匹配的生物医学顶刊。"
    return json.dumps([r.model_dump() for r in resources], ensure_ascii=False, indent=2)


@mcp.tool()
def search_physics_inspire_hep(query: str, max_results: int = 4) -> str:
    """
    在线检索 CERN INSPIRE-HEP (欧洲核子研究中心) 全球高能物理、量子物理、广义相对论与凝聚态学术总库。
    100% 免登录公立顶尖物理数据库。

    Args:
        query: 物理学定理、量子纠缠、高能粒子或拓扑物态检索词
        max_results: 返回结果篇数 (默认 4)
    """
    agent = get_paper_agent()
    resources = agent.search_inspire_hep(query=query, max_results=max_results)
    if not resources:
        return f"未在 CERN INSPIRE-HEP 上找到与「{query}」匹配的物理文献。"
    return json.dumps([r.model_dump() for r in resources], ensure_ascii=False, indent=2)


@mcp.tool()
def search_doaj_open_access(query: str, max_results: int = 4) -> str:
    """
    在线检索 DOAJ (Directory of Open Access Journals) 全球全学科同行评议开放获取顶刊 (20,000+ 本高质量期刊)。
    涵盖材料科学、绿色能源、农业、工程与经济管理。

    Args:
        query: 综合行业学术检索词
        max_results: 返回篇数 (默认 4)
    """
    agent = get_paper_agent()
    resources = agent.search_doaj(query=query, max_results=max_results)
    if not resources:
        return f"未在 DOAJ 上找到与「{query}」匹配的开放获取期刊。"
    return json.dumps([r.model_dump() for r in resources], ensure_ascii=False, indent=2)


@mcp.tool()
def search_cs_dblp(query: str, max_results: int = 4) -> str:
    """
    在线检索 DBLP 计算机科学、软件工程、电子通信与人工智能顶级会议与 IEEE/ACM Transactions 期刊录用。

    Args:
        query: 计算机算法、大模型、体系结构或视觉网络检索词
        max_results: 返回篇数 (默认 4)
    """
    agent = get_paper_agent()
    resources = agent.search_dblp(query=query, max_results=max_results)
    if not resources:
        return f"未在 DBLP 上找到与「{query}」匹配的计算机顶级出版物。"
    return json.dumps([r.model_dump() for r in resources], ensure_ascii=False, indent=2)


@mcp.tool()
def search_crossref_journals(query: str, max_results: int = 4) -> str:
    """
    在线检索 CrossRef 权威已发表期刊与顶会出版物 (IEEE/ACM/Nature/Science/Springer 等 15,000+ 出版社权威 DOI 文献)。

    Args:
        query: 学术检索关键词或期刊主题
        max_results: 返回篇数 (默认 4)
    """
    agent = get_paper_agent()
    resources = agent.search_crossref(query=query, max_results=max_results)
    if not resources:
        return f"未在 CrossRef 上找到与「{query}」匹配的期刊文献。"
    return json.dumps([r.model_dump() for r in resources], ensure_ascii=False, indent=2)


@mcp.tool()
def search_github_repositories(query: str, max_results: int = 3) -> str:
    """
    在线检索 GitHub 开源算法库、论文官方复现代码、Star 趋势与技术栈。

    Args:
        query: 开源仓库名称或算法关键词
        max_results: 返回仓库数量 (默认 3)
    """
    agent = get_paper_agent()
    repos = agent.search_github(query=query, max_results=max_results)
    if not repos:
        return f"未在 GitHub 上找到与「{query}」匹配的开源仓库。"
    return json.dumps([r.model_dump() for r in repos], ensure_ascii=False, indent=2)


@mcp.tool()
def search_multisource_academic(query: str) -> str:
    """
    跨 OpenAlex (2.5亿篇) + arXiv 预印本 + CrossRef DOI + GitHub 开源实现的一体化多源学术融合检索。

    Args:
        query: 综合科研检索关键词
    """
    agent = get_paper_agent()
    bundle = agent.search_multisource(query=query, include_github=True)
    return json.dumps(bundle, ensure_ascii=False, indent=2)


@mcp.tool()
def download_and_ingest_arxiv_paper(arxiv_id: str) -> str:
    """
    根据 arXiv 论文 ID 自动下载官方 PDF 并切片构建 BGE-M3 向量索引。

    Args:
        arxiv_id: 论文的 arXiv 编号
    """
    agent = get_paper_agent()
    rag = get_rag()
    papers = agent.search_arxiv(query=f"id:{arxiv_id}", max_results=1)
    if not papers:
        # Fallback to direct URL construction
        from online_paper_search_agent import ArxivPaper
        paper = ArxivPaper(
            paper_id=arxiv_id,
            title=f"arXiv_{arxiv_id}",
            pdf_url=f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
        )
    else:
        paper = papers[0]

    res = agent.download_and_ingest(paper, rag_engine=rag)
    if res.get("status") == "success":
        return f"✅ 论文 [{paper.title}] 下载并入库成功！共生成 {res.get('chunks_indexed', 0)} 个向量索引切片。"
    return f"❌ 论文下载或入库失败: {res.get('reason')}"


@mcp.tool()
def ingest_document(file_path: str) -> str:
    """
    解析并导入任意格式的本地学术资料（PDF 论文、Word 讲义、PPT 课件、Markdown、TXT）到本地向量知识库。
    自动提取页码、幻灯片序号并生成 BGE-M3 密集向量索引。

    Args:
        file_path: 本地文件绝对路径
    """
    rag = get_rag()
    path = Path(file_path)
    if not path.exists():
        return f"[错误] 文件不存在: {file_path}"

    chunks_count = rag.ingest_file(path)
    return f"✅ 文件 [{path.name}] 解析入库成功！共生成 {chunks_count} 个带页码/章节标记的结构化切片。"


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    在本地全格式知识库（PDF/PPT/Word）中执行两级混合检索 (BGE-M3 向量初筛 + BGE-Reranker 交叉精排)。
    返回最相关的证据段落以及精确的页码（第 X 页 / 第 X 张幻灯片）引用。

    Args:
        query: 用户的学术问题、定理概念或课件对照查询
        top_k: 返回最相关的段落数量 (默认 3)
    """
    rag = get_rag()
    results = rag.search(query=query, top_k=top_k)
    if not results:
        return "本地知识库中未检索到相关内容。"

    out_parts = []
    for i, r in enumerate(results, 1):
        out_parts.append(
            f"【证据 {i}】 来源: 《{r['doc_name']}》 {r['location_label']} (匹配得分: {r['score']:.4f})\n{r['text']}"
        )
    return "\n\n".join(out_parts)


@mcp.tool()
def derive_formula_stepwise(topic_or_equation: str) -> str:
    """
    数理公式与学术推导演算工具。调用大模型学术导师按严密逻辑分步推演数学、物理方程，输出标准 LaTeX 格式。

    Args:
        topic_or_equation: 要推导的方程、物理定理或数学问题
    """
    from master_agent_workflow import MasterEngineRegistry, safe_chat_completion
    client = MasterEngineRegistry.get_llm_client()
    name = get_assistant_name()
    prompt = f"""你是一位数理逻辑极其严密的科研搭档“{name}”。请针对以下学术或工程主题（数理、物理、化学、计算机算法或连续介质模型等）进行严密、清晰、分步的 LaTeX 公式推演：
主题: "{topic_or_equation}"

要求：
1. 明确写出初始控制方程、第一性原理假设、基本定义与已知条件；
2. 逐行分步推演，关键变换给出严谨的数学或物理依据；
3. 公式使用标准 LaTeX 块级格式 ($$ ... $$) 或行内格式 ($ ... $)；
4. 总结推导结果的理论意义或算法应用价值。
"""
    try:
        model = get_default_llm_model()
        return safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=get_llm_timeout(90.0),
        )
    except Exception as e:
        return f"公式推演服务异常: {e}"


@mcp.tool()
def recognize_intent(query: str) -> str:
    """
    跨学科通用意图识别工具。准确分析用户输入所属的研究领域、核心意图类型与任务范式。

    Args:
        query: 用户口语或文字提问
    """
    intent_agent = get_intent()
    decision = intent_agent.parse_intent(query)
    return json.dumps(decision.model_dump(), ensure_ascii=False, indent=2)


@mcp.tool()
def summarize_meeting(record_jsonl_path: Optional[str] = None) -> str:
    """
    对长会议录音、讲座流水进行分段 Map-Reduce 归纳总结，自动提取《核心决议》、《核心议题》与《待办事项 Action Items 清单》。

    Args:
        record_jsonl_path: 可选的会议流水 .jsonl 路径，若不填则自动使用 records/ 下最新记录
    """
    summarizer = get_summarizer()
    if record_jsonl_path:
        target_path = Path(record_jsonl_path)
    else:
        records_dir = BASE_DIR / "records"
        latest_records = list(records_dir.glob("record_*.jsonl"))
        if not latest_records:
            return "未找到任何会议录音记录。"
        latest_records.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        target_path = latest_records[0]

    report = summarizer.summarize_session(target_path)
    return summarizer._render_markdown_report(report)


@mcp.tool()
def text_to_speech(text: str, output_wav_path: Optional[str] = None) -> str:
    """
    将文本合成为高保真自然女声语音文件 (WAV)。

    Args:
        text: 要朗读的文本内容
        output_wav_path: 输出音频的绝对路径
    """
    import subprocess
    clean_text = text.replace("#", "").replace("*", "").strip()[:150]
    out_file = Path(output_wav_path) if output_wav_path else (BASE_DIR / "tts_response.wav")
    tts_voice = os.getenv("DEFAULT_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    try:
        cmd = [
            "edge-tts",
            "--voice", tts_voice,
            "--text", clean_text,
            "--write-media", str(out_file),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"✅ 音频合成成功: {out_file}"
    except Exception as e:
        return f"[TTS 失败]: {e}"


# ==========================================
# 3. 暴露 MCP Resources (标准只读数据资源)
# ==========================================
@mcp.resource("records://latest")
def get_latest_meeting_transcript() -> str:
    """获取最新一场会议/讲座的完整 Markdown 逐字记录流水"""
    records_dir = BASE_DIR / "records"
    latest_mds = list(records_dir.glob("record_*.md"))
    if not latest_mds:
        return "暂无会议记录流水。"
    latest_mds.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return latest_mds[0].read_text(encoding="utf-8")


@mcp.resource("knowledge://manifest")
def get_knowledge_manifest() -> str:
    """获取当前已构建 BGE-M3 向量索引的学术文档资产清单"""
    rag = get_rag()
    manifest = rag.get_ingested_manifest()
    return json.dumps(manifest, ensure_ascii=False, indent=2)


# ==========================================
# 4. 暴露 MCP Prompts (标准提示词模板)
# ==========================================
@mcp.prompt()
def paper_review_prompt(paper_title: str, research_focus: str) -> str:
    """学术论文深度研读与批判性审稿的零样本提示词模板"""
    return (
        f"请以资深期刊审稿人与科研导师身份，深入评析论文《{paper_title}》。\n"
        f"重点关注研究方向：{research_focus}。\n"
        "请从研究动机、核心理论机制、数学公式推导、实验可复现性与潜在局限性展开全面评价。"
    )


@mcp.prompt()
def formula_derivation_prompt(target_formula: str, physical_domain: str) -> str:
    """数理方程第一性原理分步推演的零样本提示词模板"""
    return (
        f"请针对【{physical_domain}】领域中的核心命题/方程【{target_formula}】，"
        "从第一性原理出发，写出详细的数学物理定义、守恒定律推导过程以及边界条件。"
    )


if __name__ == "__main__":
    print("🚀 [MCP Server] 正在启动科研多智能体 MCP 服务 (基于 stdio 协议)...")
    mcp.run()
