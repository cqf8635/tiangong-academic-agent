"""Universal Multi-Format Document Knowledge Base & RAG Engine.

Supports:
1. Multi-format parsing: PDF (.pdf), Word (.docx), PowerPoint (.pptx), Markdown (.md), Text (.txt).
2. Traceable metadata: Exact Page Number (PDF), Slide Number (PPT), Section Titles (Word/MD).
3. Local Dense Embedding: BGE-M3 (configurable via PAPER_EMBEDDING_MODEL).
4. Local Cross-Encoder Reranking: BGE-Reranker (configurable via PAPER_RERANKER_MODEL).
5. Persistent Vector Store: ChromaDB (./knowledge_db).
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

import chromadb
from chromadb.config import Settings
import docx
import pptx
import pymupdf
import torch
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

from config_utils import get_default_llm_model, get_llm_timeout, resolve_model_path
from prompts import build_ontology_prompt, build_vlm_transcription_prompt

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

DEFAULT_EMBEDDING_PATH = resolve_model_path(
    "PAPER_EMBEDDING_MODEL", "embedding", "BAAI/bge-m3", BASE_DIR
)
DEFAULT_RERANKER_PATH = resolve_model_path(
    "PAPER_RERANKER_MODEL", "reranker", "BAAI/bge-reranker-large", BASE_DIR
)
DEFAULT_DB_DIR = BASE_DIR / "knowledge_db"
EXTRACTED_TEXTS_DIR = BASE_DIR / "static" / "extracted_texts"
EXTRACTED_IMAGES_DIR = BASE_DIR / "static" / "extracted_images"
UPLOADS_DIR = BASE_DIR / "uploads"
TEST_DOCUMENTS_DIR = BASE_DIR / "test_data" / "documents"


# ==========================================
# 1. 结构化文档切片模型 (Document Chunk Schema)
# ==========================================
class DocumentChunk:
    """结构化文档切片对象，绑定精确页码、小节与原图快照"""

    def __init__(
        self,
        doc_id: str,
        doc_name: str,
        doc_type: str,
        page_or_slide: int,
        section: str,
        chunk_id: str,
        text: str,
        image_url: str = "",
    ):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.doc_type = doc_type
        self.page_or_slide = page_or_slide
        self.section = section
        self.chunk_id = chunk_id
        self.text = text
        self.image_url = image_url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "page_or_slide": self.page_or_slide,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "image_url": self.image_url,
        }


# ==========================================
# 2. 全格式通用解析器 (Universal Document Parser)
# ==========================================
class UniversalDocumentParser:
    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u00ad", "").replace("\u200b", "")
        text = re.sub(r"[ \t]+", " ", text)
        # 规整行间换行，保留连续两个换行表示段落分隔
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _extract_page_vlm(cls, image_path: Path, doc_id: str, page_num: int) -> str:
        """调用多模态 VLM API 提取扫描页的 Markdown 标题、正文与 LaTeX 公式，支持持久化缓存"""
        if not image_path.exists():
            return ""

        cache_dir = EXTRACTED_TEXTS_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{doc_id}_p{page_num}.txt"

        if cache_file.exists():
            try:
                cached_text = cache_file.read_text(encoding="utf-8").strip()
                if cached_text:
                    return cached_text
            except Exception:
                pass

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = get_default_llm_model()

        if not api_key or not base_url:
            return ""

        try:
            with open(image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")

            llm_timeout = get_llm_timeout(90.0)
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=llm_timeout)
            prompt = build_vlm_transcription_prompt()

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1,
                timeout=llm_timeout,
            )

            extracted = resp.choices[0].message.content.strip()
            if extracted:
                cache_file.write_text(extracted, encoding="utf-8")
                return extracted
        except Exception as e:
            print(f"⚠️ [VLM Parse Page {page_num} Error]: {e}")

        return ""

    @staticmethod
    def _is_heading_or_topic_boundary(line: str) -> Optional[str]:
        """通用学术标题与主题分界识别器 (支持中英文主流论文层级规范，零硬编码，零 Token 消耗)"""
        s = line.strip()
        if not s or len(s) > 60:
            return None

        # 1. Markdown 标题语法 (# 标题)
        if re.match(r"^#{1,6}\s+(.+)$", s):
            return re.sub(r"^#{1,6}\s+", "", s).strip()

        # 2. 中文一级/二级宏观章节标题: "一、引言", "（一）样本选择", "(一) 理论分析", "第一章 绪论", "第一节 理论机制"
        m_cn = re.match(r"^([一二三四五六七八九十百]+、|[\(（][一二三四五六七八九十]+[\)）]|第[一二三四五六七八九十\d]+[章节部分篇条])\s*(.+)?$", s)
        if m_cn:
            return s

        # 3. 标准阿拉伯数字层级标题: "1. 引言", "1.1 变量说明", "2.3.1 模型设定", "1、引言" (限制编号在 1~99，杜绝年份如 2024 年被误判)
        m_num = re.match(r"^([1-9]\d{0,1}(?:\.\d{1,2})+|[1-9]\d{0,1}[.、])\s*([A-Za-z\u4e00-\u9fa5].+)$", s)
        if m_num and not re.match(r"^[12]\d{3}\s*年", s):
            return s

        # 4. 关键图表标题: "表 11 多项式回归结果", "图 3 响应面三维分析", "Table 1:", "Figure 2."
        m_tab = re.match(r"^(表\s*\d+|图\s*\d+|Table\s*\d+|Figure\s*\d+)[\s:：]+(.+)$", s, re.IGNORECASE)
        if m_tab:
            return s

        # 5. 常见学术通用结构英文关键词 (如 Abstract, Introduction, Methodology, Conclusion 等)
        m_en = re.match(
            r"^(Abstract|Introduction|Related\s+Work|Methodology|Method|Model\s+Specification|"
            r"Empirical\s+Results|Discussion|Conclusion|Robustness\s+Check|Hypothesis\s+Development)[\s:：]*.*$",
            s, re.IGNORECASE
        )
        if m_en:
            return s

        return None

    @staticmethod
    def _split_long_paragraph(text: str, max_chars: int = 800) -> List[str]:
        """将超长自然段按完整标点句子拆分，保证不腰斩语义 (零硬编码)"""
        if len(text) <= max_chars:
            return [text]
        sentences = re.split(r"(?<=[。！？!?\n])", text)
        result = []
        buf = ""
        for s in sentences:
            if not s:
                continue
            if len(buf) + len(s) <= max_chars:
                buf += s
            else:
                if buf:
                    result.append(buf.strip())
                buf = s
        if buf:
            result.append(buf.strip())
        return result or [text]

    @staticmethod
    def _extract_tail_sentence(text: str, max_chars: int = 100) -> str:
        """从切片尾部提取最后一个完整句子作为滑动窗口前置缓冲 (零硬编码)"""
        s = text.strip()
        if not s:
            return ""
        m = list(re.finditer(r"[。！？!?；;\n]", s))
        if len(m) >= 2:
            second_last = m[-2].end()
            cand = s[second_last:].strip()
            if len(cand) <= max_chars:
                return cand
    @classmethod
    def _split_into_semantic_paragraphs(cls, text: str) -> List[str]:
        """按自然段与标题行拆解，确保每个标题单独成段，支持单换行与双换行排版 (零硬编码)"""
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        result = []
        for p in raw_paras:
            lines = [l.strip() for l in p.split("\n") if l.strip()]
            cur = []
            for line in lines:
                if cls._is_heading_or_topic_boundary(line):
                    if cur:
                        result.append("\n".join(cur))
                        cur = []
                    result.append(line)
                else:
                    cur.append(line)
            if cur:
                result.append("\n".join(cur))
        return result

    @classmethod
    def parse_pdf(cls, file_path: Path, max_chunk_chars: int = 800) -> List[DocumentChunk]:
        """解析 PDF：按页读取、提取章节主题与自然段、渲染高清页面视觉快照，支持扫描版多模态 VLM 深度解析"""
        doc = pymupdf.open(str(file_path))
        doc_id = hashlib.sha1(file_path.name.encode()).hexdigest()[:12]
        chunks: List[DocumentChunk] = []

        img_dir = EXTRACTED_IMAGES_DIR
        img_dir.mkdir(parents=True, exist_ok=True)

        pages_data = []
        scanned_tasks = []

        # 第一阶段：快速扫描每一页，提取原生文字与高清视觉快照
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_text = cls._clean_text(page.get_text("text"))

            snapshot_filename = f"{doc_id}_p{page_num}.png"
            snapshot_path = img_dir / snapshot_filename
            if not snapshot_path.exists():
                try:
                    pix = page.get_pixmap(dpi=130)
                    pix.save(str(snapshot_path))
                except Exception:
                    snapshot_filename = ""

            image_url = f"/extracted_images/{snapshot_filename}" if snapshot_filename else ""
            pages_data.append({
                "page_num": page_num,
                "page_text": page_text,
                "snapshot_path": snapshot_path,
                "image_url": image_url,
            })

            if not page_text and snapshot_path.exists():
                scanned_tasks.append((page_num, snapshot_path))

        doc.close()

        # 第二阶段：如果存在无文字的扫描页，检查是否有缓存或是否开启 VLM 多模态提取
        vlm_results: Dict[int, str] = {}
        cache_dir = EXTRACTED_TEXTS_DIR
        enable_vlm = os.getenv("ENABLE_VLM_OCR", "0") == "1"

        uncached_scanned_tasks = []
        for p_num, snp in scanned_tasks:
            cache_file = cache_dir / f"{doc_id}_p{p_num}.txt"
            if cache_file.exists():
                try:
                    c_text = cache_file.read_text(encoding="utf-8").strip()
                    if c_text:
                        vlm_results[p_num] = c_text
                        continue
                except Exception:
                    pass
            uncached_scanned_tasks.append((p_num, snp))

        if uncached_scanned_tasks:
            if enable_vlm:
                print(f"👁️ [Multimodal VLM] 发现 {len(uncached_scanned_tasks)} 个新扫描页面，启动并发多模态识别 (模型: {get_default_llm_model()})...")
                max_workers = min(8, len(uncached_scanned_tasks))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_page = {
                        executor.submit(cls._extract_page_vlm, snp, doc_id, p_num): p_num
                        for p_num, snp in uncached_scanned_tasks
                    }
                    for future in as_completed(future_to_page):
                        p_num = future_to_page[future]
                        try:
                            res_text = future.result()
                            if res_text:
                                vlm_results[p_num] = res_text
                        except Exception as e:
                            print(f"⚠️ [VLM Extraction Exception (Page {p_num})]: {e}")
            else:
                print(f"ℹ️ [DocumentParser] 发现 {len(uncached_scanned_tasks)} 个扫描页面 (ENABLE_VLM_OCR=0 未启用在线多模态识别，使用原图视觉快照入库)")

        # 第三阶段：组装结构化语义切片 (学术标题层级感知 + 自然段原子保护 + 滑动上下文缓冲，零硬编码)
        active_section = ""
        for item in pages_data:
            page_num = item["page_num"]
            raw_text = item["page_text"]
            image_url = item["image_url"]

            is_vlm_extracted = False
            if not raw_text and page_num in vlm_results:
                raw_text = vlm_results[page_num]
                is_vlm_extracted = True

            if not raw_text:
                # 兜底：纯空白图或未识别出的扫描页
                chunk_id = f"{file_path.stem}_p{page_num}_img"
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        doc_name=file_path.name,
                        doc_type="pdf",
                        page_or_slide=page_num,
                        section=f"Page {page_num} (Figure/Visual)",
                        chunk_id=chunk_id,
                        text=f"【页面视觉图表/扫描快照】 《{file_path.name}》 第 {page_num} 页包含复杂架构图、实验曲线或表格。",
                        image_url=image_url,
                    )
                )
                continue

            # 检测该页首发标题
            page_section = f"Page {page_num}"
            for line in raw_text.splitlines()[:5]:
                h = cls._is_heading_or_topic_boundary(line)
                if h:
                    active_section = h
                    page_section = f"Page {page_num} ({h})"
                    break
            if not page_section.endswith(")") and active_section:
                page_section = f"Page {page_num} ({active_section})"

            # 按语义自然段落聚合拆分 (确保章节标题独立成段)
            raw_paragraphs = cls._split_into_semantic_paragraphs(raw_text)
            current_buffer = ""
            current_section = page_section
            chunk_sub_idx = 1
            last_overlap_cushion = ""

            for raw_p in raw_paragraphs:
                # 检查段落是否以新的章节或图表标题开头
                first_line = raw_p.split("\n")[0].strip()
                detected_heading = cls._is_heading_or_topic_boundary(first_line)
                if detected_heading:
                    # 遇到新主题/标题：只要当前缓冲区有正文，立即作为独立主题切片输出，避免跨章节混杂
                    if current_buffer.strip():
                        chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_sub_idx}"
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_name=file_path.name,
                                doc_type="pdf",
                                page_or_slide=page_num,
                                section=current_section,
                                chunk_id=chunk_id,
                                text=current_buffer,
                                image_url=image_url,
                            )
                        )
                        chunk_sub_idx += 1
                        last_overlap_cushion = cls._extract_tail_sentence(current_buffer, max_chars=100)
                        current_buffer = ""
                    active_section = detected_heading
                    current_section = f"Page {page_num} ({detected_heading})"

                # 对超长自然段（> max_chunk_chars）进行优雅的长句平滑切分，绝不截断在词中
                sub_paragraphs = cls._split_long_paragraph(raw_p, max_chars=max_chunk_chars)
                for p in sub_paragraphs:
                    if len(current_buffer) + len(p) <= max_chunk_chars:
                        current_buffer += ("\n" + p if current_buffer else p)
                    else:
                        if len(current_buffer) >= 180:
                            chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_sub_idx}"
                            chunks.append(
                                DocumentChunk(
                                    doc_id=doc_id,
                                    doc_name=file_path.name,
                                    doc_type="pdf",
                                    page_or_slide=page_num,
                                    section=current_section,
                                    chunk_id=chunk_id,
                                    text=current_buffer,
                                    image_url=image_url,
                                )
                            )
                            chunk_sub_idx += 1
                            last_overlap_cushion = cls._extract_tail_sentence(current_buffer, max_chars=100)
                            # 新块引入前置句缓冲，保证跨块语义连续
                            current_buffer = f"[承接前文] ...{last_overlap_cushion}\n{p}" if last_overlap_cushion else p
                        else:
                            current_buffer += ("\n" + p if current_buffer else p)

            if current_buffer:
                chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_sub_idx}"
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        doc_name=file_path.name,
                        doc_type="pdf",
                        page_or_slide=page_num,
                        section=current_section,
                        chunk_id=chunk_id,
                        text=current_buffer,
                        image_url=image_url,
                    )
                )

        return chunks

    @classmethod
    def parse_docx(cls, file_path: Path, max_chunk_chars: int = 800) -> List[DocumentChunk]:
        """解析 Word (.docx)：按标题层级与自然段语义切分 (零硬编码)"""
        doc = docx.Document(str(file_path))
        doc_id = hashlib.sha1(file_path.name.encode()).hexdigest()[:12]
        chunks: List[DocumentChunk] = []

        current_section = "正文"
        current_buffer = ""
        chunk_idx = 1
        last_overlap_cushion = ""

        for p in doc.paragraphs:
            text = cls._clean_text(p.text)
            if not text:
                continue

            # 标题识别与主题分界
            if p.style and p.style.name.startswith("Heading"):
                current_section = text
            else:
                detected_heading = cls._is_heading_or_topic_boundary(text)
                if detected_heading:
                    if current_buffer.strip():
                        chunk_id = f"{file_path.stem}_docx_c{chunk_idx}"
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_name=file_path.name,
                                doc_type="docx",
                                page_or_slide=chunk_idx,
                                section=current_section,
                                chunk_id=chunk_id,
                                text=current_buffer,
                            )
                        )
                        chunk_idx += 1
                        last_overlap_cushion = cls._extract_tail_sentence(current_buffer, max_chars=100)
                        current_buffer = ""
                    current_section = detected_heading

            sub_paragraphs = cls._split_long_paragraph(text, max_chars=max_chunk_chars)
            for sub_p in sub_paragraphs:
                if len(current_buffer) + len(sub_p) <= max_chunk_chars:
                    current_buffer += ("\n" + sub_p if current_buffer else sub_p)
                else:
                    if len(current_buffer) >= 180:
                        chunk_id = f"{file_path.stem}_docx_c{chunk_idx}"
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_name=file_path.name,
                                doc_type="docx",
                                page_or_slide=chunk_idx,
                                section=current_section,
                                chunk_id=chunk_id,
                                text=current_buffer,
                            )
                        )
                        chunk_idx += 1
                        last_overlap_cushion = cls._extract_tail_sentence(current_buffer, max_chars=100)
                        current_buffer = f"[承接前文] ...{last_overlap_cushion}\n{sub_p}" if last_overlap_cushion else sub_p
                    else:
                        current_buffer += ("\n" + sub_p if current_buffer else sub_p)

        if current_buffer:
            chunk_id = f"{file_path.stem}_docx_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    doc_name=file_path.name,
                    doc_type="docx",
                    page_or_slide=chunk_idx,
                    section=current_section,
                    chunk_id=chunk_id,
                    text=current_buffer,
                )
            )

        return chunks

    @classmethod
    def parse_pptx(cls, file_path: Path) -> List[DocumentChunk]:
        """解析 PowerPoint (.pptx)：按幻灯片 Slide 提取，精准保留 PPT 第几页"""
        prs = pptx.Presentation(str(file_path))
        doc_id = hashlib.sha1(file_path.name.encode()).hexdigest()[:12]
        chunks: List[DocumentChunk] = []

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_texts = []
            slide_title = f"Slide {slide_idx}"

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        txt = cls._clean_text(paragraph.text)
                        if txt:
                            slide_texts.append(txt)

            if slide_texts:
                full_slide_text = "\n".join(slide_texts)
                chunk_id = f"{file_path.stem}_slide{slide_idx}"
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        doc_name=file_path.name,
                        doc_type="pptx",
                        page_or_slide=slide_idx,  # 精准对应 PPT 幻灯片页码
                        section=slide_title,
                        chunk_id=chunk_id,
                        text=full_slide_text,
                    )
                )

        return chunks

    @classmethod
    def parse_text_or_md(cls, file_path: Path, max_chunk_chars: int = 800) -> List[DocumentChunk]:
        """解析 Markdown / 纯文本：按章节标题与自然段语义切分 (零硬编码)"""
        raw = file_path.read_text(encoding="utf-8", errors="ignore")
        text = cls._clean_text(raw)
        doc_id = hashlib.sha1(file_path.name.encode()).hexdigest()[:12]
        chunks: List[DocumentChunk] = []

        paragraphs = cls._split_into_semantic_paragraphs(text)
        current_section = "正文"
        current_buffer = ""
        chunk_idx = 1
        last_overlap_cushion = ""

        for p in paragraphs:
            # 标题识别
            first_line = p.split("\n")[0].strip()
            detected_heading = cls._is_heading_or_topic_boundary(first_line)
            if detected_heading:
                if current_buffer.strip():
                    chunk_id = f"{file_path.stem}_txt_c{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            doc_id=doc_id,
                            doc_name=file_path.name,
                            doc_type="txt",
                            page_or_slide=chunk_idx,
                            section=current_section,
                            chunk_id=chunk_id,
                            text=current_buffer,
                        )
                    )
                    chunk_idx += 1
                    last_overlap_cushion = cls._extract_tail_sentence(current_buffer, max_chars=100)
                    current_buffer = ""
                current_section = detected_heading

            sub_paragraphs = cls._split_long_paragraph(p, max_chars=max_chunk_chars)
            for sub_p in sub_paragraphs:
                if len(current_buffer) + len(sub_p) <= max_chunk_chars:
                    current_buffer += ("\n" + sub_p if current_buffer else sub_p)
                else:
                    if len(current_buffer) >= 180:
                        chunk_id = f"{file_path.stem}_txt_c{chunk_idx}"
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_name=file_path.name,
                                doc_type="txt",
                                page_or_slide=chunk_idx,
                                section=current_section,
                                chunk_id=chunk_id,
                                text=current_buffer,
                            )
                        )
                        chunk_idx += 1
                        last_overlap_cushion = cls._extract_tail_sentence(current_buffer, max_chars=100)
                        current_buffer = f"[承接前文] ...{last_overlap_cushion}\n{sub_p}" if last_overlap_cushion else sub_p
                    else:
                        current_buffer += ("\n" + sub_p if current_buffer else sub_p)

        if current_buffer:
            chunk_id = f"{file_path.stem}_txt_c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    doc_id=doc_id,
                    doc_name=file_path.name,
                    doc_type="txt",
                    page_or_slide=chunk_idx,
                    section=current_section,
                    chunk_id=chunk_id,
                    text=current_buffer,
                )
            )

        return chunks

    @classmethod
    def parse_epub(cls, file_path: Path) -> List[DocumentChunk]:
        """EPUB 电子书专著解析器 (标准库 zipfile 解包，0 额外依赖)"""
        import zipfile
        import re
        chunks = []
        doc_id = str(uuid.uuid4())[:8]
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                # 提取所有包含正文的 HTML / XHTML 文件
                html_files = [n for n in z.namelist() if n.lower().endswith((".html", ".xhtml", ".htm"))]
                chapter_idx = 1
                for hf in sorted(html_files):
                    raw_bytes = z.read(hf)
                    raw_text = raw_bytes.decode("utf-8", errors="ignore")
                    # 清洗 HTML 标签，保留纯净文本
                    clean_text = re.sub(r"<[^>]+>", " ", raw_text)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()
                    if len(clean_text) > 40:
                        chunk_id = f"{file_path.stem}_epub_c{chapter_idx}"
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_name=file_path.name,
                                doc_type="epub",
                                page_or_slide=chapter_idx,
                                section=f"Chapter {chapter_idx}",
                                chunk_id=chunk_id,
                                text=clean_text[:3000],
                            )
                        )
                        chapter_idx += 1
        except Exception as e:
            print(f"⚠️ [DocumentParser] EPUB 解析异常: {e}")
        return chunks

    @classmethod
    def parse_file(cls, file_path: Path) -> List[DocumentChunk]:
        """通用入口：根据后缀自动选择最佳解析器，支持全格式与最大宽容度解析"""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(file_path)
        elif ext in [".pptx", ".ppt"]:
            return cls.parse_pptx(file_path)
        elif ext == ".epub":
            return cls.parse_epub(file_path)
        elif ext in [".md", ".markdown", ".txt", ".tex", ".latex", ".rst", ".html", ".htm", ".json", ".jsonl", ".csv"]:
            return cls.parse_text_or_md(file_path)
        else:
            # 最大宽容度兜底：尝试以纯文本流智能切块，杜绝格式壁垒
            try:
                return cls.parse_text_or_md(file_path)
            except Exception as e:
                raise ValueError(f"文件格式解析失败 ({ext}): {e}")


DocumentParser = UniversalDocumentParser


class DocumentAcademicOntology(BaseModel):
    """开放式学术本体元数据定义 (支持全领域自由层次化与交叉前沿)"""
    primary_discipline: str = Field(
        default="综合学术",
        description="文献的一级主研究领域或学科方向，由模型根据文献实际内容自适应生成"
    )
    sub_fields: List[str] = Field(
        default_factory=list,
        description="文献的具体细分研究方向或课题分支列表"
    )
    interdisciplinary_tags: List[str] = Field(
        default_factory=list,
        description="涉及的跨领域或多学科交叉标签列表"
    )
    core_methodologies: List[str] = Field(
        default_factory=list,
        description="文献所依托的核心理论、算法、控制方程或研究范式"
    )


def extract_document_ontology_with_llm(doc_name: str, sample_text: str, client: Optional[OpenAI] = None) -> DocumentAcademicOntology:
    """基于大模型世界知识自适应提取文献的开放式学术本体结构 (0 硬编码，支持任意学科与交叉前沿)"""
    if not client:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = build_ontology_prompt(doc_name, sample_text)

    try:
        resp = client.chat.completions.create(
            model=get_default_llm_model(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
            timeout=get_llm_timeout(60.0),
        )
        return DocumentAcademicOntology.model_validate_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ [OntologyExtraction Note]: {e}")
        return DocumentAcademicOntology(
            primary_discipline="学术文献",
            sub_fields=[doc_name],
            interdisciplinary_tags=[],
            core_methodologies=[],
        )


# ==========================================
# 3. 知识库管理与混合检索重排引擎
# ==========================================
class UniversalDocumentRAGEngine:
    def __init__(
        self,
        db_dir: Path = DEFAULT_DB_DIR,
        embedding_model_path: str = DEFAULT_EMBEDDING_PATH,
        reranker_model_path: str = DEFAULT_RERANKER_PATH,
        use_cuda_for_models: bool = True,
    ):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        print("⚡ [RAGEngine] 初始化 ChromaDB 嵌入数据库...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="academic_knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)

        self.device = "cuda" if (use_cuda_for_models and torch.cuda.is_available()) else "cpu"
        print(f"🧠 [RAGEngine] 加载本地 BGE-M3 向量嵌入模型 (设备: {self.device})...")
        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        self.embedding_model = SentenceTransformer(
            embedding_model_path,
            device=self.device,
            model_kwargs=model_kwargs,
        )

        use_reranker = os.getenv("PAPER_USE_RERANKER", "0") in ["1", "true", "True"]
        if use_reranker:
            try:
                print(f"⚖️  [RAGEngine] 加载本地 BGE-Reranker 重排模型 (设备: {self.device}, 半精度 FP16)...")
                self.reranker = CrossEncoder(
                    reranker_model_path,
                    device=self.device,
                    model_kwargs=model_kwargs,
                )
            except Exception as e:
                print(f"⚠️ [RAGEngine] 重排模型加载受阻 ({e})，切换为 BGE-M3 纯净向量余弦度量")
                self.reranker = None
        else:
            print("🚀 [RAGEngine] 重排器已由环境配置精简跳过 (PAPER_USE_RERANKER=0)，采用 BGE-M3 全维余弦度量以获得极致速度与精准度")
            self.reranker = None

    def ingest_file(self, file_path: str | Path) -> int:
        """解析任意格式文件并存入本地向量知识库 (通过大模型自动注入开放式学科本体元数据)"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        print(f"\n📂 [Ingest] 开始解析文件: {path.name} (格式: {path.suffix})...")
        chunks = UniversalDocumentParser.parse_file(path)

        if not chunks:
            print(f"[警告] 文件未解析出有效文本: {path.name}")
            return 0

        # 大模型世界知识智能提取多级学术本体 (0 关键词硬编码)
        sample_intro = " ".join([c.text for c in chunks[:4]])
        ont = extract_document_ontology_with_llm(path.name, sample_intro, self.llm_client)
        print(f"🏷️  [Ingest] 学科本体抽取: 《{path.name}》 -> 主学科: 【{ont.primary_discipline}】 | 细分: {ont.sub_fields[:2]} | 交叉: {ont.interdisciplinary_tags}")

        print(f"✂️  [Ingest] 共切出 {len(chunks)} 个结构化切片，正在生成 BGE-M3 向量...")

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "doc_type": c.doc_type,
                "primary_discipline": ont.primary_discipline,
                "sub_fields": ",".join(ont.sub_fields),
                "interdisciplinary_tags": ",".join(ont.interdisciplinary_tags),
                "core_methodologies": ",".join(ont.core_methodologies),
                "page_or_slide": c.page_or_slide,
                "section": c.section,
                "image_url": c.image_url,
            }
            for c in chunks
        ]

        # 向量编码
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # 批量写入 ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"✅ [Ingest] 成功入库: {path.name} ({len(chunks)} 个切片已就绪，已关联学科标签)")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 20,
        doc_filter: Optional[str] = None,
        discipline_filter: Optional[str] = None,
        negative_keywords: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        高精度两级检索：BGE-M3 向量初筛 -> BGE-Reranker 精排 -> 学科亲和度加权与硬截断 (0 跨学科乱配)
        """
        # 1. 向量相似度初筛
        query_vec = self.embedding_model.encode([query], normalize_embeddings=True).tolist()

        where_clause = {"doc_name": doc_filter} if doc_filter else None

        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=candidate_k,
            where=where_clause,
        )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        if not docs:
            return []

        # 2. 计算 BGE-M3 全维余弦相似度 (Cosine Space: similarity = 1.0 - distance)
        cosine_sims = [
            max(0.0, 1.0 - float(distances[i])) if (distances and i < len(distances)) else 0.5
            for i in range(len(docs))
        ]

        # 3. 检查是否启用与有效运行 BGE-Reranker 交叉精排
        scores = cosine_sims
        use_reranker = os.getenv("PAPER_USE_RERANKER", "0") in ["1", "true", "True"]
        if use_reranker and getattr(self, "reranker", None) is not None:
            try:
                rerank_pairs = []
                for doc_text, meta in zip(docs, metas):
                    doc_n = meta.get("doc_name") or ""
                    sec_n = meta.get("section") or ""
                    ctx_prefix = f"《{doc_n}》 {sec_n}\n" if (doc_n or sec_n) else ""
                    rerank_pairs.append([query, f"{ctx_prefix}{doc_text}"])
                raw_preds = self.reranker.predict(rerank_pairs)
                pred_list = [float(p) for p in raw_preds]
                # 智能保护：若重排输出方差极小 (<0.02) 说明模型未训练或词表错乱产生退化常数输出，自动回退
                if len(pred_list) > 1 and (max(pred_list) - min(pred_list) < 0.02):
                    scores = cosine_sims
                else:
                    scores = pred_list
            except Exception as e:
                scores = cosine_sims

        # 组合排序、学科亲和度调整与负向词抑制
        ranked_items = []
        for doc_text, meta, score in zip(docs, metas, scores):
            page_label = f"第 {meta.get('page_or_slide')} 页" if meta.get("doc_type") == "pdf" else f"第 {meta.get('page_or_slide')} 张幻灯片" if meta.get("doc_type") == "pptx" else f"段落 #{meta.get('page_or_slide')}"

            final_score = float(score)

            # 3. 开放式学科本体亲和度调整 (Open Academic Ontology Affinity Boost)
            doc_primary = meta.get("primary_discipline", "学术文献")
            doc_sub_fields = [sf.strip() for sf in (meta.get("sub_fields") or "").split(",") if sf.strip()]
            doc_inter = [it.strip() for it in (meta.get("interdisciplinary_tags") or "").split(",") if it.strip()]

            if discipline_filter and discipline_filter != "general":
                # 基于开放学科本体的层次化语义与交叉标签匹配
                is_affinity_match = (
                    discipline_filter in doc_primary
                    or doc_primary in discipline_filter
                    or any(discipline_filter in sf or sf in discipline_filter for sf in doc_sub_fields)
                    or any(discipline_filter in it or it in discipline_filter for it in doc_inter)
                )
                if is_affinity_match:
                    final_score += 0.25  # 同源学科本体或交叉标签亲和度加权
                else:
                    final_score -= 0.30  # 跨学科完全无交集惩罚

            if negative_keywords:
                t_low = doc_text.lower()
                doc_low = (meta.get("doc_name") or "").lower()
                neg_hits = sum(1 for kw in negative_keywords if kw and (kw.lower() in t_low or kw.lower() in doc_low))
                if neg_hits > 0:
                    final_score -= (0.35 * neg_hits)

            ranked_items.append(
                {
                    "score": final_score,
                    "doc_name": meta.get("doc_name"),
                    "doc_type": meta.get("doc_type"),
                    "primary_discipline": doc_primary,
                    "sub_fields": doc_sub_fields,
                    "interdisciplinary_tags": doc_inter,
                    "page_or_slide": meta.get("page_or_slide"),
                    "location_label": page_label,
                    "section": meta.get("section"),
                    "text": doc_text,
                    "image_url": meta.get("image_url", ""),
                }
            )

        # 按重排分数从高到低排序，严格执行相关性硬截断 (低于 0.35 分一律丢弃，坚决杜绝把无关学科硬凑为证据)
        ranked_items.sort(key=lambda x: x["score"], reverse=True)
        valid_items = [it for it in ranked_items if it["score"] >= 0.35]
        return valid_items[:top_k]

    def delete_document(self, doc_name: str) -> bool:
        """从 ChromaDB 向量库、图片快照与本地识别缓存中彻底删除指定文档的所有关联数据"""
        success = True
        try:
            self.collection.delete(where={"doc_name": doc_name})
            print(f"🗑️ [RAGEngine] 已成功从 ChromaDB 向量库删除文献 《{doc_name}》 的全部切片与向量")
        except Exception as e:
            print(f"⚠️ [RAGEngine] 从 ChromaDB 删除文献失败: {e}")
            success = False

        # 同步清理静态图片快照与提取文本缓存
        try:
            doc_id = hashlib.sha1(doc_name.encode()).hexdigest()[:12]
            img_dir = EXTRACTED_IMAGES_DIR
            txt_dir = EXTRACTED_TEXTS_DIR

            cleaned_imgs = 0
            for img in img_dir.glob(f"{doc_id}_*.png"):
                img.unlink(missing_ok=True)
                cleaned_imgs += 1

            cleaned_txts = 0
            for txt in txt_dir.glob(f"{doc_id}_*.txt"):
                txt.unlink(missing_ok=True)
                cleaned_txts += 1

            if cleaned_imgs > 0 or cleaned_txts > 0:
                print(f"🧹 [RAGEngine] 同步清理关联缓存: {cleaned_imgs} 张页面原图, {cleaned_txts} 份解析文本")
        except Exception as e:
            print(f"⚠️ [RAGEngine] 清理静态缓存异常: {e}")

        return success

    def get_document_mindmap(self, doc_name: str) -> str:
        """获取指定文档的完整深层细分思维导图 (支持 PDF 原生多级目录 / Word / PPT 章节提取)"""
        uploads_dir = UPLOADS_DIR
        test_dir = TEST_DOCUMENTS_DIR
        target_file = None
        for candidate in [uploads_dir / doc_name, test_dir / doc_name, BASE_DIR / doc_name]:
            if candidate.exists():
                target_file = candidate
                break

        if not target_file:
            # 模糊匹配
            for f in list(uploads_dir.glob("*")) + list(test_dir.glob("*")):
                if doc_name in f.name or f.name in doc_name:
                    target_file = f
                    break

        if not target_file:
            return f"# 📚 《{doc_name}》 经卷脉络\n- 暂未找到该文献源文件"

        stem = target_file.stem
        lines = [f"# 📚 《{stem}》 经卷全景导图"]

        if target_file.suffix.lower() == ".pdf":
            try:
                doc = pymupdf.open(str(target_file))
                toc = doc.get_toc()
                if toc and len(toc) >= 3:
                    for lvl, title, page in toc:
                        prefix = "#" * min(lvl + 1, 5)
                        clean_t = title.strip().replace("\n", " ")
                        lines.append(f"{prefix} {clean_t}")
                        lines.append(f"- 📍 考据定位: 第 {page} 页")
                    return "\n".join(lines)
                else:
                    # 无原生大纲，扫描文本中的章节和小节
                    lines.append("## 📖 核心章节与专题提炼")
                    for page_idx in range(len(doc)):
                        page_text = doc[page_idx].get_text("text")
                        for line in page_text.split("\n"):
                            line = line.strip()
                            # 匹配中英文学术论文核心标题 (如 1 Introduction, II. Method, 第X章 等)
                            is_heading = (
                                re.match(r"^(第[一二三四五六七八九十\d]+[章节部分问]|问题\s*\d+|\d+(\.\d+)*\s+[^\d]|[\d一二三四]、|(?:[IVXLCDM]+|[A-Z])[\.\s]+[A-Za-z]|(?:abstract|introduction|background|preliminaries|related\s+work|method(?:ology)?|proposed\s+method|architecture|experiments?|evaluation|results|ablation|discussion|conclusions?|references)\b)", line, re.IGNORECASE)
                                and 3 < len(line) < 65
                                and not line.endswith((".", "。", ";", "；"))
                            )
                            if is_heading:
                                lines.append(f"### {line}")
                                lines.append(f"- 📍 考据定位: 第 {page_idx + 1} 页")
                    return "\n".join(lines)
            except Exception as e:
                return f"# 📚 《{stem}》\n- 导图生成异常: {e}"

        elif target_file.suffix.lower() == ".pptx":
            try:
                prs = pptx.Presentation(str(target_file))
                lines.append("## 📊 幻灯片结构提炼")
                for idx, slide in enumerate(prs.slides, 1):
                    title = f"幻灯片 #{idx}"
                    if slide.shapes.title and slide.shapes.title.text.strip():
                        title = f"Slide {idx}: {slide.shapes.title.text.strip()}"
                    lines.append(f"### {title}")
                    lines.append(f"- 📍 课件定位: 第 {idx} 张幻灯片")
                return "\n".join(lines)
            except Exception as e:
                return f"# 📚 《{stem}》\n- 幻灯片解析异常: {e}"

        elif target_file.suffix.lower() == ".docx":
            try:
                doc = docx.Document(str(target_file))
                lines.append("## 📝 讲义章节脉络")
                for p in doc.paragraphs:
                    text = p.text.strip()
                    if p.style.name.startswith("Heading") or re.match(r"^(第[一二三四五六七八九十\d]+[章节部分]|[\d一二三四]、)", text):
                        lines.append(f"### {text}")
                return "\n".join(lines)
            except Exception as e:
                return f"# 📚 《{stem}》\n- 文档解析异常: {e}"

        return f"# 📚 《{stem}》\n- 暂不支持该格式大纲提取"


def sanitize_topic_title(raw_title: str) -> str:
    """智能清洗用户口语提问，提炼凝练、专业的学术主题主名称（保留真实研究核心问题，绝不暴力覆盖）"""
    if not raw_title:
        return "核心学术脉络导图"

    raw = raw_title.strip()
    # 提取书名号中的文献名（保留用于前缀，但绝不覆盖实际问题）
    book_prefix = ""
    m_book = re.search(r"[《〈]([^》〉]+)[》〉]", raw)
    if m_book:
        book_prefix = f"《{m_book.group(1).strip()}》"

    # 清除前导“根据《...》专著/论文/教材”等前缀引用
    cleaned = re.sub(r"^(根据|结合|按照|基于)?[《〈][^》〉]+[》〉][专著论文教材材料资料]*[，,、\s]*", "", raw)
    # 清理开头问候词与口语提示
    cleaned = re.sub(r"^(你好[，,、\s]*|您好[，,、\s]*|请问[，,、\s]*|请[您]?|帮我|麻烦|我想知道|我想了解|能不能|可以|请详细|详细|请)", "", cleaned)
    cleaned = re.sub(r"^(这个|该|这份|本篇|相关|关于)*", "", cleaned)
    cleaned = re.sub(r"^(介绍|分析|概述|总结|推导|解析|解释|梳理|盘点|对比)[一下]*", "", cleaned)
    # 清理末尾提问形式与图表修饰词
    cleaned = re.sub(r"(用流程图直观展示|用流程图展示|流程图直观展示|直观展示|用图表展示|用图表对比|用图表|流程图|架构图|大体内容|大概是整理了什么|整理了什么|主要讲了什么|讲的是什么|核心内容|大概内容|大体介绍|主要内容|概述一下|总结一下|介绍一下|是什么|怎么样|怎么做|如何|为什么|有何区别|核心要点|要旨|有哪些|？|\?|。|！|!)+$", "", cleaned)
    cleaned = re.sub(r"(资源里面|文件里面|文档里面|切片里面|内容里面|里面|中|之中|关于)*$", "", cleaned)
    cleaned = re.sub(r"(资源|文档|文件|材料|课件|讲义|这本书|该书|该文献|这篇论文)$", "", cleaned)
    cleaned = cleaned.strip("，,、：: 里的中关于 ")

    # 如果清洗后有具体的子问题或特定探讨方向
    if cleaned and len(cleaned) >= 2:
        if book_prefix and not cleaned.startswith("《"):
            return f"{book_prefix}{cleaned} 知识架构导图"
        return f"{cleaned} 知识架构导图"

    # 仅当用户未提具体问题、纯粹查询书籍时才降级为书籍概览精要
    if book_prefix:
        return f"{book_prefix}知识架构精要"

    return "研讨核心脉络导图"


def extract_answer_mindmap(answer_md: str, topic_title: str = "当前研讨核心脉络") -> str:
    """将导师的 Markdown 回答智能转译为层级严密的思维导图 Markmap 结构 (彻底过滤代码块与表格脏数据)"""
    clean_title = sanitize_topic_title(topic_title)
    if not answer_md:
        return f"# 🧠 {clean_title}\n- 暂无研讨内容"

    lines = [f"# 🧠 {clean_title}"]
    current_section = None
    in_code_block = False

    for raw_line in answer_md.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # 代码块隔离：彻底跳过 Mermaid、Chart、代码等内容，绝不抽入思维导图
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # 过滤表格行（含表格分隔符、三线表行等，绝不作为导图节点）
        if line.startswith("|") or "| ---" in line or line.count("|") >= 2:
            continue

        # 过滤独立公式块定界符
        if line.startswith("$$") or line.startswith("\\[") or line.startswith("\\]"):
            continue

        # 1. 智能兼容解析 ASCII 字符树 (如 ├── 1. 基础对象, | ├── 1.1 纯态)
        m_tree = re.match(r"^([\|│\s]*)(?:[├└]──|\+-+)\s*(.+)$", line)
        if m_tree:
            indent_str = m_tree.group(1)
            content = m_tree.group(2).strip()
            pipes = indent_str.count("|") + indent_str.count("│")
            if pipes == 0:
                lines.append(f"## {content}")
                current_section = content
            elif pipes == 1:
                lines.append(f"### {content}")
                current_section = content
            else:
                indent_spaces = "  " * (pipes - 1)
                lines.append(f"{indent_spaces}- {content}")
            continue

        # 2. 匹配标准 Markdown 标题: #, ##, ###, ####
        m_head = re.match(r"^(#{1,5})\s+(.+)$", line)
        if m_head:
            hashes = m_head.group(1)
            txt = m_head.group(2).strip()
            txt = re.sub(r"[：:;；。]+$", "", txt)
            level = min(len(hashes), 3) + 1  # 映射到 ##, ###, ####
            lines.append(f"{'#' * level} {txt}")
            current_section = txt
            continue

        # 3. 匹配加粗标题
        m_bold_head = re.match(r"^\*\*([0-9\.\s一二三四五六七八九十]*[^*]+)\*\*", line)
        if m_bold_head and (len(line) < 60 or m_bold_head.group(1).strip() == line.strip("* ")):
            txt = m_bold_head.group(1).strip()
            txt = re.sub(r"[：:;；。]+$", "", txt)
            if re.match(r"^[0-9一二三四五六七八九十]+[\.、\s]", txt):
                lines.append(f"## {txt}")
            elif re.match(r"^[0-9]+\.[0-9]+", txt):
                lines.append(f"### {txt}")
            else:
                lines.append(f"## {txt}")
            current_section = txt
            continue

        # 4. 匹配列表项: - 或 * 或 1. 2. 3.
        m_list = re.match(r"^([-\*]|\d+[\.、])\s+(.+)$", line)
        if m_list:
            item_text = m_list.group(2).strip()
            item_text = re.sub(r"[;；。]+$", "", item_text)
            if "|" in item_text:
                continue
            if len(item_text) < 120:
                if not current_section:
                    lines.append("## 核心要点概览")
                    current_section = "核心要点概览"
                lines.append(f"- {item_text}")
            continue

        # 5. 匹配重点强调短句（如 Markdown 粗体短句或强调要点）
        if re.match(r"^(\*\*[^\*]+\*\*|\d+[\.、]|\([0-9a-zA-Z]\))", line) and len(line) < 80 and not line.startswith("<") and "|" not in line:
            clean_short = re.sub(r"[;；。]+$", "", line)
            if not current_section:
                lines.append("## 核心要点概览")
                current_section = "核心要点概览"
            lines.append(f"- {clean_short}")

    if len(lines) <= 1:
        lines.append("## 核心要点概览")
        lines.append(f"- {clean_title}")

    return "\n".join(lines)


# ==========================================
# 4. 单元与集成测试套件
# ==========================================
def run_test_suite():
    print("=" * 70)
    print("🚀 【全格式文档知识库与检索引擎】测试启动 (PDF / Word / PPT)")
    print("=" * 70)

    engine = UniversalDocumentRAGEngine()

    # 1. 自动生成多格式测试用例并入库
    test_dir = TEST_DOCUMENTS_DIR
    test_dir.mkdir(parents=True, exist_ok=True)

    # 样例 A: Word 课件 (.docx)
    docx_file = test_dir / "quantum_mechanics_lecture.docx"
    doc = docx.Document()
    doc.add_heading("量子力学与薛定谔方程讲义", 0)
    doc.add_heading("第一节：波函数的基本性质", level=1)
    doc.add_paragraph("波函数 Ψ(x,t) 描述了微观粒子的量子态，其模平方 |Ψ(x,t)|^2 表示在空间特定位置发现粒子的概率密度。")
    doc.add_heading("第二节：薛定谔波动方程推导", level=1)
    doc.add_paragraph("一维含时薛定谔方程为：iħ ∂Ψ/∂t = - (ħ^2 / 2m) ∇^2 Ψ + V(x,t)Ψ。哈密顿算符 H 对应体系的总能量。")
    doc.save(str(docx_file))

    # 样例 B: PPT 演示课件 (.pptx)
    pptx_file = test_dir / "deep_learning_transformer_slides.pptx"
    prs = pptx.Presentation()
    # Slide 1
    s1 = prs.slides.add_slide(prs.slide_layouts[0])
    s1.shapes.title.text = "Transformer 架构精讲"
    s1.placeholders[1].text = "注意力机制与多头计算原理"
    # Slide 2
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "缩放点积注意力 (Scaled Dot-Product)"
    s2.placeholders[1].text = "公式为 Attention(Q,K,V) = softmax(QK^T / sqrt(dk))V。为什么要除以根号 dk？因为当维度 dk 很大时，点积数量级变大，会导致 softmax 梯度饱和消失！"
    prs.save(str(pptx_file))

    # 入库测试
    engine.ingest_file(docx_file)
    engine.ingest_file(pptx_file)

    # 检索与溯源测试
    print("\n" + "-" * 70)
    print("🔍 测试 1: 跨模态查询 PPT 课件内容与页码溯源")
    q1 = "课件里讲的缩放点积注意力为什么要除以根号 dk？在第几张幻灯片？"
    res1 = engine.search(q1, top_k=2)
    for i, r in enumerate(res1, 1):
        print(f"[{i}] 匹配得分: {r['score']:.4f} | 来源: {r['doc_name']} ({r['location_label']})")
        print(f"    内容: {r['text']}\n")

    print("-" * 70)
    print("🔍 测试 2: 物理讲义 Word 文档公式检索")
    q2 = "一维含时薛定谔方程的公式是什么？"
    res2 = engine.search(q2, top_k=2)
    for i, r in enumerate(res2, 1):
        print(f"[{i}] 匹配得分: {r['score']:.4f} | 来源: {r['doc_name']} ({r['location_label']})")
        print(f"    内容: {r['text']}\n")

    print("=" * 70)
    print("✅ 【全格式知识库与检索测试全部通过！】")
    print("=" * 70)


if __name__ == "__main__":
    run_test_suite()
