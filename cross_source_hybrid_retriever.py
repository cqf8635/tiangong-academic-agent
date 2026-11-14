"""Cross-Source Hybrid Deduplication & Fusion Retrieval Engine.

Implements:
1. 5-Way Concurrent Hybrid Retrieval (Local Dense, Local Sparse, arXiv, CrossRef, GitHub).
2. Deterministic & Fuzzy Semantic Deduplication (ArXiv ID, DOI, Title Normalization & Jaccard/Levenshtein matching).
3. Attribute Enrichment & Node Fusion (Merging Preprint + Published DOI + GitHub Repo + Local Slices).
4. Global BGE-Reranker cross-encoder unified scoring.
"""

from __future__ import annotations

from datetime import date, datetime
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from document_rag_engine import UniversalDocumentRAGEngine
from online_paper_search_agent import OnlinePaperSearchAgent
from config_utils import resolve_retrieval_time_constraint, retrieval_date_bounds


def normalize_title(title: str) -> str:
    """标准化论文与文献标题指纹（移除标点符号、版本后缀与多余空格）"""
    if not title:
        return ""
    # 移除常见前缀/后缀，如 "Review for", "arXiv:", "[v1]" 等
    t = re.sub(r"^(review\s+for\s+|arxiv:\s*|paper:\s*|\[.*?\]\s*)", "", title.lower())
    # 移除非字母数字字符
    t = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", " ", t)
    return " ".join(t.split())


def calculate_title_similarity(title1: str, title2: str) -> float:
    """计算两篇文献标题的混合相似度 (Jaccard 词袋 + SequenceMatcher)"""
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)
    if not norm1 or not norm2:
        return 0.0
    if norm1 == norm2:
        return 1.0

    # 1. 词袋 Jaccard 相似度
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if not words1 or not words2:
        return 0.0
    jaccard = len(words1 & words2) / len(words1 | words2)

    # 2. 序列 Levenshtein 相似度
    seq_sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()

    # 综合打分 (短标题重序列，长标题重词袋)
    return 0.6 * seq_sim + 0.4 * jaccard


class UnifiedEvidenceCard:
    """多源融合后的黄金证据节点"""

    def __init__(
        self,
        doc_name: str,
        doc_type: str = "doc",
        text: str = "",
        score: float = 0.0,
        year: Optional[int] = None,
        location_label: str = "",
        section: str = "",
        arxiv_id: Optional[str] = None,
        doi: Optional[str] = None,
        pdf_url: Optional[str] = None,
        github_url: Optional[str] = None,
        stars: Optional[int] = None,
        authors: Optional[List[str]] = None,
        is_local_indexed: bool = False,
        summary: str = "",
        page_or_slide: Optional[Any] = None,
        evidence_level: Optional[str] = None,
        content_scope: Optional[str] = None,
        version_ref: Optional[str] = None,
        version_ref_type: Optional[str] = None,
        published_at: Optional[str] = None,
        temporal_match: Optional[str] = None,
    ):
        self.doc_name = doc_name
        self.doc_type = doc_type  # local_pdf | arxiv | openalex | crossref | github | fused_academic
        self.text = text
        self.score = score
        self.year = year
        self.location_label = location_label
        self.section = section
        self.page_or_slide = page_or_slide
        self.arxiv_id = arxiv_id
        self.doi = doi
        self.pdf_url = pdf_url
        self.github_url = github_url
        self.stars = stars
        self.authors = authors or []
        self.is_local_indexed = is_local_indexed
        self.summary = summary
        self.evidence_level = evidence_level
        self.content_scope = content_scope
        self.version_ref = version_ref
        self.version_ref_type = version_ref_type
        self.published_at = published_at
        self.temporal_match = temporal_match
        self.sources_merged = [doc_type]

    def to_dict(self) -> Dict[str, Any]:
        badge_list = []
        if self.is_local_indexed:
            badge_list.append("🏷️ 本地精读")
        if self.arxiv_id:
            badge_list.append(f"📚 arXiv:{self.arxiv_id}")
        if self.doi:
            badge_list.append(f"🏛️ 期刊/会议 DOI:{self.doi}")
        if self.github_url:
            badge_list.append(f"💻 GitHub 开源 ⭐{self.stars or 0}")

        fused_label = self.location_label
        if badge_list:
            fused_label = f"{' | '.join(badge_list)} ({self.location_label})"

        return {
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "location_label": fused_label,
            "page_or_slide": self.page_or_slide if self.page_or_slide is not None else self.year,
            "section": self.section or "Core Theory",
            "score": self.score,
            "text": self.text,
            "summary": self.summary,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "url": self.pdf_url or self.github_url,
            "version_ref": self.version_ref,
            "version_ref_type": self.version_ref_type,
            "github_url": self.github_url,
            "stars": self.stars,
            "authors": self.authors,
            "is_local_indexed": self.is_local_indexed,
            "sources_merged": list(set(self.sources_merged)),
            "evidence_level": self.evidence_level or (
                "local_snippet" if self.is_local_indexed else
                "repository_metadata" if self.github_url else
                "dataset_repository_metadata" if self.doc_type == "dataset" else
                "web_search_snippet" if self.doc_type == "web" else
                "abstract" if self.doc_type == "arxiv" else
                "index_metadata_or_abstract" if self.doc_type in {"openalex", "crossref", "fused_academic"} else
                "unknown"
            ),
            "provenance": {
                "source_types": list(dict.fromkeys(self.sources_merged)),
                "location": self.location_label,
                "url": self.pdf_url or self.github_url,
                "year": self.year,
                "published_at": self.published_at,
                "temporal_match": self.temporal_match,
                "version_ref": self.version_ref,
                "version_ref_type": self.version_ref_type,
                "content_scope": self.content_scope or (
                    "indexed local passage" if self.is_local_indexed else
                    "repository metadata; source code not inspected" if self.github_url else
                    "dataset repository metadata; dataset content not fetched" if self.doc_type == "dataset" else
                    "web result snippet; destination page not fetched" if self.doc_type == "web" else
                    "abstract or bibliographic record; full text not confirmed"
                ),
            },
        }


class CrossSourceHybridRetriever:
    """跨源多路召回、确定性/语义去重与全局重排引擎"""

    def __init__(
        self,
        rag_engine: Optional[UniversalDocumentRAGEngine] = None,
        paper_agent: Optional[OnlinePaperSearchAgent] = None,
    ):
        self.rag_engine = rag_engine
        self.paper_agent = paper_agent
        self.last_source_errors: Dict[str, str] = {}
        self.last_source_counts: Dict[str, int] = {}

    def hybrid_search_and_deduplicate(
        self,
        query: str,
        negative_keywords: Optional[List[str]] = None,
        top_k: int = 6,
        include_local: bool = True,
        include_online: bool = True,
        time_constraint: Optional[Dict[str, Any]] = None,
        english_query: Optional[str] = None,
        max_chunks_per_doc: int = 4,
        include_github: bool = True,
        include_web: bool = False,
        include_datasets: bool = False,
        discipline_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """执行全源混合并发召回、跨源语义去重与全局融合重排 (同文献多页保护，支持全面文献深读)"""
        time_constraint = resolve_retrieval_time_constraint(time_constraint)
        self.last_source_errors = {}
        self.last_source_counts = {}
        raw_candidates: List[UnifiedEvidenceCard] = []

        # 1. 并发执行多路召回
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 路线 A: 本地 RAG 语义检索 (扩大候选窗口至 max(top_k, 8)，获取同文献更多代表性页面)
            local_limit = max(top_k, 8)
            fut_local = executor.submit(self._fetch_local_evidence, query, negative_keywords, time_constraint, local_limit, discipline_filter) if include_local else None
            # 路线 B: 线上多源学术与开源工程探针
            fut_online = executor.submit(self._fetch_online_evidence, query, time_constraint, english_query, include_github, include_web, include_datasets) if include_online else None

            local_cards = fut_local.result() if fut_local else []
            online_cards = fut_online.result() if fut_online else []

        raw_candidates.extend(local_cards)
        raw_candidates.extend(online_cards)

        print(f"📦 [CrossSourceHybrid] 多路召回原始候选总数: {len(raw_candidates)} 项 (本地符合时间切片: {len(local_cards)} 个, 线上文献候选: {len(online_cards)} 篇)")

        if not raw_candidates:
            return []

        # 2. 执行跨源多维去重与节点属性融合 (保留同文献不同页码的互补切片)
        deduped_cards = self._deduplicate_and_merge(raw_candidates, max_chunks_per_doc=max_chunks_per_doc)
        print(f"✨ [CrossSourceHybrid] 跨源去重与融合后唯一候选数: {len(deduped_cards)} 个 (消除冗余 {len(raw_candidates) - len(deduped_cards)} 个)")

        # 3. 全局重排：BGE-M3 MMR 横向去冗 + 纵向时间演进脉络排序
        selected_cards = self._mmr_and_evolution_rank(query, deduped_cards, top_k)
        # 4. 按需临时抓取 Top 1~2 篇核心 arXiv 论文的 HTML 方法/实验节（不入本地库）
        self._enrich_top_arxiv_html(selected_cards)
        return [c.to_dict() for c in selected_cards]

    def _enrich_top_arxiv_html(self, cards: List[UnifiedEvidenceCard], max_papers: int = 2) -> None:
        if os.getenv("ENABLE_ARXIV_HTML_ENRICHMENT", "true").lower() in ("0", "false", "no"):
            return
        try:
            limit = max(0, min(int(os.getenv("ARXIV_HTML_ENRICH_LIMIT", str(max_papers))), 3))
        except ValueError:
            limit = max_papers
        if limit <= 0:
            return
        targets = sorted(
            [
                c for c in cards
                if c.doc_type == "arxiv"
                and c.arxiv_id
                and not c.is_local_indexed
                and c.evidence_level == "abstract"
            ],
            key=lambda c: float(c.score),
            reverse=True,
        )[:limit]
        if not targets:
            return

        from online_resource_adapters import fetch_arxiv_html_sections

        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            futures = {executor.submit(fetch_arxiv_html_sections, c.arxiv_id): c for c in targets}
            for fut, card in futures.items():
                try:
                    enriched = fut.result(timeout=10)
                except Exception:
                    continue
                if not enriched or not enriched.get("text"):
                    continue
                card.text = f"{card.text}\n\n【arXiv HTML 方法/实验节临时摘录】\n{enriched['text']}".strip()
                card.evidence_level = enriched.get("evidence_level", "full_text_excerpt")
                card.content_scope = enriched.get("content_scope", "arXiv HTML 临时抓取方法/实验节（未入本地库）")
                if "[HTML" not in card.location_label:
                    card.location_label = f"{card.location_label} [HTML 方法/实验节]"

    def _mmr_and_evolution_rank(
        self,
        query: str,
        cards: List[UnifiedEvidenceCard],
        top_k: int,
        lambda_param: float = 0.72,
    ) -> List[UnifiedEvidenceCard]:
        """利用 BGE-M3 向量执行最大边界相关性 (MMR) 过滤高度同质论文，并按年份正序保留技术演进脉络。"""
        if not cards:
            return []

        def _base_priority(c: UnifiedEvidenceCard, q_sim: float = 0.0) -> float:
            title_low = (c.doc_name or "").lower()
            is_survey = 1.0 if any(kw in title_low for kw in ("survey", "review", "overview", "综述", "进展")) else 0.0
            return (
                0.55 * float(c.score)
                + 0.45 * q_sim
                + (0.15 if c.is_local_indexed else 0.0)
                + (0.05 if c.github_url else 0.0)
                + (0.06 * is_survey)
            )

        emb_model = getattr(self.rag_engine, "embedding_model", None) if self.rag_engine else None
        selected: List[UnifiedEvidenceCard] = []

        if emb_model is not None and len(cards) > 2:
            try:
                texts = [query] + [
                    f"{c.doc_name}. {(c.summary or c.text or '')[:450]}".strip()
                    for c in cards
                ]
                encoded = emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                vecs = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
                if len(vecs) == len(texts) and len(vecs[0]) > 4:
                    q_vec = vecs[0]
                    doc_vecs = vecs[1:]

                    def _dot(a, b) -> float:
                        return sum(float(x) * float(y) for x, y in zip(a, b))

                    q_sims = [_dot(q_vec, dv) for dv in doc_vecs]
                    base_scores = [_base_priority(c, qs) for c, qs in zip(cards, q_sims)]
                    remaining = list(range(len(cards)))
                    selected_indices: List[int] = []

                    while remaining and len(selected_indices) < top_k:
                        if not selected_indices:
                            best_idx = max(remaining, key=lambda idx: base_scores[idx])
                        else:
                            best_idx = max(
                                remaining,
                                key=lambda idx: (
                                    lambda_param * base_scores[idx]
                                    - (1.0 - lambda_param) * max(_dot(doc_vecs[idx], doc_vecs[s]) for s in selected_indices)
                                ),
                            )
                        selected_indices.append(best_idx)
                        remaining.remove(best_idx)
                    selected = [cards[i] for i in selected_indices]
            except Exception:
                selected = []

        if not selected:
            ranked = sorted(cards, key=lambda c: _base_priority(c), reverse=True)
            selected = ranked[:top_k]

        # 按纵向演进脉络排序：本地精读在前，在线文献按发表年份从旧到新排列，方便模型识别“背景->突破->前沿”演进链
        local_cards = [c for c in selected if c.is_local_indexed]
        online_cards = [c for c in selected if not c.is_local_indexed]
        online_cards.sort(key=lambda c: (c.year if isinstance(c.year, int) else 9999, -float(c.score)))
        return local_cards + online_cards

    @staticmethod
    def _resolve_local_doc_year(doc_name: str, chunk_meta: Dict[str, Any]) -> Optional[int]:
        """解析本地文献的真实发表年份 (优先侧车元数据/文件名/arXiv号，拒绝盲目填充当前年份)"""
        # 1. 切片自带年份
        if chunk_meta.get("year"):
            try:
                return int(chunk_meta["year"])
            except Exception:
                pass
        # 2. 检查 uploads 目录下的下载论文侧车元数据
        try:
            from online_paper_search_agent import UPLOADS_DIR
            stem = Path(doc_name).stem
            meta_path = UPLOADS_DIR / f"{stem}.json"
            if meta_path.exists():
                meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_dict.get("year"):
                    return int(meta_dict["year"])
        except Exception:
            pass
        # 3. 从文件名提取 arXiv 年份 (如 2403.xxxx -> 2024, 2501.xxxx -> 2025)
        m_arxiv = re.search(r'\b(2[0-9])(0[1-9]|1[0-2])\.[0-9]{4,5}\b', doc_name)
        if m_arxiv:
            return 2000 + int(m_arxiv.group(1))
        # 4. 从文件名提取合理出版年份 (1950 - 2030)
        m_yr = re.search(r'\b(19[5-9][0-9]|20[0-3][0-9])\b', doc_name)
        if m_yr:
            return int(m_yr.group(1))
        return None

    @classmethod
    def filter_local_dates(cls, chunks, time_constraint):
        bounds = retrieval_date_bounds(resolve_retrieval_time_constraint(time_constraint))
        filtered = []
        for raw in chunks:
            item = dict(raw)
            name = item.get("doc_name", "")
            year = cls._resolve_local_doc_year(name, item)
            published = cls._resolve_local_publication_date(name, item)
            if published:
                year = int(published[:4])
            match = None
            if bounds:
                start, end = bounds
                if published and not start <= published <= end:
                    continue
                if not published and year is not None and not int(start[:4]) <= year <= int(end[:4]):
                    continue
                match = ("within_range" if published or year is not None and
                    start <= f"{year:04d}-01-01" and f"{year:04d}-12-31" <= end
                    else "requires_date_verification")
            filtered.append({**item, "year": year, "published_at": published, "temporal_match": match})
        return filtered

    def _fetch_local_evidence(
        self,
        query: str,
        negative_keywords: Optional[List[str]],
        time_constraint: Optional[Dict[str, Any]] = None,
        limit: int = 8,
        discipline_filter: Optional[str] = None,
    ) -> List[UnifiedEvidenceCard]:
        """获取本地精准切片并转换为统一卡片 (基于真实年份严格执行时效过滤，支持同文献多页召回)"""
        if not self.rag_engine:
            return []
        try:
            time_constraint = resolve_retrieval_time_constraint(time_constraint)
            candidate_k = max(25, limit * 3)
            chunks = self.rag_engine.search(
                query,
                top_k=limit,
                candidate_k=candidate_k,
                discipline_filter=discipline_filter,
                negative_keywords=negative_keywords,
            )
            cards = []
            for c in self.filter_local_dates(chunks, time_constraint):
                text = c.get("text", "")
                if self._is_low_info_toc_snippet(text):
                    continue
                doc_name = c.get("doc_name", "本地文献")
                doc_year = c.get("year")
                published_at = c.get("published_at")
                temporal_match = c.get("temporal_match")

                cards.append(UnifiedEvidenceCard(
                    doc_name=doc_name,
                    doc_type="local_pdf",
                    text=text,
                    score=c.get("score", 0.8),
                    year=doc_year,
                    location_label=c.get("location_label", "本地切片"),
                    section=c.get("section", ""),
                    page_or_slide=c.get("page_or_slide"),
                    is_local_indexed=True,
                    doi=c.get("doi"),
                    arxiv_id=c.get("arxiv_id"),
                    summary=c.get("summary", ""),
                    published_at=published_at,
                    temporal_match=temporal_match,
                ))
            return cards
        except Exception as e:
            print(f"⚠️ [LocalEvidenceFetch Note]: {e}")
            return []

    @staticmethod
    def _is_low_info_toc_snippet(text: str) -> bool:
        """过滤纯目录/页码引导线切片（如 '11.2.3 多智能体系统的构建 . . . 268'），避免低信息密度索引页污染主证据。"""
        raw = (text or "").strip()
        if not raw:
            return False
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return False
        toc_pat = re.compile(r"(?:\.\s*){4,}\s*\d+\s*$|…{2,}\s*\d+\s*$")
        non_toc = [ln for ln in lines if not toc_pat.search(ln)]
        if not non_toc:
            return True
        return len(lines) - len(non_toc) >= 2 and sum(len(ln) for ln in non_toc) < 30

    @staticmethod
    def _resolve_local_publication_date(doc_name: str, chunk_meta: Dict[str, Any]) -> Optional[str]:
        """Use publication metadata only; ingestion/download timestamps are not publication dates."""
        metadata = [chunk_meta]
        try:
            from online_paper_search_agent import UPLOADS_DIR
            sidecar = UPLOADS_DIR / f"{Path(doc_name).stem}.json"
            if sidecar.exists():
                metadata.append(json.loads(sidecar.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        for item in metadata:
            for key in ("published_at", "publication_date", "published_date"):
                try:
                    return date.fromisoformat(str(item.get(key) or "")[:10]).isoformat()
                except ValueError:
                    continue
        return None

    def _fetch_online_evidence(
        self,
        query: str,
        time_constraint: Optional[Dict[str, Any]] = None,
        english_query: Optional[str] = None,
        include_github: bool = True,
        include_web: bool = False,
        include_datasets: bool = False,
    ) -> List[UnifiedEvidenceCard]:
        """获取线上多源资产 (arXiv, CrossRef, GitHub) 并转换为统一卡片"""
        if not self.paper_agent:
            return []
        try:
            bundle = self.paper_agent.search_multisource(
                query,
                include_github=include_github,
                include_web=include_web,
                include_datasets=include_datasets,
                time_constraint=time_constraint,
                english_query=english_query,
            )
            self.last_source_errors = dict(bundle.get("source_errors") or {})
            self.last_source_counts = {
                key: len(bundle.get(key) or []) for key in
                ("arxiv_papers", "openalex_papers", "crossref_papers", "github_repos", "web_results", "datasets")
            }
            cards = []

            # 1. arXiv
            for p in bundle.get("arxiv_papers", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=p.get("title", ""),
                    doc_type="arxiv",
                    text=f"【arXiv 论文题目】: {p.get('title')}\n【主要作者】: {', '.join(p.get('authors', [])[:3])}\n【发表年份】: {p.get('year')}\n【核心摘要】: {p.get('summary')}\n【PDF链接】: {p.get('pdf_url')}",
                    score=0.92,
                    year=p.get("year"),
                    location_label=f"arXiv:{p.get('paper_id')} ({p.get('year')}年)",
                    section="预印本摘要",
                    arxiv_id=p.get("paper_id"),
                    pdf_url=p.get("pdf_url"),
                    authors=p.get("authors", []),
                    summary=p.get("summary", ""),
                ))

            # 2. OpenAlex
            for op in bundle.get("openalex_papers", []):
                op_level = op.get("evidence_level")
                op_score = 0.91 if op_level == "abstract" else 0.68
                cards.append(UnifiedEvidenceCard(
                    doc_name=op.get("title", ""),
                    doc_type="openalex",
                    text=f"【OpenAlex 期刊/会议】: {op.get('title')}\n【主要作者】: {', '.join(op.get('authors', [])[:3])}\n【发表年份】: {op.get('year')}\n【出版/收录】: {op.get('summary')}\n【URL】: {op.get('url')}",
                    score=op_score,
                    year=op.get("year"),
                    location_label=f"OpenAlex ({op.get('year')}年)",
                    section="OpenAlex 学术索引",
                    doi=op.get("doi"),
                    pdf_url=op.get("url"),
                    authors=op.get("authors", []),
                    summary=op.get("summary", ""),
                    evidence_level=op_level,
                    content_scope=op.get("content_scope"),
                    published_at=op.get("published_at"),
                ))

            # 3. CrossRef
            for c in bundle.get("crossref_papers", []):
                c_title = (c.get("title") or "").strip()
                if c_title.startswith("语料："):
                    continue
                c_level = c.get("evidence_level")
                c_score = 0.90 if c_level == "abstract" else 0.64
                cards.append(UnifiedEvidenceCard(
                    doc_name=c_title,
                    doc_type="crossref",
                    text=f"【CrossRef 收录】: {c_title}\n【作者团队】: {', '.join(c.get('authors', [])[:3])}\n【来源信息】: {c.get('summary')}\n【DOI链接】: {c.get('url')}",
                    score=c_score,
                    year=c.get("year"),
                    location_label=f"DOI 出版文献 ({c.get('year')}年)",
                    section="DOI 已出版文献",
                    doi=c.get("doi"),
                    pdf_url=c.get("url"),
                    authors=op_authors if (op_authors := c.get("authors", [])) else [],
                    summary=c.get("summary", ""),
                    evidence_level=c_level,
                    content_scope=c.get("content_scope"),
                    published_at=c.get("published_at"),
                ))

            # 4. GitHub
            for g in bundle.get("github_repos", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=f"GitHub 开源仓库: {g.get('id')}",
                    doc_type="github",
                    text=f"【开源仓库】: {g.get('title')}\n【技术简介】: {g.get('summary')}\n【项目链接】: {g.get('url')}",
                    score=0.88,
                    year=None,
                    location_label=f"GitHub ⭐{g.get('stars', 0)}",
                    section="开源代码仓库",
                    github_url=g.get("url"),
                    stars=g.get("stars", 0),
                    summary=g.get("summary", ""),
                    evidence_level=g.get("evidence_level", "repository_metadata"),
                    content_scope=g.get("content_scope", "仓库元数据与说明；未执行源码验证"),
                    version_ref=g.get("version_ref"),
                    version_ref_type=g.get("version_ref_type"),
                ))

            for result in bundle.get("web_results", []):
                ev_level = result.get("evidence_level", "search_snippet")
                if ev_level == "web_commentary":
                    loc_label = result.get("location_label") or "网页/社区解读 (web_commentary)"
                    sec_label = "网页/社区解读"
                elif ev_level == "search_snippet":
                    loc_label = "网页检索摘要"
                    sec_label = "网页检索结果"
                else:
                    loc_label = "公开网页正文"
                    sec_label = "网页检索结果"
                cards.append(UnifiedEvidenceCard(
                    doc_name=result.get("title", "网页检索结果"), doc_type="web",
                    text=result.get("summary", ""), score=0.72,
                    location_label=loc_label,
                    section=sec_label,
                    pdf_url=result.get("url"), summary=result.get("summary", ""),
                    evidence_level=ev_level,
                    content_scope=result.get("content_scope", "网页检索摘要片段"),
                    arxiv_id=result.get("arxiv_id"),
                ))
            for result in bundle.get("datasets", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=result.get("title", "公开数据集"), doc_type="dataset",
                    text=result.get("summary", ""), score=0.78,
                    location_label="Hugging Face 数据集元数据",
                    section="数据集元数据", pdf_url=result.get("url"),
                    summary=result.get("summary", ""),
                    evidence_level=result.get("evidence_level", "dataset_repository_metadata"),
                    content_scope="公开数据集仓库元数据（未下载原始数据文件）",
                ))

            return cards
        except Exception as e:
            print(f"⚠️ [OnlineEvidenceFetch Note]: {e}")
            return []

    def _deduplicate_and_merge(
        self,
        candidates: List[UnifiedEvidenceCard],
        similarity_threshold: float = 0.82,
        max_chunks_per_doc: int = 4,
    ) -> List[UnifiedEvidenceCard]:
        """
        核心跨源去重与融合算法：
        1. 按照 DOI / arXiv ID 强对齐
        2. 按照 Title 归一化语义相似度 (Jaccard + SequenceMatcher >= 0.82) 模糊对齐
        3. 关键机制（同文献多页深度互补保护）：
           - 若两张卡片均来自【本地同一文献】，但位于【不同页码/切片】（如第 2 页与第 17 页），
             不作为重复文献折叠，保留为互补证据（单篇文献保留上限受 max_chunks_per_doc 控制）。
           - 若两张卡片来自【不同源】（如本地 PDF 与线上 arXiv/DOI/CrossRef），执行跨源元数据深度融合升级为超级节点。
           - 若两张卡片【同源且同页码/位置】或【线上重复抓取】，执行正文与得分去重合并。
        """
        if not candidates:
            return []

        merged_cards: List[UnifiedEvidenceCard] = []
        doc_chunk_counts: Dict[str, int] = {}

        for cand in candidates:
            matched_index = -1
            cand_norm_title = normalize_title(cand.doc_name)

            for idx, existing in enumerate(merged_cards):
                # 1. 判定是否指向同一篇学术文献实体
                is_same_paper = False
                if cand.arxiv_id and existing.arxiv_id and cand.arxiv_id.lower().strip() == existing.arxiv_id.lower().strip():
                    is_same_paper = True
                elif cand.doi and existing.doi and cand.doi.lower().strip() == existing.doi.lower().strip():
                    is_same_paper = True
                else:
                    sim = calculate_title_similarity(cand.doc_name, existing.doc_name)
                    if sim >= similarity_threshold:
                        is_same_paper = True

                if is_same_paper:
                    # 场景 A: 双方均是本地已索引的切片
                    if cand.is_local_indexed and existing.is_local_indexed:
                        same_loc = False
                        if cand.location_label and existing.location_label and cand.location_label == existing.location_label:
                            same_loc = True
                        elif cand.page_or_slide is not None and existing.page_or_slide is not None and cand.page_or_slide == existing.page_or_slide:
                            same_loc = True

                        if same_loc:
                            # 真正同页/同位置的重复切片，合并正文与得分
                            matched_index = idx
                            break
                        else:
                            # 同篇文献但属于不同页码/不同章节（如第 2 页引言 vs 第 17 页实证），互为补充，不作为重复项折叠
                            continue

                    # 场景 B: 跨源（一本地一线上）或双方均线上（arXiv vs CrossRef）
                    else:
                        matched_index = idx
                        break

            if matched_index >= 0:
                self._merge_two_cards(merged_cards[matched_index], cand)
            else:
                # 检查单文献配额上限，防止同一篇长文献无限制霸占所有卡片槽位
                current_cnt = doc_chunk_counts.get(cand_norm_title, 0)
                if cand.is_local_indexed:
                    if current_cnt < max_chunks_per_doc:
                        merged_cards.append(cand)
                        if cand_norm_title:
                            doc_chunk_counts[cand_norm_title] = current_cnt + 1
                else:
                    merged_cards.append(cand)
                    if cand_norm_title:
                        doc_chunk_counts[cand_norm_title] = current_cnt + 1

        return merged_cards

    def _merge_two_cards(self, base: UnifiedEvidenceCard, incoming: UnifiedEvidenceCard) -> None:
        """Keep all inspected passages while merging source identity and provenance."""
        from workflow_contracts import merge_text
        was_local = base.is_local_indexed
        base.doc_type = "fused_academic"
        base.sources_merged = list(dict.fromkeys([*base.sources_merged, *incoming.sources_merged]))
        for field in ("arxiv_id", "doi", "pdf_url", "github_url", "stars", "authors", "year",
                      "published_at", "temporal_match", "version_ref", "version_ref_type"):
            if not getattr(base, field) and getattr(incoming, field):
                setattr(base, field, getattr(incoming, field))
        if incoming.is_local_indexed and not was_local:
            base.text, _ = merge_text(incoming.text, base.text)
            base.location_label = incoming.location_label
            base.page_or_slide = incoming.page_or_slide
        else:
            base.text, _ = merge_text(base.text, incoming.text)
        base.summary, _ = merge_text(base.summary, incoming.summary)
        base.content_scope, _ = merge_text(base.content_scope, incoming.content_scope)
        base.is_local_indexed = was_local or incoming.is_local_indexed
        if base.is_local_indexed:
            base.evidence_level = "local_snippet"
        elif incoming.evidence_level == "abstract" or base.evidence_level == "abstract":
            base.evidence_level = "abstract"
        elif not base.evidence_level:
            base.evidence_level = incoming.evidence_level
        base.score = max(base.score, incoming.score) + 0.05
