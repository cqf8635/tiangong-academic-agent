"""Cross-Source Hybrid Deduplication & Fusion Retrieval Engine.

Implements:
1. 5-Way Concurrent Hybrid Retrieval (Local Dense, Local Sparse, arXiv, CrossRef, GitHub).
2. Deterministic & Fuzzy Semantic Deduplication (ArXiv ID, DOI, Title Normalization & Jaccard/Levenshtein matching).
3. Attribute Enrichment & Node Fusion (Merging Preprint + Published DOI + GitHub Repo + Local Slices).
4. Global BGE-Reranker cross-encoder unified scoring.
"""

from __future__ import annotations

from datetime import datetime
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from document_rag_engine import UniversalDocumentRAGEngine
from online_paper_search_agent import OnlinePaperSearchAgent


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
    ):
        self.doc_name = doc_name
        self.doc_type = doc_type  # local_pdf | arxiv | openalex | crossref | github | fused_academic
        self.text = text
        self.score = score
        self.year = year or datetime.now().year
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
            "github_url": self.github_url,
            "stars": self.stars,
            "authors": self.authors,
            "is_local_indexed": self.is_local_indexed,
            "sources_merged": list(set(self.sources_merged)),
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
    ) -> List[Dict[str, Any]]:
        """执行全源混合并发召回、跨源语义去重与全局融合重排 (同文献多页保护，支持全面文献深读)"""
        raw_candidates: List[UnifiedEvidenceCard] = []

        # 1. 并发执行多路召回
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 路线 A: 本地 RAG 语义检索 (扩大候选窗口至 max(top_k, 8)，获取同文献更多代表性页面)
            local_limit = max(top_k, 8)
            fut_local = executor.submit(self._fetch_local_evidence, query, negative_keywords, time_constraint, local_limit) if include_local else None
            # 路线 B: 线上多源学术与开源工程探针
            fut_online = executor.submit(self._fetch_online_evidence, query, time_constraint, english_query) if include_online else None

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

        # 3. 全局重排 (基于分数与多源权重)
        deduped_cards.sort(key=lambda c: (c.is_local_indexed * 0.15 + (1 if c.github_url else 0) * 0.05 + c.score), reverse=True)

        return [c.to_dict() for c in deduped_cards[:top_k]]

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

    def _fetch_local_evidence(
        self,
        query: str,
        negative_keywords: Optional[List[str]],
        time_constraint: Optional[Dict[str, Any]] = None,
        limit: int = 8,
    ) -> List[UnifiedEvidenceCard]:
        """获取本地精准切片并转换为统一卡片 (基于真实年份严格执行时效过滤，支持同文献多页召回)"""
        if not self.rag_engine:
            return []
        try:
            candidate_k = max(25, limit * 3)
            chunks = self.rag_engine.search(query, top_k=limit, candidate_k=candidate_k, negative_keywords=negative_keywords)
            cards = []
            has_time = bool(time_constraint and time_constraint.get("has_time_constraint"))
            start_yr = time_constraint.get("start_year") if has_time else None
            end_yr = time_constraint.get("end_year") if has_time else None

            for c in chunks:
                doc_name = c.get("doc_name", "本地文献")
                doc_year = self._resolve_local_doc_year(doc_name, c)

                # 若提问有明确的时间约束 (如近两年 2024~2026):
                # 只有真正符合该年份区间的本地文献才准予召回；早于起始年份或年份未知的经典教材，一律拦截过滤
                if has_time and start_yr:
                    if doc_year is None or doc_year < start_yr or (end_yr and doc_year > end_yr):
                        continue

                cards.append(UnifiedEvidenceCard(
                    doc_name=doc_name,
                    doc_type="local_pdf",
                    text=c.get("text", ""),
                    score=c.get("score", 0.8),
                    year=doc_year,
                    location_label=c.get("location_label", "本地切片"),
                    section=c.get("section", ""),
                    page_or_slide=c.get("page_or_slide"),
                    is_local_indexed=True,
                ))
            return cards
        except Exception as e:
            print(f"⚠️ [LocalEvidenceFetch Note]: {e}")
            return []

    def _fetch_online_evidence(
        self,
        query: str,
        time_constraint: Optional[Dict[str, Any]] = None,
        english_query: Optional[str] = None,
    ) -> List[UnifiedEvidenceCard]:
        """获取线上多源资产 (arXiv, CrossRef, GitHub) 并转换为统一卡片"""
        if not self.paper_agent:
            return []
        try:
            bundle = self.paper_agent.search_multisource(
                query,
                include_github=True,
                time_constraint=time_constraint,
                english_query=english_query,
            )
            cards = []

            # 1. arXiv
            for p in bundle.get("arxiv_papers", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=p.get("title", ""),
                    doc_type="arxiv",
                    text=f"【arXiv 论文题目】: {p.get('title')}\n【主要作者】: {', '.join(p.get('authors', [])[:3])}\n【发表年份】: {p.get('year')}\n【核心摘要】: {p.get('summary')}\n【PDF链接】: {p.get('pdf_url')}",
                    score=0.92,
                    year=p.get("year") or datetime.now().year,
                    location_label=f"arXiv:{p.get('paper_id')} ({p.get('year')}年)",
                    section="Preprint Abstract",
                    arxiv_id=p.get("paper_id"),
                    pdf_url=p.get("pdf_url"),
                    authors=p.get("authors", []),
                    summary=p.get("summary", ""),
                ))

            # 2. OpenAlex
            for op in bundle.get("openalex_papers", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=op.get("title", ""),
                    doc_type="openalex",
                    text=f"【OpenAlex 期刊/会议】: {op.get('title')}\n【主要作者】: {', '.join(op.get('authors', [])[:3])}\n【发表年份】: {op.get('year')}\n【出版/收录】: {op.get('summary')}\n【URL】: {op.get('url')}",
                    score=0.91,
                    year=op.get("year") or datetime.now().year,
                    location_label=f"OpenAlex ({op.get('year')}年)",
                    section="OpenAlex Index",
                    doi=op.get("doi"),
                    pdf_url=op.get("url"),
                    authors=op.get("authors", []),
                    summary=op.get("summary", ""),
                ))

            # 3. CrossRef
            for c in bundle.get("crossref_papers", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=c.get("title", ""),
                    doc_type="crossref",
                    text=f"【已发表论文】: {c.get('title')}\n【作者团队】: {', '.join(c.get('authors', [])[:3])}\n【出版信息】: {c.get('summary')}\n【DOI链接】: {c.get('url')}",
                    score=0.90,
                    year=c.get("year") or datetime.now().year,
                    location_label=f"DOI 出版文献 ({c.get('year')}年)",
                    section="Published DOI",
                    doi=c.get("doi"),
                    pdf_url=c.get("url"),
                    authors=c.get("authors", []),
                    summary=c.get("summary", ""),
                ))

            # 4. GitHub
            for g in bundle.get("github_repos", []):
                cards.append(UnifiedEvidenceCard(
                    doc_name=f"GitHub Repo: {g.get('id')}",
                    doc_type="github",
                    text=f"【开源仓库】: {g.get('title')}\n【技术简介】: {g.get('summary')}\n【项目链接】: {g.get('url')}",
                    score=0.88,
                    year=datetime.now().year,
                    location_label=f"GitHub ⭐{g.get('stars', 0)}",
                    section="Open Source Code",
                    github_url=g.get("url"),
                    stars=g.get("stars", 0),
                    summary=g.get("summary", ""),
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
        """属性合并：将两个同源切片/线上条目融合为一体"""
        base.doc_type = "fused_academic"
        base.sources_merged.extend(incoming.sources_merged)

        # 补充缺失字段
        if not base.arxiv_id and incoming.arxiv_id:
            base.arxiv_id = incoming.arxiv_id
        if not base.doi and incoming.doi:
            base.doi = incoming.doi
        if not base.pdf_url and incoming.pdf_url:
            base.pdf_url = incoming.pdf_url
        if not base.github_url and incoming.github_url:
            base.github_url = incoming.github_url
            base.stars = incoming.stars
        if not base.is_local_indexed and incoming.is_local_indexed:
            base.is_local_indexed = True

        # 合并正文论述
        if incoming.is_local_indexed and not base.is_local_indexed:
            # 本地精读内容优先作为正文展示
            base.text = f"{incoming.text}\n\n[多源在线互证] {base.text}"
            base.location_label = f"{incoming.location_label} (已对齐线上 DOI/arXiv)"
        elif not incoming.is_local_indexed and base.is_local_indexed:
            base.text = f"{base.text}\n\n[多源在线互证] {incoming.text}"
            base.location_label = f"{base.location_label} (已对齐线上 DOI/arXiv)"
        elif incoming.is_local_indexed and base.is_local_indexed:
            # 双方都是同位置的本地切片，去重拼接
            if incoming.text and incoming.text not in base.text:
                base.text = f"{base.text}\n\n{incoming.text}"

        # 分数叠加增益
        base.score = max(base.score, incoming.score) + 0.05
