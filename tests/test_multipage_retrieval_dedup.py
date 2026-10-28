import pytest
from cross_source_hybrid_retriever import (
    CrossSourceHybridRetriever,
    UnifiedEvidenceCard,
    calculate_title_similarity,
)


def test_deduplicate_preserves_different_pages_of_same_document():
    """测试：同篇本地文献的不同页码切片（如引言、模型、实证检验）互为补充，不作为重复文献粗暴折叠"""
    retriever = CrossSourceHybridRetriever()

    candidates = [
        UnifiedEvidenceCard(
            doc_name="象征性还是实质性：合法性压力下的企业ESG行动策略研究_李雅文.pdf",
            doc_type="local_pdf",
            text="【表11】多项式回归与响应面分析结果，b1=0.235, b2=-0.112...",
            score=1.248,
            location_label="第 17 页",
            page_or_slide=17,
            is_local_indexed=True,
        ),
        UnifiedEvidenceCard(
            doc_name="象征性还是实质性：合法性压力下的企业ESG行动策略研究_李雅文.pdf",
            doc_type="local_pdf",
            text="构建二次多项式回归模型：Y = b0 + b1*X + b2*M + b3*X^2 + ...",
            score=1.120,
            location_label="第 16 页",
            page_or_slide=16,
            is_local_indexed=True,
        ),
        UnifiedEvidenceCard(
            doc_name="象征性还是实质性：合法性压力下的企业ESG行动策略研究_李雅文.pdf",
            doc_type="local_pdf",
            text="【假说提出】H1: 合法性压力促使企业采取象征性ESG策略...",
            score=0.980,
            location_label="第 2 页",
            page_or_slide=2,
            is_local_indexed=True,
        ),
        UnifiedEvidenceCard(
            doc_name="象征性还是实质性：合法性压力下的企业ESG行动策略研究_李雅文.pdf",
            doc_type="local_pdf",
            text="稳健性检验与结论讨论...",
            score=0.950,
            location_label="第 18 页",
            page_or_slide=18,
            is_local_indexed=True,
        ),
        # 相同页码的重复抓取（真正需要去重的切片）
        UnifiedEvidenceCard(
            doc_name="象征性还是实质性：合法性压力下的企业ESG行动策略研究_李雅文.pdf",
            doc_type="local_pdf",
            text="【表11】补充说明：显著性水平在1%下通过...",
            score=1.150,
            location_label="第 17 页",
            page_or_slide=17,
            is_local_indexed=True,
        ),
    ]

    deduped = retriever._deduplicate_and_merge(candidates, max_chunks_per_doc=4)

    # 验证：不同页码均被保留（17, 16, 2, 18），同页的第 17 页去重合并，最终应当有 4 项
    assert len(deduped) == 4

    pages = [c.page_or_slide for c in deduped]
    assert 17 in pages
    assert 16 in pages
    assert 2 in pages
    assert 18 in pages

    # 验证第 17 页的合并正文互补
    page_17_card = next(c for c in deduped if c.page_or_slide == 17)
    assert "显著性水平在1%下通过" in page_17_card.text
    assert "多项式回归与响应面分析结果" in page_17_card.text


def test_cross_source_fusion_with_online():
    """测试：线上文献与本地文献依然保持跨源融合"""
    retriever = CrossSourceHybridRetriever()

    candidates = [
        UnifiedEvidenceCard(
            doc_name="Quantum Simulation of Gauge Theories.pdf",
            doc_type="local_pdf",
            text="本地全文正文：我们推导了哈密顿量...",
            score=1.0,
            location_label="第 5 页",
            page_or_slide=5,
            is_local_indexed=True,
            doi="10.1103/PhysRevD.100.012345",
        ),
        UnifiedEvidenceCard(
            doc_name="Quantum Simulation of Gauge Theories",
            doc_type="crossref",
            text="线上 CrossRef 出版物元数据...",
            score=0.9,
            location_label="CrossRef 2024",
            is_local_indexed=False,
            doi="10.1103/PhysRevD.100.012345",
            pdf_url="https://doi.org/10.1103/PhysRevD.100.012345",
        ),
    ]

    deduped = retriever._deduplicate_and_merge(candidates)
    assert len(deduped) == 1
    assert deduped[0].is_local_indexed is True
    assert deduped[0].pdf_url == "https://doi.org/10.1103/PhysRevD.100.012345"
