import pytest
from pathlib import Path
from document_rag_engine import UniversalDocumentParser


def test_is_heading_or_topic_boundary():
    """验证学术标题与主题边界识别器（0 硬编码，涵盖中英文标准规范）"""
    # 1. 中文一级/二级标题
    assert UniversalDocumentParser._is_heading_or_topic_boundary("一、引言与研究背景") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("（二）变量测度与数据说明") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("第三章 实证分析") is not None

    # 2. 阿拉伯数字层级标题
    assert UniversalDocumentParser._is_heading_or_topic_boundary("1. Introduction") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("2.1 研究假说提出") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("3.2.1 响应面多项式回归模型") is not None

    # 3. 关键图表标题
    assert UniversalDocumentParser._is_heading_or_topic_boundary("表 11 多项式回归与响应面分析结果") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("图 3 响应面三维分析曲面图") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("Table 2: Regression Estimates") is not None

    # 4. 英文标准学术小节
    assert UniversalDocumentParser._is_heading_or_topic_boundary("Methodology") is not None
    assert UniversalDocumentParser._is_heading_or_topic_boundary("Robustness Check") is not None

    # 5. 普通正文长句不应误判为标题
    normal_text = "随着经济高质量发展的推进，企业在合法性压力下如何采取ESG行动成为学术界关注的重点问题。"
    assert UniversalDocumentParser._is_heading_or_topic_boundary(normal_text) is None


def test_split_long_paragraph_preserves_sentences():
    """验证超长段落按自然句末标点平滑切分，绝不截断在词语中间"""
    s1 = "这是第一个完整的句子，阐述了假说H1的理论推导。"
    s2 = "这是第二个完整的句子，进一步讨论了实质性与象征性策略的异同。"
    s3 = "这是第三个完整的句子，给出了回归模型的边界条件。"
    long_p = s1 + s2 + s3

    # 当 max_chars 较小时，应在标点处拆开，且拼起来依然完整
    sub_parts = UniversalDocumentParser._split_long_paragraph(long_p, max_chars=len(s1) + len(s2) - 5)
    assert len(sub_parts) >= 2
    for part in sub_parts:
        assert part.endswith("。") or part.endswith("？") or part.endswith("！") or part == sub_parts[-1]


def test_parse_text_or_md_heading_and_paragraph_chunking(tmp_path):
    """测试 Markdown/纯文本按主题标题和自然段落切片"""
    sample_text = """
一、引言与研究背景

随着经济高质量发展的推进，企业在合法性压力下如何采取ESG行动成为学术界关注的重点问题。本文基于组织合法性理论，探讨中国上市公司的策略选择。

二、理论假说

【假说1】在较强合法性压力下，企业倾向于优先采取象征性ESG行动以快速满足监管要求。

【假说2】长期而言，随着利益相关者监督深化，实质性行动将逐步取代象征性行动。

三、研究设计

表 1 变量定义与测度说明
被解释变量为企业实质性ESG得分与象征性ESG得分，解释变量为区域合法性压力指数。
"""
    test_file = tmp_path / "sample_paper.txt"
    test_file.write_text(sample_text.strip(), encoding="utf-8")

    chunks = UniversalDocumentParser.parse_text_or_md(test_file, max_chunk_chars=300)

    # 验证切出的切片带有具体的 section 标题
    sections = [c.section for c in chunks]
    assert any("引言" in s for s in sections)
    assert any("假说" in s for s in sections)
    assert any("研究设计" in s or "表 1" in s for s in sections)
