"""Offline checks: time windows affect online retrieval only."""
from unittest.mock import Mock

import pytest

from cross_source_hybrid_retriever import CrossSourceHybridRetriever
from intent_recognition_agent import IntentDecision
import master_agent_workflow as workflow


WINDOW = {'has_time_constraint': True, 'start_year': 2024, 'end_year': 2026}


@pytest.mark.parametrize('include_online', [False, True])
def test_local_old_unknown_and_newer_documents_survive_online_window(monkeypatch, include_online):
    chunks = [
        {'doc_name': '经典教材', 'year': 1990, 'text': '经典内容', 'score': 0.8},
        {'doc_name': '无年份笔记', 'text': '笔记内容', 'score': 0.7},
        {'doc_name': '较新资料', 'year': 2027, 'text': '较新内容', 'score': 0.6},
    ]
    rag = Mock()
    rag.search.return_value = chunks
    papers = Mock()
    papers.search_multisource.return_value = {}
    retriever = CrossSourceHybridRetriever(rag_engine=rag, paper_agent=papers)
    # Avoid reading actual document sidecars in this offline test.
    monkeypatch.setattr(retriever, '_resolve_local_doc_year', lambda name, chunk: chunk.get('year'))

    results = retriever.hybrid_search_and_deduplicate(
        '内容', include_local=True, include_online=include_online,
        time_constraint=WINDOW, english_query='content')

    assert {result['doc_name'] for result in results} == {chunk['doc_name'] for chunk in chunks}
    assert 'time_constraint' not in rag.search.call_args.kwargs
    if include_online:
        assert papers.search_multisource.call_args.kwargs['time_constraint'] == WINDOW
    else:
        papers.search_multisource.assert_not_called()


@pytest.mark.parametrize('sources', [[], ['uploaded_pdf_docs']])
def test_time_field_alone_does_not_enable_online_search(monkeypatch, sources):
    decision = IntentDecision(
        domain='教材阅读', intent='concept_simplify', target_agent='ConceptTutorAgent',
        subject_action='INIT', subject_entity='概念', search_query='概念',
        required_data_sources=sources, need_online_paper_search=False,
        time_constraint=WINDOW,
    )
    parser = Mock()
    parser.parse_intent.return_value = decision
    hybrid = Mock()
    hybrid.hybrid_search_and_deduplicate.return_value = [
        {'doc_name': '经典教材', 'doc_type': 'local_pdf', 'text': '概念解释', 'score': 0.8}]
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_intent_agent', lambda: parser)
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_hybrid_retriever', lambda: hybrid)

    workflow.intent_routing_node({'user_query': '解释教材概念', 'recent_paper_candidates': []})

    if sources:
        hybrid.hybrid_search_and_deduplicate.assert_called_once()
        assert hybrid.hybrid_search_and_deduplicate.call_args.kwargs['include_online'] is False
        assert hybrid.hybrid_search_and_deduplicate.call_args.kwargs['include_local'] is True
    else:
        hybrid.hybrid_search_and_deduplicate.assert_not_called()
