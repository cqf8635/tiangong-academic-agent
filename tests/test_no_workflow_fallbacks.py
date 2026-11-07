"""Offline failure-boundary checks: no model loading, network, or paid API calls."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import master_agent_workflow as workflow
from evidence_scoring import score_evidence
from intent_recognition_agent import IntentDecision, IntentParsingError
from online_paper_search_agent import OnlinePaperSearchAgent
from workflow_errors import WorkflowExecutionError


@pytest.mark.parametrize('failure', [False, True])
def test_empty_or_failed_retrieval_stops_graph_without_local_substitution(monkeypatch, failure):
    parser = Mock()
    parser.parse_intent.return_value = IntentDecision(
        domain='统计', intent='compare_contrast', target_agent='ComparisonAgent',
        subject_action='INIT', subject_entity='估计方法', search_query='estimator comparison',
        need_online_paper_search=True, required_data_sources=['arxiv_live_search'],
    )
    hybrid = Mock()
    hybrid.hybrid_search_and_deduplicate.return_value = []
    if failure:
        hybrid.hybrid_search_and_deduplicate.side_effect = RuntimeError('source unavailable')
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_intent_agent', lambda: parser)
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_hybrid_retriever', lambda: hybrid)
    forbidden = Mock(side_effect=AssertionError('retrieval failed: downstream must not run'))
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_rag_engine', forbidden)
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_llm_client', forbidden)
    with pytest.raises(WorkflowExecutionError) as error:
        workflow.build_master_workflow(checkpointer=None).invoke({
            'user_query': '比较估计方法', 'recent_paper_candidates': [], 'retrieved_evidence': [],
        })
    assert error.value.stage == 'retrieval'
    forbidden.assert_not_called()


def test_reranking_failure_never_switches_to_embeddings(monkeypatch):
    monkeypatch.setenv('PAPER_USE_RERANKER', '1')
    engine = SimpleNamespace(reranker=Mock(), embedding_model=Mock())
    engine.reranker.predict.side_effect = RuntimeError('reranker unavailable')
    with pytest.raises(RuntimeError, match='重排失败'):
        score_evidence('query', [{'text': 'evidence'}], engine)
    engine.embedding_model.encode.assert_not_called()


def test_missing_semantic_model_does_not_use_lexical_scores(monkeypatch):
    monkeypatch.setenv('PAPER_USE_RERANKER', '0')
    with pytest.raises(RuntimeError, match='语义评分模型不可用'):
        score_evidence('query', [{'text': 'query'}])


def test_multisource_uses_supplied_query_without_model_rewrite():
    agent = OnlinePaperSearchAgent.__new__(OnlinePaperSearchAgent)
    agent.extract_academic_keywords = Mock(side_effect=AssertionError('no extra LLM'))
    for name in ('search_arxiv', 'search_openalex', 'search_crossref', 'search_github'):
        setattr(agent, name, Mock(return_value=[]))
    constraint = {'has_time_constraint': True, 'start_year': 2024, 'end_year': 2026}
    result = agent.search_multisource('显式检索词', time_constraint=constraint)
    assert result['total_found'] == 0
    agent.extract_academic_keywords.assert_not_called()
    agent.search_arxiv.assert_called_once_with('显式检索词', 10, constraint)
    agent.search_crossref.assert_called_once_with('显式检索词', 10, constraint)


def test_multisource_exposes_failed_source():
    agent = OnlinePaperSearchAgent.__new__(OnlinePaperSearchAgent)
    for name in ('search_arxiv', 'search_openalex', 'search_crossref', 'search_github'):
        setattr(agent, name, Mock(return_value=[]))
    agent.search_openalex.side_effect = RuntimeError('OpenAlex unavailable')
    with pytest.raises(RuntimeError, match='OpenAlex unavailable'):
        agent.search_multisource('query')


def test_llm_retries_same_model_then_raises(monkeypatch):
    monkeypatch.setenv('OPENAI_FALLBACK_MODEL', 'other-model')
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError('unavailable')
    with pytest.raises(WorkflowExecutionError):
        workflow.safe_chat_completion(client, 'selected-model', [], max_retries=2)
    assert [c.kwargs['model'] for c in client.chat.completions.create.call_args_list] == ['selected-model'] * 2


@pytest.mark.parametrize('finish', [None, 'length', 'content_filter'])
def test_stream_without_successful_finish_is_not_completed(finish):
    client = Mock()
    client.chat.completions.create.return_value = [SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, delta=SimpleNamespace(content='partial response'))])]
    with pytest.raises(WorkflowExecutionError):
        workflow.safe_chat_completion(client, 'selected-model', [], max_retries=1)


@pytest.mark.parametrize('node,stage,flag', [
    (workflow.evidence_synthesis_node, 'evidence_synthesis', 'ENABLE_LLM_EVIDENCE_SYNTHESIS'),
    (workflow.dialectical_inquiry_node, 'dialectical_inquiry', 'ENABLE_LLM_DIALECTICAL_INQUIRY'),
    (workflow.reflective_revision_node, 'reflective_revision', 'ENABLE_SECOND_PASS_REVISION'),
])
def test_review_failure_cannot_continue_with_draft(monkeypatch, node, stage, flag):
    monkeypatch.setenv(flag, 'true')
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_llm_client', lambda: Mock())
    monkeypatch.setattr(workflow, 'safe_chat_completion', Mock(side_effect=RuntimeError('upstream failed')))
    with pytest.raises(WorkflowExecutionError) as error:
        node({'user_query': 'question', 'final_answer': 'valid draft', 'target_agent': 'ComparisonAgent',
              'dialectical_notes': 'review notes', 'retrieved_evidence': [],
              'review_verdict': {'needs_revision': True, 'summary': '需要修正', 'issues': ['推导缺少条件']},
              'stage_statuses': {'dialectical_inquiry': 'completed'}})
    assert error.value.stage == stage


def test_chat_workflow_failure_is_not_archived_as_answer():
    path = Path(__file__).resolve().parents[1] / 'backend_api.py'
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_execute_chat_workflow')
    manager = Mock()
    manager.get_session.return_value = SimpleNamespace(topic_catalog=[])
    manager.get_recent_candidates.return_value = []
    app = Mock()
    app.invoke.side_effect = WorkflowExecutionError('retrieval', 'no evidence')
    namespace = {'ChatRequest': object, 'Dict': dict, 'Any': object, 'SessionManager': manager,
                 'ChatMessage': lambda **kw: SimpleNamespace(**kw), 'MasterAgentState': dict,
                 'master_app': app, 'IntentParsingError': IntentParsingError}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    result = namespace['_execute_chat_workflow'](SimpleNamespace(session_id='test', query='question', image_data=None, voice=None))
    assert result['status'] == 'workflow_failed'
    assert result['stage_statuses']['retrieval'] == 'failed'
    assert result['mindmap_markdown'] is None
    manager.record_turn_topic.assert_not_called()
    manager.update_topic_state.assert_not_called()
    manager.update_recent_candidates.assert_not_called()
    manager.save_session.assert_not_called()
