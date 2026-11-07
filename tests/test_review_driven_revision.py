"""Exercise real review/revision nodes with simulated model replies; no API calls."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import master_agent_workflow as workflow
from intent_recognition_agent import INTENT_DEFINITIONS, IntentDecision
from session_manager import ChatMessage
from workflow_errors import WorkflowExecutionError


def draft_state(answer='原始初稿，保持措辞与 Markdown。'):
    return {
        'user_query': '比较主体 A 与 B，只用给定条件，不扩展任务',
        'intent_decision': {'intent': 'compare_contrast', 'extracted_entities': {
            'objects': ['A', 'B'], 'actions': ['比较'], 'conditions': ['只用给定条件']}},
        'target_agent': 'ComparisonAgent', 'final_answer': answer,
        'retrieved_evidence': [{'doc_name': '原始资料', 'text': '完整资料末尾标记'}],
        'stage_statuses': {'generation': 'completed'},
        'agent_dialogue_history': [],
    }


def verdict(needs_revision):
    return {'needs_revision': needs_revision,
            'summary': '缺少条件解释' if needs_revision else '未发现需修正的问题',
            'issues': ['补充给定条件的作用'] if needs_revision else []}


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_llm_client', lambda: Mock())
    monkeypatch.setattr(workflow, 'get_default_llm_model', lambda: 'configured-model')
    # These obsolete flags must not suppress review or force unconditional revision.
    monkeypatch.setenv('ENABLE_LLM_DIALECTICAL_INQUIRY', 'false')
    monkeypatch.setenv('ENABLE_SECOND_PASS_REVISION', 'false')
    completion = Mock()
    monkeypatch.setattr(workflow, 'safe_chat_completion', completion)
    return completion


@pytest.mark.parametrize('length', [20, 4000])
def test_default_review_keeps_accepted_draft_exactly(model, length):
    state = draft_state('初稿' * length + '\n**末尾不能丢失**')
    model.return_value = json.dumps(verdict(False), ensure_ascii=False)
    reviewed = workflow.dialectical_inquiry_node(state)
    result = workflow.reflective_revision_node({**state, **reviewed})
    assert model.call_count == 1  # Actual review call; no author rewrite.
    assert result['final_answer'] == state['final_answer']
    assert result['stage_statuses']['reflective_revision'] == 'skipped'
    assert reviewed['agent_dialogue_history'][-1]['action'] == 'APPROVE_INITIAL'
    prompt = model.call_args.kwargs['messages'][0]['content']
    assert json.dumps(state["final_answer"], ensure_ascii=False) in prompt
    assert '完整资料末尾标记' in prompt


@pytest.mark.parametrize('length', [20, 4000])
def test_revision_is_triggered_only_by_review_not_length_or_old_flag(model, length):
    state = draft_state('完整初稿' * length + '\n最后一节必须保留')
    model.side_effect = [json.dumps(verdict(True), ensure_ascii=False), '完整修订稿']
    reviewed = workflow.dialectical_inquiry_node(state)
    result = workflow.reflective_revision_node({**state, **reviewed})
    assert model.call_count == 2  # One review plus one requested revision; no hidden re-review.
    assert result['final_answer'] == '完整修订稿'
    assert result['stage_statuses']['reflective_revision'] == 'completed'
    assert result['reflection_notes'] == '已生成修订稿，尚未复核。'
    revision_prompt = model.call_args.kwargs['messages'][0]['content']
    assert '最后一节必须保留' in revision_prompt
    assert '补充给定条件的作用' in revision_prompt
    assert '完整资料末尾标记' in revision_prompt
    assert reviewed['agent_dialogue_history'][-1]['action'] == 'CRITIQUE_AND_REQUIRE_REVISION'
    assert not any(row['action'] == 'CONSENSUS_ACCEPTED' for row in result['agent_dialogue_history'])


@pytest.mark.parametrize('response', [
    '审核通过', '{}', 'null',
    '{"needs_revision":"false","summary":"通过","issues":[]}',
    '{"needs_revision":true,"summary":"需要修改","issues":[]}',
    '{"needs_revision":false,"summary":"通过","issues":["存在错误"]}',
])
def test_invalid_review_stops_instead_of_being_treated_as_approval(model, response):
    model.return_value = response
    with pytest.raises(WorkflowExecutionError) as error:
        workflow.dialectical_inquiry_node(draft_state())
    assert error.value.stage == 'dialectical_inquiry'
    assert model.call_count == 1


def test_revision_rejects_checklist_without_completed_model_review(model):
    state = {**draft_state(), 'peer_review_notes': '固定清单',
             'review_verdict': verdict(True), 'stage_statuses': {'dialectical_inquiry': 'checklist'}}
    with pytest.raises(WorkflowExecutionError, match='真实模型审核未完成'):
        workflow.reflective_revision_node(state)
    model.assert_not_called()


def test_unchanged_revision_does_not_claim_issues_resolved(model):
    state = draft_state()
    model.side_effect = [json.dumps(verdict(True), ensure_ascii=False), state['final_answer']]
    reviewed = workflow.dialectical_inquiry_node(state)
    result = workflow.reflective_revision_node({**state, **reviewed})
    assert result['stage_statuses']['reflective_revision'] == 'unchanged'
    assert '尚未确认解决' in result['reflection_notes']
    assert result['agent_dialogue_history'][-1]['action'] == 'REVISION_UNCHANGED'


@pytest.mark.parametrize('needs_revision', [False, True])
def test_normal_graph_executes_review_before_delivery(monkeypatch, model, needs_revision):
    parser = Mock()
    parser.parse_intent.return_value = IntentDecision(
        domain='通用比较', intent='compare_contrast', target_agent='ComparisonAgent',
        subject_action='INIT', subject_entity='A 与 B', required_data_sources=[], search_query='')
    monkeypatch.setattr(workflow.MasterEngineRegistry, 'get_intent_agent', lambda: parser)
    monkeypatch.setenv('ENABLE_LLM_EVIDENCE_SYNTHESIS', 'false')
    monkeypatch.setattr(workflow, 'general_tutor_node', lambda state: {
        'final_answer': '初稿', 'target_agent': 'ComparisonAgent',
        'stage_statuses': {**state.get('stage_statuses', {}), 'generation': 'completed'}})
    delivery = Mock(return_value={'tts_audio_file': None})
    monkeypatch.setattr(workflow, 'tts_synthesis_node', delivery)
    model.side_effect = [json.dumps(verdict(needs_revision), ensure_ascii=False)] + (
        ['修订后的回答'] if needs_revision else [])
    graph = workflow.build_master_workflow(checkpointer=None)
    result = graph.invoke({'user_query': '比较 A 与 B', 'recent_paper_candidates': [],
                           'retrieved_evidence': [], 'review_verdict': None})
    assert result['review_verdict'] == verdict(needs_revision)
    assert result['stage_statuses']['dialectical_inquiry'] == 'completed'
    assert result['final_answer'] == ('修订后的回答' if needs_revision else '初稿')
    assert model.call_count == (2 if needs_revision else 1)
    delivery.assert_called_once()
    assert delivery.call_args.args[0]['review_verdict'] == verdict(needs_revision)


def test_system_control_does_not_claim_unexecuted_action_completed():
    result = workflow.system_control_node({'user_query': '开启录音'})
    assert result['stage_statuses']['system_control'] == 'not_executed'
    assert '本次没有执行' in result['final_answer']
    assert '已为您执行' not in result['final_answer']


@pytest.mark.parametrize('agent', sorted({item['agent'] for item in INTENT_DEFINITIONS}))
def test_all_supported_experts_have_explicit_routes(agent):
    assert workflow.route_selector({'intent_decision': {'target_agent': agent}}) in {
        'system_control_pipeline', 'online_paper_search_pipeline', 'cross_modal_pipeline',
        'meeting_summary_pipeline', 'document_rag_pipeline', 'general_tutor_pipeline'}


def test_unknown_expert_still_raises_instead_of_using_a_default():
    with pytest.raises(WorkflowExecutionError, match='未知智能体'):
        workflow.route_selector({'intent_decision': {'target_agent': 'UnsupportedAgent'}})


def test_unexecuted_control_is_not_saved_as_a_study_conclusion(monkeypatch):
    # Isolate the real handler without starting the backend or loading its engines.
    path = Path(__file__).resolve().parents[1] / 'backend_api.py'
    handler = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == '_execute_chat_workflow')
    manager = Mock()
    manager.get_session.return_value = SimpleNamespace(topic_catalog=[], messages=[])
    manager.get_recent_candidates.return_value = []
    app = Mock()
    app.invoke.return_value = {**workflow.system_control_node({'user_query': '开启录音'}),
                              'target_agent': 'SystemControlAgent'}
    mindmap = Mock(side_effect=AssertionError('unexecuted controls are not study content'))
    monkeypatch.setattr('document_rag_engine.extract_answer_mindmap', mindmap)
    namespace = {'ChatRequest': object, 'Dict': dict, 'Any': object,
                 'SessionManager': manager, 'ChatMessage': ChatMessage, 'MasterAgentState': dict,
                 'master_app': app, 'IntentParsingError': workflow.IntentParsingError}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), str(path), 'exec'), namespace)
    result = namespace['_execute_chat_workflow'](SimpleNamespace(
        session_id='test', query='开启录音', image_data=None, voice=None))
    assert result['status'] == 'system_control_not_executed'
    assert result['mindmap_markdown'] is None
    manager.record_turn_topic.assert_not_called()
    mindmap.assert_not_called()


def test_review_result_survives_message_serialization():
    message = ChatMessage(sender='agent', content='初稿', review_verdict=verdict(False),
                          stage_statuses={'dialectical_inquiry': 'completed'})
    restored = ChatMessage.model_validate_json(message.model_dump_json())
    assert restored.review_verdict == verdict(False)
