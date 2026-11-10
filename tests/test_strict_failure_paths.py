"""Offline checks for failure paths that must not produce substitute results."""
import ast
import json
import re
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config_utils
import online_paper_search_agent as online
from meeting_summarizer import MeetingSummarizer


def backend_functions(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'backend_api.py'
    names = {'_safe_translation_stem', '_extract_paper_pdf_text', '_translation_json', '_sanitize_academic_text'}
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = {'UPLOAD_DIR': tmp_path, 'PAPER_TRANSLATION_CACHE_DIR': tmp_path,
                 're': re, 'json': json, 'urllib': SimpleNamespace(request=Mock())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


def paper_request(**changes):
    return SimpleNamespace(**{'paper_id': 'test-paper', 'title': 'Test paper',
                              'abstract': 'Abstract is not the full paper.',
                              'text': 'Core thesis', 'pdf_url': '', **changes})


def test_missing_pdf_url_never_returns_abstract_or_legacy_cache(tmp_path):
    namespace = backend_functions(tmp_path)
    (tmp_path / 'test-paper.txt').write_text('old substituted abstract')
    with pytest.raises(ValueError, match='PDF 链接'):
        namespace['_extract_paper_pdf_text'](paper_request())
    namespace['urllib'].request.urlopen.assert_not_called()
    assert not (tmp_path / 'test-paper.body-v5.txt').exists()


@pytest.mark.parametrize('scanned', [False, True])
def test_unextractable_pdf_never_returns_abstract(tmp_path, scanned):
    import pymupdf
    namespace = backend_functions(tmp_path)
    with pymupdf.open() as document:
        page = document.new_page()
        if scanned:
            page.draw_rect(pymupdf.Rect(10, 10, 40, 40))
        document.save(tmp_path / '.paper_translation_test-paper.body-v5.pdf')
    with pytest.raises(ValueError, match='无法提取文字|未提取到正文'):
        namespace['_extract_paper_pdf_text'](paper_request())
    assert not (tmp_path / 'test-paper.body-v5.txt').exists()


def test_interrupted_pdf_download_does_not_leave_reusable_partial_file(tmp_path):
    namespace = backend_functions(tmp_path)
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.side_effect = [b'%PDF-1.5 partial', OSError('connection interrupted')]
    namespace['urllib'].request.urlopen.return_value = response
    with pytest.raises(OSError, match='interrupted'):
        namespace['_extract_paper_pdf_text'](paper_request(pdf_url='https://example.org/paper.pdf'))
    assert not list(tmp_path.glob('*.pdf'))
    assert not list(tmp_path.glob('*.part'))


def test_pdf_body_is_extracted_with_blank_pages_allowed(tmp_path):
    import pymupdf
    namespace = backend_functions(tmp_path)
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), 'Actual full paper body.')
        document.new_page()
        document.save(tmp_path / '.paper_translation_test-paper.body-v5.pdf')
    text = namespace['_extract_paper_pdf_text'](paper_request())
    assert 'Actual full paper body.' in text
    assert 'Abstract is not the full paper.' not in text
    assert (tmp_path / 'test-paper.body-v5.txt').read_text() == text


@pytest.mark.parametrize('finish,content', [
    ('length', '{"translation":"incomplete"}'),
    ('content_filter', '{"translation":"filtered"}'),
    ('stop', '{}'), ('stop', '{"translation":""}'),
    ('stop', 'plain text'), ('stop', '["unexpected"]'),
])
def test_translation_rejects_truncated_or_malformed_output(tmp_path, finish, content):
    namespace = backend_functions(tmp_path)
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content))])
    with pytest.raises(ValueError):
        namespace['_translation_json'](response, ['translation'])


def test_complete_translation_is_accepted(tmp_path):
    namespace = backend_functions(tmp_path)
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason='stop', message=SimpleNamespace(content='{"translation":"完整译文"}'))])
    assert namespace['_translation_json'](response, ['translation']) == {'translation': '完整译文'}


def test_cached_pdf_without_index_cannot_report_ingest_success(tmp_path, monkeypatch):
    monkeypatch.setattr(online, 'UPLOADS_DIR', tmp_path)
    (tmp_path / 'paper_Title.pdf').write_bytes(b'%PDF-1.5 cached')
    rag = Mock()
    rag.collection.get.return_value = {'ids': []}
    agent = online.OnlinePaperSearchAgent.__new__(online.OnlinePaperSearchAgent)
    with pytest.raises(ValueError, match='无索引'):
        agent.download_and_ingest({'paper_id': 'paper', 'title': 'Title'}, rag)
    rag.ingest_file.assert_not_called()


def test_meeting_map_failure_stops_before_reduce(tmp_path):
    path = tmp_path / 'meeting.jsonl'
    path.write_text('\n'.join(json.dumps({'text': f'sentence {i}'}) for i in range(3)))
    summarizer = MeetingSummarizer.__new__(MeetingSummarizer)
    summarizer.model = 'selected-model'
    summarizer.client = Mock()
    summarizer.client.chat.completions.create.side_effect = RuntimeError('map failed')
    with pytest.raises(RuntimeError, match='未进入全局归纳'):
        summarizer.summarize_session(path, chunk_sentences=2)
    assert summarizer.client.chat.completions.create.call_count == 1


def test_europe_pmc_failure_does_not_return_partial_results(monkeypatch):
    monkeypatch.setattr(online.urllib.request, 'urlopen', Mock(side_effect=OSError('unavailable')))
    agent = online.OnlinePaperSearchAgent.__new__(online.OnlinePaperSearchAgent)
    with pytest.raises(RuntimeError, match='EuropePMC'):
        agent.search_europe_pmc('query')


def test_empty_search_brief_does_not_generate_from_model_knowledge():
    agent = online.OnlinePaperSearchAgent.__new__(online.OnlinePaperSearchAgent)
    agent.llm_client = Mock()
    with pytest.raises(RuntimeError, match='未检索到文献'):
        agent.generate_search_brief('query', [])
    agent.llm_client.chat.completions.create.assert_not_called()


def test_explicit_missing_model_path_does_not_search_alternatives(tmp_path, monkeypatch):
    monkeypatch.setenv('WHISPER_MODEL_PATH', str(tmp_path / 'missing'))
    with pytest.raises(FileNotFoundError):
        config_utils.resolve_whisper_path(tmp_path)


def test_monitor_detection_failure_does_not_select_microphone(monkeypatch):
    monkeypatch.delenv('AUDIO_MONITOR_SOURCE', raising=False)
    run = Mock(side_effect=OSError('pactl unavailable'))
    monkeypatch.setattr(config_utils.subprocess, 'run', run)
    with pytest.raises(OSError, match='pactl'):
        config_utils.get_default_monitor_source()
    assert run.call_count == 1
