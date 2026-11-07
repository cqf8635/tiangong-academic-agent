"""Offline tests for LLM token usage tracking and conservative accounting."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from llm_accounting import BudgetExceeded, TrackingClient, UsageLedger
import master_agent_workflow as workflow


def client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def request(**extra):
    base = {"model": "fixture", "messages": [{"role": "user", "content": "测试"}], "max_tokens": 20}
    base.update(extra)
    return base


def test_budget_stops_before_network_request():
    calls = []
    ledger = UsageLedger(1, 2)
    tracked = TrackingClient(client(lambda **kw: calls.append(kw)), ledger)
    with pytest.raises(BudgetExceeded):
        tracked.chat.completions.create(**request())
    assert calls == []
    assert ledger.exhausted
    assert ledger.snapshot()["reported_total_tokens"] == 0


def test_exact_usage_includes_input_output_and_cached_details():
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 6},
    }
    ledger = UsageLedger(10000, 3)
    tracked = TrackingClient(client(lambda **kw: SimpleNamespace(usage=usage)), ledger)
    tracked.chat.completions.create(**request())
    report = ledger.snapshot()
    assert report["reported_total_tokens"] == report["charged_tokens"] == 14
    assert report["actual_usage_complete"]
    assert report["calls"][0]["usage"]["prompt_tokens_details"] == {"cached_tokens": 6}


def test_missing_usage_is_unknown_and_conservatively_charged():
    ledger = UsageLedger(10000, 2)
    tracked = TrackingClient(client(lambda **kw: SimpleNamespace(usage=None)), ledger)
    tracked.chat.completions.create(**request())
    report = ledger.snapshot()
    assert not report["actual_usage_complete"]
    assert report["calls"][0]["usage"] is None
    assert report["charged_tokens"] == report["calls"][0]["reserved_tokens"] > 0


def test_stream_usage_chunk_without_choices_is_recorded(monkeypatch):
    seen = []
    emitted = []

    def create(**kw):
        seen.append(kw)
        return iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="正文"))]),
            SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14)),
        ])

    ledger = UsageLedger(10000, 3)
    tracked = TrackingClient(client(create), ledger)
    monkeypatch.setattr(workflow, "get_stream_callback", lambda: emitted.append)
    result = workflow.safe_chat_completion(client=tracked, **request())
    assert result == "正文"
    assert emitted == ["正文"]
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert ledger.snapshot()["reported_total_tokens"] == 14


def test_every_retry_is_charged_and_budget_error_is_not_retried(monkeypatch):
    calls = []

    def fail(**kw):
        calls.append(kw)
        raise TimeoutError("fixture timeout")

    ledger = UsageLedger(100000, 2)
    tracked = TrackingClient(client(fail), ledger)
    monkeypatch.setattr(workflow.time, "sleep", lambda _: None)
    with pytest.raises(BudgetExceeded):
        workflow.safe_chat_completion(client=tracked, max_retries=5, **request())
    report = ledger.snapshot()
    assert len(calls) == len(report["calls"]) == 2
    assert all(row["status"] == "failed" for row in report["calls"])
    assert report["charged_tokens"] == sum(row["reserved_tokens"] for row in report["calls"])


def test_parallel_requests_cannot_exceed_shared_call_budget():
    ledger = UsageLedger(100000, 2)
    tracked = TrackingClient(client(lambda **kw: SimpleNamespace()), ledger)

    def call(i):
        try:
            tracked.chat.completions.create(**request())
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, range(8)))
    assert sum(results) == len(ledger.snapshot()["calls"]) == 2


def test_sdk_transport_retries_are_disabled():
    options = []

    class Configurable:
        def with_options(self, **kwargs):
            options.append(kwargs)
            return client(lambda **kw: SimpleNamespace())

    TrackingClient(Configurable(), UsageLedger(10000, 2))
    assert options == [{"max_retries": 0}]
