from email.message import Message
from email.utils import formatdate
from types import SimpleNamespace

import pytest

import online_paper_search_agent as mod


@pytest.fixture(autouse=True)
def isolated_scheduler(monkeypatch):
    monkeypatch.setattr(mod, "_ARXIV_SCHEDULER", mod.ArxivScheduler())
    updates = mod.ArxivUpdateFeedClient()
    updates.min_interval = 0.0
    monkeypatch.setattr(mod, "_ARXIV_UPDATES", updates)
    monkeypatch.setattr(mod, "_ARXIV_CACHE", mod.OrderedDict())
    monkeypatch.setattr(mod, "_RADAR_SEARCH_CACHE", mod.OrderedDict())
    monkeypatch.setattr(mod, "ARXIV_MIN_INTERVAL", mod.ARXIV_MIN_INTERVAL)
    monkeypatch.setattr(mod, "ARXIV_MAX_RETRIES", mod.ARXIV_MAX_RETRIES)
    monkeypatch.setattr(mod, "_S2_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(mod, "_OPENALEX_COOLDOWN_UNTIL", 0.0)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>"


def test_arxiv_429_stops_retries_and_blocks_other_callers(monkeypatch):
    mod._ARXIV_CACHE.clear()
    mod._ARXIV_LAST_REQUEST_AT = 0.0
    mod.ARXIV_MIN_INTERVAL = 0.0
    mod.ARXIV_MAX_RETRIES = 2
    calls = []

    headers = Message()
    headers["Retry-After"] = "1200"

    def fake_urlopen(_req, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise mod.urllib.error.HTTPError("https://export.arxiv.org", 429, "rate limited", headers, None)
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)

    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    monkeypatch.setattr(agent, "_parse_atom_feed", lambda _xml: [])

    assert agent.search_arxiv("quantum field theory", max_results=3) == []
    assert len(calls) == 1
    assert mod._ARXIV_SCHEDULER.is_in_cooldown()[1] > 1190
    assert not mod._ARXIV_CACHE  # A failed request is not a successful zero-result query.
    req = mod.urllib.request.Request("https://export.arxiv.org/api/query?different=query")
    with pytest.raises(TimeoutError, match="静默保护期"):
        mod._ARXIV_SCHEDULER.execute(req)
    assert len(calls) == 1


@pytest.mark.parametrize("retry_after", ["1200", formatdate(2_000_001_200, usegmt=True), "invalid", None])
def test_retry_after_seconds_and_http_date(monkeypatch, retry_after):
    monkeypatch.setattr(mod.time, "time", lambda: 2_000_000_000)
    headers = Message()
    if retry_after:
        headers["Retry-After"] = retry_after
    expected = 1200 if retry_after not in ("invalid", None) else mod.ARXIV_RATE_LIMIT_COOLDOWN
    assert mod._arxiv_retry_after_seconds(headers) == expected


@pytest.mark.parametrize("status", [403, 429, 503])
def test_curl_preserves_http_error_and_real_retry_after(monkeypatch, status):
    import subprocess

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                f"HTTP/1.1 200 Connection established\r\n\r\n"
                f"HTTP/2 {status}\r\nRetry-After: 1800\r\n\r\n"
                f"blocked\n__HTTP_STATUS__:{status}"
            ).encode(),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    def forbidden_urlopen(*args, **kwargs):
        pytest.fail("HTTP errors must not trigger a second transport request")
    forbidden_urlopen.__module__ = "urllib.request"
    monkeypatch.setattr(mod.urllib.request, "urlopen", forbidden_urlopen)
    scheduler = mod.ArxivScheduler()
    req = mod.urllib.request.Request("https://export.arxiv.org/api/query?test=1")
    with pytest.raises(mod.urllib.error.HTTPError) as caught:
        scheduler.execute(req)
    assert caught.value.code == status
    assert caught.value.headers["Retry-After"] == "1800"
    assert len(calls) == 1  # No second urllib request on HTTP errors.
    assert scheduler.is_in_cooldown()[0] == (status in (403, 429))
    if status in (403, 429):
        assert scheduler.is_in_cooldown()[1] > 1790


def test_curl_success_preserves_atom_body(monkeypatch):
    import subprocess

    body = FakeResponse().read()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0,
        stdout=b"HTTP/2 200\r\nContent-Type: application/atom+xml\r\n\r\n" + body + b"\n__HTTP_STATUS__:200",
    ))
    req = mod.urllib.request.Request("https://export.arxiv.org/api/query")
    assert mod.ArxivScheduler().execute(req) == body


def test_curl_dns_failure_is_not_reported_as_read_timeout(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=6, stdout=b"\n__HTTP_STATUS__:000"
    ))
    req = mod.urllib.request.Request("https://export.arxiv.org/api/query")
    with pytest.raises(mod.urllib.error.URLError, match="code=6"):
        mod.ArxivScheduler().execute(req)


def test_radar_successful_empty_result_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(mod._ARXIV_SCHEDULER, "execute", lambda *a, **kw: calls.append(1) or FakeResponse().read())
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    monkeypatch.setattr(agent, "_parse_atom_feed", lambda xml: [])
    first = agent.search_radar(query="no matches", major="CS")
    assert first["papers"] == []
    assert not first.get("is_cached_snapshot")
    assert agent.search_radar(query="no matches", major="CS") == first
    assert len(calls) == 1


def test_arxiv_cache_is_bounded(monkeypatch):
    mod._ARXIV_CACHE.clear()
    mod.ARXIV_MIN_INTERVAL = 0.0
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    monkeypatch.setattr(agent, "_parse_atom_feed", lambda _xml: [])

    for i in range(mod.ARXIV_CACHE_MAX_ENTRIES + 10):
        agent.search_arxiv(f"unique query {i}", max_results=1)

    assert len(mod._ARXIV_CACHE) == mod.ARXIV_CACHE_MAX_ENTRIES


def test_arxiv_scheduler_strict_single_concurrency_and_interval(monkeypatch):
    """验证全局调度器保证同一时刻仅有单连接，且请求完成与下一次发起之间有严格物理间隔"""
    import threading
    import time

    mod._ARXIV_SCHEDULER.reset_cooldown()
    mod._ARXIV_SCHEDULER.last_completed_at = 0.0
    mod.ARXIV_MIN_INTERVAL = 0.1  # 缩短测试间隔为 0.1s

    active_connections = 0
    max_concurrent_connections = 0
    lock = threading.Lock()
    intervals = []
    last_end = 0.0

    def fake_urlopen(_req, timeout):
        nonlocal active_connections, max_concurrent_connections, last_end
        now = time.monotonic()
        with lock:
            if last_end > 0.0:
                intervals.append(now - last_end)
            active_connections += 1
            if active_connections > max_concurrent_connections:
                max_concurrent_connections = active_connections
        time.sleep(0.05)
        with lock:
            active_connections -= 1
            last_end = time.monotonic()
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    # 启动 3 个并发线程同时发起 execute 请求
    threads = []
    for _ in range(3):
        t = threading.Thread(
            target=lambda: mod._ARXIV_SCHEDULER.execute(
                mod.urllib.request.Request("https://export.arxiv.org/api/query?search_query=test"),
                timeout=5,
            )
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 1. 验证最大并发连接数绝对为 1，杜绝并发撞车
    assert max_concurrent_connections == 1
    # 2. 验证各个请求之间满足最小间隔 >= 0.09s
    assert len(intervals) == 2
    for interval in intervals:
        assert interval >= 0.09


def test_arxiv_scheduler_yields_for_active_user():
    """验证后台任务在用户活跃时主动让出通道"""
    mod._ARXIV_SCHEDULER.reset_cooldown()
    mod._ARXIV_SCHEDULER.mark_user_active()

    # 此时后台任务发起请求，应抛出 ConnectionAbortedError 主动礼让
    req = mod.urllib.request.Request("https://export.arxiv.org/api/query?search_query=test")
    try:
        mod._ARXIV_SCHEDULER.execute(req, is_background=True)
        assert False, "后台任务应该在用户活跃期被拦截让行"
    except ConnectionAbortedError as e:
        assert "前台用户正在活跃检索" in str(e)


def test_arxiv_circuit_breaker_on_consecutive_timeouts(monkeypatch):
    """验证连续超时触发全局防雪崩熔断机制"""
    mod._ARXIV_SCHEDULER.reset_cooldown()
    mod._ARXIV_SCHEDULER.last_completed_at = 0.0
    mod.ARXIV_MIN_INTERVAL = 0.0

    def fake_timeout(*args, **kwargs):
        raise TimeoutError("Read operation timed out")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_timeout)
    req = mod.urllib.request.Request("https://export.arxiv.org/api/query?search_query=test")

    # 第一次超时
    try:
        mod._ARXIV_SCHEDULER.execute(req)
    except TimeoutError:
        pass
    assert not mod._ARXIV_SCHEDULER.is_in_cooldown()[0]

    # 第二次超时 -> 触发熔断
    try:
        mod._ARXIV_SCHEDULER.execute(req)
    except TimeoutError:
        pass
    in_cd, remain = mod._ARXIV_SCHEDULER.is_in_cooldown()
    assert in_cd is True
    assert remain > 0

    # 第三次请求直接快速失败，不再发起实际网络调用
    try:
        mod._ARXIV_SCHEDULER.execute(req)
        assert False, "应快速熔断"
    except TimeoutError as e:
        assert "静默保护期" in str(e)

    mod._ARXIV_SCHEDULER.reset_cooldown()


def update_feed(*entries):
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        '<updated>2026-11-17T08:26:34Z</updated>'
        + ''.join(entries) + '</feed>'
    ).encode()


def update_entry(paper_id="2611.18642v1", category="cs.AI"):
    return (
        f'<entry><id>oai:arXiv.org:{paper_id}</id>'
        '<title>New paper from the current announcement batch</title>'
        '<published>2026-11-17T00:00:00-05:00</published>'
        f'<summary>arXiv:{paper_id} Announce Type: new\nAbstract: Actual abstract.</summary>'
        '<dc:creator>Alice Smith, Bob Jones</dc:creator>'
        f'<category term="{category}"/><arxiv:announce_type>new</arxiv:announce_type></entry>'
    )


class UpdateResponse(FakeResponse):
    def __init__(self, body):
        self.body = body
        self.headers = {"ETag": '"batch-v1"'}

    def read(self, _limit=None):
        return self.body


def test_update_feed_parses_real_rss_atom_fields_and_versions():
    batch = mod._parse_arxiv_update_feed(update_feed(
        update_entry(), update_entry("2611.18642v2"), update_entry("hep-th/9901001v1", "hep-th"),
    ))
    assert len(batch["papers"]) == 2
    paper = next(p for p in batch["papers"] if p["paper_id"].startswith("2611"))
    assert paper["paper_id"] == "2611.18642v2"
    assert paper["authors"] == ["Alice Smith", "Bob Jones"]
    assert paper["summary"] == "Actual abstract."
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2611.18642v2"
    assert paper["date_kind"] == "announcement"
    assert paper["primary_category"] == "cs.AI"
    assert batch["feed_updated_at"] == "2026-11-17T08:26:34Z"


def test_update_feed_categories_expand_archives_without_wildcard_queries():
    assert mod._arxiv_feed_categories("CS") == ("cs",)
    assert mod._arxiv_feed_categories("CS", ["cs.LG", "cs.AI", "cs.AI"]) == ("cs.AI", "cs.LG")
    physics = mod._arxiv_feed_categories("PHYSICS")
    assert "hep-th" in physics and "nucl-ex" in physics
    assert "hep" not in physics and "nucl" not in physics
    assert "cs" in mod._arxiv_feed_categories("ALL")
    with pytest.raises(ValueError):
        mod._arxiv_feed_categories("CS", ["../../api/query"])


def test_latest_updates_work_during_api_cooldown_and_share_batch_across_pages(monkeypatch):
    calls = []
    body = update_feed(update_entry("2611.18642v1"), update_entry("2611.18643v1"))

    def fetch(req, timeout):
        calls.append(req.full_url)
        return UpdateResponse(body)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    monkeypatch.setattr(mod._ARXIV_SCHEDULER, "execute", lambda *a, **kw: pytest.fail("Latest browsing must not query the API when its feed works"))
    mod._ARXIV_SCHEDULER.trigger_cooldown(900)
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    first = agent.search_radar(major="CS", max_results=1)
    second = agent.search_radar(major="CS", start=1, max_results=1)
    assert first["source"] == second["source"] == "arxiv_rss"
    assert first["total_results"] == 2
    assert first["total_results_scope"] == "announcement_batch"
    assert first["papers"][0]["paper_id"] != second["papers"][0]["paper_id"]
    assert first["snapshot_synced_at"] == second["snapshot_synced_at"]
    first["papers"][0]["title"] = "changed by caller"
    assert agent.search_radar(major="CS", max_results=1)["papers"][0]["title"] != "changed by caller"
    assert calls == ["https://rss.arxiv.org/atom/cs"]


def test_update_feed_uses_conditional_revalidation(monkeypatch):
    client = mod._ARXIV_UPDATES
    calls = []

    def fetch(req, timeout):
        calls.append(req)
        if len(calls) == 2:
            raise mod.urllib.error.HTTPError(req.full_url, 304, "not modified", {}, None)
        return UpdateResponse(update_feed(update_entry()))

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    original = client.fetch(("cs.AI",))
    key = ("cs.AI",)
    _, data, etag = client._cache[key]
    client._cache[key] = (mod.time.monotonic() - client.cache_ttl - 1, data, etag)
    refreshed = client.fetch(key)
    assert calls[1].get_header("If-none-match") == '"batch-v1"'
    assert refreshed["papers"] == original["papers"]
    assert refreshed["feed_updated_at"] == original["feed_updated_at"]
    assert not client.cooldown_remaining()


def test_update_feed_rate_limit_respects_retry_after(monkeypatch):
    calls = []
    headers = Message()
    headers["Retry-After"] = "1800"

    def fetch(req, timeout):
        calls.append(req.full_url)
        raise mod.urllib.error.HTTPError(req.full_url, 429, "limited", headers, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    with pytest.raises(mod.urllib.error.HTTPError):
        mod._ARXIV_UPDATES.fetch(("cs",))
    assert mod._ARXIV_UPDATES.cooldown_remaining() > 1790
    with pytest.raises(TimeoutError):
        mod._ARXIV_UPDATES.fetch(("math",))
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"<html>blocked</html>", b"<feed/>", b"not XML"])
def test_invalid_update_responses_are_not_cached(monkeypatch, body):
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **kw: UpdateResponse(body))
    with pytest.raises((ValueError, mod.ET.ParseError)):
        mod._ARXIV_UPDATES.fetch(("cs",))
    assert not mod._ARXIV_UPDATES._cache
    assert mod._ARXIV_UPDATES.cooldown_remaining() > 0


def test_feed_failure_uses_original_api_before_snapshot(monkeypatch):
    monkeypatch.setattr(mod._ARXIV_UPDATES, "fetch", lambda *a: (_ for _ in ()).throw(TimeoutError("feed down")))
    calls = []
    monkeypatch.setattr(mod._ARXIV_SCHEDULER, "execute", lambda *a, **kw: calls.append(1) or FakeResponse().read())
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    result = agent.search_radar(major="CS")
    assert result["source"] == "arxiv_api"
    assert result["update_status"] == "no_announcements"
    assert not result["is_cached_snapshot"]
    assert calls == [1]


def test_failed_refresh_returns_paged_stale_snapshot_without_poisoning_cache(tmp_path, monkeypatch):
    import json
    import radar_sync_engine

    monkeypatch.setattr(radar_sync_engine, "RADAR_CACHE_DIR", tmp_path)
    old = [{"paper_id": str(n), "title": "Old paper", "categories": ["cs.AI"]} for n in range(3)]
    snapshot = {"papers": old, "total_results": 99999, "snapshot_synced_at": "2026-11-10T00:00:00Z"}
    (tmp_path / "radar_CS_all.json").write_text(json.dumps(snapshot))

    def unavailable(*a, **kw):
        raise TimeoutError("offline")

    monkeypatch.setattr(mod._ARXIV_UPDATES, "fetch", unavailable)
    monkeypatch.setattr(mod._ARXIV_SCHEDULER, "execute", unavailable)
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    result = agent.search_radar(major="CS", start=1, max_results=1)
    assert result["papers"] == [old[1]]
    assert result["total_results"] == 3
    assert result["update_status"] == "stale_snapshot"
    assert result["snapshot_synced_at"] == snapshot["snapshot_synced_at"]
    assert result["is_cached_snapshot"]
    assert not mod._RADAR_SEARCH_CACHE
    assert json.loads((tmp_path / "radar_CS_all.json").read_text()) == snapshot


def test_keyword_queries_do_not_use_updates_feed(monkeypatch):
    monkeypatch.setattr(mod._ARXIV_UPDATES, "fetch", lambda *a: pytest.fail("Keyword search must retain its existing API"))
    monkeypatch.setattr(mod._ARXIV_SCHEDULER, "execute", lambda *a, **kw: FakeResponse().read())
    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    result = agent.search_radar(query="graph learning", major="CS")
    assert "graph learning" in result["query_expression"]
    assert "update_status" not in result


@pytest.mark.parametrize("update_status", ["no_announcements", "live", "stale_snapshot"])
def test_update_endpoint_does_not_replace_empty_batches_or_pages_with_old_snapshot(update_status):
    import ast
    import asyncio
    from pathlib import Path

    # Exercise the endpoint without initializing the unrelated GPU/LLM services.
    tree = ast.parse((Path(mod.__file__).parent / "backend_api.py").read_text())
    endpoint = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "radar_search")
    endpoint.decorator_list = []
    result = {"papers": [], "total_results": 0, "update_status": update_status}
    namespace = {
        "RadarSearchRequest": SimpleNamespace,
        "paper_searcher": SimpleNamespace(search_radar=lambda **kw: result),
        "radar_sync_engine": SimpleNamespace(read_snapshot=lambda *a: pytest.fail("A valid empty update response must not load old papers")),
    }
    exec(compile(ast.Module(body=[endpoint], type_ignores=[]), "backend_api.py", "exec"), namespace)
    req = SimpleNamespace(query=None, major="CS", sub_categories=[], search_in="all", sort_by="submittedDate", start=0, max_results=25)
    response = asyncio.run(namespace["radar_search"](req))
    assert response == {"status": "success", "data": result}


def test_academic_cache_keeps_query_bounds_and_limits_separate(monkeypatch):
    import json

    calls = []

    def fetch(req, **kwargs):
        calls.append(req.full_url)
        return UpdateResponse(json.dumps({"message": {"items": [{
            "DOI": "10.test/study", "title": ["Actual study"],
            "abstract": "<p>Original abstract</p>",
        }]}}).encode())

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    bounds = {"start_year": 2025, "end_year": 2025}
    first = agent.search_crossref("mechanism", max_results=2, time_constraint=bounds)
    first[0].summary = "caller changed it"
    cached = agent.search_crossref(query="mechanism", max_results=2, time_constraint=bounds)
    assert cached.cache_hit and "Original abstract" in cached[0].summary
    assert len(calls) == 1
    agent.search_crossref("mechanism", max_results=3, time_constraint=bounds)
    agent.search_crossref("mechanism", max_results=2, time_constraint={"start_year": 2024})
    assert len(calls) == 3


def test_academic_cache_expires_and_failed_queries_are_not_zero_hits(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    calls = []

    def fetch(req, **kwargs):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise TimeoutError("offline")
        return UpdateResponse(b'{"message":{"items":[]}}')

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    failed = agent.search_crossref("mechanism")
    assert failed.source_error == "TimeoutError"
    assert not mod._search_state(agent)["cache"]
    assert "cooldown" in agent.search_crossref("other mechanism").source_error
    assert len(calls) == 1
    clock[0] += 31
    assert agent.search_crossref("mechanism").source_error is None
    assert agent.search_crossref("mechanism").cache_hit
    clock[0] += mod.ACADEMIC_EMPTY_CACHE_TTL + 1
    agent.search_crossref("mechanism")
    assert len(calls) == 3


def test_academic_identical_concurrent_queries_share_one_request(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    entered, release = threading.Event(), threading.Event()
    calls = []

    def fetch(req, **kwargs):
        calls.append(req.full_url)
        entered.set()
        assert release.wait(2)
        return UpdateResponse(b'{"message":{"items":[]}}')

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(agent.search_crossref, "shared query")
        assert entered.wait(2)
        second = executor.submit(agent.search_crossref, "shared query")
        release.set()
        assert first.result(2) == second.result(2) == []
    assert len(calls) == 1


def test_dblp_challenge_cools_new_queries_then_recovers(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])

    def fetch(req, **kwargs):
        calls.append(req.full_url)
        response = UpdateResponse(b"<html>Making sure you're not a bot!</html>" if len(calls) == 1 else b'{"result":{"hits":{"hit":[]}}}')
        response.status = 202 if len(calls) == 1 else 200
        response.headers = {"Content-Type": "text/html" if len(calls) == 1 else "application/json"}
        return response

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    assert agent.search_dblp("one").source_error == "Http202NonJson"
    assert agent.search_dblp("two").source_error == "cooldown: bot_challenge"
    assert len(calls) == 1
    clock[0] += 901
    assert agent.search_dblp("two").source_error is None
    assert len(calls) == 2


def test_semanticscholar_cooldown_honors_retry_after(monkeypatch):
    headers = Message()
    headers["Retry-After"] = "1200"
    calls = []

    def fetch(req, **kwargs):
        calls.append(req.full_url)
        raise mod.urllib.error.HTTPError(req.full_url, 429, "limited", headers, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    assert agent.search_semanticscholar("one").source_error == "RateLimited"
    agent.search_semanticscholar("two")
    assert len(calls) == 1
    until, _ = mod._search_state(agent)["cooldowns"]["semanticscholar"]
    assert until - mod.time.monotonic() > 1190
    assert mod._S2_COOLDOWN_UNTIL - mod.time.monotonic() > 1190


def test_multisource_deadline_keeps_fast_results_without_waiting_for_stragglers(monkeypatch, capsys):
    import threading
    import time

    monkeypatch.setattr(mod, "ACADEMIC_SEARCH_BUDGET", 0.08)
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    release, entered, finished = threading.Event(), threading.Event(), threading.Event()

    def slow(*args):
        entered.set()
        try:
            assert release.wait(2)
            return []
        finally:
            finished.set()

    paper = mod.AcademicResource(id="test", title="Fast result", source="crossref", summary="Abstract")
    for method in ("search_openalex", "search_semanticscholar", "search_dblp"):
        monkeypatch.setattr(agent, method, lambda *args: [])
    monkeypatch.setattr(agent, "search_crossref", lambda *args: [paper])
    monkeypatch.setattr(agent, "search_arxiv", slow)
    try:
        started = time.monotonic()
        result = agent.search_multisource("query", english_query="query", include_github=False)
        assert time.monotonic() - started < 0.5
        assert entered.is_set() and not finished.is_set()
        assert result["crossref_papers"][0]["title"] == "Fast result"
        assert result["source_errors"]["arxiv"] == "TimeoutError"
        assert "TimeoutError:" in capsys.readouterr().out
    finally:
        release.set()
        assert finished.wait(2)


def test_github_reuses_english_query_and_retains_metadata_when_details_exceed_budget(monkeypatch):
    import json
    clock = [100.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("GITHUB_DETAIL_FETCH_LIMIT", "1")
    monkeypatch.setenv("GITHUB_DETAIL_BUDGET", "3")
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    calls = []

    def fetch(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            return UpdateResponse(json.dumps({"items": [{"full_name": "org/repo", "html_url": "https://github.com/org/repo", "default_branch": "main"}]}).encode())
        assert timeout <= 3
        clock[0] += 4
        raise TimeoutError("detail slow")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fetch)
    repos = agent.search_github("hypergraph learning")
    assert len(repos) == 1 and repos[0].evidence_level == "repository_metadata"
    assert len(calls) == 2 and calls[-1].endswith("/readme")


def test_multisource_uses_prepared_english_for_github_and_caches_web(monkeypatch):
    import online_resource_adapters
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    for method in ("search_arxiv", "search_crossref", "search_openalex", "search_semanticscholar", "search_dblp"):
        monkeypatch.setattr(agent, method, lambda *args: [])
    github_queries, web_queries, dataset_queries = [], [], []
    monkeypatch.setattr(agent, "search_github", lambda query, *a: github_queries.append(query) or [])
    monkeypatch.setattr(agent, "extract_academic_keywords", lambda *a: pytest.fail("Already have English terms"))
    monkeypatch.setattr(online_resource_adapters, "search_web", lambda query, *a: web_queries.append(query) or [{"title": "Page", "summary": "Snippet"}])
    monkeypatch.setattr(online_resource_adapters, "search_datasets", lambda query, *a: dataset_queries.append(query) or [])
    first = agent.search_multisource("中文提问", english_query="hypergraph learning", include_web=True, include_datasets=True)
    second = agent.search_multisource("中文提问", english_query="hypergraph learning", include_web=True, include_datasets=True)
    assert github_queries == ["hypergraph learning", "hypergraph learning"]
    assert web_queries == dataset_queries == ["hypergraph learning"]
    assert first["web_results"] == second["web_results"]
    assert set(second["source_cache_hits"]) == {"web", "datasets"}


def test_budgeted_arxiv_does_not_wait_or_retry_beyond_remaining_time(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "ARXIV_MIN_INTERVAL", 0)
    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout: calls.append(timeout) or (_ for _ in ()).throw(TimeoutError("slow")))
    agent = object.__new__(mod.OnlinePaperSearchAgent)
    with mod._search_budget(mod.time.monotonic() + 0.1):
        result = agent.search_arxiv("budgeted query")
    assert result.source_error == "TimeoutError"
    assert len(calls) == 1 and 0 < calls[0] <= 0.1


def test_budgeted_arxiv_waits_for_scheduler_lock_only_within_deadline():
    import time
    lock = mod._ARXIV_SCHEDULER._lock
    lock.acquire()
    try:
        started = time.monotonic()
        with mod._search_budget(time.monotonic() + 0.03):
            with pytest.raises(TimeoutError, match="排队"):
                mod._ARXIV_SCHEDULER.execute(mod.urllib.request.Request("https://export.arxiv.org/api/query"))
        assert time.monotonic() - started < 0.3
    finally:
        lock.release()
