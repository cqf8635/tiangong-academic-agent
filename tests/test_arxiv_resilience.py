from email.message import Message

import online_paper_search_agent as mod


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b"<feed xmlns='http://www.w3.org/2005/Atom'></feed>"


def test_arxiv_429_retries_then_caches(monkeypatch):
    mod._ARXIV_CACHE.clear()
    mod._ARXIV_LAST_REQUEST_AT = 0.0
    mod.ARXIV_MIN_INTERVAL = 0.0
    mod.ARXIV_MAX_RETRIES = 2
    calls = []

    headers = Message()
    headers["Retry-After"] = "0"

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
    assert len(calls) == 2
    agent.search_arxiv("quantum field theory", max_results=3)
    assert len(calls) == 2


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
