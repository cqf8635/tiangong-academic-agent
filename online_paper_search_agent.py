"""Online ArXiv Academic Paper Search & Ingestion Agent.

Integrates:
1. ArXiv Atom API querying & metadata parsing (title, authors, year, abstract, PDF url).
2. Automated paper downloading into the local uploads directory.
3. Direct ingestion into UniversalDocumentRAGEngine (BGE-M3 vector indexing).
4. SOTA literature brief generation for academic research.
"""

from __future__ import annotations

import json
import copy
import inspect
import base64
import os
import re
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from openai import OpenAI

from config_utils import (
    get_default_llm_model, get_llm_max_tokens, get_llm_timeout,
    resolve_retrieval_time_constraint, retrieval_date_bounds,
)
from prompts import build_academic_keyword_prompt, get_assistant_name, format_evidence_for_prompt, build_lead_tutor_system_prompt

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_REQUEST_TIMEOUT = float(os.getenv("ARXIV_REQUEST_TIMEOUT", "8"))
ARXIV_MAX_RETRIES = int(os.getenv("ARXIV_MAX_RETRIES", "1"))
ARXIV_MIN_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL", "3.2"))
ARXIV_RATE_LIMIT_COOLDOWN = float(os.getenv("ARXIV_RATE_LIMIT_COOLDOWN", "900"))
ARXIV_CACHE_TTL = float(os.getenv("ARXIV_CACHE_TTL", "900"))
ARXIV_EMPTY_CACHE_TTL = float(os.getenv("ARXIV_EMPTY_CACHE_TTL", "15"))
ARXIV_CACHE_MAX_ENTRIES = 64
_ARXIV_CACHE: "OrderedDict[tuple[str, int], tuple[float, List['ArxivPaper']]]" = OrderedDict()
_ARXIV_CACHE_LOCK = threading.Lock()
_S2_COOLDOWN_UNTIL = 0.0
_S2_LOCK = threading.Lock()
ACADEMIC_SEARCH_BUDGET = max(0.1, float(os.getenv("ACADEMIC_SEARCH_BUDGET", "18")))
ACADEMIC_SEARCH_CACHE_TTL = max(0.0, float(os.getenv("ACADEMIC_SEARCH_CACHE_TTL", "300")))
ACADEMIC_EMPTY_CACHE_TTL = 30.0
ACADEMIC_SEARCH_CACHE_MAX = 256
_SEARCH_CONTEXT = threading.local()
_SEARCH_STATE_LOCK = threading.Lock()
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="academic-search")
_SEARCH_WORK_SLOTS = threading.BoundedSemaphore(32)


@contextmanager
def _search_budget(deadline):
    previous = getattr(_SEARCH_CONTEXT, "deadline", None)
    _SEARCH_CONTEXT.deadline = min(previous, deadline) if previous is not None else deadline
    try:
        yield
    finally:
        _SEARCH_CONTEXT.deadline = previous


def _search_timeout(seconds):
    deadline = getattr(_SEARCH_CONTEXT, "deadline", None)
    if deadline is None:
        return seconds
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("学术网络检索时间预算已耗尽")
    return min(seconds, remaining)


def _search_pause(seconds):
    if _search_timeout(seconds) < seconds:
        raise TimeoutError("剩余检索预算不足以等待后重试")
    time.sleep(seconds)


@contextmanager
def _search_lock(lock):
    deadline = getattr(_SEARCH_CONTEXT, "deadline", None)
    acquired = lock.acquire(timeout=_search_timeout(ACADEMIC_SEARCH_BUDGET)) if deadline is not None else lock.acquire()
    if not acquired:
        raise TimeoutError("检索排队超过本轮时间预算")
    try:
        yield
    finally:
        lock.release()


def _retry_after_seconds(headers, default):
    value = headers.get("Retry-After") if headers else None
    if value:
        try:
            return max(1.0, float(value))
        except (TypeError, ValueError):
            try:
                return max(1.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return default


def _search_state(agent):
    with _SEARCH_STATE_LOCK:
        state = agent.__dict__.get("_academic_search_state")
        if state is None:
            state = {"lock": threading.Lock(), "cache": OrderedDict(), "inflight": {},
                     "cooldowns": {}, "source_locks": {}}
            agent._academic_search_state = state
        return state


def _cool_down_source(agent, source, seconds, reason):
    state = _search_state(agent)
    with state["lock"]:
        until, _ = state["cooldowns"].get(source, (0.0, ""))
        new_until = time.monotonic() + seconds
        if new_until > until:
            state["cooldowns"][source] = (new_until, reason)


def _source_http_failure(agent, source, exc):
    if exc.code in (403, 429, 503):
        seconds = _retry_after_seconds(exc.headers, 180.0 if exc.code == 429 else 60.0)
        if source == "github" and exc.headers and exc.headers.get("X-RateLimit-Remaining") == "0":
            try:
                seconds = max(seconds, float(exc.headers.get("X-RateLimit-Reset", "0")) - time.time())
            except ValueError:
                pass
        _cool_down_source(agent, source, seconds, f"HTTP {exc.code}")


def _cached_academic_search(source):
    """Cache successes, share identical in-flight calls, and cool failed providers."""
    def decorate(method):
        signature = inspect.signature(method)

        @wraps(method)
        def wrapped(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            if "time_constraint" in bound.arguments:
                bound.arguments["time_constraint"] = resolve_retrieval_time_constraint(bound.arguments["time_constraint"])
            arguments = {k: v for k, v in bound.arguments.items() if k != "self"}
            key = (source, json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str))
            state = _search_state(self)
            with state["lock"]:
                cached = state["cache"].get(key)
                if cached and time.monotonic() < cached[0]:
                    state["cache"].move_to_end(key)
                    result = copy.deepcopy(cached[1])
                    result.cache_hit = True
                    return result
                state["cache"].pop(key, None)
                until, reason = state["cooldowns"].get(source, (0.0, ""))
                if time.monotonic() < until:
                    return SearchResults(source_error=f"cooldown: {reason}")
                future = state["inflight"].get(key)
                owner = future is None
                if owner:
                    future = Future()
                    state["inflight"][key] = future
                source_lock = state["source_locks"].setdefault(source, threading.Lock())
            if not owner:
                return copy.deepcopy(future.result(timeout=_search_timeout(ACADEMIC_SEARCH_BUDGET)))
            try:
                with _search_lock(source_lock):
                    _search_timeout(ACADEMIC_SEARCH_BUDGET)
                    # A different query may have restricted the source while we queued.
                    with state["lock"]:
                        until, reason = state["cooldowns"].get(source, (0.0, ""))
                    if time.monotonic() < until:
                        result = SearchResults(source_error=f"cooldown: {reason}")
                    else:
                        records = method(*bound.args, **bound.kwargs)
                        result = SearchResults(records, getattr(records, "source_error", None))
                if result.source_error:
                    _cool_down_source(self, source, 30.0, result.source_error)
                else:
                    ttl = ACADEMIC_SEARCH_CACHE_TTL if result else ACADEMIC_EMPTY_CACHE_TTL
                    with state["lock"]:
                        state["cache"][key] = (time.monotonic() + ttl, copy.deepcopy(result))
                        while len(state["cache"]) > ACADEMIC_SEARCH_CACHE_MAX:
                            state["cache"].popitem(last=False)
                future.set_result(copy.deepcopy(result))
                return result
            except BaseException as exc:
                future.set_exception(exc)
                raise
            finally:
                with state["lock"]:
                    state["inflight"].pop(key, None)

        return wrapped
    return decorate


def _s2_headers() -> Dict[str, str]:
    headers = {"User-Agent": "ResearchScholar/1.0", "Accept": "application/json"}
    api_key = (os.getenv("SEMANTICSCHOLAR_API_KEY") or "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _s2_in_cooldown() -> bool:
    with _S2_LOCK:
        return time.monotonic() < _S2_COOLDOWN_UNTIL


def _s2_trigger_cooldown(seconds: float = 60.0) -> None:
    global _S2_COOLDOWN_UNTIL
    with _S2_LOCK:
        _S2_COOLDOWN_UNTIL = max(_S2_COOLDOWN_UNTIL, time.monotonic() + max(10.0, seconds))


def _arxiv_retry_after_seconds(headers) -> float:
    value = headers.get("Retry-After") if headers else None
    if value:
        try:
            return max(3.2, float(value))
        except (TypeError, ValueError):
            try:
                return max(3.2, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    return max(3.2, ARXIV_RATE_LIMIT_COOLDOWN)


_UPDATES_ATOM = "{http://www.w3.org/2005/Atom}"
_UPDATES_ARXIV = "{http://arxiv.org/schemas/atom}"
_UPDATES_DC = "{http://purl.org/dc/elements/1.1/}"


def _arxiv_feed_categories(major=None, sub_categories=None):
    from academic_taxonomy import ACADEMIC_TAXONOMY, get_subcategory_meta

    expansions = {"hep": ["hep-ex", "hep-lat", "hep-ph", "hep-th"],
                  "nucl": ["nucl-ex", "nucl-th"]}
    archives = {g["arxiv_prefix"] for m in ACADEMIC_TAXONOMY.values() for g in m["groups"].values()}
    if sub_categories:
        categories = [c.strip().rstrip("*") for c in sub_categories if c.strip()]
        if any(c not in archives and not get_subcategory_meta(c) for c in categories):
            raise ValueError("不支持的 arXiv 订阅分类")
    else:
        if major and major != "ALL" and major not in ACADEMIC_TAXONOMY:
            raise ValueError("不支持的 arXiv 学科")
        majors = [ACADEMIC_TAXONOMY[major]] if major and major != "ALL" else ACADEMIC_TAXONOMY.values()
        categories = [g["arxiv_prefix"] for m in majors for g in m["groups"].values()]
    return tuple(sorted({expanded for c in categories for expanded in expansions.get(c, [c])}))


def _parse_arxiv_update_feed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    updated_at = root.findtext(f"{_UPDATES_ATOM}updated")
    if root.tag != f"{_UPDATES_ATOM}feed" or not updated_at:
        raise ValueError("arXiv 更新订阅未返回有效 Atom 批次")
    papers = {}
    for entry in root.findall(f"{_UPDATES_ATOM}entry"):
        entry_id = (entry.findtext(f"{_UPDATES_ATOM}id") or "").strip()
        paper_id = entry_id.removeprefix("oai:arXiv.org:")
        if "/abs/" in paper_id:
            paper_id = paper_id.split("/abs/", 1)[1]
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z.-]*/\d{7})(?:v\d+)?", paper_id):
            raise ValueError("arXiv 更新订阅包含无效论文编号")
        title = " ".join((entry.findtext(f"{_UPDATES_ATOM}title") or "").split())
        if not title:
            raise ValueError("arXiv 更新订阅包含无标题论文")
        summary = entry.findtext(f"{_UPDATES_ATOM}summary") or ""
        if summary.startswith("arXiv:") and "Abstract:" in summary:
            summary = summary.split("Abstract:", 1)[1]
        authors = [a.findtext(f"{_UPDATES_ATOM}name") or "" for a in entry.findall(f"{_UPDATES_ATOM}author")]
        if not authors:
            creators = [c.text or "" for c in entry.findall(f"{_UPDATES_DC}creator")]
            authors = [name.strip() for c in creators for name in c.split(",") if name.strip()]
        categories = [c.get("term") for c in entry.findall(f"{_UPDATES_ATOM}category") if c.get("term")]
        announced_at = entry.findtext(f"{_UPDATES_ATOM}published") or updated_at
        primary = entry.find(f"{_UPDATES_ARXIV}primary_category")
        paper = {
            "paper_id": paper_id, "title": title, "authors": authors,
            "year": int(announced_at[:4]), "published_at": announced_at,
            "announced_at": announced_at, "date_kind": "announcement",
            "announce_type": entry.findtext(f"{_UPDATES_ARXIV}announce_type") or "",
            "summary": " ".join(summary.split()),
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}", "source": "arxiv",
            "primary_category": primary.get("term") if primary is not None else (categories[0] if categories else None),
            "categories": categories,
        }
        base_id = re.sub(r"v\d+$", "", paper_id)
        previous = papers.get(base_id)
        version = int((re.search(r"v(\d+)$", paper_id) or [None, "0"])[1])
        previous_version = int((re.search(r"v(\d+)$", previous["paper_id"]) or [None, "0"])[1]) if previous else -1
        if version > previous_version:
            papers[base_id] = paper
    return {
        "papers": sorted(papers.values(), key=lambda p: (p["announced_at"], p["paper_id"]), reverse=True),
        "feed_updated_at": updated_at,
    }


class ArxivUpdateFeedClient:
    """Serialize subscription fetches; share a whole batch across pages and sizes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._last_completed_at = 0.0
        self._cooldown_until = 0.0
        self.cache_ttl = max(60.0, float(os.getenv("ARXIV_UPDATES_CACHE_TTL", "300")))
        self.timeout = float(os.getenv("ARXIV_UPDATES_TIMEOUT", "25"))
        self.min_interval = 3.2

    def cooldown_remaining(self):
        return max(0.0, self._cooldown_until - time.monotonic())

    def fetch(self, categories):
        key = tuple(sorted(set(categories)))
        if not key:
            raise ValueError("缺少 arXiv 订阅分类")
        with self._lock:
            cached = self._cache.get(key)
            if cached and time.monotonic() - cached[0] < self.cache_ttl:
                self._cache.move_to_end(key)
                return copy.deepcopy(cached[1])
            if self.cooldown_remaining():
                raise TimeoutError("arXiv 更新订阅暂时不可用，等待冷却")
            wait_for = self.min_interval - (time.monotonic() - self._last_completed_at)
            if wait_for > 0:
                time.sleep(wait_for)
            headers = {"User-Agent": os.getenv("ARXIV_USER_AGENT", "TianGongAcademicRadar/2.0"),
                       "Accept": "application/atom+xml"}
            if cached and cached[2]:
                headers["If-None-Match"] = cached[2]
            req = urllib.request.Request("https://rss.arxiv.org/atom/" + "+".join(key), headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    body = response.read(20 * 1024 * 1024 + 1)
                    if len(body) > 20 * 1024 * 1024:
                        raise ValueError("arXiv 更新订阅超出大小限制")
                    data = _parse_arxiv_update_feed(body)
                    etag = response.headers.get("ETag")
                data["fetched_at"] = datetime.now(timezone.utc).isoformat()
            except urllib.error.HTTPError as exc:
                if exc.code == 304 and cached:
                    data, etag = copy.deepcopy(cached[1]), cached[2]
                    data["fetched_at"] = datetime.now(timezone.utc).isoformat()
                else:
                    cooldown = 900.0 if exc.code in (403, 429) else 60.0
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    if retry_after:
                        try:
                            cooldown = max(cooldown, float(retry_after))
                        except ValueError:
                            try:
                                cooldown = max(cooldown, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (TypeError, ValueError, OverflowError):
                                pass
                    self._cooldown_until = time.monotonic() + cooldown
                    raise
            except Exception:
                self._cooldown_until = time.monotonic() + 60.0
                raise
            finally:
                self._last_completed_at = time.monotonic()
            self._cache[key] = (time.monotonic(), data, etag)
            self._cache.move_to_end(key)
            while len(self._cache) > 64:
                self._cache.popitem(last=False)
            return copy.deepcopy(data)


class ArxivScheduler:
    """全局 arXiv 请求调度器：保证物理级严格单并发、请求结束后合规冷却、前台请求优先与防雪崩熔断"""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_completed_at = 0.0
        self._cooldown_until = 0.0
        self._consecutive_timeouts = 0
        self._user_last_active_at = 0.0

    def mark_user_active(self):
        """记录前台用户主动检索交互时刻"""
        self._user_last_active_at = time.monotonic()

    def is_user_active(self, window: float = 30.0) -> bool:
        """检查用户在最近 window 秒内是否有前台检索交互"""
        return (time.monotonic() - self._user_last_active_at) < window

    def is_in_cooldown(self) -> tuple[bool, float]:
        """检查当前是否处于 429 冷却或超时熔断冷静期"""
        now = time.monotonic()
        if now < self._cooldown_until:
            return True, self._cooldown_until - now
        return False, 0.0

    def trigger_cooldown(self, duration: float, reason: str = ""):
        """开启全局静默保护"""
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + duration)
        print(f"⚠️ [ArxivScheduler] 开启全局静默保护 {duration:.0f}s ({reason})，学术请求由本地快照与全球备选库平滑接管。")

    def reset_cooldown(self):
        """重置冷却与熔断状态 (用于测试及手动恢复)"""
        self._cooldown_until = 0.0
        self._consecutive_timeouts = 0

    @property
    def last_completed_at(self) -> float:
        return self._last_completed_at

    @last_completed_at.setter
    def last_completed_at(self, val: float):
        self._last_completed_at = val

    def record_success(self):
        """成功恢复计数器"""
        self._consecutive_timeouts = 0

    def record_timeout(self):
        """累积超时失败，连续2次超时自动触发60秒熔断，防止主线程雪崩卡死"""
        self._consecutive_timeouts += 1
        if self._consecutive_timeouts >= 2:
            self.trigger_cooldown(60.0, reason=f"连续超时 {self._consecutive_timeouts} 次，防雪崩自动熔断")

    def _fetch_http2_fast(self, req: urllib.request.Request, timeout: float) -> Optional[bytes]:
        """Use curl HTTP/2 while preserving server status and Retry-After headers."""
        import subprocess
        from email.parser import Parser

        # 检查是否处于单测 Mock 环境 (如 urllib.request.urlopen 被 patch，优先兼容单元测试)
        if hasattr(urllib.request, "urlopen") and getattr(urllib.request.urlopen, "__module__", "") != "urllib.request":
            return None

        try:
            url = req.full_url
            ua = req.headers.get("User-agent") or req.headers.get("User-Agent") or ARXIV_USER_AGENT
            accept_hdr = req.headers.get("Accept") or "application/atom+xml, application/xml"
            cmd = [
                "curl",
                "-s",
                "--http2",
                "--max-time", str(max(0.001, timeout)),
                "-H", f"User-Agent: {ua}",
                "-H", f"Accept: {accept_hdr}",
                "-w", "\n__HTTP_STATUS__:%{http_code}",
                url,
            ]
            proc = subprocess.run(
                cmd[:1] + ["--dump-header", "-"] + cmd[1:],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_search_timeout(timeout + 2),
            )
            # curl emits headers before the response body. HTTPS proxies may add
            # CONNECT headers, so consume all leading HTTP header blocks.
            stdout = proc.stdout
            headers = None
            while stdout.startswith(b"HTTP/") and b"\r\n\r\n" in stdout:
                header_block, stdout = stdout.split(b"\r\n\r\n", 1)
                headers = Parser().parsestr(header_block.decode("iso-8859-1").partition("\r\n")[2])
            marker = b"\n__HTTP_STATUS__:"
            if marker in stdout:
                body, code_str = stdout.rsplit(marker, 1)
                status_code = int(code_str.strip() or 0)
            else:
                body = stdout
                status_code = 200 if proc.returncode == 0 else 0

            if proc.returncode == 28:
                raise TimeoutError("arXiv 请求超过传输时限 (curl 28)")
            if proc.returncode == 2:
                # Some curl builds lack HTTP/2; urllib is the compatibility path.
                return None
            if proc.returncode != 0:
                raise urllib.error.URLError(f"arXiv curl 连接失败 (code={proc.returncode})")
            if status_code == 200 and b"<feed" in body:
                return body
            if status_code >= 400:
                raise urllib.error.HTTPError(url, status_code, "arXiv HTTP error", headers, None)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("arXiv curl 进程超时") from exc
        except (TimeoutError, urllib.error.URLError):
            raise
        except FileNotFoundError:
            return None
        return None

    def execute(
        self,
        req: urllib.request.Request,
        timeout: float = ARXIV_REQUEST_TIMEOUT,
        is_background: bool = False,
    ) -> bytes:
        """在全局唯一互斥排队下执行 arXiv HTTP 请求，彻底杜绝多线程并发撞车与 Tarpit 惩罚"""
        in_cd, remain = self.is_in_cooldown()
        if in_cd:
            raise TimeoutError(f"arXiv 处于静默保护期 (剩余 {int(remain)}s)")

        if is_background and self.is_user_active(30.0):
            raise ConnectionAbortedError("前台用户正在活跃检索，后台同步线程主动礼让避让网络通道")

        with _search_lock(self._lock):
            # 进入锁内二次检查冷却
            in_cd, remain = self.is_in_cooldown()
            if in_cd:
                raise TimeoutError(f"arXiv 处于静默保护期 (剩余 {int(remain)}s)")

            # 严格计算距离上一次请求【完成】的时间间隔，动态获取当前配置的最小间隔
            now = time.monotonic()
            min_interval = float(globals().get("ARXIV_MIN_INTERVAL", 3.2))
            wait_for = min_interval - (now - self._last_completed_at)
            if wait_for > 0:
                _search_pause(wait_for)

            try:
                timeout = _search_timeout(timeout)
                # Prefer curl HTTP/2 when available; protocol choice does not bypass limits.
                data = self._fetch_http2_fast(req, timeout=timeout)
                if data is None:
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        data = resp.read()
                self.record_success()
                return data
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 429):
                    self.trigger_cooldown(
                        _arxiv_retry_after_seconds(exc.headers),
                        reason=f"arXiv HTTP {exc.code}，停止请求并遵循服务端冷却时间",
                    )
                raise
            except (TimeoutError, urllib.error.URLError, socket.timeout) as e:
                self.record_timeout()
                raise
            finally:
                # 无论成功还是抛出异常，均在网络完全断开/读取结束时记录时刻
                self._last_completed_at = time.monotonic()


_ARXIV_SCHEDULER = ArxivScheduler()
_ARXIV_UPDATES = ArxivUpdateFeedClient()
_ARXIV_RATE_LOCK = _ARXIV_SCHEDULER._lock
_ARXIV_LAST_REQUEST_AT = 0.0
_ARXIV_429_COOLDOWN_UNTIL = 0.0

# 🌟 天工雷达专用高吞吐内存 LRU 缓存 (15分钟 TTL，秒级命中防 429)
_RADAR_SEARCH_CACHE: "OrderedDict[tuple[Any, ...], tuple[float, Dict[str, Any]]]" = OrderedDict()
_RADAR_SEARCH_CACHE_LOCK = threading.Lock()
_RADAR_SEARCH_CACHE_TTL = 900.0
_RADAR_SEARCH_CACHE_MAX = 128
ARXIV_USER_AGENT = os.getenv("ARXIV_USER_AGENT", "TianGongAcademicRadar/2.0")

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class ArxivPaper(BaseModel):
    paper_id: str
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    published_at: Optional[str] = None
    summary: str = ""
    pdf_url: Optional[str] = ""
    source: str = "arxiv"
    primary_category: Optional[str] = None
    categories: List[str] = []
    category_names_zh: List[str] = []
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    ai_summary: Optional[str] = None


class SearchResults(list):
    """List-compatible source response carrying failures alongside any usable records."""
    def __init__(self, items=(), source_error=None):
        super().__init__(items)
        self.source_error = source_error
        self.cache_hit = False


class AcademicResource(BaseModel):
    id: str
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    source: str = "arxiv"  # arxiv | crossref | openalex | semanticscholar | github
    summary: str = ""
    url: Optional[str] = ""
    stars: Optional[int] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    published_at: Optional[str] = None
    evidence_level: Optional[str] = None
    content_scope: Optional[str] = None
    version_ref: Optional[str] = None
    version_ref_type: Optional[str] = None


class AcademicSearchTerms(BaseModel):
    english_keywords: List[str] = Field(
        default_factory=list,
        description="保留当前任务各必要语义片段的英文检索表达；不限定数量"
    )


_OPENALEX_COOLDOWN_UNTIL = 0.0


_QUERY_STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "for", "to", "from", "with", "by", "at", "and", "or", "not",
    "recent", "latest", "research", "advances", "directions", "representative", "studies", "papers",
    "articles", "between", "since", "after", "before", "combined", "using", "based", "into", "about",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december",
}


def _clean_search_phrase(phrase: str) -> str:
    """移除搜索引擎伪语法 (如 site:xxx)、多余日期限定与引号，保留纯净学术词。"""
    if not phrase:
        return ""
    cleaned = re.sub(r"\bsite:[^\s)]+", " ", phrase, flags=re.I)
    cleaned = re.sub(
        r"\b(?:from|between|since|after|before)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{4})\b[^;]*",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r'["\'`()]+', " ", cleaned)
    cleaned = re.sub(r"\b(?:AND|OR|NOT)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ,，-")


def _extract_academic_phrases(query: str) -> List[str]:
    """按显式任务表达分隔符分组，保留各组完整术语和关系，清理无效搜索语法。"""
    if not query:
        return []
    raw_groups = re.split(r"[;；\n]+", query)
    groups = [_clean_search_phrase(item) for item in raw_groups]
    return list(dict.fromkeys(item for item in groups if item))


def _format_single_arxiv_clause(phrase: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9][\w-]*", phrase) if w.lower() not in _QUERY_STOP_WORDS and not re.fullmatch(r"20\d{2}", w)]
    if not words:
        return f'all:"{phrase}"'
    if len(words) == 1:
        return f"all:{words[0]}"
    core_terms = words[:6]
    return f"all:({' AND '.join(core_terms)})"


def _build_arxiv_search_expression(phrases: List[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return _format_single_arxiv_clause(phrases[0])
    return " OR ".join(_format_single_arxiv_clause(phrase) for phrase in phrases[:3])


def _build_arxiv_fallback_expressions(phrases: List[str], primary_expr: str) -> List[str]:
    """Build progressively relaxed arXiv clauses when an over-constrained AND query yields 0 hits."""
    candidates: List[str] = []
    if not phrases:
        return candidates
    if len(phrases) > 1:
        candidates.append(_format_single_arxiv_clause(phrases[0]))
    words = [
        w for w in re.findall(r"[A-Za-z0-9][\w-]*", phrases[0])
        if w.lower() not in _QUERY_STOP_WORDS and not re.fullmatch(r"20\d{2}", w)
    ]
    if len(words) >= 3:
        candidates.append(f"all:({' AND '.join(words[:2])})")
    seen = {primary_expr}
    unique: List[str] = []
    for expr in candidates:
        if expr and expr not in seen:
            seen.add(expr)
            unique.append(expr)
    return unique


def _build_github_repository_query(query: str) -> str:
    """Keep full semantic expressions; semicolon-separated expressions are alternatives."""
    phrases = _extract_academic_phrases(query)
    if not phrases:
        return query.strip()
    return " OR ".join(phrases[:3])


def extract_prf_terms_from_papers(query: str, records: List[Dict[str, Any]], max_terms: int = 4) -> str:
    """Pseudo-Relevance Feedback: extract real subfield terms from initial retrieved titles/abstracts."""
    from collections import Counter
    if not records:
        return ""
    query_tokens = {w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query or "")}
    phrase_counts: Counter[str] = Counter()
    for rec in records[:6]:
        text = f"{rec.get('title') or rec.get('doc_name') or ''} {(rec.get('summary') or rec.get('text') or '')[:900]}"
        # Capture hyphenated terms, acronyms (len>=2), and capitalized multi-word technical phrases
        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9]+(?:[- ][A-Z][A-Za-z0-9]+){1,2}|[A-Z]{2,8}|[A-Za-z0-9]+-[A-Za-z0-9-]+)\b",
            text,
        )
        seen_in_doc = set()
        for cand in candidates:
            norm = cand.strip()
            low = norm.lower()
            if len(norm) < 2 or low in _QUERY_STOP_WORDS or low in query_tokens:
                continue
            if any(tok in _QUERY_STOP_WORDS for tok in low.split()):
                continue
            if re.fullmatch(r"(?:DOI|URL|PDF|HTTP|HTTPS|API|TL|DR|ARXIV|GITHUB|OPENALEX|CROSSREF)", norm, re.I):
                continue
            if low not in seen_in_doc:
                seen_in_doc.add(low)
                phrase_counts[norm] += 1
    ranked = [term for term, _ in phrase_counts.most_common(max_terms)]
    return " ".join(ranked)


class OnlinePaperSearchAgent:
    """多源学术文献与开源代码生态发现智能体 (arXiv + CrossRef + GitHub)"""

    def __init__(self):
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise ValueError("[错误] 未在 .env 中正确配置 OPENAI_BASE_URL 或 OPENAI_API_KEY！")
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._keyword_cache: Dict[str, str] = {}

    def mark_user_active(self):
        """标记用户前台活跃，促使后台同步任务主动礼让"""
        _ARXIV_SCHEDULER.mark_user_active()

    def is_user_active(self, window: float = 30.0) -> bool:
        """检查用户在最近 window 秒内是否有前台检索交互"""
        return _ARXIV_SCHEDULER.is_user_active(window=window)

    def arxiv_cooldown_remaining(self) -> float:
        """Expose the shared arXiv circuit-breaker state to background schedulers."""
        _, remaining = _ARXIV_SCHEDULER.is_in_cooldown()
        return remaining

    def extract_academic_keywords(self, query: str) -> str:
        """使用大模型直接提取核心学术概念并精准翻译为英文学术关键词（纯大模型提取，确保信息高保真准确）"""
        cache_k = query.strip()
        if cache_k in self._keyword_cache:
            return self._keyword_cache[cache_k]

        prompt = build_academic_keyword_prompt(query)
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=get_default_llm_model(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                max_tokens=256,
                    timeout=min(get_llm_timeout(100.0), 25.0),
                )
                raw = (resp.choices[0].message.content or "").strip()
                clean_terms = raw.replace('"', '').replace("'", "").replace("\n", " ").strip()
                if clean_terms:
                    print(f"🎯 [LLM Academic Keywords] 大模型精准提炼并翻译英文学术词: \"{clean_terms}\"")
                    self._keyword_cache[cache_k] = clean_terms
                    return clean_terms
            except Exception as e:
                print(f"⚠️ [Keyword Extraction Fast Fallback (attempt {attempt}/{max_retries})]: {e}")

        # 兜底保留原句中的英文专有术语/缩写，避免因翻译超时而彻底跳过 arXiv 等英文学术库
        ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{1,}", query)
        if ascii_terms:
            dedup_ascii = " ".join(dict.fromkeys(ascii_terms))
            print(f"ℹ️ [Keyword Extraction ASCII Fallback] 保留英文学术词: \"{dedup_ascii}\"")
            return dedup_ascii
        return query.strip()

    @_cached_academic_search("arxiv")
    def search_arxiv(
        self,
        query: str,
        max_results: int = 5,
        time_constraint: Optional[Dict[str, Any]] = None,
    ) -> List[ArxivPaper]:
        """调用 arXiv API 检索学术论文预印本 (基于短语级布尔检索与动态时间约束)"""
        time_constraint = resolve_retrieval_time_constraint(time_constraint)
        date_bounds = retrieval_date_bounds(time_constraint)
        # 1. 标记用户活跃，压制后台可能存在的雷达同步线程，独占网络带宽
        self.mark_user_active()

        if re.search(r"[\u4e00-\u9fff]", query):
            translated = self.extract_academic_keywords(query)
            if translated and not re.search(r"[\u4e00-\u9fff]", translated):
                query = translated

        phrases = _extract_academic_phrases(query)
        arxiv_query = _build_arxiv_search_expression(phrases)
        if not arxiv_query:
            clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
            arxiv_query = f'all:"{clean_q}"'

        sort_by_date = bool(time_constraint and time_constraint.get("sort_by_date"))

        # 闭合时域区间支持: 针对特定年份或历史区间 (如 2021 或 2018-2020) 进行严格时域拦截
        date_filter = ""
        if date_bounds:
            start_day, end_day = (value.replace("-", "") for value in date_bounds)
            date_filter = f" AND submittedDate:[{start_day}000000 TO {end_day}235959]"
            arxiv_query += date_filter

        params = {
            "search_query": arxiv_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate" if sort_by_date else "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": ARXIV_USER_AGENT,
                "Accept": "application/atom+xml, application/xml",
            },
        )
        cache_key = (arxiv_query.lower(), max_results, sort_by_date)
        now = time.monotonic()
        with _ARXIV_CACHE_LOCK:
            cached = _ARXIV_CACHE.get(cache_key)
            if cached:
                cached_at, cached_papers = cached
                ttl = ARXIV_CACHE_TTL if cached_papers else ARXIV_EMPTY_CACHE_TTL
                if now - cached_at < ttl:
                    _ARXIV_CACHE.move_to_end(cache_key)
                    return [paper.model_copy(deep=True) for paper in cached_papers]
                _ARXIV_CACHE.pop(cache_key, None)

        last_error: Optional[Exception] = None
        for attempt in range(ARXIV_MAX_RETRIES + 1):
            try:
                xml_data = _ARXIV_SCHEDULER.execute(
                    req,
                    timeout=ARXIV_REQUEST_TIMEOUT,
                    is_background=False,
                )
                papers = self._parse_atom_feed(xml_data)
                if not papers:
                    primary_expr = arxiv_query[:-len(date_filter)] if date_filter else arxiv_query
                    for relaxed_expr in _build_arxiv_fallback_expressions(phrases, primary_expr):
                        fallback_query = relaxed_expr + date_filter
                        fb_params = {
                            "search_query": fallback_query,
                            "start": 0,
                            "max_results": max_results,
                            "sortBy": "submittedDate" if sort_by_date else "relevance",
                            "sortOrder": "descending",
                        }
                        fb_url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(fb_params)}"
                        fb_req = urllib.request.Request(
                            fb_url,
                            headers={
                                "User-Agent": ARXIV_USER_AGENT,
                                "Accept": "application/atom+xml, application/xml",
                            },
                        )
                        try:
                            fb_xml = _ARXIV_SCHEDULER.execute(
                                fb_req,
                                timeout=ARXIV_REQUEST_TIMEOUT,
                                is_background=False,
                            )
                            fb_papers = self._parse_atom_feed(fb_xml)
                            if fb_papers:
                                papers = fb_papers
                                print(f"🔄 [ArxivSearch] 逐级放宽检索命中 {len(papers)} 篇: {relaxed_expr}")
                                break
                        except Exception as fb_err:
                            print(f"ℹ️ [ArxivSearch] 逐级放宽检索提示: {fb_err}")
                            break

                with _ARXIV_CACHE_LOCK:
                    _ARXIV_CACHE[cache_key] = (time.monotonic(), papers)
                    _ARXIV_CACHE.move_to_end(cache_key)
                    while len(_ARXIV_CACHE) > ARXIV_CACHE_MAX_ENTRIES:
                        _ARXIV_CACHE.popitem(last=False)
                return [paper.model_copy(deep=True) for paper in papers]
            except urllib.error.HTTPError as e:
                last_error = e
                # The scheduler has already recorded server restrictions for all
                # callers. Retrying 403/429 here would prolong those restrictions.
                break
            except (TimeoutError, urllib.error.URLError, socket.timeout, ConnectionError) as e:
                last_error = e
                if attempt >= ARXIV_MAX_RETRIES or _ARXIV_SCHEDULER.is_in_cooldown()[0]:
                    break
                try:
                    _search_pause(1.0 + random.uniform(0, 0.5))
                except TimeoutError:
                    break
            except Exception as e:
                last_error = e
                break

        print(f"ℹ️ [ArxivSearch] arXiv 响应超时或受限 ({last_error})，自动由 CrossRef/OpenAlex 全球学术库平滑接管证据链。")
        return SearchResults(source_error=type(last_error).__name__)

    def _radar_openalex_fallback(
        self,
        query: Optional[str],
        major: Optional[str],
        sub_categories: Optional[List[str]],
        sort_by: str,
        start: int,
        max_results: int,
        arxiv_query: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Fallback to OpenAlex arXiv-indexed works when arXiv RSS/API times out or is rate-limited."""
        if time.time() < _OPENALEX_COOLDOWN_UNTIL:
            return None
        try:
            from academic_taxonomy import ACADEMIC_TAXONOMY, get_subcategory_meta
            from collections import Counter

            mailto = os.getenv("OPENALEX_MAILTO", "").strip() or "yunshan_scholar@research-institute.org"
            page_size = max(1, min(int(max_results or 15), 50))
            page_num = (max(0, int(start or 0)) // page_size) + 1

            search_text = (query or "").strip()
            if not search_text and sub_categories:
                names = []
                for code in sub_categories[:3]:
                    meta = get_subcategory_meta(code.rstrip("*"))
                    if meta and meta.get("name_en"):
                        names.append(meta["name_en"])
                search_text = " ".join(names).strip()
            if not search_text and major and major in ACADEMIC_TAXONOMY:
                search_text = ACADEMIC_TAXONOMY[major].get("name_en") or ""

            params: Dict[str, Any] = {
                "filter": "primary_location.source.id:S4306400194,has_abstract:true",
                "per-page": page_size,
                "page": page_num,
                "mailto": mailto,
            }
            if search_text:
                params["search"] = search_text
            if sort_by == "submittedDate" or not (query or "").strip():
                params["sort"] = "publication_date:desc"

            url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": f"YunshanScholar/2.0 (mailto:{mailto})"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = data.get("results") or []
            if not results:
                return None

            fallback_cat = (sub_categories[0].rstrip("*") if sub_categories else (major or "cs"))
            papers_out: List[Dict[str, Any]] = []
            for w in results:
                title = (w.get("title") or "").strip()
                if not title:
                    continue
                inv = w.get("abstract_inverted_index") or {}
                abstract = ""
                if isinstance(inv, dict) and inv:
                    pos_map = {}
                    for word, positions in inv.items():
                        for pos in positions:
                            pos_map[pos] = word
                    abstract = " ".join(pos_map[k] for k in sorted(pos_map.keys()))
                authors = [
                    (a.get("author") or {}).get("display_name", "")
                    for a in (w.get("authorships") or [])[:8]
                    if (a.get("author") or {}).get("display_name")
                ]
                pub_date = w.get("publication_date") or (f"{w.get('publication_year')}-01-01" if w.get("publication_year") else "")
                loc = w.get("primary_location") or {}
                landing = loc.get("landing_page_url") or w.get("doi") or w.get("id") or ""
                pdf_url = loc.get("pdf_url") or landing
                arxiv_m = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", f"{landing} {pdf_url} {w.get('doi') or ''}")
                paper_id = arxiv_m.group(1) if arxiv_m else str(w.get("id") or "").rstrip("/").split("/")[-1]
                if arxiv_m and not str(pdf_url).startswith("https://arxiv.org/pdf/"):
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_m.group(1)}.pdf"
                papers_out.append({
                    "paper_id": paper_id,
                    "title": title,
                    "authors": authors,
                    "summary": abstract,
                    "published": pub_date,
                    "pdf_url": pdf_url,
                    "primary_category": fallback_cat,
                    "categories": [fallback_cat],
                })

            if not papers_out:
                return None

            meta_total = int((data.get("meta") or {}).get("count") or len(papers_out))
            cat_meta = get_subcategory_meta(fallback_cat)
            return {
                "query_expression": arxiv_query or search_text,
                "papers": papers_out,
                "total_fetched": len(papers_out),
                "total_results": meta_total,
                "total_results_scope": "openalex_arxiv_fallback",
                "sub_distribution": [{
                    "code": fallback_cat,
                    "name_zh": cat_meta["name_zh"] if cat_meta else fallback_cat,
                    "count": len(papers_out),
                }],
                "source": "openalex_fallback",
                "update_status": "live",
                "is_cached_snapshot": False,
            }
        except Exception as exc:
            print(f"⚠️ [RadarOpenAlexFallback]: {exc}")
            return None

    def _radar_latest_updates(self, major, sub_categories, start, max_results):
        """Read a daily announcement batch without using the historical query API."""
        from academic_taxonomy import build_arxiv_query_expression, get_subcategory_meta
        from collections import Counter

        batch = _ARXIV_UPDATES.fetch(_arxiv_feed_categories(major, sub_categories))
        all_papers = batch["papers"]
        start = max(0, start)
        page_size = max(1, min(max_results, 60))
        papers = all_papers[start:start + page_size]
        counts = Counter(cat for paper in papers for cat in set(paper.get("categories", [])))
        distribution = []
        for code, count in counts.most_common():
            meta = get_subcategory_meta(code)
            distribution.append({"code": code, "name_zh": meta["name_zh"] if meta else code, "count": count})
        return {
            "query_expression": build_arxiv_query_expression(major=major, sub_categories=sub_categories),
            "papers": papers,
            "total_fetched": len(papers),
            "total_results": len(all_papers),
            "total_results_scope": "announcement_batch",
            "sub_distribution": distribution,
            "source": "arxiv_rss",
            "update_status": "live" if all_papers else "no_announcements",
            "is_cached_snapshot": False,
            "feed_updated_at": batch["feed_updated_at"],
            "fetched_at": batch["fetched_at"],
            "snapshot_synced_at": batch["fetched_at"],
            "feed_limit_reached": len(all_papers) >= 2000,
        }

    def search_radar(
        self,
        query: Optional[str] = None,
        major: Optional[str] = None,
        sub_categories: Optional[List[str]] = None,
        search_in: str = "all",
        sort_by: str = "submittedDate",  # submittedDate | relevance
        start: int = 0,
        max_results: int = 15,
        is_background: bool = False,
    ) -> Dict[str, Any]:
        """天工雷达专用的全学科精准检索引擎，具备内存LRU秒级防限流、本地快照无缝容灾与子学科分布统计"""
        import copy
        from academic_taxonomy import build_arxiv_query_expression, get_subcategory_meta
        from collections import Counter

        # 若是前台用户交互，主动标记前台活跃以压制后台同步
        if not is_background:
            self.mark_user_active()

        # Latest browsing uses the official daily feed on a separate host. Search
        # terms and relevance sorting continue through the existing search path.
        latest_browse = not (query or "").strip() and sort_by == "submittedDate"
        if latest_browse:
            try:
                rss_res = self._radar_latest_updates(major, sub_categories, start, max_results)
                if rss_res and rss_res.get("papers"):
                    return rss_res
                print("ℹ️ [RadarUpdates]: 当日 arXiv RSS 无新公告（如周末休刊），自动回退至近期论文查询/OpenAlex。")
            except Exception as exc:
                print(f"⚠️ [RadarUpdates]: 更新订阅暂不可用 ({exc})，尝试原查询接口。")

        # 1. 检查内存 LRU 缓存 (命中直接 0 毫秒秒出，彻底免疫 429 与网络超时)
        cache_key = (
            (query or "").strip().lower(),
            major or "",
            tuple(sorted(sub_categories or [])),
            search_in,
            sort_by,
            start,
            max_results,
        )
        with _RADAR_SEARCH_CACHE_LOCK:
            if cache_key in _RADAR_SEARCH_CACHE:
                cached_time, cached_res = _RADAR_SEARCH_CACHE[cache_key]
                ttl = min(_RADAR_SEARCH_CACHE_TTL, 60.0) if latest_browse else _RADAR_SEARCH_CACHE_TTL
                if time.monotonic() - cached_time < ttl and not (latest_browse and cached_res.get("is_cached_snapshot")):
                    return copy.deepcopy(cached_res)

        arxiv_query = build_arxiv_query_expression(
            query=query,
            major=major,
            sub_categories=sub_categories,
            search_in=search_in,
        )

        params = {
            "search_query": arxiv_query,
            "start": max(0, start),
            "max_results": max(1, min(max_results, 60)),
            "sortBy": "submittedDate" if sort_by == "submittedDate" else "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": ARXIV_USER_AGENT,
                "Accept": "application/atom+xml, application/xml",
            },
        )

        papers: List[ArxivPaper] = []
        total_results_count = 0
        api_succeeded = False
        try:
            xml_data = _ARXIV_SCHEDULER.execute(
                req,
                timeout=ARXIV_REQUEST_TIMEOUT,
                is_background=is_background,
            )
            papers = self._parse_atom_feed(xml_data)
            api_succeeded = True
            try:
                root = ET.fromstring(xml_data)
                opensearch_ns = "{http://a9.com/-/spec/opensearch/1.1/}"
                total_elem = root.findtext(f"{opensearch_ns}totalResults")
                if total_elem and total_elem.isdigit():
                    total_results_count = int(total_elem)
            except Exception:
                pass
        except urllib.error.HTTPError as e:
            print(f"⚠️ [SearchRadar Note]: arXiv HTTP 异常 ({e})，平滑切换至本地经卷快照。")
        except Exception as e:
            print(f"⚠️ [SearchRadar Note]: arXiv 请求受阻或避让 ({e})，平滑切换至本地经卷快照。")

        # 2. 若 arXiv RSS/API 均超时或受限，先尝试 OpenAlex (arXiv 源 + Polite Pool) 实时拉取最新论文
        if not papers and not api_succeeded:
            oa_fallback = self._radar_openalex_fallback(
                query=query,
                major=major,
                sub_categories=sub_categories,
                sort_by=sort_by,
                start=start,
                max_results=max_results,
                arxiv_query=arxiv_query,
            )
            if oa_fallback and oa_fallback.get("papers"):
                if not latest_browse:
                    with _RADAR_SEARCH_CACHE_LOCK:
                        _RADAR_SEARCH_CACHE[cache_key] = (time.monotonic(), copy.deepcopy(oa_fallback))
                return oa_fallback

        # 3. 若在线源均不可用，自动无缝触发本地快照容灾
        if not papers and not api_succeeded:
            try:
                from radar_sync_engine import RADAR_CACHE_DIR
                sub_key = sub_categories[0] if (sub_categories and len(sub_categories) == 1) else "all"
                safe_sub = sub_key.replace(".", "_").replace("*", "_all")
                snapshot_file = RADAR_CACHE_DIR / f"radar_{major or 'CS'}_{safe_sub}.json"
                if not snapshot_file.exists():
                    snapshot_file = RADAR_CACHE_DIR / f"radar_{major or 'CS'}_all.json"
                if snapshot_file.exists():
                    with open(snapshot_file, "r", encoding="utf-8") as sf:
                        snap_data = json.load(sf)
                        if snap_data and snap_data.get("papers"):
                            snap_data["is_cached_snapshot"] = True
                            if latest_browse:
                                snapshot_papers = snap_data["papers"]
                                if sub_categories:
                                    selected = [c.rstrip("*") for c in sub_categories]
                                    snapshot_papers = [p for p in snapshot_papers if any(
                                        cat == sub or cat.startswith(sub + ".")
                                        for cat in p.get("categories", []) for sub in selected
                                    )]
                                snap_data["papers"] = snapshot_papers[max(0, start):max(0, start) + max(1, min(max_results, 60))]
                                snap_data["total_fetched"] = len(snap_data["papers"])
                                snap_data["total_results"] = len(snapshot_papers)
                                snap_data["total_results_scope"] = "snapshot"
                                snap_data["update_status"] = "stale_snapshot"
                                snap_data["query_expression"] = arxiv_query
                                snap_data["snapshot_synced_at"] = snap_data.get("snapshot_synced_at") or datetime.fromtimestamp(snapshot_file.stat().st_mtime, timezone.utc).isoformat()
                                # A failed refresh must not poison the 15-minute
                                # successful-search cache with a stale snapshot.
                                return snap_data
                            # 写入内存缓存，避免重复磁盘I/O与等待
                            with _RADAR_SEARCH_CACHE_LOCK:
                                _RADAR_SEARCH_CACHE[cache_key] = (time.monotonic(), copy.deepcopy(snap_data))
                            return snap_data
            except Exception:
                pass

        # 统计当前查询结果在各子学科中的真实分布
        sub_dist_counter = Counter()
        for p in papers:
            if p.primary_category:
                sub_dist_counter[p.primary_category] += 1
            for cat in p.categories:
                if cat != p.primary_category:
                    sub_dist_counter[cat] += 1

        sub_distribution = []
        for cat_code, count in sub_dist_counter.most_common():
            meta = get_subcategory_meta(cat_code)
            sub_distribution.append({
                "code": cat_code,
                "name_zh": meta["name_zh"] if meta else cat_code,
                "count": count,
            })

        final_result = {
            "query_expression": arxiv_query,
            "total_fetched": len(papers),
            "total_results": max(total_results_count, len(papers)),
            "papers": [p.model_dump() for p in papers],
            "sub_distribution": sub_distribution,
        }
        if latest_browse:
            final_result.update({
                "source": "arxiv_api",
                "update_status": ("live" if papers else "no_announcements") if api_succeeded else "unavailable",
                "is_cached_snapshot": False,
            })
            if api_succeeded:
                final_result["fetched_at"] = datetime.now(timezone.utc).isoformat()
                final_result["snapshot_synced_at"] = final_result["fetched_at"]

        # Cache successful empty queries too; failed requests are not zero matches.
        if api_succeeded:
            with _RADAR_SEARCH_CACHE_LOCK:
                _RADAR_SEARCH_CACHE[cache_key] = (time.monotonic(), copy.deepcopy(final_result))
                _RADAR_SEARCH_CACHE.move_to_end(cache_key)
                while len(_RADAR_SEARCH_CACHE) > _RADAR_SEARCH_CACHE_MAX:
                    _RADAR_SEARCH_CACHE.popitem(last=False)

        return final_result

    @_cached_academic_search("crossref")
    def search_crossref(
        self,
        query: str,
        max_results: int = 4,
        time_constraint: Optional[Dict[str, Any]] = None,
    ) -> List[AcademicResource]:
        """调用 CrossRef API 检索正式发表的权威期刊/顶会出版文献 (基于闭合时域区间过滤)"""
        time_constraint = resolve_retrieval_time_constraint(time_constraint)
        if re.search(r"[\u4e00-\u9fff]", query):
            translated = self.extract_academic_keywords(query)
            if translated and not re.search(r"[\u4e00-\u9fff]", translated):
                query = translated

        phrases = _extract_academic_phrases(query)
        short_q = " ".join(phrases) if phrases else query.strip()
        params = {"query": short_q, "rows": max_results}
        date_bounds = retrieval_date_bounds(time_constraint)
        if date_bounds:
            params["filter"] = f"from-pub-date:{date_bounds[0]},until-pub-date:{date_bounds[1]}"
        mailto = os.getenv("OPENALEX_MAILTO", "").strip() or "yunshan_scholar@research-institute.org"
        url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"YunshanScholar/2.0 (mailto:{mailto})"}
        )

        resources = []
        source_error = None
        try:
            with urllib.request.urlopen(req, timeout=_search_timeout(12)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("message", {}).get("items", [])
            for it in items:
                title = (it.get("title") or [""])[0]
                if not title:
                    continue
                year = None
                date_parts = next((it[field]["date-parts"] for field in ("published", "published-print", "published-online", "issued")
                    if isinstance(it.get(field), dict) and it[field].get("date-parts")), [[]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0]
                doi = it.get("DOI", "")
                authors = [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in it.get("author", [])
                ]
                container = (it.get("container-title") or [""])[0]
                summary = f"发表期刊/会议: 《{container}》 | DOI: {doi}" if container else f"DOI: {doi}"
                abstract = re.sub(r"<[^>]+>", " ", str(it.get("abstract") or ""))
                from html import unescape
                abstract = re.sub(r"\s+", " ", unescape(abstract)).strip()
                if abstract:
                    summary = abstract + "\n\n" + summary
                resources.append(AcademicResource(
                    id=doi or title,
                    title=title,
                    authors=authors[:4],
                    year=year,
                    source="crossref",
                    evidence_level="abstract" if abstract else "bibliographic_metadata",
                    content_scope="indexed abstract" if abstract else "bibliographic record only",
                    summary=summary,
                    url=f"https://doi.org/{doi}" if doi else "",
                    doi=doi
                ))
        except urllib.error.HTTPError as e:
            source_error = f"HTTP {e.code}"
            _source_http_failure(self, "crossref", e)
            print(f"⚠️ [CrossRefSearch Note]: {source_error}")
        except Exception as e:
            source_error = type(e).__name__
            print(f"⚠️ [CrossRefSearch Note]: {e}")
        return SearchResults(resources, source_error)

    @_cached_academic_search("openalex")
    def search_openalex(
        self,
        query: str,
        max_results: int = 3,
        time_constraint: Optional[Dict[str, Any]] = None,
    ) -> List[AcademicResource]:
        """调用全球开放学术图谱 OpenAlex API 检索同行评议权威期刊/顶会文献与高被引引用量 (基于动态时间约束过滤)"""
        time_constraint = resolve_retrieval_time_constraint(time_constraint)
        global _OPENALEX_COOLDOWN_UNTIL
        import time
        now = time.time()
        if now < _OPENALEX_COOLDOWN_UNTIL:
            return SearchResults(source_error="rate_limit_cooldown")

        if re.search(r"[\u4e00-\u9fff]", query):
            translated = self.extract_academic_keywords(query)
            if translated and not re.search(r"[\u4e00-\u9fff]", translated):
                query = translated

        phrases = _extract_academic_phrases(query)
        if phrases:
            search_term = " ".join(phrases[:2])
        else:
            clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
            search_term = clean_q or query.strip()

        mailto = os.getenv("OPENALEX_MAILTO", "").strip() or "yunshan_scholar@research-institute.org"
        params = {
            "search": search_term,
            "per-page": max_results,
            "mailto": mailto,
        }
        date_bounds = retrieval_date_bounds(time_constraint)
        if date_bounds and (time_constraint.get("start_date") or time_constraint.get("end_date")):
            params["filter"] = f"from_publication_date:{date_bounds[0]},to_publication_date:{date_bounds[1]}"
        elif date_bounds:
            start_y = int(time_constraint["start_year"])
            end_y = int(time_constraint.get("end_year") or start_y)
            if start_y == end_y:
                params["filter"] = f"publication_year:{start_y}"
            else:
                params["filter"] = f"publication_year:{start_y}-{end_y}"
        url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"YunshanScholar/2.0 (mailto:{mailto})"}
        )

        resources = []
        source_error = None
        try:
            data = None
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(req, timeout=_search_timeout(10)) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as http_exc:
                    if getattr(http_exc, "code", None) in (502, 503, 504) and attempt == 0:
                        _search_pause(1.0)
                        continue
                    raise
            if data is None:
                data = {}
            for it in data.get("results", []):
                title = it.get("title") or ""
                if not title:
                    continue
                year = it.get("publication_year")
                doi = it.get("doi") or ""
                cited = it.get("cited_by_count", 0)
                venue = ((it.get("primary_location") or {}).get("source") or {}).get("display_name") or "来源未标注"
                authorships = it.get("authorships", [])
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in authorships if a.get("author", {}).get("display_name")
                ]
                # 提取最佳开放获取全文 PDF 链接 (优先提取真实开放仓储地址)
                pdf_url = ""
                best_oa = it.get("best_oa_location") or {}
                if best_oa.get("pdf_url"):
                    pdf_url = best_oa["pdf_url"]
                else:
                    for loc in it.get("locations", []):
                        if loc.get("is_oa") and loc.get("pdf_url"):
                            pdf_url = loc["pdf_url"]
                            break
                if not pdf_url:
                    pdf_url = doi or str(it.get("id", ""))

                summary = f"来源: 《{venue}》 | 被引: {cited} 次 | DOI: {doi}"
                positions = {}
                for word, offsets in (it.get("abstract_inverted_index") or {}).items():
                    if isinstance(offsets, list):
                        for offset in offsets:
                            if type(offset) is int and offset >= 0:
                                positions[offset] = word
                abstract = " ".join(str(positions[idx]) for idx in sorted(positions))
                if abstract:
                    summary = abstract + "\n\n" + summary
                resources.append(AcademicResource(
                    id=doi or str(it.get("id", title)),
                    title=title,
                    authors=authors[:4],
                    year=year,
                    source="openalex",
                    published_at=it.get("publication_date"),
                    evidence_level="abstract" if abstract else "bibliographic_metadata",
                    content_scope="indexed abstract" if abstract else "bibliographic record only",
                    summary=summary,
                    url=pdf_url,
                    doi=doi,
                    citation_count=cited,
                    venue=venue
                ))
        except urllib.error.HTTPError as he:
            source_error = f"HTTP {he.code}"
            _source_http_failure(self, "openalex", he)
            if he.code == 429:
                _OPENALEX_COOLDOWN_UNTIL = time.time() + _retry_after_seconds(he.headers, 180.0)
            else:
                print(f"⚠️ [OpenAlexSearch Note]: {he}")
        except Exception as e:
            source_error = type(e).__name__
            print(f"⚠️ [OpenAlexSearch Error]: {e}")
        return SearchResults(resources, source_error)

    @_cached_academic_search("semanticscholar")
    def search_semanticscholar(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 Semantic Scholar API 检索文献极速摘要与学术引用影响力 (防噪高信噪比)"""
        if _s2_in_cooldown():
            return SearchResults([], "RateLimited")
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        params = {
            "query": clean_q,
            "limit": max_results,
            "fields": "title,authors,year,abstract,citationCount,influentialCitationCount,venue,url,tldr"
        }
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=_s2_headers())

        resources = []
        source_error = None
        try:
            data = None
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(req, timeout=_search_timeout(10)) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as http_exc:
                    if getattr(http_exc, "code", None) in (502, 503, 504) and attempt == 0:
                        _search_pause(1.0)
                        continue
                    raise
            if data is None:
                data = {}
            for it in data.get("data", []):
                title = it.get("title") or ""
                if not title:
                    continue
                year = it.get("year")
                cited = it.get("citationCount", 0)
                venue = it.get("venue") or "国际同行评议学术出版物"
                authors = [a.get("name", "") for a in it.get("authors", []) if a.get("name")]
                tldr = ((it.get("tldr") or {}).get("text") or "").strip()
                abstract = (it.get("abstract") or "").strip()
                body_parts = []
                if tldr:
                    body_parts.append(f"📌 TL;DR: {tldr}")
                if abstract:
                    body_parts.append(abstract[:1200])
                body_parts.append(f"📊 引用: {cited} 次 | 录用: {venue}")
                summary = "\n".join(body_parts)
                resources.append(AcademicResource(
                    id=it.get("paperId", title),
                    title=title,
                    authors=authors[:4],
                    year=year,
                    source="semanticscholar",
                    evidence_level="abstract" if (abstract or tldr) else "bibliographic_metadata",
                    content_scope="indexed abstract" if (abstract or tldr) else "bibliographic record only",
                    summary=summary,
                    url=it.get("url") or "",
                    citation_count=cited,
                    venue=venue
                ))
        except urllib.error.HTTPError as he:
            source_error = "RateLimited" if he.code == 429 else type(he).__name__
            _source_http_failure(self, "semanticscholar", he)
            if he.code == 429:
                seconds = _retry_after_seconds(he.headers, 180.0)
                _s2_trigger_cooldown(seconds)
                print(f"ℹ️ [SemanticScholar Note]: 公共接口限流 (429)，暂停请求 {seconds:.0f}s；其他来源独立检索。")
            else:
                print(f"⚠️ [SemanticScholar Note]: {he}")
        except Exception as e:
            source_error = type(e).__name__
            print(f"⚠️ [SemanticScholar Note]: {e}")
        return SearchResults(resources, source_error)

    def search_citation_snowball(
        self,
        seed_cards: List[Dict[str, Any]],
        max_per_seed: int = 4,
        time_constraint: Optional[Dict[str, Any]] = None,
    ) -> List[AcademicResource]:
        """引文雪球扩展：基于首轮种子论文在 Semantic Scholar 图谱中抓取前向引用与后向参考文献，突破关键词猜词限制。"""
        from concurrent.futures import ThreadPoolExecutor

        time_constraint = resolve_retrieval_time_constraint(time_constraint)
        is_explicit_tc = bool(time_constraint and time_constraint.get("basis") != "default_three_years")
        date_bounds = retrieval_date_bounds(time_constraint) if is_explicit_tc else None
        min_year = int(date_bounds[0][:4]) if date_bounds else None
        max_year = int(date_bounds[1][:4]) if date_bounds else None

        seed_refs: List[tuple[str, str]] = []
        seen_refs = set()
        for card in seed_cards or []:
            if not isinstance(card, dict):
                continue
            title = str(card.get("title") or card.get("doc_name") or "").strip()
            arxiv_id = str(card.get("arxiv_id") or "").strip()
            doi = str(card.get("doi") or "").strip()
            raw_id = str(card.get("paper_id") or card.get("id") or "").strip()
            ref = ""
            if arxiv_id:
                clean_arx = re.sub(r"v\d+$", "", arxiv_id, flags=re.I)
                ref = f"ARXIV:{clean_arx}"
            elif doi:
                clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
                ref = f"DOI:{clean_doi}"
            elif re.fullmatch(r"[0-9a-f]{40}", raw_id, re.I):
                ref = raw_id
            if ref and ref not in seen_refs:
                seen_refs.add(ref)
                seed_refs.append((ref, title))
            if len(seed_refs) >= 2:
                break

        if not seed_refs:
            return SearchResults()

        def _fetch_openalex_citations(paper_ref: str, seed_title: str) -> List[AcademicResource]:
            doi = ""
            if paper_ref.startswith("DOI:"):
                doi = paper_ref[4:]
            elif paper_ref.startswith("ARXIV:"):
                doi = f"10.48550/arXiv.{paper_ref[6:]}"
            if not doi:
                return []
            mailto = os.getenv("OPENALEX_MAILTO", "").strip() or "yunshan_scholar@research-institute.org"
            lookup_url = f"https://api.openalex.org/works?filter=doi:{urllib.parse.quote(doi)}&per-page=1&mailto={urllib.parse.quote(mailto)}"
            lookup_req = urllib.request.Request(lookup_url, headers={"User-Agent": f"YunshanScholar/2.0 (mailto:{mailto})"})
            with urllib.request.urlopen(lookup_req, timeout=6) as resp:
                wdata = json.loads(resp.read().decode("utf-8"))
            works = wdata.get("results") or []
            if not works:
                return []
            work_id = str(works[0].get("id") or "").rstrip("/").split("/")[-1]
            if not work_id:
                return []
            cites_url = (
                f"https://api.openalex.org/works?filter=cites:{work_id}"
                f"&sort=cited_by_count:desc&per-page={max_per_seed}&mailto={urllib.parse.quote(mailto)}"
            )
            cites_req = urllib.request.Request(cites_url, headers={"User-Agent": f"YunshanScholar/2.0 (mailto:{mailto})"})
            with urllib.request.urlopen(cites_req, timeout=6) as resp:
                cdata = json.loads(resp.read().decode("utf-8"))
            out: List[AcademicResource] = []
            scope_seed = f"《{seed_title[:45]}》" if seed_title else paper_ref
            for w in cdata.get("results") or []:
                p_title = (w.get("title") or "").strip()
                if not p_title:
                    continue
                year = w.get("publication_year")
                if isinstance(year, int):
                    if min_year and year < min_year:
                        continue
                    if max_year and year > max_year:
                        continue
                cited = w.get("cited_by_count", 0)
                p_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(w.get("doi") or ""), flags=re.I)
                authors = [((a.get("author") or {}).get("display_name") or "") for a in (w.get("authorships") or [])[:4]]
                authors = [a for a in authors if a]
                venue = (((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "国际学术出版物")
                inv_abs = w.get("abstract_inverted_index")
                abstract = ""
                if isinstance(inv_abs, dict) and inv_abs:
                    pos_words = [(pos, word) for word, positions in inv_abs.items() for pos in positions]
                    pos_words.sort(key=lambda x: x[0])
                    abstract = " ".join(word for _, word in pos_words)
                summary = (
                    f"{abstract[:1000]}\n" if abstract else ""
                ) + f"🕸️ 引文雪球[前向引用(后续工作) <- {scope_seed}] | 📊 引用: {cited} 次 | 载体: {venue}"
                out.append(AcademicResource(
                    id=str(w.get("id") or p_doi or p_title),
                    title=p_title,
                    authors=authors,
                    year=year,
                    source="citation_snowball",
                    doi=p_doi or None,
                    evidence_level="abstract" if abstract else "bibliographic_metadata",
                    content_scope=f"openalex citation snowball from {paper_ref}",
                    summary=summary,
                    url=str(w.get("id") or (f"https://doi.org/{p_doi}" if p_doi else "")),
                    citation_count=cited,
                    venue=venue,
                ))
            return out

        def _fetch_edge(paper_ref: str, seed_title: str, edge: str) -> List[AcademicResource]:
            if _s2_in_cooldown():
                return _fetch_openalex_citations(paper_ref, seed_title) if edge == "citations" else []
            rel_key = "citingPaper" if edge == "citations" else "citedPaper"
            fields = "title,authors,year,abstract,citationCount,venue,url,externalIds"
            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(paper_ref, safe=':')}/{edge}"
                f"?fields={fields}&limit={max_per_seed}"
            )
            req = urllib.request.Request(url, headers=_s2_headers())
            out: List[AcademicResource] = []
            try:
                with urllib.request.urlopen(req, timeout=6) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as he:
                if he.code == 429:
                    _s2_trigger_cooldown(60.0)
                    return _fetch_openalex_citations(paper_ref, seed_title) if edge == "citations" else []
                raise
            for row in data.get("data", []):
                p = row.get(rel_key) if isinstance(row, dict) else None
                if not isinstance(p, dict):
                    continue
                p_title = (p.get("title") or "").strip()
                if not p_title:
                    continue
                year = p.get("year")
                if edge == "citations" and isinstance(year, int):
                    if min_year and year < min_year:
                        continue
                    if max_year and year > max_year:
                        continue
                cited = p.get("citationCount", 0)
                venue = p.get("venue") or "国际学术出版物"
                authors = [a.get("name", "") for a in (p.get("authors") or []) if a.get("name")]
                abstract = (p.get("abstract") or "").strip()
                ext = p.get("externalIds") or {}
                p_doi = ext.get("DOI") or ""
                rel_label = "前向引用(后续工作)" if edge == "citations" else "后向参考文献(基线/源头)"
                scope_seed = f"《{seed_title[:45]}》" if seed_title else paper_ref
                summary = (
                    f"{abstract[:1000]}\n" if abstract else ""
                ) + f"🕸️ 引文雪球[{rel_label} <- {scope_seed}] | 📊 引用: {cited} 次 | 载体: {venue}"
                out.append(AcademicResource(
                    id=p.get("paperId") or p_doi or p_title,
                    title=p_title,
                    authors=authors[:4],
                    year=year,
                    source="citation_snowball",
                    doi=p_doi or None,
                    evidence_level="abstract" if abstract else "bibliographic_metadata",
                    content_scope=f"citation snowball ({edge}) from {paper_ref}",
                    summary=summary,
                    url=p.get("url") or (f"https://doi.org/{p_doi}" if p_doi else ""),
                    citation_count=cited,
                    venue=venue,
                ))
            return out

        resources: List[AcademicResource] = []
        source_error = None
        tasks = [(ref, title, edge) for ref, title in seed_refs for edge in ("citations", "references")]
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = [executor.submit(_fetch_edge, ref, title, edge) for ref, title, edge in tasks]
            for fut in futures:
                try:
                    resources.extend(fut.result(timeout=7))
                except Exception as e:
                    source_error = source_error or type(e).__name__
        return SearchResults(resources, source_error)

    def search_europe_pmc(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 Europe PMC / PubMed API 检索全球生物医药、临床健康与生命科学顶刊 (免登录)"""
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        params = {
            "query": clean_q,
            "format": "json",
            "pageSize": max_results
        }
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchScholar/1.0"})

        resources = []
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("resultList", {}).get("result", [])
            for r in results:
                title = r.get("title") or ""
                if not title:
                    continue
                year = None
                try:
                    year = int(r.get("pubYear")) if r.get("pubYear") else None
                except Exception:
                    pass
                doi = r.get("doi") or ""
                journal = r.get("journalTitle") or "生物医药权威同行评议期刊"
                authors_str = r.get("authorString") or ""
                authors = [a.strip() for a in authors_str.split(",") if a.strip()]
                summary = f"🩺 生物医学顶刊: 《{journal}》 | DOI: {doi} | PMID: {r.get('pmid', '')}"
                resources.append(AcademicResource(
                    id=doi or r.get("id", title),
                    title=title,
                    authors=authors[:4],
                    year=year,
                    source="europe_pmc",
                    summary=summary,
                    url=f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/MED/{r.get('pmid', '')}",
                    doi=doi,
                    venue=journal
                ))
        except Exception as e:
            print(f"⚠️ [EuropePMCSearch Note]: {e}")
        return resources

    def search_inspire_hep(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 CERN INSPIRE-HEP API 检索全球物理学、量子场论、高能与凝聚态顶刊 (免登录)"""
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        params = {"q": clean_q, "size": max_results}
        url = f"https://inspirehep.net/api/literature?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchScholar/1.0"})

        resources = []
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hits = data.get("hits", {}).get("hits", [])
            for h in hits:
                meta = h.get("metadata", {})
                titles = meta.get("titles", [{}])
                title = titles[0].get("title", "") if titles else ""
                if not title:
                    continue
                year = meta.get("earliest_date", "")[:4]
                year_int = int(year) if year.isdigit() else None
                cites = meta.get("citation_count", 0)
                pubs = meta.get("publication_info", [{}])
                journal = pubs[0].get("journal_title", "权威学术出版物") if pubs else "权威学术出版物"
                doi_list = meta.get("dois", [{}])
                doi = doi_list[0].get("value", "") if doi_list else ""
                summary = f"🔬 INSPIRE-HEP 学术总库 | 出版载体: 《{journal}》 | 📊 被引: {cites} 次"
                resources.append(AcademicResource(
                    id=doi or str(h.get("id", title)),
                    title=title,
                    authors=[a.get("full_name", "") for a in meta.get("authors", [])[:4]],
                    year=year_int,
                    source="inspire_hep",
                    summary=summary,
                    url=f"https://doi.org/{doi}" if doi else f"https://inspirehep.net/literature/{h.get('id')}",
                    doi=doi,
                    citation_count=cites,
                    venue=journal
                ))
        except Exception as e:
            print(f"⚠️ [InspireHEPSearch Note]: {e}")
        return resources

    def search_doaj(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 DOAJ (Directory of Open Access Journals) 全球全学科开放获取同行评议顶刊总库 (免登录)"""
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        url = f"https://doaj.org/api/search/articles/{urllib.parse.quote(clean_q)}?pageSize={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchScholar/1.0"})

        resources = []
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            for r in results:
                bib = r.get("bibjson", {})
                title = bib.get("title") or ""
                if not title:
                    continue
                year = None
                try:
                    year = int(bib.get("year")) if bib.get("year") else None
                except Exception:
                    pass
                journal = bib.get("journal", {}).get("title") or "DOAJ 开放获取同行评议顶刊"
                doi = ""
                for ident in bib.get("identifier", []):
                    if ident.get("type") == "doi":
                        doi = ident.get("id")
                        break
                authors = [a.get("name", "") for a in bib.get("author", [])[:4] if a.get("name")]
                summary = f"📚 DOAJ 开放获取顶刊: 《{journal}》 | DOI: {doi}"
                resources.append(AcademicResource(
                    id=doi or title,
                    title=title,
                    authors=authors,
                    year=year,
                    source="doaj",
                    summary=summary,
                    url=f"https://doi.org/{doi}" if doi else "",
                    doi=doi,
                    venue=journal
                ))
        except Exception as e:
            print(f"⚠️ [DOAJSearch Note]: {e}")
        return resources

    @_cached_academic_search("dblp")
    def search_dblp(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 DBLP 检索计算机科学、电子信息与 AI 顶会期刊录用 (免登录)"""
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        params = {"q": clean_q, "format": "json", "h": max_results}
        url = f"https://dblp.org/search/publ/api?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchScholar/1.0", "Accept": "application/json"},
        )

        resources = []
        source_error = None
        try:
            with urllib.request.urlopen(req, timeout=_search_timeout(10)) as resp:
                status = getattr(resp, "status", 200)
                content_type = str(resp.headers.get("Content-Type", "")).casefold()
                raw_text = resp.read().decode("utf-8", errors="replace").strip()
            if status != 200 or ("json" not in content_type and not raw_text.startswith("{")):
                source_error = f"Http{status}NonJson"
                challenge = any(marker in raw_text.lower() for marker in ("not a bot", "captcha", "challenge"))
                seconds = 900.0 if challenge else 60.0
                _cool_down_source(self, "dblp", seconds, "bot_challenge" if challenge else source_error)
                print(f"ℹ️ [DBLPSearch Note]: HTTP {status} 未返回论文 JSON，暂停请求 {seconds:.0f}s；其他来源独立检索。")
                return SearchResults(resources, source_error)
            if not raw_text:
                return SearchResults(resources, "EmptyResponse")
            data = json.loads(raw_text)
            hits = data.get("result", {}).get("hits", {}).get("hit", [])
            for h in hits:
                info = h.get("info", {})
                title = info.get("title") or ""
                if not title:
                    continue
                year = None
                try:
                    year = int(info.get("year")) if info.get("year") else None
                except Exception:
                    pass
                venue = info.get("venue") or "国际学术出版物"
                doi = info.get("doi") or ""
                authors_raw = info.get("authors", {}).get("author", [])
                if isinstance(authors_raw, list):
                    authors = [a.get("text", "") if isinstance(a, dict) else str(a) for a in authors_raw[:4]]
                else:
                    authors = [authors_raw.get("text", "") if isinstance(authors_raw, dict) else str(authors_raw)]
                summary = f"💻 DBLP 检索收录: 《{venue}》 | 出版年份: {year or 'Recent'}"
                resources.append(AcademicResource(
                    id=doi or info.get("url", title),
                    title=title,
                    authors=authors,
                    year=year,
                    source="dblp",
                    evidence_level="bibliographic_metadata",
                    content_scope="bibliographic record only",
                    summary=summary,
                    url=info.get("ee") or info.get("url") or "",
                    doi=doi,
                    venue=venue
                ))
        except urllib.error.HTTPError as e:
            source_error = f"HTTP {e.code}"
            _source_http_failure(self, "dblp", e)
            print(f"⚠️ [DBLPSearch Note]: {source_error}")
        except Exception as e:
            source_error = type(e).__name__
            print(f"⚠️ [DBLPSearch Note]: {e}")
        return SearchResults(resources, source_error)

    @_cached_academic_search("github")
    def search_github(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 GitHub API 检索开源代码仓库、论文官方实现与 Star 趋势"""
        if re.search(r"[\u4e00-\u9fff]", query):
            translated = self.extract_academic_keywords(query)
            if translated and not re.search(r"[\u4e00-\u9fff]", translated):
                query = translated

        complete_query = _build_github_repository_query(query)
        params = {
            "q": complete_query,
            "sort": "stars",
            "order": "desc",
            "per_page": max_results,
        }
        url = f"https://api.github.com/search/repositories?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ResearchScholar/1.0",
                "Accept": "application/vnd.github+json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=_search_timeout(8)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            resources = []
            detail_limit = max(0, min(int(os.getenv("GITHUB_DETAIL_FETCH_LIMIT", "1")), max_results, 4))
            detail_deadline = time.monotonic() + max(0.0, float(os.getenv("GITHUB_DETAIL_BUDGET", "3")))
            for index, repo in enumerate(data.get("items", [])):
                full_name = repo.get("full_name", "")
                desc = repo.get("description") or "无描述"
                stars = repo.get("stargazers_count", 0)
                html_url = repo.get("html_url", "")
                lang = repo.get("language") or "未知"
                pushed_at = repo.get("pushed_at") or repo.get("updated_at") or "未知"
                readme_text = ""
                version_ref = str(repo.get("default_branch") or "")
                version_ref_type = "branch" if version_ref else None
                evidence_level = "repository_metadata"
                if index < detail_limit and full_name and time.monotonic() < detail_deadline:
                    with _search_budget(detail_deadline):
                        api_headers = {
                            "User-Agent": "ResearchScholar/1.0",
                            "Accept": "application/vnd.github+json",
                        }
                        try:
                            readme_req = urllib.request.Request(
                                f"https://api.github.com/repos/{full_name}/readme", headers=api_headers
                            )
                            with urllib.request.urlopen(readme_req, timeout=_search_timeout(4)) as readme_resp:
                                readme_data = json.loads(readme_resp.read().decode("utf-8"))
                            encoded = readme_data.get("content", "")
                            if encoded:
                                readme_text = base64.b64decode(encoded).decode("utf-8", errors="replace")[:5000]
                                evidence_level = "repository_metadata_and_readme"
                        except Exception:
                            pass
                        try:
                            commit_req = urllib.request.Request(
                                f"https://api.github.com/repos/{full_name}/commits?per_page=1", headers=api_headers
                            )
                            with urllib.request.urlopen(commit_req, timeout=_search_timeout(4)) as commit_resp:
                                commits = json.loads(commit_resp.read().decode("utf-8"))
                            if commits:
                                sha = str(commits[0].get("sha") or "")
                                if sha:
                                    version_ref = sha
                                    version_ref_type = "commit"
                        except Exception:
                            pass
                content_scope = (
                    f"repository metadata and README inspected; latest/default ref={version_ref or 'unknown'}; source files not inspected or executed"
                    if readme_text else
                    f"repository metadata only; latest/default ref={version_ref or 'unknown'}; README/source files not inspected"
                )
                summary = f"⭐ Stars: {stars} | 主语言: {lang} | 最近更新: {pushed_at} | 简介: {desc}"
                if readme_text:
                    summary += f"\nREADME 摘录（前 5000 字符）:\n{readme_text}"
                resources.append(AcademicResource(
                    id=full_name,
                    title=f"GitHub 开源实现: {full_name}",
                    authors=[repo.get("owner", {}).get("login", "community")],
                    year=None,
                    source="github",
                    summary=summary,
                    url=html_url,
                    stars=stars,
                    evidence_level=evidence_level,
                    content_scope=content_scope,
                    version_ref=version_ref or None,
                    version_ref_type=version_ref_type,
                ))
            return resources
        except urllib.error.HTTPError as e:
            _source_http_failure(self, "github", e)
            print(f"⚠️ [GitHubSearch Note]: HTTP {e.code}")
            return SearchResults(source_error=f"HTTP {e.code}")
        except Exception as e:
            print(f"⚠️ [GitHubSearch Note]: {type(e).__name__}: {e}")
            return SearchResults(source_error=type(e).__name__)

    @_cached_academic_search("web")
    def _search_web(self, query, max_results, time_constraint=None):
        from online_resource_adapters import search_web
        return search_web(query, max_results, time_constraint)

    @_cached_academic_search("datasets")
    def _search_datasets(self, query, max_results, time_constraint=None):
        from online_resource_adapters import search_datasets
        return search_datasets(query, max_results)

    def search_multisource(
        self,
        query: str,
        include_github: bool = True,
        include_web: bool = False,
        include_datasets: bool = False,
        time_constraint: Optional[Dict[str, Any]] = None,
        english_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Collect independent sources within one shared network-retrieval budget."""
        time_constraint = resolve_retrieval_time_constraint(time_constraint)
        academic_target = (english_query or "").strip()
        if not academic_target or re.search(r"[\u4e00-\u9fff]", academic_target):
            academic_target = self.extract_academic_keywords(query)
            if re.search(r"[\u4e00-\u9fff]", academic_target):
                academic_target = ""
        effective_query = academic_target or query.strip()
        limit_per_source = 10
        started = time.monotonic()
        deadline = started + ACADEMIC_SEARCH_BUDGET
        inherited = getattr(_SEARCH_CONTEXT, "deadline", None)
        if inherited is not None:
            deadline = min(deadline, inherited)
        print(f'🔍 [MultiSourceSearch] 多源网络检索 (单源上限: {limit_per_source}, 总等待预算: {max(0, deadline-started):.1f}s): "{effective_query}"')

        # Submit the healthy primary indexes first when concurrent requests queue.
        jobs = {
            "openalex": (self.search_openalex, (effective_query, limit_per_source, time_constraint)),
            "crossref": (self.search_crossref, (effective_query, limit_per_source, time_constraint)),
            "semanticscholar": (self.search_semanticscholar, (effective_query, min(limit_per_source, 6))),
        }
        if academic_target:
            jobs["arxiv"] = (self.search_arxiv, (academic_target, limit_per_source, time_constraint))
        jobs["dblp"] = (self.search_dblp, (effective_query, min(limit_per_source, 5)))
        if include_github:
            # The English query has already been prepared; do not translate it again.
            jobs["github"] = (self.search_github, (effective_query, limit_per_source))
        if include_web:
            jobs["web"] = (self._search_web, (effective_query, limit_per_source, time_constraint))
        if include_datasets:
            jobs["datasets"] = (self._search_datasets, (effective_query, limit_per_source, time_constraint))

        def invoke(method, args):
            with _search_budget(deadline):
                _search_timeout(ACADEMIC_SEARCH_BUDGET)
                return method(*args)

        futures = {}
        records_by_source = {}
        source_errors = {}
        cache_hits = []
        for source, (method, args) in jobs.items():
            if not _SEARCH_WORK_SLOTS.acquire(blocking=False):
                source_errors[source] = "busy"
                continue
            try:
                future = _SEARCH_EXECUTOR.submit(invoke, method, args)
            except BaseException:
                _SEARCH_WORK_SLOTS.release()
                raise
            future.add_done_callback(lambda _future: _SEARCH_WORK_SLOTS.release())
            futures[source] = future
        if futures:
            wait(futures.values(), timeout=max(0.0, deadline - time.monotonic()))
        # Do not enter a per-call executor context: its exit waits for unfinished I/O.
        # Running I/O cannot be forcibly cancelled; it stays in the bounded pool,
        # uses the same deadline where supported, and cannot mutate this response.
        for source, future in futures.items():
            if not future.done():
                future.cancel()
                source_errors[source] = "TimeoutError"
                print(f"⚠️ [{source} Thread Note]: TimeoutError: 本轮网络检索预算已耗尽，保留其他来源结果。")
                continue
            try:
                records = future.result()
                records_by_source[source] = records
                if getattr(records, "source_error", None):
                    source_errors[source] = records.source_error
                if getattr(records, "cache_hit", False):
                    cache_hits.append(source)
            except Exception as exc:
                source_errors[source] = type(exc).__name__
                print(f"⚠️ [{source} Thread Note]: {type(exc).__name__}: {exc}")

        paper_sources = {"arxiv": "arxiv_papers", "openalex": "openalex_papers", "crossref": "crossref_papers",
                         "semanticscholar": "semanticscholar_papers", "dblp": "dblp_papers", "github": "github_repos"}
        result = {key: [p.model_dump() for p in records_by_source.get(source, [])]
                  for source, key in paper_sources.items()}
        result.update({"web_results": records_by_source.get("web", []),
                       "datasets": records_by_source.get("datasets", []),
                       "source_errors": source_errors, "source_cache_hits": cache_hits,
                       "search_elapsed_seconds": round(time.monotonic() - started, 3),
                       "search_budget_seconds": ACADEMIC_SEARCH_BUDGET,
                       "total_found": sum(len(records) for records in records_by_source.values())})
        counts = {source: len(records) for source, records in records_by_source.items()}
        print(f"📊 [MultiSourceSearch] 实际返回={counts}；来源异常={source_errors}；缓存命中={cache_hits}")
        return result

    def _parse_atom_feed(self, xml_bytes: bytes) -> List[ArxivPaper]:
        from academic_taxonomy import get_subcategory_meta
        root = ET.fromstring(xml_bytes)
        arxiv_ns = "{http://arxiv.org/schemas/atom}"
        papers = []
        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            entry_id = (entry.findtext(f"{{{ATOM_NS}}}id") or "").strip()
            paper_id = entry_id.rsplit("/", 1)[-1]
            title = " ".join((entry.findtext(f"{{{ATOM_NS}}}title") or "").split())
            published_full = (entry.findtext(f"{{{ATOM_NS}}}published") or "").strip()
            published = published_full[:4]
            summary = " ".join((entry.findtext(f"{{{ATOM_NS}}}summary") or "").split())
            authors = [
                (a.findtext(f"{{{ATOM_NS}}}name") or "").strip()
                for a in entry.findall(f"{{{ATOM_NS}}}author")
            ]
            pdf_link = next(
                (l.attrib.get("href") for l in entry.findall(f"{{{ATOM_NS}}}link") if l.attrib.get("title") == "pdf"),
                f"https://export.arxiv.org/pdf/{paper_id}.pdf",
            )
            if not pdf_link.endswith(".pdf"):
                pdf_link += ".pdf"

            # 提取权威主学科与全部交叉分类
            pri_elem = entry.find(f"{arxiv_ns}primary_category")
            primary_cat = pri_elem.attrib.get("term", "") if pri_elem is not None else ""
            cat_elems = entry.findall(f"{{{ATOM_NS}}}category")
            categories = [c.attrib.get("term", "") for c in cat_elems if c.attrib.get("term")]
            if primary_cat and primary_cat not in categories:
                categories.insert(0, primary_cat)

            # 解析分类的权威中文标签
            zh_names = []
            for c in categories:
                cm = get_subcategory_meta(c)
                if cm:
                    zh_names.append(f"{c} ({cm['name_zh']})")
                else:
                    zh_names.append(c)

            comment = (entry.findtext(f"{arxiv_ns}comment") or "").strip()
            journal_ref = (entry.findtext(f"{arxiv_ns}journal_ref") or "").strip()

            papers.append(ArxivPaper(
                paper_id=paper_id,
                title=title,
                authors=authors[:5],
                year=int(published) if published.isdigit() else None,
                published_at=published_full or None,
                summary=summary,
                pdf_url=pdf_link,
                primary_category=primary_cat,
                categories=categories,
                category_names_zh=zh_names,
                comment=comment or None,
                journal_ref=journal_ref or None,
            ))
        return papers

    def download_and_ingest(self, paper: Any, rag_engine=None) -> Dict[str, Any]:
        """下载论文全文并自动触发 4D 学术本体提炼与向量建库 (借鉴 073001 审计与侧车标准)"""
        import hashlib

        # 兼容 ArxivPaper 对象或 Dict
        if isinstance(paper, dict):
            paper_id = paper.get("paper_id") or paper.get("id") or "paper"
            title = paper.get("title") or "academic_paper"
            pdf_url = paper.get("pdf_url") or paper.get("url") or ""
            authors = paper.get("authors") or []
            year = paper.get("year")
            published_at = paper.get("published_at")
            summary = paper.get("summary") or ""
        else:
            paper_id = getattr(paper, "paper_id", "paper")
            title = getattr(paper, "title", "academic_paper")
            pdf_url = getattr(paper, "pdf_url", "")
            authors = getattr(paper, "authors", [])
            year = getattr(paper, "year", None)
            published_at = getattr(paper, "published_at", None)
            summary = getattr(paper, "summary", "")

        clean_pid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(paper_id)).strip("._")
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title)).strip("._")[:60]
        stem = f"{clean_pid}_{safe_title}" if clean_pid else safe_title
        pdf_name = f"{stem}.pdf"
        pdf_path = UPLOADS_DIR / pdf_name
        meta_path = UPLOADS_DIR / f"{stem}.json"

        # 如果已下载并入库过，直接复用
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            print(f"📄 [PaperIngest] 本地已有该论文缓存: {pdf_name}")
            return {
                "status": "success",
                "paper_id": str(paper_id),
                "file_path": str(pdf_path),
                "file_name": pdf_name,
                "title": title,
                "already_existed": True,
            }

        # 构建全网候选开放下载源列表 (原链优先，备选开源仓储 OSTI/arXiv/EuropePMC 跟进)
        candidate_urls = []
        if pdf_url and pdf_url.startswith("http"):
            candidate_urls.append(pdf_url)

        doi_or_id = paper.get("doi") if isinstance(paper, dict) else getattr(paper, "doi", "")
        doi_or_id = doi_or_id or str(paper_id)
        oa_mirrors = self._find_oa_pdf_urls(doi_or_id)
        for m in oa_mirrors:
            if m not in candidate_urls:
                candidate_urls.append(m)

        download_success = False
        last_download_error = None
        digest = hashlib.sha256()

        for try_url in candidate_urls:
            req = urllib.request.Request(
                try_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ResearchScholar/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    headers = getattr(resp, "headers", None)
                    c_type = (headers.get("Content-Type", "") if headers else "").lower()
                    if "html" in c_type and not try_url.endswith(".pdf"):
                        continue
                    digest = hashlib.sha256()
                    with open(pdf_path, "wb") as out_f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            digest.update(chunk)

                if pdf_path.exists() and pdf_path.stat().st_size > 0:
                    file_hash = digest.hexdigest()
                    print(f"📥 [PaperIngest] 成功下载论文全文: {pdf_name} (源: {try_url[:60]}... | SHA-256: {file_hash[:12]}...)")
                    download_success = True
                    break
                else:
                    if pdf_path.exists():
                        pdf_path.unlink()
            except Exception as e:
                last_download_error = e
                if pdf_path.exists():
                    try:
                        pdf_path.unlink()
                    except Exception:
                        pass
                print(f"⚠️ [PaperIngest] 镜像源受限 ({try_url[:55]}...): {e}")

        if download_success:
            file_hash = digest.hexdigest()
            meta_data = {
                "paper_id": str(paper_id),
                "title": title,
                "authors": authors,
                "year": year,
                "published_at": published_at,
                "summary": summary,
                "pdf_url": try_url,
                "sha256": file_hash,
                "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            meta_path.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), encoding="utf-8")
            chunks_indexed = 0
            if rag_engine is not None:
                print(f"🧠 [PaperIngest] 正在对新抓取论文自动提炼 4D 学术本体并建库...")
                chunks_indexed = rag_engine.ingest_file(pdf_path)

            return {
                "status": "success",
                "paper_id": str(paper_id),
                "file_path": str(pdf_path),
                "file_name": pdf_name,
                "title": title,
                "chunks_indexed": chunks_indexed,
                "already_existed": False,
            }
        else:
            # 优雅降级兜底: 若全网均无开放全文 PDF (商业付费闭源)，将权威摘要、DOI与学术元数据入库构建学术证据
            fallback_name = f"{stem}_abstract.txt"
            fallback_path = UPLOADS_DIR / fallback_name
            abstract_content = (
                f"# 学术文献研读: {title}\n"
                f"- 作者团队: {', '.join(authors) if isinstance(authors, list) else authors}\n"
                f"- 出版年份: {year}\n"
                f"- 权威出版/DOI: {paper_id}\n\n"
                f"## 核心学术摘要与方案结论\n{summary}\n"
            )
            fallback_path.write_text(abstract_content, encoding="utf-8")
            chunks_indexed = 0
            if rag_engine is not None:
                print(f"📑 [PaperIngest] 原文受商业出版权限限制 ({last_download_error})，已自动将权威文献摘要与元数据建库入卷...")
                chunks_indexed = rag_engine.ingest_file(fallback_path)

            return {
                "status": "success",
                "paper_id": str(paper_id),
                "file_path": str(fallback_path),
                "file_name": fallback_name,
                "title": title,
                "chunks_indexed": chunks_indexed,
                "already_existed": False,
                "note": "abstract_indexed"
            }

    def _find_oa_pdf_urls(self, identifier: str) -> List[str]:
        """依据 DOI 或 arXiv ID 动态解析全球开放获取全文 PDF 镜像 (OSTI, arXiv, EuropePMC, Unpaywall 等)"""
        candidates = []
        clean_id = str(identifier).strip()
        m_doi = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", clean_id)
        if m_doi:
            doi = m_doi.group(0)
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            req = urllib.request.Request(url, headers={"User-Agent": "ResearchScholar/1.0 (mailto:scholar@test.local)"})
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                for loc in data.get("locations", []):
                    p_url = loc.get("pdf_url")
                    if p_url and p_url not in candidates:
                        candidates.append(p_url)
            except Exception:
                pass

        m_arx = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", clean_id)
        if m_arx:
            arx_id = m_arx.group(1)
            arx_url = f"https://arxiv.org/pdf/{arx_id}.pdf"
            if arx_url not in candidates:
                candidates.append(arx_url)
        return candidates

    def generate_search_brief(
        self,
        query: str,
        papers: List[ArxivPaper],
        openalex_papers: Optional[List[AcademicResource]] = None,
        crossref_papers: Optional[List[AcademicResource]] = None,
        github_repos: Optional[List[AcademicResource]] = None,
        history_context: Optional[str] = None,
        active_topic: Optional[str] = None,
        requires_long_form: bool = False,
        long_form_type: str = "concise_analysis",
        evidence: Optional[List[Dict[str, Any]]] = None,
        shared_task: Optional[Dict[str, Any]] = None,
    ) -> str:
        """基于多源检索证据组织学术综述；紧密关联会话上下文与主干研究主题，由羽若展开系统性学术剖析与前瞻研讨"""
        openalex_papers = openalex_papers or []
        crossref_papers = crossref_papers or []
        github_repos = github_repos or []

        if evidence is None:
            # Standalone callers use the same complete records and global numbering.
            evidence = []
            for source, resources in (
                ("openalex", openalex_papers), ("arxiv", papers),
                ("crossref", crossref_papers), ("github", github_repos),
            ):
                for resource in resources:
                    evidence.append({
                        "doc_name": resource.title, "doc_type": source,
                        "text": resource.summary, "summary": resource.summary,
                        "authors": resource.authors, "year": resource.year,
                        "location_label": str(resource.year or ""),
                        "url": resource.pdf_url if source == "arxiv" else resource.url,
                        "arxiv_id": resource.paper_id if source == "arxiv" else None,
                        "doi": getattr(resource, "doi", None),
                        "version_ref": getattr(resource, "version_ref", None),
                        "version_ref_type": getattr(resource, "version_ref_type", None),
                        "evidence_level": getattr(resource, "evidence_level", None) or (
                            "abstract" if source == "arxiv" else
                            "repository_metadata" if source == "github" else "index_metadata_or_abstract"
                        ),
                    })
        evidence_section = format_evidence_for_prompt(evidence)

        partner_name = get_assistant_name()
        system_prompt = build_lead_tutor_system_prompt(
            assistant_name=partner_name, requires_long_form=requires_long_form,
            long_form_type=long_form_type, active_subject=active_topic, history_summary=history_context,
        )
        task = shared_task or {"original_user_query": query}
        prompt = ("本轮共享任务（原话为准）：\n" + json.dumps(task, ensure_ascii=False)
                  + "\n\n检索资料：\n" + (evidence_section or "本次未取得可用来源。"))
        if evidence_section:
            prompt += (
                "\n\n【生成执行硬要求（务必落实）】：概念科普简明带过，主要篇幅用于深入剖析核心机制与检索文献的具体方法/实验对比；"
                "必须在「前沿研究路线与实证分析」中给出「核心文献实证对照表」（文献 [n] | 针对痛点 | 核心方法/模块 | 原文已证实结论，未披露数字直写“未披露”），不得只把 [n] 当作句尾装饰。"
            )

        try:
            from master_agent_workflow import safe_chat_completion
            model_name = get_default_llm_model()
            print(f"\n✍️ [OnlinePaperSearchAgent] 正在流式生成前沿文献综合综述 (模型: {model_name})...")
            res_str = safe_chat_completion(
                client=self.llm_client,
                model=model_name,
                messages=[{"role": "user", "content": system_prompt + "\n\n" + prompt}],
                temperature=0.3,
                max_tokens=get_llm_max_tokens(),
                timeout=get_llm_timeout(100.0),
            )
            return res_str or f"{partner_name}已完成文献检索研讨分析。"
        except Exception as e:
            return f"{partner_name}在解析学术文献时发生异常: {e}"
