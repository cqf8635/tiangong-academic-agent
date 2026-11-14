"""Online ArXiv Academic Paper Search & Ingestion Agent.

Integrates:
1. ArXiv Atom API querying & metadata parsing (title, authors, year, abstract, PDF url).
2. Automated paper downloading into the local uploads directory.
3. Direct ingestion into UniversalDocumentRAGEngine (BGE-M3 vector indexing).
4. SOTA literature brief generation for academic research.
"""

from __future__ import annotations

import json
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
from datetime import datetime
from email.utils import parsedate_to_datetime
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from openai import OpenAI

from config_utils import (
    get_default_llm_model, get_llm_timeout,
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
ARXIV_EMPTY_CACHE_TTL = float(os.getenv("ARXIV_EMPTY_CACHE_TTL", "60"))
ARXIV_CACHE_MAX_ENTRIES = 64
_ARXIV_CACHE: "OrderedDict[tuple[str, int], tuple[float, List['ArxivPaper']]]" = OrderedDict()
_ARXIV_CACHE_LOCK = threading.Lock()


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
                "--max-time", str(max(5, int(timeout))),
                "-H", f"User-Agent: {ua}",
                "-H", f"Accept: {accept_hdr}",
                "-w", "\n__HTTP_STATUS__:%{http_code}",
                url,
            ]
            proc = subprocess.run(
                cmd[:1] + ["--dump-header", "-"] + cmd[1:],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 2,
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

        with self._lock:
            # 进入锁内二次检查冷却
            in_cd, remain = self.is_in_cooldown()
            if in_cd:
                raise TimeoutError(f"arXiv 处于静默保护期 (剩余 {int(remain)}s)")

            # 严格计算距离上一次请求【完成】的时间间隔，动态获取当前配置的最小间隔
            now = time.monotonic()
            min_interval = float(globals().get("ARXIV_MIN_INTERVAL", 3.2))
            wait_for = min_interval - (now - self._last_completed_at)
            if wait_for > 0:
                time.sleep(wait_for)

            try:
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
    if len(words) <= 4:
        return f'all:"{" ".join(words)}"'
    core_terms = words[:6]
    return f"all:({' AND '.join(core_terms)})"


def _build_arxiv_search_expression(phrases: List[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return _format_single_arxiv_clause(phrases[0])
    return " OR ".join(_format_single_arxiv_clause(phrase) for phrase in phrases[:3])


def _build_github_repository_query(query: str) -> str:
    """Keep full semantic expressions; semicolon-separated expressions are alternatives."""
    phrases = _extract_academic_phrases(query)
    if not phrases:
        return query.strip()
    return " OR ".join(phrases[:3])


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
        max_retries = 1
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=get_default_llm_model(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                max_tokens=256,
                    timeout=min(get_llm_timeout(100.0), 15.0),
                )
                raw = (resp.choices[0].message.content or "").strip()
                clean_terms = raw.replace('"', '').replace("'", "").replace("\n", " ").strip()
                if clean_terms:
                    print(f"🎯 [LLM Academic Keywords] 大模型精准提炼并翻译英文学术词: \"{clean_terms}\"")
                    self._keyword_cache[cache_k] = clean_terms
                    return clean_terms
            except Exception as e:
                print(f"⚠️ [Keyword Extraction Fast Fallback]: {e}")

        return query.strip()

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
                if not papers and len(phrases) > 1:
                    fallback_query = f'all:"{phrases[0]}"'
                    fallback_query += date_filter
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
                    except Exception as fb_err:
                        print(f"ℹ️ [ArxivSearch] 双短语回退单核心短语提示: {fb_err}")

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
                if attempt >= ARXIV_MAX_RETRIES:
                    break
                time.sleep(1.0 + random.uniform(0, 0.5))
            except Exception as e:
                last_error = e
                break

        print(f"ℹ️ [ArxivSearch] arXiv 响应超时或受限 ({last_error})，自动由 CrossRef/OpenAlex 全球学术库平滑接管证据链。")
        return []

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
                if time.monotonic() - cached_time < _RADAR_SEARCH_CACHE_TTL:
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

        # 2. 若网络波动、超时或频次受限导致获取为空，自动无缝触发本地快照容灾
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

        # Cache successful empty queries too; failed requests are not zero matches.
        if api_succeeded:
            with _RADAR_SEARCH_CACHE_LOCK:
                _RADAR_SEARCH_CACHE[cache_key] = (time.monotonic(), copy.deepcopy(final_result))
                _RADAR_SEARCH_CACHE.move_to_end(cache_key)
                while len(_RADAR_SEARCH_CACHE) > _RADAR_SEARCH_CACHE_MAX:
                    _RADAR_SEARCH_CACHE.popitem(last=False)

        return final_result

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
        url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchScholar/1.0 (mailto:scholar@researchscholar.local)"}
        )

        resources = []
        source_error = None
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
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
        except Exception as e:
            source_error = type(e).__name__
            print(f"⚠️ [CrossRefSearch Note]: {e}")
        return SearchResults(resources, source_error)

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

        params = {
            "search": search_term,
            "per-page": max_results,
            "mailto": "yunshan_scholar@research-institute.org"
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
            headers={"User-Agent": "YunshanScholar/2.0 (mailto:yunshan_scholar@research-institute.org)"}
        )

        resources = []
        source_error = None
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
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
            if he.code == 429:
                _OPENALEX_COOLDOWN_UNTIL = time.time() + 180.0  # 自动进入 3 分钟熔断冷却，由 CrossRef 与 arXiv 全权兜底
            else:
                print(f"⚠️ [OpenAlexSearch Note]: {he}")
        except Exception as e:
            source_error = type(e).__name__
            print(f"⚠️ [OpenAlexSearch Error]: {e}")
        return SearchResults(resources, source_error)

    def search_semanticscholar(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 Semantic Scholar API 检索文献极速摘要与学术引用影响力 (防噪高信噪比)"""
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        params = {
            "query": clean_q,
            "limit": max_results,
            "fields": "title,authors,year,abstract,citationCount,influentialCitationCount,venue,url,tldr"
        }
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchScholar/1.0"}
        )

        resources = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for it in data.get("data", []):
                title = it.get("title") or ""
                if not title:
                    continue
                year = it.get("year")
                cited = it.get("citationCount", 0)
                venue = it.get("venue") or "国际同行评议学术出版物"
                authors = [a.get("name", "") for a in it.get("authors", []) if a.get("name")]
                tldr = (it.get("tldr") or {}).get("text") or it.get("abstract") or ""
                summary = f"📌 TL;DR: {tldr[:200]}... | 📊 引用: {cited} 次 | 录用: {venue}"
                resources.append(AcademicResource(
                    id=it.get("paperId", title),
                    title=title,
                    authors=authors[:4],
                    year=year,
                    source="semanticscholar",
                    summary=summary,
                    url=it.get("url") or "",
                    citation_count=cited,
                    venue=venue
                ))
        except Exception as e:
            print(f"⚠️ [SemanticScholar Note]: {e}")
        return resources

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

    def search_dblp(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 DBLP 检索计算机科学、电子信息与 AI 顶会期刊录用 (免登录)"""
        clean_q = re.sub(r"[\"'`]", "", query).strip()
        params = {"q": clean_q, "format": "json", "h": max_results}
        url = f"https://dblp.org/search/publ/api?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "ResearchScholar/1.0"})

        resources = []
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
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
                    summary=summary,
                    url=info.get("ee") or info.get("url") or "",
                    doi=doi,
                    venue=venue
                ))
        except Exception as e:
            print(f"⚠️ [DBLPSearch Note]: {e}")
        return resources

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
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            resources = []
            detail_limit = max(0, min(int(os.getenv("GITHUB_DETAIL_FETCH_LIMIT", "2")), max_results, 4))
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
                if index < detail_limit and full_name:
                    api_headers = {
                        "User-Agent": "ResearchScholar/1.0",
                        "Accept": "application/vnd.github+json",
                    }
                    try:
                        readme_req = urllib.request.Request(
                            f"https://api.github.com/repos/{full_name}/readme", headers=api_headers
                        )
                        with urllib.request.urlopen(readme_req, timeout=8) as readme_resp:
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
                        with urllib.request.urlopen(commit_req, timeout=8) as commit_resp:
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
        except Exception as e:
            print(f"⚠️ [GitHubSearch Note]: {e}")
            return SearchResults(source_error=type(e).__name__)

    def search_multisource(
        self,
        query: str,
        include_github: bool = True,
        include_web: bool = False,
        include_datasets: bool = False,
        time_constraint: Optional[Dict[str, Any]] = None,
        english_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """跨 arXiv 预印本 + OpenAlex 权威引用 + CrossRef 期刊 + GitHub 开源实现的一体化多源学术并发探针"""
        from concurrent.futures import ThreadPoolExecutor
        time_constraint = resolve_retrieval_time_constraint(time_constraint)

        # 1. 优先使用已翻译的英文学术词；若无或包含中文，通过 LLM 高精度提炼标准学术短语
        academic_target = (english_query or "").strip()
        if not academic_target or re.search(r"[\u4e00-\u9fff]", academic_target):
            academic_target = self.extract_academic_keywords(query)
            if re.search(r"[\u4e00-\u9fff]", academic_target):
                academic_target = ""

        # 全球学术引擎优先使用精准规范的英文学术短语，回退保护采用原始 query
        effective_query = academic_target or query.strip()
        tc_log = f" (时间约束: {time_constraint})" if time_constraint else ""

        # 时间条件只负责筛选，不改变首轮取数；有无时间范围时各来源保持同一上限。
        limit_per_source = 10
        print(f"🔍 [MultiSourceSearch] 正在全域学术并发检索 (单源上限: {limit_per_source}): 规范学术短语=\"{academic_target or '回退原始query'}\" | 原始输入=\"{query.strip()}\"{tc_log} ...")

        arxiv_papers = []
        openalex_papers = []
        crossref_papers = []
        github_repos = []
        web_results = []
        datasets = []
        source_errors = {}

        with ThreadPoolExecutor(max_workers=6) as executor:
            fut_arxiv = executor.submit(self.search_arxiv, academic_target, limit_per_source, time_constraint) if academic_target else None
            fut_openalex = executor.submit(self.search_openalex, effective_query, limit_per_source, time_constraint)
            fut_crossref = executor.submit(self.search_crossref, effective_query, limit_per_source, time_constraint)
            github_query = f"{query.strip()} {academic_target}".strip()
            fut_github = executor.submit(self.search_github, github_query, limit_per_source) if include_github else None
            if include_web or include_datasets:
                from online_resource_adapters import search_datasets, search_web
                fut_web = executor.submit(search_web, query, limit_per_source) if include_web else None
                fut_datasets = executor.submit(search_datasets, query, limit_per_source) if include_datasets else None
            else:
                fut_web = fut_datasets = None

            if fut_arxiv:
                try:
                    arxiv_papers = fut_arxiv.result(timeout=15)
                except Exception as e:
                    source_errors["arxiv"] = type(e).__name__
                    print(f"⚠️ [ArXiv Thread Note]: {e or type(e).__name__}")

            try:
                openalex_papers = fut_openalex.result(timeout=15)
            except Exception as e:
                source_errors["openalex"] = type(e).__name__
                print(f"⚠️ [OpenAlex Thread Note]: {e or type(e).__name__}")

            try:
                crossref_papers = fut_crossref.result(timeout=15)
            except Exception as e:
                source_errors["crossref"] = type(e).__name__
                print(f"⚠️ [CrossRef Thread Note]: {e or type(e).__name__}")

            if fut_github:
                try:
                    github_repos = fut_github.result(timeout=15)
                except Exception as e:
                    source_errors["github"] = type(e).__name__
                    print(f"⚠️ [GitHub Thread Note]: {e or type(e).__name__}")

            for source_name, future in (("web", fut_web), ("datasets", fut_datasets)):
                if future:
                    try:
                        results = future.result(timeout=15)
                        if source_name == "web":
                            web_results = results
                        else:
                            datasets = results
                    except Exception as e:
                        source_errors[source_name] = type(e).__name__
                        print(f"⚠️ [{source_name} Thread Note]: {e or type(e).__name__}")

        for source, records in (("arxiv", arxiv_papers), ("openalex", openalex_papers),
                ("crossref", crossref_papers), ("github", github_repos), ("web", web_results), ("datasets", datasets)):
            if getattr(records, "source_error", None):
                source_errors[source] = records.source_error

        return {
            "arxiv_papers": [p.model_dump() for p in arxiv_papers],
            "openalex_papers": [p.model_dump() for p in openalex_papers],
            "crossref_papers": [p.model_dump() for p in crossref_papers],
            "github_repos": [r.model_dump() for r in github_repos],
            "web_results": web_results,
            "datasets": datasets,
            "source_errors": source_errors,
            "total_found": len(arxiv_papers) + len(openalex_papers) + len(crossref_papers) + len(github_repos) + len(web_results) + len(datasets)
        }

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

        try:
            from master_agent_workflow import safe_chat_completion
            model_name = get_default_llm_model()
            print(f"\n✍️ [OnlinePaperSearchAgent] 正在流式生成前沿文献综合综述 (模型: {model_name})...")
            res_str = safe_chat_completion(
                client=self.llm_client,
                model=model_name,
                messages=[{"role": "user", "content": system_prompt + "\n\n" + prompt}],
                temperature=0.3,
                max_tokens=4096,
                timeout=get_llm_timeout(100.0),
            )
            return res_str or f"{partner_name}已完成文献检索研讨分析。"
        except Exception as e:
            return f"{partner_name}在解析学术文献时发生异常: {e}"
