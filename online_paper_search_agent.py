"""Online ArXiv Academic Paper Search & Ingestion Agent.

Integrates:
1. ArXiv Atom API querying & metadata parsing (title, authors, year, abstract, PDF url).
2. Automated paper downloading into the local uploads directory.
3. Direct ingestion into UniversalDocumentRAGEngine (BGE-M3 vector indexing).
4. SOTA literature brief generation for academic research.
"""

from __future__ import annotations

import json
import os
import re
import random
import socket
import ssl
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
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from openai import OpenAI

from config_utils import get_default_llm_model, get_llm_timeout
from prompts import build_academic_keyword_prompt, get_assistant_name

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_REQUEST_TIMEOUT = float(os.getenv("ARXIV_REQUEST_TIMEOUT", "8"))
ARXIV_MAX_RETRIES = int(os.getenv("ARXIV_MAX_RETRIES", "1"))
ARXIV_MIN_INTERVAL = float(os.getenv("ARXIV_MIN_INTERVAL", "3.2"))
ARXIV_CACHE_TTL = float(os.getenv("ARXIV_CACHE_TTL", "900"))
ARXIV_EMPTY_CACHE_TTL = float(os.getenv("ARXIV_EMPTY_CACHE_TTL", "60"))
ARXIV_CACHE_MAX_ENTRIES = 64
_ARXIV_CACHE: "OrderedDict[tuple[str, int], tuple[float, List['ArxivPaper']]]" = OrderedDict()
_ARXIV_CACHE_LOCK = threading.Lock()


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
        print(f"⚠️ [ArxivScheduler] 开启全局静默保护 {duration:.0f}s ({reason})，本次请求明确暂停。")

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
                last_err = None
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(req, timeout=timeout) as resp:
                            data = resp.read()
                        self.record_success()
                        return data
                    except urllib.error.HTTPError as e:
                        last_err = e
                        if e.code == 429:
                            raise
                        if attempt < 2 and e.code in (500, 502, 503, 504):
                            time.sleep(2.0 * (attempt + 1))
                            continue
                        raise
                    except (TimeoutError, urllib.error.URLError, socket.timeout, ConnectionResetError, ssl.SSLError) as e:
                        last_err = e
                        if attempt < 2:
                            time.sleep(2.0 * (attempt + 1))
                            continue
                        self.record_timeout()
                        raise
                if last_err:
                    raise last_err
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
ARXIV_USER_AGENT = "Mozilla/5.0 (compatible; TianGongAcademicRadar/2.0; +https://arxiv.org; mailto:academic_research@tian-gong.org)"

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


class AcademicSearchTerms(BaseModel):
    english_keywords: List[str] = Field(
        default_factory=list,
        description="2~3 个最核心、最适合学术数据库检索的英文专业学术术语列表（无序号、无标点）"
    )


_OPENALEX_COOLDOWN_UNTIL = 0.0


def extract_arxiv_id(text: str) -> Optional[str]:
    """从文本或 URL 中提取规范的 arXiv 论文 ID（如 YYMM.NNNNN 格式或旧版学科格式）"""
    if not text:
        return None
    # 匹配标准 arXiv URL (abs 或 pdf) 或显式带有前缀 arxiv: 的格式
    m = re.search(r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # 匹配文本中直接包含的纯 arXiv ID 格式 (例如 YYMM.NNNNN 或带版本号格式)
    m2 = re.search(r'\b(\d{4}\.\d{4,5})(?:v\d+)?\b', text)
    if m2:
        return m2.group(1)
    return None


def _extract_academic_phrases(query: str) -> List[str]:
    """从输入检索词中解析出保真的学术检索短语（语义保真，杜绝粗暴截断丢词）"""
    if not query:
        return []

    # 1. 显式双引号短语优先完整保留
    quoted = re.findall(r'"([^"]+)"', query)
    if quoted:
        clean_quotes = [re.sub(r"\s+", " ", q).strip() for q in quoted if q.strip()]
        if clean_quotes:
            return clean_quotes

    # 2. 标点明确分隔的短语完整保留
    if any(sep in query for sep in [",", "，", ";", "；"]):
        raw_parts = re.split(r"[,，;；]+", query)
        phrases = [re.sub(r"\s+", " ", p).strip() for p in raw_parts if p.strip()]
        if phrases:
            return phrases

    # 3. 针对含并列/对比连词的查询进行语义自然分段
    conjunction_split = re.split(r"\s+(?:and|versus|vs\.?|with)\s+", query, flags=re.IGNORECASE)
    if len(conjunction_split) > 1:
        phrases = [re.sub(r"\s+", " ", p).strip() for p in conjunction_split if p.strip()]
        if phrases:
            return phrases

    # 4. 整体查询保真返回（仅清洗首尾与多余空格，绝不按词数暴力截断）
    clean_q = re.sub(r"[^\w\s-]", " ", query)
    clean_q = re.sub(r"\s+", " ", clean_q).strip()
    return [clean_q] if clean_q else []


class OnlinePaperSearchAgent:
    """多源学术文献与开源代码生态发现智能体 (arXiv + CrossRef + GitHub)"""

    def __init__(self, llm_client: Optional[Any] = None):
        if llm_client is not None:
            self.llm_client = llm_client
        else:
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
                    max_tokens=40,
                    timeout=min(get_llm_timeout(100.0), 15.0),
                )
                if getattr(resp.choices[0], "finish_reason", None) != "stop":
                    raise ValueError("模型响应未完整结束，本阶段未完成")
                raw = (resp.choices[0].message.content or "").strip()
                clean_terms = raw.replace('"', '').replace("'", "").replace("\n", " ").strip()
                if clean_terms:
                    print(f"🎯 [LLM Academic Keywords] 大模型精准提炼并翻译英文学术词: \"{clean_terms}\"")
                    self._keyword_cache[cache_k] = clean_terms
                    return clean_terms
            except Exception as e:
                raise RuntimeError(f"学术关键词提取失败: {e}") from e

        raise RuntimeError("学术关键词提取未返回有效内容")

    def search_arxiv(
        self,
        query: str,
        max_results: int = 5,
        time_constraint: Optional[Dict[str, Any]] = None,
    ) -> List[ArxivPaper]:
        """调用 arXiv API 检索学术论文预印本 (支持精准 ID 提取直通、专有实体识别与动态时间约束)"""
        # 1. 标记用户活跃，压制后台可能存在的雷达同步线程，独占网络带宽
        self.mark_user_active()

        if not query.strip():
            raise ValueError("缺少检索词")

        # 优先检测直接的 arXiv ID 或论文链接 (精准直通)
        direct_arxiv_id = extract_arxiv_id(query)
        if direct_arxiv_id:
            cache_key = (f"id:{direct_arxiv_id.lower()}", 1, False)
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

            params = {
                "id_list": direct_arxiv_id,
                "start": 0,
                "max_results": 1,
            }
            url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": ARXIV_USER_AGENT,
                    "Accept": "application/atom+xml, application/xml",
                },
            )
            try:
                xml_data = _ARXIV_SCHEDULER.execute(
                    req,
                    timeout=ARXIV_REQUEST_TIMEOUT,
                    is_background=False,
                )
                papers = self._parse_atom_feed(xml_data)
                if papers:
                    with _ARXIV_CACHE_LOCK:
                        _ARXIV_CACHE[cache_key] = (time.monotonic(), papers)
                        _ARXIV_CACHE.move_to_end(cache_key)
                    return [papers[0].model_copy(deep=True)]
            except Exception as e:
                print(f"⚠️ [search_arxiv] 通过 id_list={direct_arxiv_id} 检索失败: {e}，回退至关键词检索")

        # 2. 通用学术查询布尔表达式构建
        # 显式双引号短语优先保留
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            arxiv_query = " AND ".join([f'all:"{q.strip()}"' for q in quoted[:2]])
        else:
            # 过滤常见英文停用词与虚词
            stop_words = {"and", "or", "the", "for", "with", "in", "of", "to", "a", "an", "on", "at", "by", "from", "after", "is", "are", "which", "how", "what"}
            words = [w for w in re.findall(r'[a-zA-Z0-9_\-]+', query) if w.lower() not in stop_words]
            if not words:
                clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
                arxiv_query = f'all:{clean_q}' if clean_q else 'all:AI'
            else:
                # 复合专有技术实体（驼峰命名或含数字的大写专有模型/架构名）优先定位
                compound_entities = [w for w in words if any(c.isupper() for c in w[1:]) or any(c.isdigit() for c in w)]
                if compound_entities:
                    arxiv_query = f"all:{compound_entities[0]}"
                elif len(words) <= 3:
                    arxiv_query = " AND ".join([f"all:{w}" for w in words])
                else:
                    arxiv_query = " AND ".join([f"all:{w}" for w in words[:3]])

        sort_by_date = bool(time_constraint and time_constraint.get("sort_by_date"))

        # 闭合时域区间支持: 针对特定年份或历史区间 (如 2021 或 2018-2020) 进行严格时域拦截
        if time_constraint and time_constraint.get("start_year"):
            start_y = int(time_constraint["start_year"])
            end_y = int(time_constraint.get("end_year") or start_y)
            arxiv_query = f"{arxiv_query} AND submittedDate:[{start_y:04d}0101000000 TO {end_y:04d}1231235959]"

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
                with _ARXIV_CACHE_LOCK:
                    _ARXIV_CACHE[cache_key] = (time.monotonic(), papers)
                    _ARXIV_CACHE.move_to_end(cache_key)
                    while len(_ARXIV_CACHE) > ARXIV_CACHE_MAX_ENTRIES:
                        _ARXIV_CACHE.popitem(last=False)
                return [paper.model_copy(deep=True) for paper in papers]
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:
                    if attempt >= ARXIV_MAX_RETRIES:
                        _ARXIV_SCHEDULER.trigger_cooldown(180.0, reason="HTTP 429 访问频次受限 (重试耗尽)")
                        break
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 1.5
                    except (TypeError, ValueError) as error:
                        raise ValueError("arXiv Retry-After 响应头无效") from error
                    time.sleep(delay)
                    continue
                break
            except (TimeoutError, urllib.error.URLError, socket.timeout, ConnectionError) as e:
                last_error = e
                if attempt >= ARXIV_MAX_RETRIES:
                    break
                time.sleep(1.0 + random.uniform(0, 0.5))
            except Exception as e:
                last_error = e
                break

        raise RuntimeError(f"arXiv 检索失败: {last_error}") from last_error

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
        """天工雷达专用的全学科精准检索引擎，具备内存LRU缓存与子学科分布统计；请求失败明确报错"""
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
        try:
            xml_data = _ARXIV_SCHEDULER.execute(
                req,
                timeout=ARXIV_REQUEST_TIMEOUT,
                is_background=is_background,
            )
            papers = self._parse_atom_feed(xml_data)
            try:
                root = ET.fromstring(xml_data)
                opensearch_ns = "{http://a9.com/-/spec/opensearch/1.1/}"
                total_elem = root.findtext(f"{opensearch_ns}totalResults")
                if total_elem and total_elem.isdigit():
                    total_results_count = int(total_elem)
            except Exception as e:
                raise ValueError("arXiv 返回的检索统计无效") from e
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _ARXIV_SCHEDULER.trigger_cooldown(180.0, reason="雷达检索触发 429 访问受限")
            raise RuntimeError(f"雷达在线检索失败: {e}") from e
        except Exception as e:
            raise RuntimeError(f"雷达在线检索失败: {e}") from e

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

        # 3. 沉淀至 LRU 内存缓存 (仅缓存非空有效结果)
        if papers:
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
        if not query.strip():
            raise ValueError("缺少检索词")

        phrases = _extract_academic_phrases(query)
        short_q = " ".join(phrases[:2]) if phrases else query.strip()
        params = {"query": short_q, "rows": max_results}
        if time_constraint and time_constraint.get("start_year"):
            start_y = int(time_constraint["start_year"])
            end_y = int(time_constraint.get("end_year") or start_y)
            params["filter"] = f"from-pub-date:{start_y:04d}-01-01,until-pub-date:{end_y:04d}-12-31"
        url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchScholar/1.0 (mailto:scholar@researchscholar.local)"}
        )

        resources = []
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise RuntimeError(f"CrossRefSearch Note: {e}") from e

        if not data:
            return resources

        items = data.get("message", {}).get("items", [])
        for it in items:
            title = (it.get("title") or [""])[0]
            if not title:
                continue
            year = None
            date_parts = it.get("created", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
            doi = it.get("DOI", "")
            authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in it.get("author", [])
            ]
            container = (it.get("container-title") or [""])[0]
            summary = f"发表期刊/会议: 《{container}》 | DOI: {doi}" if container else f"DOI: {doi}"
            resources.append(AcademicResource(
                id=doi or title,
                title=title,
                authors=authors[:4],
                year=year,
                source="crossref",
                summary=summary,
                url=f"https://doi.org/{doi}" if doi else "",
                doi=doi
            ))
        return resources

    def search_openalex(
        self,
        query: str,
        max_results: int = 3,
        time_constraint: Optional[Dict[str, Any]] = None,
    ) -> List[AcademicResource]:
        """调用全球开放学术图谱 OpenAlex API 检索同行评议权威期刊/顶会文献与高被引引用量 (基于动态时间约束过滤)"""
        global _OPENALEX_COOLDOWN_UNTIL
        import time
        now = time.time()
        if now < _OPENALEX_COOLDOWN_UNTIL:
            raise RuntimeError("OpenAlex 访问受限，仍处于冷却期")

        if not query.strip():
            raise ValueError("OpenAlex 缺少检索词")

        phrases = _extract_academic_phrases(query)
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            search_term = " ".join([f'"{q.strip()}"' for q in quoted[:2]])
        elif phrases:
            search_term = " ".join(phrases[:2])
        else:
            clean_q = re.sub(r"[^\w\s-]", " ", query).strip()
            search_term = clean_q or query.strip()

        params = {
            "search": search_term,
            "per-page": max_results,
            "mailto": "yunshan_scholar@research-institute.org"
        }
        if time_constraint and time_constraint.get("start_year"):
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
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as he:
                if he.code == 429:
                    _OPENALEX_COOLDOWN_UNTIL = time.time() + 180.0
                raise RuntimeError(f"OpenAlex 检索失败: {he}") from he
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAlex 检索失败: {e}") from e

        if not data:
            return resources

        for it in data.get("results", []):
            title = it.get("title") or ""
            if not title:
                continue
            year = it.get("publication_year")
            doi = it.get("doi") or ""
            cited = it.get("cited_by_count", 0)
            venue = ((it.get("primary_location") or {}).get("source") or {}).get("display_name") or "出版信息未提供"
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

            summary = f"🏛️ 出版信息: 《{venue}》 | 📊 被引: {cited} 次 | DOI: {doi}"
            resources.append(AcademicResource(
                id=doi or str(it.get("id", title)),
                title=title,
                authors=authors[:4],
                year=year,
                source="openalex",
                summary=summary,
                url=pdf_url,
                doi=doi,
                citation_count=cited,
                venue=venue
            ))
        return resources

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
                venue = it.get("venue") or "出版信息未提供"
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
            raise RuntimeError(f"SemanticScholar Note: {e}") from e
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
                except Exception as e:
                    raise ValueError("在线文献年份字段无效") from e
                doi = r.get("doi") or ""
                journal = r.get("journalTitle") or "出版信息未提供"
                authors_str = r.get("authorString") or ""
                authors = [a.strip() for a in authors_str.split(",") if a.strip()]
                summary = f"🩺 生物医学文献: 《{journal}》 | DOI: {doi} | PMID: {r.get('pmid', '')}"
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
            raise RuntimeError(f"EuropePMC 检索失败: {e}") from e
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
                journal = pubs[0].get("journal_title", "出版信息未提供") if pubs else "出版信息未提供"
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
            raise RuntimeError(f"InspireHEP 检索失败: {e}") from e
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
                except Exception as e:
                    raise ValueError("在线文献年份字段无效") from e
                journal = bib.get("journal", {}).get("title") or "出版信息未提供"
                doi = ""
                for ident in bib.get("identifier", []):
                    if ident.get("type") == "doi":
                        doi = ident.get("id")
                        break
                authors = [a.get("name", "") for a in bib.get("author", [])[:4] if a.get("name")]
                summary = f"📚 DOAJ 收录: 《{journal}》 | DOI: {doi}"
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
            raise RuntimeError(f"DOAJSearch Note: {e}") from e
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
                except Exception as e:
                    raise ValueError("在线文献年份字段无效") from e
                venue = info.get("venue") or "国际学术出版物"
                doi = info.get("doi") or ""
                authors_raw = info.get("authors", {}).get("author", [])
                if isinstance(authors_raw, list):
                    authors = [a.get("text", "") if isinstance(a, dict) else str(a) for a in authors_raw[:4]]
                else:
                    authors = [authors_raw.get("text", "") if isinstance(authors_raw, dict) else str(authors_raw)]
                summary = f"💻 DBLP 检索收录: 《{venue}》 | 出版年份: {year or '年份未提供'}"
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
            raise RuntimeError(f"DBLPSearch Note: {e}") from e
        return resources

    def search_github(self, query: str, max_results: int = 3) -> List[AcademicResource]:
        """调用 GitHub API 检索开源代码仓库、论文官方实现与 Star 趋势"""
        if not query.strip():
            raise ValueError("缺少检索词")

        phrases = _extract_academic_phrases(query)
        clean_target = phrases[0] if phrases else query
        clean_words = [w for w in clean_target.split() if w.lower() not in ["and", "or", "the", "for", "with", "in", "of"]]
        short_q = " ".join(clean_words[:2]) or clean_target

        for stars_filter in ["stars:>5"]:
            params = {
                "q": f"{short_q} {stars_filter}",
                "sort": "stars",
                "order": "desc",
                "per_page": max_results
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
                items = data.get("items", [])
                if items:
                    resources = []
                    for repo in items:
                        full_name = repo.get("full_name", "")
                        desc = repo.get("description") or "无描述"
                        stars = repo.get("stargazers_count", 0)
                        html_url = repo.get("html_url", "")
                        lang = repo.get("language") or "语言未提供"
                        created_at = repo.get("created_at")
                        year = int(created_at[:4]) if created_at else None
                        owner = (repo.get("owner") or {}).get("login")
                        resources.append(AcademicResource(
                            id=full_name,
                            title=f"GitHub 开源实现: {full_name}",
                            authors=[owner] if owner else [],
                            year=year,
                            source="github",
                            summary=f"⭐ Stars: {stars} | 主语言: {lang} | 简介: {desc}",
                            url=html_url,
                            stars=stars
                        ))
                    return resources
            except Exception as e:
                raise RuntimeError(f"GitHub 检索失败: {e}") from e
        return []

    def search_multisource(
        self,
        query: str,
        include_github: bool = True,
        time_constraint: Optional[Dict[str, Any]] = None,
        english_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """跨 arXiv 预印本 + OpenAlex 权威引用 + CrossRef 期刊 + GitHub 开源实现的一体化多源学术并发探针"""
        from concurrent.futures import ThreadPoolExecutor

        # 直接使用意图解析提供的检索表达，优先识别显式 arXiv ID 或论文链接 (精准直通)
        direct_arxiv_id = extract_arxiv_id(query) or extract_arxiv_id(english_query or "")
        if direct_arxiv_id:
            academic_target = direct_arxiv_id
            effective_query = direct_arxiv_id
            eff_time_constraint = None
        else:
            academic_target = (english_query or "").strip()
            effective_query = academic_target if academic_target else query.strip()
            if not effective_query:
                raise ValueError("在线检索缺少检索词")
            academic_target = effective_query
            # 默认回溯近三年文献 (时效保障)，若调用方已指定显式时域则遵从显式约束
            from datetime import datetime
            if not time_constraint or not time_constraint.get("has_time_constraint"):
                cur_year = datetime.now().year
                eff_time_constraint = {
                    "has_time_constraint": True,
                    "start_year": cur_year - 3,
                    "end_year": cur_year,
                    "is_default_window": True,
                }
            else:
                eff_time_constraint = time_constraint

        # 单源最多允许 10 个：显式时效约束时单源上限直接统一设为 10，常规/默认时域检索取 5
        has_explicit_time = bool(eff_time_constraint and eff_time_constraint.get("has_time_constraint") and not eff_time_constraint.get("is_default_window"))
        limit_per_source = 10 if has_explicit_time else 5
        tc_log = f" (时间约束: {eff_time_constraint})" if eff_time_constraint else ""
        print(f"🔍 [MultiSourceSearch] 正在全域学术并发检索 (单源上限: {limit_per_source}): 规范学术短语=\"{academic_target}\" | 原始输入=\"{query.strip()}\"{tc_log} ...")

        arxiv_papers = []
        openalex_papers = []
        crossref_papers = []
        github_repos = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_arxiv = executor.submit(self.search_arxiv, academic_target, limit_per_source, eff_time_constraint) if academic_target else None
            fut_openalex = executor.submit(self.search_openalex, effective_query, limit_per_source, eff_time_constraint)
            fut_crossref = executor.submit(self.search_crossref, effective_query, limit_per_source, eff_time_constraint)
            fut_github = executor.submit(self.search_github, effective_query, 2) if include_github else None

            # 多源弹性容灾：各渠道独立容错，避免任一偶发网络超时掀翻全源已检索成果
            source_errors = []
            def _get_res(fut, name):
                if fut is None:
                    return []
                try:
                    return fut.result(timeout=15) or []
                except TimeoutError as te:
                    err = TimeoutError(f"{name} 接口响应超时 (15s)")
                    source_errors.append((name, err))
                    print(f"⚠️ [MultiSourceSearch] 渠道 [{name}] 响应超时 (15s)")
                    return []
                except Exception as e:
                    source_errors.append((name, e))
                    print(f"⚠️ [MultiSourceSearch] 渠道 [{name}] 检索异常: {e}")
                    return []

            arxiv_papers = _get_res(fut_arxiv, "arXiv")
            openalex_papers = _get_res(fut_openalex, "OpenAlex")
            crossref_papers = _get_res(fut_crossref, "CrossRef")
            github_repos = _get_res(fut_github, "GitHub")

        total = len(arxiv_papers) + len(openalex_papers) + len(crossref_papers) + len(github_repos)
        if total == 0 and source_errors:
            first_err = source_errors[0][1]
            raise first_err

        return {
            "arxiv_papers": [p.model_dump() for p in arxiv_papers],
            "openalex_papers": [p.model_dump() for p in openalex_papers],
            "crossref_papers": [p.model_dump() for p in crossref_papers],
            "github_repos": [r.model_dump() for r in github_repos],
            "total_found": total
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
        """只使用提供的全文地址；下载与索引均成功后才返回成功。"""
        import hashlib

        if rag_engine is None:
            raise ValueError("全文入库需要可用的知识库引擎")
        data = paper if isinstance(paper, dict) else paper.model_dump()
        paper_id = data.get("paper_id") or data.get("id")
        title = data.get("title")
        pdf_url = data.get("pdf_url") or data.get("url")
        if not paper_id or not title:
            raise ValueError("论文缺少标识或标题")
        clean_pid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(paper_id)).strip("._")
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", str(title)).strip("._")[:60]
        stem = f"{clean_pid}_{safe_title}" if clean_pid else safe_title
        pdf_path = UPLOADS_DIR / f"{stem}.pdf"
        meta_path = UPLOADS_DIR / f"{stem}.json"

        if pdf_path.exists():
            if not pdf_path.stat().st_size:
                raise ValueError("已缓存 PDF 为空，未确认入库成功")
            indexed = rag_engine.collection.get(where={"doc_name": pdf_path.name}, limit=1)
            if not indexed.get("ids"):
                raise ValueError("PDF 已下载但知识库中无索引，未确认入库成功")
            return {"status": "success", "paper_id": str(paper_id),
                    "file_path": str(pdf_path), "file_name": pdf_path.name,
                    "title": title, "already_existed": True}

        if not isinstance(pdf_url, str) or not pdf_url.startswith(("http://", "https://")):
            raise ValueError("论文缺少有效全文地址，未自动寻找其他下载源")
        req = urllib.request.Request(pdf_url, headers={"User-Agent": "ResearchScholar/1.0"})
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                headers = getattr(resp, "headers", None)
                if headers and "html" in headers.get("Content-Type", "").lower():
                    raise ValueError("全文地址返回 HTML 页面")
                with pdf_path.open("wb") as out:
                    first = True
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        if first and not chunk.startswith(b"%PDF-"):
                            raise ValueError("全文地址返回的内容不是 PDF")
                        first = False
                        out.write(chunk)
                        digest.update(chunk)
            if not pdf_path.stat().st_size:
                raise ValueError("全文下载为空")
        except Exception as e:
            pdf_path.unlink(missing_ok=True)
            raise RuntimeError(f"论文全文下载失败: {e}") from e

        metadata = {"paper_id": str(paper_id), "title": title,
                    "authors": data.get("authors") or [], "year": data.get("year"),
                    "summary": data.get("summary") or "", "pdf_url": pdf_url,
                    "sha256": digest.hexdigest(), "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        chunks_indexed = rag_engine.ingest_file(pdf_path)
        if not isinstance(chunks_indexed, int) or chunks_indexed <= 0:
            raise RuntimeError("论文全文未生成有效切片，入库未完成")
        return {"status": "success", "paper_id": str(paper_id), "title": title,
                "file_path": str(pdf_path), "file_name": pdf_path.name,
                "chunks_indexed": chunks_indexed, "already_existed": False}

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
        long_form_type: str = "concise_analysis"
    ) -> str:
        """基于多源检索证据组织学术综述；紧密关联会话上下文与主干研究主题，由羽若展开系统性学术剖析与前瞻研讨"""
        openalex_papers = openalex_papers or []
        crossref_papers = crossref_papers or []
        github_repos = github_repos or []

        if requires_long_form:
            length_constraint = f"10. 【系统化成果篇幅规范】：依据学者规划要求（类型：{long_form_type}），展开系统性论述与论证，全面展开各章节。"
            budget_max_tokens = 4096
        else:
            length_constraint = (
                "10. 【学术研讨篇幅与深度规范】：\n"
                "    - 篇幅由课题本身的思想深度与因果复杂度自然展开，杜绝空泛注水，保持紧凑的高信息密度；\n"
                "    - 首句直接切入核心命题，聚焦于所涉对象在【做什么、怎么做、什么结果】层面的深度剖析与对比；\n"
                "    - 若资料仅提供题名与书目线索，如实呈现其方向脉络与已知线索，基于第一性原理客观界定推演边界；\n"
                "    - 语言平实严谨，省去宏观通识套话与口头免责声明。"
            )
            budget_max_tokens = 4096

        evidence_section = ""
        if openalex_papers:
            evidence_section += "\n【OpenAlex 全球权威期刊/顶会高被引文献】\n"
            for i, op in enumerate(openalex_papers, 1):
                authors_str = ", ".join(op.authors) if op.authors else "学术团队"
                evidence_section += f"- [{i}] 《{op.title}》({op.year or '年份未提供'}) | 录用/出版: {op.summary} | 链接: {op.url}\n"

        if papers:
            evidence_section += "\n【arXiv 预印本文献】\n"
            for i, p in enumerate(papers, 1):
                authors_str = ", ".join(p.authors) if p.authors else "未知作者"
                evidence_section += f"- [{i}] 《{p.title}》({p.year or '年份未提供'}) | 作者: {authors_str} | arXiv:{p.paper_id}\n  摘要: {p.summary[:300]}...\n  PDF: {p.pdf_url}\n"

        if crossref_papers:
            evidence_section += "\n【CrossRef 权威发表期刊/会议 (DOI)】\n"
            for i, cp in enumerate(crossref_papers, 1):
                authors_str = ", ".join(cp.authors) if cp.authors else "科研学者团队"
                evidence_section += f"- [{i}] 《{cp.title}》({cp.year or '年份未提供'}) | 出版: {cp.summary} | 链接: {cp.url}\n"

        if github_repos:
            evidence_section += "\n【GitHub 官方开源工程生态】\n"
            for i, gh in enumerate(github_repos, 1):
                evidence_section += f"- [{i}] {gh.title} ({gh.stars or 0}⭐) | 链接: {gh.url}\n  简介: {gh.summary}\n"

        context_prompt = ""
        if history_context:
            context_prompt += f"\n【当前研讨会话多轮历史背景】\n{history_context}\n"
        if active_topic:
            context_prompt += f"\n【当前会话锁定的核心研究主题】: {active_topic}\n"

        if not evidence_section.strip():
            raise RuntimeError("未检索到文献，不能用模型自身知识代替检索综述")
        partner_name = get_assistant_name()
        prompt = f"""你是一位充满敏锐洞察力、学养深厚的学术搭档“{partner_name}”。
你的任务是针对学者探讨的研究主题：“{query}”，组织一份逻辑严密、兼具理论深度与前沿学术视野的【学术文献综述与前瞻方案研讨】。
{context_prompt}
【检索到的多源学术资产证据】：
{evidence_section}

【事实依据与严谨论证准则】：
- 依据上述真实检索资产中提供的题名、作者、摘要与方案信息展开对比与机理剖析；
- 贯彻【做什么、怎么做、什么结果】的核心闭环，透视各方案的研究目标、实现机制与最终成效；
- 若某文献条目缺少具体正文，客观陈述其研究线索，理性呈现推演边界；
- 直奔学术内核，保持高信息密度，省去一切无关垫话。

请遵循以下学术研讨规范进行解析：
1. 研讨搭档身份与称谓：
   - 统一自称为“{partner_name}”，语调亲切、平等、敏锐、严谨且专业，营造学者同侪并肩探讨的沉浸氛围；
2. 紧扣会话主题与承前启后：
   - 严格依据当前研讨主题与会话上下文展开，精准解答学者的具体提问；
3. 核心机理与第一性原理剖析：
   - 深入解释该方向或著作主题的底层机理、形式化模型或核心逻辑，若涉及数理表达需严谨书写公式（遵循 LaTeX 行内 $...$ 与行间 $$...$$ 规范）；
4. 关键演进脉络与主流范式（Architecture & Taxonomy）：
   - 系统梳理该领域当前的主流范式、演进脉络与代表性方案；
5. 多源成果深度对比与实证论证：
   - 结合检索到的前沿文献或权威专著，围绕【做什么、怎么做、什么结果】展开实质性对比；若涉及工程技术方向且存在优质开源生态，可自然结合其实践落地价值；
6. 内在矛盾、开放性挑战与启发探索：
   - 深入剖析当前面临的内在矛盾、代偿代价与开放性挑战（Open Challenges），提出具有实质启发价值的前瞻思考方向与研究设想（Research Ideas）；
7. 逻辑脉络、技术路线或验证流程：
   - 若包含明确的先后、分支、输入输出或决策关系，可使用 Mermaid `flowchart TD`；若只是步骤清单，使用 Markdown 有序列表；
8. 检索关键词与符号呈现规范：
   - 检索关键词使用 Markdown 列表、行内代码或表格展示；数学/科学符号直接使用标准 LaTeX 表达；除非内容本身是可执行代码，文本步骤不使用代码块包裹；Mermaid 必须使用 ` ```mermaid ` 围栏。
9. 遵循专业术语规范与章节编号自洽性。
{length_constraint}"""

        try:
            from master_agent_workflow import safe_chat_completion
            model_name = get_default_llm_model()
            print(f"\n✍️ [OnlinePaperSearchAgent] 正在流式生成前沿文献综合综述 (模型: {model_name})...")
            res_str = safe_chat_completion(
                client=self.llm_client,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=budget_max_tokens,
                timeout=get_llm_timeout(100.0),
            )
            if not res_str or not res_str.strip():
                raise RuntimeError("文献综述未返回有效正文")
            return res_str
        except Exception as e:
            raise RuntimeError(f"文献综述生成失败: {e}") from e
