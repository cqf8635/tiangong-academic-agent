"""Optional open web and dataset discovery adapters.

These adapters return source-labelled records. They never claim that a result
was read in full or independently verified.
"""
from __future__ import annotations

import html as html_lib
import json
import ipaddress
import os
import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from typing import Any, Dict, List
from urllib.parse import parse_qs, quote_plus, urlsplit
from urllib.request import Request, urlopen


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items: List[Dict[str, str]] = []
        self._current: Dict[str, str] | None = None
        self._capture = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "")
        if tag == "a" and "result__a" in cls:
            href = attrs.get("href", "")
            if href.startswith("//"):
                href = "https:" + href
            parsed = urlsplit(href)
            if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
                href = parse_qs(parsed.query).get("uddg", [href])[0]
            self._current = {"url": href, "title": ""}
            self._capture = True
        elif tag in {"a", "div", "td"} and "result__snippet" in cls and self.items:
            self._capture = True
            self.items[-1]["_snippet_open"] = "1"

    def handle_endtag(self, tag):
        if tag == "a" and self._current is not None:
            self.items.append(self._current)
            self._current = None
            self._capture = False
        elif tag in {"div", "td"} and self.items:
            self.items[-1].pop("_snippet_open", None)
            self._capture = False

    def handle_data(self, data):
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._current is not None and self._capture:
            self._current["title"] += text
        elif self.items and self.items[-1].get("_snippet_open"):
            self.items[-1]["summary"] = (self.items[-1].get("summary", "") + " " + text).strip()


class _PageTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title_parts: List[str] = []
        self.text_parts: List[str] = []
        self._in_title = False
        self._hidden_depth = 0
        self.description = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._hidden_depth += 1
        if tag == "meta" and attrs.get("name", "").casefold() in {"description", "og:description"}:
            self.description = attrs.get("content", "")[:1200]

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"} and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data):
        value = re.sub(r"\s+", " ", data).strip()
        if not value:
            return
        if self._in_title:
            self.title_parts.append(value)
        elif not self._hidden_depth:
            self.text_parts.append(value)


def _public_http_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False
        host = parsed.hostname.casefold().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal")):
            return False
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global:
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


def fetch_web_page(url: str) -> Dict[str, str]:
    """Fetch limited public HTML text for source verification, without downloading files."""
    if not _public_http_url(url):
        raise ValueError("Only public HTTP(S) pages can be fetched")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"})
    with urlopen(request, timeout=10) as response:
        final_url = response.geturl()
        if not _public_http_url(final_url):
            raise ValueError("Redirect target is not a public HTTP(S) page")
        content_type = str(response.headers.get("Content-Type", "")).casefold()
        if "html" not in content_type and "text/" not in content_type:
            raise ValueError("Target is not an HTML/text page")
        raw = response.read(2_000_000)
        charset = response.headers.get_content_charset() or "utf-8"
    parser = _PageTextParser()
    parser.feed(raw.decode(charset, errors="replace"))
    page_text = " ".join(parser.text_parts)
    page_text = re.sub(r"\s+", " ", page_text).strip()[:7000]
    return {
        "title": " ".join(parser.title_parts).strip(),
        "summary": page_text or parser.description,
        "url": final_url,
        "evidence_level": "page_text" if page_text else "page_metadata",
        "content_scope": "accessible HTML text; page may be incomplete" if page_text else "page metadata only",
    }


def _search_tavily(query: str, limit: int) -> List[Dict[str, Any]]:
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return []
    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max(1, min(limit, 10)),
        "include_answer": False,
    }).encode("utf-8")
    request = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)",
        },
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        data = json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for item in data.get("results") or []:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        if title and url:
            results.append({
                "title": title,
                "url": url,
                "summary": content,
                "source": "web",
                "provider": "tavily",
                "location_label": "Tavily 网页解读",
                "evidence_level": "web_commentary",
                "content_scope": "Tavily 网页/技术解读（仅用于架构直觉与工程对比，实验指标以论文原文/摘要为准）",
            })
    return results[:limit]


def _search_searxng(query: str, limit: int) -> List[Dict[str, Any]]:
    base_url = (os.getenv("SEARXNG_URL") or "").strip().rstrip("/")
    if not base_url or not _public_http_url(base_url):
        return []
    url = f"{base_url}/search?q={quote_plus(query)}&format=json"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)",
        },
    )
    with urlopen(request, timeout=10) as response:
        data = json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for item in data.get("results") or []:
        title = str(item.get("title") or "").strip()
        item_url = str(item.get("url") or "").strip()
        content = re.sub(r"\s+", " ", str(item.get("content") or item.get("snippet") or "")).strip()
        if title and item_url:
            results.append({
                "title": title,
                "url": item_url,
                "summary": content,
                "source": "web",
                "provider": "searxng",
                "location_label": "SearXNG 网页解读",
                "evidence_level": "web_commentary",
                "content_scope": "SearXNG 网页/技术解读（仅用于架构直觉与工程对比，实验指标以论文原文/摘要为准）",
            })
    return results[:limit]


def _clean_html_text(raw: str) -> str:
    text = html_lib.unescape(str(raw or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _search_huggingface_papers(query: str, limit: int) -> List[Dict[str, Any]]:
    url = f"https://huggingface.co/api/papers/search?q={quote_plus(query)}&limit={max(1, min(limit, 15))}"
    request = Request(url, headers={"User-Agent": "ResearchAssistant/1.0"})
    with urlopen(request, timeout=6) as response:
        payload = json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for raw_item in payload if isinstance(payload, list) else []:
        if not isinstance(raw_item, dict):
            continue
        paper = raw_item.get("paper") if isinstance(raw_item.get("paper"), dict) else raw_item
        paper_id = str(paper.get("id") or "").strip()
        title = re.sub(r"\s+", " ", str(raw_item.get("title") or paper.get("title") or "")).strip()
        if not paper_id or not title:
            continue
        ai_summary = re.sub(r"\s+", " ", str(paper.get("ai_summary") or "")).strip()
        abstract_summary = re.sub(r"\s+", " ", str(raw_item.get("summary") or paper.get("summary") or "")).strip()
        keywords = [str(k).strip() for k in (paper.get("ai_keywords") or []) if str(k).strip()]
        upvotes = paper.get("upvotes")
        comments = raw_item.get("numComments")
        meta_parts = []
        if upvotes is not None:
            meta_parts.append(f"upvotes={upvotes}")
        if comments is not None:
            meta_parts.append(f"comments={comments}")
        if keywords:
            meta_parts.append("keywords=" + ", ".join(keywords[:6]))
        body = ai_summary or abstract_summary
        if meta_parts:
            body = f"[HF Papers {' | '.join(meta_parts)}] {body}".strip()
        results.append({
            "title": title,
            "url": f"https://huggingface.co/papers/{paper_id}",
            "summary": body,
            "arxiv_id": paper_id,
            "source": "web",
            "provider": "huggingface_papers",
            "location_label": "Hugging Face Papers 社区讨论",
            "evidence_level": "web_commentary",
            "content_scope": "Hugging Face Papers 社区讨论与 AI 提炼（仅用于架构直觉与工程对比，实验指标以论文原文/摘要为准）",
        })
    return results[:limit]


def _search_quanta_magazine(query: str, limit: int) -> List[Dict[str, Any]]:
    url = (
        "https://www.quantamagazine.org/wp-json/wp/v2/posts"
        f"?search={quote_plus(query)}&per_page={max(1, min(limit, 5))}&_fields=title,link,excerpt,date"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"})
    with urlopen(request, timeout=6) as response:
        payload = json.loads(response.read(1_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        title = _clean_html_text((item.get("title") or {}).get("rendered") or "")
        link = str(item.get("link") or "").strip()
        excerpt = _clean_html_text((item.get("excerpt") or {}).get("rendered") or "")
        date_str = str(item.get("date") or "")[:10]
        if not title or not link:
            continue
        summary = f"[Quanta Magazine {date_str}] {excerpt}".strip() if date_str else excerpt
        results.append({
            "title": title,
            "url": link,
            "summary": summary,
            "source": "web",
            "provider": "quanta_magazine",
            "location_label": "Quanta Magazine 科学深度解读",
            "evidence_level": "web_commentary",
            "content_scope": "Quanta Magazine 物理/数学/生物/计算机科学解读（仅用于研究脉络与直觉理解，实验指标以论文原文/摘要为准）",
        })
    return results[:limit]


def _search_wikipedia(query: str, limit: int) -> List[Dict[str, Any]]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={quote_plus(query)}&srlimit={max(1, min(limit, 4))}&format=json"
    )
    request = Request(url, headers={"User-Agent": "ResearchAssistant/1.0"})
    with urlopen(request, timeout=6) as response:
        payload = json.loads(response.read(1_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for item in (payload.get("query") or {}).get("search") or []:
        title = str(item.get("title") or "").strip()
        snippet = _clean_html_text(item.get("snippet") or "")
        if not title:
            continue
        page_url = "https://en.wikipedia.org/wiki/" + quote_plus(title.replace(" ", "_"))
        results.append({
            "title": title,
            "url": page_url,
            "summary": f"[Wikipedia 学科百科] {snippet}".strip(),
            "source": "web",
            "provider": "wikipedia",
            "location_label": "Wikipedia 学科百科",
            "evidence_level": "web_commentary",
            "content_scope": "Wikipedia 全学科概念百科（仅用于背景与理论脉络补充，前沿进展与指标以论文原文/摘要为准）",
        })
    return results[:limit]


def _search_europe_pmc(query: str, limit: int) -> List[Dict[str, Any]]:
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={quote_plus(query)}&format=json&pageSize={max(1, min(limit, 4))}"
    )
    request = Request(url, headers={"User-Agent": "ResearchAssistant/1.0"})
    with urlopen(request, timeout=6) as response:
        payload = json.loads(response.read(1_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for item in (payload.get("resultList") or {}).get("result") or []:
        title = _clean_html_text(item.get("title") or "")
        pmid = str(item.get("pmid") or "").strip()
        pmcid = str(item.get("pmcid") or "").strip()
        doi = str(item.get("doi") or "").strip()
        journal = str(item.get("journalTitle") or "").strip()
        year = str(item.get("pubYear") or "").strip()
        if not title:
            continue
        if pmcid:
            item_url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif pmid:
            item_url = f"https://europepmc.org/article/MED/{pmid}"
        elif doi:
            item_url = f"https://doi.org/{doi}"
        else:
            continue
        meta = " | ".join(p for p in (journal, year, f"DOI:{doi}" if doi else "") if p)
        results.append({
            "title": title,
            "url": item_url,
            "doi": doi or None,
            "summary": f"[Europe PMC {meta}] {title}".strip() if meta else title,
            "source": "web",
            "provider": "europe_pmc",
            "location_label": "Europe PMC 生物医学/交叉学科",
            "evidence_level": "web_commentary",
            "content_scope": "Europe PMC 生命科学/交叉学科网页索引（仅用于跨学科线索与脉络补充）",
        })
    return results[:limit]


def _search_inspire_hep(query: str, limit: int) -> List[Dict[str, Any]]:
    url = f"https://inspirehep.net/api/literature?q={quote_plus(query)}&size={max(1, min(limit, 4))}"
    request = Request(url, headers={"User-Agent": "ResearchAssistant/1.0"})
    with urlopen(request, timeout=6) as response:
        payload = json.loads(response.read(1_500_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for hit in (payload.get("hits") or {}).get("hits") or []:
        hit_id = str(hit.get("id") or "").strip()
        meta = hit.get("metadata") or {}
        titles = meta.get("titles") or []
        title = _clean_html_text(titles[0].get("title") if titles and isinstance(titles[0], dict) else "")
        if not hit_id or not title:
            continue
        abstracts = meta.get("abstracts") or []
        abstract = _clean_html_text(abstracts[0].get("value") if abstracts and isinstance(abstracts[0], dict) else "")
        eprints = meta.get("arxiv_eprints") or []
        arxiv_id = str(eprints[0].get("value") if eprints and isinstance(eprints[0], dict) else "").strip() or None
        cites = meta.get("citation_count")
        date_str = str(meta.get("earliest_date") or meta.get("preprint_date") or "")[:10]
        tags = [p for p in (date_str, f"cites={cites}" if cites is not None else "", f"arXiv:{arxiv_id}" if arxiv_id else "") if p]
        prefix = f"[INSPIRE-HEP {' | '.join(tags)}]" if tags else "[INSPIRE-HEP]"
        results.append({
            "title": title,
            "url": f"https://inspirehep.net/literature/{hit_id}",
            "arxiv_id": arxiv_id,
            "summary": f"{prefix} {abstract or title}".strip(),
            "source": "web",
            "provider": "inspire_hep",
            "location_label": "INSPIRE-HEP 物理学专门库",
            "evidence_level": "web_commentary",
            "content_scope": "INSPIRE-HEP 物理学专门数据库（高能/理论/量子物理线索与摘要）",
        })
    return results[:limit]


def _search_zbmath(query: str, limit: int) -> List[Dict[str, Any]]:
    url = (
        "https://api.zbmath.org/v1/document/_search"
        f"?search_string={quote_plus(query)}&results_per_page={max(1, min(limit, 3))}"
    )
    request = Request(url, headers={"User-Agent": "ResearchAssistant/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=6) as response:
        payload = json.loads(response.read(1_000_000).decode("utf-8", errors="replace"))
    results: List[Dict[str, Any]] = []
    for item in payload.get("result") or []:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id") or "").strip()
        raw_title = item.get("title")
        title = _clean_html_text(raw_title.get("title") if isinstance(raw_title, dict) else raw_title or "")
        if not doc_id or not title:
            continue
        contribs = item.get("editorial_contributions") or []
        review_text = ""
        for c in contribs:
            if isinstance(c, dict) and c.get("text"):
                review_text = _clean_html_text(c.get("text"))
                break
        year = str(item.get("year") or "").strip()
        prefix = f"[zbMATH {year}]" if year else "[zbMATH]"
        results.append({
            "title": title,
            "url": f"https://zbmath.org/{doc_id}",
            "summary": f"{prefix} {review_text or title}".strip(),
            "source": "web",
            "provider": "zbmath",
            "location_label": "zbMATH Open 数学专门库",
            "evidence_level": "web_commentary",
            "content_scope": "zbMATH Open 数学专门文摘与专家评论库",
        })
    return results[:limit]


def _search_duckduckgo_html(query: str, limit: int) -> List[Dict[str, Any]]:
    request = Request(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
        headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"},
    )
    with urlopen(request, timeout=12) as response:
        html = response.read(2_000_000).decode("utf-8", errors="replace")
    parser = _DuckDuckGoParser()
    parser.feed(html)
    results = [
        {"title": item.get("title", "").strip(), "url": item.get("url", ""),
         "summary": item.get("summary", "").strip(), "source": "web",
         "provider": "duckduckgo",
         "evidence_level": "search_snippet", "content_scope": "search result snippet only"}
        for item in parser.items if item.get("title") and item.get("url")
    ][:limit]
    fetch_limit = max(0, min(int(os.getenv("WEB_PAGE_FETCH_LIMIT", "3")), limit, 8))
    for item in results[:fetch_limit]:
        try:
            page = fetch_web_page(item["url"])
            item.update(page)
        except Exception as exc:
            item["fetch_error"] = type(exc).__name__
    return results


def search_web(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Three-tier multidisciplinary web & paper commentary discovery:
    1) Optional Tavily / SearXNG API (if configured),
    2) Zero-config multidisciplinary commentary pool (Quanta Magazine, Wikipedia, Europe PMC, HF Papers),
    3) Isolated DuckDuckGo HTML fallback.
    """
    limit = max(1, min(limit, 15))
    combined: List[Dict[str, Any]] = []
    seen_urls = set()

    def _append_unique(items: List[Dict[str, Any]]) -> None:
        for item in items:
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            combined.append(item)
            if len(combined) >= limit:
                break

    # Layer 1: Optional Tavily / SearXNG API
    for provider_fn in (_search_tavily, _search_searxng):
        if len(combined) >= limit:
            break
        try:
            _append_unique(provider_fn(query, limit - len(combined)))
        except Exception:
            continue

    # Layer 2: Zero-config multidisciplinary commentary pool (run concurrently)
    if len(combined) < limit:
        per_source = max(2, min(4, limit))
        layer2_providers = (
            _search_inspire_hep,
            _search_quanta_magazine,
            _search_huggingface_papers,
            _search_zbmath,
            _search_europe_pmc,
            _search_wikipedia,
        )
        layer2_lists: List[List[Dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=len(layer2_providers)) as executor:
            futures = [executor.submit(fn, query, per_source) for fn in layer2_providers]
            for fut in futures:
                try:
                    items = fut.result(timeout=7)
                    if items:
                        layer2_lists.append(items)
                except Exception:
                    continue
        if layer2_lists:
            max_len = max(len(lst) for lst in layer2_lists)
            for idx in range(max_len):
                if len(combined) >= limit:
                    break
                round_items = [lst[idx] for lst in layer2_lists if idx < len(lst)]
                _append_unique(round_items)

    # Layer 3: Isolated DuckDuckGo HTML fallback
    if len(combined) < limit:
        try:
            _append_unique(_search_duckduckgo_html(query, limit - len(combined)))
        except Exception:
            pass

    return combined[:limit]


def _normalize_arxiv_id(raw_id: str) -> str:
    value = str(raw_id or "").strip()
    if not value:
        return ""
    value = re.sub(r"^https?://arxiv\.org/(?:abs|pdf|html)/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^arXiv:", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE).strip("/")
    if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?", value, flags=re.IGNORECASE):
        return value
    return ""


class _ArxivHTMLSectionParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections: List[Dict[str, str]] = []
        self._hidden_depth = 0
        self._in_heading = False
        self._heading_parts: List[str] = []
        self._current_title = ""
        self._current_body: List[str] = []

    def _flush(self) -> None:
        title = re.sub(r"\s+", " ", self._current_title).strip()
        body = re.sub(r"\s+", " ", " ".join(self._current_body)).strip()
        if title or body:
            self.sections.append({"title": title, "body": body})
        self._current_title = ""
        self._current_body = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"}:
            self._hidden_depth += 1
            return
        if self._hidden_depth:
            return
        if tag in {"h1", "h2", "h3"}:
            self._flush()
            self._in_heading = True
            self._heading_parts = []

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header"} and self._hidden_depth:
            self._hidden_depth -= 1
            return
        if self._hidden_depth:
            return
        if tag in {"h1", "h2", "h3"} and self._in_heading:
            self._current_title = " ".join(self._heading_parts)
            self._in_heading = False
            self._heading_parts = []

    def handle_data(self, data):
        if self._hidden_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_heading:
            self._heading_parts.append(text)
        else:
            self._current_body.append(text)

    def close(self):
        super().close()
        self._flush()


_ARXIV_PRIORITY_SECTION_RE = re.compile(
    r"\b(?:method|methodology|approach|architecture|model|framework|algorithm|"
    r"experiment|experiments|evaluation|result|results|ablation|implementation|setup|benchmark)\b",
    re.IGNORECASE,
)
_ARXIV_SKIP_SECTION_RE = re.compile(
    r"\b(?:abstract|reference|references|bibliography|acknowledg|appendix)\b",
    re.IGNORECASE,
)


def fetch_arxiv_html_sections(arxiv_id: str, max_chars: int = 4500) -> Dict[str, Any]:
    """Fetch method/experiment sections from arXiv HTML on demand without indexing into ChromaDB."""
    clean_id = _normalize_arxiv_id(arxiv_id)
    if not clean_id:
        return {}
    url = f"https://arxiv.org/html/{clean_id}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)"})
    with urlopen(request, timeout=10) as response:
        content_type = str(response.headers.get("Content-Type", "")).casefold()
        if content_type and "html" not in content_type and "text/" not in content_type:
            return {}
        raw = response.read(2_000_000)
        charset = response.headers.get_content_charset() or "utf-8"

    parser = _ArxivHTMLSectionParser()
    parser.feed(raw.decode(charset, errors="replace"))
    parser.close()

    priority_chunks: List[str] = []
    priority_titles: List[str] = []
    fallback_chunks: List[str] = []
    fallback_titles: List[str] = []

    for sec in parser.sections:
        title = sec["title"]
        body = sec["body"]
        if len(body) < 80 or _ARXIV_SKIP_SECTION_RE.search(title):
            continue
        snippet = f"[{title}] {body}" if title else body
        if _ARXIV_PRIORITY_SECTION_RE.search(title):
            priority_titles.append(title)
            priority_chunks.append(snippet)
        else:
            fallback_titles.append(title)
            fallback_chunks.append(snippet)

    chosen_chunks = priority_chunks or fallback_chunks
    chosen_titles = priority_titles or fallback_titles
    if not chosen_chunks:
        return {}

    combined = "\n\n".join(chosen_chunks).strip()[:max_chars]
    return {
        "arxiv_id": clean_id,
        "url": url,
        "sections": [t for t in chosen_titles if t][:8],
        "text": combined,
        "evidence_level": "full_text_excerpt",
        "content_scope": "arXiv HTML 临时抓取方法/实验节（未入本地库）",
    }


def search_datasets(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Discover public Hugging Face datasets through its public search API."""
    url = f"https://huggingface.co/api/datasets?search={quote_plus(query)}&limit={max(1, min(limit, 20))}"
    request = Request(url, headers={"User-Agent": "ResearchAssistant/1.0"})
    with urlopen(request, timeout=12) as response:
        payload = json.loads(response.read(2_000_000).decode("utf-8"))
    items = []
    for item in payload if isinstance(payload, list) else []:
        repo_id = item.get("id") or item.get("repo_id")
        if not repo_id:
            continue
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        items.append({
            "title": repo_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "summary": str(card_data.get("description") or item.get("description") or "Dataset repository metadata"),
            "source": "huggingface_dataset",
            "downloads": item.get("downloads"),
            "evidence_level": "repository_metadata",
        })
    return items
