"""Optional open web and dataset discovery adapters.

These adapters return source-labelled records. They never claim that a result
was read in full or independently verified.
"""
from __future__ import annotations

import json
import ipaddress
import os
import re
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


def search_web(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """General web discovery via DuckDuckGo HTML; results are snippets only."""
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
