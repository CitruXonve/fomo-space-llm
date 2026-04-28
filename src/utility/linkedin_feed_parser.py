"""Extract LinkedIn feed post summaries from HTML.

Uses BeautifulSoup plus regex over embedded Voyager-style JSON (saved pages often
lack classic ``feed-shared-update-v2`` cards). Falls back to DOM card heuristics
when those elements exist after hydration.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

LINKEDIN_ORIGIN = "https://www.linkedin.com"
_SNIPPET_MAX_DEFAULT = 400
_PHASE_A_DOM_THRESHOLD = 3

_FEED_UPDATE_ABS_RE = re.compile(
    r"https://(?:[\w.-]+\.)?linkedin\.com/feed/update/[^\s\"\'<>\\\]\)]+",
    re.IGNORECASE,
)
_FEED_UPDATE_ANY_RE = re.compile(
    r"(?:https?:)?//(?:[\w.-]+\.)?linkedin\.com/feed/update/[^\s\"\'<>\\\]\)]+",
    re.IGNORECASE,
)
_ACTIVITY_ID_RE = re.compile(r"urn:li:activity:(\d+)", re.I)
_POSTS_PATH_RE = re.compile(
    r"https://(?:[\w.-]+\.)?linkedin\.com/posts/[^\s\"\'<>\\\]\)]+", re.I
)


def _truncate(s: str, max_len: int) -> str:
    if max_len <= 0 or len(s) <= max_len:
        return s
    return s[:max_len].rstrip() + "…"


def _clean_captured_url(raw: str) -> str:
    raw = raw.rstrip(' \t\n\r,.;)]\\"\'')
    if "&quot;" in raw:
        raw = raw.split("&quot;", 1)[0]
    if '"' in raw:
        raw = raw.split('"', 1)[0]
    return raw.strip()


def normalize_linkedin_url(raw: str) -> str:
    """Resolve relative URLs and strip trailing JSON-quote junk."""
    raw = html_lib.unescape(raw.strip())
    raw = _clean_captured_url(raw)
    if raw.startswith("//"):
        raw = "https:" + raw
    elif raw.startswith("/"):
        raw = urljoin(LINKEDIN_ORIGIN, raw)
    elif not raw.startswith("http"):
        raw = urljoin(LINKEDIN_ORIGIN + "/", raw.lstrip("/"))
    host = urlparse(raw).netloc.lower()
    if "linkedin.com" not in host:
        return ""
    return raw


def activity_id_from_url(url: str) -> str | None:
    m = _ACTIVITY_ID_RE.search(url)
    return m.group(1) if m else None


def _dedupe_key(post_url: str) -> str:
    aid = activity_id_from_url(post_url)
    return f"activity:{aid}" if aid else post_url.split("?", 1)[0]


def _guess_snippet_near_activity(plain: str, url: str) -> str:
    aid = activity_id_from_url(url)
    if not aid:
        return ""
    idx = plain.find(aid)
    if idx == -1:
        return ""
    window = plain[max(0, idx - 2000) : idx + 4000]
    tm = re.search(
        r'"commentary"\s*:\s*\{[^\}]{0,800}"text"\s*:\s*"((?:[^"\\]|\\.)*)"',
        window,
        re.DOTALL,
    )
    if not tm:
        return ""
    raw_txt = tm.group(1)
    try:
        decoded = json.loads(f'"{raw_txt}"')
        if isinstance(decoded, str):
            return decoded.replace("\n", " ").strip()
    except json.JSONDecodeError:
        pass
    return raw_txt.replace("\\n", " ").strip()[:800]


def _collect_urls_phase_a(plain: str, soup: BeautifulSoup) -> set[str]:
    found: set[str] = set()
    for rx in (_FEED_UPDATE_ABS_RE, _FEED_UPDATE_ANY_RE, _POSTS_PATH_RE):
        for m in rx.finditer(plain):
            u = normalize_linkedin_url(_clean_captured_url(m.group(0)))
            if u and ("feed/update" in u or "/posts/" in u):
                found.add(u)
    for tag in soup.find_all("a", href=True):
        href = html_lib.unescape(tag["href"])
        if "/feed/update/" in href or "/posts/" in href:
            u = normalize_linkedin_url(href)
            if u:
                found.add(u)
    return found


def _rows_from_urls(
    urls: set[str],
    plain: str,
    snippet_max_len: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for u in urls:
        key = _dedupe_key(u)
        if key in seen:
            continue
        seen.add(key)
        snippet = _guess_snippet_near_activity(plain, u)
        rows.append(
            {
                "post_url": u,
                "author_profile_url": "",
                "text_snippet": _truncate(snippet, snippet_max_len),
                "relative_time": "",
            }
        )
    return rows


def _extract_phase_b_cards(soup: BeautifulSoup, snippet_max_len: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def is_card_div(tag) -> bool:
        if tag.name != "div":
            return False
        classes = tag.get("class") or []
        return any(
            "feed-shared-update" in str(c).lower()
            or "feed-shared-update-v2" in str(c).lower()
            for c in classes
        )

    candidates = soup.find_all(is_card_div)
    if not candidates:
        candidates = list(soup.find_all("article"))

    for card in candidates:
        post_url = ""
        for a in card.find_all("a", href=True):
            href = html_lib.unescape(a["href"])
            if "/feed/update/" in href or "/posts/" in href:
                nu = normalize_linkedin_url(href)
                if nu:
                    post_url = nu
                    break
        if not post_url:
            continue
        key = _dedupe_key(post_url)
        if key in seen:
            continue
        seen.add(key)

        author = ""
        for a in card.find_all("a", href=True):
            href = html_lib.unescape(a["href"])
            if "/in/" in href and "/company/" not in href:
                nu = normalize_linkedin_url(href)
                if nu and "/feed/" not in nu:
                    author = nu
                    break

        text = card.get_text(separator=" ", strip=True)
        rel_time = ""
        tel = card.find("time")
        if tel:
            rel_time = (tel.get("datetime") or tel.get_text(strip=True) or "")[:120]

        rows.append(
            {
                "post_url": post_url,
                "author_profile_url": author,
                "text_snippet": _truncate(text, snippet_max_len),
                "relative_time": rel_time,
            }
        )
    return rows


def parse_feed_posts(
    html: str,
    *,
    snippet_max_len: int = _SNIPPET_MAX_DEFAULT,
) -> list[dict[str, str]]:
    """Parse LinkedIn feed HTML into compact post rows.

    Phase A collects ``feed/update`` and ``/posts/`` URLs from anchors and regex
    over the raw document (works for Voyager JSON blobs). Phase B runs when
    Phase A finds fewer than ``_PHASE_A_DOM_THRESHOLD`` rows and scans for
    classic feed card markup.

    Each dict has: ``post_url``, ``author_profile_url``, ``text_snippet``,
    ``relative_time``. Never raises; logs parse failures at warning level.
    """
    if not html or not html.strip():
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        logger.warning("BeautifulSoup parse failed: %s", exc)
        return []

    plain = html_lib.unescape(html)
    urls = _collect_urls_phase_a(plain, soup)
    rows_a = _rows_from_urls(urls, plain, snippet_max_len)

    if len(rows_a) >= _PHASE_A_DOM_THRESHOLD:
        return rows_a

    rows_b = _extract_phase_b_cards(soup, snippet_max_len)

    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []

    def add_row(r: dict[str, str]) -> None:
        key = _dedupe_key(r["post_url"])
        if key in merged:
            prev = merged[key]
            if not prev.get("text_snippet") and r.get("text_snippet"):
                prev["text_snippet"] = r["text_snippet"]
            if not prev.get("author_profile_url") and r.get("author_profile_url"):
                prev["author_profile_url"] = r["author_profile_url"]
            if not prev.get("relative_time") and r.get("relative_time"):
                prev["relative_time"] = r["relative_time"]
        else:
            merged[key] = dict(r)
            order.append(key)

    for r in rows_a:
        add_row(r)
    for r in rows_b:
        add_row(r)

    return [merged[k] for k in order]


__all__ = [
    "LINKEDIN_ORIGIN",
    "normalize_linkedin_url",
    "parse_feed_posts",
    "activity_id_from_url",
]
