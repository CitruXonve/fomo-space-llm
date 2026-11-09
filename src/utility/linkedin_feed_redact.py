"""Redact a local LinkedIn feed HTML fragment before it leaves the machine.

Keeps visible card text and LinkedIn hrefs for ``/feed/update/``, ``/posts/``,
and ``/in/``. Drops scripts, embedded frames, comments, and session-like
values. This is the privacy boundary for the Firecrawl parse arm: Firecrawl's
``redactPII`` option only scrubs returned markdown.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup, Comment

from src.utility.linkedin_feed_parser import normalize_linkedin_url

_DROP_TAGS = ("script", "style", "noscript", "iframe", "meta")
_DROP_ATTRS = {"nonce", "integrity", "src", "srcset", "action", "csrf", "csrf-token", "csrf_token"}
_KEEP_HREF_MARKERS = ("/feed/update/", "/posts/", "/in/")

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)"
    r"(?:li_at|jsessionid)\s*[=:]\s*[^\s\"'<>;&]+"
    r"|ajax:\d+"
    r"|csrf(?:[-_]?token)?[\"']?\s*[:=]\s*[\"'][^\"']+[\"']"
)


def redact_feed_html(html: str) -> str:
    """Return a feed fragment safe to upload, or ``""`` when ``html`` is empty."""
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in _DROP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    for tag in soup.find_all(True):
        for name in list(tag.attrs):
            lowered = name.lower()
            if lowered.startswith("on") or lowered in _DROP_ATTRS or "csrf" in lowered:
                del tag[name]
                continue
            if lowered == "href":
                kept = _kept_href(tag.get(name))
                if kept:
                    tag[name] = kept
                else:
                    del tag[name]
                continue
            scrubbed = _scrub_attr(tag.get(name))
            if scrubbed is None:
                del tag[name]
            else:
                tag[name] = scrubbed

    for text in list(soup.find_all(string=True)):
        if isinstance(text, Comment):
            text.extract()
            continue
        cleaned = _scrub_secrets(str(text))
        if cleaned != str(text):
            text.replace_with(cleaned)

    return str(soup)


def _kept_href(raw: object) -> str:
    href = _attr_text(raw)
    if not href:
        return ""
    normalized = normalize_linkedin_url(href)
    if not normalized:
        return ""
    parsed = urlparse(normalized)
    path = parsed.path.lower()
    if not any(marker in path for marker in _KEEP_HREF_MARKERS):
        return ""
    return urlunparse(parsed._replace(query="", fragment=""))


def _scrub_attr(raw: object) -> str | None:
    text = _scrub_secrets(_attr_text(raw))
    return text or None


def _attr_text(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return " ".join(str(part) for part in raw)
    return str(raw)


def _scrub_secrets(text: str) -> str:
    text = _EMAIL_RE.sub("", text)
    text = _SECRET_RE.sub("", text)
    return text
