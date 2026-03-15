"""
Wikipedia article downloader for the Narrative Forensics Tool.

Downloads the current revision of a Wikipedia article via the
MediaWiki REST API and returns structured text suitable for the
analysis pipeline.

Optionally fetches multiple historical revisions to enable version
diff analysis (Phase 2).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse, unquote

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_BASE = "https://en.wikipedia.org/api/rest_v1"
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "NarrativeForensicsTool/1.0 (research; contact: local)"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WikiRevision:
    revid: int
    timestamp: str
    user: str
    comment: str
    text: str


@dataclass
class WikiArticle:
    url: str
    title: str
    page_id: int
    current_text: str
    sections: List[dict] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    revisions: List[WikiRevision] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "page_id": self.page_id,
            "current_text": self.current_text,
            "sections": self.sections,
            "categories": self.categories,
            "references": self.references,
            "revisions": [
                {
                    "revid": r.revid,
                    "timestamp": r.timestamp,
                    "user": r.user,
                    "comment": r.comment,
                }
                for r in self.revisions
            ],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _title_from_url(url: str) -> str:
    """Extract the Wikipedia article title from a URL."""
    parsed = urlparse(url)
    path = parsed.path  # e.g. /wiki/Some_Article_Title
    parts = path.split("/wiki/", 1)
    if len(parts) == 2:
        return unquote(parts[1]).replace("_", " ")
    # Fallback: use the last path segment
    return unquote(path.rstrip("/").split("/")[-1]).replace("_", " ")


def _strip_wiki_markup(wikitext: str) -> str:
    """
    Very lightweight wikitext cleaner that removes the most common
    markup so the resulting text is readable plain text.

    This is intentionally simple – a full parser is not needed for
    structural signal detection.
    """
    text = wikitext

    # Remove nowiki blocks
    text = re.sub(r"<nowiki>.*?</nowiki>", "", text, flags=re.DOTALL)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove infobox / template blocks (double-brace)
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if text[i : i + 2] == "{{":
            depth += 1
            i += 2
        elif text[i : i + 2] == "}}":
            depth = max(0, depth - 1)
            i += 2
        else:
            if depth == 0:
                result.append(text[i])
            i += 1
    text = "".join(result)

    # Remove [[File:...]] and [[Image:...]] blocks
    text = re.sub(r"\[\[(File|Image):.*?\]\]", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Convert [[link|label]] → label, [[link]] → link
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # Remove remaining [ ] external links, keep label if present
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove section headings markup but keep text
    text = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\n\n\1\n", text)

    # Remove bold/italic markup
    text = re.sub(r"'{2,3}", "", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _extract_references(wikitext: str) -> List[str]:
    """Pull raw citation strings from wikitext."""
    # Named and unnamed <ref> tags
    refs = re.findall(r"<ref[^>]*>(.*?)</ref>", wikitext, flags=re.DOTALL)
    # {{cite …}} templates (rough)
    cites = re.findall(r"\{\{cite[^}]+\}\}", wikitext, flags=re.IGNORECASE)
    combined = refs + cites
    # Strip markup from each reference string minimally
    cleaned = [re.sub(r"\s+", " ", r).strip() for r in combined if r.strip()]
    return cleaned


def _extract_sections(wikitext: str) -> List[dict]:
    """Return a list of {title, level, text} for each section."""
    heading_pat = re.compile(r"(={2,6})\s*(.*?)\s*\1", re.MULTILINE)
    matches = list(heading_pat.finditer(wikitext))
    sections = []
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(wikitext)
        raw_body = wikitext[start:end]
        sections.append({"title": title, "level": level, "text": _strip_wiki_markup(raw_body)})
    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_article(url: str, fetch_revisions: int = 0) -> WikiArticle:
    """
    Download a Wikipedia article and return structured text.

    Args:
        url: Full Wikipedia article URL
              (e.g. https://en.wikipedia.org/wiki/Some_Topic).
        fetch_revisions: Number of recent revisions to also download
                         (0 = current version only).

    Returns:
        WikiArticle with plain text, sections, references, and optional
        revision history.
    """
    session = _make_session()
    title = _title_from_url(url)

    # ------------------------------------------------------------------
    # Step 1: get current wikitext via the MediaWiki API
    # ------------------------------------------------------------------
    params: dict = {
        "action": "query",
        "titles": title,
        "prop": "revisions|categories|info",
        "rvprop": "ids|timestamp|user|comment|content",
        "rvslots": "main",
        "rvlimit": 1,
        "inprop": "url",
        "cllimit": "max",
        "format": "json",
        "formatversion": 2,
    }

    try:
        response = session.get(WIKIPEDIA_API_BASE, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return WikiArticle(url=url, title=title, page_id=-1, current_text="", error=str(exc))

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return WikiArticle(url=url, title=title, page_id=-1, current_text="", error="No pages returned by API")

    page = pages[0]
    if "missing" in page:
        return WikiArticle(url=url, title=title, page_id=-1, current_text="", error=f"Article not found: {title!r}")

    page_id: int = page.get("pageid", -1)
    revisions = page.get("revisions", [])
    raw_wikitext = ""
    if revisions:
        slots = revisions[0].get("slots", {})
        raw_wikitext = slots.get("main", {}).get("content", "")

    categories = [c.get("title", "").replace("Category:", "") for c in page.get("categories", [])]

    plain_text = _strip_wiki_markup(raw_wikitext)
    sections = _extract_sections(raw_wikitext)
    references = _extract_references(raw_wikitext)

    article = WikiArticle(
        url=url,
        title=title,
        page_id=page_id,
        current_text=plain_text,
        sections=sections,
        categories=categories,
        references=references,
    )

    # ------------------------------------------------------------------
    # Step 2: optionally fetch N historical revisions
    # ------------------------------------------------------------------
    if fetch_revisions > 0:
        rev_params: dict = {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "ids|timestamp|user|comment|content",
            "rvslots": "main",
            "rvlimit": fetch_revisions,
            "format": "json",
            "formatversion": 2,
        }
        try:
            rev_response = session.get(WIKIPEDIA_API_BASE, params=rev_params, timeout=REQUEST_TIMEOUT)
            rev_response.raise_for_status()
            rev_data = rev_response.json()
            rev_pages = rev_data.get("query", {}).get("pages", [])
            if rev_pages:
                for rev in rev_pages[0].get("revisions", []):
                    slots = rev.get("slots", {})
                    content = slots.get("main", {}).get("content", "")
                    article.revisions.append(
                        WikiRevision(
                            revid=rev.get("revid", 0),
                            timestamp=rev.get("timestamp", ""),
                            user=rev.get("user", ""),
                            comment=rev.get("comment", ""),
                            text=_strip_wiki_markup(content),
                        )
                    )
        except requests.RequestException:
            pass  # revision fetch failure is non-fatal

    return article


def fetch_article_by_title(title: str, fetch_revisions: int = 0) -> WikiArticle:
    """Convenience wrapper that accepts a title instead of a full URL."""
    safe_title = title.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{safe_title}"
    return fetch_article(url, fetch_revisions=fetch_revisions)
