import json
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse as _urlparse

import re as _re

_GENERIC_SELECTORS = {
    "a", "button", "div", "span", "input", "form", "select",
    "textarea", "li", "ul", "ol", "nav", "header", "footer", "section",
    "article", "main", "p", "h1", "h2", "h3", "h4", "h5", "h6", "label",
    "img", "svg", "table", "tr", "td", "th", "dl", "dt", "dd", "figure",
    "aside", "summary", "details",
}

_GENERIC_ATTR_PATTERNS = (
    '[type="button"]',
    '[type="submit"]',
    '[role="button"]',
    '[role="link"]',
    '[role="menuitem"]',
    '[tabindex="0"]',
    '[tabindex="-1"]',
)

_COOKIE_FRAGMENTS = (
    "onetrust", "cookieconsent", "cookie-consent", "cookie-banner",
    "gdpr", "ccpa", "consent", "privacy-banner", "cookie-notice",
    "banner-close", "accept-all", "cookie-law", "cookie-popup",
)


def _is_valid_selector(selector: str) -> bool:
    s = (selector or "").strip()
    if not s:
        return False

    if s in _GENERIC_SELECTORS:
        return False

    if s in _GENERIC_ATTR_PATTERNS:
        return False

    if len(s) > 200:
        return False

    s_lower = s.lower()
    if any(frag in s_lower for frag in _COOKIE_FRAGMENTS):
        return False

    if _re.search(r":nth-(?:child|of-type)\(\d+\)", s) and "#" not in s and "." not in s:
        return False

    return True


def _path_prefix(url: str) -> str:
    """Extract a stable 2-segment path prefix from a URL for memory scoping.

    Examples:
      https://www.booking.com/search/hotels  -> "/search"
      https://www.booking.com/hotel/gb/lon   -> "/hotel"
      https://www.apple.com/shop/buy-mac/mac-mini -> "/shop"
      https://example.com/                   -> "/"
    """
    try:
        path = _urlparse(url).path or "/"
        parts = [p for p in path.split("/") if p]
        return "/" + parts[0] if parts else "/"
    except Exception:
        return "/"


@dataclass
class MemoryEntry:
    host: str
    intent: str
    action: str
    selector: str
    selector_type: str
    strategy: str
    success_count: int = 0
    last_score: Optional[float] = None
    # First path segment of the URL when this selector was recorded.
    # Empty string means "recorded before path-awareness was added" (matches any path).
    url_path: str = ""


class SelectorMemory:
    """Persistent, path-aware memory of successful selector resolutions.

    Key: (host, url_path, intent, action)

    Lookup prefers entries recorded on the same url_path; falls back to
    entries with url_path="" (recorded by older code) so no history is lost.

    Writes are batched: disk I/O only happens every ``write_batch`` records or
    when ``flush()`` is called explicitly.
    """

    def __init__(
        self,
        path: str = "./cache/selector_memory.json",
        write_batch: int = 10,
        max_entries: int = 2_000,
    ):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_batch = max(1, write_batch)
        self._max_entries = max_entries
        self._entries: list[MemoryEntry] = []
        self._pending_writes = 0
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            entries = []
            for x in raw:
                if not isinstance(x, dict):
                    continue
                # Backward compatibility: old entries lack url_path
                x.setdefault("url_path", "")
                try:
                    entries.append(MemoryEntry(**x))
                except TypeError:
                    pass  # Unknown field from future version - skip safely
            self._entries = entries
        except Exception:
            self._entries = []

    def flush(self) -> None:
        """Persist all pending in-memory entries to disk immediately."""
        with self._lock:
            self._write_locked()
            self._pending_writes = 0

    def _write_locked(self) -> None:
        """Write entries to disk. Must be called while holding self._lock."""
        try:
            self._path.write_text(
                json.dumps([asdict(e) for e in self._entries], indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_success(
        self,
        host: str,
        intent: str,
        action: str,
        selector: str,
        selector_type: str,
        strategy: str,
        score: Optional[float],
        url: str = "",
    ) -> None:
        host = host.lower().strip()
        intent = intent.strip().lower()
        action = action.strip().lower()
        selector = selector.strip()
        selector_type = selector_type.strip().lower()
        strategy = strategy.strip().lower()
        url_path = _path_prefix(url) if url else ""

        if not _is_valid_selector(selector):
            return  # Reject generic or cookie-banner selectors

        with self._lock:
            for e in self._entries:
                if (
                    e.host == host
                    and e.intent == intent
                    and e.action == action
                    and e.selector == selector
                    and e.selector_type == selector_type
                    and e.url_path == url_path
                ):
                    e.success_count += 1
                    e.last_score = score
                    self._pending_writes += 1
                    if self._pending_writes >= self._write_batch:
                        self._write_locked()
                        self._pending_writes = 0
                    return

            self._entries.append(
                MemoryEntry(
                    host=host,
                    intent=intent,
                    action=action,
                    selector=selector,
                    selector_type=selector_type,
                    strategy=strategy,
                    success_count=1,
                    last_score=score,
                    url_path=url_path,
                )
            )
            self._entries.sort(key=lambda x: x.success_count, reverse=True)
            self._entries = self._entries[: self._max_entries]
            self._pending_writes += 1
            if self._pending_writes >= self._write_batch:
                self._write_locked()
                self._pending_writes = 0

    def best_for(
        self,
        host: str,
        intent: str,
        action: str,
        url: str = "",
    ) -> Optional[MemoryEntry]:
        host = host.lower().strip()
        intent = intent.strip().lower()
        action = action.strip().lower()
        url_path = _path_prefix(url) if url else ""

        with self._lock:
            all_matches = [
                e for e in self._entries
                if (
                    e.host == host
                    and e.intent == intent
                    and e.action == action
                    and _is_valid_selector(e.selector)
                )
            ]

        if not all_matches:
            return None

        # Prefer exact path match; fall back to path-agnostic entries (url_path="").
        path_matches = [e for e in all_matches if e.url_path == url_path]
        fallback_matches = [e for e in all_matches if e.url_path == ""]
        candidates = path_matches or fallback_matches or all_matches

        candidates.sort(key=lambda e: (e.success_count, e.last_score or 0.0), reverse=True)
        return candidates[0]

    def __del__(self) -> None:
        """Flush remaining unsaved entries on garbage collection."""
        try:
            if self._pending_writes > 0:
                self.flush()
        except Exception:
            pass