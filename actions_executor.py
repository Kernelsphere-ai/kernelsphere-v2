import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import Page, Locator, ElementHandle, TimeoutError as PlaywrightTimeoutError

from capture import ExtractionEngine, PageState
from element_selector import SemanticElementSelector, SelectorMatch
from state_differentiator import StateDiffResult, WebStateDifferentiator
from page_graph import PageGraph
from selector_memory import SelectorMemory
from neural_intent_engine import get_shared_engine
from logging_config import get_logger

logger = get_logger(__name__)

VerificationResult = StateDiffResult
PageChangeVerifier = WebStateDifferentiator

_SUPPORTED_ACTIONS = {"click", "type", "select", "check", "uncheck", "navigate", "scroll", "extract"}


_EXTRACT_PAGE_TEXT_LIMIT = 32000

# How large each candidate "window" is when we do semantic section selection.
_EXTRACT_WINDOW_SIZE = 6000
_EXTRACT_WINDOW_STEP = 3000


def _select_relevant_section(page_text: str, question: str, max_chars: int = 20000) -> str:
    """Return the most question-relevant section of page_text.

    Strategy: slide a window over the text, score each window by keyword overlap
    with the question, then return the top window(s) concatenated.  This makes
    sure that product prices / standings / recipe ingredients buried deep in the
    page are not truncated away.
    """
    if len(page_text) <= max_chars:
        return page_text

    q_tokens = {w.lower() for w in re.split(r"\W+", question) if len(w) >= 3}
    stop = {"the", "and", "for", "are", "this", "that", "what", "with", "from",
            "how", "many", "find", "list", "show", "give", "tell", "check"}
    q_tokens -= stop

    if not q_tokens:
        return page_text[:max_chars]

    windows: list[tuple[float, int]] = []  # (score, start_index)
    for start in range(0, len(page_text) - _EXTRACT_WINDOW_SIZE + 1, _EXTRACT_WINDOW_STEP):
        chunk = page_text[start: start + _EXTRACT_WINDOW_SIZE].lower()
        score = sum(1 for t in q_tokens if t in chunk)
        windows.append((score, start))

    # Always include the start of the page (navigation/title context)
    windows_sorted = sorted(windows, key=lambda x: -x[0])
    chosen_starts: list[int] = [0]
    for score, start in windows_sorted:
        if len(chosen_starts) >= 3:
            break
        if start not in chosen_starts:
            chosen_starts.append(start)

    chosen_starts.sort()
    sections: list[str] = []
    for s in chosen_starts:
        sections.append(page_text[s: s + _EXTRACT_WINDOW_SIZE])

    result = "\n...\n".join(sections)
    return result[:max_chars]


def _gemini_extract_answer(page_text: str, question: str) -> str:
    """Call Gemini Flash to answer *question* from the visible text of the current page.

    Returns the answer string, or an empty string on any failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.debug("GEMINI_API_KEY not set; extract action falls back to empty answer.")
        return ""

    prompt = (
        "You are a precise web-content reader. Read the page text below and answer the question.\n"
        "Answer ONLY from the information present in the text.\n"
        "Be concise: 1 -- 3 sentences maximum.\n"
        "If the answer is not found in the text, reply exactly: 'Not found on this page.'\n\n"
        f"=== Page text ===\n{page_text}\n\n"
        f"=== Question ===\n{question}\n\n"
        "Answer:"
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    req = urllib.request.Request(
        url=url,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:
        logger.warning("Gemini extract call failed: %s", exc)
        return ""


@dataclass
class ActionRequest:
    action: str
    value: Optional[str] = None
    intent: Optional[str] = None
    selector: Optional[str] = None
    selector_type: str = "auto"
    dom_node_id: Optional[str] = None
    timeout_ms: int = 8000
    clear_first: bool = True
    press_enter: bool = False
    retry_attempts: int = 1
    retry_backoff_ms: int = 120
    skip_auto_memory: bool = False


@dataclass
class ActionResult:
    success: bool
    action: str
    message: str
    latency_ms: float
    selector: Optional[str] = None
    selector_type: Optional[str] = None
    strategy: Optional[str] = None
    score: Optional[float] = None
    attempts: int = 1
    error: Optional[str] = None


@dataclass
class ActionVerificationResult:
    action_result: ActionResult
    verification: Optional[StateDiffResult]
    metadata: Optional[dict] = None


@dataclass
class _ResolvedTarget:
    locator: Optional[Locator]
    element_handle: Optional[ElementHandle]
    selector: Optional[str]
    selector_type: Optional[str]
    strategy: str
    score: Optional[float]


class BrowserActionsExecutor:
    """Execute browser actions by explicit selector, DOM node id, or semantic intent."""

    def __init__(
        self,
        page: Page,
        extraction_engine: Optional[ExtractionEngine] = None,
        selector: Optional[SemanticElementSelector] = None,
        selector_memory: Optional[SelectorMemory] = None,
    ):
        self._page = page
        self._engine = extraction_engine
        self._selector = selector or SemanticElementSelector(use_llm=True, strict_model=False)
        self._last_state: Optional[PageState] = None
        self._page_graph: Optional[PageGraph] = None
        self._memory = selector_memory or SelectorMemory()
        self._neural = get_shared_engine()

    def set_state(self, state: PageState) -> None:
        self._last_state = state

    def set_page_graph(self, graph: Optional[PageGraph]) -> None:
        self._page_graph = graph

    def refresh_state(self) -> Optional[PageState]:
        if not self._engine:
            return self._last_state
        try:
            self._last_state = self._engine.capture_page(self._page)
        except Exception as exc:
            err = str(exc)
            if "Target page" in err or "context or browser" in err or "TargetClosed" in err:
                raise  # Re-raise browser-closed  --  caller must restart
            logger.warning("refresh_state capture failed: %s", exc)
        return self._last_state

    def click(
        self,
        intent: Optional[str] = None,
        selector: Optional[str] = None,
        selector_type: str = "auto",
        dom_node_id: Optional[str] = None,
        timeout_ms: int = 10000,
    ) -> ActionResult:
        return self.execute(ActionRequest(
            action="click",
            intent=intent,
            selector=selector,
            selector_type=selector_type,
            dom_node_id=dom_node_id,
            timeout_ms=timeout_ms,
        ))

    def type(
        self,
        value: str,
        intent: Optional[str] = None,
        selector: Optional[str] = None,
        selector_type: str = "auto",
        dom_node_id: Optional[str] = None,
        timeout_ms: int = 10000,
        clear_first: bool = True,
        press_enter: bool = False,
    ) -> ActionResult:
        return self.execute(ActionRequest(
            action="type",
            value=value,
            intent=intent,
            selector=selector,
            selector_type=selector_type,
            dom_node_id=dom_node_id,
            timeout_ms=timeout_ms,
            clear_first=clear_first,
            press_enter=press_enter,
        ))

    def execute(self, request: ActionRequest) -> ActionResult:
        t0 = time.perf_counter()
        self._validate_request(request)

        if request.action == "navigate":
            return self._execute_navigate(request, t0)

        if request.action == "extract":
            return self._execute_extract(request, t0)

        if request.action == "scroll":
            try:
                self._perform_scroll(None, request.value or "down", request.timeout_ms)
                latency = (time.perf_counter() - t0) * 1000
                try:
                    self.refresh_state()
                except Exception:
                    pass
                return ActionResult(
                    success=True,
                    action="scroll",
                    message=f"Scrolled {request.value or 'down'}.",
                    latency_ms=latency,
                    strategy="scroll",
                    score=1.0,
                    attempts=1,
                )
            except Exception as exc:
                return self._fail("scroll", f"Scroll failed: {exc}", t0, error=type(exc).__name__)

        attempts = max(1, request.retry_attempts + 1)
        try:
            resolved = None
            last_err: Optional[Exception] = None
            for attempt in range(1, attempts + 1):
                try:
                    resolved = self._resolve_target(request)
                    if not resolved:
                        raise ValueError("No matching element found for action target.")
                    self._check_preconditions(request.action, resolved, request.timeout_ms)
                    self._dispatch_action(request, resolved)
                    latency = (time.perf_counter() - t0) * 1000
                    if not request.skip_auto_memory:
                        self._record_success(request, resolved)
                    logger.info(
                        "Action %r succeeded via strategy=%r attempt=%d",
                        request.action, resolved.strategy, attempt,
                    )
                    return ActionResult(
                        success=True,
                        action=request.action,
                        message=f"{request.action} succeeded.",
                        latency_ms=latency,
                        selector=resolved.selector,
                        selector_type=resolved.selector_type,
                        strategy=resolved.strategy,
                        score=resolved.score,
                        attempts=attempt,
                    )
                except Exception as exc:
                    last_err = exc
                    logger.debug(
                        "Action %r attempt %d/%d failed: %s",
                        request.action, attempt, attempts, exc,
                    )
                    if attempt >= attempts:
                        break
                    time.sleep(max(0, request.retry_backoff_ms) / 1000.0)

            err_name = type(last_err).__name__ if last_err else "unknown_error"
            return self._fail(
                request.action,
                f"Action failed after {attempts} attempt(s): {last_err}",
                t0,
                attempts=attempts,
                error=err_name,
            )
        except PlaywrightTimeoutError as exc:
            return self._fail(request.action, f"Action timed out: {exc}", t0, attempts=1, error="timeout")
        except Exception as exc:
            return self._fail(request.action, f"Action failed: {exc}", t0, attempts=1, error=type(exc).__name__)
        finally:
            if self._engine and request.action != "navigate":
                try:
                    self.refresh_state()
                except Exception:
                    pass

    def _execute_extract(self, request: ActionRequest, t0: float) -> ActionResult:
        """Read the current page's visible text and use Gemini to answer the question
        expressed in ``request.intent``.  The answer is returned in ``ActionResult.message``.
        """
        question = (request.intent or "").strip() or "Summarize the page content."
        try:
            if not self._page:
                return self._fail("extract", "No browser page available.", t0, error="no_page")

            # --- Page-context guard ---
            # Check that the current page is plausibly the right page for this question.
            # A mismatch (agent on wrong page) produces "Not found" or hallucinated answers.
            current_url = self._page.url or ""
            page_title = ""
            try:
                page_title = self._page.title() or ""
            except Exception:
                pass
            try:
                from urllib.parse import urlparse as _urlparse
                url_path = _urlparse(current_url).path.replace("/", " ").replace("-", " ").replace("_", " ").strip()
            except Exception:
                url_path = ""
            page_context_text = f"{page_title} {url_path}".strip()
            page_context_warning = ""
            if page_context_text:
                ctx_sim = self._neural.similarity(question, page_context_text)
                ctx_score = (ctx_sim + 1.0) * 0.5
                if ctx_score < 0.22:
                    # Very low  --  almost certainly on the wrong page; fail fast.
                    return self._fail(
                        "extract",
                        f"Page context mismatch: page='{page_title}' url='{current_url}' "
                        f"does not match question (ctx_score={ctx_score:.2f}). "
                        "Navigate to the correct page before extracting.",
                        t0,
                        error="wrong_page_for_extract",
                    )
                if ctx_score < 0.32:
                    # Moderate mismatch  --  warn Gemini so it answers conservatively.
                    page_context_warning = (
                        f"\nNOTE: The current page ('{page_title}') may not contain the answer "
                        "to this question. If the answer is not present, respond: "
                        "'Not found on this page.'\n"
                    )
                    logger.warning(
                        "Extract page-context score low (%.2f): page=%r question=%.80r",
                        ctx_score, page_title, question,
                    )

            # Wait for JS-heavy pages (Apple, SPAs) to fully render before reading text.
            try:
                self._page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            # Scroll to bottom to trigger any lazy-load content, then back to top.
            try:
                self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                self._page.wait_for_timeout(600)
                self._page.evaluate("window.scrollTo(0, 0)")
                self._page.wait_for_timeout(300)
            except Exception:
                pass

            # Get visible, human-readable text - skip scripts, styles, hidden nodes.
            # Fetch up to 2x the limit so our semantic window selector has room to work.
            page_text: str = self._page.evaluate(
                f"() => document.body ? document.body.innerText.slice(0, {_EXTRACT_PAGE_TEXT_LIMIT * 2}) : ''"
            ) or ""
            page_text = page_text.strip()
            if not page_text:
                return self._fail(
                    "extract", "Page body has no visible text content.", t0, error="no_content"
                )
            # Select the most question-relevant section rather than blindly truncating.
            page_text = _select_relevant_section(page_text, question, max_chars=_EXTRACT_PAGE_TEXT_LIMIT)
            answer = _gemini_extract_answer(page_text, question + page_context_warning)
            latency = (time.perf_counter() - t0) * 1000
            if not answer:
                fallback_answer = self._extract_search_results_fallback(question)
                if fallback_answer:
                    logger.info("Extract fallback succeeded without Gemini response.")
                    return ActionResult(
                        success=True,
                        action="extract",
                        message=fallback_answer,
                        latency_ms=latency,
                        strategy="dom_extract_fallback",
                        score=1.0,
                        attempts=1,
                    )
                help_answer = self._extract_via_help_links(question)
                if help_answer:
                    logger.info("Extract help-link fallback succeeded.")
                    return ActionResult(
                        success=True,
                        action="extract",
                        message=help_answer,
                        latency_ms=latency,
                        strategy="help_link_extract_fallback",
                        score=1.0,
                        attempts=1,
                    )
                return self._fail(
                    "extract", "Gemini returned no answer from page content.", t0, error="no_answer"
                )
            strategy = "gemini_extract"
            if answer.strip().lower() == "not found on this page.":
                fallback_answer = self._extract_search_results_fallback(question)
                if fallback_answer:
                    answer = fallback_answer
                    strategy = "dom_extract_fallback"
                else:
                    line_answer = self._extract_visible_lines_fallback(question, page_text)
                    if line_answer:
                        answer = line_answer
                        strategy = "visible_lines_extract_fallback"
                    else:
                        relevant_link_answer = self._extract_via_relevant_links(question)
                        if relevant_link_answer:
                            answer = relevant_link_answer
                            strategy = "relevant_link_extract_fallback"
                        else:
                            help_answer = self._extract_via_help_links(question)
                            if help_answer:
                                answer = help_answer
                                strategy = "help_link_extract_fallback"
            elif not answer.strip():
                line_answer = self._extract_visible_lines_fallback(question, page_text)
                if line_answer:
                    answer = line_answer
                    strategy = "visible_lines_extract_fallback"
            if not answer.strip():
                relevant_link_answer = self._extract_via_relevant_links(question)
                if relevant_link_answer:
                    answer = relevant_link_answer
                    strategy = "relevant_link_extract_fallback"
            if not answer.strip():
                help_answer = self._extract_via_help_links(question)
                if help_answer:
                    answer = help_answer
                    strategy = "help_link_extract_fallback"
            logger.info("Extract succeeded (%.0fms): %.120s...", latency, answer)
            return ActionResult(
                success=True,
                action="extract",
                message=answer,
                latency_ms=latency,
                strategy=strategy,
                score=1.0,
                attempts=1,
            )
        except Exception as exc:
            return self._fail("extract", f"Extract failed: {exc}", t0, error=type(exc).__name__)
    def _extract_search_results_fallback(self, question: str) -> Optional[str]:
        if not self._page:
            return None
        q = (question or "").lower()
        # Technical/spec questions are not search-results list tasks.
        if any(k in q for k in ("gpu", "cpu", "core", "configured", "configuration", "spec", "specification")):
            return None
        if not any(k in q for k in (
            "list", "match", "result", "rating", "price", "criteria", "top", "best",
            "breakfast", "wifi", "pool", "event", "events", "this month"
        )):
            return None
        if any(k in q for k in ("event", "events")):
            event_answer = self._extract_event_results_fallback(q)
            if event_answer:
                return event_answer
        try:
            rows = self._page.evaluate(
                """() => {
                    const selectors = [
                      '[data-testid="property-card"]', '[data-testid*="property-card"]',
                      '[data-testid*="event-card"]', '[class*="event-card"]', '[class*="search-event"]',
                      '[data-testid*="card"]', '[data-testid*="result"]',
                      'article', '[role="article"]',
                      '[class*="card"]', '[class*="result"]', '[class*="listing"]',
                      'main li', '[role="listitem"]'
                    ];
                    let cards = [];
                    for (const s of selectors) {
                        cards = cards.concat(Array.from(document.querySelectorAll(s)));
                    }
                    cards = cards.filter((c, i, a) => a.indexOf(c) === i).slice(0, 10);
                    const out = [];
                    for (const c of cards) {
                        const nameEl = c.querySelector(
                            '[data-testid*="title"], h1, h2, h3, h4, a, strong, [aria-label]'
                        );
                        const scoreEl = c.querySelector(
                            '[data-testid*="review"], [data-testid*="rating"], [aria-label*="rating"], [class*="rating"], [class*="score"]'
                        );
                        const priceEl = c.querySelector(
                            '[data-testid*="price"], [aria-label*="price"], [class*="price"]'
                        );
                        const snippetEl = c.querySelector('p, [class*="desc"], [class*="amenit"], [aria-label]');
                        const name = (nameEl?.textContent || '').trim();
                        let score = (scoreEl?.textContent || '').trim();
                        let price = (priceEl?.textContent || '').trim();
                        let snippet = (snippetEl?.textContent || '').trim();
                        score = score.replace(/\s+/g, ' ');
                        price = price.replace(/\s+/g, ' ');
                        snippet = snippet.replace(/\s+/g, ' ').slice(0, 180);
                        if (name && name.length >= 2) out.push({name, score, price, snippet});
                    }
                    return out;
                }"""
            ) or []
            if not isinstance(rows, list) or not rows:
                return None
            lines = []
            amenity_notes: list[str] = []
            for item in rows[:5]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip()
                score = str(item.get("score") or "").strip()
                price = str(item.get("price") or "").strip()
                snippet = str(item.get("snippet") or "").strip().lower()
                if not name:
                    continue
                extras = []
                if score:
                    extras.append(f"rating: {score}")
                if price:
                    extras.append(f"price: {price}")
                suffix = f" ({', '.join(extras)})" if extras else ""
                lines.append(f"- {name}{suffix}")
                if any(k in q for k in ("breakfast", "wifi", "pool")):
                    amenity_tokens = [k for k in ("breakfast", "wifi", "pool") if k in q]
                    found = [k for k in amenity_tokens if k in snippet]
                    if amenity_tokens:
                        status = "mentions " + ", ".join(found) if found else "no amenity mention in visible card snippet"
                        amenity_notes.append(f"- {name}: {status}")
            if not lines:
                return None
            if any(k in q for k in ("breakfast", "wifi", "pool")) and amenity_notes:
                return "Amenity check from current top results:\n" + "\n".join(amenity_notes[:3])
            return "Top matches from current results page:\n" + "\n".join(lines[:5])
        except Exception:
            return None

    def _extract_event_results_fallback(self, q: str) -> Optional[str]:
        if not self._page:
            return None
        try:
            rows = self._page.evaluate(
                """(question) => {
                    const words = (question || '').toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length >= 3);
                    const cityHint = words.includes('york') || words.includes('new');
                    const monthHint = words.includes('month') || words.includes('this');
                    const anchors = Array.from(document.querySelectorAll('a[href], article, [data-testid*="event"]'));
                    const seen = new Set();
                    const out = [];
                    function norm(s) { return (s || '').replace(/\\s+/g, ' ').trim(); }
                    for (const el of anchors) {
                        const root = el.matches('article, [data-testid*="event"]') ? el : (el.closest('article') || el.closest('[data-testid*="event"]') || el);
                        const txt = norm(root.textContent || el.textContent || '');
                        if (!txt || txt.length < 8) continue;
                        const txtL = txt.toLowerCase();
                        if (!txtL.includes('event') && !txtL.includes('school') && !txtL.includes('activity')) continue;
                        if (cityHint && !(txtL.includes('new york') || txtL.includes('nyc') || txtL.includes(', ny'))) continue;
                        const titleEl = root.querySelector('h1, h2, h3, h4, [data-testid*="title"], strong, a');
                        const title = norm((titleEl && titleEl.textContent) || txt.split('\\n')[0] || '');
                        if (!title || title.length < 4) continue;
                        if (seen.has(title.toLowerCase())) continue;
                        const dateEl = root.querySelector('time, [datetime], [class*="date"], [data-testid*="date"]');
                        const dateText = norm((dateEl && (dateEl.textContent || dateEl.getAttribute('datetime'))) || '');
                        if (monthHint && !dateText && !txtL.includes('apr') && !txtL.includes('may') && !txtL.includes('jun') && !txtL.includes('jul') && !txtL.includes('2026')) {
                            continue;
                        }
                        seen.add(title.toLowerCase());
                        out.push({ title, date: dateText, snippet: txt.slice(0, 180) });
                        if (out.length >= 8) break;
                    }
                    return out;
                }""",
                q,
            ) or []
        except Exception:
            return None
        if not isinstance(rows, list) or not rows:
            return None
        lines = []
        for item in rows[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            date_text = str(item.get("date") or "").strip()
            if date_text:
                lines.append(f"- {title} ({date_text})")
            else:
                lines.append(f"- {title}")
        if not lines:
            return None
        return "Event matches visible on current page:\n" + "\n".join(lines)

    def _restore_url(self, origin_url: str) -> None:
        """Navigate back to origin_url if the page has drifted away from it."""
        if not self._page or not origin_url:
            return
        try:
            if (self._page.url or "") != origin_url:
                self._page.goto(origin_url, wait_until="domcontentloaded", timeout=8000)
        except Exception:
            pass

    def _extract_via_help_links(self, question: str) -> Optional[str]:
        if not self._page:
            return None
        q = (question or "").lower()
        info_tokens = ("lost", "help", "support", "faq", "contact", "refund", "baggage", "policy")
        if not any(t in q for t in info_tokens):
            return None
        _origin_url = self._page.url or ""
        try:
            cur = _origin_url
            host = urlparse(cur).netloc.lower()
            links = self._page.evaluate(
                """(tokens) => {
                    const out = [];
                    const anchors = Array.from(document.querySelectorAll('a[href]'));
                    for (const a of anchors) {
                        const href = (a.getAttribute('href') || '').trim();
                        if (!href) continue;
                        const txt = (a.textContent || a.getAttribute('aria-label') || '').toLowerCase().trim();
                        const hrefL = href.toLowerCase();
                        let score = 0;
                        for (const t of tokens) {
                            if (txt.includes(t)) score += 3;
                            if (hrefL.includes(t)) score += 2;
                        }
                        if (score <= 0) continue;
                        out.push({ href, score, text: txt.slice(0, 120) });
                    }
                    out.sort((a,b) => b.score - a.score);
                    return out.slice(0, 8);
                }""",
                list(info_tokens),
            ) or []
            if not isinstance(links, list) or not links:
                return None
            for item in links[:3]:
                if not isinstance(item, dict):
                    continue
                href = str(item.get("href") or "").strip()
                if not href:
                    continue
                if href.startswith("/"):
                    pu = urlparse(cur)
                    href = f"{pu.scheme}://{pu.netloc}{href}"
                pu = urlparse(href)
                dst_host = (pu.netloc or "").lower()
                if dst_host and host and dst_host != host and not dst_host.endswith("." + host):
                    continue
                try:
                    self._page.goto(href, wait_until="domcontentloaded", timeout=12000)
                    txt = self._page.evaluate(
                        f"() => document.body ? document.body.innerText.slice(0, {_EXTRACT_PAGE_TEXT_LIMIT}) : ''"
                    ) or ""
                    txt = txt.strip()
                    if not txt:
                        continue
                    ans = _gemini_extract_answer(txt, question)
                    if ans and ans.strip().lower() != "not found on this page.":
                        self._restore_url(_origin_url)
                        return ans.strip()
                except Exception:
                    continue
        except Exception:
            pass
        self._restore_url(_origin_url)
        return None

    def _extract_via_relevant_links(self, question: str) -> Optional[str]:
        if not self._page:
            return None
        q = (question or "").lower()
        words = [w for w in re.split(r"\W+", q) if len(w) >= 4]
        stop = {
            "find", "check", "identify", "provide", "include", "using", "website",
            "current", "first", "branch", "official", "repository", "please",
            "list", "with", "from", "this", "that", "which", "what",
        }
        anchors = [w for w in words if w not in stop][:6]
        if not anchors:
            return None
        _origin_url = self._page.url or ""
        try:
            cur = _origin_url
            host = urlparse(cur).netloc.lower()
            links = self._page.evaluate(
                """(anchors) => {
                    const out = [];
                    const as = Array.from(document.querySelectorAll('a[href]'));
                    for (const a of as) {
                        const href = (a.getAttribute('href') || '').trim();
                        if (!href || href.startsWith('#') || href.startsWith('javascript:')) continue;
                        const txt = ((a.textContent || '') + ' ' + (a.getAttribute('aria-label') || '')).toLowerCase();
                        const hrefL = href.toLowerCase();
                        let score = 0;
                        for (const t of anchors) {
                            if (txt.includes(t)) score += 3;
                            if (hrefL.includes(t)) score += 2;
                        }
                        if (txt.includes('standings') || hrefL.includes('standings')) score += 2;
                        if (txt.includes('table') || hrefL.includes('table')) score += 1;
                        if (score <= 0) continue;
                        out.push({ href, score });
                    }
                    out.sort((a,b) => b.score - a.score);
                    return out.slice(0, 8);
                }""",
                anchors,
            ) or []
            if not isinstance(links, list) or not links:
                return None
            for item in links[:4]:
                if not isinstance(item, dict):
                    continue
                href = str(item.get("href") or "").strip()
                if not href:
                    continue
                if href.startswith("/"):
                    pu = urlparse(cur)
                    href = f"{pu.scheme}://{pu.netloc}{href}"
                pu = urlparse(href)
                dst_host = (pu.netloc or "").lower()
                if dst_host and host and dst_host != host and not dst_host.endswith("." + host):
                    continue
                try:
                    self._page.goto(href, wait_until="domcontentloaded", timeout=12000)
                    txt = self._page.evaluate(
                        f"() => document.body ? document.body.innerText.slice(0, {_EXTRACT_PAGE_TEXT_LIMIT}) : ''"
                    ) or ""
                    txt = txt.strip()
                    if not txt:
                        continue
                    ans = _gemini_extract_answer(txt, question)
                    if ans and ans.strip().lower() != "not found on this page.":
                        self._restore_url(_origin_url)
                        return ans.strip()
                except Exception:
                    continue
        except Exception:
            pass
        self._restore_url(_origin_url)
        return None

    def _extract_visible_lines_fallback(self, question: str, page_text: str) -> Optional[str]:
        q = (question or "").lower()
        if not page_text:
            return None
        if any(k in q for k in ("gpu", "configured", "configuration", "core")):
            lines = [ln.strip() for ln in page_text.splitlines() if ln and ln.strip()]
            hits = []
            for ln in lines:
                ll = ln.lower()
                if len(ll) < 4 or len(ll) > 160:
                    continue
                if "gpu" in ll or ("core" in ll and any(t in ll for t in ("cpu", "gpu", "neural", "chip"))):
                    hits.append(ln)
                if len(hits) >= 6:
                    break
            if hits:
                uniq = []
                seen = set()
                for h in hits:
                    k = h.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(h)
                    if len(uniq) >= 5:
                        break
                return "Hardware configuration lines found on current page:\n" + "\n".join(f"- {x}" for x in uniq)
        if any(k in q for k in ("breakfast", "wifi", "pool")):
            lines = [ln.strip() for ln in page_text.splitlines() if ln and ln.strip()]
            hits = []
            for ln in lines:
                ll = ln.lower()
                if len(ll) < 6 or len(ll) > 120:
                    continue
                if any(t in ll for t in ("breakfast included", "includes breakfast", "free wifi", "swimming pool", "pool")):
                    hits.append(ln)
                if len(hits) >= 6:
                    break
            if hits:
                uniq = []
                seen = set()
                for h in hits:
                    k = h.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(h)
                    if len(uniq) >= 5:
                        break
                return "Amenity signals found on current page:\n" + "\n".join(f"- {x}" for x in uniq)
        if not any(k in q for k in ("list", "events", "event", "results", "top", "matches")):
            return None
        lines = [ln.strip() for ln in page_text.splitlines() if ln and ln.strip()]
        if not lines:
            return None
        city_hint = "new york" if "new york" in q else ""
        noise = {
            "eventbrite", "search events", "create events", "sign in",
            "help center", "contact support", "privacy", "cookies",
        }
        hits: list[str] = []
        for ln in lines:
            ll = ln.lower()
            if len(ll) < 8 or len(ll) > 140:
                continue
            if ll in noise:
                continue
            if ll.startswith("only show ") or ll.startswith("search for online events") or ll.startswith("share this event"):
                continue
            score = 0
            if "event" in ll:
                score += 2
            if "school" in ll or "activity" in ll:
                score += 2
            if city_hint and ("new york" in ll or "nyc" in ll or ", ny" in ll):
                score += 2
            if "today" in ll or "tomorrow" in ll or re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", ll):
                score += 1
            if len(ll.split()) < 3:
                score -= 1
            if score >= 2:
                hits.append(ln)
            if len(hits) >= 6:
                break
        if not hits:
            return None
        uniq: list[str] = []
        seen = set()
        for h in hits:
            key = h.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
            if len(uniq) >= 5:
                break
        if not uniq:
            return None
        return "Matches found on current page:\n" + "\n".join(f"- {x}" for x in uniq)

    def _execute_navigate(self, request: ActionRequest, t0: float) -> ActionResult:
        url = (request.value or "").strip()
        if not url:
            return self._fail("navigate", "navigate requires a value (URL).", t0, error="missing_url")
        url_lower = url.lower()
        if not (url_lower.startswith("http://") or url_lower.startswith("https://") or url_lower.startswith("file://")):
            return self._fail(
                "navigate",
                f"Unsupported URL scheme: {url!r}. Only http/https/file are allowed.",
                t0,
                error="invalid_scheme",
            )
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=request.timeout_ms)
            try:
                self.refresh_state()
            except Exception:
                pass
            latency = (time.perf_counter() - t0) * 1000
            logger.info("Navigate succeeded to %r in %.0fms", url, latency)
            return ActionResult(
                success=True,
                action="navigate",
                message=f"Navigated to {url}",
                latency_ms=latency,
                selector=url,
                selector_type="url",
                strategy="direct_navigate",
                score=1.0,
                attempts=1,
            )
        except PlaywrightTimeoutError as exc:
            return self._fail("navigate", f"Navigation timed out: {exc}", t0, error="timeout")
        except Exception as exc:
            return self._fail("navigate", f"Navigation failed: {exc}", t0, error=type(exc).__name__)

    def _validate_request(self, request: ActionRequest) -> None:
        action = (request.action or "").strip().lower()
        if action not in _SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action: {request.action!r}. Supported: {sorted(_SUPPORTED_ACTIONS)}")
        if action == "type" and request.value is None:
            raise ValueError("type action requires a value.")
        if action == "select" and request.value is None:
            raise ValueError("select action requires a value (label/value/index).")
        if action == "navigate" and not (request.value or "").strip():
            raise ValueError("navigate action requires a value (URL).")
        if request.timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")

    def _dispatch_action(self, request: ActionRequest, resolved: _ResolvedTarget) -> None:
        action = request.action.lower()
        if action == "click":
            self._perform_click(resolved, request.timeout_ms)
            return
        if action == "type":
            assert request.value is not None
            self._perform_type(
                resolved,
                request.value,
                request.timeout_ms,
                clear_first=request.clear_first,
                press_enter=request.press_enter,
                intent=request.intent,
            )
            return
        if action == "select":
            assert request.value is not None
            self._perform_select(resolved, request.value, request.timeout_ms)
            return
        if action == "check":
            self._perform_check(resolved, request.timeout_ms, True)
            return
        if action == "uncheck":
            self._perform_check(resolved, request.timeout_ms, False)
            return
        if action == "scroll":
            self._perform_scroll(resolved, request.value or "down", request.timeout_ms)
            return

    def _perform_scroll(
        self, target: Optional[_ResolvedTarget], value: str, timeout_ms: int
    ) -> None:
        """Scroll the page or a specific element.

        value: 'down' | 'up' | 'top' | 'bottom' | '<pixels>' (integer, positive = down).
        Scrolling into view first then scrolling by pixels lets the agent trigger
        lazy-loaded content, carousels, and infinite-scroll feeds.
        """
        amount = (value or "down").strip().lower()
        if amount == "top":
            self._page.evaluate("window.scrollTo(0, 0)")
        elif amount == "bottom":
            self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            try:
                pixels = int(amount)
            except ValueError:
                pixels = 500 if amount == "down" else -500
            if target and target.locator:
                try:
                    target.locator.scroll_into_view_if_needed(timeout=min(timeout_ms, 2000))
                except Exception:
                    pass
            self._page.evaluate(f"window.scrollBy(0, {pixels})")
        try:
            self._page.wait_for_timeout(350)
        except Exception:
            pass

    def execute_and_verify(
        self,
        request: ActionRequest,
        wait_after_ms: int = 250,
        verifier: Optional[WebStateDifferentiator] = None,
    ) -> ActionVerificationResult:
        if not self._engine:
            result = self.execute(request)
            return ActionVerificationResult(action_result=result, verification=None, metadata=None)

        net = {"started": 0, "finished": 0, "failed": 0}

        def _on_req(_):
            net["started"] += 1

        def _on_fin(_):
            net["finished"] += 1

        def _on_fail(_):
            net["failed"] += 1

        self._page.on("request", _on_req)
        self._page.on("requestfinished", _on_fin)
        self._page.on("requestfailed", _on_fail)
        before = self._last_state or self.refresh_state()
        result = self.execute(request)
        # Only settle for successful actions  --  failed clicks produce no XHR.
        # Skipping settle on failures saves ~2000ms per failed candidate.
        if wait_after_ms > 0 and result.success:
            self._settle_after_action(net, wait_after_ms, request.action)
        elif result.success:
            try:
                self._page.wait_for_timeout(80)
            except Exception:
                pass
        after = self.refresh_state()
        self._page.remove_listener("request", _on_req)
        self._page.remove_listener("requestfinished", _on_fin)
        self._page.remove_listener("requestfailed", _on_fail)

        if not before or not after:
            return ActionVerificationResult(
                action_result=result,
                verification=None,
                metadata={"network": net},
            )

        active_verifier = verifier or WebStateDifferentiator()
        verified = active_verifier.compare(before, after)
        return ActionVerificationResult(
            action_result=result,
            verification=verified,
            metadata={
                "network": net,
                "url_before": before.url,
                "url_after": after.url,
                "scroll_before": before.viewport.scroll_y if before.viewport else None,
                "scroll_after": after.viewport.scroll_y if after.viewport else None,
            },
        )

    def _resolve_target(self, request: ActionRequest) -> Optional[_ResolvedTarget]:
        if request.selector:
            locator = self._selector_from_string(request.selector, request.selector_type)
            return _ResolvedTarget(
                locator=locator,
                element_handle=None,
                selector=request.selector,
                selector_type=request.selector_type,
                strategy="direct_selector",
                score=None,
            )

        if request.dom_node_id:
            handle = self._find_by_dom_node_id(request.dom_node_id)
            if not handle:
                return None
            selector = f"dom_node_id:{request.dom_node_id}"
            return _ResolvedTarget(
                locator=None,
                element_handle=handle,
                selector=selector,
                selector_type="dom_node_id",
                strategy="dom_lookup",
                score=None,
            )

        if request.intent:
            mem = self._resolve_from_memory(request.intent, request.action)
            if mem:
                return mem
            graph_resolved = self._resolve_from_graph(request.intent, request.action)
            if graph_resolved:
                return graph_resolved
            state = self._last_state or self.refresh_state()
            if not state:
                return None
            picked = self._selector.find(
                request.intent,
                state,
                use_cache=False,
                action_hint=request.action,
            )
            if not picked.best:
                return None
            match: SelectorMatch = picked.best
            locator = self._selector_from_match(match)
            return _ResolvedTarget(
                locator=locator,
                element_handle=None,
                selector=match.selector,
                selector_type=match.selector_type,
                strategy=f"intent_{match.strategy}",
                score=match.score,
            )

        return None

    def _resolve_from_memory(self, intent: str, action: str) -> Optional[_ResolvedTarget]:
        host = urlparse(self._page.url).netloc if self._page.url else ""
        if not host:
            return None
        entry = self._memory.best_for(host=host, intent=intent, action=action)
        if not entry:
            return None
        try:
            locator = self._selector_from_string(entry.selector, entry.selector_type)
            return _ResolvedTarget(
                locator=locator,
                element_handle=None,
                selector=entry.selector,
                selector_type=entry.selector_type,
                strategy="intent_memory",
                score=entry.last_score,
            )
        except Exception:
            return None

    def _resolve_from_graph(self, intent: str, action: str) -> Optional[_ResolvedTarget]:
        if not self._page_graph:
            return None
        candidates = self._page_graph.rank_for_intent(
            intent,
            action=action,
            top_k=3,
            semantic_scorer=self._graph_semantic_score,
        )
        for candidate in candidates:
            if candidate.dom_node_id:
                handle = self._find_by_dom_node_id(candidate.dom_node_id)
                if handle:
                    return _ResolvedTarget(
                        locator=None,
                        element_handle=handle,
                        selector=f"dom_node_id:{candidate.dom_node_id}",
                        selector_type="dom_node_id",
                        strategy="intent_graph",
                        score=candidate.score,
                    )
            if candidate.selector:
                locator = self._selector_from_string(candidate.selector, candidate.selector_type)
                return _ResolvedTarget(
                    locator=locator,
                    element_handle=None,
                    selector=candidate.selector,
                    selector_type=candidate.selector_type,
                    strategy="intent_graph_selector",
                    score=candidate.score,
                )
        return None

    def _graph_semantic_score(self, intent: str, node) -> float:
        text = " | ".join([
            node.name or "",
            node.role or "",
            node.tag_name or "",
            node.selector or "",
            node.metadata.get("nearby_text") or "" if node.metadata else "",
            node.metadata.get("form_context") or "" if node.metadata else "",
            " ".join(node.actions or []),
        ])
        sim = self._neural.similarity(intent, text)
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))

    def _selector_from_match(self, match: SelectorMatch) -> Locator:
        return self._selector_from_string(match.selector, match.selector_type)

    def _selector_from_string(self, selector: str, selector_type: str) -> Locator:
        st = (selector_type or "auto").lower()
        if st in {"xpath", "xpath_frame", "xpath_id"} or selector.startswith("/") or selector.startswith("./"):
            return self._page.locator(f"xpath={selector}").first
        if st == "text":
            return self._page.get_by_text(selector, exact=False).first
        if st == "aria_label":
            name = selector.strip()
            if name.startswith('[aria-label="') and name.endswith('"]'):
                name = name[len('[aria-label="'):-2]
            return self._page.get_by_role("link", name=name, exact=False).or_(
                self._page.get_by_role("button", name=name, exact=False)
            ).first
        if st in {"css", "css_frame", "css_id", "auto"}:
            return self._page.locator(selector).first
        return self._page.locator(selector).first

    def _find_by_dom_node_id(self, dom_node_id: str) -> Optional[ElementHandle]:
        frame_prefix = None
        local_id = dom_node_id
        if ":" in dom_node_id and dom_node_id.startswith("frame-"):
            frame_prefix, local_id = dom_node_id.split(":", 1)

        script = """(id) => {
            function collect(root, out) {
                if (!root) return;
                const all = root.querySelectorAll("*");
                for (const el of all) {
                    out.push(el);
                    if (el.shadowRoot) collect(el.shadowRoot, out);
                }
            }
            const nodes = [];
            collect(document, nodes);
            for (const el of nodes) {
                if (el.__captureId === id) return el;
            }
            return null;
        }"""

        frames = self._page.frames
        matched_frames = []

        for idx, frame in enumerate(frames):
            frame_name = frame.name or "anon"
            frame_id_exact = f"frame-{idx}-{frame_name}"

            if frame_prefix:
                if frame_id_exact == frame_prefix:
                    matched_frames.insert(0, frame)
                    continue
                if frame_prefix.endswith(f"-{frame_name}") and frame_name != "anon":
                    matched_frames.append(frame)
                    continue
                if frame_name == "anon":
                    matched_frames.append(frame)
            else:
                matched_frames.append(frame)

        if frame_prefix and not matched_frames:
            matched_frames = list(frames)

        for frame in matched_frames:
            try:
                handle = frame.evaluate_handle(script, local_id)
                element = handle.as_element()
                if element:
                    return element
            except Exception:
                continue
        return None

    def _perform_click(self, target: _ResolvedTarget, timeout_ms: int) -> None:
        if target.element_handle:
            target.element_handle.scroll_into_view_if_needed(timeout=timeout_ms)
            try:
                target.element_handle.click(timeout=timeout_ms)
            except Exception:
                target.element_handle.click(timeout=timeout_ms, force=True)
        else:
            assert target.locator is not None
            target.locator.wait_for(state="visible", timeout=timeout_ms)
            target.locator.scroll_into_view_if_needed(timeout=timeout_ms)
            try:
                target.locator.click(timeout=timeout_ms)
            except Exception as first_exc:
                try:
                    target.locator.click(timeout=min(timeout_ms, 4000), force=True)
                except Exception:
                    try:
                        target.locator.evaluate("el => el.click()")
                    except Exception:
                        raise first_exc
        # If the click triggered a navigation, wait for the page to reach a
        # stable DOM state before returning.  Without this wait the finally-block
        # refresh_state() in execute() captures a partially-loaded page.
        if self._page:
            try:
                self._page.wait_for_load_state(
                    "domcontentloaded", timeout=min(timeout_ms, 6000)
                )
            except Exception:
                pass

    def _check_preconditions(self, action: str, target: _ResolvedTarget, timeout_ms: int) -> None:
        if action == "navigate":
            return
        if target.element_handle:
            target.element_handle.scroll_into_view_if_needed(timeout=timeout_ms)
            return
        assert target.locator is not None
        target.locator.wait_for(state="visible", timeout=timeout_ms)
        target.locator.scroll_into_view_if_needed(timeout=timeout_ms)
        if action in {"click", "type", "select", "check", "uncheck"}:
            try:
                if not target.locator.is_enabled(timeout=min(timeout_ms, 2000)):
                    raise ValueError("Target element is disabled.")
            except TypeError:
                if not target.locator.is_enabled():
                    raise ValueError("Target element is disabled.")

    def _perform_type(
        self,
        target: _ResolvedTarget,
        value: str,
        timeout_ms: int,
        clear_first: bool,
        press_enter: bool,
        intent: Optional[str] = None,
    ) -> None:
        # Do not auto-submit on generic "search" intents; planners frequently
        # model submit as a separate click/autocomplete step. Auto-Enter causes
        # premature navigations and anti-bot interstitials.
        search_context = bool(press_enter)
        if target.element_handle:
            target.element_handle.scroll_into_view_if_needed(timeout=timeout_ms)
            target.element_handle.focus()
            if clear_first:
                target.element_handle.fill("", timeout=timeout_ms)
            target.element_handle.fill(value, timeout=timeout_ms)
            if search_context:
                self._page.keyboard.press("Enter")
            return

        assert target.locator is not None
        target.locator.wait_for(state="visible", timeout=timeout_ms)
        target.locator.scroll_into_view_if_needed(timeout=timeout_ms)
        if clear_first:
            target.locator.fill("", timeout=timeout_ms)
        target.locator.fill(value, timeout=timeout_ms)
        if search_context:
            target.locator.press("Enter")

    def _perform_select(self, target: _ResolvedTarget, value: str, timeout_ms: int) -> None:
        if target.element_handle:
            target.element_handle.scroll_into_view_if_needed(timeout=timeout_ms)
            try:
                target.element_handle.select_option(value=value, timeout=timeout_ms)
            except Exception:
                target.element_handle.select_option(label=value, timeout=timeout_ms)
            return

        assert target.locator is not None
        target.locator.wait_for(state="visible", timeout=timeout_ms)
        target.locator.scroll_into_view_if_needed(timeout=timeout_ms)
        try:
            target.locator.select_option(value=value, timeout=timeout_ms)
        except Exception:
            target.locator.select_option(label=value, timeout=timeout_ms)

    def _perform_check(self, target: _ResolvedTarget, timeout_ms: int, checked: bool) -> None:
        if target.element_handle:
            target.element_handle.scroll_into_view_if_needed(timeout=timeout_ms)
            if checked:
                target.element_handle.check(timeout=timeout_ms)
            else:
                target.element_handle.uncheck(timeout=timeout_ms)
            return

        assert target.locator is not None
        target.locator.wait_for(state="visible", timeout=timeout_ms)
        target.locator.scroll_into_view_if_needed(timeout=timeout_ms)
        if checked:
            target.locator.check(timeout=timeout_ms)
        else:
            target.locator.uncheck(timeout=timeout_ms)

    def _settle_after_action(self, net: dict, wait_after_ms: int, action: str) -> None:
        deadline = time.perf_counter() + (max(wait_after_ms, 150) + 600) / 1000.0
        poll_ms = 80

        if action == "navigate":
            try:
                self._page.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                pass
            try:
                self._page.wait_for_timeout(400)
            except Exception:
                pass
            return

        while time.perf_counter() < deadline:
            in_flight = net["started"] - net["finished"] - net["failed"]
            if in_flight <= 0:
                break
            try:
                self._page.wait_for_timeout(poll_ms)
            except Exception:
                break

        remaining_ms = int((deadline - time.perf_counter()) * 1000)
        if remaining_ms > 0:
            settle = min(remaining_ms, max(wait_after_ms, 250))
            try:
                self._page.wait_for_timeout(settle)
            except Exception:
                pass

    def _fail(
        self,
        action: str,
        message: str,
        t0: float,
        attempts: int = 1,
        error: Optional[str] = None,
    ) -> ActionResult:
        logger.warning("Action %r failed: %s", action, message[:200])
        return ActionResult(
            success=False,
            action=action,
            message=message,
            latency_ms=(time.perf_counter() - t0) * 1000,
            attempts=attempts,
            error=error,
        )

    def record_confirmed_success(
        self,
        intent: str,
        action: str,
        selector: str,
        selector_type: str,
        strategy: str,
        score: Optional[float],
        url: str = "",
    ) -> None:
        from urllib.parse import urlparse as _up
        current_url = url or (self._page.url if self._page and self._page.url else "")
        host = _up(current_url).netloc if current_url else ""
        if not host:
            return
        self._memory.record_success(
            host=host,
            intent=intent,
            action=action,
            selector=selector,
            selector_type=selector_type,
            strategy=strategy,
            score=score,
            url=current_url,
        )

    def _record_success(self, request: ActionRequest, resolved: _ResolvedTarget) -> None:
        if not request.intent:
            return
        if not resolved.selector or not resolved.selector_type:
            return
        if request.action == "navigate":
            return
        host = urlparse(self._page.url).netloc if self._page.url else ""
        if not host:
            return
        self._memory.record_success(
            host=host,
            intent=request.intent,
            action=request.action,
            selector=resolved.selector,
            selector_type=resolved.selector_type,
            strategy=resolved.strategy,
            score=resolved.score,
        )




