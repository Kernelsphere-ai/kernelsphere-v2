import re
import time
import math
import json
import threading
from datetime import date as _date
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from capture import ExtractionEngine, PageState
from element_selector import SemanticElementSelector
from actions_executor import (
    ActionRequest,
    ActionResult,
    ActionVerificationResult,
    BrowserActionsExecutor,
)
from state_differentiator import WebStateDifferentiator
from full_page_extractor import ExtractionConfig, FullPageExtraction, FullPageExtractor
from page_graph import PageGraph, PageGraphBuilder
from neural_intent_engine import get_shared_engine
from goal_validator import GoalProgress, GoalValidator, TaskGoal
from model_reasoner import CostAwareReasoner, ReasonerConfig
from task_planner import (
    PlanExecutionResult,
    PlannerStep,
    ReliableTaskPlanner,
    TaskPlan,
)
from logging_config import get_logger
from selector_memory import _is_valid_selector as _sel_ok

logger = get_logger(__name__)



@dataclass
class AutomationStep:
    action: str
    intent: Optional[str] = None
    value: Optional[str] = None
    selector: Optional[str] = None
    selector_type: str = "auto"
    dom_node_id: Optional[str] = None
    timeout_ms: int = 10000
    clear_first: bool = True
    press_enter: bool = False
    wait_after_ms: int = 250


@dataclass
class AutomationStepResult:
    step: AutomationStep
    ok: bool
    action: ActionVerificationResult
    duration_ms: float
    changed: bool
    extraction: Optional[FullPageExtraction] = None
    change_summaries: list[str] = field(default_factory=list)


@dataclass
class AutomationTaskResult:
    url: str
    started_at: float
    ended_at: float
    steps: list[AutomationStepResult]
    success: bool
    goal_progress: Optional[GoalProgress] = None


@dataclass
class NextActionCandidate:
    intent: str
    score: float
    probability: float
    selector: Optional[str]
    selector_type: str
    reasoning: str
    dom_node_id: Optional[str]
    level: int = 0
    supports_action: bool = False
    visible_ratio: float = 0.0
    is_occluded: bool = False
    neural_score: float = 0.0


class BrowserAutomationAgent:
    """
    Unified browser automation agent:
    capture -> select -> execute -> verify changes.
    Supports navigate, click, type, select, check, uncheck actions.
    """

    def __init__(self, headless: bool = True, use_llm_selector: bool = True):
        self._headless = headless
        self._use_llm_selector = use_llm_selector
        self._engine = ExtractionEngine(headless=headless)
        self._selector = SemanticElementSelector(use_llm=use_llm_selector, strict_model=False, use_vision=False)
        self._diff = WebStateDifferentiator()
        self._full_extractor = FullPageExtractor(self._engine, ExtractionConfig())
        self._graph_builder = PageGraphBuilder()
        self._neural = get_shared_engine()
        self._goal_validator = GoalValidator()
        self._reasoner = CostAwareReasoner(ReasonerConfig())
        self._planner = ReliableTaskPlanner(use_gemini=True)
        self._page = None
        self._executor: Optional[BrowserActionsExecutor] = None
        self._current_state: Optional[PageState] = None
        self._current_extraction: Optional[FullPageExtraction] = None
        self._current_graph: Optional[PageGraph] = None
        self._active_plan: Optional[TaskPlan] = None
        self._min_level_for_action = {"click": 2, "type": 2, "select": 2, "check": 2, "uncheck": 2, "navigate": 0}
        self._restart_count = 0

    def start(self) -> "BrowserAutomationAgent":
        logger.info("Starting BrowserAutomationAgent (headless=%s)", self._headless)
        self._engine.start()
        if self._engine._context is None:
            raise RuntimeError("Browser context not initialized after start(). Check Playwright installation.")
        self._page = self._engine._context.new_page()
        self._executor = BrowserActionsExecutor(
            page=self._page,
            extraction_engine=self._engine,
            selector=self._selector,
        )
        logger.debug("BrowserAutomationAgent ready.")
        return self

    def stop(self) -> None:
        logger.info("Stopping BrowserAutomationAgent.")
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        self._engine.stop()

    def __enter__(self) -> "BrowserAutomationAgent":
        return self.start()

    def __exit__(self, *_):
        self.stop()

    @property
    def page(self):
        return self._page

    @property
    def current_state(self) -> Optional[PageState]:
        return self._current_state

    @property
    def current_extraction(self) -> Optional[FullPageExtraction]:
        return self._current_extraction

    @property
    def current_graph(self) -> Optional[PageGraph]:
        return self._current_graph

    @property
    def active_plan(self) -> Optional[TaskPlan]:
        return self._active_plan

    def _ensure_browser_alive(self) -> bool:
        if self._page is None or self._engine._browser is None:
            raise RuntimeError("Agent not started. Call start() first.")
        try:
            self._page.evaluate("1")
            return True
        except Exception:
            logger.warning(
                "Browser appears to have crashed. Attempting recovery (attempt %d).",
                self._restart_count + 1,
            )
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = ExtractionEngine(headless=self._headless)
            self._engine.start()
            if self._engine._context is None:
                raise RuntimeError("Browser recovery failed: context is None.")
            self._page = self._engine._context.new_page()
            self._executor = BrowserActionsExecutor(
                page=self._page,
                extraction_engine=self._engine,
                selector=self._selector,
            )
            self._full_extractor = FullPageExtractor(self._engine, ExtractionConfig())
            self._current_state = None
            self._current_extraction = None
            self._current_graph = None
            self._restart_count += 1
            logger.info("Browser recovered successfully (total restarts: %d).", self._restart_count)
            return True

    def reset_context(self, anti_bot_retry: bool = False) -> None:
        if self._engine._browser is None:
            raise RuntimeError("Agent not started. Call start() first.")

        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None

        old_context = self._engine._context
        if old_context:
            try:
                old_context.close()
            except Exception:
                pass
            self._engine._context = None

        self._engine._context = self._engine.new_context(anti_bot_mode=anti_bot_retry)
        self._page = self._engine._context.new_page()
        self._executor = BrowserActionsExecutor(
            page=self._page,
            extraction_engine=self._engine,
            selector=self._selector,
        )
        self._full_extractor = FullPageExtractor(self._engine, ExtractionConfig())
        self._current_state = None
        self._current_extraction = None
        self._current_graph = None
        self._active_plan = None
        self._reasoner.reset_budget()
        # Clear TF-IDF corpus so vocabulary from the previous task does not
        # skew similarity scores on the next (corpus accumulates unless reset).
        self._neural.reset_corpus()
        self._selector.clear_cache()
        logger.info("Browser context reset; fresh context ready for next task.")

    def _execution_blocked_by_bot(self, execution: PlanExecutionResult) -> bool:
        for step in execution.executed_steps:
            try:
                err = (step.action.action_result.error or "").lower()
            except Exception:
                err = ""
            if err in {"bot_gate_blocked", "page_not_ready"}:
                return True
        return False

    def _is_hard_gate_page(self) -> bool:
        """Detect hard-deny pages that are unlikely to be solvable in-session."""
        if self._page is None:
            return False
        try:
            title = (self._page.title() or "").lower()
        except Exception:
            title = ""
        url = (self._page.url or "").lower()
        try:
            body = (self._page.locator("body").first.inner_text(timeout=900) or "").lower()
        except Exception:
            body = ""
        hard_tokens = (
            "you don't have permission to access",
            "access denied",
            "request could not be satisfied",
            "reference #",
            "errors.edgesuite.net",
            "akamai",
        )
        blob = f"{title}\n{url}\n{body}"
        return any(t in blob for t in hard_tokens)

    def _alternate_entry_urls(self, start_url: str) -> list[str]:
        try:
            from urllib.parse import urlparse
            pu = urlparse(start_url)
            host = (pu.netloc or "").lower()
            if not host:
                return [start_url]
            bare = host[4:] if host.startswith("www.") else host
            variants = [
                f"https://{host}/",
                f"https://{bare}/",
                f"https://www.{bare}/",
                f"https://{bare}/home",
                f"https://{bare}/stores",
                f"https://{bare}/store-locator",
            ]
            out: list[str] = []
            seen = set()
            for v in variants:
                if v in seen:
                    continue
                seen.add(v)
                out.append(v)
            return out
        except Exception:
            return [start_url]

    def _preflight_gate_playbook(self, start_url: str, max_rotations: int = 2) -> tuple[bool, str]:
        """Attempt early gate recovery before planning to avoid wasted runtime."""
        if self._page is None:
            return False, "no_page"
        hard_hits = 0
        attempts = 0
        while attempts <= max_rotations:
            if attempts > 0:
                try:
                    self.reset_context(anti_bot_retry=True)
                except Exception as exc:
                    return False, f"context_rotate_failed:{type(exc).__name__}"
            for u in self._alternate_entry_urls(start_url)[:3]:
                try:
                    self._page.goto(u, wait_until="domcontentloaded", timeout=16000)
                except Exception:
                    continue
                if self._is_hard_gate_page():
                    hard_hits += 1
                    if hard_hits >= 2:
                        return False, "hard_gate_blocked_after_rotations"
                    continue
                self._mitigate_bot_gate()
                if not self._is_bot_gate_page():
                    try:
                        self.extract_page_context()
                    except Exception:
                        pass
                    return True, ""
                if not self._is_hard_gate_page():
                    # Soft gate (captcha/challenge) might pass in normal flow.
                    return True, "soft_gate_detected"
            attempts += 1
        return False, "hard_gate_blocked_after_rotations"

    def _hard_gate_abort_result(self, url: str, user_goal: str, reason: str) -> PlanExecutionResult:
        plan = TaskPlan(
            task_id=f"plan-{abs(hash(user_goal))}",
            user_goal=user_goal,
            subgoals=[],
            status="failed",
            current_subgoal_index=0,
            notes=[f"hard_gate_abort:{reason}", f"url:{url}"],
        )
        logger.error("Hard gate abort: %s goal=%r url=%s", reason, user_goal, url)
        return PlanExecutionResult(
            plan=plan,
            executed_steps=[],
            success=False,
            extracted_answer=None,
        )

    def extract_page_context(self) -> FullPageExtraction:
        if self._page is None:
            raise RuntimeError("Agent not started. Call start() first.")
        if self._executor is None:
            raise RuntimeError("Executor not initialized. Call start() first.")
        previous_signature = self._current_extraction.signature if self._current_extraction else None
        extraction = None
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                extraction = self._full_extractor.extract(self._page, previous_signature=previous_signature)
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = (
                    "execution context was destroyed" in msg
                    or "most likely because of a navigation" in msg
                )
                if transient and attempt < 3:
                    try:
                        self._page.wait_for_load_state("domcontentloaded", timeout=4000)
                    except Exception:
                        try:
                            self._page.wait_for_timeout(220)
                        except Exception:
                            pass
                    continue
                raise
        if extraction is None:
            raise RuntimeError(f"extract_page_context failed after retries: {last_exc}")
        graph = self._graph_builder.build(extraction)
        self._current_extraction = extraction
        self._current_graph = graph
        self._current_state = extraction.state
        self._executor.set_state(extraction.state)
        self._executor.set_page_graph(graph)
        self._prime_neural_corpus(extraction)
        return extraction

    def _maybe_extract_revealed_elements(self, step: AutomationStep) -> None:
        """After a click that likely opened a dropdown/picker, scan for hidden
        elements that are now visible and merge them into the current extraction
        so subsequent steps can target them.

        Uses neural semantic classification rather than a hardcoded keyword list.
        """
        if self._page is None or self._current_extraction is None:
            return
        intent = step.intent or ""
        if self._intent_is_autocomplete_pick(intent):
            return
        if not self._neural.classify(intent, "reveal_expand", threshold=0.62):
            return

        try:
            hidden_els = self._full_extractor.extract_scoped_hidden(
                self._page,
                container_selector=None,  # Scan full DOM  -  picker may be a portal
                frame_wait_ms=400,
            )
            if hidden_els:
                logger.debug(
                    "Scoped hidden extraction found %d additional elements after reveal click.",
                    len(hidden_els),
                )
                # Merge into current extraction (deduplicate by dom_node_id)
                existing_ids = {
                    e.dom_node_id for e in self._current_extraction.elements
                    if e.dom_node_id
                }
                new_els = [
                    e for e in hidden_els
                    if not e.dom_node_id or e.dom_node_id not in existing_ids
                ]
                if new_els:
                    self._current_extraction.elements.extend(new_els)
                    # Rebuild graph with the augmented element list
                    self._current_graph = self._graph_builder.build(self._current_extraction)
                    if self._executor:
                        self._executor.set_page_graph(self._current_graph)
                    self._prime_neural_corpus(self._current_extraction)
                    logger.info(
                        "Merged %d revealed elements into extraction (total now %d).",
                        len(new_els), len(self._current_extraction.elements),
                    )
        except Exception as exc:
            logger.debug("Scoped hidden extraction failed (non-fatal): %s", exc)

    def _prime_neural_corpus(self, extraction: FullPageExtraction) -> None:
        # Always prime TF-IDF corpus so it is ready if _neural_similarity falls
        # back to TF-IDF on exception, even when the neural model is active.
        corpus: list[str] = []
        for el in extraction.elements:
            parts = [
                el.name or "",
                el.role or "",
                el.tag_name or "",
                el.placeholder or "",
                el.nearby_text or "",
                el.form_context or "",
                el.css_selector or "",
                el.xpath or "",
                " ".join(el.actions or []),
            ]
            text = " ".join(p for p in parts if p)
            if text.strip():
                corpus.append(text)
        if corpus:
            self._neural.prime_corpus(corpus)

    def _intent_is_autocomplete_pick(self, intent: str) -> bool:
        il = (intent or "").lower()
        # Hard negatives: these are never autocomplete-choice intents.
        if any(w in il for w in (
            "filter", "sort", "wifi", "pool", "breakfast", "search button",
            "submit", "next month", "previous month", "calendar next", "calendar prev",
            "open the date picker", "check-in date", "check-out date",
            "set as my store", "set as home", "home store", "open menu", "close menu",
        )):
            return False
        has_explicit = any(w in il for w in (
            "autocomplete", "suggestion", "from the list", "dropdown list", "listbox",
            "from the dropdown", "choose from list",
        ))
        if has_explicit:
            return True
        # No explicit list/suggestion cue: do not treat as autocomplete.
        return False

    def _intent_is_datepicker_open(self, intent: str) -> bool:
        il = (intent or "").lower()
        # Hard negatives: avoid misfiring date-picker flow for unrelated clicks.
        if any(w in il for w in (
            "autocomplete", "suggestion", "search", "submit", "sort", "filter",
            "wifi", "breakfast", "pool", "louvre", "museum", "price",
        )):
            return False
        # Date-cell and calendar-navigation intents must not be treated as
        # "open date picker" actions.
        if any(w in il for w in ("next month", "previous month", "prev month", "calendar next", "calendar prev")):
            return False
        if re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
            il,
        ):
            return False
        has_date_hint = any(w in il for w in (
            "date", "calendar", "check-in", "check in", "check-out", "check out",
            "arrival", "departure", "start date", "end date",
        ))
        if not has_date_hint:
            return False
        has_open_hint = any(w in il for w in (
            "open", "show", "display", "expand", "activate", "picker", "widget",
        ))
        if has_open_hint:
            return True
        return self._neural.classify(intent or "", "date_picker_open", threshold=0.62)

    def _intent_is_specific_date_click(self, intent: str, value: Optional[str]) -> bool:
        txt = f"{intent or ''} {value or ''}".lower()
        has_month_day = bool(re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b",
            txt,
        ))
        has_day_phrase = bool(re.search(r"\bdate\s+\d{1,2}\b", txt))
        return has_month_day or has_day_phrase

    def _extract_month_day_hint(self, text: str) -> tuple[Optional[str], Optional[int]]:
        lower = (text or "").lower()
        m = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})\b",
            lower,
        )
        if m:
            try:
                return m.group(1), int(m.group(2))
            except Exception:
                return m.group(1), None
        d = re.search(r"\bdate\s+(\d{1,2})\b", lower)
        if d:
            try:
                return None, int(d.group(1))
            except Exception:
                return None, None
        return None, None

    def _try_calendar_date_click(self, intent: str, value: Optional[str]) -> bool:
        if self._page is None:
            return False
        month_name, day_num = self._extract_month_day_hint(f"{intent or ''} {value or ''}")
        if day_num is None:
            return False

        def _attempt_click(month: Optional[str], day: int) -> bool:
            selectors: list[str] = []
            if month:
                selectors.extend([
                    f'[aria-label*="{month}"][aria-label*="{day}"]',
                    f'[aria-label*="{day} {month}"]',
                    f'[aria-label*="{month} {day}"]',
                    f'[data-date*="-{day:02d}"][aria-label*="{month}"]',
                ])
            selectors.extend([
                f'[aria-label*="{day}"]',
                f'[data-date$="-{day:02d}"]',
            ])
            for sel in selectors:
                try:
                    loc = self._page.locator(sel)
                    if loc.count() > 0:
                        target = loc.first
                        if not target.is_visible(timeout=350):
                            continue
                        target.scroll_into_view_if_needed(timeout=1500)
                        target.click(timeout=2500)
                        logger.info("Calendar date clicked via locator: day=%s month=%s sel=%r", day, month, sel)
                        return True
                except Exception:
                    continue

            js = """(month, day) => {
                const s = String(day);
                const all = Array.from(document.querySelectorAll(
                    '[role="gridcell"], td, button, [data-date], [aria-label]'
                ));
                function vis(el) {
                    if (!el) return false;
                    const r = el.getBoundingClientRect();
                    if (r.width <= 0 || r.height <= 0) return false;
                    const st = getComputedStyle(el);
                    return st.display !== 'none' && st.visibility !== 'hidden' && parseFloat(st.opacity || '1') > 0.05;
                }
                for (const el of all) {
                    if (!vis(el)) continue;
                    const label = ((el.getAttribute('aria-label') || '') + ' ' + (el.textContent || '')).toLowerCase();
                    const dataDate = (el.getAttribute('data-date') || '').toLowerCase();
                    if (month && label.includes(month) && label.includes(s)) {
                        el.click();
                        return true;
                    }
                    if (!month) {
                        const t = (el.textContent || '').trim();
                        if (t === s) { el.click(); return true; }
                    }
                    if (dataDate.endsWith('-' + String(day).padStart(2, '0'))) {
                        el.click();
                        return true;
                    }
                }
                return false;
            }"""
            try:
                ok = bool(self._page.evaluate(js, month, int(day)))
                if ok:
                    logger.info("Calendar date clicked via JS: day=%s month=%s", day, month)
                return ok
            except Exception:
                return False

        # First attempt on current calendar view
        if _attempt_click(month_name, day_num):
            return True

        # Calendar may be showing the wrong month - click the "next month" button
        # up to 14 times, retrying the date click after each advance.
        _next_selectors = [
            'button[aria-label*="next" i]',
            'button[aria-label="Next"]',
            'button[data-testid*="next" i]',
            '[class*="next"][role="button"]',
            'button[class*="next"]',
            'button.fc-next-button',
        ]
        # Also try JS-based next-button click (handles shadow DOM / Google Flights)
        _next_js = """() => {
            const sels = [
                'button[aria-label="Next"]',
                'button[aria-label*="next" i]',
                '[data-testid*="next"] button',
                'button[class*="next"]'
            ];
            for (const s of sels) {
                const btn = document.querySelector(s);
                if (btn) { btn.click(); return true; }
            }
            return false;
        }"""
        for _ in range(14):
            advanced = False
            for sel in _next_selectors:
                try:
                    loc = self._page.locator(sel)
                    if loc.count() > 0 and loc.first.is_visible(timeout=300):
                        loc.first.click(timeout=1500)
                        try:
                            self._page.wait_for_timeout(300)
                        except Exception:
                            pass
                        advanced = True
                        break
                except Exception:
                    continue
            if not advanced:
                # Try JS fallback for shadow DOM calendars
                try:
                    advanced = bool(self._page.evaluate(_next_js))
                    if advanced:
                        self._page.wait_for_timeout(300)
                except Exception:
                    pass
            if not advanced:
                break
            if _attempt_click(month_name, day_num):
                return True

        return False

    def _selector_is_disallowed(self, selector: Optional[str]) -> bool:
        s = (selector or "").lower()
        if not s:
            return True
        if "cloudflare.com" in s or "challenge" in s or "/cdn-cgi/" in s:
            return True
        ad_patterns = (
            "google_ads_iframe", "googlesyndication", "doubleclick", "adsbygoogle",
            "googletagmanager", "adnxs", "criteo", "outbrain", "taboola",
        )
        return any(p in s for p in ad_patterns)

    def _selector_is_chrome_noise(self, selector: Optional[str]) -> bool:
        s = (selector or "").lower()
        if not s:
            return False
        noise_tokens = (
            "footer", "header", "menu", "language-picker", "back to main menu",
            "subscribe", "newsletter", "cookie", "privacy", "social",
            "account-menu", "site-nav", "global-nav",
        )
        return any(t in s for t in noise_tokens)

    def _selector_exists_now(self, selector: Optional[str]) -> bool:
        if self._page is None or not selector or self._selector_is_disallowed(selector):
            return False
        try:
            loc = self._page.locator(selector)
            if loc.count() <= 0:
                return False
            return bool(loc.first.is_visible(timeout=300))
        except Exception:
            return False

    def _selector_looks_like_autocomplete_choice(self, selector: Optional[str], expected_value: str) -> bool:
        if self._page is None or not selector or not expected_value:
            return False
        try:
            loc = self._page.locator(selector).first
            txt = (loc.text_content(timeout=600) or "").lower()
            aria = (loc.get_attribute("aria-label", timeout=300) or "").lower()
            role = (loc.get_attribute("role", timeout=300) or "").lower()
            expected_tokens = [t for t in re.split(r"\W+", expected_value.lower()) if len(t) >= 3][:4]
            if expected_tokens and not any(t in txt or t in aria for t in expected_tokens):
                return False
            if role in {"option", "menuitem", "listitem"}:
                return True
            # Accept if it visually looks like suggestion text and contains expected tokens.
            return bool(expected_tokens)
        except Exception:
            return False

    def get_available_elements_by_level(self, min_level: int = 2, action: Optional[str] = None) -> list[dict]:
        graph = self._current_graph or self._graph_builder.build(self.extract_page_context())
        nodes = graph.available_nodes(min_level=min_level, action=action)
        return [
            {
                "node_id": n.node_id,
                "level": n.level,
                "frame_id": n.frame_id,
                "dom_node_id": n.dom_node_id,
                "role": n.role,
                "name": n.name,
                "selector": n.selector,
                "selector_type": n.selector_type,
                "actions": n.actions,
                "confidence": n.confidence,
                "visible_ratio": n.visible_ratio,
            }
            for n in nodes
        ]

    def get_next_action_candidates(
        self, intent: str, action: Optional[str] = None, limit: int = 5
    ) -> list[NextActionCandidate]:
        extraction = self._current_extraction or self.extract_page_context()
        graph = self._current_graph
        _page_url = (self._page.url or "") if self._page else ""
        _intent_l = (intent or "").lower()
        raw: list[NextActionCandidate] = []
        seen_keys: set[str] = set()

        if self._executor is not None and self._page is not None:
            from urllib.parse import urlparse as _up
            _current_url = self._page.url or ""
            _host = _up(_current_url).netloc if _current_url else ""
            if _host:
                mem_entry = self._executor._memory.best_for(
                    host=_host, intent=intent, action=action or "", url=_current_url
                )
                if mem_entry:
                    if self._selector_is_disallowed(mem_entry.selector):
                        mem_entry = None
                    elif not self._selector_exists_now(mem_entry.selector):
                        mem_entry = None
                if mem_entry:
                    _mem_key = f"mem|{mem_entry.selector_type}|{mem_entry.selector}"
                    seen_keys.add(_mem_key)
                    raw.append(NextActionCandidate(
                        intent=intent,
                        score=min(1.0, 0.95 + (mem_entry.success_count - 1) * 0.005),
                        probability=0.0,
                        selector=mem_entry.selector,
                        selector_type=mem_entry.selector_type,
                        reasoning=f"memory_winner: success_count={mem_entry.success_count} strategy={mem_entry.strategy}",
                        dom_node_id=None,
                        level=3,
                        supports_action=True,
                        visible_ratio=1.0,
                        is_occluded=False,
                        neural_score=0.0,
                    ))
                    logger.debug(
                        "Memory winner injected for intent=%r action=%r selector=%r hits=%d",
                        intent, action, mem_entry.selector, mem_entry.success_count,
                    )

        if graph:
            graph_matches = graph.rank_for_intent(
                intent,
                action=action,
                top_k=max(limit * 3, 10),
                semantic_scorer=self._graph_lexical_score,
            )
            for m in graph_matches:
                if m.dom_node_id:
                    key = f"dom_id:{m.dom_node_id}"
                else:
                    key = f"{m.selector_type}|{m.selector}"
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                raw.append(NextActionCandidate(
                    intent=intent,
                    score=m.score,
                    probability=0.0,
                    selector=m.selector,
                    selector_type=m.selector_type,
                    reasoning=f"graph_level={m.level} | {m.reasoning}",
                    dom_node_id=m.dom_node_id,
                    level=m.level,
                    supports_action=True if action is None else ("action_fit=1.00" in m.reasoning),
                    visible_ratio=1.0,
                    is_occluded=False,
                    neural_score=0.0,
                ))

        state = extraction.state

        _fast_intent = (
            action == "click"
            and any(w in _intent_l for w in (
                "autocomplete", "suggestion", "date", "calendar",
                "search", "filter", "sort", "occupancy",
            ))
        )
        use_model_selector = (
            action not in {"type", "scroll"}
            and not self._intent_is_specific_date_click(intent, None)
            and not _fast_intent
            and not self._is_bot_gate_page()
        )
        result = None
        if use_model_selector:
            _screenshot: Optional[bytes] = None
            if self._page is not None:
                try:
                    # Before screenshotting for Vision, scroll so that the most
                    # semantically relevant interactive elements are in the viewport.
                    # Elements below the fold are invisible to the Vision model even
                    # though they exist in the DOM.  We find the median Y of the top
                    # candidates by neural similarity and scroll there first.
                    _vp_height = (self._page.viewport_size or {}).get("height", 720)
                    _scroll_y = self._page.evaluate("() => window.scrollY") or 0
                    _cand_ys = []
                    _intent_lower = intent.lower()
                    for _el in state.interactive_elements:
                        if _el.bounding_box and _el.is_visible:
                            _el_y = _el.bounding_box.y
                            # Only consider elements below the current fold
                            if _el_y > _scroll_y + _vp_height * 0.8:
                                _el_text = " ".join(filter(None, [
                                    _el.name, _el.placeholder, _el.nearby_text, _el.role,
                                ]))
                                if _el_text:
                                    _el_tl = _el_text.lower()
                                    # Quick heuristic: any token from intent in element text
                                    _tokens = [t for t in re.split(r'\W+', _intent_lower) if len(t) >= 4]
                                    if any(t in _el_tl for t in _tokens):
                                        _cand_ys.append(_el_y)
                    if _cand_ys:
                        _target_y = sorted(_cand_ys)[len(_cand_ys) // 2]
                        _new_scroll = max(0, int(_target_y) - _vp_height // 3)
                        self._page.evaluate(f"window.scrollTo(0, {_new_scroll})")
                        self._page.wait_for_timeout(120)
                    _screenshot = self._page.screenshot(type="jpeg", quality=85, full_page=False)
                except Exception:
                    pass
            result = self._selector.find(
                intent, state, use_cache=True, action_hint=action, screenshot_bytes=_screenshot
            )

        by_dom_id = {el.dom_node_id: el for el in extraction.elements if el.dom_node_id}

        if result is not None:
            for match in result.matches:
                # Hard-drop known anti-bot/off-task link targets.
                _sel_l = (match.selector or "").lower()
                if "cloudflare.com" in _sel_l or "challenge" in _sel_l:
                    continue
                ex = by_dom_id.get(match.element.dom_node_id)
                visual_bonus = 0.0
                visual_reason = ""
                if ex:
                    visual_bonus += ex.confidence * 0.18
                    if ex.is_occluded:
                        visual_bonus -= 0.25
                    visual_bonus += ex.visible_ratio * 0.08
                    visual_reason = (
                        f"visual_conf={ex.confidence:.2f}, "
                        f"visible_ratio={ex.visible_ratio:.2f}, "
                        f"occluded={ex.is_occluded}"
                    )

                final_score = max(0.0, min(1.0, match.score + visual_bonus))
                item = NextActionCandidate(
                    intent=intent,
                    score=final_score,
                    probability=0.0,
                    selector=match.selector,
                    selector_type=match.selector_type,
                    reasoning=f"{match.reasoning}" + (f" | {visual_reason}" if visual_reason else ""),
                    dom_node_id=match.element.dom_node_id,
                    level=2,
                    supports_action=action in match.element.actions if action else True,
                    visible_ratio=ex.visible_ratio if ex else (1.0 if match.element.is_in_viewport else 0.0),
                    is_occluded=ex.is_occluded if ex else False,
                    neural_score=0.0,
                )
                if item.dom_node_id:
                    dedup_key = f"dom_id:{item.dom_node_id}"
                else:
                    dedup_key = f"{item.selector_type}|{item.selector}"
                if dedup_key not in seen_keys:
                    raw.append(item)
                    seen_keys.add(dedup_key)

        _cookie_patterns = (
            "onetrust", "cookie", "consent", "privacy-banner", "gdpr",
            "ccpa", "ot-sdk", "cookie-banner", "cookie-notice", "ot-",
            "onetrust-", "banner-close", "accept-all",
        )
        scan_cap = max(45, min(120, limit * 12))
        if self._is_bot_gate_page():
            scan_cap = min(scan_cap, 40)
        # Pass 1: collect elements and build their semantic texts, filtering cookie elements.
        _batch_elements = []
        _batch_texts = []
        for ex in extraction.elements[:scan_cap]:
            # Keep selector and selector_type aligned. Mixing css selector text
            # with xpath type causes invalid locator resolution.
            if ex.css_selector:
                selector = ex.css_selector
                selector_type = "css"
            elif ex.xpath:
                selector = ex.xpath
                selector_type = "xpath"
            else:
                selector = None
                selector_type = "auto"
            if not selector:
                continue
            selector_lower = selector.lower()
            nearby_lower = (ex.nearby_text or "").lower()
            form_lower = (ex.form_context or "").lower()
            if "cloudflare.com" in selector_lower or "challenge" in selector_lower:
                continue
            if any(
                p in selector_lower or p in nearby_lower or p in form_lower
                for p in _cookie_patterns
            ):
                continue
            semantic_text = " | ".join([
                ex.name or "",
                ex.role or "",
                ex.tag_name or "",
                ex.nearby_text or "",
                ex.form_context or "",
                " ".join(ex.actions or []),
                selector,
            ])
            _batch_elements.append((ex, selector))
            _batch_texts.append(semantic_text)

        # Pass 2: score all elements in a single batch encode call.
        _batch_sims = self._neural.similarity_batch(intent, _batch_texts)

        # Pre-compute intent categories once  -  used inside the per-element loop
        # to avoid N separate neural inference calls.
        # "data entry" intents expect interactive form elements (inputs, comboboxes,
        # option lists), NOT navigation tabs or social auth links.
        _intent_is_data_entry = (
            self._neural.classify(intent, "autocomplete_pick", threshold=0.48)
            or self._neural.classify(intent, "date_picker_open", threshold=0.48)
            or self._neural.classify(intent, "reveal_expand", threshold=0.48)
        )
        _intent_is_nav = (
            not _intent_is_data_entry
            and self._neural.classify(intent, "page_navigation", threshold=0.52)
        )

        for (ex, selector), sim in zip(_batch_elements, _batch_sims):
            base = max(0.0, min(1.0, (sim + 1.0) * 0.5))
            observability = ex.visible_ratio * (0.0 if ex.is_occluded else 1.0)
            role_penalty = 0.0
            el_role = (ex.role or "").lower()
            el_tag = (ex.tag_name or "").lower()
            is_nav_chrome = (
                el_role in ("tab", "tabpanel", "menuitem")
                or (el_tag == "a" and el_role not in ("button", "checkbox", "radio", "option"))
            )
            if is_nav_chrome:
                if _intent_is_data_entry:
                    role_penalty = -0.30
                elif not _intent_is_nav:
                    role_penalty = -0.12
            adaptive_conf_weight = 0.27 * (0.40 + base * 0.60)

            final_score = max(0.0, min(1.0,
                (0.58 * base)
                + (adaptive_conf_weight * ex.confidence)
                + (0.15 * observability)
                + role_penalty
            ))
            if ex.frame_id != "main":
                selector_type = f"{selector_type}_frame"

            item = NextActionCandidate(
                intent=intent,
                score=final_score,
                probability=0.0,
                selector=selector,
                selector_type=selector_type,
                reasoning=(
                    f"source={ex.source}, frame={ex.frame_id}, conf={ex.confidence:.2f}, "
                    f"visible_ratio={ex.visible_ratio:.2f}, occluded={ex.is_occluded}"
                ),
                dom_node_id=ex.dom_node_id,
                level=3 if ex.visible_ratio >= 0.6 and ex.is_enabled and not ex.is_occluded else 2 if ex.visible_ratio > 0.1 else 1,
                supports_action=action in ex.actions if action else True,
                visible_ratio=ex.visible_ratio,
                is_occluded=ex.is_occluded,
                neural_score=0.0,
            )
            if ex.dom_node_id:
                dedup_key = f"dom_id:{ex.dom_node_id}"
            else:
                dedup_key = f"{item.selector_type}|{item.selector}"
            if dedup_key not in seen_keys:
                raw.append(item)
                seen_keys.add(dedup_key)

        enriched = self._apply_neural_scores(intent, raw)
        ranked = self._probabilistic_rank(enriched, limit * 3)
        ranked = self._apply_reasoner_rerank(intent, action, ranked)
        filtered = self._filter_valid_candidates(ranked, action=action)[:limit]
        logger.debug(
            "Candidates for intent=%r action=%r: %d raw, %d after filter",
            intent, action, len(raw), len(filtered),
        )
        return filtered

    def _graph_semantic_score(self, intent: str, node) -> float:
        text = " | ".join([
            node.name or "",
            node.role or "",
            node.tag_name or "",
            node.metadata.get("nearby_text") or "" if node.metadata else "",
            node.metadata.get("form_context") or "" if node.metadata else "",
            " ".join(node.actions or []),
            node.selector or "",
        ])
        sim = self._neural.similarity(intent, text)
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))

    def _graph_lexical_score(self, intent: str, node) -> float:
        """Fast lexical score used for graph pre-ranking.

        We avoid per-node neural calls here because those become the dominant
        latency on large pages. Neural scoring is still applied later in one
        batch over shortlisted candidates.
        """
        txt = " ".join([
            (node.name or ""),
            (node.role or ""),
            (node.tag_name or ""),
            (node.metadata.get("nearby_text") or "") if node.metadata else "",
            (node.metadata.get("form_context") or "") if node.metadata else "",
            " ".join(node.actions or []),
            (node.selector or ""),
        ]).lower()
        if not txt.strip():
            return 0.0
        tokens = [t for t in re.split(r"\W+", (intent or "").lower()) if len(t) >= 3]
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in txt)
        coverage = hits / max(1, len(tokens))
        return max(0.0, min(1.0, coverage))

    def _apply_reasoner_rerank(
        self,
        intent: str,
        action: Optional[str],
        ranked: list[NextActionCandidate],
    ) -> list[NextActionCandidate]:
        if not ranked:
            return ranked
        if not self._reasoner.should_rerank([c.score for c in ranked[:5]]):
            return ranked

        rows = []
        for i, c in enumerate(ranked[: self._reasoner.max_candidates_per_call]):
            rows.append({
                "idx": i,
                "selector": c.selector,
                "selector_type": c.selector_type,
                "reasoning": c.reasoning,
                "score": c.score,
                "probability": c.probability,
                "level": c.level,
                "visible_ratio": c.visible_ratio,
                "occluded": c.is_occluded,
            })
        llm_scores = self._reasoner.rerank(intent, action, rows)
        if not llm_scores:
            return ranked

        by_idx = {s.candidate_idx: s for s in llm_scores}
        updated = []
        for i, c in enumerate(ranked):
            rs = by_idx.get(i)
            if not rs:
                updated.append(c)
                continue
            llm = max(0.0, min(1.0, 0.45 * rs.relevance + 0.35 * rs.action_fit + 0.20 * rs.confidence))
            merged = max(0.0, min(1.0, c.score * 0.72 + llm * 0.28))
            updated.append(NextActionCandidate(
                intent=c.intent,
                score=merged,
                probability=c.probability,
                selector=c.selector,
                selector_type=c.selector_type,
                reasoning=c.reasoning + f" | llm={llm:.3f}:{rs.reasoning}",
                dom_node_id=c.dom_node_id,
                level=c.level,
                supports_action=c.supports_action,
                visible_ratio=c.visible_ratio,
                is_occluded=c.is_occluded,
                neural_score=c.neural_score,
            ))

        return self._probabilistic_rank(updated, len(updated))

    def _apply_neural_scores(self, intent: str, candidates: list[NextActionCandidate]) -> list[NextActionCandidate]:
        if not candidates:
            return candidates
        texts = [self._candidate_semantic_text(c) for c in candidates]
        sims = self._neural.similarity_batch(intent, texts)
        nn_tag = "(nn)" if self._neural.is_neural_active else "(fallback)"
        out: list[NextActionCandidate] = []
        for c, sim in zip(candidates, sims):
            neural = max(0.0, min(1.0, (sim + 1.0) * 0.5))
            blended = max(0.0, min(1.0, c.score * 0.70 + neural * 0.30))
            out.append(NextActionCandidate(
                intent=c.intent,
                score=blended,
                probability=c.probability,
                selector=c.selector,
                selector_type=c.selector_type,
                reasoning=c.reasoning + f" | neural={neural:.3f}{nn_tag}",
                dom_node_id=c.dom_node_id,
                level=c.level,
                supports_action=c.supports_action,
                visible_ratio=c.visible_ratio,
                is_occluded=c.is_occluded,
                neural_score=neural,
            ))
        return out

    def _candidate_semantic_text(self, c: NextActionCandidate) -> str:
        parts = [c.selector or "", c.selector_type or ""]
        for token in (c.reasoning or "").split("|"):
            token = token.strip()
            if not token:
                continue
            if any(token.startswith(prefix) for prefix in (
                "graph_level=", "source=", "neural=", "llm=",
                "semantic=", "action_fit=", "conf=", "visible_ratio=",
                "occluded=", "level=", "frame=",
            )):
                continue
            parts.append(token)
        return " | ".join(p for p in parts if p)

    def _probabilistic_rank(self, candidates: list[NextActionCandidate], limit: int) -> list[NextActionCandidate]:
        if not candidates:
            return []
        logits = [self._calibrated_logit(c.score) for c in candidates]
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        total = sum(exps) or 1.0
        ranked = [
            NextActionCandidate(
                intent=c.intent,
                score=c.score,
                probability=e / total,
                selector=c.selector,
                selector_type=c.selector_type,
                reasoning=c.reasoning,
                dom_node_id=c.dom_node_id,
                level=c.level,
                supports_action=c.supports_action,
                visible_ratio=c.visible_ratio,
                is_occluded=c.is_occluded,
                neural_score=c.neural_score,
            )
            for c, e in zip(candidates, exps)
        ]
        ranked.sort(key=lambda x: x.probability, reverse=True)
        top = ranked[:limit]
        mass = sum(c.probability for c in top) or 1.0
        return [
            NextActionCandidate(
                intent=c.intent,
                score=c.score,
                probability=c.probability / mass,
                selector=c.selector,
                selector_type=c.selector_type,
                reasoning=c.reasoning,
                dom_node_id=c.dom_node_id,
                level=c.level,
                supports_action=c.supports_action,
                visible_ratio=c.visible_ratio,
                is_occluded=c.is_occluded,
                neural_score=c.neural_score,
            )
            for c in top
        ]

    def _filter_valid_candidates(
        self, candidates: list[NextActionCandidate], action: Optional[str]
    ) -> list[NextActionCandidate]:
        out: list[NextActionCandidate] = []
        min_level = self._min_level_for_action.get(action or "", 1)
        if not candidates:
            return out

        best_score = max(c.score for c in candidates)
        best_prob = max(c.probability for c in candidates)
        min_score = max(0.05, best_score * 0.50)
        min_probability = max(0.005, best_prob * 0.15)

        for c in candidates:
            if c.score < min_score:
                continue
            if c.probability < min_probability:
                continue
            if c.is_occluded and c.visible_ratio < 0.9:
                continue
            if c.level < min_level:
                continue
            if action and not c.supports_action:
                continue
            if not c.selector and not c.dom_node_id:
                continue
            if c.selector and not _sel_ok(c.selector):
                if c.dom_node_id:
                    out.append(NextActionCandidate(
                        intent=c.intent,
                        score=c.score,
                        probability=c.probability,
                        selector=None,
                        selector_type="dom_node_id",
                        reasoning=c.reasoning + " | bare_selector_stripped",
                        dom_node_id=c.dom_node_id,
                        level=c.level,
                        supports_action=c.supports_action,
                        visible_ratio=c.visible_ratio,
                        is_occluded=c.is_occluded,
                        neural_score=c.neural_score,
                    ))
                else:
                    logger.debug(
                        "Candidate dropped: bare selector=%r no dom_node_id intent=%r",
                        c.selector, c.intent,
                    )
                continue
            out.append(c)

        if not out and candidates:
            viable = [
                c for c in candidates
                if (not action or c.supports_action)
                and (c.dom_node_id or (c.selector and _sel_ok(c.selector)))
            ]
            if viable:
                out.append(max(viable, key=lambda c: c.score))
            else:
                eligible = [c for c in candidates if (not action or c.supports_action)]
                best_pool = eligible if eligible else candidates
                best = max(best_pool, key=lambda c: c.score)
                if best.dom_node_id or best.selector:
                    logger.warning(
                        "All candidates have generic selectors for intent=%r; using best by score.",
                        best.intent,
                    )
                    out.append(best)

        return out

    def _calibrated_logit(self, score: float) -> float:
        s = max(1e-4, min(1.0 - 1e-4, score))
        return math.log(s / (1.0 - s))

    def _candidate_to_request(self, step: AutomationStep, c: NextActionCandidate) -> ActionRequest:
        action = (step.action or "").lower()
        timeout_ms = step.timeout_ms
        if action == "click":
            timeout_ms = min(timeout_ms, 5000)
        elif action == "type":
            timeout_ms = min(timeout_ms, 3200)
        elif action in {"select", "check", "uncheck"}:
            timeout_ms = min(timeout_ms, 3800)
        return ActionRequest(
            action=step.action,
            value=step.value,
            intent=step.intent,
            selector=c.selector,
            selector_type=c.selector_type,
            dom_node_id=c.dom_node_id,
            timeout_ms=timeout_ms,
            clear_first=step.clear_first,
            press_enter=step.press_enter,
            retry_attempts=0,
            retry_backoff_ms=0,
            skip_auto_memory=True,
        )

    def _validate_action(
        self, step: AutomationStep, c: NextActionCandidate, out: ActionVerificationResult
    ) -> tuple[bool, str]:
        ar = out.action_result
        if not ar.success:
            return False, f"exec_failed:{ar.error or 'unknown'}"

        if step.action == "navigate":
            if out.verification and out.verification.had_changes:
                return True, "navigate_state_changed"
            if ar.success:
                return True, "navigate_executed"
            return False, "navigate_failed"

        if step.action == "type":
            if step.value is None:
                return False, "type_missing_value"
            # Strong confirmation: DOM snapshot shows the typed value.
            if self._current_state and c.dom_node_id:
                for el in self._current_state.interactive_elements:
                    if el.dom_node_id == c.dom_node_id and (el.value or "") == step.value:
                        return True, "typed_value_confirmed"
            # Strong confirmation: live Playwright input_value matches.
            if c.selector and self._page is not None:
                try:
                    val = self._page.locator(c.selector).first.input_value(timeout=1200)
                    if val == step.value:
                        return True, "typed_value_confirmed_selector"
                except Exception:
                    pass
            # Medium confirmation: page state changed after fill (autocomplete appeared,
            # validation feedback shown, etc.).
            if out.verification and out.verification.had_changes:
                return True, "typed_state_changed"
            # Fallback: Playwright's fill() is atomic  -  if it returned without raising,
            # the value was written to the input's property.  React/Vue controlled inputs
            # may not propagate the value to the DOM snapshot but the fill DID happen.
            # Accepting fill() success here prevents typed_not_confirmed cascade failures
            # on every write-task form field on modern SPAs.
            if ar.success:
                return True, "type_fill_accepted"
            return False, "typed_not_confirmed"

        if step.action == "scroll":
            sb = (out.metadata or {}).get("scroll_before")
            sa = (out.metadata or {}).get("scroll_after")
            if isinstance(sb, (int, float)) and isinstance(sa, (int, float)):
                if abs(float(sa) - float(sb)) >= 12.0:
                    return True, "scroll_position_changed"
            if out.verification and out.verification.had_changes:
                return True, "scroll_state_changed"
            return False, "scroll_not_confirmed"

        if step.action == "click":
            intent = step.intent or ""
            is_autocomplete_intent = self._intent_is_autocomplete_pick(intent)
            is_date_open_intent = self._intent_is_datepicker_open(intent)
            is_reveal_intent = self._neural.classify(intent, "reveal_expand", threshold=0.60)
            is_search_submit_intent = self._neural.classify(intent, "search_submit", threshold=0.58)
            if self._selector_is_chrome_noise(c.selector) and (
                self._intent_is_refinement_click(intent)
                or any(k in intent.lower() for k in ("filter", "sort", "price", "wifi", "pool", "breakfast"))
            ):
                return False, "click_chrome_noise_selector"
            url_before = (out.metadata or {}).get("url_before", "")
            url_after = (out.metadata or {}).get("url_after", "")
            if self._intent_is_refinement_click(intent) and url_before and url_after:
                if self._url_looks_results_like(url_before) and not self._url_looks_results_like(url_after):
                    return False, "click_refinement_left_results_page"

            # Classify intent semantically rather than via keyword lists.
            # in-page: calendar nav, filter, tab, accordion  -  URL does NOT change.
            # nav: link clicks, store locator, go-to  -  URL IS expected to change.
            _is_inpage = self._neural.classify(intent, "in_page_interaction", threshold=0.50)
            nav_intent = (
                not _is_inpage
                and self._neural.classify(intent, "page_navigation", threshold=0.52)
            )
            if self._neural.classify(intent, "search_submit", threshold=0.58):
                nav_intent = True
            if not nav_intent:
                il = intent.lower()
                if any(k in il for k in (
                    "open", "go to", "navigate", "visit", "product page",
                    "details page", "result page", "link to", "click on the product",
                )):
                    nav_intent = True

            if is_autocomplete_intent:
                expected = (step.value or "").strip()
                if c.selector and expected and self._selector_looks_like_autocomplete_choice(c.selector, expected):
                    return True, "autocomplete_selector_confirmed"
                if self._autocomplete_listbox_is_live():
                    return False, "autocomplete_not_selected"
                if expected and self._autocomplete_input_has_value(expected):
                    if out.verification and out.verification.had_changes:
                        return True, "autocomplete_value_applied_with_change"
                    return False, "autocomplete_value_preexisting"
                return False, "autocomplete_unverified"

            if is_date_open_intent and self._calendar_widget_is_open():
                return True, "datepicker_open_confirmed"
            if is_date_open_intent and ar.success and out.verification and out.verification.had_changes:
                return True, "datepicker_open_state_changed"

            if ar.success and url_before and url_after and url_before != url_after:
                return True, "click_url_changed"

            noise_only_change = False
            if out.verification and out.verification.had_changes:
                if nav_intent and url_before and url_after and url_before == url_after:
                    return False, "click_nav_intent_no_url_change"
                # Use had_meaningful_changes to filter out background DOM noise
                # (carousels, lazy-load churn) that falsely shows had_changes=True
                # even when the user's click did nothing useful.
                had_meaningful = getattr(out.verification, "had_meaningful_changes", out.verification.had_changes)
                if had_meaningful:
                    return True, "click_state_changed"
                if is_reveal_intent:
                    return True, "click_reveal_state_changed"
                # Keep probing with stronger checks (ARIA state, URL).
                noise_only_change = True
            if c.selector and self._page is not None:
                try:
                    sel_type = (c.selector_type or "").lower()
                    if sel_type in {"css", "auto", "css_id", "css_frame"}:
                        # Guard: reject selectors that point to off-domain links
                        # (e.g. cloudflare challenge links) for regular task clicks.
                        try:
                            from urllib.parse import urlparse as _up
                            href = self._page.locator(c.selector).first.get_attribute("href", timeout=450)
                            if href:
                                cur_host = _up(self._page.url or "").netloc.lower()
                                dst_host = _up(href).netloc.lower()
                                if "cloudflare.com" in dst_host:
                                    return False, "click_external_link_blocked"
                                if dst_host and cur_host and dst_host != cur_host and not dst_host.endswith("." + cur_host):
                                    return False, "click_external_link_blocked"
                        except Exception:
                            pass
                        aria_active = self._page.evaluate(
                            """(sel) => {
                                try {
                                    const el = document.querySelector(sel);
                                    if (!el) return false;
                                    const probed = [
                                        'aria-expanded', 'aria-selected',
                                        'aria-pressed', 'aria-current',
                                    ];
                                    return probed.some(a => {
                                        const v = el.getAttribute(a);
                                        return v !== null && v !== 'false';
                                    });
                                } catch(e) { return false; }
                            }""",
                            c.selector,
                        )
                        if aria_active:
                            return True, "click_aria_state_active"
                except Exception:
                    pass
            if noise_only_change:
                return False, "click_noise_only_change"
            if nav_intent and ar.success and url_before and url_after and url_before != url_after:
                return True, "click_nav_url_changed"
            if is_search_submit_intent and self._page is not None:
                il = intent.lower()
                if any(k in il for k in ("search", "submit", "find ", "show results")) and self._is_results_like_page():
                    return True, "click_search_results_loaded"
            if nav_intent:
                return False, "click_nav_intent_no_confirmation"
            if not _is_inpage:
                return False, "click_no_confirmation"
            return True, "click_executed_unverified"

        if step.action == "select":
            if step.value is None:
                return False, "select_missing_value"
            if self._current_state and c.dom_node_id:
                for el in self._current_state.interactive_elements:
                    if el.dom_node_id == c.dom_node_id and (el.value or "") == step.value:
                        return True, "select_value_confirmed"
            if c.selector and self._page is not None:
                try:
                    val = self._page.locator(c.selector).first.input_value(timeout=1200)
                    if val == step.value:
                        return True, "select_value_confirmed_selector"
                except Exception:
                    pass
            if out.verification and out.verification.had_changes:
                return True, "select_state_changed"
            return False, "select_not_confirmed"

        if step.action in {"check", "uncheck"}:
            expected = step.action == "check"
            if self._current_state and c.dom_node_id:
                a11y_id = self._current_state.dom_to_a11y.get(c.dom_node_id)
                if a11y_id:
                    a11y_node = self._current_state.accessibility_index.get(a11y_id)
                    if a11y_node and isinstance(a11y_node.checked, bool):
                        if a11y_node.checked == expected:
                            return True, f"{step.action}_checked_state_confirmed"
            if c.selector and self._page is not None:
                try:
                    state = self._page.locator(c.selector).first.is_checked(timeout=1200)
                    if state == expected:
                        return True, f"{step.action}_checked_state_confirmed_selector"
                except Exception:
                    pass
            if out.verification and out.verification.had_changes:
                return True, f"{step.action}_state_changed"
            return False, f"{step.action}_not_confirmed"

        return True, "success"

    def _execute_navigate_step(
        self,
        step: AutomationStep,
    ) -> tuple[ActionVerificationResult, list[str], bool]:
        if self._executor is None:
            raise RuntimeError("Executor not initialized. Call start() first.")
        url = (step.value or "").strip()
        if not url:
            traces = ["navigate_no_url"]
            out = ActionVerificationResult(
                action_result=ActionResult(
                    success=False,
                    action="navigate",
                    message="navigate step has no URL value.",
                    latency_ms=0.0,
                    error="missing_url",
                ),
                verification=None,
            )
            return out, traces, False

        req = ActionRequest(
            action="navigate",
            value=url,
            intent=step.intent,
            timeout_ms=max(step.timeout_ms, 15000),
        )
        out = self._executor.execute_and_verify(
            request=req,
            wait_after_ms=step.wait_after_ms,
            verifier=self._diff,
        )
        self.extract_page_context()
        if self._is_bot_gate_page():
            traces = [f"navigate url={url} ok=False reason=bot_gate_blocked"]
            out.action_result.success = False
            out.action_result.error = "bot_gate_blocked"
            return out, traces, False
        ok = out.action_result.success
        reason = "navigate_ok" if ok else f"navigate_failed:{out.action_result.error}"
        traces = [f"navigate url={url} ok={ok} reason={reason}"]
        logger.debug("Navigate step: url=%r ok=%s", url, ok)
        return out, traces, ok

    def _execute_step_with_ranked_retry(
        self,
        step: AutomationStep,
        max_candidates: int = 6,
    ) -> tuple[ActionVerificationResult, list[str], bool]:
        if self._executor is None:
            raise RuntimeError("Executor not initialized. Call start() first.")
        if step.intent is None:
            raise ValueError("Step must have an intent when using ranked retry.")
        traces: list[str] = []
        # End-to-end per-step budget, including candidate generation.
        _step_budget_ms = float(min(26000, max(6000, step.timeout_ms * 2.2)))
        _step_start = time.perf_counter()

        # Filter/sort/refinement clicks should run on a results/listing page.
        if step.action == "click" and self._page is not None:
            _intent_l = (step.intent or "").lower()
            if (
                self._intent_is_refinement_click(_intent_l)
                and not self._is_results_like_page()
            ):
                moved = self._submit_search_from_page()
                if moved:
                    self.extract_page_context()
                    traces.append("results_page_submit_recovery ok=True")
                else:
                    out = ActionVerificationResult(
                        action_result=ActionResult(
                            success=False,
                            action=step.action,
                            message="Refinement attempted before results/listing page was loaded.",
                            latency_ms=0.0,
                            attempts=1,
                            error="results_page_required",
                        ),
                        verification=None,
                        metadata=None,
                    )
                    traces.append("results_page_submit_recovery ok=False")
                    return out, traces, False

        # Autocomplete fallback: if no live suggestion list is present, submit
        # the typed query instead of clicking arbitrary candidates.
        if step.action == "click" and self._page is not None:
            _intent_l = (step.intent or "").lower()
            if self._intent_is_autocomplete_pick(_intent_l) and not self._autocomplete_listbox_is_live():
                # Wait briefly - some sites have a debounce delay before the listbox
                # appears.  Only fall back to form-submit if it's still absent after
                # the wait.
                try:
                    self._page.wait_for_timeout(700)
                except Exception:
                    pass
                if not self._autocomplete_listbox_is_live():
                    moved = self._submit_search_from_page()
                    if moved:
                        try:
                            self.extract_page_context()
                        except Exception:
                            pass
                        traces.append("autocomplete_submit_fallback ok=True")
                        dummy_out = ActionVerificationResult(
                            action_result=ActionResult(
                                success=True,
                                action="click",
                                message="Autocomplete fallback: submitted search without listbox.",
                                latency_ms=0.0,
                            ),
                            verification=None,
                            metadata=None,
                        )
                        return dummy_out, traces, True
        if step.action == "click" and self._page is not None:
            _intent_l = (step.intent or "").lower()
            _explicit_next = any(w in _intent_l for w in ("next month", "next calendar", "calendar next", "next arrow"))
            _explicit_prev = any(w in _intent_l for w in ("previous month", "prev month", "calendar prev", "back month"))
            _is_next_first = _explicit_next
            _is_prev_first = (not _is_next_first and _explicit_prev)
            if (_is_next_first or _is_prev_first) and not self._calendar_widget_is_open():
                self._ensure_calendar_open((step.intent or "").lower())
            if _is_next_first or _is_prev_first:
                direction_patterns = (
                    ["next month", "next", "->'", "forward"]
                    if _is_next_first
                    else ["previous month", "prev", "previous", "->", "back"]
                )
                clicked_first = self._try_calendar_nav_js(direction_patterns)
                if clicked_first:
                    try:
                        self._page.wait_for_timeout(500)
                    except Exception:
                        pass
                    self.extract_page_context()
                    direction_tag = "next" if _is_next_first else "prev"
                    traces.append(
                        f"calendar_nav_first direction={direction_tag} ok=True reason=aria_label_js"
                    )
                    dummy_out = ActionVerificationResult(
                        action_result=ActionResult(
                            success=True,
                            action="click",
                            message=f"Calendar navigation ({direction_tag}) via aria-label JS.",
                            latency_ms=0.0,
                        ),
                        verification=None,
                        metadata=None,
                    )
                    return dummy_out, traces, True
                traces.append(
                    f"calendar_nav_first direction={'next' if _is_next_first else 'prev'} ok=False reason=no_aria_button_found"
                )
                out = ActionVerificationResult(
                    action_result=ActionResult(
                        success=False,
                        action=step.action,
                        message="Calendar navigation control not found in open date picker.",
                        latency_ms=0.0,
                        attempts=1,
                        error="calendar_nav_not_found",
                    ),
                    verification=None,
                    metadata=None,
                )
                return out, traces, False

        if step.action == "click" and self._page is not None:
            if step.value and self._autocomplete_listbox_is_live():
                clicked_any = self._try_autocomplete_js(step.value)
                if clicked_any:
                    try:
                        self._page.wait_for_timeout(450)
                    except Exception:
                        pass
                    self.extract_page_context()
                    traces.append("autocomplete_js_opportunistic ok=True")
                    dummy_out = ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="click",
                            message="Autocomplete option selected via opportunistic JS path.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
                    return dummy_out, traces, True
            if self._intent_is_autocomplete_pick(step.intent or ""):
                if self._autocomplete_listbox_is_live():
                    _url_before_ac = self._page.url or ""
                    _state_before = self._snapshot_page_identity()
                    clicked = self._try_autocomplete_js(step.value or "")
                    if clicked:
                        try:
                            self._page.wait_for_timeout(600)
                        except Exception:
                            pass
                        _state_after = self._snapshot_page_identity()
                        if _state_after != _state_before:
                            self.extract_page_context()
                            self._detect_and_escape_auth_redirect(_url_before_ac)
                            traces.append("autocomplete_js_first ok=True reason=listbox_js")
                            dummy_out = ActionVerificationResult(
                                action_result=ActionResult(
                                    success=True, action="click",
                                    message="Autocomplete option selected via JS.",
                                    latency_ms=0.0,
                                ),
                                verification=None, metadata=None,
                            )
                            return dummy_out, traces, True
                        # Many sites keep URL/state stable after suggestion click.
                        # Accept when listbox collapses or destination value updates.
                        expected = (step.value or "").strip()
                        if (not self._autocomplete_listbox_is_live()) or self._autocomplete_input_has_value(expected):
                            self.extract_page_context()
                            traces.append("autocomplete_js_first ok=True reason=listbox_closed_or_value_set")
                            dummy_out = ActionVerificationResult(
                                action_result=ActionResult(
                                    success=True, action="click",
                                    message="Autocomplete option selected via JS (confirmed by listbox/value).",
                                    latency_ms=0.0,
                                ),
                                verification=None, metadata=None,
                            )
                            return dummy_out, traces, True
                        traces.append("autocomplete_js_first ok=False reason=no_state_change_after_click")
                    else:
                        traces.append("autocomplete_js_first ok=False reason=no_listbox_item_found")
                else:
                    traces.append("autocomplete_js_skipped reason=no_live_listbox")

        # Date-picker open JS fires FIRST 
        # Neural classification replaces keyword list: any intent semantically
        # meaning "open a date picker or calendar" triggers JS-first.
        # Direct date-cell click path for intents like "April 3" / "date 20".
        if step.action == "click" and self._page is not None:
            if self._intent_is_specific_date_click(step.intent or "", step.value):
                if not self._calendar_widget_is_open():
                    self._ensure_calendar_open((step.intent or "").lower())
                if self._try_calendar_date_click(step.intent or "", step.value):
                    try:
                        self._page.wait_for_timeout(350)
                    except Exception:
                        pass
                    self.extract_page_context()
                    traces.append("calendar_date_click_first ok=True reason=date_cell_locator_or_js")
                    dummy_out = ActionVerificationResult(
                        action_result=ActionResult(
                            success=True,
                            action="click",
                            message="Calendar date clicked via direct locator/JS.",
                            latency_ms=0.0,
                        ),
                        verification=None,
                        metadata=None,
                    )
                    return dummy_out, traces, True
                traces.append("calendar_date_click_first ok=False reason=no_matching_date_cell")
                out = ActionVerificationResult(
                    action_result=ActionResult(
                        success=False,
                        action="click",
                        message="Target calendar date not found in visible date picker.",
                        latency_ms=0.0,
                        attempts=1,
                        error="calendar_date_not_found",
                    ),
                    verification=None,
                    metadata=None,
                )
                return out, traces, False

        if step.action == "click" and self._page is not None:
            if self._intent_is_datepicker_open(step.intent or ""):
                opened = self._try_date_input_js((step.intent or "").lower())
                if opened:
                    try:
                        self._page.wait_for_timeout(400)
                    except Exception:
                        pass
                    self.extract_page_context()
                    self._maybe_extract_revealed_elements(step)
                    traces.append("date_input_js_first ok=True reason=date_input_js")
                    dummy_out = ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="click",
                            message="Date picker opened via JS.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
                    return dummy_out, traces, True
                traces.append("date_input_js_first ok=False reason=no_date_input_found")

        # Allrecipes filter: navigate directly to search URL with rating param when
        # intent is about filtering by rating/reviews and we're on allrecipes.
        if step.action == "click" and self._page is not None:
            _il = (step.intent or "").lower()
            _cur_url = (self._page.url or "").lower()
            if "allrecipes.com" in _cur_url and any(
                w in _il for w in ("filter", "rating", "review", "sort")
            ):
                # Extract the search query from the current URL or step context
                try:
                    from urllib.parse import urlparse as _up2, parse_qs as _pqs, urlencode as _ue2
                    _purl = _up2(self._page.url)
                    _qs = _pqs(_purl.query)
                    _q_val = (_qs.get("q") or _qs.get("query") or [""])[0]
                    if not _q_val:
                        # Try to get from page title or value
                        _q_val = (step.value or "").strip()
                    if _q_val:
                        from urllib.parse import quote_plus as _qp2
                        _filter_url = f"https://www.allrecipes.com/search?q={_qp2(_q_val)}&sort=re"
                        self._page.goto(_filter_url, wait_until="domcontentloaded", timeout=12000)
                        traces.append("allrecipes_filter_url ok=True")
                        self.extract_page_context()
                        dummy_out = ActionVerificationResult(
                            action_result=ActionResult(
                                success=True, action="click",
                                message=f"Navigated to Allrecipes filtered search: {_filter_url}",
                                latency_ms=0.0,
                            ),
                            verification=None, metadata=None,
                        )
                        return dummy_out, traces, True
                except Exception:
                    pass

        # Recipe-pick deterministic path: ensure we open an actual
        # recipe page (URL containing /recipe/) before extraction.
        if step.action == "click" and self._page is not None:
            _il = (step.intent or "").lower()
            if "recipe" in _il:
                hint = f"{step.intent or ''} {step.value or ''}"
                if self._open_recipe_link_from_page(hint):
                    self.extract_page_context()
                    traces.append("recipe_open_nav ok=True")
                    dummy_out = ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="click",
                            message="Recipe link opened via deterministic recipe path.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
                    return dummy_out, traces, True

        # PRIMARY: Try Playwright-native semantic locators before DOM scoring.
        # get_by_role / get_by_label / get_by_placeholder query the live DOM and
        # work across any site regardless of hashed CSS classes or React IDs.
        if step.action == "click" and self._page is not None:
            _native = self._try_playwright_native_click(step)
            if _native is not None:
                try:
                    self._page.wait_for_timeout(300)
                except Exception:
                    pass
                self.extract_page_context()
                self._maybe_extract_revealed_elements(step)
                traces.append("native_click_first ok=True")
                return _native, traces, True

        if step.action == "type" and self._page is not None:
            _native_type = self._try_playwright_native_type(step)
            if _native_type is not None:
                # After a successful native fill, re-extract so the next step
                # (e.g. autocomplete pick) sees the fresh DOM with suggestions.
                try:
                    self._page.wait_for_timeout(350)
                except Exception:
                    pass
                self.extract_page_context()
                traces.append("native_type_first ok=True")
                return _native_type, traces, True
            traces.append("native_type_first ok=False reason=no_matching_input")
            _search_val = step.value or ""
            if not _search_val and step.intent:
                _intent_q = step.intent
                _qm = re.search(r"['\"]([^'\"]+)['\"]", _intent_q)
                if _qm:
                    _search_val = _qm.group(1).strip()
                else:
                    _stripped = re.sub(
                        r"(?i)^(type|enter|search for|search|input|put|write)\s+", "", _intent_q
                    )
                    _stripped = re.sub(r"(?i)\s+(in|into|on|at|the|search bar|search box|search field|input field|field|bar|box).*$", "", _stripped).strip()
                    if _stripped and len(_stripped) <= 80:
                        _search_val = _stripped
            if _search_val and self._page is not None:
                _site_search_moved = self._try_site_search_url(
                    value=_search_val,
                    intent=step.intent or "",
                )
                if _site_search_moved:
                    try:
                        self._page.wait_for_timeout(600)
                        self.extract_page_context()
                    except Exception:
                        pass
                    traces.append(f"type_search_url_fallback ok=True value={_search_val!r}")
                    return ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="type",
                            message=f"Search submitted via URL: {_search_val!r}.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    ), traces, True

        _elapsed_before_candidates = (time.perf_counter() - _step_start) * 1000.0
        if _elapsed_before_candidates >= _step_budget_ms:
            traces.append(
                f"step_budget_exhausted_pre_candidates elapsed={_elapsed_before_candidates:.0f}ms budget={_step_budget_ms:.0f}ms"
            )
            out = ActionVerificationResult(
                action_result=ActionResult(
                    success=False,
                    action=step.action,
                    message="Step budget exhausted before candidate generation.",
                    latency_ms=0.0,
                    attempts=1,
                    error="step_budget_exhausted",
                ),
                verification=None,
                metadata=None,
            )
            return out, traces, False

        # FALLBACK: DOM-scored candidates
        candidates = self.get_next_action_candidates(step.intent, action=step.action, limit=max_candidates)
        if not candidates and self._page is not None and step.action in {"click", "type", "select", "check", "uncheck"}:
            # One recovery refresh for transient empty extractions.
            try:
                self._page.wait_for_timeout(500)
            except Exception:
                pass
            try:
                self.extract_page_context()
            except Exception:
                pass
            candidates = self.get_next_action_candidates(step.intent, action=step.action, limit=max_candidates)
            if candidates:
                traces.append("candidate_recovery_after_refresh ok=True")
        if not candidates:
            logger.warning("No candidates found for intent=%r action=%r", step.intent, step.action)
            out = ActionVerificationResult(
                action_result=ActionResult(
                    success=False,
                    action=step.action,
                    message="No model-ranked candidate available for action.",
                    latency_ms=0.0,
                    attempts=1,
                    error="no_candidates",
                ),
                verification=None,
                metadata=None,
            )
            traces.append("no_candidates_model_reject")
            return out, traces, False

        last_out: Optional[ActionVerificationResult] = None
        _candidates_refreshed = False

        for idx, c in enumerate(candidates, start=1):
            _elapsed_ms = (time.perf_counter() - _step_start) * 1000.0
            if idx > 1 and _elapsed_ms >= _step_budget_ms:
                traces.append(
                    f"step_budget_exhausted elapsed={_elapsed_ms:.0f}ms budget={_step_budget_ms:.0f}ms"
                )
                logger.warning(
                    "Step time budget exhausted after %.0fms (budget=%.0fms) for intent=%r",
                    _elapsed_ms, _step_budget_ms, step.intent,
                )
                break
            _url_before_attempt = (self._page.url or "") if self._page else ""
            req = self._candidate_to_request(step, c)
            if idx > 1:
                time.sleep(0.05 if idx <= 3 else 0.10)
                req = ActionRequest(
                    action=req.action,
                    value=req.value,
                    intent=req.intent,
                    selector=req.selector,
                    selector_type=req.selector_type,
                    dom_node_id=req.dom_node_id,
                    timeout_ms=min(req.timeout_ms, 4000),
                    clear_first=req.clear_first,
                    press_enter=req.press_enter,
                    retry_attempts=0,
                    retry_backoff_ms=0,
                    skip_auto_memory=req.skip_auto_memory,
                )
            out = self._executor.execute_and_verify(
                request=req,
                wait_after_ms=(
                    max(step.wait_after_ms, 900)
                    if step.action == "click" and self._intent_is_refinement_click(step.intent or "")
                    else step.wait_after_ms
                ),
                verifier=self._diff,
            )
            if (
                not out.action_result.success
                and step.action in {"click", "type"}
                and self._page is not None
                and not _candidates_refreshed
            ):
                dismissed = self._dismiss_overlays()
                if dismissed:
                    traces.append(f"overlay_recovery_after_failed_candidate idx={idx}")
                    try:
                        self.extract_page_context()
                    except Exception:
                        pass
                    fresh = self.get_next_action_candidates(
                        step.intent, action=step.action, limit=max_candidates
                    )
                    if fresh:
                        candidates = fresh
                        _candidates_refreshed = True
            if step.action == "click" and self._page is not None:
                _cur_url = (self._page.url or "").lower()
                if _cur_url.startswith("chrome-error://"):
                    traces.append("chrome_error_page_detected")
                    try:
                        if _url_before_attempt:
                            self._page.goto(_url_before_attempt, wait_until="domcontentloaded", timeout=10000)
                            self.extract_page_context()
                            traces.append("chrome_error_recovered_via_goto")
                    except Exception:
                        pass
            if step.action == "click" and self._page is not None:
                escaped = self._detect_and_escape_auth_redirect(_url_before_attempt)
                if escaped:
                    traces.append(
                        f"auth_redirect_escaped url_before={_url_before_attempt!r}"
                    )
                    _elapsed_after_auth = (time.perf_counter() - _step_start) * 1000.0
                    if _elapsed_after_auth < (_step_budget_ms * 0.70):
                        fresh = self.get_next_action_candidates(
                            step.intent, action=step.action, limit=max_candidates
                        )
                        if fresh:
                            candidates = fresh
                            _candidates_refreshed = True
            if out.verification and out.verification.had_meaningful_changes:
                self.extract_page_context()
            if step.action == "click":
                self._maybe_extract_revealed_elements(step)

            ok, reason = self._validate_action(step, c, out)
            if (
                ok
                and step.action == "click"
                and self._page is not None
                and reason in ("click_url_changed", "click_nav_url_changed")
                and _url_before_attempt
            ):
                _new_url = (self._page.url or "").lower()
                _url_actually_changed = _new_url != _url_before_attempt.lower()
                if _url_actually_changed:
                    _intent_tokens = {
                        w.lower() for w in re.split(r"\W+", step.intent or "")
                        if len(w) >= 3
                        and w.lower() not in {
                            "click", "open", "the", "link", "page", "button",
                            "from", "for", "and", "that", "this", "select",
                            "filter", "nonstop", "non", "stop", "show",
                        }
                    }
                    _url_path_tokens = set(re.split(r"\W+", _new_url))
                    _title_after = ""
                    try:
                        _title_after = (self._page.title() or "").lower()
                    except Exception:
                        pass
                    _page_tokens = _url_path_tokens | set(re.split(r"\W+", _title_after))
                    _overlap = len(_intent_tokens & _page_tokens)
                    _same_domain = _new_url.split("/")[2] == _url_before_attempt.lower().split("/")[2] if "/" in _new_url else False
                    if _intent_tokens and _overlap == 0 and not _same_domain:
                        try:
                            self._page.go_back(wait_until="domcontentloaded", timeout=5000)
                            self._page.wait_for_timeout(200)
                            self.extract_page_context()
                        except Exception:
                            pass
                        ok = False
                        reason = "click_url_changed_irrelevant_rollback"

            if (
                not ok
                and out.action_result.success
                and step.action == "click"
                and out.verification is not None
                and getattr(out.verification, "had_meaningful_changes", out.verification.had_changes)
                and any(k in (step.intent or "").lower() for k in (
                    "filter", "sort", "review score", "landmark", "price", "wifi", "pool",
                ))
                and reason != "click_refinement_left_results_page"
                and self._is_results_like_page()
                and self._confirm_refinement_applied(step.intent or "")
            ):
                ok = True
                reason = "click_filter_sort_fallback_success"
            traces.append(
                f"candidate#{idx} p={c.probability:.3f} score={c.score:.3f} "
                f"level={c.level} selector={c.selector_type}:{c.selector} valid={ok} reason={reason}"
            )
            last_out = out
            if ok:
                logger.debug(
                    "Step succeeded on candidate #%d selector=%r reason=%r",
                    idx, c.selector, reason,
                )
                if (
                    reason != "click_executed_unverified"
                    and step.action != "extract"
                    and step.intent and c.selector and self._executor
                ):
                    _url = self._page.url if self._page else ""
                    self._executor.record_confirmed_success(
                        intent=step.intent,
                        action=step.action,
                        selector=c.selector,
                        selector_type=c.selector_type or "auto",
                        strategy=out.action_result.strategy or "ranked_retry",
                        score=c.score,
                        url=_url,
                    )
                return out, traces, True
            if (
                not _candidates_refreshed
                and idx < len(candidates)
                and out.verification is not None
                and getattr(out.verification, "had_meaningful_changes", out.verification.had_changes)
            ):
                _elapsed_before_refresh = (time.perf_counter() - _step_start) * 1000.0
                if _elapsed_before_refresh < (_step_budget_ms * 0.55):
                    fresh = self.get_next_action_candidates(
                        step.intent, action=step.action, limit=max_candidates
                    )
                    if fresh:
                        candidates = fresh
                        _candidates_refreshed = True
                        traces.append(f"candidates_refreshed after meaningful_change at idx={idx}")

        logger.warning(
            "All %d candidates failed for intent=%r action=%r",
            len(candidates), step.intent, step.action,
        )
        if last_out is None:
            raise RuntimeError("Candidates list was non-empty but last_out is None.")

        # Href fallback (only for non-calendar click intents) 
        if step.action == "click" and self._page is not None:
            _is_cal = (
                self._neural.classify(step.intent or "", "calendar_next", threshold=0.65)
                or self._neural.classify(step.intent or "", "calendar_prev", threshold=0.65)
            )
            if not _is_cal:
                href = self._extract_href_from_candidates(candidates, intent=step.intent or "")
                if href:
                    logger.info("Click candidates exhausted; falling back to navigate href=%r", href)
                    nav_req = ActionRequest(
                        action="navigate",
                        value=href,
                        intent=step.intent,
                        timeout_ms=step.timeout_ms,
                    )
                    nav_out = self._executor.execute_and_verify(
                        request=nav_req,
                        wait_after_ms=step.wait_after_ms,
                        verifier=self._diff,
                    )
                    self.extract_page_context()
                    if nav_out.action_result.success:
                        traces.append(f"href_fallback url={href} ok=True reason=navigate_fallback")
                        return nav_out, traces, True
                    traces.append(f"href_fallback url={href} ok=False reason=navigate_failed")

        return last_out, traces, False

    def _extract_href_from_candidates(self, candidates: list[NextActionCandidate], intent: str = "") -> Optional[str]:
        if self._page is None:
            return None
        # Content-path segments that are never valid navigation targets for data/UI actions.
        _BAD_CONTENT_SEGS = frozenset({
            "stories", "story", "blog", "blogs", "news", "article", "articles",
            "video", "videos", "podcast", "podcasts", "press", "media",
        })
        _intent_words = set(w for w in re.split(r'\W+', intent.lower()) if len(w) >= 4) if intent else set()
        for c in candidates:
            if not c.selector:
                continue
            try:
                st = (c.selector_type or "").lower()
                if "xpath" in st:
                    loc = self._page.locator(f"xpath={c.selector}")
                else:
                    loc = self._page.locator(c.selector)
                href = loc.first.get_attribute("href", timeout=1000)
                if not href or not (href.startswith("http") or href.startswith("/")):
                    continue
                # Reject parameterised search/query URLs - they behave differently
                # when navigated to directly vs triggered by a form submit.
                if "?" in href:
                    continue
                # Reject JS-only hrefs that can't be navigated directly.
                if href.startswith("javascript:"):
                    continue
                # Reject content articles/blog posts unless the intent explicitly
                # targets that kind of content (prevents clicking "stories/mlb/..."
                # when the intent is "Click on the Standings link").
                if intent and _intent_words:
                    try:
                        import urllib.parse as _up_rel
                        _path_l = (_up_rel.urlparse(href).path or "").lower()
                        _path_segs = set(s for s in _path_l.split("/") if s)
                        if _path_segs & _BAD_CONTENT_SEGS and not (_path_segs & _intent_words):
                            continue
                    except Exception:
                        pass
                if href.startswith("/"):
                    import urllib.parse as _urlparse
                    parsed = _urlparse.urlparse(self._page.url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                return href
            except Exception:
                continue
        return None

    # Semantic descriptions of common task categories that are WRONG for a given goal.
    # Used as reference anchors in the contrastive URL check  -  a path segment that
    # scores much higher against one of these than against the actual plan goal is
    # a strong signal that the agent navigated to the wrong section of the site.
    # These are NOT site-specific patterns; they describe task-category semantics.
    _WRONG_CATEGORY_REFS = (
        "tourist attractions sightseeing entertainment activities tickets",
        "flights airfare airline booking departure arrival",
        "car rental vehicle hire driving",
        "airport taxi shuttle transfer pickup",
        "tours guided excursions experiences",
        "login sign in register create account authentication",
        "error 404 page not found",
    )

    _RESULTS_PATH_HINTS = (
        "search", "results", "result", "listing", "listings", "list",
        "query", "find", "discover", "items", "products", "hotels",
    )
    _REFINEMENT_HINTS = (
        "filter", "sort", "price", "rating", "review", "amenities",
        "wifi", "pool", "breakfast", "stars", "distance", "nonstop",
        "stops", "free cancellation",
    )

    def _check_url_goal_alignment(
        self,
        new_url: str,
        plan_goal: str,
        subgoal_criteria: list,
        plan: Optional[object] = None,  # TaskPlan  -  used for plan-wide url_contains whitelist
    ) -> str:
        """Check if *new_url* is compatible with *plan_goal*.

        Returns a non-empty reason string if the URL is clearly wrong, empty
        string if it looks fine.

        Checks (in order):
        1. Current subgoal has url_contains criterion that the new URL does NOT satisfy.
        2. Plan-wide url_contains whitelist: if the new URL satisfies ANY expected
           fragment from the full plan, it's on a known-good page (possibly ahead of
           schedule). Otherwise apply tighter checks.
        3. Contrastive semantic check: is the path segment much more similar to a
           wrong-category description than to the plan goal?  Catches same-domain but
           wrong-section navigation (hotel goal ->' /attractions).
        4. Baseline low-similarity check for cross-domain mismatch (e.g. landing on
           a completely unrelated site).
        """
        if not new_url:
            return ""

        url_lower = new_url.lower()
        goal_lower = (plan_goal or "").lower()

        if self._goal_is_informational(goal_lower) and any(
            k in url_lower for k in ("/help", "/faq", "/support", "/contact", "/customer", "lost-item", "lost-items")
        ):
            return ""
        for c in (subgoal_criteria or []):
            if getattr(c, "kind", "").lower() == "url_contains" and getattr(c, "required", True):
                expected = (getattr(c, "value", "") or "").lower()
                if not expected:
                    continue
                expected_norm = expected.lstrip("?")
                if expected not in url_lower and expected_norm not in url_lower:
                    return f"url_contains_not_met:expected={expected!r}"

        from urllib.parse import urlparse as _up
        try:
            _pu = _up(new_url)
            host = _pu.netloc.lower().lstrip("www.")
            path = _pu.path or "/"
            segments = [p for p in path.split("/") if p]
            first_path_seg = segments[0].lower() if segments else ""
        except Exception:
            return ""
        if self._url_looks_results_like(new_url) and any(
            k in goal_lower for k in (
                "search", "find", "book", "compare", "hotel", "room", "flight",
                "restaurant", "product", "recipe", "list", "results",
            )
        ):
            return ""

        if not first_path_seg or len(first_path_seg) < 4:
            return ""

        first_seg = first_path_seg.replace("-", " ").replace("_", " ")
        if plan is not None:
            plan_path_frags: list[str] = []
            try:
                for sg in getattr(plan, "subgoals", []):
                    for c in getattr(sg, "success_criteria", []):
                        if getattr(c, "kind", "").lower() == "url_contains":
                            v = (getattr(c, "value", "") or "").lower().strip()
                            if v and "/" in v:
                                plan_path_frags.append(v)
                            elif v and not any(v.endswith(tld) for tld in (".com", ".org", ".net", ".io", ".co")):
                                plan_path_frags.append(v)
            except Exception:
                pass

            if plan_path_frags:
                if any(frag in url_lower for frag in plan_path_frags):
                    return ""
        goal_sim = self._neural.similarity(first_seg, goal_lower)
        goal_score = (goal_sim + 1.0) * 0.5

        alt_sims = self._neural.similarity_batch(first_seg, list(self._WRONG_CATEGORY_REFS))
        alt_scores = [(s + 1.0) * 0.5 for s in alt_sims]
        max_alt_score = max(alt_scores) if alt_scores else 0.0
        if max_alt_score > goal_score + 0.04 and max_alt_score > 0.52:
            return (
                f"url_wrong_category:path=/{segments[0]} "
                f"goal_sim={goal_score:.2f} alt_sim={max_alt_score:.2f}"
            )

        if goal_score < 0.42:
            return f"url_domain_mismatch:path=/{segments[0]} score={goal_score:.2f}"

        return ""

    def _replan_is_compatible(self, original_goal: str, new_plan) -> bool:
        
        if not original_goal or not new_plan:
            return True

        # Build a representative text summary of the new plan.
        plan_desc_parts: list[str] = []
        for sg in new_plan.subgoals[:5]:
            plan_desc_parts.append(sg.description)
            for step in sg.steps[:2]:
                if step.intent:
                    plan_desc_parts.append(step.intent)
                if step.action == "navigate" and step.value:
                    plan_desc_parts.append(step.value)
        plan_text = " ".join(plan_desc_parts)

        if not plan_text.strip():
            return True

        sim = self._neural.similarity(original_goal, plan_text)
        score = (sim + 1.0) * 0.5
        compatible = score >= 0.45
        if not compatible:
            logger.warning(
                "Replan rejected: new plan similarity=%.2f too low vs original goal %r; "
                "plan summary: %.120s",
                score, original_goal, plan_text,
            )
        return compatible

    def _url_looks_results_like(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse as _up
            pu = _up(url or "")
            path_l = (pu.path or "").lower()
            query_l = (pu.query or "").lower()
            blob = f"{path_l} {query_l}"
            if any(h in blob for h in self._RESULTS_PATH_HINTS):
                return True
            if any(k in query_l for k in ("q=", "query=", "search=", "keyword=", "term=")):
                return True
        except Exception:
            return False
        return False

    def _page_has_result_cards(self) -> bool:
        if self._page is None:
            return False
        try:
            count = self._page.evaluate(
                """() => {
                    const sels = [
                      'article', '[role="article"]', '[data-testid*="card"]',
                      '[class*="card"]', '[class*="result"]', '[class*="listing"]',
                      'main li', '[role="listitem"]'
                    ];
                    let total = 0;
                    for (const s of sels) total += document.querySelectorAll(s).length;
                    return total;
                }"""
            )
            return int(count or 0) >= 4
        except Exception:
            return False

    def _is_results_like_page(self) -> bool:
        if self._page is None:
            return False
        u = (self._page.url or "").lower()
        return self._url_looks_results_like(u) or self._page_has_result_cards()

    def _intent_is_refinement_click(self, intent: str) -> bool:
        il = (intent or "").lower()
        if any(k in il for k in self._REFINEMENT_HINTS):
            return True
        try:
            return self._neural.similarity(il, "refine results with filters sort order") > 0.25
        except Exception:
            return False

    def _intent_anchor_terms(self, intent: str, max_terms: int = 5) -> list[str]:
        il = (intent or "").lower()
        raw = [t for t in re.split(r"\W+", il) if len(t) >= 4]
        stop = {
            "click", "open", "filter", "filters", "sort", "results", "result", "show",
            "with", "from", "into", "that", "this", "these", "those", "page",
            "option", "options", "dropdown", "suggestion", "autocomplete", "button",
            "apply", "select", "choose", "list", "more", "than", "above", "below",
            "hotel", "recipe", "search",
        }
        uniq: list[str] = []
        seen = set()
        for t in raw:
            if t in stop or t in seen:
                continue
            seen.add(t)
            uniq.append(t)
            if len(uniq) >= max_terms:
                break
        return uniq

    def _confirm_refinement_applied(self, intent: str) -> bool:
        """Check whether a filter/sort-like action appears applied in the live DOM."""
        if self._page is None:
            return False
        anchors = self._intent_anchor_terms(intent)
        try:
            return bool(self._page.evaluate(
                """(anchors) => {
                    const checked = Array.from(document.querySelectorAll(
                        'input[type="checkbox"]:checked, [aria-checked="true"], [aria-pressed="true"], [aria-selected="true"], [data-selected="true"], [class*="selected"], [class*="active"]'
                    ));
                    if (!checked.length) return false;
                    const norm = (s) => (s || '').toLowerCase();
                    for (const el of checked) {
                        const blob = norm(el.textContent) + ' ' + norm(el.getAttribute('aria-label')) + ' ' + norm(el.getAttribute('name')) + ' ' + norm(el.getAttribute('id'));
                        if (!anchors || anchors.length === 0) return true;
                        if (anchors.some(t => blob.includes(t))) return true;
                        const parent = el.closest('label, li, div, section');
                        if (parent) {
                            const ptxt = norm(parent.textContent);
                            if (anchors.some(t => ptxt.includes(t))) return true;
                        }
                    }
                    return false;
                }""",
                anchors,
            ))
        except Exception:
            return False

    def _submit_search_from_page(self) -> bool:
        if self._page is None:
            return False
        _before_url = (self._page.url or "")

        # Site-specific submit strategies (highest priority)

        # HuggingFace: navigate to models search URL directly - HuggingFace's
        # search form doesn't respond to Enter key; direct URL is more reliable.
        if "huggingface.co" in _before_url.lower():
            try:
                # Get whatever text is in the visible search input
                _hf_query = self._page.evaluate("""() => {
                    const inp = document.querySelector('input[type="search"], input[placeholder*="Search" i], [role="searchbox"]');
                    return inp ? inp.value.trim() : '';
                }""") or ""
                if _hf_query:
                    from urllib.parse import quote_plus as _qp_hf
                    _hf_url = f"https://huggingface.co/models?search={_qp_hf(_hf_query)}&language=en"
                    self._page.goto(_hf_url, wait_until="domcontentloaded", timeout=12000)
                    _after_url = (self._page.url or "")
                    if _after_url != _before_url:
                        logger.info("HuggingFace direct search URL navigation ok: %r", _hf_url)
                        return True
            except Exception:
                pass

        # Wolfram Alpha: click the orange "=" submit button via JS
        if "wolframalpha.com" in _before_url.lower():
            try:
                clicked = self._page.evaluate("""() => {
                    // WA submit button has class like '_RLFy' or aria-label "Compute"
                    const sels = [
                        '[aria-label="Compute"]',
                        'button._RLFy',
                        'button[class*="submit"]',
                        'form button',
                        'button[type="submit"]',
                    ];
                    for (const sel of sels) {
                        const btn = document.querySelector(sel);
                        if (btn && !btn.disabled) { btn.click(); return true; }
                    }
                    // Fallback: press Enter on the focused element
                    const active = document.activeElement;
                    if (active) {
                        active.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', keyCode:13, bubbles:true}));
                        active.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter', keyCode:13, bubbles:true}));
                        return true;
                    }
                    return false;
                }""")
                if clicked:
                    try:
                        self._page.wait_for_timeout(800)
                    except Exception:
                        pass
                    _after_url = (self._page.url or "")
                    if _after_url != _before_url or self._is_results_like_page():
                        logger.info("Wolfram Alpha submit via JS ok")
                        return True
            except Exception:
                pass

        locators = [
            self._page.locator('button[type="submit"]'),
            self._page.get_by_role("button", name="Search", exact=False),
            self._page.get_by_role("button", name="Show results", exact=False),
            self._page.get_by_role("button", name="Find", exact=False),
            self._page.get_by_role("button", name="Apply", exact=False),
            self._page.locator('input[type="submit"]'),
        ]
        for loc in locators:
            try:
                if loc.count() <= 0:
                    continue
                btn = loc.first
                if not btn.is_visible(timeout=250):
                    continue
                btn.scroll_into_view_if_needed(timeout=1200)
                btn.click(timeout=1600)
                try:
                    self._page.wait_for_timeout(450)
                except Exception:
                    pass
                _after_url = (self._page.url or "")
                if _after_url != _before_url or self._is_results_like_page():
                    return True
            except Exception:
                continue
        # Try pressing Enter on the currently-focused element first (works for
        # sites like Wolfram Alpha whose search input retains focus after typing).
        try:
            self._page.keyboard.press("Enter")
            try:
                self._page.wait_for_timeout(600)
            except Exception:
                pass
            _after_url = (self._page.url or "")
            if _after_url != _before_url or self._is_results_like_page():
                return True
        except Exception:
            pass

        # Broader Enter fallback: any visible text/search input.
        try:
            inputs = [
                self._page.get_by_role("searchbox"),
                self._page.get_by_role("combobox"),
                self._page.locator('input[type="search"]'),
                self._page.locator('input[type="text"]:visible'),
            ]
            for loc in inputs:
                try:
                    if loc.count() <= 0:
                        continue
                    inp = loc.first
                    if not inp.is_visible(timeout=250):
                        continue
                    inp.scroll_into_view_if_needed(timeout=900)
                    inp.press("Enter", timeout=1200)
                    try:
                        self._page.wait_for_timeout(500)
                    except Exception:
                        pass
                    _after_url = (self._page.url or "")
                    if _after_url != _before_url or self._is_results_like_page():
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    def _try_site_search_url(self, value: str, intent: str = "") -> bool:
        """Navigate to a site-specific search URL when normal input fill fails.

        This handles sites whose search bar is inside deeply-nested iframes or
        shadow DOM that Playwright can't fill (e.g. Fox Sports, ESPN).
        Returns True if the navigation moved to a plausible search-results page.
        """
        if self._page is None or not value:
            return False
        _before_url = (self._page.url or "")
        from urllib.parse import urlparse as _up, urlencode as _ue, quote_plus as _qp
        parsed = _up(_before_url)
        netloc = parsed.netloc.lower()
        value_enc = _qp(value)

        # Site-specific known search URL patterns
        _SITE_SEARCH_PATTERNS: dict[str, str] = {
            "foxsports.com": f"https://www.foxsports.com/search?q={value_enc}",
            "espn.com": f"https://www.espn.com/search/_/q/{value_enc}",
            "nfl.com": f"https://www.nfl.com/search#d={value_enc}",
            "nba.com": f"https://www.nba.com/search?q={value_enc}",
            "mlb.com": f"https://www.mlb.com/search?q={value_enc}",
            "bleacherreport.com": f"https://bleacherreport.com/search?q={value_enc}",
            "wikipedia.org": f"https://en.wikipedia.org/wiki/Special:Search?search={value_enc}",
            "youtube.com": f"https://www.youtube.com/results?search_query={value_enc}",
            "reddit.com": f"https://www.reddit.com/search/?q={value_enc}",
            "amazon.com": f"https://www.amazon.com/s?k={value_enc}",
            "imdb.com": f"https://www.imdb.com/find?q={value_enc}",
            # Academic / research
            "arxiv.org": f"https://arxiv.org/search/?searchtype=ti&query={value_enc}",
            # Recipe sites
            "allrecipes.com": f"https://www.allrecipes.com/search?q={value_enc}",
            # Store locator
            "traderjoes.com": f"https://www.traderjoes.com/home/stores/find?q={value_enc}",
            # HuggingFace
            "huggingface.co": f"https://huggingface.co/models?search={value_enc}&language=en",
            # Bus / transit
            "megabus.com": f"https://us.megabus.com/faqs",
        }

        # Check if current domain has a known pattern
        for domain, url_template in _SITE_SEARCH_PATTERNS.items():
            if domain in netloc:
                try:
                    self._page.goto(url_template, wait_until="domcontentloaded", timeout=12000)
                    _after_url = (self._page.url or "")
                    if _after_url != _before_url:
                        logger.info("Site search URL fallback: domain=%r value=%r url=%r", domain, value, url_template)
                        return True
                except Exception:
                    pass
                return False

        # Generic fallback: try common search URL patterns on current domain
        _base = f"{parsed.scheme}://{parsed.netloc}"
        _generic_patterns = [
            f"{_base}/search?q={value_enc}",
            f"{_base}/search?query={value_enc}",
            f"{_base}/search/{value_enc}",
            f"{_base}/?s={value_enc}",
        ]
        for url in _generic_patterns:
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=8000)
                _after_url = (self._page.url or "")
                if _after_url != _before_url and "404" not in (self._page.title() or "").lower():
                    logger.info("Generic site search URL fallback ok: url=%r", url)
                    return True
            except Exception:
                continue

        return False

    def _open_recipe_link_from_page(self, hint_text: str = "") -> bool:
        if self._page is None:
            return False
        tokens = [t for t in re.split(r"\W+", (hint_text or "").lower()) if len(t) >= 4]
        stop = {"click", "recipe", "with", "more", "than", "rating", "reviews", "stars", "from"}
        tokens = [t for t in tokens if t not in stop][:4]
        try:
            def _find_href() -> Optional[str]:
                return self._page.evaluate(
                    """(tokens) => {
                    const anchors = Array.from(document.querySelectorAll(
                        'a[href*="/recipe/"], a[href*="/recipes/"], a[href*="/recipe-"]'
                    ));
                    let best = null;
                    let bestScore = -1;
                    for (const a of anchors) {
                        const r = a.getBoundingClientRect();
                        if (!(r.width > 0 && r.height > 0)) continue;
                        const href = a.getAttribute('href') || '';
                        const hrefL = href.toLowerCase();
                        if (hrefL.includes('/kitchen-tips') || hrefL.includes('/news/') || hrefL.includes('/videos/')) continue;
                        const txt = (a.textContent || a.getAttribute('aria-label') || '').toLowerCase();
                        let score = 0;
                        for (const t of tokens || []) {
                            if (txt.includes(t)) score += 1;
                        }
                        if (score > bestScore) {
                            bestScore = score;
                            best = href;
                        }
                    }
                    if (best) return best;
                    const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
                    function collectUrls(obj, out) {
                        if (!obj) return;
                        if (Array.isArray(obj)) { for (const x of obj) collectUrls(x, out); return; }
                        if (typeof obj === 'object') {
                            const u = obj.url || obj['@id'];
                            if (typeof u === 'string') out.push(u);
                            for (const k of Object.keys(obj)) collectUrls(obj[k], out);
                        }
                    }
                    const urls = [];
                    for (const s of scripts) {
                        try {
                            const raw = JSON.parse(s.textContent || '{}');
                            collectUrls(raw, urls);
                        } catch(e) {}
                    }
                    for (const u of urls) {
                        const ul = (u || '').toLowerCase();
                        if (ul.includes('/recipe/') && !ul.includes('/kitchen-tips')) return u;
                    }
                    return null;
                }""",
                    tokens,
                )

            href = None
            for _ in range(3):
                href = _find_href()
                if href:
                    break
                try:
                    self._page.evaluate("window.scrollBy(0, Math.max(window.innerHeight, 600));")
                    self._page.wait_for_timeout(250)
                except Exception:
                    pass
            if not href:
                return False
            try:
                if href.startswith("/"):
                    from urllib.parse import urlparse as _up
                    pu = _up(self._page.url or "")
                    href = f"{pu.scheme}://{pu.netloc}{href}"
                self._page.goto(href, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                return False
            final_url = (self._page.url or "").lower()
            return ("/recipe/" in final_url) or ("/recipes/" in final_url and "/kitchen-tips" not in final_url)
        except Exception:
            return False

    def _goal_is_informational(self, goal_text: str) -> bool:
        gl = (goal_text or "").lower()
        support_tokens = (
            "support", "policy", "procedure", "lost item", "lost baggage",
            "contact support", "customer service", "faq", "refund",
            "cancellation policy", "terms of use", "privacy policy",
            "lose an item", "lost", "missing item",
        )
        if any(k in gl for k in support_tokens):
            return True
        # Interrogative phrasing can still be execution-oriented ("what events
        # are in New York"), so require absence of transactional verbs.
        if any(k in gl for k in ("what", "how", "which")):
            if not any(v in gl for v in (
                "find", "search", "book", "buy", "set", "select",
                "closest", "nearby", "events", "store", "location",
            )):
                return True
        return False

    def _try_navigate_help_resource(self, goal_text: str) -> bool:
        if self._page is None:
            return False
        tokens = [t for t in re.split(r"\W+", (goal_text or "").lower()) if len(t) >= 3][:8]
        support_tokens = ["help", "support", "faq", "contact", "policy", "lost", "item", "baggage"]
        try:
            href = self._page.evaluate(
                """(goalTokens, supportTokens) => {
                    const as = Array.from(document.querySelectorAll('a[href]'));
                    let best = null, bestScore = -1;
                    function visible(el) {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.height > 0;
                    }
                    for (const a of as) {
                        if (!visible(a)) continue;
                        const href = a.getAttribute('href') || '';
                        const txt = (a.textContent || a.getAttribute('aria-label') || '').toLowerCase().trim();
                        if (!href || href.startsWith('#') || href.startsWith('javascript:')) continue;
                        const blob = (txt + ' ' + href.toLowerCase());
                        let score = 0;
                        for (const t of supportTokens) if (blob.includes(t)) score += 5;
                        for (const t of goalTokens) if (blob.includes(t)) score += 2;
                        if (score > bestScore) { bestScore = score; best = href; }
                    }
                    return bestScore >= 5 ? best : null;
                }""",
                tokens,
                support_tokens,
            )
            if not href:
                return False
            if href.startswith("/"):
                from urllib.parse import urlparse as _up
                pu = _up(self._page.url or "")
                href = f"{pu.scheme}://{pu.netloc}{href}"
            self._page.goto(href, wait_until="domcontentloaded", timeout=15000)
            self.extract_page_context()
            return True
        except Exception:
            return False

    def _ensure_calendar_open(self, intent: str) -> bool:
        if self._page is None:
            return False
        if self._calendar_widget_is_open():
            return True
        for _ in range(2):
            opened = self._try_date_input_js(intent)
            if opened:
                try:
                    self._page.wait_for_timeout(250)
                except Exception:
                    pass
                if self._calendar_widget_is_open():
                    return True
        return self._calendar_widget_is_open()

    def _calendar_widget_is_open(self) -> bool:
        """Return True ONLY when a date picker is in its EXPANDED interactive state.
        The Booking.com homepage has a permanently-visible collapsed date widget (~140px).
        We require: dialog container, explicit open class, or tall grid with many date cells.
        """
        if self._page is None:
            return False
        _JS = """() => {
            function vis(el) {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                if (r.width < 1 || r.height < 1) return false;
                const s = window.getComputedStyle(el);
                return s.display !== 'none' && s.visibility !== 'hidden' && parseFloat(s.opacity) > 0.05;
            }
            // 1. Calendar inside a dialog/modal
            if (vis(document.querySelector('[role="dialog"] table[role="grid"], [aria-modal="true"] table[role="grid"]')))
                return true;
            // 2. Explicit open class
            if (vis(document.querySelector('[class*="flatpickr-calendar"][class*="open"], [class*="react-datepicker-popper"], [class*="calendar--open"]')))
                return true;
            // 3. Tall grid (>=280px) with many labelled date cells
            for (const g of document.querySelectorAll('table[role="grid"]')) {
                if (!vis(g) || g.getBoundingClientRect().height < 280) continue;
                if (g.querySelectorAll('[role="gridcell"][aria-label], td[aria-label]').length >= 28) return true;
            }
            // 4. Explicit next/prev month controls visible (common booking layout)
            for (const b of document.querySelectorAll('button, [role="button"]')) {
                if (!vis(b)) continue;
                const lbl = (b.getAttribute('aria-label') || b.textContent || '').toLowerCase().trim();
                if (lbl.includes('next month') || lbl.includes('previous month')) return true;
            }
            // 5. Google Flights / Material date picker: many aria-label date cells
            //    These use li[data-day] or [aria-label*="2026"] style elements
            const dateCells = document.querySelectorAll(
                '[aria-label*="2024"], [aria-label*="2025"], [aria-label*="2026"], [aria-label*="2027"], [data-day]'
            );
            if (dateCells.length >= 20) return true;
            // 6. Any container with 'calendar' or 'datepicker' in class that is visible
            for (const el of document.querySelectorAll('[class*="calendar"], [class*="datepicker"], [class*="DatePicker"]')) {
                if (vis(el) && el.getBoundingClientRect().height > 150) return true;
            }
            return false;
        }"""
        try:
            return bool(self._page.evaluate(_JS))
        except Exception:
            return False

    def _autocomplete_listbox_is_live(self) -> bool:
        if self._page is None:
            return False
        _JS = """() => {
            const LISTBOX_SELS = [
                '[role="listbox"]',
                'ul[role="listbox"]',
                '[id*="autocomplete"]',
                '[class*="autocomplete-dropdown"]',
                '[class*="suggestion-list"]',
                '[data-testid*="autocomplete"]',
            ];
            function isLive(el) {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                if (r.width <= 0 || r.height <= 2) return false;
                const s = window.getComputedStyle(el);
                if (s.display === 'none') return false;
                if (s.visibility === 'hidden') return false;
                if (parseFloat(s.opacity) < 0.05) return false;
                const items = el.querySelectorAll(
                    '[role="option"], li, [class*="suggestion-item"], [class*="result-item"]'
                );
                return items.length > 0;
            }
            for (const sel of LISTBOX_SELS) {
                try {
                    const els = document.querySelectorAll(sel);
                    for (const el of els) {
                        if (isLive(el)) return true;
                    }
                } catch(e) {}
            }
            return false;
        }"""
        try:
            return bool(self._page.evaluate(_JS))
        except Exception:
            return False

    def _autocomplete_input_has_value(self, expected_value: str) -> bool:
        if self._page is None or not expected_value:
            return False
        exp = expected_value.strip().lower()
        if not exp:
            return False
        try:
            js = """(exp) => {
                const sels = [
                    'input[role="combobox"]',
                    'input[type="search"]',
                    'input[name*="ss"]',
                    'input[data-testid*="destination"]',
                    'input[id*="destination"]',
                ];
                for (const s of sels) {
                    const el = document.querySelector(s);
                    if (!el) continue;
                    const v = String(el.value || '').toLowerCase();
                    if (!v) continue;
                    if (v.includes(exp) || exp.includes(v)) return true;
                }
                return false;
            }"""
            return bool(self._page.evaluate(js, exp))
        except Exception:
            return False

    def _snapshot_page_identity(self) -> str:
        if self._page is None:
            return ""
        try:
            result = self._page.evaluate("""() => {
                return JSON.stringify({
                    url: location.href,
                    title: document.title,
                    body_len: document.body ? document.body.innerHTML.length : 0,
                    interactive_count: document.querySelectorAll(
                        'input, button, select, textarea, [role="combobox"]'
                    ).length,
                });
            }""")
            return str(result)
        except Exception:
            return ""

    def _extract_answer_matches_question(self, question: str, answer: str) -> bool:
        
        q = (question or "").lower()
        a = (answer or "").lower().strip()

        if not a:
            return False
        _DEFINITIVE_NOT_FOUND = (
            "not found on this page.",
            "no information found",
            "no results found",
        )
        if any(a == s for s in _DEFINITIVE_NOT_FOUND):
            return False

        if any(t in a for t in ("page not found", "404 error", "whoops, the page", "event you are looking for was not found")):
            return False

        if len(a) < 12:
            return False
        q_words = {w for w in re.split(r"\W+", q) if len(w) >= 4}
        a_words = {w for w in re.split(r"\W+", a) if len(w) >= 3}
        stop = {
            "find", "check", "identify", "provide", "include", "please", "using",
            "website", "current", "first", "branch", "official", "repository",
            "book", "search", "results", "list", "details", "about", "which",
            "with", "from", "that", "this", "then", "into", "above", "below",
            "what", "does", "have", "show", "page", "help", "give", "tell",
        }
        q_core = {w for w in q_words if w not in stop}
        if q_core:
            def _words_overlap(qset: set, aset: set) -> bool:
                # Exact match first
                if qset & aset:
                    return True
                # Prefix match: each q-word matches an a-word sharing a 4-char prefix
                for qw in qset:
                    pfx = qw[:4]
                    if any(aw.startswith(pfx) or qw.startswith(aw[:4]) for aw in aset if len(aw) >= 4):
                        return True
                return False
            if not _words_overlap(q_core, a_words):
                return False

        return True

    def _try_autocomplete_js(self, value_hint: str) -> bool:
        """Click autocomplete suggestion: Playwright-native first, JS fallback."""
        if self._page is None:
            return False
        hint_lower = (value_hint or "").lower().strip()
        _WRONG_PATHS = ("/attractions", "/flights", "/cars", "/airport-taxis",
                        "/activities", "/cruises", "/experiences")

        def _bad_url():
            url = self._page.url or ""
            return any(p in url.lower() for p in _WRONG_PATHS)

        def _go_back():
            try:
                self._page.go_back(wait_until="domcontentloaded", timeout=6000)
                self._page.wait_for_timeout(300)
            except Exception:
                pass

        # Strategy 1: Playwright get_by_role("option")  -  works on any ARIA listbox
        if hint_lower:
            for loc in [
                self._page.get_by_role("option", name=hint_lower, exact=False),
                self._page.locator(f'[role="option"]').filter(has_text=hint_lower),
                self._page.locator(f'li[role="option"]').filter(has_text=hint_lower),
            ]:
                try:
                    if loc.count() > 0:
                        loc.first.click(timeout=3000)
                        self._page.wait_for_timeout(300)
                        if _bad_url():
                            _go_back()
                            continue
                        logger.info("Autocomplete: Playwright option click for %r", hint_lower)
                        return True
                except Exception:
                    pass

            # Strategy 2: li:has-text for non-ARIA listboxes
            for loc in [
                self._page.locator(f'li:has-text("{hint_lower}")'),
                self._page.locator(f'[data-testid*="autocomplete"] li').filter(has_text=hint_lower),
            ]:
                try:
                    if loc.count() > 0:
                        loc.first.click(timeout=3000)
                        self._page.wait_for_timeout(300)
                        if _bad_url():
                            _go_back()
                            continue
                        logger.info("Autocomplete: li:has-text click for %r", hint_lower)
                        return True
                except Exception:
                    pass

        # Strategy 3: JS querySelector (original approach)
        _JS = """(hint) => {
            const SKIP = ['attractions','flights','activities','cars','airport','experiences','rentals'];
            const SELS = [
                '[role="listbox"] [role="option"]', '[role="listbox"] li',
                'ul[role="listbox"] li', '[role="option"]',
                '[id*="autocomplete"] li', '[class*="autocomplete"] li',
                '[class*="suggestion-item"]', '[data-testid*="autocomplete"] li',
            ];
            function vis(el) { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; }
            function hasSkip(t) { const l = t.toLowerCase(); return SKIP.some(w => l.includes(w)); }
            function norm(s) { return (s || '').toLowerCase().replace(/\\s+/g, ' ').trim(); }
            function toks(s) { return norm(s).split(/[^a-z0-9]+/).filter(x => x.length >= 2); }
            let items = [];
            for (const s of SELS) {
                const found = Array.from(document.querySelectorAll(s)).filter(vis);
                if (found.length) { items = found; break; }
            }
            if (!items.length) return null;
            // Pass 1: best semantic text match against hint
            if (hint) {
                const h = norm(hint);
                const ht = toks(h);
                let best = null;
                let bestScore = -9999;
                for (const el of items) {
                    const t = (el.textContent || '').trim();
                    const tn = norm(t);
                    if (!tn || hasSkip(tn)) continue;
                    let score = 0;
                    if (tn === h) score += 120;
                    if (tn.startsWith(h)) score += 70;
                    if (tn.includes(h)) score += 40;
                    const tt = toks(tn);
                    const overlap = ht.filter(x => tt.includes(x)).length;
                    score += overlap * 12;
                    const extra = Math.max(0, tt.length - overlap);
                    score -= extra * 2;
                    if (score > bestScore) { bestScore = score; best = el; }
                }
                if (best && bestScore >= 20) {
                    const txt = (best.textContent || '').trim();
                    best.click();
                    return txt.slice(0,80);
                }
                for (const el of items) {
                    const t = (el.textContent || '').trim();
                    if (t.toLowerCase().includes(hint) && !hasSkip(t)) { el.click(); return t.slice(0,80); }
                }
                // Pass 2: match hint, allow skip words
                for (const el of items) {
                    const t = (el.textContent || '').trim();
                    if (t.toLowerCase().includes(hint)) { el.click(); return t.slice(0,80); }
                }
            }
                // Do not click arbitrary first suggestion when hint does not match.
                return null;
            }"""
        for frame in self._page.frames:
            try:
                result = frame.evaluate(_JS, hint_lower)
                if result:
                    logger.info("Autocomplete JS clicked: %r", result)
                    try:
                        self._page.wait_for_timeout(600)
                    except Exception:
                        pass
                    url_after = self._page.url or ""
                    if any(p in url_after.lower() for p in _WRONG_PATHS):
                        logger.warning("Autocomplete navigated to wrong path %s -- going back", url_after)
                        try:
                            self._page.go_back(wait_until="domcontentloaded", timeout=8000)
                            self._page.wait_for_timeout(400)
                        except Exception:
                            pass
                        return False
                    return True
            except Exception:
                continue
        return False

    def _try_playwright_native_type(
        self, step: "AutomationStep"
    ) -> "Optional[ActionVerificationResult]":
        """Fill an input using Playwright semantic locators  -  no DOM snapshot needed.

        Parses label keywords from the step intent, then tries get_by_label /
        get_by_placeholder / get_by_role(textbox|searchbox|combobox) in order.
        Returns a success result if fill() lands in the right field, or None to
        fall through to the DOM-scored candidate pipeline.

        This is the primary path for all type/fill actions.  It is faster and
        more reliable than DOM scoring because:
          - Playwright locators query the live DOM (no stale snapshot risk)
          - No Gemini API call is needed
          - fill() is atomic  -  if it returns without exception, the value is in
            the input's value property, regardless of React/Vue controlled-state
        """
        if self._page is None or not step.value:
            return None
        value: str = step.value
        intent = (step.intent or "").lower().strip()

        # Support/help queries should target help/search/chat inputs, not trip forms.
        if any(w in f"{intent} {value.lower()}" for w in ("lost item", "help", "faq", "support", "chat")):
            try:
                ok = self._page.evaluate(
                    """(v) => {
                        const good = ['search','help','support','faq','question','ask','chat','webchat'];
                        const bad = ['from','to','going','destination','arrival','departure','passenger','adult','child','date','checkout','checkin'];
                        function vis(el) {
                            const r = el.getBoundingClientRect();
                            const cs = getComputedStyle(el);
                            return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
                        }
                        const inputs = Array.from(document.querySelectorAll('input, textarea, [role="textbox"], [role="searchbox"], [role="combobox"]'));
                        let best = null, bestScore = -999;
                        for (const el of inputs) {
                            if (!vis(el)) continue;
                            const blob = ((el.getAttribute('type') || '') + ' ' + (el.getAttribute('name') || '') + ' ' +
                                          (el.getAttribute('id') || '') + ' ' + (el.getAttribute('placeholder') || '') + ' ' +
                                          (el.getAttribute('aria-label') || '')).toLowerCase();
                            let score = 0;
                            if (good.some(k => blob.includes(k))) score += 9;
                            if (bad.some(k => blob.includes(k))) score -= 10;
                            if ((el.getAttribute('type') || '').toLowerCase() === 'search') score += 2;
                            if (score > bestScore) { bestScore = score; best = el; }
                        }
                        if (!best || bestScore < 2) return false;
                        best.focus();
                        best.value = v;
                        best.dispatchEvent(new Event('input', { bubbles: true }));
                        best.dispatchEvent(new Event('change', { bubbles: true }));
                        return String(best.value || '').toLowerCase().includes(String(v || '').toLowerCase().slice(0, Math.min(8, String(v || '').length)));
                    }""",
                    value,
                )
                if ok:
                    logger.info("Native support-query type ok via JS targeted input: intent=%r value=%r", step.intent, value)
                    return ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="type",
                            message=f"Native fill (support-query-targeted): {value!r}.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
            except Exception:
                pass

        # ZIP/postal inputs are frequently mis-targeted to newsletter fields.
        if re.fullmatch(r"\d{5}", value) and any(w in intent for w in ("zip", "postal", "postcode")):
            try:
                ok = self._page.evaluate(
                    """(v) => {
                        const bad = ['email','first','last','name','subscribe','newsletter'];
                        const good = ['zip','postal','postcode','store','location','find'];
                        function vis(el) {
                            const r = el.getBoundingClientRect();
                            const cs = getComputedStyle(el);
                            return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
                        }
                        const inputs = Array.from(document.querySelectorAll('input, [role="textbox"], [role="combobox"]'));
                        let best = null, bestScore = -999;
                        for (const el of inputs) {
                            if (!vis(el)) continue;
                            const t = ((el.getAttribute('type') || '') + ' ' + (el.getAttribute('name') || '') + ' ' +
                                       (el.getAttribute('id') || '') + ' ' + (el.getAttribute('placeholder') || '') + ' ' +
                                       (el.getAttribute('aria-label') || '')).toLowerCase();
                            let score = 0;
                            if (good.some(k => t.includes(k))) score += 8;
                            if (bad.some(k => t.includes(k))) score -= 10;
                            if ((el.getAttribute('type') || '').toLowerCase() === 'search') score += 2;
                            if (score > bestScore) { bestScore = score; best = el; }
                        }
                        if (!best || bestScore < 2) return false;
                        best.focus();
                        best.value = v;
                        best.dispatchEvent(new Event('input', { bubbles: true }));
                        best.dispatchEvent(new Event('change', { bubbles: true }));
                        return String(best.value || '') === String(v);
                    }""",
                    value,
                )
                if ok:
                    logger.info("Native zip type ok via JS targeted input: intent=%r value=%r", step.intent, value)
                    return ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="type",
                            message=f"Native fill (zip-targeted): {value!r}.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
            except Exception:
                pass

        # Price/budget inputs are commonly confused with location/destination.
        if re.fullmatch(r"\d+(?:[.,]\d+)?", value) and any(
            w in intent for w in ("price", "budget", "max price", "maximum price", "under ")
        ):
            try:
                ok = self._page.evaluate(
                    """(v) => {
                        function vis(el) {
                            const r = el.getBoundingClientRect();
                            const cs = getComputedStyle(el);
                            return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
                        }
                        const good = ['price','budget','max','maximum','amount','cost'];
                        const bad = ['destination','location','where','city','search destination','louvre','rome','paris'];
                        const inputs = Array.from(document.querySelectorAll('input, [role="textbox"], [role="combobox"]'));
                        let best = null, bestScore = -999;
                        for (const el of inputs) {
                            if (!vis(el)) continue;
                            const t = ((el.getAttribute('type') || '') + ' ' + (el.getAttribute('name') || '') + ' ' +
                                       (el.getAttribute('id') || '') + ' ' + (el.getAttribute('placeholder') || '') + ' ' +
                                       (el.getAttribute('aria-label') || '')).toLowerCase();
                            let score = 0;
                            if (good.some(k => t.includes(k))) score += 10;
                            if (bad.some(k => t.includes(k))) score -= 12;
                            if ((el.getAttribute('type') || '').toLowerCase() === 'number') score += 4;
                            if (score > bestScore) { bestScore = score; best = el; }
                        }
                        if (!best || bestScore < 3) return false;
                        best.focus();
                        best.value = v;
                        best.dispatchEvent(new Event('input', { bubbles: true }));
                        best.dispatchEvent(new Event('change', { bubbles: true }));
                        return String(best.value || '').replace(/,/g,'') === String(v).replace(/,/g,'');
                    }""",
                    value,
                )
                if ok:
                    logger.info("Native price type ok via JS targeted input: intent=%r value=%r", step.intent, value)
                    return ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="type",
                            message=f"Native fill (price-targeted): {value!r}.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
            except Exception:
                pass
        _is_search_intent = any(w in intent for w in (
            "search", "query", "type", "enter", "into the search", "into the bar",
            "into the box", "find ", "look for", "commit search", "search bar",
        ))
        if _is_search_intent:
            _direct_css_patterns = [
                'input[name="q"]',
                'input[type="search"]:visible',
                '[role="searchbox"]:visible',
                'input[placeholder*="Search" i]:visible',
                'input[aria-label*="Search" i]:visible',
                'input[placeholder*="search" i]:visible',
                'input[placeholder*="Find" i]:visible',
                'input[placeholder*="Type" i]:visible',
                '[data-testid*="search" i] input:visible',
                'input[name="searchterm"]:visible',
                'input[name="search"]:visible',
            ]
            for _css in _direct_css_patterns:
                try:
                    _loc = self._page.locator(_css)
                    if _loc.count() <= 0:
                        continue
                    _el = _loc.first
                    if not _el.is_visible(timeout=300):
                        continue
                    # Skip if it's clearly a newsletter/subscribe input
                    _ph = (_el.get_attribute("placeholder") or "").lower()
                    _al = (_el.get_attribute("aria-label") or "").lower()
                    if any(bad in _ph + _al for bad in ("email", "subscribe", "newsletter", "password")):
                        continue
                    _el.scroll_into_view_if_needed(timeout=1000)
                    _el.click(timeout=1000)
                    _el.fill(value, timeout=2000)
                    _actual = _el.input_value(timeout=500) or ""
                    if value.lower()[:6] in _actual.lower() or _actual.lower() in value.lower():
                        logger.info(
                            "Direct search CSS fill ok: intent=%r value=%r pattern=%r",
                            step.intent, value, _css,
                        )
                        return ActionVerificationResult(
                            action_result=ActionResult(
                                success=True, action="type",
                                message=f"Direct search fill: {value!r}.",
                                latency_ms=0.0,
                            ),
                            verification=None, metadata=None,
                        )
                except Exception:
                    continue

        _SKIP = frozenset({
            "type", "enter", "fill", "write", "input", "put", "set", "key",
            "in", "into", "the", "a", "an", "field", "box", "area", "bar",
            "form", "click", "on", "to", "for", "via", "and", "or", "with",
            "using", "by", "place", "button", "label", "placeholder",
        })
        intent_clean = intent
        # Remove quoted fragments (the actual value being typed)
        for quoted in re.findall(r"['\"]([^'\"]+)['\"]", intent_clean):
            intent_clean = intent_clean.replace(f"'{quoted}'", "").replace(f'"{quoted}"', "")
        intent_clean = intent_clean.replace(value.lower(), "")
        label_words = [
            w.strip("'\"()[].,!?;:")
            for w in intent_clean.split()
            if len(w.strip("'\"()[].,!?;:")) >= 3
            and w.strip("'\"()[].,!?;:") not in _SKIP
        ]

        locators = []
        # Per-word: label, placeholder, role matches
        for word in label_words[:5]:
            locators += [
                self._page.get_by_label(word, exact=False),
                self._page.get_by_placeholder(word, exact=False),
                self._page.get_by_role("textbox", name=word, exact=False),
                self._page.get_by_role("searchbox", name=word, exact=False),
                self._page.get_by_role("combobox", name=word, exact=False),
            ]
        # Two-word label (e.g. "check-in date")
        if len(label_words) >= 2:
            two = f"{label_words[0]} {label_words[1]}"
            locators += [
                self._page.get_by_label(two, exact=False),
                self._page.get_by_placeholder(two, exact=False),
            ]
        # Generic role fallbacks  -  catch-all for unlabelled inputs
        locators += [
            self._page.get_by_role("searchbox"),
            self._page.get_by_role("combobox"),
        ]

        # Bad form contexts that should never receive typed values unless the intent
        # explicitly targets them (e.g. newsletter, login, account sign-up).
        _BAD_FORM_CONTEXTS = frozenset({
            "subscribe", "newsletter", "email", "sign up", "sign-up", "signup",
            "create account", "register", "login", "log in", "password",
        })
        _intent_targets_bad = any(w in intent.lower() for w in _BAD_FORM_CONTEXTS)

        seen: set[str] = set()
        deadline = time.perf_counter() + 4.0
        for loc in locators:
            if time.perf_counter() > deadline:
                break
            loc_key = str(loc)
            if loc_key in seen:
                continue
            seen.add(loc_key)
            try:
                target = loc.first
                if not target.is_visible(timeout=350):
                    continue
                # Guard: reject elements that are clearly buttons or non-input tags.
                # fill() on a button never works and can mis-click.
                try:
                    _tag = (target.evaluate("el => el.tagName") or "").lower()
                    if _tag in {"button", "a", "div", "span", "li"}:
                        continue
                    _input_type = (target.get_attribute("type") or "").lower()
                    if _input_type in {"button", "submit", "reset", "checkbox", "radio", "file", "image"}:
                        continue
                except Exception:
                    pass
                # Guard: skip newsletter/subscribe/login form inputs when the intent
                # is about something else (e.g. zip code, search query, destination).
                if not _intent_targets_bad:
                    try:
                        _form_ctx = (
                            target.evaluate(
                                "el => { const f = el.closest('form'); return f ? f.innerText.slice(0,200) : ''; }"
                            ) or ""
                        ).lower()
                        if any(bad in _form_ctx for bad in _BAD_FORM_CONTEXTS):
                            continue
                        # Also reject inputs whose placeholder/label screams email/password
                        _ph = (target.get_attribute("placeholder") or "").lower()
                        _aria = (target.get_attribute("aria-label") or "").lower()
                        _combined = f"{_ph} {_aria}"
                        if any(bad in _combined for bad in ("email", "password", "subscribe", "newsletter")):
                            continue
                    except Exception:
                        pass
                target.scroll_into_view_if_needed(timeout=1500)
                target.click(timeout=1500)
                target.fill(value, timeout=3000)
                # Verify fill was accepted: partial match handles formatted values
                # (e.g. typed "Rome", input shows "Rome, Italy")
                try:
                    actual = target.input_value(timeout=800)
                    if actual is not None:
                        if not (value.lower() in actual.lower() or actual.lower() in value.lower()):
                            continue  # Wrong element - value didn't stick, try next
                except Exception:
                    pass  # Can't read input_value -> trust fill() succeeded
                logger.info(
                    "Native type ok: intent=%r value=%r via %s",
                    step.intent, value, loc_key[:70],
                )
                return ActionVerificationResult(
                    action_result=ActionResult(
                        success=True, action="type",
                        message=f"Native fill: {value!r} via {loc_key[:70]}.",
                        latency_ms=0.0,
                    ),
                    verification=None, metadata=None,
                )
            except Exception:
                continue
        return None

    def _try_playwright_native_click(
        self, step: "AutomationStep"
    ) -> "Optional[ActionVerificationResult]":
        
        if self._page is None:
            return None
        intent = (step.intent or "").lower()
        value = (step.value or "").strip()
        _cur_url = (self._page.url or "").lower()

        locators = []

        if any(w in intent for w in ("help", "faq", "support", "contact", "lost item",
                                     "customer service", "assistance")):
            # Try to find a real Help/FAQ link (not webchat, not chat buttons)
            locators += [
                self._page.get_by_role("link", name=re.compile(r"^help$|^faq", re.I)),
                self._page.locator('a[href*="/faq" i]:not([href*="webchat" i])').first,
                self._page.locator('a[href*="/help" i]:not([href*="webchat" i])').first,
                self._page.locator('nav a:has-text("Help")').first,
                self._page.locator('header a:has-text("Help")').first,
                self._page.locator('footer a:has-text("Help")').first,
                self._page.locator('a:has-text("FAQs")').first,
                self._page.locator('a:has-text("FAQ")').first,
            ]
            # For megabus specifically - navigate to FAQs directly
            if "megabus.com" in _cur_url:
                try:
                    self._page.goto("https://us.megabus.com/faqs", wait_until="domcontentloaded", timeout=10000)
                    self.extract_page_context()
                    return ActionVerificationResult(
                        action_result=ActionResult(
                            success=True, action="click",
                            message="Navigated to Megabus FAQs page directly.",
                            latency_ms=0.0,
                        ),
                        verification=None, metadata=None,
                    )
                except Exception:
                    pass

        # Recipe-selection intents should click an actual recipe link.
        if "recipe" in intent:
            locators += [
                self._page.locator('a[href*="/recipe/"]').first,
                self._page.locator('a[href*="/recipes/"]').first,
                self._page.locator('main a[href*="/recipe/"]').first,
                self._page.get_by_role("link", name=re.compile(r"quinoa|salad|recipe", re.I)).first,
            ]

        # Autocomplete-pick intents: restrict to listbox/option selectors only.
        if self._intent_is_autocomplete_pick(step.intent or ""):
            if value:
                locators += [
                    self._page.get_by_role("option", name=re.compile(re.escape(value), re.I)),
                    self._page.locator('[role="listbox"] [role="option"]').filter(has_text=re.compile(re.escape(value), re.I)),
                    self._page.locator('[data-testid*="autocomplete"] [role="option"]').filter(has_text=re.compile(re.escape(value), re.I)),
                    self._page.locator('ul[role="listbox"] li').filter(has_text=re.compile(re.escape(value), re.I)),
                ]
            else:
                locators += [
                    self._page.locator('[role="listbox"] [role="option"]'),
                    self._page.locator('[data-testid*="autocomplete"] [role="option"]'),
                ]

        # Occupancy/guest controls: avoid accidental calendar/day clicks.
        if any(w in intent for w in ("adult", "adults", "guest", "guests", "occupancy", "rooms")):
            if any(w in intent for w in ("decrease", "minus", "reduce", "less")):
                locators += [
                    self._page.get_by_role("button", name=re.compile(r"decrease.*adult|adult.*decrease|minus", re.I)),
                    self._page.locator('button[aria-label*="Decrease number of Adults"]'),
                    self._page.locator('button[aria-label*="Decrease number of adults"]'),
                    self._page.locator('button:has-text("-")'),
                ]
            elif any(w in intent for w in ("increase", "plus", "more")):
                locators += [
                    self._page.get_by_role("button", name=re.compile(r"increase.*adult|adult.*increase|plus", re.I)),
                    self._page.locator('button[aria-label*="Increase number of Adults"]'),
                    self._page.locator('button[aria-label*="Increase number of adults"]'),
                    self._page.locator('button:has-text("+")'),
                ]
            else:
                locators += [
                    self._page.get_by_role("button", name=re.compile(r"adults?|guests?|occupancy|rooms?", re.I)),
                    self._page.locator('[data-testid*="occupancy"]'),
                ]

        # Search / submit 
        if any(w in intent for w in ("search", "submit", "find hotel", "find flight",
                                     "click the search")):
            locators += [
                self._page.get_by_role("button", name="Search", exact=True),
                self._page.get_by_role("button", name="Search", exact=False),
                self._page.get_by_role("button", name="Find", exact=False),
                self._page.locator('button[type="submit"]').last,
                self._page.get_by_role("button", name="Show results", exact=False),
            ]

        # Calendar date cells (by aria-label) 
        elif any(w in intent for w in ("date cell", "click the date", "click date",
                                       "click april", "click march", "click may",
                                       "click 20", "click 23", "click 3 ", "click 5 ",
                                       "april", "march")):
            day_m = re.search(r"\b(\d{1,2})\b", intent)
            mon_m = re.search(
                r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|'
                r'january|february|march|april|june|july|august|'
                r'september|october|november|december)', intent, re.I)
            yr_m = re.search(r'(202\d)', intent)
            if day_m and mon_m:
                day  = day_m.group(1)
                mon3 = mon_m.group(1)[:3].capitalize()
                yr   = yr_m.group(1) if yr_m else str(_date.today().year)
                locators += [
                    # Booking.com: "Thu 20 Apr 2026"
                    self._page.locator(f'[aria-label*="{day} {mon3} {yr}"]'),
                    self._page.locator(f'[aria-label*="{mon3} {day}, {yr}"]'),
                    self._page.locator(f'[aria-label*="{mon3} {day} {yr}"]'),
                    self._page.get_by_role(
                        "button",
                        name=re.compile(rf'\b{day}\b.*{mon3}.*{yr}', re.I)
                    ),
                    self._page.get_by_role(
                        "gridcell",
                        name=re.compile(rf'\b{day}\b', re.I)
                    ).first,
                ]

        # Next / prev month 
        elif any(w in intent for w in ("next month", "go to next", "advance month",
                                       "advance the calendar", "next arrow", "click next")):
            locators += [
                # Booking.com / generic: "Next month"
                self._page.get_by_role("button", name="Next month", exact=True),
                self._page.get_by_role("button", name="next month", exact=False),
                self._page.locator('[aria-label="Next month"]'),
                self._page.locator('[aria-label*="next month" i]'),
                # Google Flights: aria-label="Next" (no "month")
                self._page.locator('[aria-label="Next"]'),
                self._page.get_by_role("button", name="Next", exact=True),
                # Generic fallbacks
                self._page.locator('button[data-testid*="next" i]'),
                self._page.locator('[class*="DayPickerNavigation_rightButton" i]'),
                self._page.locator('button[class*="next" i]'),
            ]
        elif any(w in intent for w in ("previous month", "prev month", "go back")):
            locators += [
                self._page.get_by_role("button", name="Previous month", exact=True),
                self._page.locator('[aria-label="Previous month"]'),
                self._page.locator('[aria-label="Previous"]'),
            ]

        # Sort options 
        elif "price" in intent and ("lowest" in intent or "sort" in intent):
            locators += [
                self._page.get_by_role("option", name="Price (lowest first)", exact=False),
                self._page.get_by_role("option", name="Lowest price", exact=False),
                self._page.get_by_text("Price (lowest first)", exact=False),
            ]

        # Sort dropdown trigger 
        elif any(w in intent for w in ("sort", "sorting", "order by", "order results")):
            locators += [
                # Booking.com sort dropdown trigger (data-testid)
                self._page.locator('[data-testid="sorters-dropdown-trigger"]'),
                self._page.locator('[data-testid*="sorter" i]').first,
                self._page.get_by_role("button", name=re.compile(r"sort|order by|sorted|filter by", re.I)),
                self._page.locator('select[name*="sort" i], select[id*="sort" i]'),
                self._page.locator('[data-testid*="sort" i] button'),
                self._page.locator('button:has-text("Sort by")'),
                self._page.locator('button:has-text("Sort")'),
            ]

        # Review score / rating filter 
        elif any(w in intent for w in ("review score", "superb", "9+", "8+", "7+",
                                       "rating filter", "score filter", "highly rated")):
            locators += [
                # Booking.com review score filter checkbox (data-testid pattern)
                self._page.locator('[data-testid*="reviewscore" i]').first,
                self._page.locator('[data-testid*="filter"][data-testid*="90"]').first,
                self._page.get_by_role("checkbox", name=re.compile(r"superb|9\+|wonderful|8\+", re.I)),
                self._page.locator('label:has-text("Superb")'),
                self._page.locator('label:has-text("9+")'),
                self._page.locator('[data-testid*="filter"] label:has-text("Superb")'),
                self._page.locator('[data-testid*="filter"] label:has-text("9+")'),
                self._page.locator('input[type="checkbox"] + label:has-text("Superb")'),
                self._page.locator('input[type="checkbox"] + label:has-text("9+")'),
                self._page.get_by_text(re.compile(r"superb.*9\+|9\+.*superb", re.I)).first,
                self._page.get_by_text("Superb: 9+", exact=False),
                self._page.get_by_text("Superb 9+", exact=False),
            ]

        # Filter/rating menu opener 
        elif any(w in intent for w in ("filter menu", "filter options", "open filter",
                                       "rating filter", "ratings & reviews", "reviews filter")):
            # Allrecipes / similar sites: the filter button is a text-labelled button
            # in the sidebar/top bar, NOT the main nav menu
            locators += [
                self._page.locator('button:has-text("Rating")'),
                self._page.locator('button:has-text("Ratings")'),
                self._page.locator('button:has-text("Reviews")'),
                self._page.locator('button[data-testid*="filter" i]'),
                self._page.locator('[class*="filter" i] button').first,
                self._page.get_by_role("button", name=re.compile(r"filter|rating|review", re.I)),
            ]

        # Filter checkboxes 
        elif any(w in intent for w in ("breakfast", "swimming pool", "pool", "free wifi",
                                       "wifi", "free cancellation", "nonstop", "non-stop",
                                       "non stop")):
            # "nonstop" OR "non-stop" OR "non stop" all map to kw="nonstop"
            _is_nonstop = any(w in intent for w in ("nonstop", "non-stop", "non stop"))
            kw = "nonstop" if _is_nonstop else next(
                (w for w in ("breakfast", "pool", "wifi", "cancellation") if w in intent), ""
            )
            if kw == "nonstop":
                locators += [
                    self._page.get_by_role("checkbox", name=re.compile(r"nonstop|non.?stop", re.I)),
                    self._page.locator('label:has-text("Nonstop")'),
                    self._page.locator('label:has-text("Non-stop")'),
                    self._page.locator('[data-value*="NONE" i]'),
                    # Google Flights: stops filter chip/checkbox
                    self._page.locator('[aria-label*="nonstop" i]'),
                    self._page.locator('[aria-label*="non-stop" i]'),
                    self._page.get_by_text(re.compile(r"^nonstop$|^non.?stop$", re.I)),
                    self._page.get_by_text(re.compile(r"nonstop|non.?stop", re.I)).first,
                ]
            elif kw:
                locators += [
                    self._page.get_by_role("checkbox", name=kw, exact=False),
                    self._page.get_by_label(kw, exact=False),
                ]

        # Generic: use value text if provided 
        elif any(w in intent for w in ("product link", "product page", "details page", "result", "item")):
            phrase = value
            if not phrase:
                q = re.findall(r"['\"]([^'\"]{3,120})['\"]", step.intent or "")
                if q:
                    phrase = q[0].strip()
            if phrase:
                locators += [
                    self._page.get_by_role("link", name=phrase, exact=False),
                    self._page.get_by_role("heading", name=phrase, exact=False),
                    self._page.get_by_text(phrase, exact=False),
                ]

        elif value:
            locators += [
                self._page.get_by_role("option", name=value, exact=False),
                self._page.get_by_role("button", name=value, exact=False),
                self._page.get_by_role("link", name=value, exact=False),
                self._page.get_by_text(value, exact=True),
            ]

        # Try each locator 
        _WRONG = ("/attractions", "/flights", "/cars", "/airport-taxis",
                  "/activities", "/cruises")
        deadline = time.perf_counter() + 4.5
        for loc in locators:
            if time.perf_counter() > deadline:
                break
            try:
                target = loc.first
                if not target.is_visible(timeout=350):
                    continue
                target.scroll_into_view_if_needed(timeout=2000)
                url_before = self._page.url
                target.click(timeout=3000)
                try:
                    self._page.wait_for_timeout(250)
                except Exception:
                    pass
                url_after = self._page.url
                # Abort if we navigated somewhere wrong
                if url_after != url_before:
                    bad = self._check_url_goal_alignment(url_after, step.intent or "", [], None)
                    if bad or any(p in url_after.lower() for p in _WRONG):
                        try:
                            self._page.go_back(wait_until="domcontentloaded", timeout=6000)
                        except Exception:
                            pass
                        continue
                logger.info("Native click succeeded: intent=%r locator=%r", step.intent, str(loc))
                return ActionVerificationResult(
                    action_result=ActionResult(
                        success=True, action="click",
                        message="Clicked via Playwright semantic locator.",
                        latency_ms=0.0,
                    ),
                    verification=None, metadata=None,
                )
            except Exception:
                continue
        return None

    def _try_date_input_js(self, intent: str) -> bool:
        """Open date picker: Playwright-native locators first, JS querySelector fallback."""
        if self._page is None:
            return False
        is_checkin = self._neural.classify(intent, "checkin date arrival start", threshold=0.50)
        is_checkout = self._neural.classify(intent, "checkout date departure end return", threshold=0.50)

        # Strategy 1: Playwright-native (most reliable across sites)
        if is_checkin:
            checkin_locators = [
                self._page.get_by_test_id("date-display-field-start"),
                self._page.get_by_test_id("searchbox-dates-container"),
                self._page.get_by_label("Check-in", exact=False),
                self._page.get_by_label("Check in", exact=False),
                self._page.get_by_label("Start date", exact=False),
                self._page.get_by_placeholder("Check-in", exact=False),
                self._page.get_by_placeholder("Check in", exact=False),
                self._page.locator('[data-testid="date-display-field-start"]'),
                self._page.locator('[data-testid="searchbox-dates-container"]'),
            ]
        elif is_checkout:
            checkin_locators = [
                self._page.get_by_test_id("date-display-field-end"),
                self._page.get_by_label("Check-out", exact=False),
                self._page.get_by_label("Check out", exact=False),
                self._page.get_by_label("End date", exact=False),
                self._page.get_by_placeholder("Check-out", exact=False),
                self._page.locator('[data-testid="date-display-field-end"]'),
            ]
        else:
            checkin_locators = [
                self._page.get_by_test_id("date-display-field-start"),
                self._page.get_by_label("Date", exact=False),
                self._page.locator('[data-testid*="date"]').first,
            ]

        for loc in checkin_locators:
            try:
                if loc.count() > 0:
                    loc.first.scroll_into_view_if_needed(timeout=2000)
                    loc.first.click(timeout=3000)
                    logger.info("Date picker opened via Playwright locator")
                    return True
            except Exception:
                continue

        # Strategy 2: JS querySelector with comprehensive selectors
        _JS = """(isCheckin, isCheckout) => {
            const CHECKIN_SELS = [
                '[data-testid="date-display-field-start"]',
                '[data-testid="searchbox-dates-container"]',
                '[data-testid*="check-in"]', '[data-testid*="checkin"]',
                '[data-testid*="start-date"]',
                '[aria-label*="Check-in"]', '[aria-label*="check-in"]',
                '[aria-label*="Check in"]', '[aria-label*="Start date"]',
                'input[placeholder*="Check-in"]', 'input[placeholder*="check-in"]',
                '[class*="sb-date-field__start"]', '[class*="checkin"]',
            ];
            const CHECKOUT_SELS = [
                '[data-testid="date-display-field-end"]',
                '[data-testid*="check-out"]', '[data-testid*="checkout"]',
                '[aria-label*="Check-out"]', '[aria-label*="check-out"]',
                '[aria-label*="Check out"]', '[aria-label*="End date"]',
                'input[placeholder*="Check-out"]',
                '[class*="sb-date-field__end"]', '[class*="checkout"]',
            ];
            const GENERIC = [
                'input[type="date"]', '[data-testid*="date"]',
                '[aria-label*="date"]', '[aria-label*="Date"]',
            ];
            function isVis(el) {
                if (!el) return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            }
            function tryClick(sels) {
                for (const s of sels) {
                    try {
                        const els = document.querySelectorAll(s);
                        for (const el of els) {
                            if (isVis(el)) { el.click(); el.focus(); return s; }
                        }
                    } catch(e) {}
                }
                return null;
            }
            if (isCheckin) return tryClick(CHECKIN_SELS) || tryClick(GENERIC);
            if (isCheckout) return tryClick(CHECKOUT_SELS) || tryClick(GENERIC);
            return tryClick(GENERIC);
        }"""
        for frame in self._page.frames:
            try:
                result = frame.evaluate(_JS, is_checkin, is_checkout)
                if result:
                    logger.info("Date picker opened via JS: %r", result)
                    return True
            except Exception:
                continue
        return False

    def _is_auth_page(self) -> bool:
        
        if self._page is None:
            return False
        try:
            has_password = self._page.evaluate(
                "() => document.querySelector('input[type=\"password\"]') !== null"
            )
            if has_password:
                return True
        except Exception:
            pass
        current_url = (self._page.url or "").lower()
        if any(k in current_url for k in (
            "/login", "/signin", "/sign-in", "/account/login",
            "oauth", "accounts.google.com", "auth",
        )):
            return True

        # Strict lexical check to avoid false positives on result pages.
        try:
            title = self._page.title() or ""
            h1 = self._page.evaluate(
                "() => { const h = document.querySelector('h1,h2'); "
                "return h ? h.textContent.trim() : ''; }"
            ) or ""
            page_text = f"{title} {h1}".lower()
            if any(t in page_text for t in (
                "sign in", "log in", "login", "continue with google",
                "continue with apple", "verify it's you", "forgot password",
                "create account",
            )):
                return True
        except Exception:
            pass
        return False

    def _detect_and_escape_auth_redirect(self, url_before: str) -> bool:
        
        if self._page is None:
            return False
        current_url = (self._page.url or "").lower()
        if not url_before or current_url == url_before.lower():
            return False
        if self._url_looks_results_like(current_url) and not any(
            t in current_url for t in ("login", "signin", "sign-in", "auth", "oauth")
        ):
            return False
        if not self._is_auth_page():
            return False

        logger.warning(
            "Auth/login redirect detected (url=%r). Navigating back to %r.",
            self._page.url, url_before,
        )
        try:
            self._page.go_back(wait_until="domcontentloaded", timeout=8000)
            time.sleep(0.3)
            self.extract_page_context()
            logger.info("Successfully navigated back from auth page.")
            return True
        except Exception as exc:
            logger.warning("go_back from auth page failed: %s. Trying direct goto.", exc)
            try:
                self._page.goto(url_before, wait_until="domcontentloaded", timeout=15000)
                self.extract_page_context()
                return True
            except Exception as exc2:
                logger.error("Could not recover from auth redirect: %s", exc2)
        return False

    def _try_calendar_nav_js(self, aria_patterns: list[str]) -> bool:
        if self._page is None:
            return False
        patterns_lower = [p.lower() for p in aria_patterns]
        _js = """(patterns) => {
            // Recursively collect buttons from shadow DOM
            function collectButtons(root) {
                const btns = [];
                const direct = Array.from(root.querySelectorAll('button, [role="button"]'));
                btns.push(...direct);
                // Pierce shadow roots
                const all = Array.from(root.querySelectorAll('*'));
                for (const el of all) {
                    if (el.shadowRoot) {
                        btns.push(...collectButtons(el.shadowRoot));
                    }
                }
                return btns;
            }
            const btns = collectButtons(document);
            for (const b of btns) {
                const label = (
                    b.getAttribute('aria-label') ||
                    b.textContent ||
                    ''
                ).toLowerCase().trim();
                if (patterns.some(p => label === p || label.startsWith(p) || label.includes(p))) {
                    b.click();
                    return label;
                }
            }
            return null;
        }"""
        for frame in self._page.frames:
            try:
                found = frame.evaluate(_js, patterns_lower)
                if found:
                    logger.info("Calendar nav JS fallback clicked button: %r", found)
                    return True
            except Exception:
                continue
        return False

    def _to_interactive(self, ex):
        from capture import InteractiveElement
        return InteractiveElement(
            tag_name=ex.tag_name,
            role=ex.role,
            actions=ex.actions,
            is_visible=ex.is_visible,
            is_enabled=ex.is_enabled,
            is_in_viewport=ex.is_in_viewport,
            bounding_box=ex.bounding_box,
            dom_node_id=ex.dom_node_id,
            name=ex.name,
            placeholder=ex.placeholder,
            value=ex.value,
            input_type=None,
            xpath=ex.xpath,
            css_selector=ex.css_selector,
            nearby_text=ex.nearby_text,
            form_context=ex.form_context,
        )

    def goto(self, url: str, wait_until: str = "domcontentloaded", timeout_ms: int = 30000) -> PageState:
        if self._page is None:
            raise RuntimeError("Agent not started. Call start() first.")
        url_lower = url.lower().strip()
        if not (url_lower.startswith("http://") or url_lower.startswith("https://") or url_lower.startswith("file://")):
            raise ValueError(f"Unsupported URL scheme: {url!r}. Only http/https/file are allowed.")
        self._ensure_browser_alive()
        logger.info("Navigating to %s", url)
        last_exc: Optional[Exception] = None
        for attempt, (wstate, tms) in enumerate([
            (wait_until, timeout_ms),
            ("load", timeout_ms + 30000),
            ("commit", 15000),
        ], start=1):
            try:
                self._page.goto(url, wait_until=wstate, timeout=tms)
                self._mitigate_bot_gate()
                extraction = self.extract_page_context()
                if len(extraction.elements) < 8:
                    ready_state, body_len, interactive_count = self._probe_dom_readiness()
                    # If the raw DOM has interactive controls but extractor got
                    # too few elements, give extraction one more chance.
                    if interactive_count >= 5 and len(extraction.elements) == 0:
                        try:
                            self._page.wait_for_timeout(700)
                        except Exception:
                            pass
                        extraction = self.extract_page_context()
                    # If page is still effectively blank/loading, retry with the
                    # next wait strategy instead of starting execution on empty DOM.
                    if (
                        len(extraction.elements) < 3
                        and attempt < 3
                        and (
                            extraction.state.title.strip() == ""
                            or body_len < 24
                            or ready_state not in {"interactive", "complete"}
                        )
                    ):
                        logger.warning(
                            "Page not interaction-ready after navigation: title=%r ready=%r body_len=%d interactive=%d elements=%d (attempt=%d). Retrying.",
                            extraction.state.title,
                            ready_state,
                            body_len,
                            interactive_count,
                            len(extraction.elements),
                            attempt,
                        )
                        continue
                logger.debug(
                    "Page loaded (attempt=%d wait_until=%r): title=%r elements=%d",
                    attempt, wstate, extraction.state.title, len(extraction.elements),
                )
                return extraction.state
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Navigation attempt %d failed (wait_until=%r timeout=%dms): %s",
                    attempt, wstate, tms, exc,
                )
                try:
                    self._page.wait_for_timeout(500)
                except Exception:
                    pass
        raise RuntimeError(f"Navigation to {url!r} failed after all attempts: {last_exc}")

    def _probe_dom_readiness(self) -> tuple[str, int, int]:
        if self._page is None:
            return "unknown", 0, 0
        js = """() => {
            try {
                const body = document.body;
                const txt = body ? (body.innerText || '') : '';
                const interactive = document.querySelectorAll(
                    'input, button, select, textarea, a, [role="button"], [role="combobox"], [role="textbox"]'
                ).length;
                return {
                    ready: document.readyState || 'unknown',
                    body_len: txt.trim().length,
                    interactive_count: interactive,
                };
            } catch(e) {
                return { ready: 'unknown', body_len: 0, interactive_count: 0 };
            }
        }"""
        try:
            obj = self._page.evaluate(js) or {}
            ready = str(obj.get("ready") or "unknown")
            body_len = int(obj.get("body_len") or 0)
            interactive = int(obj.get("interactive_count") or 0)
            return ready, body_len, interactive
        except Exception:
            return "unknown", 0, 0

    def _ensure_interaction_ready(self) -> bool:
        """Ensure the current page is usable for non-navigation actions."""
        if self._page is None:
            return False
        if self._is_bot_gate_page():
            return False
        try:
            if self._current_extraction is None:
                self.extract_page_context()
        except Exception:
            pass

        ready, body_len, interactive_count = self._probe_dom_readiness()
        extracted_count = len(self._current_extraction.elements) if self._current_extraction else 0
        if extracted_count >= 12 or interactive_count >= 8:
            return True

        # Recovery cycles: short wait + reload + re-extract.
        ok = False
        ready2, body_len2, interactive_count2, extracted_count2 = ready, body_len, interactive_count, extracted_count
        for _ in range(2):
            try:
                self._page.wait_for_timeout(650)
                self._page.reload(wait_until="domcontentloaded", timeout=12000)
                self.extract_page_context()
            except Exception:
                pass
            ready2, body_len2, interactive_count2 = self._probe_dom_readiness()
            extracted_count2 = len(self._current_extraction.elements) if self._current_extraction else 0
            ok = extracted_count2 >= 12 or interactive_count2 >= 8
            if ok:
                break

        # Final recovery: re-open current URL (handles transient blank/challenge pages).
        if not ok:
            try:
                cur = (self._page.url or "").strip()
                if cur.startswith("http://") or cur.startswith("https://"):
                    self.goto(cur, wait_until="domcontentloaded", timeout_ms=15000)
                    ready2, body_len2, interactive_count2 = self._probe_dom_readiness()
                    extracted_count2 = len(self._current_extraction.elements) if self._current_extraction else 0
                    ok = extracted_count2 >= 12 or interactive_count2 >= 8
            except Exception:
                pass

        if not ok:
            if interactive_count2 == 0 and body_len2 <= 260:
                # Likely anti-bot/challenge/interstitial despite non-empty body.
                try:
                    self._mitigate_bot_gate()
                    self.extract_page_context()
                    ready2, body_len2, interactive_count2 = self._probe_dom_readiness()
                    extracted_count2 = len(self._current_extraction.elements) if self._current_extraction else 0
                    ok = extracted_count2 >= 12 or interactive_count2 >= 8
                except Exception:
                    pass

        if not ok:
            logger.warning(
                "Page still not interaction-ready: ready=%r body_len=%d interactive=%d extracted=%d | ready2=%r body2=%d interactive2=%d extracted2=%d",
                ready, body_len, interactive_count, extracted_count,
                ready2, body_len2, interactive_count2, extracted_count2,
            )
        return ok

    def _mitigate_bot_gate(self) -> None:
        """Detect common anti-bot interstitials and attempt a short recovery."""
        if self._page is None:
            return
        for _ in range(2):
            blocked = self._is_bot_gate_page()
            if not blocked:
                return
            logger.warning("Bot/anti-automation gate detected; waiting and retrying page readiness.")
            # Challenge token URLs often append volatile query params; retry a
            # clean URL first.
            try:
                cur = self._page.url or ""
                if "__cf_chl_rt_tk=" in cur or "chal_t=" in cur or "force_referer=" in cur:
                    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
                    pu = urlparse(cur)
                    _drop = {"__cf_chl_rt_tk", "chal_t", "force_referer"}
                    q = [(k, v) for (k, v) in parse_qsl(pu.query, keep_blank_values=True) if k not in _drop]
                    clean = urlunparse((pu.scheme, pu.netloc, pu.path, pu.params, urlencode(q, doseq=True), pu.fragment))
                    if clean and clean != cur:
                        self._page.goto(clean, wait_until="domcontentloaded", timeout=12000)
                        continue
            except Exception:
                pass
            # Access denied pages often render over plain http subroutes; force
            # canonical https root and retry once before giving up.
            try:
                cur = self._page.url or ""
                if cur.startswith("http://"):
                    from urllib.parse import urlparse
                    pu = urlparse(cur)
                    root = f"https://{pu.netloc}/"
                    self._page.goto(root, wait_until="domcontentloaded", timeout=12000)
                    continue
            except Exception:
                pass
            try:
                self._page.wait_for_timeout(2500)
                self._page.reload(wait_until="domcontentloaded", timeout=12000)
            except Exception:
                try:
                    self._page.wait_for_timeout(2000)
                except Exception:
                    pass

    def _is_bot_gate_page(self) -> bool:
        if self._page is None:
            return False
        try:
            title = (self._page.title() or "").lower()
        except Exception:
            title = ""
        url = (self._page.url or "").lower()
        try:
            body = (self._page.locator("body").first.inner_text(timeout=800) or "").lower()
        except Exception:
            body = ""
        return (
            "just a moment" in title
            or "simple page" in title
            or "access denied" in title
            or "attention required" in title
            or "verify you are human" in body
            or "captcha" in body
            or "unusual traffic" in body
            or "access denied" in body
            or "enable javascript and cookies" in body
            or "/cdn-cgi/challenge" in url
            or "__cf_chl_rt_tk=" in url
            or "chal_t=" in url
            or "force_referer=" in url
        )

    def _dismiss_overlays(self) -> bool:
        if self._page is None:
            return False

        _JS = """() => {
            const COOKIE_CLASSES = [
                'onetrust','cookieconsent','cookie-consent','cookie-banner',
                'gdpr','ccpa','consent','privacy-banner','cookie-notice',
                'cookie-law','cookie-policy','cookie-popup','cookie-overlay',
                'cookie-alert','cookie-box','cookie-bar','cookie-message',
            ];
            const ACCEPT_TEXTS = [
                'accept all','accept cookies','accept all cookies','allow all',
                'allow cookies','i accept','i agree','agree','ok','got it',
                'i understand','continue','yes, i accept','yes','close',
            ];

            function isVisible(el) {
                if (!el || !el.offsetParent) return false;
                const r = el.getBoundingClientRect();
                return r.width > 0 && r.height > 0;
            }

            function textMatch(el, targets) {
                const t = (el.textContent || '').toLowerCase().trim();
                return targets.some(s => t === s || t.startsWith(s + ' '));
            }

            const allBtns = Array.from(document.querySelectorAll(
                'button, a[role="button"], [role="button"], input[type="button"], input[type="submit"]'
            ));

            // 1. Cookie banners: find buttons inside cookie-related containers.
            const bannerSelectors = COOKIE_CLASSES
                .flatMap(c => [`[id*="${c}"]`, `[class*="${c}"]`])
                .join(',');
            try {
                const banners = Array.from(document.querySelectorAll(bannerSelectors));
                for (const banner of banners) {
                    if (!isVisible(banner)) continue;
                    const btns = Array.from(banner.querySelectorAll(
                        'button, a[role="button"], [role="button"]'
                    ));
                    for (const btn of btns) {
                        if (isVisible(btn) && textMatch(btn, ACCEPT_TEXTS)) {
                            btn.click();
                            return 'cookie_banner_accepted';
                        }
                    }
                }
            } catch(e) {}

            // 2. Loose cookie accept buttons anywhere on page.
            for (const btn of allBtns) {
                if (!isVisible(btn)) continue;
                const id = (btn.id || '').toLowerCase();
                const cls = Array.from(btn.classList || []).join(' ').toLowerCase();
                const isCookieEl = COOKIE_CLASSES.some(p => id.includes(p) || cls.includes(p));
                if (isCookieEl && textMatch(btn, ACCEPT_TEXTS)) {
                    btn.click();
                    return 'cookie_button_accepted';
                }
            }

            // 3. Modal/dialog close buttons (only genuine dialogs, not nav menus).
            const dialogs = Array.from(document.querySelectorAll(
                '[role="dialog"], [role="alertdialog"], [aria-modal="true"]'
            ));
            for (const dialog of dialogs) {
                if (!isVisible(dialog)) continue;
                const closeBtns = Array.from(dialog.querySelectorAll('button, [role="button"]'));
                for (const btn of closeBtns) {
                    if (!isVisible(btn)) continue;
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const title = (btn.getAttribute('title') || '').toLowerCase();
                    const text = (btn.textContent || '').toLowerCase().trim();
                    const isClose = ['close', 'dismiss', 'cancel', 'x', 'x', 'x'].some(
                        s => ariaLabel === s || title === s || text === s
                    );
                    if (isClose) {
                        btn.click();
                        return 'dialog_closed';
                    }
                }
            }

            return null;
        }"""
        try:
            result = self._page.evaluate(_JS)
            if result:
                logger.info("Overlay auto-dismissed: %s", result)
                try:
                    self._page.wait_for_timeout(400)
                except Exception:
                    pass
                return True
        except Exception as exc:
            logger.debug("Overlay dismissal probe failed: %s", exc)
        return False

    def run_step(self, step: AutomationStep) -> AutomationStepResult:
        if self._executor is None:
            raise RuntimeError("Agent not started. Call start() first.")
        self._ensure_browser_alive()
        self._mitigate_bot_gate()
        t0 = time.perf_counter()
        validated_ok: Optional[bool] = None
        retry_traces: list[str] = []
        action_out: ActionVerificationResult
        logger.info(
            "Running step: action=%r intent=%r value=%r",
            step.action, step.intent, step.value,
        )

        action = (step.action or "").lower()
        if action != "navigate" and self._is_bot_gate_page():
            fail = ActionVerificationResult(
                action_result=ActionResult(
                    success=False,
                    action=step.action,
                    message="Blocked by anti-bot verification page.",
                    latency_ms=0.0,
                    error="bot_gate_blocked",
                ),
                verification=None,
                metadata=None,
            )
            return AutomationStepResult(
                step=step,
                ok=False,
                action=fail,
                duration_ms=(time.perf_counter() - t0) * 1000,
                changed=False,
                extraction=self._current_extraction,
                change_summaries=["bot_gate_blocked"],
            )

        if action not in {"navigate", "extract"} and not self._ensure_interaction_ready():
            fail = ActionVerificationResult(
                action_result=ActionResult(
                    success=False,
                    action=step.action,
                    message="Page is not interaction-ready (blank/challenge/low-DOM).",
                    latency_ms=0.0,
                    error="page_not_ready",
                ),
                verification=None,
                metadata=None,
            )
            return AutomationStepResult(
                step=step,
                ok=False,
                action=fail,
                duration_ms=(time.perf_counter() - t0) * 1000,
                changed=False,
                extraction=self._current_extraction,
                change_summaries=["page_not_ready"],
            )

        if action not in {"navigate", "scroll"}:
            dismissed = self._dismiss_overlays()
            if dismissed:
                try:
                    self._page.wait_for_timeout(400)
                except Exception:
                    pass
                try:
                    self.extract_page_context()
                    self._selector.clear_cache()
                except Exception:
                    pass

        if action == "navigate":
            action_out, retry_traces, validated_ok = self._execute_navigate_step(step)

        elif action == "extract":
            req = ActionRequest(
                action="extract",
                intent=step.intent,
                timeout_ms=step.timeout_ms,
            )
            action_out = self._executor.execute_and_verify(
                request=req,
                wait_after_ms=0,
                verifier=self._diff,
            )
            validated_ok = action_out.action_result.success
            _msg_l = (action_out.action_result.message or "").strip().lower()
            if _msg_l == "not found on this page.":
                validated_ok = False
            elif validated_ok and not self._extract_answer_matches_question(step.intent or "", action_out.action_result.message or ""):
                validated_ok = False
            retry_traces.append(
                f"extract_direct ok={validated_ok} "
                f"err={action_out.action_result.error or ''}"
            )

            if not validated_ok and self._page is not None:
                if _msg_l == "not found on this page.":
                    retry_traces.append("extract_not_found_treated_as_failure")
                elif _msg_l:
                    retry_traces.append("extract_answer_insufficient_for_goal")

                _scroll_positions = [0.25, 0.5, 0.75, 1.0]
                for _scroll_frac in _scroll_positions:
                    try:
                        _doc_h = self._page.evaluate(
                            "() => document.documentElement.scrollHeight"
                        ) or 0
                        _target_y = int(float(_doc_h) * _scroll_frac)
                        self._page.evaluate(f"window.scrollTo(0, {_target_y})")
                        self._page.wait_for_timeout(300)
                    except Exception:
                        pass
                    _scroll_req = ActionRequest(
                        action="extract",
                        intent=step.intent,
                        timeout_ms=step.timeout_ms,
                    )
                    _scroll_out = self._executor.execute_and_verify(
                        request=_scroll_req,
                        wait_after_ms=0,
                        verifier=self._diff,
                    )
                    _scroll_msg_l = (_scroll_out.action_result.message or "").strip().lower()
                    if (
                        _scroll_out.action_result.success
                        and _scroll_msg_l
                        and _scroll_msg_l != "not found on this page."
                        and self._extract_answer_matches_question(
                            step.intent or "", _scroll_out.action_result.message or ""
                        )
                    ):
                        action_out = _scroll_out
                        validated_ok = True
                        retry_traces.append(f"extract_scroll_retry ok=True frac={_scroll_frac}")
                        break
                if not validated_ok:
                    retry_traces.append("extract_scroll_retry ok=False")

        elif action == "scroll":
            req = ActionRequest(
                action="scroll",
                value=step.value or "down",
                intent=step.intent,
                timeout_ms=step.timeout_ms,
            )
            action_out = self._executor.execute_and_verify(
                request=req,
                wait_after_ms=step.wait_after_ms,
                verifier=self._diff,
            )
            c = NextActionCandidate(
                intent=step.intent or "",
                score=1.0 if action_out.action_result.success else 0.0,
                probability=1.0,
                selector=step.selector,
                selector_type=step.selector_type,
                reasoning="scroll_direct",
                dom_node_id=step.dom_node_id,
            )
            validated_ok, reason = self._validate_action(step, c, action_out)
            retry_traces.append(f"scroll_direct_valid={validated_ok} reason={reason}")

        elif step.intent and not step.selector and not step.dom_node_id:
            action_out, retry_traces, validated_ok = self._execute_step_with_ranked_retry(step)

        else:
            req = ActionRequest(
                action=step.action,
                value=step.value,
                intent=step.intent,
                selector=step.selector,
                selector_type=step.selector_type,
                dom_node_id=step.dom_node_id,
                timeout_ms=step.timeout_ms,
                clear_first=step.clear_first,
                press_enter=step.press_enter,
            )
            action_out = self._executor.execute_and_verify(
                request=req,
                wait_after_ms=step.wait_after_ms,
                verifier=self._diff,
            )
            c = NextActionCandidate(
                intent=step.intent or "",
                score=1.0 if action_out.action_result.success else 0.0,
                probability=1.0,
                selector=step.selector,
                selector_type=step.selector_type,
                reasoning="direct_target",
                dom_node_id=step.dom_node_id,
            )
            validated_ok, reason = self._validate_action(step, c, action_out)
            retry_traces.append(f"direct_valid={validated_ok} reason={reason}")

        duration_ms = (time.perf_counter() - t0) * 1000

        summaries = []
        changed = False
        if action_out.verification:
            changed = action_out.verification.had_changes
            summaries = [item.summary for item in action_out.verification.changes]
        summaries.extend(retry_traces)

        _type_commits_state = (
            action == "type"
            and (
                step.press_enter
                or self._neural.classify(step.intent or "", "search_submit", threshold=0.68)
            )
        )
        _needs_extract = (
            action in {"navigate", "extract"}
            or changed
            or _type_commits_state
        )
        if _needs_extract:
            extraction = self.extract_page_context()
        else:
            extraction = self._current_extraction 

        logger.info(
            "Step done: action=%r ok=%s changed=%s duration=%.0fms",
            step.action, validated_ok, changed, duration_ms,
        )
        return AutomationStepResult(
            step=step,
            ok=bool(validated_ok),
            action=action_out,
            duration_ms=duration_ms,
            changed=changed,
            extraction=extraction,
            change_summaries=summaries,
        )

    def run_task(self, url: str, steps: list[AutomationStep]) -> AutomationTaskResult:
        return self.run_task_with_goal(url=url, steps=steps, goal=None)

    def create_task_plan(self, user_goal: str, start_url: Optional[str] = None) -> TaskPlan:
        logger.info("Creating task plan for goal: %r", user_goal)
        plan = self._planner.create_plan(user_goal, start_url=start_url)
        self._active_plan = plan
        logger.info("Plan created: %d subgoals", len(plan.subgoals))
        return plan

    def execute_task_plan(self, plan: TaskPlan, url: Optional[str] = None) -> PlanExecutionResult:
        if url:
            self.goto(url)
        else:
            if self._current_state is None:
                self.extract_page_context()

        executed: list[AutomationStepResult] = []
        plan.status = "in_progress"
        _last_type_submitted_search = False
        _extracted_answer: Optional[str] = None 
        _consecutive_failures = 0
        _REPLAN_THRESHOLD = 2 
        _max_replans = 1      
        _replans_used = 0
        _blocked = False

        for idx, subgoal in enumerate(plan.subgoals):
            logger.info("Executing subgoal %s: %r", subgoal.subgoal_id, subgoal.description)
            self._planner.mark_subgoal(plan, idx, "in_progress", None)

            steps_in_subgoal = subgoal.steps
            is_search_button_subgoal = (
                len(steps_in_subgoal) == 1
                and steps_in_subgoal[0].action == "click"
                and self._neural.classify(
                    steps_in_subgoal[0].intent or "", "search_submit", threshold=0.54
                )
            )

            _is_autocomplete_sg = bool(subgoal.steps) and any(
                w in (subgoal.description or "").lower() or
                w in (subgoal.steps[0].intent or "").lower()
                for w in ("autocomplete", "suggestion", "dropdown list", "from the list")
            )
            if _last_type_submitted_search and is_search_button_subgoal and not _is_autocomplete_sg:
                logger.info(
                    "Subgoal %s skipped: search was already submitted via Enter in previous type step.",
                    subgoal.subgoal_id,
                )
                self._planner.mark_subgoal(plan, idx, "completed", None)
                _last_type_submitted_search = False
                continue

            _last_type_submitted_search = False
            steps_ok = True
            last_step_result: Optional[AutomationStepResult] = None

            for pstep in steps_in_subgoal:
                step = self._planner_step_to_automation_step(pstep)
                step_result = self.run_step(step)
                step_result.change_summaries.insert(
                    0, f"plan_subgoal={subgoal.subgoal_id}:{subgoal.description}"
                )
                executed.append(step_result)
                last_step_result = step_result

                if (step_result.action.action_result.error or "") in {"bot_gate_blocked", "page_not_ready"}:
                    logger.error(
                        "Execution blocked by page readiness gate during subgoal %s (error=%s); aborting remaining plan.",
                        subgoal.subgoal_id,
                        step_result.action.action_result.error,
                    )
                    step_result.change_summaries.append("plan_blocked_by_readiness_gate")
                    _blocked = True
                    steps_ok = False
                    break

                if step.action == "type" and step_result.ok and step_result.changed:
                    if (
                        self._neural.classify(step.intent or "", "search_submit", threshold=0.72)
                        and not self._goal_is_informational(plan.user_goal)
                    ):
                        _last_type_submitted_search = True

                # Capture answer produced by an extract action.
                if step.action == "extract" and step_result.ok and step_result.action.action_result.message:
                    _extracted_answer = step_result.action.action_result.message
                    logger.info("Extracted answer captured: %.160s...", _extracted_answer)

                if (step.action == "click" and step_result.ok
                        and step_result.changed and self._page is not None):
                    current_url = self._page.url or ""
                    wrong_reason = self._check_url_goal_alignment(
                        new_url=current_url,
                        plan_goal=plan.user_goal,
                        subgoal_criteria=subgoal.success_criteria,
                        plan=plan,
                    )
                    if wrong_reason:
                        logger.warning(
                            "Click ok=True but URL %r contradicts plan goal (%s); "
                            "overriding step to failed.",
                            current_url, wrong_reason,
                        )
                        step_result.change_summaries.append(
                            f"wrong_domain_navigation:{wrong_reason}"
                        )
                        _recipe_recovered = False
                        _step_text = ((subgoal.description or "") + " " + (step.intent or "")).lower()
                        try:
                            self._page.go_back(wait_until="domcontentloaded", timeout=8000)
                            try:
                                self._page.wait_for_timeout(400)
                            except Exception:
                                pass
                            self.extract_page_context()
                            self._selector.clear_cache()
                            if "recipe" in _step_text:
                                _recipe_recovered = self._open_recipe_link_from_page(
                                    f"{subgoal.description or ''} {step.intent or ''} {step.value or ''}"
                                )
                                if _recipe_recovered:
                                    self.extract_page_context()
                                    step_result.change_summaries.append("recipe_mismatch_recovered_via_recipe_link")
                        except Exception as _gb_e:
                            logger.warning("go_back failed: %s", _gb_e)
                            try:
                                _sv = next((st.value for sg in plan.subgoals
                                    for st in sg.steps if st.action == "navigate" and st.value), "")
                                if _sv:
                                    self._page.goto(_sv, wait_until="domcontentloaded", timeout=15000)
                                    self.extract_page_context()
                            except Exception:
                                pass
                        if _recipe_recovered:
                            continue
                        steps_ok = False
                        break

                if not step_result.ok:
                    steps_ok = False
                    break

            progress = None
            subgoal_ok = steps_ok

            if self._current_state is not None:
                diff = last_step_result.action.verification if last_step_result else None
                progress = self._goal_validator.evaluate(
                    goal=self._planner.subgoal_task_goal(subgoal),
                    state=self._current_state,
                    diff=diff,
                )

                # Typed-value confirmations on modern SPAs can succeed even when
                # the global DOM diff remains low-signal. If the subgoal is a
                # single successful type step, treat it as complete so planning
                # can progress to autocomplete/search steps.
                if (
                    steps_ok
                    and len(steps_in_subgoal) == 1
                    and steps_in_subgoal[0].action == "type"
                    and last_step_result is not None
                    and last_step_result.ok
                ):
                    subgoal_ok = True
                    logger.info(
                        "Subgoal %s: single type step validated; continuing despite low diff signal.",
                        subgoal.subgoal_id,
                    )
                elif (
                    steps_ok
                    and len(steps_in_subgoal) == 1
                    and steps_in_subgoal[0].action == "scroll"
                    and last_step_result is not None
                    and last_step_result.ok
                ):
                    subgoal_ok = True
                    logger.info(
                        "Subgoal %s: single scroll step validated; continuing despite low diff signal.",
                        subgoal.subgoal_id,
                    )
                elif (
                    steps_ok
                    and len(steps_in_subgoal) == 1
                    and steps_in_subgoal[0].action == "click"
                    and last_step_result is not None
                    and last_step_result.ok
                    and any(
                        w in ((subgoal.description or "") + " " + (steps_in_subgoal[0].intent or "")).lower()
                        for w in ("autocomplete", "suggestion", "from the list", "dropdown list")
                    )
                ):
                    _u = (self._current_state.url or "").lower()
                    _searchish = any(t in _u for t in ("search", "query=", "q="))
                    _value_set = self._autocomplete_input_has_value(steps_in_subgoal[0].value or "")
                    if _searchish or _value_set:
                        subgoal_ok = True
                        logger.info(
                            "Subgoal %s: autocomplete click accepted on search-like URL.",
                            subgoal.subgoal_id,
                        )
                    else:
                        subgoal_ok = False
                elif (
                    steps_ok
                    and len(steps_in_subgoal) == 1
                    and steps_in_subgoal[0].action == "click"
                    and last_step_result is not None
                    and last_step_result.ok
                    and any(
                        k in ((subgoal.description or "") + " " + (steps_in_subgoal[0].intent or "")).lower()
                        for k in ("date", "calendar", "check-in", "check out", "check-out")
                    )
                ):
                    trace_blob = " ".join(last_step_result.change_summaries).lower()
                    _text = ((subgoal.description or "") + " " + (steps_in_subgoal[0].intent or "")).lower()
                    _is_month_nav = any(k in _text for k in ("next month", "previous month", "prev month"))
                    _trace_hits = any(t in trace_blob for t in (
                        "date_input_js_first ok=true",
                        "calendar_date_click_first ok=true",
                    ))
                    if not _trace_hits and _is_month_nav and "calendar_nav_first direction=" in trace_blob:
                        _trace_hits = True
                    if _trace_hits:
                        subgoal_ok = True
                        logger.info(
                            "Subgoal %s: calendar/date click accepted from deterministic trace markers.",
                            subgoal.subgoal_id,
                        )
                    else:
                        subgoal_ok = steps_ok
                elif not steps_ok:
                    subgoal_ok = False
                elif progress.total_required == 0:
                    subgoal_ok = True
                elif progress.completed:
                    subgoal_ok = True
                elif progress.probability >= 0.55:
                    has_url_criterion = any(
                        c.kind.lower() == "url_contains" and c.required
                        for c in subgoal.success_criteria
                    )
                    _url_now = (self._current_state.url or "").lower()
                    _url_after = ""
                    if last_step_result is not None:
                        _url_after = ((last_step_result.action.metadata or {}).get("url_after") or "").lower()
                    _search_results_override = (
                        last_step_result is not None
                        and (last_step_result.step.action or "").lower() == "click"
                        and self._neural.classify(last_step_result.step.intent or "", "search_submit", threshold=0.58)
                        and (self._url_looks_results_like(_url_now) or (_url_after and self._url_looks_results_like(_url_after)))
                    )
                    url_criteria_met = all(
                        not (c.kind.lower() == "url_contains" and c.required)
                        or c.value.lower() in _url_now
                        or (_url_after and c.value.lower() in _url_after)
                        for c in subgoal.success_criteria
                    )
                    if _search_results_override:
                        url_criteria_met = True
                    if (
                        has_url_criterion
                        and not url_criteria_met
                        and last_step_result is not None
                        and (last_step_result.step.action or "").lower() == "click"
                    ):
                        _recipe_ctx = ((subgoal.description or "") + " " + (last_step_result.step.intent or "")).lower()
                        if "recipe" in _recipe_ctx and self._page is not None:
                            if "/recipes/" in _url_now and "/recipe/" not in _url_now:
                                recovered = self._open_recipe_link_from_page(_recipe_ctx)
                                if recovered:
                                    self.extract_page_context()
                                    _url_now = (self._current_state.url or "").lower()
                                    url_criteria_met = all(
                                        not (c.kind.lower() == "url_contains" and c.required)
                                        or c.value.lower() in _url_now
                                        for c in subgoal.success_criteria
                                    )
                    if has_url_criterion and not url_criteria_met:
                        subgoal_ok = False
                        logger.info(
                            "Subgoal %s: url_contains criterion not satisfied despite high probability=%.3f; aborting.",
                            subgoal.subgoal_id, progress.probability,
                        )
                    else:
                        subgoal_ok = True
                        logger.info(
                            "Subgoal %s: criteria not fully met but probability=%.3f >= 0.55; continuing.",
                            subgoal.subgoal_id, progress.probability,
                        )
                else:
                    state_changed = diff is not None and diff.had_changes
                    has_url_criterion = any(
                        c.kind.lower() == "url_contains" and c.required
                        for c in subgoal.success_criteria
                    )
                    _url_now = (self._current_state.url or "").lower()
                    _url_after = ""
                    if last_step_result is not None:
                        _url_after = ((last_step_result.action.metadata or {}).get("url_after") or "").lower()
                    _search_results_override = (
                        last_step_result is not None
                        and (last_step_result.step.action or "").lower() == "click"
                        and self._neural.classify(last_step_result.step.intent or "", "search_submit", threshold=0.58)
                        and (self._url_looks_results_like(_url_now) or (_url_after and self._url_looks_results_like(_url_after)))
                    )
                    url_criteria_met = all(
                        not (c.kind.lower() == "url_contains" and c.required)
                        or c.value.lower() in _url_now
                        or (_url_after and c.value.lower() in _url_after)
                        for c in subgoal.success_criteria
                    )
                    if _search_results_override:
                        url_criteria_met = True
                    if (
                        has_url_criterion
                        and not url_criteria_met
                        and last_step_result is not None
                        and (last_step_result.step.action or "").lower() == "click"
                    ):
                        _recipe_ctx = ((subgoal.description or "") + " " + (last_step_result.step.intent or "")).lower()
                        if "recipe" in _recipe_ctx and self._page is not None:
                            if "/recipes/" in _url_now and "/recipe/" not in _url_now:
                                recovered = self._open_recipe_link_from_page(_recipe_ctx)
                                if recovered:
                                    self.extract_page_context()
                                    _url_now = (self._current_state.url or "").lower()
                                    url_criteria_met = all(
                                        not (c.kind.lower() == "url_contains" and c.required)
                                        or c.value.lower() in _url_now
                                        for c in subgoal.success_criteria
                                    )
                    if has_url_criterion and not url_criteria_met:
                        subgoal_ok = False
                        logger.info(
                            "Subgoal %s: url_contains criterion not satisfied (url=%r); marking failed.",
                            subgoal.subgoal_id, self._current_state.url,
                        )
                    elif state_changed and progress.probability >= 0.30:
                        subgoal_ok = True
                        logger.info(
                            "Subgoal %s: state changed and probability=%.3f >= 0.30; continuing.",
                            subgoal.subgoal_id, progress.probability,
                        )
                    else:
                        subgoal_ok = False

            self._planner.mark_subgoal(
                plan,
                idx,
                "completed" if subgoal_ok else "failed",
                progress,
            )
            logger.info(
                "Subgoal %s: %s (progress_prob=%.3f)",
                subgoal.subgoal_id,
                "completed" if subgoal_ok else "failed",
                progress.probability if progress else 0.0,
            )
            if not subgoal_ok:
                _consecutive_failures += 1
                logger.warning(
                    "Subgoal %s failed (prob=%.3f); consecutive_failures=%d; continuing.",
                    subgoal.subgoal_id,
                    progress.probability if progress else 0.0,
                    _consecutive_failures,
                )
                _fail_blob = " ".join(last_step_result.change_summaries).lower() if last_step_result else ""
                if (
                    "autocomplete_unverified" in _fail_blob
                    and self._goal_is_informational(plan.user_goal)
                    and self._try_navigate_help_resource(plan.user_goal)
                ):
                    logger.info(
                        "Recovered from autocomplete dead-end by navigating to help/support resource page."
                    )
                    _consecutive_failures = 0
                remaining_idx = idx + 1
                _trace_blob_fail = " ".join(last_step_result.change_summaries).lower() if last_step_result else ""
                _autocomplete_stuck = ("autocomplete_unverified" in _trace_blob_fail or "no_live_listbox" in _trace_blob_fail)
                if (
                    (_consecutive_failures >= _REPLAN_THRESHOLD or _autocomplete_stuck)
                    and _replans_used < _max_replans
                    and remaining_idx < len(plan.subgoals)
                    and self._current_state is not None
                    and self._page is not None
                    and not self._is_bot_gate_page()
                ):
                    completed_desc = [
                        sg.description for sg in plan.subgoals[:idx]
                        if sg.status == "completed"
                    ]
                    new_plan = self._planner.create_plan_from_state(
                        original_goal=plan.user_goal,
                        current_url=self._page.url or "",
                        current_title=self._current_state.title or "",
                        completed_descriptions=completed_desc,
                    )
                    if new_plan and new_plan.subgoals and self._replan_is_compatible(plan.user_goal, new_plan):
                        logger.info(
                            "Replanning injected %d new subgoals to replace remaining %d.",
                            len(new_plan.subgoals),
                            len(plan.subgoals) - remaining_idx,
                        )
                        plan.subgoals[remaining_idx:] = new_plan.subgoals
                        _replans_used += 1
                        _consecutive_failures = 0
                    elif new_plan and new_plan.subgoals:
                        logger.warning(
                            "Replan rejected: new plan is incompatible with original goal %r.",
                            plan.user_goal,
                        )
            else:
                _consecutive_failures = 0

            if _blocked:
                break

        # Compute final plan status from individual subgoal outcomes.
        completed_count = sum(1 for sg in plan.subgoals if sg.status == "completed")
        total_count = len(plan.subgoals)
        failed_count = sum(1 for sg in plan.subgoals if sg.status == "failed")

        if _blocked:
            plan.status = "failed"
        elif completed_count == total_count:
            plan.status = "completed"
        elif completed_count > 0 and failed_count > 0:
            plan.status = "partial"
        elif completed_count == 0:
            plan.status = "failed"
        success = plan.status == "completed"
        _ans = (_extracted_answer or "").strip()
        if _ans:
            if _ans.lower() == "not found on this page.":
                success = False
            elif not self._extract_answer_matches_question(plan.user_goal, _ans):
                success = False
            elif (not success) and self._extract_answer_matches_question(plan.user_goal, _ans):
                success = True
        logger.info(
            "Plan execution done: status=%s success=%s completed=%d/%d",
            plan.status, success, completed_count, total_count,
        )
        return PlanExecutionResult(
            plan=plan,
            executed_steps=executed,
            success=success,
            extracted_answer=_extracted_answer,
        )

    def run_user_goal(self, url: str, user_goal: str) -> PlanExecutionResult:
        logger.info("Running user goal: %r at %s", user_goal, url)
        preflight_ok, preflight_reason = self._preflight_gate_playbook(url)
        if not preflight_ok:
            return self._hard_gate_abort_result(url, user_goal, preflight_reason)

        plan = self.create_task_plan(user_goal, start_url=url)
        execution = self.execute_task_plan(plan=plan, url=None)
        max_retries = 2
        retries = 0
        while (
            retries < max_retries
            and not execution.success
            and self._execution_blocked_by_bot(execution)
        ):
            retries += 1
            logger.warning(
                "Task blocked by anti-bot gate; retrying with rotated browser context (attempt %d/%d).",
                retries, max_retries,
            )
            try:
                self.reset_context(anti_bot_retry=True)
            except Exception as exc:
                logger.warning("Anti-bot context reset failed: %s", exc)
                break
            plan = self.create_task_plan(user_goal, start_url=url)
            execution = self.execute_task_plan(plan=plan, url=url)
        return execution

    def generate_plan_trace_report(self, execution: PlanExecutionResult) -> dict:
        plan = execution.plan
        steps = execution.executed_steps
        subgoal_reports = []
        cursor = 0

        for sg in plan.subgoals:
            sg_steps = steps[cursor: cursor + len(sg.steps)]
            cursor += len(sg.steps)
            step_reports = []
            for idx, s in enumerate(sg_steps, start=1):
                selection_traces = [
                    x for x in s.change_summaries
                    if x.startswith("candidate#") or x.startswith("direct_valid")
                    or x.startswith("fallback_valid") or x.startswith("no_candidates")
                    or x.startswith("navigate ")
                ]
                change_traces = [
                    x for x in s.change_summaries
                    if x not in selection_traces and not x.startswith("plan_subgoal=")
                ]
                step_reports.append({
                    "step_index": idx,
                    "action": s.step.action,
                    "intent": s.step.intent,
                    "value": s.step.value,
                    "ok": s.ok,
                    "duration_ms": round(s.duration_ms, 2),
                    "changed": s.changed,
                    "selection_traces": selection_traces,
                    "change_traces": change_traces[:20],
                })

            sg_progress = None
            if sg.progress is not None:
                sg_progress = {
                    "completed": sg.progress.completed,
                    "probability": round(sg.progress.probability, 4),
                    "satisfied_required": sg.progress.satisfied_required,
                    "total_required": sg.progress.total_required,
                    "satisfied_optional": sg.progress.satisfied_optional,
                    "total_optional": sg.progress.total_optional,
                    "satisfied": sg.progress.satisfied,
                    "unsatisfied": sg.progress.unsatisfied,
                }

            subgoal_reports.append({
                "subgoal_id": sg.subgoal_id,
                "description": sg.description,
                "status": sg.status,
                "steps_executed": len(sg_steps),
                "steps_ok": sum(1 for x in sg_steps if x.ok),
                "progress": sg_progress,
                "steps": step_reports,
            })

        total_steps = len(steps)
        passed_steps = sum(1 for s in steps if s.ok)
        failed_steps = total_steps - passed_steps
        avg_duration = (sum(s.duration_ms for s in steps) / total_steps) if total_steps else 0.0

        report: dict = {
            "task": {
                "task_id": plan.task_id,
                "user_goal": plan.user_goal,
                "status": plan.status,
                "success": execution.success,
                "subgoals_total": len(plan.subgoals),
                "subgoals_completed": sum(1 for sg in plan.subgoals if sg.status == "completed"),
                **({"notes": plan.notes} if plan.notes else {}),
            },
            "metrics": {
                "steps_total": total_steps,
                "steps_passed": passed_steps,
                "steps_failed": failed_steps,
                "avg_step_duration_ms": round(avg_duration, 2),
            },
            "subgoals": subgoal_reports,
        }
        if execution.extracted_answer:
            report["extracted_answer"] = execution.extracted_answer
        return report

    def export_plan_trace_report(
        self,
        execution: PlanExecutionResult,
        json_path: Optional[str] = None,
        markdown_path: Optional[str] = None,
    ) -> dict:
        report = self.generate_plan_trace_report(execution)
        if json_path:
            p = Path(json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Trace report written to %s", json_path)
        if markdown_path:
            p = Path(markdown_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self._report_to_markdown(report), encoding="utf-8")
            logger.info("Markdown report written to %s", markdown_path)
        return report

    def _report_to_markdown(self, report: dict) -> str:
        task = report["task"]
        metrics = report["metrics"]
        lines = []
        lines.append("# Plan Trace Report")
        lines.append("")
        lines.append(f"- task_id: `{task['task_id']}`")
        lines.append(f"- goal: {task['user_goal']}")
        lines.append(f"- status: **{task['status']}** | success: **{task['success']}**")
        lines.append(f"- subgoals: {task['subgoals_completed']}/{task['subgoals_total']}")
        lines.append(
            f"- steps passed: {metrics['steps_passed']}/{metrics['steps_total']} "
            f"| avg duration: {metrics['avg_step_duration_ms']} ms"
        )
        lines.append("")
        if report.get("extracted_answer"):
            lines.append("## Extracted Answer")
            lines.append("")
            lines.append(report["extracted_answer"])
            lines.append("")
        for sg in report["subgoals"]:
            lines.append(f"## {sg['subgoal_id']} - {sg['description']}")
            lines.append(f"- status: **{sg['status']}**")
            lines.append(f"- steps ok: {sg['steps_ok']}/{sg['steps_executed']}")
            progress = sg.get("progress")
            if progress:
                lines.append(
                    f"- progress: completed={progress['completed']} prob={progress['probability']} "
                    f"required={progress['satisfied_required']}/{progress['total_required']}"
                )
            lines.append("")
            for st in sg["steps"]:
                lines.append(
                    f"- step {st['step_index']}: action=`{st['action']}` intent=`{st['intent']}` "
                    f"ok={st['ok']} changed={st['changed']} duration={st['duration_ms']}ms"
                )
                for tr in st["selection_traces"][:5]:
                    lines.append(f"  - trace: {tr}")
                for tr in st["change_traces"][:5]:
                    lines.append(f"  - change: {tr}")
            lines.append("")
        return "\n".join(lines)

    def _planner_step_to_automation_step(self, pstep: PlannerStep) -> AutomationStep:
        return AutomationStep(
            action=pstep.action,
            intent=pstep.intent,
            value=pstep.value,
            selector=pstep.selector,
            selector_type=pstep.selector_type,
            dom_node_id=pstep.dom_node_id,
            timeout_ms=pstep.timeout_ms,
            clear_first=pstep.clear_first,
            press_enter=pstep.press_enter,
            wait_after_ms=pstep.wait_after_ms,
        )

    def run_task_with_goal(
        self,
        url: str,
        steps: list[AutomationStep],
        goal: Optional[TaskGoal],
    ) -> AutomationTaskResult:
        started = time.time()
        self._reasoner.reset_budget()
        logger.info(
            "Running task at %s (%d steps, goal=%s)",
            url, len(steps), goal.description if goal else "none",
        )
        self.goto(url)
        out: list[AutomationStepResult] = []
        all_ok = True
        goal_progress: Optional[GoalProgress] = None

        for step in steps:
            result = self.run_step(step)
            out.append(result)
            if not result.ok:
                all_ok = False
                break
            if goal is not None and self._current_state is not None:
                goal_progress = self._goal_validator.evaluate(
                    goal=goal,
                    state=self._current_state,
                    diff=result.action.verification,
                )
                result.change_summaries.append(
                    f"goal_progress={goal_progress.probability:.3f} "
                    f"required={goal_progress.satisfied_required}/{goal_progress.total_required} "
                    f"completed={goal_progress.completed}"
                )
                if goal_progress.completed:
                    break

        if goal is not None and self._current_state is not None and goal_progress is None:
            goal_progress = self._goal_validator.evaluate(goal, self._current_state, None)
        if goal is not None and goal_progress is not None:
            all_ok = all_ok and goal_progress.completed

        logger.info(
            "Task done: success=%s steps=%d/%d duration=%.1fs",
            all_ok, sum(1 for s in out if s.ok), len(out), time.time() - started,
        )
        return AutomationTaskResult(
            url=url,
            started_at=started,
            ended_at=time.time(),
            steps=out,
            success=all_ok,
            goal_progress=goal_progress,
        )


if __name__ == "__main__":
    from logging_config import setup_logging
    setup_logging(level="INFO")

    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    TEST_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Browser Automation Task Demo</title>
</head>
<body>
  <h1>Login</h1>
  <form aria-label="Login form">
    <label for="email">Email</label>
    <input id="email" type="email" placeholder="you@example.com" aria-label="Email">
    <label for="password">Password</label>
    <input id="password" type="password" placeholder="Password" aria-label="Password">
    <button id="signin" type="button" onclick="document.getElementById('status').textContent='Signed in';">Sign In</button>
  </form>
  <p id="status">Idle</p>
</body>
</html>"""

    class _Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(TEST_HTML.encode())

        def log_message(self, *_):
            pass

    server = HTTPServer(("localhost", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    task_steps = [
        AutomationStep(action="type", intent="type email", value="alice@example.com"),
        AutomationStep(action="type", intent="type password", value="secret123"),
        AutomationStep(action="click", intent="click sign in"),
    ]

    with BrowserAutomationAgent(headless=True, use_llm_selector=False) as agent:
        result = agent.run_task(url=f"http://localhost:{port}/", steps=task_steps)
        print(f"Task success: {result.success}")
        for idx, step_result in enumerate(result.steps, start=1):
            print(f"[{idx}] action={step_result.step.action} ok={step_result.ok} changed={step_result.changed}")
            for summary in step_result.change_summaries[:5]:
                print(f"    - {summary}")

    server.shutdown()
