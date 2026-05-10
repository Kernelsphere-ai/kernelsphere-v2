import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

from goal_validator import GoalCriterion, GoalProgress, TaskGoal
from env_loader import load_local_env
from logging_config import get_logger

logger = get_logger(__name__)


def _rewrite_past_years(text: str) -> str:
    current_year = _date.today().year

    # Signals that surround a year to indicate it is a calendar date.
    _MONTH_NAMES = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    )
    _DATE_SIGNALS = _MONTH_NAMES + (
        "date", "check-in", "checkin", "check-out", "checkout",
        "departure", "arrival", "flight", "travel", "depart", "arrive",
        "from", "to", "on", "by", "until",
    )

    def _in_date_context(year_start: int, year_end: int, s: str) -> bool:
        window = s[max(0, year_start - 40): year_end + 40].lower()
        return any(sig in window for sig in _DATE_SIGNALS)

    def _sub(m: re.Match) -> str:
        y = int(m.group(0))
        if y >= current_year:
            return m.group(0)
        if _in_date_context(m.start(), m.end(), text):
            return str(current_year)
        return m.group(0)  # Not a date context - leave unchanged

    return re.sub(r"\b(20\d{2})\b", _sub, text)


@dataclass
class PlannerStep:
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
class PlannedSubGoal:
    subgoal_id: str
    description: str
    steps: list[PlannerStep]
    success_criteria: list[GoalCriterion]
    status: str = "pending"
    progress: Optional[GoalProgress] = None


@dataclass
class TaskPlan:
    task_id: str
    user_goal: str
    subgoals: list[PlannedSubGoal]
    status: str = "pending"
    current_subgoal_index: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanExecutionResult:
    plan: TaskPlan
    executed_steps: list[object]
    success: bool
    extracted_answer: Optional[str] = None  # Answer returned by the final extract step


def _gemini_call_with_retry(
    url: str,
    payload: dict,
    timeout_s: int = 15,
    max_retries: int = 2,
    backoff_s: float = 1.5,
) -> Optional[dict]:
    for attempt in range(1, max_retries + 2):
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt <= max_retries:
                wait = backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini HTTP %d on attempt %d; retrying in %.1fs url=%s",
                    exc.code, attempt, wait, url,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Gemini HTTP error %d (non-retryable or max retries hit). url=%s",
                    exc.code, url,
                )
                return None
        except urllib.error.URLError as exc:
            if attempt <= max_retries:
                wait = backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini network error on attempt %d: %s; retrying in %.1fs",
                    attempt, exc.reason, wait,
                )
                time.sleep(wait)
            else:
                logger.warning("Gemini network error after %d attempts: %s", attempt, exc.reason)
                return None
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Gemini unexpected error on attempt %d: %s", attempt, exc)
            return None
    return None


def _ac_in(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ("autocomplete", "suggestion", "dropdown list", "from the list", "from the dropdown"))


def _repair_and_inject_autocomplete(subgoals: list) -> list:
    # search tasks ("help me find events...") into incorrect FAQ navigation.
    _SUPPORT_TOKENS = {
        "lost", "policy", "procedure", "contact", "support",
        "refund", "cancellation", "baggage", "faq", "complaint",
        "disabled", "disability", "customer service",
    }
    _QUERY_ACTION_TOKENS = {
        "find", "search", "book", "buy", "list", "events",
        "closest", "nearby", "location", "store",
    }
    _NON_AC_TYPE_TOKENS = {
        "price", "budget", "max price", "maximum price", "min price",
        "adults", "children", "rooms", "occupancy", "guest", "guests",
        "review score", "rating", "filter",
    }

    def _looks_informational(value: str, intent: str) -> bool:
        vv = (value or "").lower()
        ii = (intent or "").lower()
        blob = f"{vv} {ii}"
        # Explicit support/help-center intents only.
        if any(tok in blob for tok in _SUPPORT_TOKENS):
            return True
        if "help center" in blob or "customer support" in blob:
            return True
        if "what do i do" in blob or "how do i" in blob or "how can i" in blob:
            return True
        # "help me find/list/book" is task execution, not support navigation.
        if "help me " in blob and any(tok in blob for tok in _QUERY_ACTION_TOKENS):
            return False
        # Short noun-like typed values (e.g. "new york", "90028") should never
        # force FAQ redirection.
        if vv and len(vv.split()) <= 4 and not any(tok in vv for tok in _SUPPORT_TOKENS):
            return False
        return False

    last_typed = ""
    # Pass 1: REPAIR - description says autocomplete but intent doesn't
    for sg in subgoals:
        for st in sg.steps:
            if st.action == "type" and st.value:
                last_typed = st.value
        if not _ac_in(sg.description):
            continue
        if not sg.steps:
            continue
        first = sg.steps[0]
        if _looks_informational(last_typed or (first.value or ""), first.intent or ""):
            sg.description = f"Open help/support information related to {last_typed or 'the query'}."
            first.action = "click"
            first.intent = f"Open help or FAQ page related to {last_typed or 'the query'}"
            first.value = None
            continue
        if first.action == "click" and not _ac_in(first.intent or ""):
            tv = last_typed or (first.value or "")
            if tv:
                first.intent = f"Click {tv} from the autocomplete suggestion dropdown list"
                first.value = tv

    # Pass 2: INJECT - missing autocomplete subgoal after destination type step
    _DEST = {"destination", "city", "location", "hotel", "search field", "search bar",
             "search input", "where", "into the"}
    result = []
    for i, sg in enumerate(subgoals):
        result.append(sg)
        for st in sg.steps:
            if st.action != "type" or not st.value:
                continue
            il = (st.intent or "").lower()
            tv = st.value
            # Never inject autocomplete for numeric filter/occupancy entries.
            if re.fullmatch(r"\d+(?:[.,]\d+)?", str(tv).strip()):
                continue
            if any(tok in il for tok in _NON_AC_TYPE_TOKENS):
                continue
            if not any(w in il for w in _DEST):
                continue
            if _looks_informational(tv, il):
                continue
            nxt = subgoals[i + 1] if i + 1 < len(subgoals) else None
            if nxt and (_ac_in(nxt.description) or
                        (nxt.steps and _ac_in(nxt.steps[0].intent or ""))):
                break
            result.append(PlannedSubGoal(
                subgoal_id=f"sg-ac-{i}",
                description=f"Click {tv!r} from the autocomplete suggestion dropdown list",
                steps=[PlannerStep(
                    action="click",
                    intent=f"Click {tv} from the autocomplete suggestion dropdown list",
                    value=tv,
                )],
                success_criteria=[GoalCriterion(kind="state_changed", value="true", required=True)],
            ))
            break
    return result


def _extract_goal_dates(goal_text: str) -> list[tuple[str, int, Optional[int]]]:
    """Extract ordered month/day[/year] tuples from the user goal."""
    text = (goal_text or "")
    out: list[tuple[str, int, Optional[int]]] = []
    month_pat = (
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    )
    # e.g. "April 20", "Apr 20, 2026"
    for m in re.finditer(rf"\b{month_pat}\s+(\d{{1,2}})(?:,\s*(20\d{{2}}))?\b", text, flags=re.I):
        mon = m.group(1)
        day = int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else None
        out.append((mon, day, yr))
    # e.g. "April 3-5, 2026" -> inject second date in same month
    for m in re.finditer(rf"\b{month_pat}\s+(\d{{1,2}})\s*[--]\s*(\d{{1,2}})(?:,\s*(20\d{{2}}))?\b", text, flags=re.I):
        mon = m.group(1)
        d1 = int(m.group(2))
        d2 = int(m.group(3))
        yr = int(m.group(4)) if m.group(4) else None
        out.append((mon, d1, yr))
        out.append((mon, d2, yr))
    # De-duplicate while preserving order.
    uniq: list[tuple[str, int, Optional[int]]] = []
    seen = set()
    for item in out:
        key = (item[0].lower(), item[1], item[2] or 0)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def _repair_date_click_intents(subgoals: list, user_goal: str) -> list:
    """Make date-click intents explicit (month/day/year) to avoid wrong date picks."""
    dates = _extract_goal_dates(user_goal)
    if not dates:
        return subgoals
    month_tokens = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    }
    idx = 0
    for sg in subgoals:
        for st in sg.steps:
            if st.action != "click":
                continue
            il = (st.intent or "").lower()
            if not any(k in il for k in ("date cell", "date picker", "check-in date", "check-out date", "click the date")):
                continue
            if any(m in il for m in month_tokens):
                continue
            day_m = re.search(r"\b(\d{1,2})\b", il)
            if not day_m:
                continue
            target = dates[min(idx, len(dates) - 1)]
            t_mon, t_day, t_year = target
            # If intent already includes a day and it mismatches goal-date order,
            # keep the explicit day from intent but still inject month/year.
            day = int(day_m.group(1))
            if day not in {t_day}:
                day = t_day if idx < len(dates) else day
            mon_fmt = t_mon[:1].upper() + t_mon[1:].lower()
            yr = t_year or _date.today().year
            st.intent = f"Click the date {mon_fmt} {day}, {yr} in the calendar"
            if not st.value:
                st.value = f"{mon_fmt} {day}, {yr}"
            idx += 1
    return subgoals


class ReliableTaskPlanner:
    """
    Goal -> subgoals -> steps planner.
    Always falls back to heuristics if Gemini is unavailable or fails.
    Supports navigate, click, type, select, check, uncheck actions.
    """

    def __init__(
        self,
        use_gemini: bool = True,
        model: str = "gemini-2.0-flash",
        strict_model_planning: bool = False,
    ):
        self._use_gemini = use_gemini
        self._model = model
        self._strict_model_planning = strict_model_planning

    def create_plan(self, user_goal: str, start_url: Optional[str] = None) -> TaskPlan:
        rewritten_goal = _rewrite_past_years(user_goal)
        if rewritten_goal != user_goal:
            logger.info(
                "Goal years rewritten for planning: %r -> %r", user_goal, rewritten_goal
            )

        subgoals: Optional[list[PlannedSubGoal]] = None
        if self._use_gemini and not subgoals:
            try:
                subgoals = self._gemini_subgoals(rewritten_goal, start_url=start_url)
            except Exception as exc:
                logger.warning("Gemini planning raised exception: %s", exc)
                subgoals = None
            if not subgoals:
                logger.warning(
                    "Gemini planning unavailable for goal=%r; using heuristic fallback.",
                    rewritten_goal,
                )
        if not subgoals:
            try:
                subgoals = self._heuristic_subgoals(rewritten_goal)
            except Exception as exc:
                logger.warning("Heuristic planning raised exception: %s", exc)
                subgoals = None
        if not subgoals:
            raise RuntimeError(
                f"Planner could not produce any subgoals for goal: {rewritten_goal!r}"
            )
        subgoals = self._sanitize_subgoals_for_start_domain(
            subgoals=subgoals,
            start_url=start_url,
            user_goal=rewritten_goal,
        )
        subgoals = _repair_date_click_intents(subgoals, rewritten_goal)
        self._normalize_subgoal_criteria(subgoals)
        logger.info("Plan created: %d subgoals for goal=%r", len(subgoals), rewritten_goal)
        return TaskPlan(
            task_id=f"plan-{abs(hash(user_goal))}",
            user_goal=rewritten_goal,
            subgoals=subgoals,
            status="ready",
            current_subgoal_index=0,
        )

    def _normalize_subgoal_criteria(self, subgoals: list[PlannedSubGoal]) -> None:
        """Normalize planner criteria so runtime validation matches real web behavior."""
        for sg in subgoals:
            if not sg.steps:
                continue
            first = sg.steps[0]
            text = ((sg.description or "") + " " + (first.intent or "")).lower()

            # URL criteria from LLM often contain full URLs with volatile params.
            # Reduce to stable path fragments.
            for c in sg.success_criteria:
                if c.kind.lower() != "url_contains":
                    continue
                v = (c.value or "").strip()
                if v.startswith("http://") or v.startswith("https://"):
                    p = urlparse(v)
                    if p.path and p.path != "/":
                        c.value = p.path.lower()
                    elif p.netloc:
                        c.value = p.netloc.lower()

            # Type/scroll/autocomplete steps should not hard-fail on state diff.
            if first.action == "type":
                sg.success_criteria = [GoalCriterion(kind="state_changed", value="true", required=False)]
                continue
            if first.action == "scroll":
                sg.success_criteria = [GoalCriterion(kind="state_changed", value="true", required=False)]
                continue
            if first.action == "click" and any(
                w in text for w in ("autocomplete", "suggestion", "from the list", "dropdown list")
            ):
                sg.success_criteria = [GoalCriterion(kind="state_changed", value="true", required=False)]
                continue
            if first.action == "extract":
                sg.success_criteria = [GoalCriterion(kind="element_exists", value="body", required=False)]

    def create_plan_from_state(
        self,
        original_goal: str,
        current_url: str,
        current_title: str,
        completed_descriptions: list[str],
    ) -> Optional[TaskPlan]:
        """Create a recovery plan from the current page state.

        Called when consecutive subgoal failures indicate the agent is lost.
        Injects current page context into the goal so the planner knows where
        the agent is and what has already been completed.

        Returns None if replanning fails (caller should continue with original plan).
        """
        completed_str = "; ".join(completed_descriptions) if completed_descriptions else "none"
        # Inject a strong instruction to stay on task - the agent may be on the
        # wrong tab/page when this is called (e.g. booking.com/flights when the
        # goal is hotel booking), so explicitly anchor the task type.
        context_goal = (
            f"IMPORTANT: The overall task type must NOT change - the goal is still: {original_goal}. "
            f"The agent is currently on page: {current_url} (title: {current_title}). "
            f"Steps already completed: [{completed_str}]. "
            f"Generate recovery steps to CONTINUE the original goal from the current page. "
            f"Do NOT change the task type (e.g. do not plan flight search if the goal is hotel booking). "
            f"If the current page is wrong (e.g. on flights tab but goal is hotels), "
            f"first navigate back to the correct starting point."
        )
        logger.info("Replanning from current state: url=%r completed=%d steps", current_url, len(completed_descriptions))
        try:
            return self.create_plan(context_goal)
        except Exception as exc:
            logger.warning("Replanning failed: %s", exc)
            return None

    def mark_subgoal(
        self,
        plan: TaskPlan,
        idx: int,
        status: str,
        progress: Optional[GoalProgress],
    ) -> None:
        sg = plan.subgoals[idx]
        sg.status = status
        sg.progress = progress
        plan.current_subgoal_index = idx
        terminal = {"completed", "failed", "skipped"}
        all_done = all(s.status in terminal for s in plan.subgoals)
        if all(s.status == "completed" for s in plan.subgoals):
            plan.status = "completed"
        elif all_done:
            # Some completed, some failed - mark as partial so callers can
            # compute a success rate rather than a hard binary.
            plan.status = "partial"
        else:
            plan.status = "in_progress"

    def subgoal_task_goal(self, subgoal: PlannedSubGoal) -> TaskGoal:
        return TaskGoal(description=subgoal.description, criteria=subgoal.success_criteria)

    def _heuristic_subgoals(self, user_goal: str) -> list[PlannedSubGoal]:
        text = (user_goal or "").strip()
        if not text:
            return []
        chunks = re.split(
            r"\b(?:then|and then|after that|next|afterwards)\b",
            text,
            flags=re.IGNORECASE,
        )
        chunks = [c.strip(" .") for c in chunks if c.strip()]
        if not chunks:
            chunks = [text]
        out: list[PlannedSubGoal] = []
        for i, chunk in enumerate(chunks, start=1):
            step = self._infer_step(chunk)
            crit = self._infer_criteria(chunk, step)
            out.append(
                PlannedSubGoal(
                    subgoal_id=f"sg-{i}",
                    description=chunk,
                    steps=[step],
                    success_criteria=crit,
                )
            )
        return out

    def _infer_step(self, text: str) -> PlannerStep:
        lower = text.lower()

        if any(w in lower for w in ["navigate to", "go to", "visit", "open", "load"]):
            m = re.search(r"https?://\S+", text)
            url = m.group(0).rstrip(".,)") if m else None
            # Heuristic replans often contain "navigate back to the correct page"
            # without an explicit URL. Emitting navigate(value=None) hard-fails.
            # Fall back to click so downstream logic can still recover.
            if not url:
                return PlannerStep(action="click", intent=text, value=None)
            return PlannerStep(action="navigate", intent=text, value=url)

        if any(w in lower for w in ["type", "enter", "fill", "write", "input"]):
            value = ""
            # Prefer explicitly quoted values - they are unambiguous.
            m = re.search(r"['\"]([^'\"]{1,})['\"]", text)
            if m:
                value = m.group(1).strip()
            else:
                # Try relational prepositions before a quoted value.
                m = re.search(r"(?:as|=|:|to|with)\s+['\"]([^'\"]+)['\"]", text)
                if m:
                    value = m.group(1).strip()
            # Don't accept generic stop-words as a value - they are almost
            # always regex noise (e.g. "enter your email" -> captured "your").
            _STOP = {
                "your", "the", "a", "an", "my", "its", "this", "that",
                "some", "any", "it", "them", "their", "input", "value",
                "text", "into", "field", "box", "form",
            }
            if value.lower() in _STOP:
                value = ""
            return PlannerStep(action="type", intent=text, value=value or None)

        if any(w in lower for w in ["scroll", "swipe"]):
            direction = "down"
            if any(w in lower for w in ["up", "top", "back"]):
                direction = "up"
            elif any(w in lower for w in ["bottom", "end"]):
                direction = "bottom"
            return PlannerStep(action="scroll", intent=text, value=direction)

        if any(w in lower for w in ["select", "choose", "pick", "dropdown"]):
            return PlannerStep(action="select", intent=text)

        if any(w in lower for w in ["check", "enable", "tick"]):
            return PlannerStep(action="check", intent=text)

        if any(w in lower for w in ["uncheck", "disable", "untick"]):
            return PlannerStep(action="uncheck", intent=text)

        return PlannerStep(action="click", intent=text)

    def _infer_criteria(
        self, text: str, step: PlannerStep
    ) -> list[GoalCriterion]:
        crit = [GoalCriterion(kind="state_changed", value="true", required=True)]

        if step.action == "navigate":
            if step.value:
                crit.append(
                    GoalCriterion(kind="url_contains", value=step.value, required=True)
                )
            crit = [c for c in crit if c.kind != "state_changed"]
            if not crit:
                crit = [GoalCriterion(kind="state_changed", value="true", required=True)]
            return crit

        if step.action == "type" and step.value:
            crit.append(
                GoalCriterion(kind="text_contains", value=step.value, required=False)
            )

        m = re.search(r"https?://\S+", text)
        if m and any(k in text.lower() for k in ["open", "go to", "navigate", "visit"]):
            crit.append(
                GoalCriterion(kind="url_contains", value=m.group(0), required=False)
            )

        return crit

    def _gemini_subgoals(
        self, user_goal: str, start_url: Optional[str] = None
    ) -> Optional[list[PlannedSubGoal]]:
        load_local_env()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.debug("GEMINI_API_KEY not set; skipping Gemini planning.")
            return None

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={api_key}"
        )
        today = _date.today()
        today_str = today.strftime("%B %d, %Y")   # e.g. "March 23, 2026"
        current_year = today.year
        prompt = (
            f"Today's date is {today_str}. The current year is {current_year}.\n"
            "Decompose this browser automation task into ordered subgoals and executable UI steps.\n"
            "Return JSON only with this exact schema:\n"
            "{\n"
            '  "subgoals": [\n'
            "    {\n"
            '      "description": "short description of what this subgoal achieves",\n'
            '      "steps": [\n'
            "        {\n"
            '          "action": "navigate|click|type|select|check|uncheck|scroll|extract",\n'
            '          "intent": "natural language description of the element, action target, or question to answer",\n'
            '          "value": "URL for navigate, text for type, option label for select - null otherwise"\n'
            "        }\n"
            "      ],\n"
            '      "criteria": [\n'
            "        {\n"
            '          "kind": "state_changed|url_contains|title_contains|text_contains|element_exists",\n'
            '          "value": "expected value",\n'
            '          "required": true\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Use navigate action (with value=URL) to go to any URL, including the starting page.\n"
            "- Break the goal into atomic, independently verifiable subgoals. Maximum 10 subgoals.\n"
            "- Each step should target exactly one UI element or URL.\n"
            "- Use clear, specific intent descriptions.\n"
            "- NEVER target cookie banners, overlays, or navigation tabs (Flights/Cars/Attractions).\n"
            "=== AUTOCOMPLETE RULE (NON-NEGOTIABLE) ===\n"
            "After EVERY type step for a destination/city/query, the NEXT subgoal MUST be:\n"
            "  action=click, intent: Click [value] from the autocomplete suggestion dropdown list\n"
            "  Description AND intent must contain autocomplete or suggestion. NEVER use a date intent here.\n"
            "=== HOTEL/TRAVEL SITES - use EXACTLY this sequence ===\n"
            "1.navigate  2.type-destination  3.click-autocomplete  4.open-date-picker\n"
            "5.click-checkin-date  6.click-checkout-date  7.click-search  8.extract\n"
            "For date cells use intent: Click the date [Month Day Year] in the calendar\n"
            "- Criteria must be achievable and observable immediately after the step executes.\n"
            "- For navigate steps: use kind=url_contains with a fragment of the destination URL.\n"
            "- For click steps that open a new page: ALWAYS use kind=url_contains as required=true criterion. The value must be a unique fragment of the target page URL (e.g. '/store-locator', '/search').\n"
            "- For click steps that expand/toggle UI (accordions, dropdowns): use kind=state_changed with value=true.\n"
            "- For type steps: use kind=state_changed with value=true. Do NOT use text_contains for typed values.\n"
            "- Never use text_contains for content that requires a search result to load.\n"
            "- Keep criteria minimal - one required criterion per subgoal is usually enough.\n"
            "- Do not include criteria that depend on content from external APIs or search results.\n"
            "- For scroll steps: use kind=state_changed with value=true.\n"
            f"- Date picker calendars open on the current month ({today.strftime('%B %Y')}). "
            f"If the target date is in a different month, include a subgoal with click steps to navigate "
            f"the calendar forward or back to the correct month BEFORE clicking the date cell.\n"
            "- When clicking a specific date cell, use intent like: "
            "'Click the date cell for <day> in the calendar date picker' - never reference a year that is not the current year in date picker intents.\n"
            "- The intent for each date-picker click must reference only the day number and month name visible in the current calendar view.\n"
            "=== extract action ===\n"
            "- Use the extract action as the FINAL step in any subgoal that requires READING information from the page.\n"
            "- Trigger words that always require an extract step at the end: 'find out', 'check if', 'identify', 'what is', 'list', 'tell me', 'how many', 'which', 'report', 'note', 'provide', 'show me', 'give me', 'summarize'.\n"
            "- The extract step's intent must be a specific question whose answer is visible on the current page - e.g. 'Can the Mac Mini be configured with a GPU larger than 16-core? List all available GPU options.'\n"
            "- extract steps do NOT need a value. Their success criterion must use kind=element_exists, value='body', required=false - NEVER use state_changed for extract steps because extraction does not change page state.\n"
            "- ALWAYS place the extract step AFTER you have navigated to the exact page that contains the answer.\n"
            "=== Apple website rules ===\n"
            "- To view GPU / CPU configuration options for an Apple product, navigate directly to 'https://www.apple.com/shop/buy-mac/<product-slug>' or 'https://www.apple.com/shop/configure/<product-slug>' - NOT the marketing overview page.\n"
            "- The Mac Mini configurator URL is: https://www.apple.com/shop/buy-mac/mac-mini\n"
            "- The iPhone configurator URL is: https://www.apple.com/shop/buy-iphone\n"
            "=== GENERALIZATION + RELIABILITY GUARDRAILS ===\n"
            "- Be domain-agnostic: do not assume site-specific structure unless visible from URL/title intent.\n"
            f"- Starting URL is: {start_url or 'unknown'}.\n"
            "- Stay on the same website/domain as the starting URL unless the user explicitly asks to use another site.\n"
            "- Do NOT route to help/FAQ/support unless the user goal explicitly asks for policy/support/contact guidance.\n"
            "- For search/listing tasks, keep flow: type query -> choose autocomplete (if present) -> run search -> refine filters/location -> extract.\n"
            "- Avoid brittle assumptions about exact button text; use semantic intents like 'submit search', 'apply location filter', 'open matching result card'.\n"
            "- Include at most one exploratory click before extraction; if uncertain, add a step that narrows context (search/filter) instead of random navigation.\n"
            "- For write/transactional goals (book, buy, set, add, submit), end with a verification subgoal that confirms the action outcome.\n"
            "- Never generate contradictory subgoals (for example FAQ navigation during booking/search unless explicitly requested).\n"
            "- Keep subgoals minimal and recoverable: each step must preserve progress toward the same final user goal.\n"
            f"\nUser Goal: {user_goal}"
        )
        payload = {
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
            "contents": [{"parts": [{"text": prompt}]}],
        }
        data = _gemini_call_with_retry(url, payload, timeout_s=20, max_retries=2)
        if not data:
            return None

        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw)
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse Gemini planning response: %s", exc)
            return None

        if not isinstance(parsed, dict):
            logger.warning("Gemini planning response is not a JSON object.")
            return None

        _VALID_ACTIONS = {"navigate", "click", "type", "select", "check", "uncheck", "scroll", "extract"}
        _MAX_SUBGOALS = 12
        subgoals: list[PlannedSubGoal] = []
        raw_subgoals = parsed.get("subgoals") or []
        if not isinstance(raw_subgoals, list):
            logger.warning("Gemini planning: subgoals field is not a list.")
            return None

        for i, sg in enumerate(raw_subgoals[:_MAX_SUBGOALS], start=1):
            if not isinstance(sg, dict):
                continue
            steps = []
            for s in (sg.get("steps") or []):
                if not isinstance(s, dict):
                    continue
                intent = str(s.get("intent") or "").strip()
                if not intent:
                    continue
                raw_action = str(s.get("action") or "click").strip().lower()
                action = raw_action if raw_action in _VALID_ACTIONS else "click"
                steps.append(PlannerStep(
                    action=action,
                    intent=intent,
                    value=s.get("value") or None,
                ))
            if not steps:
                continue
            raw_criteria = sg.get("criteria") or []
            criteria = []
            if isinstance(raw_criteria, list):
                for c in raw_criteria:
                    if not isinstance(c, dict):
                        continue
                    criteria.append(GoalCriterion(
                        kind=str(c.get("kind") or "state_changed"),
                        value=str(c.get("value") or "true"),
                        required=bool(c.get("required", True)),
                    ))
            if not criteria:
                criteria = [GoalCriterion(kind="state_changed", value="true", required=True)]
            subgoals.append(
                PlannedSubGoal(
                    subgoal_id=f"sg-{i}",
                    description=str(sg.get("description") or f"Subgoal {i}"),
                    steps=steps,
                    success_criteria=criteria,
                )
            )
        # Post-parse correction: fix criteria on extract-only subgoals.
        # Gemini sometimes emits state_changed=true for extract steps, which always
        # fails because extraction does not mutate page DOM.  Replace it with a
        # non-required text_contains so the subgoal is never blocked on state change.
        for sg in subgoals:
            all_extract = sg.steps and all(s.action == "extract" for s in sg.steps)
            if not all_extract:
                continue
            has_bad_criterion = any(
                c.kind == "state_changed" and c.required for c in sg.success_criteria
            )
            if not has_bad_criterion:
                continue
            # Derive a keyword hint from the extract intent for the text_contains value.
            extract_intent = sg.steps[0].intent or sg.description
            # Use first meaningful word (>4 chars) from the intent as a loose hint.
            hint_word = next(
                (w for w in extract_intent.split() if len(w) > 4 and w.isalpha()),
                "",
            )
            sg.success_criteria = [
                GoalCriterion(kind="text_contains", value=hint_word, required=False)
            ]
            logger.debug(
                "Post-parse fix: replaced state_changed criterion on extract subgoal %r "
                "with text_contains(required=False, value=%r)",
                sg.subgoal_id, hint_word,
            )

        # Post-parse: fix extract criteria + autocomplete
        for sg in subgoals:
            if all(s.action == "extract" for s in sg.steps):
                sg.success_criteria = [
                    GoalCriterion(kind="element_exists", value="body", required=False)
                    if c.kind == "state_changed" else c
                    for c in sg.success_criteria
                ]
        subgoals = _repair_and_inject_autocomplete(subgoals)
        return subgoals or None

    def _sanitize_subgoals_for_start_domain(
        self,
        subgoals: list[PlannedSubGoal],
        start_url: Optional[str],
        user_goal: str,
    ) -> list[PlannedSubGoal]:
        if not subgoals or not start_url:
            return subgoals
        try:
            start_host = (urlparse(start_url).netloc or "").lower()
        except Exception:
            start_host = ""
        if not start_host:
            return subgoals

        goal_l = (user_goal or "").lower()
        if any(k in goal_l for k in ("google", "search engine", "wikipedia", "youtube", "bing")):
            return subgoals

        def _same_site(host_a: str, host_b: str) -> bool:
            a = host_a.lstrip("www.")
            b = host_b.lstrip("www.")
            return a == b or a.endswith("." + b) or b.endswith("." + a)

        replaced = 0
        for sg in subgoals:
            for st in sg.steps:
                if st.action != "navigate" or not st.value:
                    continue
                try:
                    nav_host = (urlparse(st.value).netloc or "").lower()
                except Exception:
                    nav_host = ""
                if not nav_host or _same_site(start_host, nav_host):
                    continue
                st.value = start_url
                st.intent = "Go to the target website start page"
                replaced += 1
        if replaced:
            logger.info(
                "Planner sanitized %d off-domain navigate step(s) to start_url host=%s",
                replaced, start_host,
            )
        return subgoals
