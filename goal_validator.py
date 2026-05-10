import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import unquote

from capture import PageState
from state_differentiator import StateDiffResult
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GoalCriterion:
    kind: str
    value: str
    required: bool = True
    target_dom_node_id: Optional[str] = None
    target_role: Optional[str] = None


@dataclass
class TaskGoal:
    description: str
    criteria: list[GoalCriterion]


@dataclass
class GoalProgress:
    description: str
    completed: bool
    satisfied_required: int
    total_required: int
    satisfied_optional: int
    total_optional: int
    probability: float
    satisfied: list[str] = field(default_factory=list)
    unsatisfied: list[str] = field(default_factory=list)


class GoalValidator:
    """Goal-level validation to measure whether page changes represent true task progress."""

    def evaluate(
        self,
        goal: TaskGoal,
        state: PageState,
        diff: Optional[StateDiffResult] = None,
    ) -> GoalProgress:
        sat_req = 0
        total_req = sum(1 for c in goal.criteria if c.required)
        sat_opt = 0
        total_opt = sum(1 for c in goal.criteria if not c.required)
        satisfied: list[str] = []
        unsatisfied: list[str] = []

        for c in goal.criteria:
            ok = self._criterion_satisfied(c, state, diff)
            label = f"{c.kind}:{c.value}"
            if ok:
                satisfied.append(label)
                if c.required:
                    sat_req += 1
                else:
                    sat_opt += 1
            else:
                unsatisfied.append(label)

        completed = sat_req == total_req
        # Beta posterior mean with weak prior.
        alpha = 1.0 + sat_req + 0.5 * sat_opt
        beta = 1.0 + (total_req - sat_req) + 0.5 * (total_opt - sat_opt)
        probability = alpha / (alpha + beta)

        logger.debug(
            "Goal eval: completed=%s prob=%.3f required=%d/%d optional=%d/%d",
            completed, probability, sat_req, total_req, sat_opt, total_opt,
        )
        return GoalProgress(
            description=goal.description,
            completed=completed,
            satisfied_required=sat_req,
            total_required=total_req,
            satisfied_optional=sat_opt,
            total_optional=total_opt,
            probability=probability,
            satisfied=satisfied,
            unsatisfied=unsatisfied,
        )

    def _criterion_satisfied(
        self,
        c: GoalCriterion,
        state: PageState,
        diff: Optional[StateDiffResult],
    ) -> bool:
        kind = c.kind.lower()
        value = (c.value or "").strip()

        if kind == "url_contains":
            url = state.url or ""
            v = value.lower()
            # Check both raw URL and URL-decoded form (handles %20, +, etc.)
            return v in url.lower() or v in unquote(url).lower()
        if kind == "title_contains":
            return value.lower() in (state.title or "").lower()
        if kind == "text_contains":
            return self._has_text(state, value)
        if kind == "element_exists":
            return self._element_exists(state, value.lower(), c.target_role)
        if kind == "element_value_equals":
            return self._element_value_equals(state, c.target_dom_node_id, value)
        if kind == "state_changed":
            return bool(diff and diff.had_changes)
        logger.warning("Unknown criterion kind=%r", kind)
        return False

    def _has_text(self, state: PageState, needle: str) -> bool:
        """Search DOM for needle using word-boundary matching to avoid false positives.

        Normalises internal whitespace so DOM text with extra/collapsed spaces
        still matches a clean needle (e.g. "Sign  Up" matches "sign up").
        """
        if not needle:
            return False
        # Normalise needle: collapse whitespace so callers don't have to.
        needle_norm = " ".join(needle.lower().split())
        try:
            pattern = re.compile(
                r"(?<!\w)" + re.escape(needle_norm) + r"(?!\w)",
                re.IGNORECASE,
            )
        except re.error:
            pattern = None

        for node in state.dom_index.values():
            raw = node.text_content or ""
            if not raw:
                continue
            # Normalise DOM text whitespace before matching.
            txt = " ".join(raw.lower().split())
            if pattern:
                if pattern.search(txt):
                    return True
            else:
                if needle_norm in txt:
                    return True
        return False

    def _element_exists(
        self, state: PageState, text: str, role: Optional[str]
    ) -> bool:
        role_l = role.lower() if role else None
        for el in state.interactive_elements:
            name = (el.name or "").lower()
            ph = (el.placeholder or "").lower()
            if role_l and (el.role or "").lower() != role_l:
                continue
            if text in name or text in ph:
                return True
        return False

    def _element_value_equals(
        self,
        state: PageState,
        dom_node_id: Optional[str],
        expected: str,
    ) -> bool:
        if not dom_node_id:
            return False
        expected_lower = expected.lower().strip()
        for el in state.interactive_elements:
            if el.dom_node_id != dom_node_id:
                continue
            val = str(el.value or "").lower().strip()
            # Normalize boolean representations
            if expected_lower in {"true", "1", "yes", "on"} and val in {"true", "1", "yes", "on"}:
                return True
            if expected_lower in {"false", "0", "no", "off"} and val in {"false", "0", "no", "off"}:
                return True
            if val == expected_lower:
                return True
        return False
