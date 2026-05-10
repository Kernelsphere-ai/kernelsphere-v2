import re
import time
from dataclasses import dataclass, field
from typing import Optional

from capture import DOMNode, InteractiveElement, PageState
from logging_config import get_logger

logger = get_logger(__name__)

_DECORATIVE_ROLES = frozenset({"presentation", "none"})
_AUTO_ROTATE_ATTRS = frozenset({
    "data-autoplay", "data-carousel", "data-slide", "data-slider",
    "data-ticker", "data-marquee", "data-lazy", "data-skeleton",
})

# Change kinds that always represent a meaningful user-visible action.
_MEANINGFUL_KINDS = frozenset({
    "url_changed", "title_changed",
    "interactive_added", "interactive_removed",
    "interactive_value_changed", "interactive_enabled_changed",
    "interactive_aria_changed",
})

# Attributes whose change on a DOM node is semantically significant.
_TRACKED_ATTRS = frozenset({
    "value",
    "aria-checked", "aria-expanded", "aria-selected",
    "aria-pressed", "aria-current", "aria-hidden",
    "aria-disabled", "aria-invalid", "aria-busy",
    "class", "disabled", "open", "data-active",
})


@dataclass
class ChangeItem:
    kind: str
    summary: str
    before: Optional[str] = None
    after: Optional[str] = None
    node_id: Optional[str] = None


@dataclass
class StateDiffResult:
    had_changes: bool
    total_changes: int
    changes: list[ChangeItem]
    before_capture_id: str
    after_capture_id: str
    latency_ms: float
    meaningful_changes: int = 0
    had_meaningful_changes: bool = False


class WebStateDifferentiator:
    """Diff two page captures and return a compact, noise-filtered list of changes.

    Produces two signals:
      * had_changes        - any DOM/interactive/URL/title mutation detected
      * had_meaningful_changes - only high-signal changes that represent a
                                 real user-visible state transition (not
                                 background carousels / lazy-load churn)
    """

    def __init__(self, max_items_per_kind: int = 20):
        self._max_items = max_items_per_kind

    def compare(self, before: PageState, after: PageState) -> StateDiffResult:
        t0 = time.perf_counter()
        changes: list[ChangeItem] = []

        if before.url != after.url:
            changes.append(ChangeItem(
                kind="url_changed",
                summary="URL changed",
                before=before.url,
                after=after.url,
            ))
        if before.title != after.title:
            changes.append(ChangeItem(
                kind="title_changed",
                summary="Page title changed",
                before=before.title,
                after=after.title,
            ))

        self._diff_meta(before, after, changes)
        self._diff_dom(before, after, changes)
        self._diff_interactive(before, after, changes)

        meaningful = sum(
            1 for c in changes
            if c.kind in _MEANINGFUL_KINDS or self._is_meaningful_dom_change(c)
        )

        latency = (time.perf_counter() - t0) * 1000
        result = StateDiffResult(
            had_changes=bool(changes),
            total_changes=len(changes),
            changes=changes,
            before_capture_id=before.capture_id,
            after_capture_id=after.capture_id,
            latency_ms=latency,
            meaningful_changes=meaningful,
            had_meaningful_changes=meaningful > 0,
        )
        logger.debug(
            "State diff: %d total changes, %d meaningful in %.1fms (url_changed=%s)",
            result.total_changes,
            result.meaningful_changes,
            latency,
            any(c.kind == "url_changed" for c in changes),
        )
        return result

    # Per-kind diffing

    def _diff_meta(self, before: PageState, after: PageState, out: list[ChangeItem]) -> None:
        pairs = [
            ("description", before.meta.description, after.meta.description),
            ("canonical", before.meta.canonical, after.meta.canonical),
            ("robots", before.meta.robots, after.meta.robots),
            ("form_count", str(before.meta.form_count), str(after.meta.form_count)),
            ("link_count", str(before.meta.link_count), str(after.meta.link_count)),
            # image_count changes constantly on lazy-loading pages; excluded.
        ]
        for key, b, a in pairs:
            if b != a:
                out.append(ChangeItem(
                    kind=f"meta_{key}",
                    summary=f"Meta changed: {key}",
                    before=b,
                    after=a,
                ))

    def _diff_dom(self, before: PageState, after: PageState, out: list[ChangeItem]) -> None:
        before_ids = set(before.dom_index.keys())
        after_ids = set(after.dom_index.keys())

        added_count = 0
        for node_id in after_ids - before_ids:
            if added_count >= self._max_items:
                break
            node = after.dom_index[node_id]
            if self._is_noise_node(node):
                continue
            out.append(ChangeItem(
                kind="dom_added",
                summary=f"DOM node added: {self._describe_node(node)}",
                node_id=node_id,
            ))
            added_count += 1

        removed_count = 0
        for node_id in before_ids - after_ids:
            if removed_count >= self._max_items:
                break
            node = before.dom_index[node_id]
            if self._is_noise_node(node):
                continue
            out.append(ChangeItem(
                kind="dom_removed",
                summary=f"DOM node removed: {self._describe_node(node)}",
                node_id=node_id,
            ))
            removed_count += 1

        modified = 0
        for node_id in before_ids & after_ids:
            if modified >= self._max_items:
                break
            b = before.dom_index[node_id]
            a = after.dom_index[node_id]
            b_snap = self._node_value_snapshot(b)
            a_snap = self._node_value_snapshot(a)
            if b_snap != a_snap:
                out.append(ChangeItem(
                    kind="dom_modified",
                    summary=f"DOM node modified: {self._describe_node(a)}",
                    before=b_snap,
                    after=a_snap,
                    node_id=node_id,
                ))
                modified += 1

    def _diff_interactive(self, before: PageState, after: PageState, out: list[ChangeItem]) -> None:
        b_map = {self._interactive_key(el): el for el in before.interactive_elements}
        a_map = {self._interactive_key(el): el for el in after.interactive_elements}
        b_keys = set(b_map.keys())
        a_keys = set(a_map.keys())

        added = 0
        for key in a_keys - b_keys:
            if added >= self._max_items:
                break
            out.append(ChangeItem(
                kind="interactive_added",
                summary=f"Interactive element added: {self._describe_interactive(a_map[key])}",
            ))
            added += 1

        removed = 0
        for key in b_keys - a_keys:
            if removed >= self._max_items:
                break
            out.append(ChangeItem(
                kind="interactive_removed",
                summary=f"Interactive element removed: {self._describe_interactive(b_map[key])}",
            ))
            removed += 1

        changed = 0
        for key in b_keys & a_keys:
            if changed >= self._max_items:
                break
            b = b_map[key]
            a = a_map[key]
            if b.value != a.value:
                out.append(ChangeItem(
                    kind="interactive_value_changed",
                    summary=f"Interactive value changed: {self._describe_interactive(a)}",
                    before=b.value,
                    after=a.value,
                ))
                changed += 1
            if b.is_enabled != a.is_enabled:
                out.append(ChangeItem(
                    kind="interactive_enabled_changed",
                    summary=f"Enabled state changed: {self._describe_interactive(a)}",
                    before=str(b.is_enabled),
                    after=str(a.is_enabled),
                ))
                changed += 1
            # Track ARIA state changes on interactive elements via their DOM node.
            if b.dom_node_id and a.dom_node_id and b.dom_node_id == a.dom_node_id:
                b_node = before.dom_index.get(b.dom_node_id)
                a_node = after.dom_index.get(a.dom_node_id)
                if b_node and a_node:
                    aria_change = self._aria_diff(b_node, a_node)
                    if aria_change:
                        out.append(ChangeItem(
                            kind="interactive_aria_changed",
                            summary=f"ARIA state changed on {self._describe_interactive(a)}: {aria_change}",
                            before=aria_change[0],
                            after=aria_change[1],
                            node_id=b.dom_node_id,
                        ))
                        changed += 1

    # Noise / meaningful classification

    def _is_noise_node(self, node: DOMNode) -> bool:
        """Return True for DOM nodes that represent background animation/lazy-load churn.

        Uses structural W3C signals instead of a class-name fragment list:

        1. role="presentation" / role="none" - explicitly decorative per spec.
        2. aria-hidden="true" - intentionally removed from the accessibility tree.
        3. data-* auto-rotate attributes - signals auto-cycling content.
        4. No interactive semantics AND no text - structural filler nodes.

        Nodes with an explicit ``id`` attribute survive all checks (1-3) because
        ID-bearing nodes are almost always load-bearing structural elements.
        """
        attrs = node.attributes

        # Nodes with an explicit id are presumed meaningful structural elements.
        if attrs.get("id"):
            return False

        # Signal 1: W3C decorative role.
        role = (attrs.get("role") or "").lower().strip()
        if role in _DECORATIVE_ROLES:
            return True

        # Signal 2: Intentionally hidden from assistive technology.
        if attrs.get("aria-hidden") == "true":
            return True

        # Signal 3: Explicit auto-rotate / lazy-load data attributes.
        if any(k in _AUTO_ROTATE_ATTRS for k in attrs):
            return True

        # Signal 4: No interactive semantics and no text content - filler node.
        text = (node.text_content or "").strip()
        has_role = bool(role)
        has_aria = any(k.startswith("aria-") for k in attrs)
        has_form = node.tag_name in ("input", "select", "textarea", "button", "label", "form")
        return not text and not has_role and not has_aria and not has_form

    def _is_meaningful_dom_change(self, change: ChangeItem) -> bool:
        """Return True if a dom_added/dom_removed/dom_modified change is high-signal."""
        if change.kind not in ("dom_added", "dom_removed", "dom_modified"):
            return False
        summary = change.summary or ""
        # If the summary mentions id= it is an ID-bearing node - meaningful.
        if 'id="' in summary:
            return True
        # Interactive tags in the summary - meaningful.
        for tag in ("input", "select", "textarea", "button", "form", "dialog", "modal"):
            if f" {tag}" in summary or summary.startswith(tag):
                return True
        return False

    def _aria_diff(self, b: DOMNode, a: DOMNode) -> Optional[tuple[str, str]]:
        """Return (before_snap, after_snap) if any tracked ARIA attr changed, else None."""
        _ARIA_ATTRS = (
            "aria-expanded", "aria-checked", "aria-selected",
            "aria-pressed", "aria-current", "aria-hidden",
            "aria-disabled",
        )
        b_vals = {k: b.attributes.get(k) for k in _ARIA_ATTRS if k in b.attributes or k in a.attributes}
        a_vals = {k: a.attributes.get(k) for k in _ARIA_ATTRS if k in b.attributes or k in a.attributes}
        if b_vals != a_vals:
            return (str(b_vals), str(a_vals))
        return None

    # Helpers

    def _describe_node(self, node: DOMNode) -> str:
        node_id = node.attributes.get("id")
        node_class = node.attributes.get("class")
        text = (node.text_content or "").strip().replace("\n", " ")
        text = text[:60] + ("..." if len(text) > 60 else "")
        bits = [node.tag_name]
        if node_id:
            bits.append(f'id="{node_id}"')
        if node_class:
            bits.append(f'class="{node_class[:30]}"')
        if text:
            bits.append(f'text="{text}"')
        return " ".join(bits)

    def _node_value_snapshot(self, node: DOMNode) -> str:
        txt = (node.text_content or "").strip()[:80]
        attrs = {
            k: v for k, v in node.attributes.items()
            if k in _TRACKED_ATTRS
        }
        return f"text={txt!r} attrs={attrs}"

    def _interactive_key(self, el: InteractiveElement) -> str:
        if el.dom_node_id:
            return f"dom:{el.dom_node_id}"
        if el.css_selector and "#" in el.css_selector and "." not in el.css_selector.split("#")[0]:
            return f"css:{el.css_selector}"
        # Use bounding-box-free key to avoid drift after scroll/layout shift.
        name = (el.name or el.placeholder or "")[:40]
        return f"role:{el.role}|tag:{el.tag_name}|name:{name}"

    def _describe_interactive(self, el: InteractiveElement) -> str:
        label = el.name or el.placeholder or el.css_selector or el.tag_name
        return f'{el.role} "{label}"'
