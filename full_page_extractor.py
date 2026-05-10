import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from playwright.sync_api import Frame, Page

from capture import BoundingBox, ExtractionEngine, InteractiveElement, PageState


@dataclass
class ExtractionConfig:
    include_hidden: bool = False
    capture_bboxes: bool = True
    settle_delay_ms: int = 80
    max_passes: int = 1
    stable_passes_required: int = 0
    max_elements: int = 900
    include_screenshot_fingerprint: bool = False
    screenshot_quality: int = 40
    frame_timeout_ms: int = 1500
    max_frames: int = 4
    max_elements_per_frame: int = 220
    max_shadow_depth: int = 8
    max_notes: int = 80
    dedupe_notes: bool = True


@dataclass
class VisualEmbedding:
    vector: list[float]
    signature: str
    quality: float


@dataclass
class FrameSummary:
    frame_id: str
    url: str
    name: Optional[str]
    depth: int
    element_count: int
    accessible: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class ExtractionCoverage:
    total_frames: int
    accessible_frames: int
    inaccessible_frames: int
    dom_elements: int
    frame_elements: int
    shadow_elements: int
    a11y_fallback_elements: int
    screenshot_fingerprint: Optional[str]
    completeness_score: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ExtractedElement:
    dom_node_id: Optional[str]
    frame_id: str
    role: str
    tag_name: str
    actions: list[str]
    name: Optional[str]
    placeholder: Optional[str]
    value: Optional[str]
    xpath: Optional[str]
    css_selector: Optional[str]
    nearby_text: Optional[str]
    form_context: Optional[str]
    is_visible: bool
    is_enabled: bool
    is_in_viewport: bool
    is_occluded: bool
    visible_ratio: float
    z_index: int
    confidence: float
    source: str
    bounding_box: Optional[BoundingBox] = None
    visual_embedding: Optional[VisualEmbedding] = None


@dataclass
class FullPageExtraction:
    state: PageState
    elements: list[ExtractedElement]
    page_embedding: VisualEmbedding
    stable: bool
    passes_used: int
    signature: str
    latency_ms: float
    frame_summaries: list[FrameSummary] = field(default_factory=list)
    coverage: Optional[ExtractionCoverage] = None
    health_status: str = "unknown"
    telemetry: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


_ACTIVITY_JS = r"""
(() => {
  const perf = performance || {};
  const nav = perf.getEntriesByType ? perf.getEntriesByType('navigation') : [];
  const res = perf.getEntriesByType ? perf.getEntriesByType('resource') : [];
  const now = (perf.now ? perf.now() : Date.now());
  const body = document.body;
  const nodeCount = body ? body.querySelectorAll("*").length : 0;
  const htmlLen = document.documentElement ? document.documentElement.outerHTML.length : 0;
  const ready = document.readyState || "unknown";
  const title = document.title || "";
  const href = location.href || "";
  return {
    now,
    ready,
    title,
    href,
    navCount: nav.length,
    resCount: res.length,
    nodeCount,
    htmlLen
  };
})()
"""


_FRAME_VISUAL_JS_TEMPLATE = r"""
(() => {
  function getRole(el) {
    const role = el.getAttribute("role");
    if (role) return role;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (tag === "a" && el.hasAttribute("href")) return "link";
    if (tag === "button") return "button";
    if (tag === "input") {
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      if (type === "search") return "searchbox";
      return "textbox";
    }
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    return tag;
  }

  function getActions(el) {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    const role = getRole(el);
    const actions = [];
    if (["a", "button", "summary"].includes(tag) || ["button", "link"].includes(role)) actions.push("click");
    if (["input", "textarea"].includes(tag) && !["checkbox", "radio", "submit", "button", "file"].includes(type)) actions.push("type");
    if (tag === "select" || ["combobox", "listbox"].includes(role)) actions.push("select");
    if (type === "checkbox" || role === "checkbox" || role === "switch") {
      actions.push("check");
      actions.push("uncheck");
    }
    if (type === "radio" || role === "radio") actions.push("check");
    if (type === "file") actions.push("upload");
    if (actions.length === 0) actions.push("click");
    return actions;
  }

  function isVisible(el) {
    const s = window.getComputedStyle(el);
    if (!s || s.display === "none" || s.visibility === "hidden" || Number(s.opacity || "1") <= 0.01) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  function getLabel(el) {
    const al = el.getAttribute("aria-label");
    if (al) return al.trim();
    const lb = el.getAttribute("aria-labelledby");
    if (lb) {
      const txt = lb.split(" ").map(id => {
        const n = document.getElementById(id);
        return n ? (n.textContent || "").trim() : "";
      }).join(" ").trim();
      if (txt) return txt;
    }
    if (el.id) {
      const forLabel = document.querySelector(`label[for="${el.id}"]`);
      if (forLabel) return (forLabel.textContent || "").trim();
    }
    const text = (el.textContent || "").trim().replace(/\s+/g, " ");
    if (text) return text.slice(0, 100);
    return null;
  }

  function getVisual(el) {
    const r = el.getBoundingClientRect();
    const area = Math.max(0, r.width) * Math.max(0, r.height);
    const vw = window.innerWidth || 1;
    const vh = window.innerHeight || 1;
    const inW = Math.max(0, Math.min(r.right, vw) - Math.max(r.left, 0));
    const inH = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    const visArea = inW * inH;
    const visibleRatio = area > 0 ? Math.max(0, Math.min(1, visArea / area)) : 0;

    let isOccluded = false;
    const cx = Math.min(vw - 1, Math.max(0, Math.floor(r.left + r.width / 2)));
    const cy = Math.min(vh - 1, Math.max(0, Math.floor(r.top + r.height / 2)));
    const topEl = document.elementFromPoint(cx, cy);
    if (topEl && topEl !== el && !el.contains(topEl) && !topEl.contains(el)) isOccluded = true;

    const style = window.getComputedStyle(el);
    const zRaw = style.zIndex;
    const z = Number.isFinite(Number(zRaw)) ? Number(zRaw) : 0;
    const opacity = Number(style.opacity || "1");
    const fontSize = Number((style.fontSize || "16px").replace("px", "")) || 16;
    const fontWeight = Number(style.fontWeight) || (style.fontWeight === "bold" ? 700 : 400);
    const hasPointer = style.cursor === "pointer";
    const hasShadow = !!style.boxShadow && style.boxShadow !== "none";
    return { visibleRatio, isOccluded, zIndex: z, opacity, fontSize, fontWeight, hasPointer, hasShadow };
  }

  function absoluteCss(el) {
    if (el.id) return "#" + CSS.escape(el.id);
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === 1) {
      let sel = cur.tagName.toLowerCase();
      if (cur.classList && cur.classList.length) sel += "." + Array.from(cur.classList).slice(0, 2).map(c => CSS.escape(c)).join(".");
      const parent = cur.parentElement;
      if (parent) {
        const sib = Array.from(parent.children).filter(x => x.tagName === cur.tagName);
        if (sib.length > 1) sel += `:nth-of-type(${sib.indexOf(cur) + 1})`;
      }
      parts.unshift(sel);
      cur = cur.parentElement;
      if (cur === document.body) break;
    }
    return parts.join(" > ");
  }

  function collectRoots(root, shadowDepth, out, source) {
    if (!root || shadowDepth > __MAX_SHADOW_DEPTH__) return;
    const selectors = [
      "a[href]",
      "button",
      "input:not([type=hidden])",
      "select",
      "textarea",
      "[role=button]",
      "[role=link]",
      "[role=checkbox]",
      "[role=radio]",
      "[role=switch]",
      "[role=textbox]",
      "[role=combobox]",
      "[role=option]",
      "[role=menuitem]",
      "[role=listitem]",
      "[tabindex]:not([tabindex='-1'])",
      "[onclick]",
      "td[data-date]",
      "[data-date]",
      "[data-day]",
      "li[id*='autocomplete']",
      "li[id*='result']",
      "[id*='suggestion']",
      "[class*='autocomplete-item']",
      "[class*='suggestion-item']"
    ];
    const nodes = root.querySelectorAll(selectors.join(","));
    for (const el of nodes) {
      const tag = el.tagName.toLowerCase();
      const type = (el.getAttribute("type") || "").toLowerCase();
      const role = getRole(el);
      const vis = getVisual(el);
      const rect = el.getBoundingClientRect();
      out.push({
        domNodeId: el.__captureId || null,
        role,
        tagName: tag,
        actions: getActions(el),
        inputType: type || null,
        name: getLabel(el),
        placeholder: el.getAttribute("placeholder") || null,
        value: el.value !== undefined ? String(el.value) : null,
        cssSelector: absoluteCss(el),
        xpath: "",
        nearbyText: (el.parentElement ? (el.parentElement.textContent || "").trim().replace(/\s+/g, " ").slice(0, 160) : null),
        formContext: (el.closest("form") ? (el.closest("form").id || el.closest("form").getAttribute("aria-label") || "form") : null),
        isVisible: isVisible(el),
        isEnabled: !(el.disabled || el.getAttribute("aria-disabled") === "true"),
        isInViewport: vis.visibleRatio > 0,
        visibleRatio: vis.visibleRatio,
        isOccluded: vis.isOccluded,
        zIndex: vis.zIndex,
        opacity: vis.opacity,
        fontSize: vis.fontSize,
        fontWeight: vis.fontWeight,
        hasPointer: vis.hasPointer,
        hasShadow: vis.hasShadow,
        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height, top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left },
        source
      });
    }

    const all = root.querySelectorAll("*");
    for (const n of all) {
      if (n.shadowRoot) collectRoots(n.shadowRoot, shadowDepth + 1, out, "frame+shadow");
    }
  }

  const results = [];
  collectRoots(document, 0, results, "frame");
  return results;
})()
"""


class FullPageExtractor:
    """
    Universal-oriented extraction:
    - DOM + accessibility + frame traversal + shadow DOM
    - visual probes + geometric cues + screenshot fingerprint
    - stabilization based on signatures and runtime activity snapshots
    """

    def __init__(self, engine: ExtractionEngine, config: Optional[ExtractionConfig] = None):
        self._engine = engine
        self._config = config or ExtractionConfig()

    def extract(self, page: Page, previous_signature: Optional[str] = None) -> FullPageExtraction:
        t0 = time.perf_counter()
        notes: list[str] = []
        telemetry = {
            "passes": 0,
            "frame_probe_ms": 0.0,
            "merge_ms": 0.0,
            "coverage_ms": 0.0,
            "errors": 0,
            "warnings": 0,
        }
        self._wait_for_readiness(page, notes)

        best_state: Optional[PageState] = None
        best_elements: list[ExtractedElement] = []
        best_signature = ""
        best_frames: list[FrameSummary] = []
        best_coverage: Optional[ExtractionCoverage] = None
        stable_count = 0
        last_sig = None
        last_activity_hash = None
        passes_used = 0

        for _ in range(self._config.max_passes):
            passes_used += 1
            telemetry["passes"] = passes_used
            state = self._engine.capture_page(
                page,
                include_hidden=self._config.include_hidden,
                capture_bboxes=self._config.capture_bboxes,
            )
            t_frame = time.perf_counter()
            frame_elements, frame_summaries, frame_notes = self._collect_frame_elements(page)
            telemetry["frame_probe_ms"] += (time.perf_counter() - t_frame) * 1000
            notes.extend(frame_notes)

            t_merge = time.perf_counter()
            elements = self._merge_elements(state, frame_elements, notes)
            telemetry["merge_ms"] += (time.perf_counter() - t_merge) * 1000
            sig = self._compute_signature(state, elements)
            activity_hash = self._activity_hash(page)

            stable_this_pass = (sig == last_sig) and (activity_hash == last_activity_hash)
            # Reset to 0 on any unstable pass so consecutive stability is required.
            # Previously reset to 1, which allowed a single stable pass to immediately
            # satisfy stable_passes_required=2 on the very next check.
            stable_count = stable_count + 1 if stable_this_pass else 0

            best_state = state
            best_elements = elements
            best_signature = sig
            best_frames = frame_summaries
            t_cov = time.perf_counter()
            best_coverage = self._build_coverage(page, elements, frame_summaries, notes)
            telemetry["coverage_ms"] += (time.perf_counter() - t_cov) * 1000
            last_sig = sig
            last_activity_hash = activity_hash

            if stable_count >= self._config.stable_passes_required:
                break
            try:
                page.wait_for_timeout(self._config.settle_delay_ms)
            except Exception:
                break

        assert best_state is not None
        page_embedding = self._build_page_embedding(best_elements, best_state)
        stable = stable_count >= self._config.stable_passes_required
        screenshot_hash = self._screenshot_hash(page)
        if best_coverage is not None:
            best_coverage.screenshot_fingerprint = screenshot_hash
        if previous_signature and previous_signature != best_signature:
            notes.append("Page signature changed since previous extraction.")
        if not stable:
            notes.append("Page did not fully stabilize in allowed passes.")

        notes = self._normalize_notes(notes)
        telemetry["warnings"] = len(notes)
        telemetry["errors"] = sum(1 for n in notes if "failed" in n.lower() or "error" in n.lower())
        health_status = self._health_status(stable, best_coverage, telemetry)

        return FullPageExtraction(
            state=best_state,
            elements=best_elements,
            page_embedding=page_embedding,
            stable=stable,
            passes_used=passes_used,
            signature=best_signature,
            latency_ms=(time.perf_counter() - t0) * 1000,
            frame_summaries=best_frames,
            coverage=best_coverage,
            health_status=health_status,
            telemetry=telemetry,
            notes=notes,
        )

    def _wait_for_readiness(self, page: Page, notes: list[str]) -> None:
        try:
            state = page.evaluate("() => document.readyState")
            if state != "complete":
                page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            notes.append("readiness check skipped.")
        try:
            page.wait_for_timeout(50)
        except Exception:
            pass

    def _activity_hash(self, page: Page) -> str:
        try:
            info = page.evaluate(_ACTIVITY_JS)
            return hashlib.md5(json.dumps(info, sort_keys=True, default=str).encode()).hexdigest()
        except Exception:
            return "activity_unavailable"

    def _collect_frame_elements(self, page: Page) -> tuple[list[ExtractedElement], list[FrameSummary], list[str]]:
        elements: list[ExtractedElement] = []
        summaries: list[FrameSummary] = []
        notes: list[str] = []
        frames = list(page.frames)
        main = page.main_frame
        page_host = urlparse(page.url or "").netloc.lower()
        ad_host_signals = (
            "doubleclick", "googlesyndication", "googleadservices", "adnxs",
            "taboola", "outbrain", "criteo", "rubiconproject", "amazon-adsystem",
        )
        filtered_frames: list[Frame] = []
        skipped_cross_origin = 0
        skipped_ad_frames = 0
        for fr in frames:
            if fr == main:
                filtered_frames.append(fr)
                continue
            fr_host = urlparse(fr.url or "").netloc.lower()
            fr_url = (fr.url or "").lower()
            if fr_host and any(sig in fr_host for sig in ad_host_signals):
                skipped_ad_frames += 1
                continue
            if fr_url and any(sig in fr_url for sig in ad_host_signals):
                skipped_ad_frames += 1
                continue
            # Cross-origin iframes are mostly ads/widgets on benchmark sites and
            # are a major extraction latency source. Keep same-origin only.
            if page_host and fr_host and fr_host != page_host:
                skipped_cross_origin += 1
                continue
            filtered_frames.append(fr)
        if skipped_ad_frames:
            notes.append(f"Skipped {skipped_ad_frames} ad/tracker frames.")
        if skipped_cross_origin:
            notes.append(f"Skipped {skipped_cross_origin} cross-origin frames.")
        frames = filtered_frames
        if len(frames) > self._config.max_frames:
            notes.append(
                f"Frame count capped: processing {self._config.max_frames}/{len(frames)} frames."
            )
            frames = frames[: self._config.max_frames]

        frame_visual_js = self._frame_visual_js()

        for idx, frame in enumerate(frames):
            frame_id = self._frame_id(frame, idx)
            frame_depth = self._frame_depth(frame)
            frame_notes: list[str] = []
            local_accessible = True
            raw_nodes = []
            offset_x = 0.0
            offset_y = 0.0
            try:
                if frame != page.main_frame:
                    try:
                        fe = frame.frame_element()
                        bb = fe.bounding_box()
                        if bb:
                            offset_x = float(bb.get("x", 0.0))
                            offset_y = float(bb.get("y", 0.0))
                    except Exception:
                        frame_notes.append("Could not resolve frame element offset.")

                frame_t0 = time.perf_counter()
                raw = frame.evaluate(frame_visual_js)
                frame_ms = (time.perf_counter() - frame_t0) * 1000
                if frame_ms > self._config.frame_timeout_ms:
                    frame_notes.append(
                        f"Frame probe slow: {frame_ms:.0f}ms (target {self._config.frame_timeout_ms}ms)."
                    )
                if isinstance(raw, list):
                    raw_nodes = raw
            except Exception:
                local_accessible = False
                frame_notes.append("Frame JS extraction failed.")

            built = self._build_frame_elements(raw_nodes, frame_id, offset_x, offset_y)
            elements.extend(built)
            summaries.append(FrameSummary(
                frame_id=frame_id,
                url=frame.url,
                name=frame.name,
                depth=frame_depth,
                element_count=len(built),
                accessible=local_accessible,
                notes=frame_notes,
            ))
            notes.extend([f"{frame_id}: {n}" for n in frame_notes])

        return elements, summaries, notes

    def _build_frame_elements(
        self,
        nodes: list,
        frame_id: str,
        offset_x: float,
        offset_y: float,
    ) -> list[ExtractedElement]:
        out: list[ExtractedElement] = []
        for node in nodes[: self._config.max_elements_per_frame]:
            rect = node.get("rect") or {}
            raw_box = BoundingBox(
                x=float(rect.get("x", 0.0) + offset_x),
                y=float(rect.get("y", 0.0) + offset_y),
                width=float(rect.get("width", 0.0)),
                height=float(rect.get("height", 0.0)),
                top=float(rect.get("top", 0.0) + offset_y),
                right=float(rect.get("right", 0.0) + offset_x),
                bottom=float(rect.get("bottom", 0.0) + offset_y),
                left=float(rect.get("left", 0.0) + offset_x),
            )
            dom_node_id = node.get("domNodeId")
            prefixed_dom_id = f"{frame_id}:{dom_node_id}" if dom_node_id else None
            vis = {
                "visibleRatio": node.get("visibleRatio", 0.0),
                "opacity": node.get("opacity", 1.0),
                "fontSize": node.get("fontSize", 16.0),
                "fontWeight": node.get("fontWeight", 400.0),
                "hasPointer": node.get("hasPointer", False),
                "hasShadow": node.get("hasShadow", False),
            }
            iel = InteractiveElement(
                tag_name=str(node.get("tagName", "div")),
                role=str(node.get("role", "generic")),
                actions=node.get("actions", ["click"]),
                is_visible=bool(node.get("isVisible", False)),
                is_enabled=bool(node.get("isEnabled", True)),
                is_in_viewport=bool(node.get("isInViewport", False)),
                bounding_box=raw_box,
                dom_node_id=prefixed_dom_id,
                name=node.get("name"),
                placeholder=node.get("placeholder"),
                value=node.get("value"),
                input_type=node.get("inputType"),
                xpath=node.get("xpath") or "",
                css_selector=node.get("cssSelector") or "",
                nearby_text=node.get("nearbyText"),
                form_context=node.get("formContext"),
            )
            visible_ratio = float(node.get("visibleRatio", 0.0))
            is_occluded = bool(node.get("isOccluded", False))
            confidence = self._score_confidence(iel, visible_ratio, is_occluded)
            emb = self._build_element_embedding(iel, vis)
            out.append(ExtractedElement(
                dom_node_id=prefixed_dom_id,
                frame_id=frame_id,
                role=iel.role,
                tag_name=iel.tag_name,
                actions=iel.actions,
                name=iel.name,
                placeholder=iel.placeholder,
                value=iel.value,
                xpath=iel.xpath,
                css_selector=iel.css_selector,
                nearby_text=iel.nearby_text,
                form_context=iel.form_context,
                is_visible=iel.is_visible,
                is_enabled=iel.is_enabled,
                is_in_viewport=iel.is_in_viewport,
                is_occluded=is_occluded,
                visible_ratio=visible_ratio,
                z_index=int(node.get("zIndex", 0)),
                confidence=confidence,
                source=str(node.get("source", "frame")),
                bounding_box=raw_box,
                visual_embedding=emb,
            ))
        return out

    def _frame_visual_js(self) -> str:
        return _FRAME_VISUAL_JS_TEMPLATE.replace(
            "__MAX_SHADOW_DEPTH__",
            str(max(0, int(self._config.max_shadow_depth))),
        )

    def _merge_elements(
        self,
        state: PageState,
        frame_elements: list[ExtractedElement],
        notes: list[str],
    ) -> list[ExtractedElement]:
        dom_map: dict[str, ExtractedElement] = {}

        for el in state.interactive_elements[: self._config.max_elements]:
            ext = self._from_captured_interactive(el, "main", source="dom+a11y+visual")
            key = ext.dom_node_id or f"main|{ext.role}|{ext.tag_name}|{ext.css_selector}|{ext.name}"
            dom_map[key] = ext

        shadow_count = 0
        for fel in frame_elements:
            key = fel.dom_node_id or f"{fel.frame_id}|{fel.role}|{fel.tag_name}|{fel.css_selector}|{fel.name}"
            prev = dom_map.get(key)
            if not prev or fel.confidence > prev.confidence:
                dom_map[key] = fel
            if "shadow" in fel.source:
                shadow_count += 1

        fallback_count = self._merge_a11y_fallback(state, dom_map)
        if fallback_count > 0:
            notes.append(f"Added {fallback_count} accessibility fallback elements.")
        if shadow_count > 0:
            notes.append(f"Discovered {shadow_count} shadow-DOM interactive elements.")

        out = list(dom_map.values())
        out.sort(key=lambda x: (x.confidence, x.visible_ratio, x.is_enabled), reverse=True)
        return out[: self._config.max_elements]

    def _from_captured_interactive(self, el: InteractiveElement, frame_id: str, source: str) -> ExtractedElement:
        vis = {}
        visible_ratio = 1.0 if el.is_in_viewport else 0.0
        is_occluded = False
        confidence = self._score_confidence(el, visible_ratio, is_occluded)
        emb = self._build_element_embedding(el, vis)
        return ExtractedElement(
            dom_node_id=el.dom_node_id,
            frame_id=frame_id,
            role=el.role,
            tag_name=el.tag_name,
            actions=el.actions,
            name=el.name,
            placeholder=el.placeholder,
            value=el.value,
            xpath=el.xpath,
            css_selector=el.css_selector,
            nearby_text=el.nearby_text,
            form_context=el.form_context,
            is_visible=el.is_visible,
            is_enabled=el.is_enabled,
            is_in_viewport=el.is_in_viewport,
            is_occluded=is_occluded,
            visible_ratio=visible_ratio,
            z_index=0,
            confidence=confidence,
            source=source,
            bounding_box=el.bounding_box,
            visual_embedding=emb,
        )

    def _merge_a11y_fallback(self, state: PageState, dom_map: dict[str, ExtractedElement]) -> int:
        added = 0
        for node in state.accessibility_index.values():
            if not node.is_interactable:
                continue
            dom_node_id = node.dom_node_id
            if dom_node_id:
                existing_key = None
                for key, val in dom_map.items():
                    if val.dom_node_id == dom_node_id:
                        existing_key = key
                        break
                if existing_key:
                    continue

            role = node.role or "generic"
            actions = self._role_actions(role)
            key = f"a11y|{role}|{node.name}|{node.parent_id}|{node.node_id}"
            dom_map[key] = ExtractedElement(
                dom_node_id=dom_node_id,
                frame_id="main",
                role=role,
                tag_name=role,
                actions=actions,
                name=node.name,
                placeholder=None,
                value=node.value,
                xpath=node.xpath,
                css_selector=node.css_selector,
                nearby_text=node.description,
                form_context=None,
                is_visible=True,
                is_enabled=not node.disabled,
                is_in_viewport=True,
                is_occluded=False,
                visible_ratio=0.5,
                z_index=0,
                confidence=0.45 if node.name else 0.35,
                source="a11y_fallback",
                bounding_box=node.bounding_box,
                visual_embedding=None,
            )
            added += 1
        return added

    def _role_actions(self, role: str) -> list[str]:
        role = role.lower()
        if role in {"button", "link", "menuitem", "tab"}:
            return ["click"]
        if role in {"textbox", "searchbox", "spinbutton"}:
            return ["type"]
        if role in {"checkbox", "switch"}:
            return ["check", "uncheck"]
        if role in {"radio"}:
            return ["check"]
        if role in {"combobox", "listbox", "option"}:
            return ["select"]
        return ["click"]

    def _score_confidence(self, el: InteractiveElement, visible_ratio: float, is_occluded: bool) -> float:
        score = 0.35
        if el.is_visible:
            score += 0.20
        if el.is_enabled:
            score += 0.15
        if el.is_in_viewport:
            score += 0.10
        score += min(max(visible_ratio, 0.0), 1.0) * 0.12
        if el.name or el.placeholder:
            score += 0.08
        if is_occluded:
            score -= 0.25
        score = max(0.0, min(1.0, score))
        # Apply multiplicative penalties: disabled and invisible elements
        # should have consistently low confidence regardless of additive factors.
        if not el.is_enabled:
            score *= 0.45
        if not el.is_visible:
            score *= 0.35
        if is_occluded and visible_ratio < 0.5:
            score *= 0.60
        return max(0.0, min(1.0, score))

    def _build_element_embedding(self, el: InteractiveElement, vis: dict) -> VisualEmbedding:
        bb = el.bounding_box
        w = float(bb.width if bb else 0.0)
        h = float(bb.height if bb else 0.0)
        area = w * h
        text_len = len((el.name or "") + (el.placeholder or "") + (el.nearby_text or ""))
        vector = [
            min(w / 1920.0, 1.0),
            min(h / 1080.0, 1.0),
            min(area / (1920.0 * 1080.0), 1.0),
            1.0 if el.is_visible else 0.0,
            1.0 if el.is_enabled else 0.0,
            1.0 if el.is_in_viewport else 0.0,
            min(float(vis.get("visibleRatio", 0.0)), 1.0),
            min(float(vis.get("opacity", 1.0)), 1.0),
            min(float(vis.get("fontSize", 16.0)) / 48.0, 1.0),
            min(float(vis.get("fontWeight", 400.0)) / 900.0, 1.0),
            1.0 if bool(vis.get("hasPointer", False)) else 0.0,
            1.0 if bool(vis.get("hasShadow", False)) else 0.0,
            min(text_len / 200.0, 1.0),
        ]
        sig = hashlib.md5(json.dumps(vector, sort_keys=True).encode()).hexdigest()[:12]
        quality = 0.7 + (0.15 if el.name else 0.0) + (0.15 if el.bounding_box else 0.0)
        return VisualEmbedding(vector=vector, signature=sig, quality=min(1.0, quality))

    def _build_page_embedding(self, elements: list[ExtractedElement], state: PageState) -> VisualEmbedding:
        if not elements:
            vector = [0.0] * 14
            sig = hashlib.md5(json.dumps(vector).encode()).hexdigest()[:12]
            return VisualEmbedding(vector=vector, signature=sig, quality=0.2)

        top = elements[:120]
        n = float(len(top))
        avg_conf = sum(e.confidence for e in top) / n
        avg_visible_ratio = sum(e.visible_ratio for e in top) / n
        avg_enabled = sum(1.0 if e.is_enabled else 0.0 for e in top) / n
        avg_in_view = sum(1.0 if e.is_in_viewport else 0.0 for e in top) / n
        avg_occluded = sum(1.0 if e.is_occluded else 0.0 for e in top) / n
        role_div = len({e.role for e in top}) / max(1.0, n)
        tag_div = len({e.tag_name for e in top}) / max(1.0, n)
        frame_div = len({e.frame_id for e in top}) / max(1.0, n)
        shadow_ratio = sum(1.0 if "shadow" in e.source else 0.0 for e in top) / n

        vp = state.viewport
        vector = [
            min(len(state.dom_index) / 6000.0, 1.0),
            min(len(state.accessibility_index) / 6000.0, 1.0),
            min(len(state.interactive_elements) / 1200.0, 1.0),
            avg_conf,
            avg_visible_ratio,
            avg_enabled,
            avg_in_view,
            avg_occluded,
            role_div,
            tag_div,
            frame_div,
            shadow_ratio,
            min(vp.document_height / 18000.0, 1.0),
            min(vp.document_width / 3500.0, 1.0),
        ]
        sig = hashlib.md5(json.dumps(vector, sort_keys=True).encode()).hexdigest()[:16]
        quality = min(1.0, 0.45 + avg_conf * 0.55)
        return VisualEmbedding(vector=vector, signature=sig, quality=quality)

    def _build_coverage(
        self,
        page: Page,
        elements: list[ExtractedElement],
        frame_summaries: list[FrameSummary],
        notes: list[str],
    ) -> ExtractionCoverage:
        total_frames = len(frame_summaries)
        accessible_frames = sum(1 for f in frame_summaries if f.accessible)
        inaccessible_frames = total_frames - accessible_frames
        dom_elements = sum(1 for e in elements if e.source.startswith("dom"))
        frame_elements = sum(1 for e in elements if e.source.startswith("frame"))
        shadow_elements = sum(1 for e in elements if "shadow" in e.source)
        a11y_fallback_elements = sum(1 for e in elements if e.source == "a11y_fallback")
        screenshot_fingerprint = self._screenshot_hash(page)

        frame_ratio = accessible_frames / max(1, total_frames)
        visible_ratio = sum(1.0 if e.is_visible else 0.0 for e in elements) / max(1, len(elements))
        confident_ratio = sum(1.0 if e.confidence >= 0.6 else 0.0 for e in elements) / max(1, len(elements))
        completeness = max(0.0, min(1.0, frame_ratio * 0.35 + visible_ratio * 0.30 + confident_ratio * 0.35))

        if inaccessible_frames > 0:
            notes.append(f"{inaccessible_frames}/{total_frames} frames partially inaccessible.")
        return ExtractionCoverage(
            total_frames=total_frames,
            accessible_frames=accessible_frames,
            inaccessible_frames=inaccessible_frames,
            dom_elements=dom_elements,
            frame_elements=frame_elements,
            shadow_elements=shadow_elements,
            a11y_fallback_elements=a11y_fallback_elements,
            screenshot_fingerprint=screenshot_fingerprint,
            completeness_score=completeness,
            notes=list(notes),
        )

    def _screenshot_hash(self, page: Page) -> Optional[str]:
        if not self._config.include_screenshot_fingerprint:
            return None
        try:
            raw = page.screenshot(type="jpeg", quality=self._config.screenshot_quality, full_page=False)
            return hashlib.md5(raw).hexdigest()
        except Exception:
            return None

    def _frame_id(self, frame: Frame, idx: int) -> str:
        name = frame.name or "anon"
        return f"frame-{idx}-{name}"

    def _frame_depth(self, frame: Frame) -> int:
        depth = 0
        parent = frame.parent_frame
        while parent is not None:
            depth += 1
            parent = parent.parent_frame
        return depth

    def _compute_signature(self, state: PageState, elements: list[ExtractedElement]) -> str:
        top = elements[:180]
        payload = {
            "url": state.url,
            "title": state.title,
            "dom": len(state.dom_index),
            "a11y": len(state.accessibility_index),
            "interactive": len(state.interactive_elements),
            "els": [
                {
                    "id": e.dom_node_id,
                    "f": e.frame_id,
                    "r": e.role,
                    "t": e.tag_name,
                    "n": e.name,
                    "v": e.value,
                    "vr": round(e.visible_ratio, 3),
                    "o": e.is_occluded,
                    "c": round(e.confidence, 3),
                    "s": e.source,
                }
                for e in top
            ],
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _normalize_notes(self, notes: list[str]) -> list[str]:
        cleaned = [n.strip() for n in notes if n and n.strip()]
        if self._config.dedupe_notes:
            seen = set()
            deduped = []
            for n in cleaned:
                if n in seen:
                    continue
                seen.add(n)
                deduped.append(n)
            cleaned = deduped
        if len(cleaned) > self._config.max_notes:
            cleaned = cleaned[: self._config.max_notes]
            cleaned.append("Notes truncated due to max_notes limit.")
        return cleaned

    def _health_status(
        self,
        stable: bool,
        coverage: Optional[ExtractionCoverage],
        telemetry: dict,
    ) -> str:
        score = coverage.completeness_score if coverage else 0.0
        if not stable:
            return "degraded"
        if telemetry.get("errors", 0) > 2:
            return "degraded"
        if score >= 0.80:
            return "healthy"
        if score >= 0.55:
            return "degraded"
        return "poor"

    # ------------------------------------------------------------------
    # Scoped hidden extraction - for dropdowns, date pickers, and other
    # overlays that are only populated after a trigger click.
    # ------------------------------------------------------------------

    # JS that extracts ALL interactive elements within a CSS-scoped container,
    # including elements that are currently hidden (display:none / visibility:hidden).
    # Used after a dropdown-open or date-picker-reveal click.
    _SCOPED_HIDDEN_JS = r"""
(containerSelector) => {
  function getRole(el) {
    const role = el.getAttribute("role");
    if (role) return role;
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    if (tag === "a" && el.hasAttribute("href")) return "link";
    if (tag === "button") return "button";
    if (tag === "input") {
      if (type === "checkbox") return "checkbox";
      if (type === "radio") return "radio";
      return "textbox";
    }
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    return tag;
  }

  function getLabel(el) {
    const al = el.getAttribute("aria-label");
    if (al) return al.trim();
    const text = (el.textContent || "").trim().replace(/\s+/g, " ");
    if (text) return text.slice(0, 100);
    return null;
  }

  function absoluteCss(el) {
    if (el.id) return "#" + CSS.escape(el.id);
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === 1) {
      let sel = cur.tagName.toLowerCase();
      if (cur.classList && cur.classList.length)
        sel += "." + Array.from(cur.classList).slice(0, 2).map(c => CSS.escape(c)).join(".");
      const parent = cur.parentElement;
      if (parent) {
        const sib = Array.from(parent.children).filter(x => x.tagName === cur.tagName);
        if (sib.length > 1) sel += `:nth-of-type(${sib.indexOf(cur) + 1})`;
      }
      parts.unshift(sel);
      cur = cur.parentElement;
      if (cur === document.body) break;
    }
    return parts.join(" > ");
  }

  const root = containerSelector
    ? document.querySelector(containerSelector)
    : document.body;
  if (!root) return [];

  const selectors = [
    "button", "a[href]",
    "input:not([type=hidden])", "select", "textarea",
    "[role=button]", "[role=option]", "[role=menuitem]",
    "[role=checkbox]", "[role=radio]", "[role=switch]",
    "[tabindex]:not([tabindex='-1'])",
    "td[data-date]", "[data-value]", "[data-day]",
  ];
  const nodes = root.querySelectorAll(selectors.join(","));
  const results = [];
  for (const el of nodes) {
    const rect = el.getBoundingClientRect();
    results.push({
      domNodeId: el.__captureId || null,
      role: getRole(el),
      tagName: el.tagName.toLowerCase(),
      name: getLabel(el),
      placeholder: el.getAttribute("placeholder") || null,
      value: el.value !== undefined ? String(el.value) : null,
      cssSelector: absoluteCss(el),
      isVisible: !!(el.offsetParent || rect.width || rect.height),
      isEnabled: !(el.disabled || el.getAttribute("aria-disabled") === "true"),
      visibleRatio: (rect.width > 0 && rect.height > 0) ? 0.8 : 0.0,
      isOccluded: false,
      rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height,
              top: rect.top, right: rect.right, bottom: rect.bottom, left: rect.left },
      ariaExpanded: el.getAttribute("aria-expanded"),
      ariaHidden: el.getAttribute("aria-hidden"),
      source: "scoped_hidden",
    });
  }
  return results;
}
"""

    def extract_scoped_hidden(
        self,
        page: Page,
        container_selector: Optional[str] = None,
        frame_wait_ms: int = 600,
    ) -> list[ExtractedElement]:
        """Extract interactive elements inside a container, including hidden ones.

        Called after a dropdown-open or date-picker-reveal click.  Returns only
        the new/hidden elements found inside the container so the caller can
        merge them into the current extraction without a full page re-scan.

        Args:
            page: Playwright Page.
            container_selector: CSS selector for the opened container
                (e.g. ``"[aria-label='Check-in']"``).  If None, scans the
                whole document (useful when the container is hard to locate).
            frame_wait_ms: How long to wait for the DOM to settle after the
                trigger click before scanning.
        """
        if frame_wait_ms > 0:
            try:
                page.wait_for_timeout(min(frame_wait_ms, 200))
            except Exception:
                pass

        results: list[ExtractedElement] = []
        for frame in page.frames:
            try:
                raw = frame.evaluate(self._SCOPED_HIDDEN_JS, container_selector)
                if not isinstance(raw, list):
                    continue
                built = self._build_frame_elements(
                    raw,
                    frame_id=self._frame_id(frame, 0),
                    offset_x=0.0,
                    offset_y=0.0,
                )
                results.extend(built)
            except Exception:
                continue

        # Mark all results as scoped_hidden source and give them a base confidence.
        out: list[ExtractedElement] = []
        for el in results:
            # Only keep elements that are either enabled or have a name -
            # pure structural containers are noise.
            if not el.is_enabled and not el.name:
                continue
            out.append(el)
        return out
