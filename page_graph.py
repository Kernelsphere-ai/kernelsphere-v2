from dataclasses import dataclass, field
from typing import Optional

from capture import InteractiveElement
from full_page_extractor import ExtractedElement, FullPageExtraction
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    level: int
    frame_id: Optional[str] = None
    dom_node_id: Optional[str] = None
    role: Optional[str] = None
    tag_name: Optional[str] = None
    name: Optional[str] = None
    selector: Optional[str] = None
    selector_type: str = "auto"
    actions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    is_visible: bool = False
    is_enabled: bool = False
    is_occluded: bool = False
    visible_ratio: float = 0.0
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    edge_id: str
    from_node: str
    to_node: str
    edge_type: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class GraphMatch:
    node_id: str
    dom_node_id: Optional[str]
    selector: Optional[str]
    selector_type: str
    score: float
    level: int
    reasoning: str


@dataclass
class PageGraph:
    graph_id: str
    signature: str
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    level_index: dict[int, list[str]]
    frame_index: dict[str, list[str]]
    dom_index: dict[str, str]
    action_index: dict[str, list[str]]

    def available_nodes(self, min_level: int = 2, action: Optional[str] = None) -> list[GraphNode]:
        out: list[GraphNode] = []
        for level, node_ids in self.level_index.items():
            if level < min_level:
                continue
            for node_id in node_ids:
                node = self.nodes[node_id]
                if node.node_type != "element":
                    continue
                if action and action not in node.actions:
                    continue
                out.append(node)
        out.sort(key=lambda n: (n.level, n.confidence, n.visible_ratio), reverse=True)
        return out

    def rank_for_intent(
        self,
        intent: str,
        action: Optional[str] = None,
        top_k: int = 5,
        semantic_scorer=None,
    ) -> list[GraphMatch]:
        # Retrieve all level>=1 nodes regardless of action (soft penalty applied below)
        candidates = self.available_nodes(min_level=1, action=None)
        ranked: list[GraphMatch] = []
        for node in candidates:
            if semantic_scorer:
                base = float(semantic_scorer(intent, node))
            else:
                base = 0.0

            action_fit = 1.0 if (not action or action in node.actions) else 0.0
            # Soft penalty: wrong action type reduces score to 30% rather than zeroing it.
            # This keeps valid elements when action inference is incomplete.
            action_fit_mult = 1.0 if action_fit == 1.0 else 0.30
            observability = max(0.0, min(1.0, node.visible_ratio * (0.0 if node.is_occluded else 1.0)))
            confidence = max(0.0, min(1.0, node.confidence))
            level_prior = max(0.0, min(1.0, node.level / 3.0))
            score = max(
                0.0,
                min(
                    1.0,
                    (0.46 * base)
                    + (0.24 * confidence)
                    + (0.20 * observability)
                    + (0.10 * level_prior),
                ),
            ) * action_fit_mult
            reasoning = (
                f"semantic={base:.2f}, action_fit={action_fit:.2f}, conf={node.confidence:.2f}, "
                f"visible_ratio={node.visible_ratio:.2f}, level={node.level}"
            )
            ranked.append(GraphMatch(
                node_id=node.node_id,
                dom_node_id=node.dom_node_id,
                selector=node.selector,
                selector_type=node.selector_type,
                score=score,
                level=node.level,
                reasoning=reasoning,
            ))
        ranked.sort(key=lambda m: m.score, reverse=True)
        return ranked[:top_k]

    def _to_interactive(self, node: GraphNode) -> InteractiveElement:
        return InteractiveElement(
            tag_name=node.tag_name or "div",
            role=node.role or "generic",
            actions=node.actions,
            is_visible=node.is_visible,
            is_enabled=node.is_enabled,
            is_in_viewport=node.visible_ratio > 0.0,
            bounding_box=None,
            dom_node_id=node.dom_node_id,
            name=node.name,
            placeholder=None,
            value=None,
            input_type=None,
            xpath=node.selector if node.selector_type.startswith("xpath") else None,
            css_selector=node.selector if node.selector_type.startswith("css") else None,
            nearby_text=node.metadata.get("nearby_text") or None,
            form_context=node.metadata.get("form_context") or None,
        )


class PageGraphBuilder:
    def build(self, extraction: FullPageExtraction) -> PageGraph:
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        level_index: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
        frame_index: dict[str, list[str]] = {}
        dom_index: dict[str, str] = {}
        action_index: dict[str, list[str]] = {}

        page_node_id = "page:root"
        nodes[page_node_id] = GraphNode(
            node_id=page_node_id,
            node_type="page",
            level=3,
            source="page",
            metadata={
                "url": extraction.state.url,
                "title": extraction.state.title,
                "signature": extraction.signature,
                "stable": extraction.stable,
                "coverage_score": extraction.coverage.completeness_score if extraction.coverage else None,
            },
        )

        for frame in extraction.frame_summaries:
            fid = f"frame:{frame.frame_id}"
            frame_level = 3 if frame.accessible else 1
            nodes[fid] = GraphNode(
                node_id=fid,
                node_type="frame",
                level=frame_level,
                frame_id=frame.frame_id,
                source="frame",
                metadata={
                    "url": frame.url,
                    "name": frame.name,
                    "depth": frame.depth,
                    "accessible": frame.accessible,
                },
            )
            edges.append(GraphEdge(
                edge_id=f"edge:contains:{page_node_id}->{fid}",
                from_node=page_node_id,
                to_node=fid,
                edge_type="contains",
                weight=1.0,
            ))
            level_index[frame_level].append(fid)

        for idx, el in enumerate(extraction.elements):
            level = self._element_level(el)
            node_id = f"el:{el.frame_id}:{idx}"
            selector, selector_type = self._pick_selector(el)
            node = GraphNode(
                node_id=node_id,
                node_type="element",
                level=level,
                frame_id=el.frame_id,
                dom_node_id=el.dom_node_id,
                role=el.role,
                tag_name=el.tag_name,
                name=el.name,
                selector=selector,
                selector_type=selector_type,
                actions=list(el.actions),
                confidence=el.confidence,
                is_visible=el.is_visible,
                is_enabled=el.is_enabled,
                is_occluded=el.is_occluded,
                visible_ratio=el.visible_ratio,
                source=el.source,
                metadata={
                    "z_index": el.z_index,
                    "nearby_text": el.nearby_text or "",
                    "form_context": el.form_context or "",
                },
            )
            nodes[node_id] = node
            level_index[level].append(node_id)
            frame_index.setdefault(el.frame_id, []).append(node_id)
            if el.dom_node_id:
                dom_index[el.dom_node_id] = node_id
            for action in node.actions:
                action_index.setdefault(action, []).append(node_id)

            frame_node_id = f"frame:{el.frame_id}"
            parent_id = frame_node_id if frame_node_id in nodes else page_node_id
            edges.append(GraphEdge(
                edge_id=f"edge:contains:{parent_id}->{node_id}",
                from_node=parent_id,
                to_node=node_id,
                edge_type="contains",
                weight=1.0,
            ))

        self._add_contextual_edges(nodes, edges)

        return PageGraph(
            graph_id=f"graph:{extraction.signature[:12]}",
            signature=extraction.signature,
            nodes=nodes,
            edges=edges,
            level_index=level_index,
            frame_index=frame_index,
            dom_index=dom_index,
            action_index=action_index,
        )

    def _pick_selector(self, el: ExtractedElement) -> tuple[Optional[str], str]:
        if el.xpath:
            if el.frame_id != "main":
                return el.xpath, "xpath_frame"
            return el.xpath, "xpath"
        if el.css_selector:
            if el.frame_id != "main":
                return el.css_selector, "css_frame"
            return el.css_selector, "css"
        return None, "auto"

    def _element_level(self, el: ExtractedElement) -> int:
        if el.is_visible and el.is_enabled and not el.is_occluded and el.visible_ratio >= 0.6:
            return 3
        # Require at least 25% in-viewport to reach the action stage.  The old 0.10 threshold
        # allowed elements that are 90% off-screen to become candidates; scroll_into_view covers
        # the rest but better candidates should be preferred when available.
        if el.is_visible and el.is_enabled and el.visible_ratio >= 0.25:
            return 2
        if el.is_visible or el.is_enabled:
            return 1
        return 0

    @staticmethod
    def _form_context_key(form_context: str) -> str:
        """Normalize a form_context string to a stable grouping key.

        Dynamic SPAs often capture slightly different surrounding-text snippets
        for elements that share the same logical form.  We normalize by:
          - lowercasing and stripping whitespace
          - truncating to 80 chars so minor suffix differences are ignored
          - collapsing runs of whitespace to a single space
        """
        normalized = " ".join(form_context.lower().split())
        return normalized[:80]

    def _add_contextual_edges(self, nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> None:
        # Primary group: normalized form_context key (handles minor text variations in SPAs)
        # Secondary group: same frame_id + same normalized prefix, for cross-variation linking
        groups: dict[str, list[GraphNode]] = {}
        for node in nodes.values():
            if node.node_type != "element":
                continue
            form = node.metadata.get("form_context")
            if not form:
                continue
            # Scope the key to the frame so cross-frame false positives are avoided.
            key = f"{node.frame_id or 'main'}::{self._form_context_key(form)}"
            groups.setdefault(key, []).append(node)

        for members in groups.values():
            if len(members) < 2:
                continue
            hub = members[0]
            for member in members[1:]:
                edges.append(GraphEdge(
                    edge_id=f"edge:form:{hub.node_id}->{member.node_id}",
                    from_node=hub.node_id,
                    to_node=member.node_id,
                    edge_type="same_form",
                    weight=0.85,
                ))