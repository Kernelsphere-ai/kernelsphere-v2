import json
import time
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from capture import ExtractionEngine, PageState, InteractiveElement
from env_loader import load_local_env
from neural_intent_engine import get_shared_engine
from vision_enhancer import VisionEnhancer, VisionScore
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SelectorMatch:
    element: InteractiveElement
    score: float
    strategy: str
    reasoning: str
    selector: str
    selector_type: str


@dataclass
class SelectionResult:
    intent: str
    matches: list
    best: Optional[SelectorMatch]
    fallback_used: bool
    latency_ms: float
    vision_enhanced: bool = False


_GEMINI_SYSTEM = (
    "You are a production-grade web element selector for browser automation. "
    "Return only strict JSON. Rules: "
    "(1) pos field = screen position. Use it to pick the right element when names are identical. "
    "(2) Never pick clear/reset/cancel unless the intent asks for it. "
    "(3) For autocomplete: prefer role=option whose name matches the typed value. "
    "(4) For recipe clicks: only pick links whose href contains /recipe/ (not /recipes/). "
    "(5) For date cells: use nearby_text and pos to identify the correct day. "
    "(6) Prefer main-content controls over header/footer/nav/sidebar links unless intent explicitly targets navigation. "
    "(7) For type actions, never pick buttons/links. For click actions, avoid plain inputs unless intent is to focus/open. "
    "(8) Ignore cookie banners, ads, newsletter popups, and consent modals unless intent explicitly references them."
)


def best_selector(el: InteractiveElement) -> tuple[str, str]:
    if el.xpath and el.xpath.startswith('//*[@id='):
        return el.xpath, "xpath_id"
    if el.css_selector and el.css_selector.startswith("#"):
        return el.css_selector, "css_id"
    if el.xpath:
        return el.xpath, "xpath"
    if el.css_selector:
        return el.css_selector, "css"
    if el.name:
        name = el.name.replace('"', '\\"')
        return f'[aria-label="{name}"]', "aria_label"
    # placeholder is reliable for input/textarea without explicit labels
    if el.placeholder:
        ph = el.placeholder.replace('"', '\\"')
        return f'[placeholder="{ph}"]', "placeholder"
    # Text-content selector for buttons/links - more specific than bare tag
    if el.nearby_text and el.tag_name in {"button", "a", "summary"}:
        text = el.nearby_text.strip()[:60].replace('"', '\\"')
        return f'{el.tag_name}:has-text("{text}")', "text_content"
    # Bare tag is a last resort and almost always wrong on real pages;
    # return it so callers can decide whether to use it.
    return el.tag_name, "tag"


def _bbox(el: InteractiveElement) -> Optional[dict]:
    if not el.bounding_box:
        return None
    return {
        "x": round(el.bounding_box.x),
        "y": round(el.bounding_box.y),
        "w": round(el.bounding_box.width),
        "h": round(el.bounding_box.height),
    }


def _pos_desc(bb: Optional[dict]) -> str:
    if not bb:
        return ""
    cx, cy = bb["x"] + bb["w"] / 2, bb["y"] + bb["h"] / 2
    col = "left" if cx < 448 else ("right" if cx > 832 else "center")
    row = "top" if cy < 252 else ("bottom" if cy > 468 else "middle")
    return "center" if row == "middle" and col == "center" else (
        f"{row}-{col}" if col != "center" else row)


def _grid_pos(elements: list, idx: int) -> str:
    el = elements[idx]
    if el.name is not None:
        return ""
    bb = _bbox(el)
    if not bb:
        return ""
    peers = [(i, _bbox(e)) for i, e in enumerate(elements)
             if e.name is None and e.role == el.role and e.tag_name == el.tag_name and _bbox(e)]
    if len(peers) < 4:
        return ""
    ys = sorted({round(b["y"] / 8) * 8 for _, b in peers})
    xs = sorted({round(b["x"] / 8) * 8 for _, b in peers})
    if len(ys) < 2 or len(xs) < 2:
        return ""
    ry, rx = round(bb["y"] / 8) * 8, round(bb["x"] / 8) * 8
    r = ys.index(ry) + 1 if ry in ys else 0
    c = xs.index(rx) + 1 if rx in xs else 0
    return f"grid row {r} col {c} of {len(ys)}x{len(xs)}" if r and c else ""


def _serialize_elements(elements: list[InteractiveElement]) -> list[dict]:
    out: list[dict] = []
    for i, el in enumerate(elements):
        bb = _bbox(el)
        pos = _grid_pos(elements, i) or _pos_desc(bb)
        entry: dict = {
            "idx": i,
            "role": el.role,
            "tag": el.tag_name,
            "name": el.name,
            "placeholder": el.placeholder,
            "type": el.input_type,
            "actions": el.actions,
            "enabled": el.is_enabled,
            "visible": el.is_visible,
            "in_viewport": el.is_in_viewport,
            "nearby_text": (el.nearby_text or "")[:180],
            "form_context": (el.form_context or "")[:180],
            "bbox": bb,
        }
        if pos:
            entry["pos"] = pos
        out.append(entry)
    return out


def _gemini_call_with_retry(
    url: str,
    payload: dict,
    timeout_s: int = 5,
    max_retries: int = 0,
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
                wait = 1.5 * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini selector HTTP %d attempt %d; retry in %.1fs url=%s",
                    exc.code, attempt, wait, url,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Gemini selector HTTP error %d (giving up). url=%s", exc.code, url,
                )
                return None
        except urllib.error.URLError as exc:
            if attempt <= max_retries:
                wait = 1.5 * attempt
                logger.warning(
                    "Gemini selector network error attempt %d: %s; retry in %.1fs",
                    attempt, exc.reason, wait,
                )
                time.sleep(wait)
            else:
                logger.warning("Gemini selector network error after %d attempts.", attempt)
                return None
        except Exception as exc:
            logger.warning("Gemini selector unexpected error attempt %d: %s", attempt, exc)
            return None
    return None


def _gemini_select(
    intent: str,
    action_hint: Optional[str],
    state: PageState,
    candidates: list[InteractiveElement],
    model: str,
    timeout_s: int,
) -> Optional[list[SelectorMatch]]:
    load_local_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.debug("GEMINI_API_KEY not set; skipping LLM selection.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    page_ctx = {
        "url": state.url,
        "title": state.title,
        "description": state.meta.description,
        "action_hint": action_hint,
    }
    serialized = _serialize_elements(candidates)
    prompt = (
        "Select the best web elements for the user intent.\n"
        "Return JSON only with this exact schema:\n"
        "{\"matches\": [{\"idx\": int, \"score\": float, \"reasoning\": \"short explanation\"}]}\n"
        "\n"
        "Constraints:\n"
        "- score must be in [0,1]\n"
        "- return at most 7 matches, ordered by relevance\n"
        "- omit elements that are hidden, disabled, or clearly wrong\n"
        "- prefer elements with strong semantic alignment to the intent\n"
        "- for 'type' actions: prefer input/textarea elements\n"
        "- for 'click' actions: prefer buttons, links, or clickable elements\n"
        "- for 'select' actions: prefer select/combobox elements\n"
        "- if multiple elements are similar, prefer the one in primary content region (forms/results/main)\n"
        "- avoid global site chrome (header/footer/nav/account/help) unless intent asks for it\n"
        "- for autocomplete intents, rank suggestion options above search/menu icons\n"
        "- do not rank overlay/consent/newsletter elements unless intent explicitly asks\n"
        f"Page context: {json.dumps(page_ctx, ensure_ascii=True)}\n"
        f"User intent: {intent}\n"
        f"Candidate elements: {json.dumps(serialized, ensure_ascii=True)}"
    )

    payload = {
        "systemInstruction": {"parts": [{"text": _GEMINI_SYSTEM}]},
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
        "contents": [{"parts": [{"text": prompt}]}],
    }

    data = _gemini_call_with_retry(url, payload, timeout_s=timeout_s, max_retries=0)
    if not data:
        return None

    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Gemini selector response parse error: %s", exc)
        return None

    index_map = {i: el for i, el in enumerate(candidates)}
    matches: list[SelectorMatch] = []
    for m in parsed.get("matches", []):
        idx = m.get("idx")
        if idx not in index_map:
            continue
        score = float(m.get("score", 0.0))
        if score < 0.01:
            continue
        el = index_map[idx]
        selector, selector_type = best_selector(el)
        matches.append(SelectorMatch(
            element=el,
            score=max(0.0, min(1.0, score)),
            strategy="gemini",
            reasoning=str(m.get("reasoning", "")),
            selector=selector,
            selector_type=selector_type,
        ))

    matches.sort(key=lambda x: x.score, reverse=True)
    logger.debug(
        "Gemini selector: %d matches for intent=%r", len(matches), intent,
    )
    return matches or None


class ElementSelector:
    def __init__(
        self,
        strict_model: bool = False,
        model: str = "gemini-2.0-flash",
        max_candidates: int = 18,
        timeout_s: int = 5,
        use_vision: bool = True,
    ):
        self._strict_model = strict_model
        self._model = model
        self._max_candidates = max_candidates
        self._timeout_s = timeout_s
        self._neural = get_shared_engine()
        self._vision = VisionEnhancer(enabled=use_vision)

    def _elem_semantic_text(self, el: InteractiveElement) -> str:
        selector, selector_type = best_selector(el)
        return " | ".join([
            el.name or "",
            el.placeholder or "",
            el.nearby_text or "",
            el.form_context or "",
            selector,
            selector_type,
            el.role or "",
            el.tag_name or "",
            " ".join(el.actions or []),
        ])

    def _semantic_score(self, intent: str, el: InteractiveElement) -> float:
        sim = self._neural.similarity(intent, self._elem_semantic_text(el))
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))

    # Ad network / tracking iframe patterns - these are never valid action targets.
    _AD_SELECTOR_PATTERNS = (
        "google_ads_iframe", "googlesyndication", "doubleclick", "adsbygoogle",
        "googletagmanager", "amazon-adsystem", "adnxs", "rubiconproject",
        "criteo", "outbrain", "taboola", "adform", "adsystem",
    )
    # Element label words that indicate a destructive / cancellation action.
    _DESTRUCTIVE_LABELS = frozenset({
        "clear", "reset", "cancel", "delete", "remove", "discard",
        "undo", "erase", "wipe", "dismiss",
    })
    # Intent words that mark the intent itself as destructive - suppress penalty then.
    _DESTRUCTIVE_INTENT_WORDS = frozenset({
        "clear", "reset", "cancel", "delete", "remove", "discard",
        "undo", "erase", "wipe",
    })

    def _is_ad_el(self, el: InteractiveElement) -> bool:
        sel = ((el.css_selector or "") + (el.xpath or "")).lower()
        return any(p in sel for p in self._AD_SELECTOR_PATTERNS)

    def _is_destructive_el(self, el: InteractiveElement) -> bool:
        label = ((el.name or "") + " " + (el.nearby_text or "")).lower()
        return any(d in label.split() for d in self._DESTRUCTIVE_LABELS)

    def _prefilter_candidates(
        self, intent: str, state: PageState
    ) -> list[InteractiveElement]:
        _cookie_patterns = (
            "onetrust", "onetrust-", "ot-sdk", "ot-", "cookie", "consent",
            "privacy-banner", "gdpr", "ccpa", "cookie-banner", "banner-close",
        )
        def _is_cookie_el(el: InteractiveElement) -> bool:
            sel = ((el.css_selector or "") + (el.xpath or "") + (el.nearby_text or "")).lower()
            return any(p in sel for p in _cookie_patterns)

        visible = [
            e for e in state.interactive_elements
            if e.is_visible and e.is_enabled
            and not _is_cookie_el(e)
            and not self._is_ad_el(e)
        ]
        if not visible:
            visible = [
                e for e in state.interactive_elements
                if e.is_visible and not _is_cookie_el(e) and not self._is_ad_el(e)
            ]
        if not visible:
            visible = [
                e for e in state.interactive_elements
                if not _is_cookie_el(e) and not self._is_ad_el(e)
            ]
        if not visible:
            visible = list(state.interactive_elements)

        texts = [self._elem_semantic_text(el) for el in visible]
        sims = self._neural.similarity_batch(intent, texts)
        scores = [max(0.0, min(1.0, (s + 1.0) * 0.5)) for s in sims]

        # Apply destructive-element penalty when the intent is not itself destructive.
        # This prevents "Clear dates" / "Reset filters" from winning over the real target.
        intent_words = set(intent.lower().split())
        intent_is_destructive = bool(intent_words & self._DESTRUCTIVE_INTENT_WORDS)
        if not intent_is_destructive:
            scores = [
                max(0.0, s - 0.40) if self._is_destructive_el(el) else s
                for el, s in zip(visible, scores)
            ]

        # Hard-exclude navigation category links when intent is about a specific recipe/item.
        # These links (/recipes/96/salad/, /food-news-trends/) never lead to a recipe page.
        intent_lower = intent.lower()
        _item_click = any(w in intent_lower for w in ("recipe", "article", "card", "item", "result", "post"))
        if _item_click:
            import re as _re
            _cat = _re.compile('/recipes?/[0-9]+/|/food-news|/product-review')
            filtered = [(el, s) for el, s in zip(visible, scores)
                        if not (_cat.search(el.css_selector or "") and '/recipe/' not in (el.css_selector or ""))]
            if filtered:
                scored = sorted(filtered, key=lambda x: x[1], reverse=True)
            else:
                scored = sorted(zip(visible, scores), key=lambda x: x[1], reverse=True)
        else:
            scored = sorted(zip(visible, scores), key=lambda x: x[1], reverse=True)
        return [el for el, _ in scored[: self._max_candidates]]

    def _apply_vision_scores(
        self,
        matches: list[SelectorMatch],
        vision_scores: list[VisionScore],
    ) -> list[SelectorMatch]:
        vision_map = {vs.element_idx: vs for vs in vision_scores}
        updated = []
        for i, match in enumerate(matches):
            vs = vision_map.get(i)
            if vs:
                if vs.visual_score < 0.10:
                    # Strong negative signal - element clearly not what was intended visually
                    blended = max(0.0, match.score * 0.35)
                else:
                    blended = max(0.0, min(1.0, match.score * 0.70 + vs.visual_score * 0.30))
                updated.append(SelectorMatch(
                    element=match.element,
                    score=blended,
                    strategy=match.strategy + "+vision",
                    reasoning=match.reasoning + f" | vision={vs.visual_score:.2f}: {vs.visual_reasoning[:60]}",
                    selector=match.selector,
                    selector_type=match.selector_type,
                ))
            else:
                updated.append(match)
        updated.sort(key=lambda x: x.score, reverse=True)
        return updated

    def select(
        self,
        intent: str,
        state: PageState,
        use_llm: bool = True,
        action_hint: Optional[str] = None,
        screenshot_bytes: Optional[bytes] = None,
    ) -> SelectionResult:
        t0 = time.perf_counter()
        candidates = self._prefilter_candidates(intent, state)
        logger.debug(
            "Selecting element for intent=%r action=%r candidates=%d",
            intent, action_hint, len(candidates),
        )
        vision_enhanced = False

        if use_llm:
            llm_matches = _gemini_select(
                intent=intent,
                action_hint=action_hint,
                state=state,
                candidates=candidates,
                model=self._model,
                timeout_s=self._timeout_s,
            )
            if llm_matches:
                if screenshot_bytes and self._vision:
                    vision_candidates = [
                        {
                            "idx": i,
                            "role": m.element.role,
                            "name": m.element.name,
                            "bbox": _bbox(m.element),
                        }
                        for i, m in enumerate(llm_matches)
                    ]
                    vision_scores = self._vision.score_candidates(
                        screenshot_bytes=screenshot_bytes,
                        intent=intent,
                        action_hint=action_hint,
                        candidates=vision_candidates,
                    )
                    if vision_scores:
                        llm_matches = self._apply_vision_scores(llm_matches, vision_scores)
                        vision_enhanced = True

                latency = (time.perf_counter() - t0) * 1000
                logger.debug(
                    "Selection succeeded: top=%r score=%.3f vision=%s latency=%.0fms",
                    llm_matches[0].selector, llm_matches[0].score, vision_enhanced, latency,
                )
                return SelectionResult(
                    intent=intent,
                    matches=llm_matches,
                    best=llm_matches[0],
                    fallback_used=False,
                    latency_ms=latency,
                    vision_enhanced=vision_enhanced,
                )

        if self._strict_model:
            latency = (time.perf_counter() - t0) * 1000
            logger.warning(
                "LLM unavailable and strict_model=True; returning no matches for intent=%r",
                intent,
            )
            return SelectionResult(
                intent=intent,
                matches=[],
                best=None,
                fallback_used=False,
                latency_ms=latency,
            )

        matches: list[SelectorMatch] = []
        cand_texts = [self._elem_semantic_text(el) for el in candidates]
        cand_sims = self._neural.similarity_batch(intent, cand_texts)
        for el, sim in zip(candidates, cand_sims):
            score = max(0.0, min(1.0, (sim + 1.0) * 0.5))
            if score < 0.30:
                continue
            selector, selector_type = best_selector(el)
            matches.append(SelectorMatch(
                element=el,
                score=score,
                strategy="semantic_fallback",
                reasoning=f"neural_similarity={score:.3f}",
                selector=selector,
                selector_type=selector_type,
            ))
        matches.sort(key=lambda x: x.score, reverse=True)
        matches = matches[:7]

        # Apply vision scoring in fallback path too - same signal, same benefit
        if screenshot_bytes and self._vision and matches:
            vision_candidates = [
                {
                    "idx": i,
                    "role": m.element.role,
                    "name": m.element.name,
                    "bbox": _bbox(m.element),
                }
                for i, m in enumerate(matches)
            ]
            vision_scores = self._vision.score_candidates(
                screenshot_bytes=screenshot_bytes,
                intent=intent,
                action_hint=action_hint,
                candidates=vision_candidates,
            )
            if vision_scores:
                matches = self._apply_vision_scores(matches, vision_scores)
                vision_enhanced = True

        latency = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Semantic fallback: %d matches for intent=%r vision=%s latency=%.0fms",
            len(matches), intent, vision_enhanced, latency,
        )
        return SelectionResult(
            intent=intent,
            matches=matches,
            best=matches[0] if matches else None,
            fallback_used=True,
            latency_ms=latency,
            vision_enhanced=vision_enhanced,
        )


class SemanticElementSelector:
    def __init__(
        self,
        use_llm: bool = True,
        cache_ttl: float = 60.0,
        strict_model: bool = False,
        use_vision: bool = True,
    ):
        self._engine = ElementSelector(strict_model=strict_model, use_vision=use_vision)
        self._use_llm = use_llm
        self._cache: dict = {}
        self._cache_ttl = cache_ttl

    def find(
        self,
        intent: str,
        state: PageState,
        use_cache: bool = True,
        action_hint: Optional[str] = None,
        screenshot_bytes: Optional[bytes] = None,
    ) -> SelectionResult:
        cache_key = f"{state.url}|{state.capture_id}|{intent}|{action_hint or ''}"
        if use_cache and cache_key in self._cache and screenshot_bytes is None:
            ts, result = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return result
        result = self._engine.select(
            intent,
            state,
            use_llm=self._use_llm,
            action_hint=action_hint,
            screenshot_bytes=screenshot_bytes,
        )
        if use_cache and screenshot_bytes is None:
            self._cache[cache_key] = (time.time(), result)
        return result

    def find_batch(self, intents: list[str], state: PageState) -> dict:
        return {intent: self.find(intent, state) for intent in intents}

    def clear_cache(self) -> None:
        self._cache.clear()


def rule_score(intent: str, el: InteractiveElement) -> float:
    engine = get_shared_engine()
    selector, selector_type = best_selector(el)
    txt = " | ".join([
        el.name or "",
        el.placeholder or "",
        el.nearby_text or "",
        el.form_context or "",
        selector,
        selector_type,
        el.role or "",
        el.tag_name or "",
        " ".join(el.actions or []),
    ])
    sim = engine.similarity(intent or "", txt)
    return max(0.0, min(1.0, (sim + 1.0) * 0.5))


def print_result(result: SelectionResult) -> None:
    mode = "llm" if not result.fallback_used else "fallback"
    vision_tag = "+vision" if result.vision_enhanced else ""
    print(f"\n  intent : \"{result.intent}\"  [{mode}{vision_tag} | {result.latency_ms:.0f}ms]")
    if not result.matches:
        print("  result : no matches")
        return
    for i, m in enumerate(result.matches[:3]):
        tag = " <--" if i == 0 else "    "
        print(f"  {tag}[{i+1}] score={m.score:.2f}  [{m.element.role}] {m.element.name or repr(m.element.tag_name)}")
        print(f"       sel: {m.selector}")
        print(f"       why: {m.reasoning}")


if __name__ == "__main__":
    import threading
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from logging_config import setup_logging
    setup_logging(level="INFO")

    TEST_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Selector Demo</title></head>
<body>
  <label>Email</label><input id="email" aria-label="Email" type="email"/>
  <label>Password</label><input id="password" aria-label="Password" type="password"/>
  <button id="signin">Sign In</button>
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

    srv = HTTPServer(("localhost", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever)
    t.daemon = True
    t.start()

    with ExtractionEngine(headless=True) as engine:
        state = engine.capture_url(f"http://localhost:{port}/")
        selector = SemanticElementSelector(use_llm=True, strict_model=False)
        for intent in ["type email", "type password", "click sign in"]:
            print_result(selector.find(intent, state, action_hint="click" if "click" in intent else "type"))

    srv.shutdown()
