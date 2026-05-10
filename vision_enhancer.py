from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from env_loader import load_local_env
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VisionScore:
    """Vision-informed score for a single element candidate."""
    element_idx: int
    visual_score: float       # 0..1: how well this element visually matches the intent
    visual_reasoning: str     # Short natural-language justification


class VisionEnhancer:
    """
    Use Gemini Vision to score element candidates from a page screenshot.

    Flow:
      1. Receive a JPEG screenshot (bytes) of the current page viewport.
      2. Serialize element candidates (idx, role, name, bounding_box).
      3. Call Gemini Vision with screenshot + intent + candidates.
      4. Return per-candidate visual scores.

    These scores are blended into the overall ranking in SemanticElementSelector.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        timeout_s: int = 15,
        enabled: bool = True,
        max_retries: int = 1,
    ):
        self._model = model
        self._timeout_s = timeout_s
        self._enabled = enabled
        self._max_retries = max_retries

    def score_candidates(
        self,
        screenshot_bytes: bytes,
        intent: str,
        action_hint: Optional[str],
        candidates: list[dict],
    ) -> Optional[list[VisionScore]]:
        """
        Score element candidates using page screenshot.

        Args:
            screenshot_bytes: Raw JPEG bytes from Playwright page.screenshot()
            intent: Natural language description of the target element
            action_hint: Action type (click/type/select/etc.) or None
            candidates: List of dicts with keys:
                - idx (int): position in the full candidates list
                - role (str): ARIA role
                - name (str or None): accessible name
                - bbox (dict or None): {x, y, w, h} in page coordinates

        Returns:
            List of VisionScore or None if vision is unavailable/failed.
        """
        if not self._enabled or not screenshot_bytes or not candidates:
            return None

        load_local_env()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None

        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        prompt = self._build_prompt(intent, action_hint, candidates)
        payload = {
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
            "contents": [
                {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": screenshot_b64,
                            }
                        },
                        {"text": prompt},
                    ]
                }
            ],
        }
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={api_key}"
        )

        data = self._call_with_retry(url, payload)
        if not data:
            return None

        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(raw)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Vision response parse error: %s", exc)
            return None

        out: list[VisionScore] = []
        for item in parsed.get("scores", []):
            idx = int(item.get("idx", -1))
            score = float(item.get("score", 0.0))
            if idx < 0:
                continue
            out.append(
                VisionScore(
                    element_idx=idx,
                    visual_score=max(0.0, min(1.0, score)),
                    visual_reasoning=str(item.get("reasoning", "")),
                )
            )

        logger.debug(
            "Vision scored %d/%d candidates for intent=%r",
            len(out), len(candidates), intent,
        )
        return out or None

    def _build_prompt(
        self,
        intent: str,
        action_hint: Optional[str],
        candidates: list[dict],
    ) -> str:
        # Limit candidates in prompt to avoid token limits
        top_candidates = candidates[:20]
        cands_text = json.dumps(top_candidates, ensure_ascii=True)
        action_context = f" The user wants to perform a '{action_hint}' action." if action_hint else ""
        return (
            "You are analyzing a webpage screenshot to find the best UI element for a user action.\n"
            f"User intent: \"{intent}\"{action_context}\n"
            "\n"
            "For each candidate element listed below, assign a visual score based on:\n"
            "1. Whether the element is visually present and interactive (not hidden/covered)\n"
            "2. Whether it semantically matches the user intent\n"
            "3. Whether it's in an appropriate state for the action\n"
            "\n"
            "Return JSON only with this exact schema:\n"
            "{\"scores\": [{\"idx\": int, \"score\": 0.0-1.0, \"reasoning\": \"1-sentence explanation\"}]}\n"
            "\n"
            "Rules:\n"
            "- Only include candidates you can assess (skip if bounding box is missing)\n"
            "- score=1.0 means this is clearly the right element\n"
            "- score=0.0 means this element is wrong or not visible\n"
            "- Be precise: two similar elements should have differentiated scores\n"
            f"\nCandidates: {cands_text}"
        )

    def _call_with_retry(self, url: str, payload: dict) -> Optional[dict]:
        for attempt in range(1, self._max_retries + 2):
            req = urllib.request.Request(
                url=url,
                method="POST",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload).encode("utf-8"),
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 500, 503} and attempt <= self._max_retries:
                    wait = 1.5 * attempt
                    logger.warning("Vision API HTTP %d; retrying in %.1fs", exc.code, wait)
                    time.sleep(wait)
                else:
                    logger.warning("Vision API failed: HTTP %d", exc.code)
                    return None
            except urllib.error.URLError as exc:
                logger.warning("Vision API network error: %s", exc.reason)
                return None
            except Exception as exc:
                logger.warning("Vision API unexpected error: %s", exc)
                return None
        return None