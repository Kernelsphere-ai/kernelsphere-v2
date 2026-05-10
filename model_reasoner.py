import json
import os
import time
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from env_loader import load_local_env
from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ReasonerConfig:
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    max_calls_per_task: int = 3
    max_candidates_per_call: int = 5
    enable_cache: bool = True
    temperature: float = 0.0
    timeout_s: int = 8
    min_uncertainty_gap: float = 0.10


@dataclass
class ReasonerScore:
    candidate_idx: int
    relevance: float
    action_fit: float
    confidence: float
    reasoning: str


class CostAwareReasoner:
    """
    Cheap-first reasoning reranker.
    Intended for ambiguous cases only.
    Uses Gemini API when key is present.
    Has hard call-budget and cache to control cost.
    """

    def __init__(self, config: Optional[ReasonerConfig] = None):
        self._cfg = config or ReasonerConfig()
        self._calls = 0
        self._cache: dict[str, list[ReasonerScore]] = {}

    def reset_budget(self) -> None:
        self._calls = 0

    @property
    def max_candidates_per_call(self) -> int:
        return self._cfg.max_candidates_per_call

    _HIGH_CONFIDENCE_BYPASS = 0.80

    def should_rerank(self, scores: list[float]) -> bool:
        if len(scores) < 2:
            return False
        top = sorted(scores, reverse=True)
        if top[0] <= 0.0:
            return False
        if top[0] >= self._HIGH_CONFIDENCE_BYPASS:
            logger.debug(
                "should_rerank: top=%.4f >= HIGH_CONFIDENCE_BYPASS=%.4f -> skip rerank",
                top[0], self._HIGH_CONFIDENCE_BYPASS,
            )
            return False
        gap = top[0] - top[1]
        logger.debug(
            "should_rerank: top=%.4f gap=%.4f threshold=%.4f",
            top[0], gap, self._cfg.min_uncertainty_gap,
        )
        return gap < self._cfg.min_uncertainty_gap

    def rerank(
        self,
        intent: str,
        action: Optional[str],
        candidate_rows: list[dict],
    ) -> Optional[list[ReasonerScore]]:
        if self._calls >= self._cfg.max_calls_per_task:
            return None
        if not candidate_rows:
            return None

        rows = candidate_rows[: self._cfg.max_candidates_per_call]
        cache_key = self._cache_key(intent, action, rows)
        if self._cfg.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self._cfg.provider == "gemini":
            out = self._call_gemini(intent, action, rows)
        else:
            out = None

        if out:
            self._calls += 1
            if self._cfg.enable_cache:
                self._cache[cache_key] = out
        return out

    def _call_gemini(self, intent: str, action: Optional[str], rows: list[dict]) -> Optional[list[ReasonerScore]]:
        load_local_env()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        model = self._cfg.model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = (
            "Rank UI candidates for a web automation action.\n"
            "Return only JSON:\n"
            "{ \"scores\": [ {\"idx\": int, \"relevance\": 0..1, \"action_fit\": 0..1, \"confidence\": 0..1, \"reasoning\": \"short\"} ] }\n"
            "Hard rules:\n"
            "- action_fit must be near 0 for action-incompatible elements (e.g. type on button, click on hidden/disabled element).\n"
            "- Penalize cookie banners, newsletter popups, ads, and generic account/help nav unless intent explicitly asks for them.\n"
            "- Prefer candidates in primary task flow (search form, result list, checkout form, settings form) over global chrome.\n"
            "- For autocomplete intents, prioritize options/list items matching typed value over menu/search icons.\n"
            "- If uncertain, lower confidence rather than overconfidently picking weak candidates.\n"
            f"Intent: {intent}\n"
            f"Action: {action or 'unspecified'}\n"
            f"Candidates: {json.dumps(rows)}"
        )
        payload = {
            "generationConfig": {
                "temperature": self._cfg.temperature,
                "responseMimeType": "application/json",
            },
            "contents": [{"parts": [{"text": prompt}]}],
        }
        req = urllib.request.Request(
            url=url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._cfg.timeout_s) as resp:
                data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            out: list[ReasonerScore] = []
            for s in parsed.get("scores", []):
                out.append(ReasonerScore(
                    candidate_idx=int(s.get("idx", -1)),
                    relevance=float(s.get("relevance", 0.0)),
                    action_fit=float(s.get("action_fit", 0.0)),
                    confidence=float(s.get("confidence", 0.0)),
                    reasoning=str(s.get("reasoning", "")),
                ))
            return out or None
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Reasoner Gemini HTTP error %d for model=%r url=%s",
                exc.code, model, url,
            )
            return None
        except (
            urllib.error.URLError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("Reasoner Gemini error: %s", exc)
            return None

    def _cache_key(self, intent: str, action: Optional[str], rows: list[dict]) -> str:
        raw = json.dumps({"intent": intent, "action": action, "rows": rows}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()
