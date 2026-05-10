import math
import re
import threading
from typing import Optional

from logging_config import get_logger

logger = get_logger(__name__)

# Web-UI synonym table
# Maps common web automation intent words to canonical forms so that TF-IDF
# (and neural) can match across naming conventions.
# "check-in" and "departure" both normalise to "checkin departure checkin",
# ensuring overlap with calendar labels like "Check-in date" or "Depart".
_SYNONYMS: dict[str, list[str]] = {
    "checkin":      ["check-in", "check_in", "departure", "depart", "from", "outbound"],
    "checkout":     ["check-out", "check_out", "return", "arrival", "inbound"],
    "destination":  ["where", "to", "going", "location", "place", "city", "airport"],
    "origin":       ["from", "leaving", "source", "start", "origin"],
    "search":       ["find", "look", "query", "discover", "browse", "explore"],
    "submit":       ["go", "search", "confirm", "proceed", "enter", "done"],
    "adult":        ["adults", "pax", "passenger", "traveller", "person", "guest"],
    "children":     ["child", "kids", "minor", "youth"],
    "room":         ["rooms", "cabin", "suite", "accommodation", "lodging", "unit"],
    "filter":       ["refine", "narrow", "sort", "order by", "limit"],
    "price":        ["cost", "rate", "fare", "fee", "charge", "amount"],
    "next":         ["forward", ">", ">", "next month", "advance"],
    "previous":     ["back", "prev", "<", "<", "previous month", "earlier"],
    "close":        ["dismiss", "cancel", "x", "exit", "hide"],
    "accept":       ["agree", "ok", "got it", "allow", "confirm", "yes"],
    "select":       ["choose", "pick", "option", "value"],
    "date":         ["day", "calendar", "when", "time", "schedule"],
    "hotel":        ["accommodation", "stay", "lodging", "inn", "property"],
    "flight":       ["airline", "plane", "air", "fly", "trip"],
    "sort":         ["order", "rank", "arrange", "by price", "by rating"],
    "rating":       ["stars", "score", "review", "rank"],
    "map":          ["location", "area", "district", "landmark"],
}

# Reverse: "check-in" -> ["checkin"]
_SYNONYM_REVERSE: dict[str, list[str]] = {}
for _canon, _variants in _SYNONYMS.items():
    for _v in _variants:
        _SYNONYM_REVERSE.setdefault(_v, []).append(_canon)


def _expand_synonyms(text: str) -> str:
    """Append canonical synonym tokens to text so TF-IDF gains semantic overlap."""
    words = text.lower().split()
    extras: list[str] = []
    for w in words:
        if w in _SYNONYM_REVERSE:
            extras.extend(_SYNONYM_REVERSE[w])
        if w in _SYNONYMS:
            extras.extend(_SYNONYMS[w])
    if extras:
        return text + " " + " ".join(extras)
    return text


def _is_cuda_torch() -> bool:
    """Return True if the installed torch was built with CUDA support.

    CPU-only builds (e.g. torch 2.x+cpu) have no CUDA DLLs and therefore
    cannot crash with DLL-load errors.  In that case the subprocess probe is
    unnecessary and we can load sentence-transformers directly.
    """
    try:
        import importlib
        torch = importlib.import_module("torch")
        return getattr(getattr(torch, "version", None), "cuda", None) is not None
    except Exception:
        return False


def _run_probe(code: str, timeout: int = 60) -> tuple:
    """Run a snippet in a subprocess and return (ok, stdout, stderr)."""
    import os
    import sys
    import subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_CUDA_ALLOC_CONF"] = ""
    env["PYTORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        ok = result.returncode == 0
        return ok, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "probe subprocess timed out"
    except Exception as exc:
        return False, "", str(exc)


def _probe_torch_usable(model_name: str) -> bool:
    """
    Run staged probes to verify torch + sentence-transformers are usable.

    Stage 1: can we import torch at all?
    Stage 2: can we import sentence_transformers?
    Stage 3: can we load the model and encode a string?
    """
    ok, _, err = _run_probe("import torch; print('torch_ok')")
    if not ok:
        err_summary = err.strip().splitlines()[-1] if err.strip() else "(no output)"
        logger.warning(
            "Neural probe stage 1 FAILED - torch import error: %s\n"
            "  Fix: pip uninstall torch torchvision torchaudio -y\n"
            "       pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "       pip install sentence-transformers",
            err_summary,
        )
        return False

    ok, _, err = _run_probe("from sentence_transformers import SentenceTransformer; print('st_ok')")
    if not ok:
        err_summary = err.strip().splitlines()[-1] if err.strip() else "(no output)"
        logger.warning(
            "Neural probe stage 2 FAILED - sentence-transformers import error: %s\n"
            "  Fix: pip install sentence-transformers",
            err_summary,
        )
        return False

    code = (
        "from sentence_transformers import SentenceTransformer; "
        f"m = SentenceTransformer({model_name!r}, device='cpu'); "
        "v = m.encode(['test'], normalize_embeddings=True); "
        "print('model_ok')"
    )
    ok, _, err = _run_probe(code, timeout=120)
    if not ok:
        err_summary = err.strip().splitlines()[-1] if err.strip() else "(no output)"
        logger.warning(
            "Neural probe stage 3 FAILED - model load/encode error for %r: %s",
            model_name, err_summary,
        )
        return False

    return True


def _try_load_sentence_transformers(model_name: str):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

    # On CPU-only torch builds (e.g. 2.x+cpu) there are no CUDA DLLs that can
    # crash the process, so the expensive subprocess probe is unnecessary.
    # Load directly with a try/except - this is both faster and reliable.
    if not _is_cuda_torch():
        logger.debug(
            "CPU-only torch detected; skipping subprocess probe for model=%r.", model_name
        )
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device="cpu")
            logger.info(
                "Neural intent engine loaded model=%r on cpu (direct, no CUDA).", model_name
            )
            return model
        except Exception as exc:
            logger.warning(
                "sentence-transformers model=%r failed to load: %s.", model_name, exc
            )
            return None

    # CUDA-capable torch: use subprocess probe first to prevent DLL crash in
    # the main process if CUDA libraries are broken.
    logger.debug("CUDA torch detected; probing sentence-transformers for model=%r.", model_name)
    if not _probe_torch_usable(model_name):
        logger.warning("Neural engine disabled for model=%r; using TF-IDF fallback.", model_name)
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")
        logger.info("Neural intent engine loaded model=%r on cpu.", model_name)
        return model
    except Exception as exc:
        logger.warning(
            "sentence-transformers model=%r failed to load after probe passed: %s.",
            model_name, exc,
        )
        return None


class TFIDFEngine:
    """
    Corpus-adaptive TF-IDF semantic scorer with unigrams, bigrams, and
    web-UI synonym expansion.

    Synonym expansion adds canonical forms of common web labels so that
    "Check-in date" and "Departure" both score highly against the intent
    "type the check-in date".
    """

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._corpus: list[str] = []
        self._lock = threading.Lock()

    def _tokenize(self, text: str) -> list[str]:
        expanded = _expand_synonyms(text)
        expanded = (expanded or "").lower()
        expanded = re.sub(r"[^a-z0-9\s_\-]", " ", expanded)
        tokens = [t for t in expanded.split() if len(t) > 1]
        bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def _build_idf(self, docs: list[str]) -> dict[str, float]:
        n = len(docs)
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(self._tokenize(doc)):
                df[term] = df.get(term, 0) + 1
        return {
            term: math.log((n + 1.0) / (count + 1.0)) + 1.0
            for term, count in df.items()
        }

    def _vec(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens)
        return {t: (count / n) * self._idf.get(t, 1.0) for t, count in tf.items()}

    def _cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[k] * b[k] for k in shared)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if not na or not nb:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def prime(self, corpus: list[str]) -> None:
        with self._lock:
            combined = self._corpus + [t for t in corpus if t not in self._corpus]
            if len(combined) != len(self._corpus):
                self._corpus = combined
                self._idf = self._build_idf(self._corpus)

    def reset(self) -> None:
        """Clear accumulated corpus and IDF weights (call between tasks)."""
        with self._lock:
            self._corpus = []
            self._idf = {}

    def similarity(self, a: str, b: str) -> float:
        return self._cosine(self._vec(a), self._vec(b))


class NeuralIntentEngine:
    """
    Semantic scorer with two-tier fallback.

    Tier 1: sentence-transformers on CPU.  Preferred models in order:
            1. BAAI/bge-small-en-v1.5  (better zero-shot retrieval, smaller)
            2. all-MiniLM-L6-v2        (general-purpose, widely cached)
            Each is probed in a subprocess to prevent CUDA DLL crashes.

    Tier 2: TF-IDF with bigrams + web-UI synonym expansion.  Zero C-extension
            dependencies.  Output remapped to [-1, 1] for API consistency.
    """

    def __init__(self, use_neural: bool = True, model_name: Optional[str] = None):
        self._use_neural = use_neural
        self._model = None
        self._model_name: str = ""
        self._is_neural_active = False
        self._tfidf = TFIDFEngine()
        self._cache_lock = threading.Lock()
        self._sim_cache: dict[tuple[str, str], float] = {}
        self._classify_cache: dict[tuple[str, str, int], bool] = {}
        self._cache_cap = 4096

        if use_neural:
            # Try preferred models in order; first success wins.
            _candidates = [model_name] if model_name else []
            _candidates += ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"]
            _seen: set[str] = set()
            for mn in _candidates:
                if mn is None or mn in _seen:
                    continue
                _seen.add(mn)
                m = _try_load_sentence_transformers(mn)
                if m is not None:
                    self._model = m
                    self._model_name = mn
                    self._is_neural_active = True
                    break

    @property
    def is_neural_active(self) -> bool:
        return self._is_neural_active

    @property
    def active_model_name(self) -> str:
        return self._model_name if self._is_neural_active else "tfidf"

    def prime_corpus(self, texts: list[str]) -> None:
        self._tfidf.prime(texts)

    def reset_corpus(self) -> None:
        """Reset TF-IDF corpus so previous-task vocabulary does not bleed into the next."""
        self._tfidf.reset()
        with self._cache_lock:
            self._sim_cache.clear()
            self._classify_cache.clear()

    def similarity(self, a: str, b: str) -> float:
        ka = (a or "").strip().lower()
        kb = (b or "").strip().lower()
        key = (ka, kb)
        with self._cache_lock:
            cached = self._sim_cache.get(key)
        if cached is not None:
            return cached

        if self._is_neural_active and self._model is not None:
            score = self._neural_similarity(a, b)
        else:
            raw = self._tfidf.similarity(a, b)
            score = raw * 2.0 - 1.0

        with self._cache_lock:
            if len(self._sim_cache) >= self._cache_cap:
                self._sim_cache.clear()
            self._sim_cache[key] = score
        return score

    def similarity_batch(self, intent: str, texts: list) -> list:
        """Compute similarity between one intent and many texts in a single encode call.

        Returns a list of floats in [-1, 1] (same scale as ``similarity``).
        """
        if not texts:
            return []
        if self._is_neural_active and self._model is not None:
            return self._neural_similarity_batch(intent, texts)
        return [self._tfidf.similarity(intent, t) * 2.0 - 1.0 for t in texts]

    # Semantic intent categories
    # Each entry maps a category name to a reference description.  classify()
    # uses neural similarity against this description instead of keyword lists.
    # Categories are intentionally verbose to give the embedding model enough
    # semantic signal.
    _CATEGORY_REFS: dict[str, str] = {
        "autocomplete_pick":   (
            "select pick choose an item option from autocomplete dropdown "
            "suggestion listbox first result city airport hotel name"
        ),
        "date_picker_open":    (
            "open click date picker calendar widget to select choose check-in "
            "check-out arrival departure date field input"
        ),
        "reveal_expand":       (
            "expand open reveal toggle show hidden dropdown picker accordion "
            "occupancy adults children guests rooms travelers"
        ),
        "in_page_interaction": (
            "interact filter tab accordion sort collapse close dismiss modal "
            "within current page no url change"
        ),
        "page_navigation":     (
            "navigate go to click link visit load new page external URL"
        ),
        # calendar_next/prev: ONLY for physically clicking the month-navigation
        # arrow INSIDE an already-open calendar widget.  Must NOT match intents
        # about opening a calendar, clicking Search, setting filters, or any
        # general 'next' concept outside a calendar widget context.
        "calendar_next":       (
            "click next month arrow button inside open calendar datepicker widget "
            "to advance forward to following month pagination arrow"
        ),
        "calendar_prev":       (
            "click previous month arrow button inside open calendar datepicker widget "
            "to go back earlier month pagination arrow"
        ),
        "search_submit":       (
            "submit search query click search button find look up execute go "
            "confirm form press enter search results"
        ),
        "auth_login":          (
            "sign in log in authenticate login oauth sso credentials password"
        ),
        # New categories that prevent semantic bleed into calendar_next
        "filter_checkbox":     (
            "check enable tick filter checkbox facet option sidebar amenity "
            "swimming pool wifi breakfast rating stars category"
        ),
        "stepper_control":     (
            "increase decrease increment decrement plus minus adults children "
            "rooms guests quantity counter number stepper"
        ),
        "sort_dropdown":       (
            "sort order arrange results by price rating distance popularity "
            "lowest highest best select dropdown"
        ),
    }

    def classify(self, intent: str, category: str, threshold: float = 0.52) -> bool:
        """Return True if *intent* semantically belongs to *category*.

        Uses neural cosine similarity when the model is loaded, falls back to
        TF-IDF + synonym expansion.  Replaces hardcoded keyword-list matching
        throughout the codebase.

        ``threshold`` is applied to the [0, 1]-normalised similarity score
        (0 = orthogonal, 1 = identical).  Default 0.52 is intentionally close
        to 0.50 so borderline intents are accepted rather than missed.
        """
        k_intent = (intent or "").strip().lower()
        k_cat = (category or "").strip().lower()
        k_thr = int(round(float(threshold) * 1000))
        ckey = (k_intent, k_cat, k_thr)
        with self._cache_lock:
            cached = self._classify_cache.get(ckey)
        if cached is not None:
            return cached

        ref = self._CATEGORY_REFS.get(category, category)
        sim = self.similarity(intent, ref)       # [-1, 1]
        score = (sim + 1.0) * 0.5               # [0, 1]
        out = score >= threshold
        with self._cache_lock:
            if len(self._classify_cache) >= self._cache_cap:
                self._classify_cache.clear()
            self._classify_cache[ckey] = out
        return out

    def _neural_similarity(self, a: str, b: str) -> float:
        """Neural cosine similarity - DO NOT apply synonym expansion here.

        Synonym expansion is a TF-IDF compensator for missing vocabulary.
        Neural models already capture semantic overlap across paraphrases, so
        appending synonym tokens pollutes the embedding and reduces precision.
        """
        try:
            ta = (a or "").strip()
            tb = (b or "").strip()
            if not ta or not tb:
                return 0.0
            vecs = self._model.encode([ta, tb], normalize_embeddings=True)
            dot = float(vecs[0] @ vecs[1])
            return max(-1.0, min(1.0, dot))
        except Exception as exc:
            logger.debug("Neural similarity failed, using TF-IDF: %s", exc)
            raw = self._tfidf.similarity(a, b)
            return raw * 2.0 - 1.0

    def _neural_similarity_batch(self, intent: str, texts: list) -> list:
        """Batch neural cosine similarity - synonym expansion NOT applied."""
        try:
            ti = (intent or "").strip()
            if not ti:
                return [0.0] * len(texts)
            clean = [(t or "").strip() for t in texts]
            all_texts = [ti] + clean
            vecs = self._model.encode(all_texts, normalize_embeddings=True)
            intent_vec = vecs[0]
            return [
                max(-1.0, min(1.0, float(intent_vec @ vecs[i])))
                for i in range(1, len(vecs))
            ]
        except Exception as exc:
            logger.debug("Batch neural similarity failed, falling back to TF-IDF: %s", exc)
            return [self._tfidf.similarity(intent, t) * 2.0 - 1.0 for t in texts]


_SHARED_ENGINE: "Optional[NeuralIntentEngine]" = None
_SHARED_ENGINE_LOCK = threading.Lock()


def get_shared_engine(
    use_neural: bool = True,
    model_name: Optional[str] = None,
) -> "NeuralIntentEngine":
    """Return the process-wide NeuralIntentEngine singleton.

    Thread-safe.  Tries BAAI/bge-small-en-v1.5 first, falls back to
    all-MiniLM-L6-v2, then TF-IDF.  Uses subprocess probes to avoid CUDA
    DLL crashes on Windows machines without a GPU.
    """
    global _SHARED_ENGINE
    if _SHARED_ENGINE is not None:
        return _SHARED_ENGINE
    with _SHARED_ENGINE_LOCK:
        if _SHARED_ENGINE is None:
            _SHARED_ENGINE = NeuralIntentEngine(use_neural=use_neural, model_name=model_name)
    return _SHARED_ENGINE
