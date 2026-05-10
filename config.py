from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from env_loader import load_local_env

_PREFIX = "KS_"


def _e(key: str, default: str) -> str:
    return os.environ.get(f"{_PREFIX}{key}", os.environ.get(key, default))


def _ei(key: str, default: int) -> int:
    try:
        return int(_e(key, str(default)))
    except ValueError:
        return default


def _eb(key: str, default: bool) -> bool:
    return _e(key, "true" if default else "false").lower() not in {"0", "false", "no", "off"}


def _ef(key: str, default: float) -> float:
    try:
        return float(_e(key, str(default)))
    except ValueError:
        return default


@dataclass
class BrowserConfig:
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    navigation_timeout_ms: int = 30_000
    page_load_wait: str = "domcontentloaded"
    max_restart_attempts: int = 3

    @classmethod
    def from_env(cls) -> "BrowserConfig":
        load_local_env()
        return cls(
            headless=_eb("BROWSER_HEADLESS", True),
            viewport_width=_ei("VIEWPORT_WIDTH", 1280),
            viewport_height=_ei("VIEWPORT_HEIGHT", 720),
            navigation_timeout_ms=_ei("NAV_TIMEOUT_MS", 30_000),
            page_load_wait=_e("PAGE_LOAD_WAIT", "domcontentloaded"),
            max_restart_attempts=_ei("BROWSER_MAX_RESTARTS", 3),
        )


@dataclass
class LLMConfig:
    enabled: bool = True
    model: str = "gemini-2.0-flash"
    timeout_s: int = 12
    max_reasoner_calls: int = 10
    reasoner_min_gap: float = 0.10

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_local_env()
        return cls(
            enabled=_eb("LLM_ENABLED", True),
            model=_e("GEMINI_MODEL", "gemini-2.0-flash"),
            timeout_s=_ei("LLM_TIMEOUT_S", 12),
            max_reasoner_calls=_ei("REASONER_MAX_CALLS", 10),
            reasoner_min_gap=_ef("REASONER_MIN_GAP", 0.10),
        )


@dataclass
class ExtractionSettings:
    max_passes: int = 2
    stable_passes_required: int = 1
    settle_delay_ms: int = 80
    max_elements: int = 1200
    max_frames: int = 8
    max_shadow_depth: int = 8
    screenshot_fingerprint: bool = True

    @classmethod
    def from_env(cls) -> "ExtractionSettings":
        load_local_env()
        return cls(
            max_passes=_ei("EXTRACT_MAX_PASSES", 2),
            stable_passes_required=_ei("EXTRACT_STABLE_PASSES", 1),
            settle_delay_ms=_ei("EXTRACT_SETTLE_MS", 80),
            max_elements=_ei("EXTRACT_MAX_ELEMENTS", 1200),
            max_frames=_ei("EXTRACT_MAX_FRAMES", 8),
            max_shadow_depth=_ei("EXTRACT_MAX_SHADOW_DEPTH", 8),
            screenshot_fingerprint=_eb("EXTRACT_SCREENSHOT", True),
        )


@dataclass
class ActionSettings:
    default_timeout_ms: int = 10_000
    retry_attempts: int = 2
    retry_backoff_ms: int = 180
    wait_after_ms: int = 250
    max_candidates: int = 6

    @classmethod
    def from_env(cls) -> "ActionSettings":
        load_local_env()
        return cls(
            default_timeout_ms=_ei("ACTION_TIMEOUT_MS", 10_000),
            retry_attempts=_ei("ACTION_RETRY_ATTEMPTS", 2),
            retry_backoff_ms=_ei("ACTION_RETRY_BACKOFF_MS", 180),
            wait_after_ms=_ei("ACTION_WAIT_AFTER_MS", 250),
            max_candidates=_ei("ACTION_MAX_CANDIDATES", 6),
        )


@dataclass
class CacheSettings:
    cache_dir: str = "./cache"
    selector_memory_path: str = "./cache/selector_memory.json"
    selector_memory_max: int = 2_000
    selector_memory_write_batch: int = 10

    @classmethod
    def from_env(cls) -> "CacheSettings":
        load_local_env()
        return cls(
            cache_dir=_e("CACHE_DIR", "./cache"),
            selector_memory_path=_e("SELECTOR_MEMORY_PATH", "./cache/selector_memory.json"),
            selector_memory_max=_ei("SELECTOR_MEMORY_MAX", 2_000),
            selector_memory_write_batch=_ei("SELECTOR_MEMORY_WRITE_BATCH", 10),
        )


@dataclass
class ProxyConfig:
    """ProxyEmpire rotating residential proxy settings.

    Set KS_PROXY_ENABLED=true and supply credentials via environment variables
    (or .env).  The proxy URL format expected by Playwright is:

        http://username:password@proxy.proxyempire.io:9000

    ProxyEmpire supports optional country/city targeting via the username suffix:
        username-country-us   - route through a US exit node
        username-country-gb   - UK exit node
    Set KS_PROXY_COUNTRY to a two-letter ISO code to enable geo-targeting.
    """
    enabled: bool = False
    host: str = "proxy.proxyempire.io"
    port: int = 9000
    username: str = ""
    password: str = ""
    # Optional ISO-3166-1 alpha-2 country code for geo-targeting ("us", "gb", ...)
    country: str = ""

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        load_local_env()
        return cls(
            enabled=_eb("PROXY_ENABLED", False),
            host=_e("PROXY_HOST", "proxy.proxyempire.io"),
            port=_ei("PROXY_PORT", 9000),
            username=_e("PROXY_USERNAME", ""),
            password=_e("PROXY_PASSWORD", ""),
            country=_e("PROXY_COUNTRY", ""),
        )

    def playwright_proxy(self) -> Optional[dict]:
        """Return a Playwright-compatible proxy dict, or None if disabled/unconfigured."""
        if not self.enabled or not self.username or not self.password:
            return None
        # Build geo-targeted username suffix when a country is specified
        user = self.username
        if self.country:
            user = f"{user}-country-{self.country.lower()}"
        return {
            "server": f"http://{self.host}:{self.port}",
            "username": user,
            "password": self.password,
        }


@dataclass
class FrameworkConfig:
    """Top-level configuration container for the automation framework."""
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    action: ActionSettings = field(default_factory=ActionSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)

    @classmethod
    def from_env(cls) -> "FrameworkConfig":
        return cls(
            browser=BrowserConfig.from_env(),
            llm=LLMConfig.from_env(),
            extraction=ExtractionSettings.from_env(),
            action=ActionSettings.from_env(),
            cache=CacheSettings.from_env(),
            proxy=ProxyConfig.from_env(),
        )


_default: Optional[FrameworkConfig] = None


def get_config() -> FrameworkConfig:
    """Return the process-wide config, loading from environment on first call."""
    global _default
    if _default is None:
        _default = FrameworkConfig.from_env()
    return _default


def reset_config() -> None:
    """Reset the process-wide config singleton (useful in tests)."""
    global _default
    _default = None