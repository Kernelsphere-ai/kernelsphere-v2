import os
from pathlib import Path

_LOADED = False


def load_local_env(env_path: str = ".env", override: bool = False) -> None:
    """Load KEY=VALUE pairs from .env into process env."""
    global _LOADED
    if _LOADED and not override:
        return
    path = Path(env_path)
    if not path.exists():
        _LOADED = True
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    _LOADED = True


def reset_env_loader() -> None:
    """Reset loaded state so the next call to load_local_env re-reads the file.

    Intended for use in tests or when the .env file changes at runtime.
    """
    global _LOADED
    _LOADED = False
