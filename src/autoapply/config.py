"""Runtime settings, read from environment variables (and a local ``.env``).

The model ID is deliberately *required* and never hardcoded, so the same code
runs against whichever Claude model the user configures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
BASELINE_ALIAS = "baseline"  # AUTOAPPLY_MODEL=baseline runs the offline no-LLM baseline


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    model: str
    effort: str = "medium"
    max_auto_revisions: int = 1
    refusal_fallback: bool = True
    data_dir: Path = Path(".autoapply")

    @property
    def checkpoint_db(self) -> Path:
        """SQLite file holding LangGraph checkpoints (paused reviews)."""
        return self.data_dir / "checkpoints.sqlite"

    @property
    def usage_log(self) -> Path:
        """JSONL token-usage log."""
        return self.data_dir / "token_usage.jsonl"

    @property
    def scam_model(self) -> Path:
        """Trained scam pre-screen (created by `autoapply scam train`)."""
        return self.data_dir / "models" / "scam_screen.joblib"

    @property
    def outputs_dir(self) -> Path:
        """Folder for human-approved outputs."""
        return self.data_dir / "outputs"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Load settings from the environment, after reading ``env_file`` if present.

    Raises:
        ConfigError: if ``AUTOAPPLY_MODEL`` is unset or a value is invalid.
    """
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)

    model = os.getenv("AUTOAPPLY_MODEL", "").strip()
    if not model:
        raise ConfigError(
            "AUTOAPPLY_MODEL is not set. Copy .env.example to .env and set the Claude model ID "
            "(e.g. AUTOAPPLY_MODEL=claude-opus-5-5)."
        )
    effort = os.getenv("AUTOAPPLY_EFFORT", "medium").strip().lower()
    if effort not in EFFORT_LEVELS:
        raise ConfigError(f"AUTOAPPLY_EFFORT must be one of {sorted(EFFORT_LEVELS)}, got {effort!r}")
    try:
        revisions = int(os.getenv("AUTOAPPLY_MAX_AUTO_REVISIONS", "1"))
    except ValueError as exc:
        raise ConfigError("AUTOAPPLY_MAX_AUTO_REVISIONS must be an integer") from exc
    if revisions < 0:
        raise ConfigError("AUTOAPPLY_MAX_AUTO_REVISIONS must be >= 0")

    return Settings(
        model=model,
        effort=effort,
        max_auto_revisions=revisions,
        refusal_fallback=_as_bool(os.getenv("AUTOAPPLY_REFUSAL_FALLBACK", "true")),
        data_dir=Path(os.getenv("AUTOAPPLY_DATA_DIR", ".autoapply")),
    )
