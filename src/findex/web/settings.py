"""Configuration from the environment (12-factor, rule III): nothing deploy-specific in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read from environment variables (and a `.env` file in development).

    INDEX_PATH=data/index.pkl LOG_LEVEL=DEBUG uvicorn findex.web.app:app
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("data/index.pkl")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True  # one JSON object per log line (what log platforms parse)
    page_size: int = Field(default=10, ge=1, le=100)
    cors_origins: list[str] = []  # e.g. CORS_ORIGINS='["https://example.org"]'


@lru_cache
def get_settings() -> Settings:
    """One Settings per process; also a FastAPI dependency (overridable in tests)."""
    return Settings()
