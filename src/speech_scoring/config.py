"""Runtime configuration from the environment / .env (see .env.example)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    whisper_model: str = "small.en"

    # Optional: rephrase grounded feedback with Claude (falls back to templates if unset).
    anthropic_api_key: str | None = None
    llm_model: str = "claude-haiku-5-5"

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "speechocean762"


def get_settings() -> Settings:
    return Settings()
