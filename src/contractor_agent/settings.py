"""Настройки проекта: pydantic-settings, значения из окружения и ``.env``.

В банке те же переменные придут из vault; у нас — из ``.env`` (в .gitignore).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from contractor_agent.data.loader import ReportSource, load_snapshot


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("data")
    index_path: Path = Path(".cache/snapshot.sqlite")
    report_source: Literal["snapshot", "sqlite"] = "snapshot"

    mcp_transport: Literal["stdio", "streamable-http"] = "stdio"
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765

    api_host: str = "127.0.0.1"
    web_password: str | None = (
        None  # WEB_PASSWORD: пароль на доступ к странице (для публичной ссылки)
    )
    api_port: int = 8080

    llm_base_url: str = "https://ai.api.cloud.yandex.net/v1"
    llm_model: str = "gpt://<folder-id>/gpt-oss-120b"
    llm_fallback_models: str = ""  # резервы только при явной настройке
    llm_timeout: float = 60
    run_timeout_s: float = 180
    llm_max_tokens: int = 4096
    llm_max_retries: int = 1
    llm_reasoning_effort: str | None = (
        "low"  # gpt-oss: low в 3 раза быстрее medium на том же ответе
    )
    judge_model: str = "openai/gpt-oss-120b"  # судья эвалов — сильнее агента, чтобы не судить себя
    judge_base_url: str = "https://openrouter.ai/api/v1"  # судья не зависит от адреса агента
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    yandex_api_key: str | None = None
    llm_api_key_override: str | None = None  # LLM_API_KEY_OVERRIDE: ключ для произвольного base_url
    recursion_limit: int = 32  # two companies, sequential tools and one repair; time stays bounded
    runs_dir: Path = Path("runs")

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.llm_fallback_models.split(",") if m.strip()]

    def api_key_for(self, base_url: str) -> str | None:
        """Ключ по адресу: Groq, Яндекс, иначе OpenRouter."""
        if "groq" in base_url:
            return self.groq_api_key
        if "yandex" in base_url:
            return self.yandex_api_key
        return self.openrouter_api_key

    @property
    def llm_api_key(self) -> str | None:
        """Ключ агента: явный override — приоритет, иначе по адресу."""
        return self.llm_api_key_override or self.api_key_for(self.llm_base_url)

    @property
    def judge_api_key(self) -> str | None:
        return self.api_key_for(self.judge_base_url)


def make_source(settings: Settings) -> ReportSource:
    """Источник отчётов по настройке: снапшот в памяти или SQLite-индекс (собирается при нужде)."""
    if settings.report_source == "sqlite":
        from contractor_agent.data.index import SqliteSource, build_index

        if not settings.index_path.is_file():
            build_index(load_snapshot(settings.data_dir), settings.index_path)
        return SqliteSource(settings.index_path)
    return load_snapshot(settings.data_dir)
