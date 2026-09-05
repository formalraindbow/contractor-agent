"""Модель за одной строкой конфига: OpenAI-совместимый клиент с запасными моделями.

Groq, OpenRouter, vLLM в контуре банка — меняется только ``LLM_BASE_URL`` и
``LLM_MODEL``. Бесплатные модели OpenRouter упираются в лимиты, поэтому основная
модель + список запасных: ``with_fallbacks`` переключает на следующую при ошибке.
``bind_tools`` и ``with_structured_output`` применяются к каждой модели отдельно —
обёртка с fallback их не умеет.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from contractor_agent.settings import Settings

HEADERS = {
    "HTTP-Referer": "https://github.com/formalraindbow/contractor-agent",
    "X-Title": "kontragent-agent",
}


class LLM:
    """Основная модель и запасные; отдаёт готовые цепочки с инструментами или схемой."""

    def __init__(self, models: list[BaseChatModel]) -> None:
        if not models:
            raise ValueError("нужна хотя бы одна модель")
        self.models = models

    @property
    def name(self) -> str:
        return getattr(self.models[0], "model_name", type(self.models[0]).__name__)

    def with_tools(self, tools: list[Any]) -> Runnable:
        bound = [m.bind_tools(tools) for m in self.models]
        return bound[0].with_fallbacks(bound[1:]) if len(bound) > 1 else bound[0]

    def structured(self, schema: type) -> Runnable:
        chains: list[Runnable] = []
        for m in self.models:
            chains.append(m.with_structured_output(schema, method="function_calling"))
            chains.append(m.with_structured_output(schema, method="json_schema"))
        return chains[0].with_fallbacks(chains[1:])


def make_llm(
    settings: Settings,
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    reasoning_effort: str | None = "inherit",
) -> LLM:
    """Модель агента по умолчанию; судья эвалов передаёт свой адрес, ключ и пустые запасные."""
    names = [
        model or settings.llm_model,
        *(settings.fallback_models if fallbacks is None else fallbacks),
    ]
    effort = settings.llm_reasoning_effort if reasoning_effort == "inherit" else reasoning_effort
    extra: dict[str, Any] = {"reasoning_effort": effort} if effort else {}
    return LLM(
        [
            ChatOpenAI(
                model=name,
                base_url=base_url or settings.llm_base_url,
                api_key=api_key or settings.llm_api_key or "missing",
                temperature=0,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
                max_tokens=settings.llm_max_tokens,
                default_headers=HEADERS,
                **extra,
            )
            for name in names
        ]
    )


def make_judge_llm(settings: Settings, model: str | None = None) -> LLM:
    return make_llm(
        settings,
        model or settings.judge_model,
        base_url=settings.judge_base_url,
        api_key=settings.judge_api_key,
        fallbacks=[],
        reasoning_effort=None,  # судья думает как умеет: качество важнее скорости
    )
