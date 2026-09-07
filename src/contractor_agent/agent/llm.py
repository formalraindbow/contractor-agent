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
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, InternalServerError

from contractor_agent.settings import Settings

HEADERS = {
    "HTTP-Referer": "https://github.com/formalraindbow/contractor-agent",
    "X-Title": "kontragent-agent",
}


class LLM:
    """Основная модель и запасные; отдаёт готовые цепочки с инструментами или схемой."""

    def __init__(self, models: list[BaseChatModel], *, retry_transport: bool = False) -> None:
        if not models:
            raise ValueError("нужна хотя бы одна модель")
        self.models = models
        self.retry_transport = retry_transport

    def _transport(self, chain: Runnable) -> Runnable:
        if not self.retry_transport:
            return chain
        # Some gateways send Retry-After: 60 with a 502/503. That SDK sleep blocks
        # the whole chat. Retry transient failures once with a short bounded wait;
        # rate limits/authentication errors deliberately do not use this policy.
        return chain.with_retry(
            retry_if_exception_type=(APIConnectionError, InternalServerError),
            stop_after_attempt=2,
            exponential_jitter_params={"initial": 0.5, "max": 2.0, "jitter": 0.5},
        )

    @property
    def name(self) -> str:
        return getattr(self.models[0], "model_name", type(self.models[0]).__name__)

    def with_tools(self, tools: list[Any], *, tool_choice: str | None = None) -> Runnable:
        options = {"tool_choice": tool_choice} if tool_choice is not None else {}
        bound = [self._transport(m.bind_tools(tools, **options)) for m in self.models]
        return bound[0].with_fallbacks(bound[1:]) if len(bound) > 1 else bound[0]

    def structured(self, schema: type, *, method: str = "json_schema") -> Runnable:
        chains: list[Runnable] = []

        def _validate(result: Any) -> Any:
            if result is None:  # модель не заполнила схему — пусть сработает запасная цепочка
                raise ValueError("структурный ответ пуст")
            return schema.model_validate(result)

        validate = RunnableLambda(_validate)
        for m in self.models:
            if method == "prompt_json":
                parser = PydanticOutputParser(pydantic_object=schema)

                def formatted(messages, parser=parser):
                    return [
                        *messages,
                        SystemMessage(
                            content="FINAL_JSON: Верни JSON-объект, "
                            "без блока кода, по приведённой схеме.\n"
                            + parser.get_format_instructions()
                        ),
                    ]

                chains.append(RunnableLambda(formatted) | self._transport(m) | parser | validate)
                continue
            methods = ("json_schema", "function_calling") if method == "json_schema" else (method,)
            for output_method in methods:
                chains.append(
                    self._transport(m.with_structured_output(schema, method=output_method))
                    | validate
                )
        return chains[0].with_fallbacks(chains[1:])


def make_llm(
    settings: Settings,
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    reasoning_effort: str | None = "inherit",
    provider_order: list[str] | None = None,
    retry_transport: bool = True,
) -> LLM:
    """Модель агента по умолчанию; судья эвалов передаёт свой адрес, ключ и пустые запасные."""
    names = [
        model or settings.llm_model,
        *(settings.fallback_models if fallbacks is None else fallbacks),
    ]
    effort = settings.llm_reasoning_effort if reasoning_effort == "inherit" else reasoning_effort
    extra: dict[str, Any] = {"reasoning_effort": effort} if effort else {}
    order = settings.provider_order if provider_order is None else provider_order
    if order:
        extra["extra_body"] = {
            "provider": {"order": order, "allow_fallbacks": settings.llm_provider_allow_fallbacks}
        }
    return LLM(
        [
            ChatOpenAI(
                model=name,
                base_url=base_url or settings.llm_base_url,
                api_key=api_key or settings.llm_api_key or "missing",
                temperature=0,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
                max_retries=0 if retry_transport else 2,
                default_headers=HEADERS,
                **extra,
            )
            for name in names
        ],
        retry_transport=retry_transport,
    )


def make_judge_llm(settings: Settings, model: str | None = None) -> LLM:
    return make_llm(
        settings,
        model or settings.judge_model,
        base_url=settings.judge_base_url,
        api_key=settings.judge_api_key,
        fallbacks=[],
        reasoning_effort=None,  # судья думает как умеет: качество важнее скорости
        provider_order=[],  # the judge family has its own available providers
        retry_transport=False,  # offline judging keeps the SDK's conservative retry policy
    )
