"""События графа → SSE-конверт ``{type, thread_id, run_id, seq, contract_version, ts, data}``.

Типы: ``token`` — кусок текста модели из узла ``agent``; ``tool`` — вызов
инструмента (имя, аргументы, доступность, размер ответа) — это и есть «агент
работает» на UI; ``end`` — ``Answer`` и ``usage``; ``error`` — исключение.
``interrupt`` зарезервирован под human-in-the-loop. Один прогон — один
``run_id`` и сквозной ``seq``; ``thread_id`` — сессия и ключ чекпоинтера.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.graph import initial_state
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Answer
from contractor_agent.agent.telemetry import RunMetrics

CONTRACT_VERSION = "1.0"
EventType = Literal["token", "tool", "interrupt", "end", "error"]


class Envelope(BaseModel):
    type: EventType
    thread_id: str
    run_id: str
    seq: int
    contract_version: str = CONTRACT_VERSION
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))
    data: dict[str, Any] = Field(default_factory=dict)


def new_run_id() -> str:
    return uuid.uuid4().hex


async def stream_run(
    runtime: AgentRuntime, question: str, thread_id: str, run_id: str | None = None
) -> AsyncIterator[Envelope]:
    assert runtime.graph is not None
    run_id = run_id or new_run_id()
    seq = 0

    def envelope(type_: EventType, data: dict[str, Any]) -> Envelope:
        nonlocal seq
        seq += 1
        return Envelope(type=type_, thread_id=thread_id, run_id=run_id, seq=seq, data=data)

    metrics = RunMetrics()
    config = {
        "callbacks": [metrics],
        "configurable": {"thread_id": thread_id},
        "recursion_limit": runtime.settings.recursion_limit,
    }
    answer: Answer | None = None
    try:
        seen_tools = 0
        async with asyncio.timeout(runtime.settings.run_timeout_s):
            async for chunk in runtime.graph.astream(
                initial_state(question), config=config, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    if node == "tools":
                        trace = update.get("trace") or []
                        for call in trace[seen_tools:]:
                            yield envelope("tool", call.model_dump())
                        seen_tools = len(trace)
                    if node in ("validate", "guard") and update.get("answer") is not None:
                        answer = update["answer"]
        if answer is None:
            raise RuntimeError("граф завершился без ответа")
        state = await runtime.graph.aget_state(config)
        answer.runtime = {
            **metrics.result(),
            "build": runtime.fingerprint,
            "provider": runtime.settings.llm_base_url,
        }
        usage = answer.runtime["usage"]
        runtime.record_trace(thread_id, question, state.values, answer)
        yield envelope("end", {"output": answer.model_dump(mode="json"), "usage": usage})
    except Exception as e:  # ошибка — событие, а не разрыв потока
        yield envelope(
            "error",
            {
                "type": type(e).__name__,
                "message": "Не удалось завершить проверку. Повторите запрос.",
                "runtime": metrics.result(),
            },
        )


def _text(message: BaseMessage) -> str:
    if not isinstance(message, AIMessage | AIMessageChunk):
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    return ""


def _usage(messages: list[BaseMessage]) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        for key in total:
            total[key] += int(usage.get(key) or 0)
    return total
