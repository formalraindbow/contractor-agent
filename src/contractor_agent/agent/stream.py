"""События графа → SSE-конверт ``{type, thread_id, run_id, seq, contract_version, ts, data}``.

Типы: ``token`` — проверенный итоговый текст (не промежуточный черновик); ``tool`` — вызов
инструмента (имя, аргументы, доступность, размер ответа) — это и есть «агент
работает» на UI; ``end`` — ``Answer`` и ``usage``; ``error`` — исключение.
``interrupt`` зарезервирован под human-in-the-loop. Один прогон — один
``run_id`` и сквозной ``seq``; ``thread_id`` — сессия и ключ чекпоинтера.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.graph import initial_state
from contractor_agent.agent.runtime import AgentRuntime, RunBusyError
from contractor_agent.agent.schema import Answer

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

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": runtime.settings.recursion_limit,
    }
    answer: Answer | None = None
    try:
        async with runtime.run_slot(thread_id):
            async for chunk in runtime.graph.astream(
                initial_state(question), config=config, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    if node in ("tools", "evidence"):
                        for call in update.get("trace") or []:
                            yield envelope("tool", call.model_dump())
                    if node in ("validate", "guard") and update.get("answer") is not None:
                        answer = update["answer"]
            if answer is None:
                raise RuntimeError("граф завершился без ответа")
            state = await runtime.graph.aget_state(config)
            messages = state.values.get("messages") or []
            turn_start = next(
                (
                    i
                    for i in range(len(messages) - 1, -1, -1)
                    if isinstance(messages[i], HumanMessage) and messages[i].content == question
                ),
                len(messages),
            )
            usage = _usage(messages[turn_start:])
            runtime.record_trace(thread_id, question, state.values, answer)
            # Clients may animate this text, but must never see unvalidated reasoning/drafts.
            if answer.text_md:
                yield envelope("token", {"text": answer.text_md})
            yield envelope("end", {"output": answer.model_dump(mode="json"), "usage": usage})
    except RunBusyError as e:
        yield envelope("error", {"type": "Busy", "message": str(e)})
    except TimeoutError:
        yield envelope(
            "error",
            {
                "type": "Timeout",
                "message": "Ответ занимает слишком много времени. Попробуйте повторить запрос.",
            },
        )
    except Exception:
        logging.getLogger(__name__).exception("Agent stream failed")
        yield envelope(
            "error",
            {"type": "AgentError", "message": "Не удалось завершить запрос. Попробуйте ещё раз."},
        )


def _usage(messages: list[BaseMessage]) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        for key in total:
            total[key] += int(usage.get(key) or 0)
    return total
