"""Прогон эталона через агента: кэш, троттлинг, повторы, судья.

Каждый ответ сохраняется в ``runs/evals/<модель>/<вопрос>-<повтор>.json`` вместе
с тем, что агент получил от инструментов, результатами детерминированных проверок
и вердиктом судьи. Повторный запуск читает кэш — так эталон гоняется по частям
и не тратит лимиты бесплатных моделей.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.llm import LLM
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Answer
from evals.checks import CheckResult, check
from evals.gold import Gold, GoldQuestion
from evals.judge import JudgeVerdict, judge

TOOL_OUTPUT_LIMIT = 20_000


class RunRecord(BaseModel):
    question_id: str
    inn: str
    type: str
    topic: str = "other"
    model: str
    repeat: int
    question: str
    answer: Answer | None = None
    tool_outputs: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)
    checks_passed: bool = False
    check_failures: list[str] = Field(default_factory=list)
    check_notes: dict[str, bool] = Field(default_factory=dict)
    judge: JudgeVerdict | None = None
    duration_s: float = 0.0
    error: str | None = None


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class EvalRunner:
    def __init__(
        self,
        gold: Gold,
        runtime: AgentRuntime,
        *,
        cache_dir: Path,
        model_name: str,
        judge_llm: LLM | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.gold = gold
        self.runtime = runtime
        self.cache_dir = cache_dir / model_slug(model_name)
        self.model_name = model_name
        self.judge_llm = judge_llm
        self.delay_s = delay_s

    def _path(self, question: GoldQuestion, repeat: int) -> Path:
        return self.cache_dir / f"{question.id}-{repeat}.json"

    async def run(
        self, questions: Sequence[GoldQuestion], *, repeats: int = 1, refresh: bool = False
    ) -> list[RunRecord]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        records: list[RunRecord] = []
        for question in questions:
            for repeat in range(repeats):
                path = self._path(question, repeat)
                if path.exists() and not refresh:
                    records.append(RunRecord.model_validate_json(path.read_text(encoding="utf-8")))
                    continue
                record = await self.run_one(question, repeat)
                path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                records.append(record)
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
        return records

    async def run_one(self, question: GoldQuestion, repeat: int) -> RunRecord:
        record = RunRecord(
            question_id=question.id,
            inn=question.inn,
            type=question.type,
            topic=question.topic,
            model=self.model_name,
            repeat=repeat,
            question=question.question,
        )
        thread_id = f"eval-{question.id}-{repeat}"
        started = time.perf_counter()
        try:
            answer = await self.runtime.ask(question.question, thread_id=thread_id)
            state = await self.runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
            values = state.values or {}
            record.answer = answer
            record.tool_outputs = _tool_outputs(values.get("messages") or [])
            record.trace = [t.model_dump() for t in values.get("trace") or []]
            result: CheckResult = check(question, answer, self.gold.card(question.inn).report_date)
            record.checks_passed = result.passed
            record.check_failures = result.failures
            record.check_notes = result.notes
            if self.judge_llm is not None:
                record.judge = await judge(self.judge_llm, question, answer, record.tool_outputs)
        except Exception as e:  # ошибка прогона — запись, а не остановка эталона
            record.error = f"{type(e).__name__}: {e}"
            record.check_failures = [record.error]
        record.duration_s = round(time.perf_counter() - started, 1)
        return record


def _tool_outputs(messages: list) -> str:
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = (
                m.content
                if isinstance(m.content, str)
                else json.dumps(m.content, ensure_ascii=False)
            )
            parts.append(f"[{m.name}] {content}")
    return "\n".join(parts)[:TOOL_OUTPUT_LIMIT]
