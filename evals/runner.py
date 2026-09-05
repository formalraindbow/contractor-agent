"""Прогон эталона через агента: кэш, троттлинг, повторы, судья.

Каждый ответ сохраняется в ``runs/evals/<модель>/<вопрос>-<повтор>.json`` вместе
с тем, что агент получил от инструментов, результатами детерминированных проверок
и вердиктом судьи. Повторный запуск читает кэш — так эталон гоняется по частям
и не тратит лимиты бесплатных моделей.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import re
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.llm import LLM
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Answer
from evals.checks import CheckResult, check
from evals.gold import Gold, GoldCard, GoldQuestion
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
    agent_duration_s: float | None = None
    judge_duration_s: float | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)


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
        agent_timeout_s: float = 900,
        judge_timeout_s: float = 120,
    ) -> None:
        self.gold = gold
        self.runtime = runtime
        self.manifest = {
            **runtime.fingerprint,
            "model": model_name,
            "judge": judge_llm.name if judge_llm else None,
            "gold": hashlib.sha256(gold.model_dump_json().encode()).hexdigest(),
            "evaluator": hashlib.sha256(
                b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
            ).hexdigest(),
            "judge_prompt": hashlib.sha256(
                Path(__file__).with_name("judge.py").read_bytes()
            ).hexdigest(),
        }
        fingerprint = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode()
        ).hexdigest()[:12]
        self.cache_dir = cache_dir / model_slug(model_name) / fingerprint
        self.model_name = model_name
        self.judge_llm = judge_llm
        self.delay_s = delay_s
        # OpenRouter держит соединение служебными строками, таймаут чтения клиента не срабатывает —
        # нужен общий дедлайн на вызов, иначе один зависший провайдер съедает час.
        self.agent_timeout_s = agent_timeout_s
        self.judge_timeout_s = judge_timeout_s

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
                    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
                    if record.answer is not None:  # ошибка без ответа — не кэш, гоняем заново
                        # Existing outputs remain immutable; a changed build gets a new directory.
                        records.append(record)
                        continue
                record = await self.run_one(question, repeat)
                path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                records.append(record)
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
        return records

    async def _judge(
        self, question: GoldQuestion, answer: Answer, tool_outputs: str
    ) -> JudgeVerdict:
        """Судья с дедлайном и второй попыткой: зависший провайдер — не приговор вопросу."""
        assert self.judge_llm is not None
        for attempt in (1, 2):
            try:
                return await asyncio.wait_for(
                    judge(self.judge_llm, question, answer, tool_outputs), self.judge_timeout_s
                )
            except TimeoutError:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    async def run_one(self, question: GoldQuestion, repeat: int) -> RunRecord:
        record = RunRecord(
            question_id=question.id,
            inn=question.inn,
            type=question.type,
            topic=question.topic,
            model=self.model_name,
            repeat=repeat,
            question=question.question,
            manifest=self.manifest,
        )
        thread_id = f"eval-{question.id}-{repeat}-{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            history = follow_up_history(self.gold.card(question.inn)) if question.follow_up else ()
            answer = await asyncio.wait_for(
                self.runtime.ask(question.question, thread_id=thread_id, history=history),
                self.agent_timeout_s,
            )
            record.agent_duration_s = round(time.perf_counter() - started, 3)
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
                judge_started = time.perf_counter()
                record.judge = await self._judge(question, answer, record.tool_outputs)
                record.judge_duration_s = round(time.perf_counter() - judge_started, 3)
        except Exception as e:  # ошибка прогона — запись, а не остановка эталона
            record.error = f"{type(e).__name__}: {e}"  # проверки, если успели, остаются как есть
        record.duration_s = round(time.perf_counter() - started, 1)
        return record


def follow_up_history(card: GoldCard) -> list[BaseMessage]:
    """Предыдущий обмен в той же сессии — без вызова модели: проверяем, что агент помнит,
    о какой компании речь, а не заставляем его отвечать на два вопроса."""
    date = datetime.date.fromisoformat(card.report_date).strftime("%d.%m.%Y")
    return [
        HumanMessage(content=f"Проверь {card.company}, ИНН {card.inn}."),
        AIMessage(
            content=f"Смотрю отчёт по {card.company} (ИНН {card.inn}) от {date}. "
            "Могу рассказать про статус, суды, долги у приставов, финансы и проверки."
        ),
    ]


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
