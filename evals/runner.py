"""Прогон эталона через агента: кэш, троттлинг, повторы, судья.

Каждый ответ сохраняется в ``runs/evals/<модель>/<вопрос>-<повтор>.json`` вместе
с тем, что агент получил от инструментов, результатами детерминированных проверок
и вердиктом судьи. Повторный запуск читает кэш — так эталон гоняется по частям
и не тратит лимиты бесплатных моделей.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.citations import check_citation
from contractor_agent.agent.llm import LLM
from contractor_agent.agent.prompt import PROMPT_VERSION, prompt_fingerprint
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
    category: str = ""
    effective_type: str = ""  # dialog меряется типом последнего вопроса
    model: str
    prompt_version: str = ""  # «v3 · a1b2c3d4»: без неё цифры качества не воспроизводимы
    repeat: int
    question: str
    tool_calls: int = 0  # инструментов на последнем ходу (guard: должно быть 0)
    answer: Answer | None = None
    tool_outputs: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)
    checks_passed: bool = False
    check_failures: list[str] = Field(default_factory=list)
    check_notes: dict[str, bool] = Field(default_factory=dict)
    judge: JudgeVerdict | None = None
    duration_s: float = 0.0
    error: str | None = None
    error_stage: str | None = None


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
        self.cache_dir = cache_dir / model_slug(model_name)
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
        self,
        questions: Sequence[GoldQuestion],
        *,
        repeats: int = 1,
        refresh: bool = False,
        rejudge: bool = False,
        cached_only: bool = False,
    ) -> list[RunRecord]:
        """``rejudge`` — ответы агента из кэша, вердикт судьи считается заново (правка рубрики);
        ``cached_only`` — вопросы без кэша пропускаются (пересуживаем старый прогон, не гоняем
        новые вопросы на другом промпте под старой меткой)."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        records: list[RunRecord] = []
        for question in questions:
            for repeat in range(repeats):
                path = self._path(question, repeat)
                if path.exists() and not refresh:
                    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
                    if record.answer is not None:  # ошибка без ответа — не кэш, гоняем заново
                        if rejudge:
                            record.judge = None
                        changed = self._recheck(question, record)
                        changed = await self._rejudge(question, record) or changed
                        if changed:
                            path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                        records.append(record)
                        continue
                if cached_only:
                    continue
                record = await self.run_one(question, repeat)
                path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
                records.append(record)
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
        return records

    def _recheck(self, question: GoldQuestion, record: RunRecord) -> bool:
        """Проверки кодом пересчитываются из кэша: правка проверки действует задним числом."""
        assert record.answer is not None
        self._revalidate_citations(record.answer)
        result = self._check(question, record)
        before = (record.checks_passed, record.check_failures, record.check_notes)
        record.checks_passed, record.check_failures, record.check_notes = (
            result.passed,
            result.failures,
            result.notes,
        )
        return before != (record.checks_passed, record.check_failures, record.check_notes)

    def _check(self, question: GoldQuestion, record: RunRecord) -> CheckResult:
        assert record.answer is not None
        expected = {inn: self.gold.card(inn).expected_verdict for inn in question.all_inns}
        return check(
            question,
            record.answer,
            self.gold.card(question.inn).report_date,
            tool_calls=record.tool_calls,
            expected_by_inn=expected if question.effective_type == "comparison" else None,
        )

    def _revalidate_citations(self, answer: Answer) -> None:
        """Цитаты проверяются заново текущим валидатором: его правки действуют задним числом."""
        inns = list(answer.report_dates)
        if not inns:
            return
        valid, invalid = [], []
        for c in [*answer.citations, *answer.invalid_citations]:
            (valid if check_citation(self.runtime.source, inns, c).ok else invalid).append(c)
        answer.citations, answer.invalid_citations = valid, invalid

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

    async def _rejudge(self, question: GoldQuestion, record: RunRecord) -> bool:
        """Ответ агента есть, а вердикта судьи нет (судья упал) — судим заново, агента не гоняем."""
        if self.judge_llm is None or record.judge is not None or record.answer is None:
            return False
        try:
            record.judge = await self._judge(question, record.answer, record.tool_outputs)
        except Exception as e:
            record.error = f"{type(e).__name__}: {e}"
            record.error_stage = "judge"
            return False
        record.error = None
        record.error_stage = None
        return True

    async def run_one(self, question: GoldQuestion, repeat: int) -> RunRecord:
        record = RunRecord(
            question_id=question.id,
            inn=question.inn,
            type=question.type,
            topic=question.topic,
            category=question.category,
            effective_type=question.effective_type,
            model=self.model_name,
            prompt_version=f"{PROMPT_VERSION} · {prompt_fingerprint()}",
            repeat=repeat,
            question=question.question,
        )
        thread_id = f"eval-{question.id}-{repeat}"
        started = time.perf_counter()
        stage = "agent"
        try:
            history = follow_up_history(self.gold.card(question.inn)) if question.follow_up else ()
            for prior in question.prior_questions:  # dialog: прошлые ходы задаются живьём
                await asyncio.wait_for(
                    self.runtime.ask(prior, thread_id=thread_id, history=history),
                    self.agent_timeout_s,
                )
                history = ()
            answer = await asyncio.wait_for(
                self.runtime.ask(question.question, thread_id=thread_id, history=history),
                self.agent_timeout_s,
            )
            state = await self.runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
            values = state.values or {}
            messages = values.get("messages") or []
            record.answer = answer
            record.tool_outputs = _tool_outputs(_last_turn(messages, question.question))
            record.trace = [t.model_dump() for t in values.get("trace") or []]
            record.tool_calls = sum(
                1 for m in _last_turn(messages, question.question) if isinstance(m, ToolMessage)
            )
            stage = "checks"
            result: CheckResult = self._check(question, record)
            record.checks_passed = result.passed
            record.check_failures = result.failures
            record.check_notes = result.notes
            if self.judge_llm is not None:
                stage = "judge"
                record.judge = await self._judge(question, answer, record.tool_outputs)
        except Exception as e:  # ошибка прогона — запись, а не остановка эталона
            record.error = f"{type(e).__name__}: {e}"  # проверки, если успели, остаются как есть
            record.error_stage = stage
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


def _last_turn(messages: list, question: str) -> list:
    """Сообщения последнего хода: от последней реплики пользователя с этим вопросом до конца.
    В диалоге судья и счётчик инструментов смотрят только на текущий ответ."""
    start = 0
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage) and str(m.content) == question:
            start = i
    return messages[start:]


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
