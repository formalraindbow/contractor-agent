"""Схема ответа агента — контракт с API и фронтом (`Answer`, `Card`).

Одна и та же модель для узла ``finalize``, для SSE-события ``end`` и для React:
фронт рисует по структуре, не по свободному тексту. Карточка (``Card``) собирается
кодом из сигналов — модель пишет только ``text_md`` и цитаты; так маленькая модель
не может испортить вердикт или потерять пробел.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from contractor_agent.signals.model import Severity, Verdict


class Citation(BaseModel):
    """Утверждение и адрес поля отчёта, на которое оно опирается."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="утверждение своими словами, с числом, если оно есть")
    source_path: str = Field(description="адрес поля: report.executionProceedings[3].amount")


class Attention(BaseModel):
    claim: str
    severity: Severity
    source_path: str


class CardLabels(BaseModel):
    """Метки банка словами, как их показывает сам банк; не объясняем и не пересчитываем."""

    riskLevel: str = Field(description="светофор: зелёный / жёлтый / красный / серый")  # noqa: N815
    zskRiskLevel: str = Field(description="ЗСК: зелёный или серый")  # noqa: N815


class Card(BaseModel):
    """Карточка-рекомендация по одной компании; собирается кодом из сигналов."""

    inn: str
    name: str
    labels: CardLabels
    verdict: Verdict
    attention: list[Attention] = Field(default_factory=list, description="на что обратить внимание")
    ask_before: list[str] = Field(default_factory=list, description="что запросить у контрагента")
    gaps: list[str] = Field(default_factory=list, description="чего в отчёте нет")
    report_date: date


class Draft(BaseModel):
    """То, что пишет модель в ``finalize``: вид ответа, текст и цитаты. Остальное — код.

    Текст — списком строк, а не одной строкой: маленькие модели теряют перенос строки
    внутри JSON-строки (проверено живьём — остаётся «n» или пробелы), а склеить список
    кодом надёжно.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["card", "answer", "comparison", "refusal"]
    lines: list[str] = Field(
        description="ответ пользователю в Markdown построчно: одна строка — один элемент, "
        "пустая строка — пустой элемент; простым языком, с датой отчёта"
    )
    citations: list[Citation] = Field(default_factory=list)

    @property
    def text_md(self) -> str:
        return "\n".join(self.lines).strip()


class Answer(BaseModel):
    """Итог прогона графа: текст, карточки, цитаты и даты отчётов."""

    kind: Literal["card", "answer", "comparison", "refusal"]
    text_md: str
    card: Card | None = None
    cards: list[Card] = Field(default_factory=list, description="для сравнения нескольких компаний")
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[Citation] = Field(
        default_factory=list, description="цитаты, не прошедшие проверку — не факты"
    )
    report_dates: dict[str, str] = Field(default_factory=dict, description="ИНН → дата отчёта ISO")
