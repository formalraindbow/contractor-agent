"""Контракт слоя сигналов: ``Signal``, ``Gap``, ``SignalSet``.

Здесь нет правил — только форма, которую правила заполняют, а инструменты MCP,
карточка агента и валидатор цитат читают.

* ``Signal`` — один факт из отчёта с адресом поля (``source_path``) и тяжестью.
  Тяжесть: ``critical`` — компания может не дожить до конца договора или уже не
  платит; ``moderate`` — стоит проверить дополнительно; ``info`` — для сведения,
  на рекомендацию не влияет. ``terminal`` — факт, который сразу даёт
  «не рекомендуем» (банкротство, исключение из ЕГРЮЛ).
* ``Gap`` — критерий, который оценить нельзя: секции нет, секция пуста, нет
  поля, или критерий не применим (ИП не сдают отчётность). Пробел — не сигнал:
  «нет данных» ≠ «нет нарушений», и на рекомендацию он не влияет, но обязан
  быть виден.
* ``SignalSet`` — итог по компании: сигналы по тяжести, пробелы, рекомендация.

Адреса полей пишутся в терминах отчёта (``report.executionProceedings[3].amount``)
и обязаны резолвиться через ``data.paths.resolve`` — ``check_paths`` это проверяет.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contractor_agent.data.model import Report
from contractor_agent.data.paths import PathNotFoundError, resolve

MAX_SOURCE_PATHS = 20


class Severity(StrEnum):
    CRITICAL = "critical"
    MODERATE = "moderate"
    INFO = "info"

    @property
    def rank(self) -> int:
        return _RANK[self]


_RANK = {Severity.CRITICAL: 0, Severity.MODERATE: 1, Severity.INFO: 2}


class Verdict(StrEnum):
    """Три исхода проверки по сигналам отчёта.

    Условия оплаты зависят от роли пользователя и конкретной сделки;
    категория риска сама по себе не предписывает предоплату.
    Коды в контракте с API и фронтом не меняются.
    """

    OK = "ok"  # можно работать
    CHECK = "check"  # до договора проверить
    NOT_RECOMMENDED = "not_recommended"  # существенные риски


VERDICT_RU: dict[Verdict, str] = {
    Verdict.OK: "можно работать",
    Verdict.CHECK: "стоит проверить до договора",
    Verdict.NOT_RECOMMENDED: "есть существенные риски",
}
TERMINAL_RU = (  # штатный исход + причина: валидатор находит фразу исхода, судья — оговорку
    f"{VERDICT_RU[Verdict.NOT_RECOMMENDED]}; компания в процедуре банкротства "
    "или исключается из реестра — сделки могут быть оспорены, обязательства не исполнены"
)


class Origin(StrEnum):
    COMPUTED = "computed"  # посчитали сами по сырым полям
    BANK_FLAG = "bank_flag"  # взяли готовый признак из reputationalRisks


class GapReason(StrEnum):
    SECTION_ABSENT = "section_absent"
    SECTION_EMPTY = "section_empty"
    FIELD_ABSENT = "field_absent"
    NOT_APPLICABLE = "not_applicable"


class SignalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Signal(SignalModel):
    code: str = Field(description="стабильный код правила, например enforcement_active")
    severity: Severity
    title_ru: str = Field(description="короткий заголовок для карточки")
    value: Any = Field(default=None, description="число, строка или словарь с числами")
    source_path: str = Field(description="главный адрес поля в отчёте — резолвится")
    source_paths: list[str] = Field(
        default_factory=list, max_length=MAX_SOURCE_PATHS, description="подтверждающие адреса"
    )
    explanation_ru: str = Field(description="1–2 предложения фактов, без вердикта")
    origin: Origin = Origin.COMPUTED
    year: int | None = Field(default=None, description="год отчётности, если факт за год")
    terminal: bool = Field(default=False, description="сразу «не рекомендуем»")
    bank_text: str | None = Field(default=None, description="текст банка как данные, не наш")

    @field_validator("value", mode="before")
    @classmethod
    def _json_native(cls, value: Any) -> Any:
        return jsonable(value)

    @model_validator(mode="after")
    def _terminal_is_critical(self) -> Signal:
        if self.terminal and self.severity is not Severity.CRITICAL:
            raise ValueError("терминальный сигнал не может быть мягче critical")
        return self


class Gap(SignalModel):
    criterion: str = Field(description="что не удалось оценить: «финансовая отчётность»")
    reason: GapReason
    source_path: str = Field(description="корень секции или поле — резолвится в None/[]")
    text_ru: str = Field(description="«в отчёте нет … — оценить по этому критерию нельзя»")
    ask_ru: str | None = Field(default=None, description="что запросить у контрагента")
    floor_check: bool = Field(
        default=False,
        description="пробел, при котором рекомендация не может быть лучше «стоит проверить»",
    )


class SignalSet(SignalModel):
    inn: str
    report_date: date
    signals: tuple[Signal, ...] = ()
    gaps: tuple[Gap, ...] = ()
    verdict: Verdict
    score: int = Field(ge=0, description="3·critical + 1·moderate")
    terminal: bool = False

    @model_validator(mode="after")
    def _sorted_by_severity(self) -> SignalSet:
        ordered = tuple(
            sorted(
                self.signals, key=lambda s: (s.severity.rank, not s.terminal, s.code, s.title_ru)
            )
        )
        if ordered != self.signals:
            object.__setattr__(self, "signals", ordered)
        return self

    def by_severity(self, severity: Severity) -> list[Signal]:
        return [s for s in self.signals if s.severity is severity]

    @property
    def codes(self) -> set[str]:
        return {s.code for s in self.signals}


class RuleResult(NamedTuple):
    """Что вернуло одно семейство правил."""

    signals: list[Signal]
    gaps: list[Gap]


def jsonable(value: Any) -> Any:
    """Правила считают в ``Decimal`` и ``date``; в контракт уходят числа и строки ISO.

    Так ``value`` одинаково читается в JSON инструмента MCP, в карточке и после
    round-trip через SQLite: ``Decimal("1571231.50")`` → ``1571231.5``, ``Decimal("2")`` → ``2``.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    return value


class SignalPathError(ValueError):
    """Сигнал или пробел ссылается на адрес, которого в отчёте нет — ошибка правила."""


def check_paths(report: Report, item: Signal | Gap) -> None:
    """Каждый адрес обязан резолвиться. Ловим ошибки правил в тестах, а не у клиента."""
    paths = [item.source_path, *(item.source_paths if isinstance(item, Signal) else [])]
    for path in paths:
        try:
            resolve(report, path)
        except PathNotFoundError as e:
            raise SignalPathError(f"{type(item).__name__} {item_code(item)}: {e}") from e


def item_code(item: Signal | Gap) -> str:
    return item.code if isinstance(item, Signal) else item.criterion
