"""Исполнительные производства: долги, которые уже дошли до приставов.

Активные и завершённые считаются отдельно и никогда не складываются. Активные —
сигнал всегда (флаг банка ставится при любом активном производстве; 33 = 33), а
тяжесть — по соразмерности чистым активам (``finance.debt_scale``), как советовал
кейсодатель: половина чистых активов и больше — критично; десять и больше
производств — критично само по себе; меньше процента при паре мелких — для сведения.

Сумма — нижняя граница: у 770 из 3873 производств суммы нет, текст обязан говорить
«не менее X, у K из N сумма не указана»; ноль не пишем никогда — нулевая сумма
считается неизвестной. Производство без признака активности — не «завершённое»,
а пробел. Список не отдаём — у ЛЕ МОНЛИД 1744 записи; в подтверждающие адреса
идут десять самых крупных.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from contractor_agent.data.model import ExecutionProceeding, Report
from contractor_agent.signals.finance import DebtScale, debt_scale
from contractor_agent.signals.model import Gap, GapReason, RuleResult, Severity, Signal
from contractor_agent.signals.text import date_ru, plural, rub, years_range

SECTION_PATH = "report.executionProceedings"
CRITICAL_ACTIVE_COUNT = 10
STALE_ACTIVE_AFTER_DAYS = 3 * 365
RECENT_DAYS = 365
TOP_ITEMS = 10
BANK_FLAG_CODE = "executionProceedings"


@dataclass(frozen=True)
class Item:
    index: int
    data: ExecutionProceeding

    @property
    def path(self) -> str:
        return f"{SECTION_PATH}[{self.index}]"

    @property
    def known_amount(self) -> Decimal | None:
        """Сумма, если она указана и больше нуля; ноль — как «не указана»."""
        amount = self.data.amount
        return amount if amount is not None and amount > 0 else None


@dataclass(frozen=True)
class Aggregate:
    """Свод по одной группе (активные или завершённые)."""

    items: list[Item]
    known_sum: Decimal
    unknown_count: int
    recent_count: int  # начаты за последние 12 месяцев до даты отчёта
    stale: list[Item]  # старше 3 лет
    earliest: date | None
    latest: date | None

    @property
    def count(self) -> int:
        return len(self.items)

    def top(self, n: int = TOP_ITEMS) -> list[Item]:
        """Самые крупные; без суммы — в конец; при равных суммах новые первыми."""
        return sorted(
            self.items,
            key=lambda i: (
                i.known_amount is None,
                -(i.known_amount or 0),
                i.data.date is None,
                -(i.data.date.toordinal() if i.data.date else 0),
            ),
        )[:n]


def aggregate(items: list[Item], report_date: date) -> Aggregate:
    recent_from = report_date - timedelta(days=RECENT_DAYS)
    stale_before = report_date - timedelta(days=STALE_ACTIVE_AFTER_DAYS)
    dates = [i.data.date for i in items if i.data.date]
    return Aggregate(
        items=items,
        known_sum=sum((i.known_amount for i in items if i.known_amount), Decimal(0)),
        unknown_count=sum(i.known_amount is None for i in items),
        recent_count=sum(1 for i in items if i.data.date and i.data.date >= recent_from),
        stale=[i for i in items if i.data.date and i.data.date < stale_before],
        earliest=min(dates) if dates else None,
        latest=max(dates) if dates else None,
    )


def split(report: Report) -> tuple[Aggregate, Aggregate, list[Item]]:
    """(активные, завершённые, без признака активности)."""
    items = [Item(i, p) for i, p in enumerate(report.execution_proceedings or [])]
    active = [i for i in items if i.data.active is True]
    finished = [i for i in items if i.data.active is False]
    unknown = [i for i in items if i.data.active is None]
    return aggregate(active, report.report_date), aggregate(finished, report.report_date), unknown


def run(report: Report) -> RuleResult:
    signals: list[Signal] = []
    gaps: list[Gap] = []
    state = report.section_state("executionProceedings")
    if state == "absent":
        gaps.append(
            Gap(
                criterion="исполнительные производства",
                reason=GapReason.SECTION_ABSENT,
                source_path=SECTION_PATH,
                text_ru=(
                    "В отчёте нет раздела об исполнительных производствах — есть ли долги "
                    "у приставов, по отчёту сказать нельзя."
                ),
                ask_ru="Проверьте контрагента в банке данных ФССП по ИНН.",
            )
        )
        return RuleResult(signals, gaps)

    active, finished, unknown_status = split(report)
    flag = _bank_flag(report)

    if active.count:
        scale = debt_scale(
            report, active.known_sum, active.unknown_count, active.count, CRITICAL_ACTIVE_COUNT
        )
        signals.append(_active_signal(active, scale, flag))
        if active.unknown_count == active.count:
            n = active.count
            if n == 1:
                text = (
                    "У единственного действующего производства в отчёте не указана сумма — "
                    "размер долга оценить нельзя."
                )
            else:
                text = (
                    f"Ни у одного из {n} действующих производств в отчёте не указана сумма — "
                    f"размер долга оценить нельзя."
                )
            gaps.append(
                Gap(
                    criterion="суммы исполнительных производств",
                    reason=GapReason.FIELD_ABSENT,
                    source_path=f"{active.items[0].path}.amount",
                    text_ru=text,
                    ask_ru="Запросите у контрагента справку ФССП о задолженности.",
                )
            )
        if active.stale:
            oldest = min(i.data.date for i in active.stale if i.data.date)
            signals.append(
                Signal(
                    code="enforcement_stale_active",
                    severity=Severity.INFO,
                    title_ru="Старые непогашенные производства",
                    value={"count": len(active.stale), "oldest_date": oldest},
                    source_path=SECTION_PATH,
                    source_paths=[i.path for i in active.stale[:TOP_ITEMS]],
                    explanation_ru=(
                        f"{len(active.stale)} из {active.count} действующих производств открыты "
                        f"больше трёх лет, самое раннее — от {date_ru(oldest)}."
                    ),
                )
            )
    if unknown_status:
        n = len(unknown_status)
        gaps.append(
            Gap(
                criterion="статус исполнительных производств",
                reason=GapReason.FIELD_ABSENT,
                source_path=f"{unknown_status[0].path}.active",
                text_ru=(
                    f"У {plural(n, 'производства', 'производств', 'производств')} в отчёте "
                    f"не указано, действующее оно или завершённое — учесть их нельзя."
                ),
            )
        )
    elif flag is not None and flag[1] != bool(active.count):
        signals.append(
            Signal(
                code="enforcement_flag_mismatch",
                severity=Severity.MODERATE if flag[1] else Severity.INFO,
                title_ru="Признак банка расходится с данными",
                value={"bank_negative": flag[1], "active_count": active.count},
                source_path=flag[0],
                source_paths=[SECTION_PATH],
                explanation_ru=(
                    "По признаку банка действующие производства "
                    + ("есть" if flag[1] else "не найдены")
                    + f", а в списке отчёта активных {active.count}."
                ),
                bank_text=flag[2],
            )
        )

    if finished.count:
        signals.append(_finished_signal(finished))
    return RuleResult(signals, gaps)


def _active_signal(active: Aggregate, scale: DebtScale, flag) -> Signal:
    n = active.count
    head = plural(
        n,
        "действующее исполнительное производство",
        "действующих исполнительных производства",
        "действующих исполнительных производств",
    )
    if active.unknown_count == n:
        amount_text = ", сумма в отчёте не указана" if n == 1 else ", суммы в отчёте не указаны"
    elif active.unknown_count:
        amount_text = (
            f" на сумму не менее {rub(active.known_sum)} "
            f"(у {active.unknown_count} из {n} сумма не указана)"
        )
    else:
        amount_text = f" на сумму {rub(active.known_sum)}"
    compare = scale.compare_text()
    if scale.by_count:
        compare += f"; таких производств {n} — много само по себе"
    recent = ""
    if active.recent_count == n:
        recent = ", " + ("начато за последний год" if n == 1 else "все начаты за последний год")
    elif active.recent_count:
        recent = f", {active.recent_count} из них начаты за последний год"
    source_paths = [i.path for i in active.top()]
    if scale.net_assets:
        source_paths.append(scale.net_assets.path)
    if flag:
        source_paths.append(flag[0])
    return Signal(
        code="enforcement_active",
        severity=scale.severity,
        title_ru="Действующие исполнительные производства",
        value={
            "active_count": n,
            "known_sum": active.known_sum,
            "unknown_amount_count": active.unknown_count,
            "sum_is_lower_bound": active.unknown_count > 0,
            "recent_count": active.recent_count,
            "latest_date": active.latest,
            "net_assets": scale.net_assets.value if scale.net_assets else None,
            "net_assets_year": scale.net_assets.year if scale.net_assets else None,
            "share_of_net_assets": scale.share_value,
            "critical_by_count": scale.by_count,
        },
        source_path=SECTION_PATH,
        source_paths=source_paths[: TOP_ITEMS + 2],
        explanation_ru=f"{head}{amount_text}{compare}{recent}.",
        bank_text=flag[2] if flag else None,
    )


def _finished_signal(finished: Aggregate) -> Signal:
    n = finished.count
    head = plural(
        n,
        "завершённое исполнительное производство",
        "завершённых исполнительных производства",
        "завершённых исполнительных производств",
    )
    if finished.unknown_count == n:
        amount_text = ", сумма не указана" if n == 1 else ", суммы не указаны"
    elif finished.unknown_count:
        amount_text = (
            f" на сумму не менее {rub(finished.known_sum)} "
            f"(у {finished.unknown_count} из {n} сумма не указана)"
        )
    else:
        amount_text = f" на сумму {rub(finished.known_sum)}"
    period = ""
    if finished.earliest and finished.latest:
        period = f" за {years_range(finished.earliest.year, finished.latest.year)}"
    recent = ""
    if finished.recent_count and finished.recent_count < n:
        recent = f", {finished.recent_count} — за последний год"
    elif finished.recent_count:
        recent = ", все — за последний год" if n > 1 else ", за последний год"
    return Signal(
        code="enforcement_finished",
        severity=Severity.INFO,
        title_ru="Завершённые исполнительные производства",
        value={
            "finished_count": n,
            "known_sum": finished.known_sum,
            "unknown_amount_count": finished.unknown_count,
            "recent_count": finished.recent_count,
            "earliest": finished.earliest,
            "latest": finished.latest,
        },
        source_path=SECTION_PATH,
        source_paths=[i.path for i in finished.top()],
        explanation_ru=f"{head}{amount_text}{period}{recent}; к действующим долгам не относятся.",
    )


def _bank_flag(report: Report) -> tuple[str, bool, str | None] | None:
    """(адрес флага, негативный ли, текст банка) для кода executionProceedings."""
    risks = report.reputational_risks
    if risks is None:
        return None
    for i, f in enumerate(risks.negative or []):
        if f.code == BANK_FLAG_CODE:
            return f"report.reputationalRisks.negative[{i}].code", True, f.name
    for i, f in enumerate(risks.positive or []):
        if f.code == BANK_FLAG_CODE:
            return f"report.reputationalRisks.positive[{i}].code", False, f.name
    return None
