"""Финансовые сигналы по ``finReports`` и опора для сравнения долгов с чистыми активами.

Правило выбора года: каждое поле берём из последней строки отчётности, где оно
заполнено, в пределах двух последних строк (2025, иначе 2024) — так считает свои
флаги сам банк; у половины компаний в свежей строке нет прибыли. Строки-заглушки
из одних нулей (годы до регистрации и несданные годы) не смотрим — их узнаём по
содержимому, а не по году. Год всегда в сигнале.

Сигналы: убыток (moderate) · отрицательный капитал (critical — обязательства
превышают активы; светофор этого не учитывает) · низкая текущая ликвидность:
оборотные / краткосрочные < 0,5 — moderate, 0,5–1 — info · нулевая выручка
(moderate) · отчётность устарела (info, с учётом срока сдачи до 31 марта) ·
убытки несколько лет (info).

Пробелы: у ИП отчётности в отчёте не бывает (критерий не применим); у компании
моложе года срок сдачи не наступил; у компании старше года без отчётности —
пробел, при котором рекомендация не лучше «стоит проверить»; нет строки прибыли,
выручки, капитала, итогов для ликвидности — пробел по полю, не «убытка нет».
Ноль — это значение, а не пропуск: нулевые краткосрочные обязательства — не пробел.

``debt_scale`` — соразмерность долга чистым активам (капитал и резервы последнего
года), общая для производств и исков: по совету кейсодателя (чат 3.09).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from contractor_agent.data.model import FinReport, Report
from contractor_agent.signals.model import Gap, GapReason, RuleResult, Severity, Signal
from contractor_agent.signals.text import ratio, rub, share, years_list, years_range

LOOKBACK_ROWS = 2
DEBT_CRITICAL_SHARE = Decimal("0.5")  # долг ≥ половины чистых активов — критично
DEBT_INFO_SHARE = Decimal("0.01")  # долг < 1 % чистых активов при паре производств — для сведения
DEBT_INFO_MAX_COUNT = 5
LIQUIDITY_MODERATE_BELOW = Decimal("0.5")
LIQUIDITY_INFO_BELOW = Decimal("1")
SOLE_PROPRIETOR_INN_LENGTH = 12
STATEMENTS_DUE_AFTER_YEARS = 1
STATEMENTS_DEADLINE_MONTH = 4  # годовая отчётность сдаётся до 31 марта

SECTION_PATH = "report.finReports"
EQUITY = "чистые активы (капитал и резервы)"
EQUITY_GEN = "чистых активов (капитал и резервы)"


@dataclass(frozen=True)
class Row:
    index: int
    year: int
    data: FinReport

    @property
    def path(self) -> str:
        return f"{SECTION_PATH}[{self.index}]"


@dataclass(frozen=True)
class Picked:
    """Значение поля с годом и адресом — то, что уходит в сигнал."""

    value: int
    year: int
    path: str


def is_sole_proprietor(report: Report) -> bool:
    return len(report.base_info.inn) == SOLE_PROPRIETOR_INN_LENGTH


def months_between(start: date, end: date) -> int:
    return max(0, (end.year - start.year) * 12 + end.month - start.month - (end.day < start.day))


def company_age_years(report: Report) -> int | None:
    """Возраст в полных годах: поле банка, иначе по дате регистрации."""
    info = report.base_info.registration_info
    if info is None:
        return None
    if info.years_from_registration is not None:
        return info.years_from_registration
    if info.registration_date is None:
        return None
    return months_between(info.registration_date, report.report_date) // 12


def expected_latest_year(report_date: date) -> int:
    """Последний год, за который отчётность уже должна быть сдана к дате отчёта."""
    if report_date.month >= STATEMENTS_DEADLINE_MONTH:
        return report_date.year - 1
    return report_date.year - 2


def _leaves(f: FinReport) -> list[int | None]:
    common, assets, liabilities = f.common, f.assets, f.liabilities
    current = assets.current_assets if assets else None
    uncurrent = assets.uncurrent_assets if assets else None
    short = liabilities.short_term_liabilities if liabilities else None
    long = liabilities.long_term_duties if liabilities else None
    return [
        common.proceeds if common else None,
        common.profit if common else None,
        assets.total_assets if assets else None,
        current.total if current else None,
        uncurrent.total if uncurrent else None,
        liabilities.capitals if liabilities else None,
        liabilities.total_liabilities if liabilities else None,
        short.total if short else None,
        long.total if long else None,
    ]


def is_placeholder(f: FinReport) -> bool:
    """Строка из одних нулей и пропусков — заглушка, не отчётность."""
    return all(v is None or v == 0 for v in _leaves(f))


def real_rows(report: Report) -> list[Row]:
    """Строки отчётности по убыванию года, без заглушек (до регистрации и из нулей)."""
    info = report.base_info.registration_info
    registered = info.registration_date.year if info and info.registration_date else None
    rows = []
    for index, data in enumerate(report.fin_reports or []):
        year = data.common.year if data.common else None
        if year is None or (registered is not None and year < registered):
            continue
        if is_placeholder(data):
            continue
        rows.append(Row(index, year, data))
    return sorted(rows, key=lambda r: -r.year)


def pick(rows: list[Row], getter: Callable[[FinReport], int | None], subpath: str) -> Picked | None:
    for row in rows[:LOOKBACK_ROWS]:
        value = getter(row.data)
        if value is not None:
            return Picked(value, row.year, f"{row.path}.{subpath}")
    return None


def _profit(f: FinReport) -> int | None:
    return f.common.profit if f.common else None


def _proceeds(f: FinReport) -> int | None:
    return f.common.proceeds if f.common else None


def _capitals(f: FinReport) -> int | None:
    return f.liabilities.capitals if f.liabilities else None


def _total_assets(f: FinReport) -> int | None:
    return f.assets.total_assets if f.assets else None


def _current_assets(f: FinReport) -> int | None:
    return f.assets.current_assets.total if f.assets and f.assets.current_assets else None


def _short_term(f: FinReport) -> int | None:
    liabilities = f.liabilities
    if liabilities and liabilities.short_term_liabilities:
        return liabilities.short_term_liabilities.total
    return None


def net_assets(report: Report) -> Picked | None:
    """Капитал и резервы последнего года — «чистые активы» для сравнения с долгами."""
    return pick(real_rows(report), _capitals, "liabilities.capitals")


@dataclass(frozen=True)
class DebtScale:
    """Соразмерность долга чистым активам — по совету кейсодателя (чат 3.09)."""

    severity: Severity
    net_assets: Picked | None
    ratio: Decimal | None  # известная сумма / чистые активы; None — сравнить не с чем
    by_count: bool  # критично по числу, независимо от суммы
    reason: str  # почему сравнить нельзя; пусто, если ratio посчитан

    @property
    def comparable(self) -> bool:
        return self.ratio is not None

    def compare_text(self) -> str:
        """Хвост объяснения: сравнение с чистыми активами или почему его нет."""
        if self.comparable:
            net = self.net_assets
            return (
                f"; это {share(self.ratio)} {EQUITY_GEN} на конец {net.year} года "
                f"({rub(net.value)})"
            )
        return f"; {self.reason}" if self.reason else ""

    @property
    def share_value(self) -> float | None:
        return float(self.ratio) if self.ratio is not None else None


def debt_scale(
    report: Report,
    known_sum: Decimal,
    unknown_count: int,
    count: int,
    critical_count: int,
) -> DebtScale:
    """Тяжесть долга: ≥ половины чистых активов или ≥ critical_count штук — критично;
    меньше 1 % при нескольких штуках с известными суммами — для сведения; иначе проверить.
    Без отчётности, без строки капитала, при нулевом капитале или без сумм сравнить
    не с чем — проверить."""
    net = net_assets(report)
    ratio_value: Decimal | None = None
    if net is None:
        if real_rows(report):
            reason = (
                "в отчётности нет строки «капитал и резервы», соразмерность долга оценить нельзя"
            )
        else:
            reason = "отчётности в отчёте нет, соразмерность долга оценить нельзя"
        severity = Severity.MODERATE
    elif net.value < 0:
        reason = f"{EQUITY} на конец {net.year} года отрицательные ({rub(net.value)})"
        severity = Severity.CRITICAL
    elif net.value == 0:
        reason = f"{EQUITY} на конец {net.year} года равны нулю, соразмерность оценить нельзя"
        severity = Severity.MODERATE
    elif known_sum <= 0:
        reason = "без суммы соразмерность оценить нельзя"
        severity = Severity.MODERATE
    else:
        reason = ""
        ratio_value = known_sum / Decimal(net.value)
        if ratio_value >= DEBT_CRITICAL_SHARE:
            severity = Severity.CRITICAL
        elif ratio_value < DEBT_INFO_SHARE and unknown_count == 0 and count < DEBT_INFO_MAX_COUNT:
            severity = Severity.INFO
        else:
            severity = Severity.MODERATE
    by_count = count >= critical_count
    if by_count:
        severity = Severity.CRITICAL
    return DebtScale(severity, net, ratio_value, by_count, reason)


def run(report: Report) -> RuleResult:
    signals: list[Signal] = []
    gaps: list[Gap] = []

    state = report.section_state("finReports")
    rows = real_rows(report) if state == "present" else []
    if not rows:
        gaps.append(_no_statements_gap(report, state))
        return RuleResult(signals, gaps)

    latest = rows[0]
    expected = expected_latest_year(report.report_date)
    if latest.year < expected:
        signals.append(
            Signal(
                code="fin_stale",
                severity=Severity.INFO,
                title_ru="Отчётность устарела",
                value={"latest_year": latest.year, "expected_year": expected},
                source_path=f"{latest.path}.common.year",
                explanation_ru=(
                    f"Последняя отчётность в отчёте — за {latest.year} год; отчётность за "
                    f"{expected} год должна была быть сдана к 31 марта {expected + 1} года."
                ),
                year=latest.year,
            )
        )

    profit = pick(rows, _profit, "common.profit")
    proceeds = pick(rows, _proceeds, "common.proceeds")
    if profit is None:
        gaps.append(
            _field_gap(rows, "прибыль или убыток", "common.profit", "строки «прибыль (убыток)»")
        )
    elif profit.value < 0:
        revenue = ""
        if proceeds and proceeds.year == profit.year:
            revenue = f" при выручке {rub(proceeds.value)}"
        signals.append(
            Signal(
                code="fin_loss",
                severity=Severity.MODERATE,
                title_ru=f"Убыток за {profit.year} год",
                value={"profit": profit.value, "year": profit.year},
                source_path=profit.path,
                source_paths=[proceeds.path] if revenue else [],
                explanation_ru=(
                    f"По отчётности за {profit.year} год — убыток {rub(-profit.value)}{revenue}."
                ),
                year=profit.year,
            )
        )

    loss_rows = [r for r in rows if (_profit(r.data) or 0) < 0]
    if len(loss_rows) >= 2:
        signals.append(
            Signal(
                code="fin_loss_streak",
                severity=Severity.INFO,
                title_ru="Убытки несколько лет",
                value={
                    "loss_years": sorted(r.year for r in loss_rows),
                    "years_in_report": len(rows),
                },
                source_path=SECTION_PATH,
                source_paths=[f"{r.path}.common.profit" for r in loss_rows],
                explanation_ru=(
                    f"Убыток в {len(loss_rows)} из {len(rows)} лет отчётности: "
                    f"{years_list([r.year for r in loss_rows])}."
                ),
            )
        )

    if proceeds is None:
        gaps.append(_field_gap(rows, "выручка", "common.proceeds", "строки «выручка»"))
    elif proceeds.value == 0:
        signals.append(
            Signal(
                code="fin_zero_revenue",
                severity=Severity.MODERATE,
                title_ru=f"Нет выручки за {proceeds.year} год",
                value={"proceeds": 0, "year": proceeds.year},
                source_path=proceeds.path,
                explanation_ru=f"Выручка за {proceeds.year} год по отчётности — 0 ₽.",
                year=proceeds.year,
            )
        )

    equity = pick(rows, _capitals, "liabilities.capitals")
    assets = pick(rows, _total_assets, "assets.totalAssets")
    if equity is None:
        gaps.append(
            _field_gap(
                rows, "капитал и резервы", "liabilities.capitals", "строки «капитал и резервы»"
            )
        )
    elif equity.value < 0:
        total = (
            f", всего активов {rub(assets.value)}" if assets and assets.year == equity.year else ""
        )
        signals.append(
            Signal(
                code="fin_negative_equity",
                severity=Severity.CRITICAL,
                title_ru="Отрицательный капитал",
                value={"capitals": equity.value, "year": equity.year},
                source_path=equity.path,
                source_paths=[assets.path] if total else [],
                explanation_ru=(
                    f"{EQUITY.capitalize()} на конец {equity.year} года отрицательные: "
                    f"{rub(equity.value)} — обязательства превышают активы{total}."
                ),
                year=equity.year,
            )
        )

    liquidity = _liquidity(rows)
    if isinstance(liquidity, Signal):
        signals.append(liquidity)
    elif liquidity is not None:
        gaps.append(liquidity)

    return RuleResult(signals, gaps)


def _liquidity(rows: list[Row]) -> Signal | Gap | None:
    had_totals = False
    for row in rows[:LOOKBACK_ROWS]:
        current, short = _current_assets(row.data), _short_term(row.data)
        if current is None or short is None:
            continue
        had_totals = True
        if short <= 0:
            if current <= 0:
                continue  # пустая строка баланса — смотрим дальше
            return None  # краткосрочных обязательств нет — ликвидность не проблема
        value = Decimal(current) / Decimal(short)
        if value >= LIQUIDITY_INFO_BELOW:
            return None
        severity = Severity.MODERATE if value < LIQUIDITY_MODERATE_BELOW else Severity.INFO
        return Signal(
            code="fin_low_liquidity",
            severity=severity,
            title_ru="Низкая текущая ликвидность",
            value={
                "ratio": value.quantize(Decimal("0.01")),
                "current_assets": current,
                "short_term_liabilities": short,
                "year": row.year,
            },
            # коэффициент — производный показатель: адрес года, операнды рядом
            source_path=row.path,
            source_paths=[
                f"{row.path}.assets.currentAssets.total",
                f"{row.path}.liabilities.shortTermLiabilities.total",
            ],
            explanation_ru=(
                f"Оборотные активы {rub(current)} против краткосрочных обязательств "
                f"{rub(short)} на конец {row.year} года: покрытие {ratio(value)}."
            ),
            year=row.year,
        )
    if had_totals:
        return None
    latest = rows[0]
    return Gap(
        criterion="текущая ликвидность",
        reason=GapReason.FIELD_ABSENT,
        source_path=f"{latest.path}.assets",
        text_ru=(
            f"В отчётности за {latest.year} год нет итога оборотных активов или краткосрочных "
            f"обязательств — текущую ликвидность по отчёту не рассчитать."
        ),
    )


def _field_gap(rows: list[Row], criterion: str, subpath: str, what: str) -> Gap:
    considered = [r.year for r in rows[:LOOKBACK_ROWS]]
    years = years_range(min(considered), max(considered))
    return Gap(
        criterion=criterion,
        reason=GapReason.FIELD_ABSENT,
        source_path=f"{rows[0].path}.{subpath}",
        text_ru=f"В отчётности за {years} нет {what} — оценить по этому критерию нельзя.",
    )


def _no_statements_gap(report: Report, state: str) -> Gap:
    reason = GapReason.SECTION_ABSENT if state == "absent" else GapReason.SECTION_EMPTY
    if is_sole_proprietor(report):
        return Gap(
            criterion="финансовая отчётность",
            reason=GapReason.NOT_APPLICABLE,
            source_path=SECTION_PATH,
            text_ru=(
                "Контрагент — индивидуальный предприниматель: бухгалтерской отчётности в отчёте "
                "нет, оценить выручку, прибыль и устойчивость по этому критерию нельзя."
            ),
            ask_ru="Запросите у контрагента налоговую декларацию или выписку по счёту за год.",
        )
    age = company_age_years(report)
    if age is not None and age < STATEMENTS_DUE_AFTER_YEARS:
        return Gap(
            criterion="финансовая отчётность",
            reason=reason,
            source_path=SECTION_PATH,
            text_ru=(
                "Компания зарегистрирована меньше года назад — срок сдачи первой бухгалтерской "
                "отчётности ещё не наступил; оценить финансы по отчёту нельзя."
            ),
            ask_ru="Запросите промежуточную отчётность или выписку по счёту.",
        )
    text = (
        "В отчёте нет финансовой отчётности: выручку, прибыль, капитал "
        "и ликвидность оценить нельзя."
    )
    return Gap(
        criterion="финансовая отчётность",
        reason=reason,
        source_path=SECTION_PATH,
        text_ru=text,
        ask_ru="Запросите у контрагента бухгалтерскую отчётность за последний год.",
        floor_check=True,
    )
