"""Арбитраж: открытые и завершённые дела, где компания — ответчик.

В отчёте нет списка дел — только агрегат по ролям и статусам
(``arbitrationByStatus``: истец / ответчик × закрытые / открытые / обжалованные)
и разбивка по годам (``arbitrationCases``, окно 2023–2026). Риск — когда с
компанией судятся: открытые и обжалованные дела как ответчик — претензии,
которые ещё могут стать долгом; тяжесть — по соразмерности чистым активам
(``finance.debt_scale``). Закрытые дела как ответчик — история, для сведения;
«проверить» только если их сумма не меньше чистых активов.

Ловушки из данных: у одной компании роль ответчика видна только в годовой
разбивке (агрегат без ролей) — есть запасной путь по годам; «агрегат больше
суммы по годам» у 55 компаний — окно 2023–2026, не ошибка, расхождением считаем
только реальное: дела без ролей внутри агрегата и агрегат меньше разбивки.
Нулевая сумма считается неизвестной — ноль не пишем никогда.

Пустой агрегат — не пробел: у банка есть положительный признак «дела не найдены»,
а «не найдено ≠ нет» — формулировка на стороне промпта (кейсодатель, QA 10:27).
Дела, где компания сама истец, риском не считаем — их покажет инструмент.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from pydantic.alias_generators import to_camel

from contractor_agent.data.model import ArbitrationByStatus, Report
from contractor_agent.signals.finance import DebtScale, debt_scale
from contractor_agent.signals.model import Gap, GapReason, RuleResult, Severity, Signal
from contractor_agent.signals.text import plural, rub, years_range

SECTION_PATH = "report.arbitrationByStatus"
CASES_PATH = "report.arbitrationCases"
DEFENDANT_PATH = f"{SECTION_PATH}.defandantArbitration"
PLAINTIFF_PATH = f"{SECTION_PATH}.plaintiffArbitration"
CRITICAL_OPEN_CASES = 10
FINISHED_MODERATE_SHARE = Decimal("1")  # закрытые дела на сумму ≥ чистых активов — проверить
BANK_FLAG_CODE = "arbitrationDefendant"


@dataclass(frozen=True)
class Bucket:
    """Одна ячейка агрегата: число дел, сумма (может отсутствовать), адреса."""

    count: int
    amount: int | None
    count_path: str
    amount_path: str

    @property
    def known_amount(self) -> int | None:
        return self.amount if self.amount else None  # ноль — как «не указана»


def _bucket(node, count_attr: str, amount_attr: str, path: str) -> Bucket | None:
    if node is None:
        return None
    count = getattr(node, count_attr)
    if not count:
        return None
    return Bucket(
        count,
        getattr(node, amount_attr),
        f"{path}.{to_camel(count_attr)}",  # адрес — в терминах отчёта: dfCount, не df_count
        f"{path}.{to_camel(amount_attr)}",
    )


@dataclass(frozen=True)
class Roles:
    finished: Bucket | None
    pending: Bucket | None
    appealed: Bucket | None

    @property
    def total(self) -> int:
        return sum(b.count for b in (self.finished, self.pending, self.appealed) if b)

    @property
    def any_present(self) -> bool:
        return any(b is not None for b in (self.finished, self.pending, self.appealed))


def defendant_roles(agg: ArbitrationByStatus | None) -> Roles:
    d = agg.defandant_arbitration if agg else None
    return Roles(
        finished=_bucket(
            d.defandant_arbitration_finished if d else None,
            "df_count",
            "df_amount",
            f"{DEFENDANT_PATH}.defandantArbitrationFinished",
        ),
        pending=_bucket(
            d.defandant_arbitration_pending if d else None,
            "dp_count",
            "dp_amount",
            f"{DEFENDANT_PATH}.defandantArbitrationPending",
        ),
        appealed=_bucket(
            d.defandant_arbitration_appealed if d else None,
            "da_count",
            "da_amount",
            f"{DEFENDANT_PATH}.defandantArbitrationAppealed",
        ),
    )


def plaintiff_roles(agg: ArbitrationByStatus | None) -> Roles:
    p = agg.plaintiff_arbitration if agg else None
    return Roles(
        finished=_bucket(
            p.plaintiff_arbitration_finished if p else None,
            "pf_count",
            "pf_amount",
            f"{PLAINTIFF_PATH}.plaintiffArbitrationFinished",
        ),
        pending=_bucket(
            p.plaintiff_arbitration_pending if p else None,
            "pp_count",
            "pp_amount",
            f"{PLAINTIFF_PATH}.plaintiffArbitrationPending",
        ),
        appealed=_bucket(
            p.plaintiff_arbitration_appealed if p else None,
            "pa_count",
            "pa_amount",
            f"{PLAINTIFF_PATH}.plaintiffArbitrationAppealed",
        ),
    )


def _sum(buckets: list[Bucket]) -> tuple[Decimal, int]:
    """(известная сумма, число дел в ячейках без суммы или с нулевой)."""
    known = sum((Decimal(b.known_amount) for b in buckets if b.known_amount), Decimal(0))
    unknown = sum(b.count for b in buckets if b.known_amount is None)
    return known, unknown


def run(report: Report) -> RuleResult:
    signals: list[Signal] = []
    gaps: list[Gap] = []
    agg = report.arbitration_by_status
    cases = report.arbitration_cases
    if agg is None and not cases:
        gaps.append(
            Gap(
                criterion="арбитражные дела",
                reason=GapReason.SECTION_ABSENT,
                source_path=SECTION_PATH,
                text_ru=(
                    "В отчёте нет раздела об арбитражных делах — судятся ли с компанией, "
                    "по отчёту сказать нельзя."
                ),
                ask_ru="Проверьте контрагента в картотеке арбитражных дел по ИНН.",
            )
        )
        return RuleResult(signals, gaps)

    roles = defendant_roles(agg)
    flag = _bank_flag(report)
    open_buckets = [b for b in (roles.pending, roles.appealed) if b]
    if open_buckets:
        signals.append(_open_signal(report, roles, open_buckets, flag))
    if roles.finished:
        signals.append(_finished_signal(report, roles.finished, flag))

    if roles.total == 0:
        by_years = _defendant_by_years(report, plaintiff_roles(agg).any_present)
        if by_years is not None:
            signals.append(by_years)

    signals.extend(_mismatch_signals(report, roles))
    return RuleResult(signals, gaps)


def _open_signal(report: Report, roles: Roles, buckets: list[Bucket], flag) -> Signal:
    count = sum(b.count for b in buckets)
    known, unknown = _sum(buckets)
    scale = debt_scale(report, known, unknown, count, CRITICAL_OPEN_CASES)
    head = plural(
        count, "открытое арбитражное дело", "открытых арбитражных дела", "открытых арбитражных дел"
    )
    detail = ""
    if roles.pending and roles.appealed:
        detail = f" ({roles.pending.count} в производстве, {roles.appealed.count} обжалуется)"
    if unknown == count:
        amount_text = "сумма в отчёте не указана"
    elif unknown:
        amount_text = f"на сумму не менее {rub(known)} (по {unknown} из {count} сумма не указана)"
    else:
        amount_text = f"на сумму {rub(known)}"
    compare = scale.compare_text()
    if scale.by_count:
        compare += f"; таких дел {count} — много само по себе"
    source_paths = []
    for b in buckets:
        source_paths.append(b.count_path)
        if b.known_amount:
            source_paths.append(b.amount_path)
    if scale.net_assets:
        source_paths.append(scale.net_assets.path)
    if flag:
        source_paths.append(flag[0])
    return Signal(
        code="arbitration_defendant_open",
        severity=scale.severity,
        title_ru="Открытые иски к компании",
        value={
            "open_count": count,
            "pending_count": roles.pending.count if roles.pending else 0,
            "appealed_count": roles.appealed.count if roles.appealed else 0,
            "known_sum": known,
            "unknown_amount_count": unknown,
            "net_assets": scale.net_assets.value if scale.net_assets else None,
            "net_assets_year": scale.net_assets.year if scale.net_assets else None,
            "share_of_net_assets": scale.share_value,
            "critical_by_count": scale.by_count,
        },
        source_path=buckets[0].count_path,
        source_paths=source_paths,
        explanation_ru=f"{head}{detail}, где компания — ответчик, {amount_text}{compare}.",
        bank_text=flag[2] if flag else None,
    )


def _finished_signal(report: Report, finished: Bucket, flag) -> Signal:
    """Закрытые дела — история, для сведения; «проверить» только если их сумма
    не меньше чистых активов: с компании уже взыскивали больше, чем она стоит."""
    known, unknown = _sum([finished])
    scale: DebtScale = debt_scale(report, known, unknown, finished.count, CRITICAL_OPEN_CASES)
    heavy = scale.comparable and scale.ratio >= FINISHED_MODERATE_SHARE
    severity = Severity.MODERATE if heavy else Severity.INFO
    head = plural(
        finished.count,
        "завершённое арбитражное дело",
        "завершённых арбитражных дела",
        "завершённых арбитражных дел",
    )
    amount_text = f"на сумму {rub(known)}" if finished.known_amount else "сумма в отчёте не указана"
    compare = scale.compare_text() if heavy else ""
    bank_note = "; признак банка по этому пункту положительный" if flag and not flag[1] else ""
    source_paths = [finished.count_path]
    if finished.known_amount:
        source_paths.append(finished.amount_path)
    if heavy and scale.net_assets:
        source_paths.append(scale.net_assets.path)
    if flag:
        source_paths.append(flag[0])
    return Signal(
        code="arbitration_defendant_finished",
        severity=severity,
        title_ru="Завершённые иски к компании",
        value={
            "finished_count": finished.count,
            "known_sum": known,
            "net_assets": scale.net_assets.value if scale.net_assets else None,
            "share_of_net_assets": scale.share_value,
        },
        source_path=finished.count_path,
        source_paths=source_paths,
        explanation_ru=(
            f"{head}, где компания была ответчиком, {amount_text}{compare}{bank_note}; "
            f"к текущим претензиям не относятся."
        ),
        bank_text=flag[2] if flag else None,
    )


def _defendant_by_years(report: Report, plaintiff_present: bool) -> Signal | None:
    rows = [
        (i, c) for i, c in enumerate(report.arbitration_cases or []) if (c.defendant_count or 0) > 0
    ]
    if not rows:
        return None
    count = sum(c.defendant_count for _, c in rows)
    known = sum((Decimal(c.defendant_amount) for _, c in rows if c.defendant_amount), Decimal(0))
    years = sorted({c.year for _, c in rows if c.year})
    period = f" за {years_range(years[0], years[-1])}" if years else ""
    amount_text = f" на сумму {rub(known)}" if known else ""
    tail = (
        "в сводке по статусам дел с ролью ответчика нет"
        if plaintiff_present
        else "в сводке по статусам роли не указаны"
    )
    return Signal(
        code="arbitration_defendant_by_years",
        severity=Severity.MODERATE,
        title_ru="Иски к компании в разбивке по годам",
        value={"defendant_count": count, "known_sum": known, "years": years},
        source_path=CASES_PATH,
        source_paths=[f"{CASES_PATH}[{i}].defendantCount" for i, _ in rows],
        explanation_ru=(
            f"По разбивке{period} у компании {plural(count, 'дело', 'дела', 'дел')} "
            f"как ответчик{amount_text}; {tail}."
        ),
    )


def _mismatch_signals(report: Report, roles: Roles) -> list[Signal]:
    agg = report.arbitration_by_status
    cases = report.arbitration_cases or []
    years_total = sum((c.plaintiff_count or 0) + (c.defendant_count or 0) for c in cases)
    out: list[Signal] = []
    if agg is None or not agg.common_count:
        if years_total and roles.total == 0 and not plaintiff_roles(agg).any_present:
            out.append(
                Signal(
                    code="arbitration_mismatch_no_summary",
                    severity=Severity.INFO,
                    title_ru="Сводка по арбитражу неполная",
                    value={"years_total": years_total, "aggregate_total": None},
                    source_path=CASES_PATH,
                    source_paths=[SECTION_PATH],
                    explanation_ru=(
                        f"В сводке по статусам дел нет, а в разбивке по годам — "
                        f"{plural(years_total, 'дело', 'дела', 'дел')}; данные есть только "
                        f"в разбивке."
                    ),
                )
            )
        return out
    nested = roles.total + plaintiff_roles(agg).total
    if agg.common_count > nested:
        out.append(
            Signal(
                code="arbitration_mismatch_roles",
                severity=Severity.INFO,
                title_ru="Часть дел без деталей",
                value={"common_count": agg.common_count, "with_roles": nested},
                source_path=f"{SECTION_PATH}.commonCount",
                source_paths=[DEFENDANT_PATH, PLAINTIFF_PATH],
                explanation_ru=(
                    f"Всего дел в отчёте {agg.common_count}, по ролям и статусам разнесены "
                    f"{nested}: {plural(agg.common_count - nested, 'дело', 'дела', 'дел')} "
                    f"без деталей."
                ),
            )
        )
    if years_total > agg.common_count:
        out.append(
            Signal(
                code="arbitration_mismatch_years",
                severity=Severity.INFO,
                title_ru="Сводка по арбитражу расходится с разбивкой",
                value={"common_count": agg.common_count, "years_total": years_total},
                source_path=f"{SECTION_PATH}.commonCount",
                source_paths=[CASES_PATH],
                explanation_ru=(
                    f"В сводке {plural(agg.common_count, 'дело', 'дела', 'дел')}, в разбивке "
                    f"по годам {plural(years_total, 'дело', 'дела', 'дел')} — данные расходятся."
                ),
            )
        )
    return out


def _bank_flag(report: Report) -> tuple[str, bool, str | None] | None:
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
