"""Сборка: все правила → ``SignalSet`` с рекомендацией.

Рекомендация считается кодом, детерминированно (кейсодатель: «рекомендация,
а не вердикт, всегда с фактами»; три исхода — его формулировка):

* терминальный сигнал (банкротство, исключение из ЕГРЮЛ, компания закрыта,
  конкурсный управляющий, флаг ликвидации) → факты, требующие особого внимания;
* любой критичный сигнал → «в отчёте есть факты, требующие особого внимания»
  (`docs/CRITERIA.md` §2: итоговый цвет в сервисах проверки — не сумма баллов,
  а наличие критического фактора);
* иначе хотя бы один сигнал «проверить» → «нужна дополнительная проверка»,
  0 → «существенных факторов риска в отчёте не выявлено»;
  баллы (3 за критичный, 1 за «проверить») остаются
  как мера для сравнения компаний, не для порога;
* нижняя граница: светофор банка HIGH → не лучше «стоит проверить» (метка
  учитывает данные, которых в отчёте нет); пробел с ``floor_check`` (компания
  старше года без отчётности) → не лучше «стоит проверить».

ЗСК в рекомендации не участвует: банк раскрывает клиенту только зелёный и серый.
Каждый адрес каждого сигнала и пробела проверяется на резолвинг — ошибка
правила всплывает здесь, а не в валидаторе цитат у клиента.
"""

from __future__ import annotations

from collections.abc import Callable

from contractor_agent.data.model import Report
from contractor_agent.signals import arbitration, enforcement, finance, flags, registry
from contractor_agent.signals.model import (
    Gap,
    GapReason,
    RuleResult,
    Severity,
    Signal,
    SignalSet,
    Verdict,
    check_paths,
)

Rule = Callable[[Report], RuleResult]

RULES: tuple[Rule, ...] = (registry.run, finance.run, enforcement.run, arbitration.run, flags.run)

SCORE_CRITICAL = 3
SCORE_MODERATE = 1
LABEL_FLOOR_LEVELS = frozenset({"HIGH"})


def compute(report: Report, rules: tuple[Rule, ...] = RULES) -> SignalSet:
    signals: list[Signal] = []
    gaps: list[Gap] = []
    for rule in rules:
        result = rule(report)
        signals.extend(result.signals)
        gaps.extend(result.gaps)
    if (report.base_info.risk_level or "").upper() == "UNKNOWN":
        gaps.append(
            Gap(
                criterion="светофор банка",
                reason=GapReason.FIELD_ABSENT,
                source_path="report.baseInfo.riskLevel",
                text_ru="Светофор банка не рассчитан (серый) — оценки банка по компании нет.",
            )
        )
    for item in (*signals, *gaps):
        check_paths(report, item)
    terminal = any(s.terminal for s in signals)
    score = sum(
        SCORE_CRITICAL if s.severity is Severity.CRITICAL else SCORE_MODERATE
        for s in signals
        if s.severity is not Severity.INFO
    )
    verdict = decide(
        terminal=terminal,
        critical=any(s.severity is Severity.CRITICAL for s in signals),
        moderate=any(s.severity is Severity.MODERATE for s in signals),
        risk_level=report.base_info.risk_level,
        floor_check=any(g.floor_check for g in gaps),
    )
    return SignalSet(
        inn=report.inn,
        report_date=report.report_date,
        signals=signals,
        gaps=gaps,
        verdict=verdict,
        score=score,
        terminal=terminal,
    )


def decide(
    *, terminal: bool, critical: bool, moderate: bool, risk_level: str, floor_check: bool
) -> Verdict:
    if terminal or critical:
        return Verdict.NOT_RECOMMENDED
    if moderate or floor_check or (risk_level or "").upper() in LABEL_FLOOR_LEVELS:
        return Verdict.CHECK
    return Verdict.OK
