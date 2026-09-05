"""Четыре доли наверху отчёта плюс стабильность — как в `CRITERIA.md` §5.

Ошибки разной цены — разные счётчики: пропуск критичного факта стоит денег,
ложная тревога — времени; они не складываются в одно число.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from evals.checks import visible_invalid_citations
from evals.runner import RunRecord

JUDGE_PASS = 4


@dataclass
class Metrics:
    model: str
    total: int = 0
    errors: int = 0
    grounded_share: float | None = None  # ответы, подтверждаемые данными (answer + infer + card)
    refusal_share: float | None = None  # корректные отказы (refuse)
    invented_share: float | None = None  # выдуманные факты по судье (все типы)
    bad_citation_share: float | None = None  # ответы с неверной ссылкой на поле (все типы)
    missed_critical_share: float | None = None  # пропущенные критичные факты (card)
    judge_mean: float | None = None
    stability: float | None = None  # доля вопросов с одинаковым исходом во всех повторах
    by_type: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    mean_duration_s: float | None = None


def _share(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def compute_metrics(records: list[RunRecord]) -> Metrics:
    if not records:
        return Metrics(model="?")
    m = Metrics(model=records[0].model, total=len(records))
    ok = [r for r in records if r.error is None]
    m.errors = len(records) - len(ok)

    substantive = [r for r in records if r.type in ("answer", "infer", "card")]
    grounded = [
        r
        for r in substantive
        if not r.error and r.checks_passed and (r.judge is None or r.judge.score >= JUDGE_PASS)
    ]
    m.grounded_share = _share(len(grounded), len(substantive))

    refuse = [r for r in records if r.type == "refuse"]
    m.refusal_share = _share(
        sum(
            1
            for r in refuse
            if not r.error
            and r.checks_passed
            and r.check_notes.get("refused")
            and (r.judge is None or r.judge.score >= JUDGE_PASS)
        ),
        len(refuse),
    )

    invented = [  # выдуманный факт по мнению судьи, который видит данные инструментов
        r
        for r in ok
        if r.judge and ("invented_fact" in r.judge.failures or "false_alarm" in r.judge.deductions)
    ]
    m.invented_share = _share(len(invented), sum(r.judge is not None for r in ok))
    bad_cit = [
        r for r in ok if r.answer and visible_invalid_citations(r.answer)
    ]  # ссылка не на то поле
    m.bad_citation_share = _share(len(bad_cit), len(ok))

    cards = [r for r in ok if r.type == "card"]
    missed = [
        r
        for r in cards
        if r.check_notes.get("missed_critical")
        or (r.judge and "missed_critical" in r.judge.failures)
    ]
    m.missed_critical_share = _share(len(missed), len(cards))

    scores = [r.judge.score for r in ok if r.judge]
    m.judge_mean = round(mean(scores), 2) if scores else None
    durations = [r.agent_duration_s for r in ok if r.agent_duration_s is not None]
    m.mean_duration_s = round(mean(durations), 1) if durations else None

    outcomes: dict[str, set[tuple]] = defaultdict(set)
    for r in records:
        verdict = r.answer.card.verdict.value if r.answer and r.answer.card else None
        outcomes[r.question_id].add((bool(r.error), r.checks_passed, verdict))
    repeated = {
        q: o for q, o in outcomes.items() if sum(1 for r in records if r.question_id == q) > 1
    }
    if repeated:
        m.stability = _share(sum(1 for o in repeated.values() if len(o) == 1), len(repeated))

    for kind in ("card", "answer", "refuse", "infer"):
        rows = [r for r in records if r.type == kind]
        if rows:
            m.by_type[kind] = {
                "n": len(rows),
                "checks_pass": _share(
                    sum(1 for r in rows if not r.error and r.checks_passed), len(rows)
                )
                or 0,
                "judge_mean": round(mean(r.judge.score for r in rows if r.judge), 2)
                if any(r.judge for r in rows)
                else None,
            }
    return m
