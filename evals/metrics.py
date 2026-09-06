"""Четыре доли наверху отчёта плюс стабильность — как в `CRITERIA.md` §5.

Ошибки разной цены — разные счётчики: пропуск критичного факта стоит денег,
ложная тревога — времени; они не складываются в одно число.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

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
    by_type: dict[str, dict[str, float | int]] = field(default_factory=dict)
    by_category: dict[str, dict[str, float | int]] = field(default_factory=dict)
    mean_duration_s: float | None = None
    prompt_version: str = ""
    guard_share: float | None = None  # посторонний ввод отбит без инструментов (guard)
    comparison_share: float | None = None  # у каждой компании свой правильный вывод (comparison)


def _share(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def compute_metrics(records: list[RunRecord]) -> Metrics:
    if not records:
        return Metrics(model="?")
    m = Metrics(model=records[0].model, total=len(records))
    ok = [r for r in records if r.error is None]
    m.errors = len(records) - len(ok)

    substantive = [  # ответ по существу: и прямой, и последний в диалоге, и сравнение
        r for r in ok if (r.effective_type or r.type) in ("answer", "infer", "card", "comparison")
    ]
    grounded = [
        r
        for r in substantive
        if r.checks_passed and (r.judge is None or r.judge.score >= JUDGE_PASS)
    ]
    m.grounded_share = _share(len(grounded), len(substantive))

    refuse = [r for r in ok if (r.effective_type or r.type) == "refuse"]
    m.refusal_share = _share(sum(1 for r in refuse if r.check_notes.get("refused")), len(refuse))

    invented = [  # выдуманный факт по мнению судьи, который видит данные инструментов
        r
        for r in ok
        if r.judge and ("invented_fact" in r.judge.failures or "false_alarm" in r.judge.deductions)
    ]
    m.invented_share = _share(len(invented), len(ok))
    bad_cit = [r for r in ok if r.answer and r.answer.invalid_citations]  # ссылка не на то поле
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
    durations = [r.duration_s for r in ok]
    m.mean_duration_s = round(mean(durations), 1) if durations else None

    outcomes: dict[str, set[tuple]] = defaultdict(set)
    for r in ok:
        verdict = r.answer.card.verdict.value if r.answer and r.answer.card else None
        outcomes[r.question_id].add((r.checks_passed, verdict))
    repeated = {q: o for q, o in outcomes.items() if sum(1 for r in ok if r.question_id == q) > 1}
    if repeated:
        m.stability = _share(sum(1 for o in repeated.values() if len(o) == 1), len(repeated))

    guards = [r for r in ok if r.type == "guard"]
    m.guard_share = _share(sum(1 for r in guards if r.check_notes.get("guarded")), len(guards))
    comparisons = [r for r in ok if r.type == "comparison"]
    m.comparison_share = _share(
        sum(1 for r in comparisons if r.check_notes.get("verdict_match")), len(comparisons)
    )
    m.prompt_version = next((r.prompt_version for r in records if r.prompt_version), "")

    def _group(rows: list[RunRecord]) -> dict[str, float | int]:
        return {
            "n": len(rows),
            "checks_pass": _share(sum(1 for r in rows if r.checks_passed), len(rows)) or 0,
            "judge_mean": round(mean(r.judge.score for r in rows if r.judge), 2)
            if any(r.judge for r in rows)
            else 0,
        }

    order = ["card", "answer", "refuse", "infer", "comparison", "dialog", "guard"]
    for kind in sorted(
        {r.type for r in ok}, key=lambda k: (order.index(k) if k in order else 99, k)
    ):
        m.by_type[kind] = _group([r for r in ok if r.type == kind])
    for cat in sorted({r.category for r in ok if r.category}):
        m.by_category[cat] = _group([r for r in ok if r.category == cat])
    return m
