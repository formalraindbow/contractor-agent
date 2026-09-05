# ruff: noqa: E501
"""Отчёт по прогонам: таблица метрик по моделям и разбор провалов."""

from __future__ import annotations

from evals.metrics import Metrics
from evals.runner import RunRecord


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def render(
    metrics: list[Metrics], records: dict[str, list[RunRecord]], max_failures: int = 40
) -> str:
    lines = ["# Эвалы — метрики по моделям", ""]
    lines.append(
        "| Модель | Вопросов | Подтверждаемых ответов | Корректных отказов | Выдуманных фактов | Неверных ссылок | Пропущенных критичных | Судья (0–5) | Стабильность | Время, с |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in metrics:
        lines.append(
            f"| `{m.model}` | {m.total}{f' ({m.errors} ошибок)' if m.errors else ''} | {_pct(m.grounded_share)} | "
            f"{_pct(m.refusal_share)} | {_pct(m.invented_share)} | {_pct(m.bad_citation_share)} | {_pct(m.missed_critical_share)} | "
            f"{m.judge_mean if m.judge_mean is not None else '—'} | {_pct(m.stability)} | "
            f"{m.mean_duration_s if m.mean_duration_s is not None else '—'} |"
        )
    for m in metrics:
        lines += [
            "",
            f"## `{m.model}` — по типам вопросов",
            "",
            "| Тип | n | Проверки пройдены | Судья |",
            "|---|---:|---:|---:|",
        ]
        for kind, row in m.by_type.items():
            lines.append(
                f"| {kind} | {row['n']} | {_pct(row['checks_pass'])} | {row['judge_mean']} |"
            )
        failures = [
            r
            for r in records.get(m.model, [])
            if not r.checks_passed or (r.judge and r.judge.score < 4)
        ]
        if failures:
            lines += ["", f"### Провалы и снижения ({len(failures)})", ""]
            for r in failures[:max_failures]:
                why = "; ".join(r.check_failures) or (r.judge.comment if r.judge else "")
                judge = (
                    f" · судья {r.judge.score}: {', '.join(r.judge.failures + r.judge.deductions)}"
                    if r.judge
                    else ""
                )
                lines.append(f"- `{r.question_id}` ({r.type}) — {why}{judge}")
    return "\n".join(lines) + "\n"
