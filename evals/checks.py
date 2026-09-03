"""Детерминированные проверки ответа — без модели, бесплатно и воспроизводимо.

Они ловят то, что судья может пропустить или выдумать: обязательные подстроки,
корректный отказ, совпадение вывода с разметкой, валидность цитат (по нашему же
валидатору), запрещённые ярлыки, дату отчёта, пропуск критичного факта.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from contractor_agent.agent.schema import Answer
from contractor_agent.signals.model import TERMINAL_RU, VERDICT_RU, Verdict
from evals.gold import FORBIDDEN_LABELS, REFUSAL_MARKERS, GoldQuestion


@dataclass
class CheckResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: dict[str, bool] = field(default_factory=dict)


def verdict_in_text(text: str) -> str | None:
    """Исход по формулировке в тексте, когда карточки нет (ответ на вопрос, а не проверка)."""
    lowered = text.casefold()
    if TERMINAL_RU.split("— ")[-1].casefold() in lowered:  # хвост терминальной фразы
        return Verdict.NOT_RECOMMENDED.value
    for verdict in (Verdict.NOT_RECOMMENDED, Verdict.CHECK, Verdict.OK):
        if VERDICT_RU[verdict].casefold() in lowered:
            return verdict.value
    return None


def _has(text: str, needle: str) -> bool:
    """Подстрока без учёта регистра; варианты через « | »; пробелы любые."""
    lowered = re.sub(r"\s+", " ", text.casefold())
    return any(
        re.sub(r"\s+", " ", v.strip().casefold()) in lowered
        for v in needle.split(" | ")
        if v.strip()
    )


def is_refusal(answer: Answer) -> bool:
    return answer.kind == "refusal" or any(_has(answer.text_md, m) for m in REFUSAL_MARKERS)


def mentions_report_date(answer: Answer, report_date: str) -> bool:
    iso = report_date
    y, m, d = iso.split("-")
    return any(
        _has(answer.text_md, v) for v in (iso, f"{d}.{m}.{y}", f"{int(d)} ", f"{y} г")
    ) or iso in str(answer.report_dates.values())


def check(question: GoldQuestion, answer: Answer, report_date: str) -> CheckResult:
    failures: list[str] = []
    notes: dict[str, bool] = {}
    text = answer.text_md

    for needle in question.must_not_mention + (
        FORBIDDEN_LABELS if question.type != "refuse" else []
    ):
        if _has(text, needle):
            failures.append(f"запрещённое: «{needle}»")
    notes["labels_ok"] = not any(_has(text, n) for n in FORBIDDEN_LABELS)

    if question.type == "refuse":
        refused = is_refusal(answer)
        notes["refused"] = refused
        if not refused:
            failures.append("нет корректного отказа — ответ по существу при отсутствии данных")
        for needle in question.must_mention:
            if not _has(text, needle):
                failures.append(f"в отказе нет «{needle}»")
    else:
        for needle in question.must_mention:
            if not _has(text, needle):
                failures.append(f"не названо «{needle}»")
        notes["citations_valid"] = not answer.invalid_citations
        if answer.invalid_citations:
            failures.append(f"невалидных цитат: {len(answer.invalid_citations)}")
        if (
            question.type in ("answer", "infer")
            and not answer.citations
            and not answer.invalid_citations
        ):
            failures.append("ни одной цитаты с адресом поля")

    if question.type in ("card", "infer") and question.expected_verdict:
        actual = answer.card.verdict.value if answer.card else verdict_in_text(answer.text_md)
        notes["verdict_match"] = actual == question.expected_verdict
        if actual != question.expected_verdict:
            failures.append(f"вывод {actual} вместо {question.expected_verdict}")

    if question.type == "card":
        missed = [fact for fact in question.must_name if not _has(text, fact)]
        notes["missed_critical"] = bool(missed)
        for fact in missed:
            failures.append(f"не назван критичный факт: «{fact}»")
        notes["report_date"] = mentions_report_date(answer, report_date)
        if not notes["report_date"]:
            failures.append("нет даты отчёта")

    return CheckResult(passed=not failures, failures=failures, notes=notes)
