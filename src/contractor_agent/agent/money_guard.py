"""Validate amounts in the visible answer, not only in the model's citation claims.

The model can omit an amount from a citation's claim while still printing it.
This independent check catches that gap. It is not semantic proof of every claim:
roles, periods and causal statements still require their own checks and review.
"""

import re
from bisect import bisect_left
from decimal import Decimal
from typing import Any

from contractor_agent.agent.citations import _MONEY, CitationCheck, number_matches, numbers_in
from contractor_agent.agent.schema import Citation
from contractor_agent.data.loader import normalize_name, strip_legal_form
from contractor_agent.mcp_server.tools import Tools


def amounts(text: str) -> list[Decimal]:
    return [n for part in _MONEY.findall(text) for n in numbers_in(part)]


def _values(value: Any) -> set[Decimal]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return set().union(*(_values(v) for v in value.values()))
    if isinstance(value, list | tuple):
        return set().union(*(_values(v) for v in value))
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        number = Decimal(str(value))
        return {abs(number)} if number.is_finite() else set()
    return set()


def monetary_violations(
    tools: Tools, inns: list[str], text: str, question: str
) -> list[CitationCheck]:
    if not amounts(text) or not inns:
        return []
    # A hypothetical amount supplied by the user may be repeated as context.
    allowed = {abs(n) for n in amounts(question)}
    for inn in inns:
        report = tools.source.get(inn)
        if report is None:
            continue
        allowed |= _values(report)
        for method in (
            tools.get_risk_signals,
            tools.get_financials,
            tools.get_arbitration_summary,
            tools.get_enforcement_summary,
        ):
            response = method(inn)
            if response.available:
                allowed |= _values(response.data)
    ordered = sorted(allowed)

    def supported(number):
        index = bisect_left(ordered, abs(number))
        return any(
            number_matches(number, value) for value in ordered[max(0, index - 1) : index + 1]
        )

    bad = []
    for line in text.splitlines():
        missing = [str(number) for number in amounts(line) if not supported(number)]
        if missing:
            bad.append(
                CitationCheck(
                    Citation(
                        claim=line.strip().lstrip("-*• "), source_path="report.__unverified_amount"
                    ),
                    False,
                    "В тексте есть сумма, которой нет в полях отчёта или расчётах: "
                    + ", ".join(missing),
                )
            )
    return bad


_ASSIGNED_LABEL = re.compile(
    r"\b(светофор(?:\s+банка)?|зск)[\s:*—–=-]{0,15}"
    r"(зел[её]ный|ж[её]лтый|красный|серый)\b",
    re.I,
)


def label_violations(tools: Tools, inns: list[str], text: str) -> list[CitationCheck]:
    """Check explicit assignments even when the model omits them from its citations."""
    reports = {inn: tools.source.get(inn) for inn in inns}
    names = {
        inn: strip_legal_form(normalize_name(r.base_info.short_name))
        for inn, r in reports.items()
        if r is not None
    }
    current = inns[0] if len(inns) == 1 else None
    bad = []
    for line in text.splitlines():
        named = [
            inn
            for inn, name in names.items()
            if inn in line or (name and re.search(rf"\b{re.escape(name)}\b", normalize_name(line)))
        ]
        if named:
            current = named[0] if len(named) == 1 else None
        if current is None or reports.get(current) is None:
            continue
        labels = tools.get_report_summary(current).data["labels"]
        for match in _ASSIGNED_LABEL.finditer(line):
            is_zsk = match[1].lower() == "зск"
            expected = labels["zsk" if is_zsk else "svetofor"]
            if match[2].lower().replace("ё", "е") != expected.replace("ё", "е"):
                bad.append(
                    CitationCheck(
                        Citation(
                            claim=line.strip(),
                            inn=current,
                            source_path="report.zskRiskLevel"
                            if is_zsk
                            else "report.baseInfo.riskLevel",
                        ),
                        False,
                        f"В отчёте исходная оценка: {expected}. Не пересчитывай её.",
                    )
                )
    return bad
