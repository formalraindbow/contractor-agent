from decimal import Decimal

import pytest

from contractor_agent.agent.citations import (
    check_citation,
    normalize_path,
    number_matches,
    numbers_in,
)
from contractor_agent.agent.schema import Citation
from contractor_agent.data.loader import Snapshot


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("убыток 26,2 млн ₽", [Decimal("26200000")]),
        ("на сумму 180 809,59 ₽", [Decimal("180809.59")]),
        ("54 действующих производства", [Decimal("54")]),
        (
            "на конец\u202f2025\u202fгода: 3\u202f782\u202f000\u202f₽",
            [Decimal("2025"), Decimal("3782000")],
        ),
        ("это 1,1 % чистых активов", [Decimal("0.011"), Decimal("1.1")]),
        ("капитал -14 000 ₽", [Decimal("-14000")]),
        ("без чисел", []),
    ],
)
def test_numbers_in(text: str, expected: list[Decimal]) -> None:
    assert numbers_in(text) == expected


def test_number_matches_tolerance() -> None:
    assert number_matches(Decimal("26200000"), Decimal("26249000"))  # округление в тексте
    assert not number_matches(Decimal("100"), Decimal("54"))
    assert (
        normalize_path("[executionProceedings[3].amount]")
        == "report.executionProceedings[3].amount"
    )


def test_check_citation_against_report(snapshot: Snapshot) -> None:
    inns = ["6165169320"]  # ГДК
    ok = check_citation(
        snapshot,
        inns,
        Citation(claim="убыток 26,2 млн ₽", source_path="report.finReports[1].common.profit"),
    )
    assert ok.ok and ok.value == -26_249_000
    wrong = check_citation(
        snapshot,
        inns,
        Citation(claim="убыток 99 млн ₽", source_path="report.finReports[1].common.profit"),
    )
    assert not wrong.ok and "не совпадает" in wrong.why
    missing = check_citation(
        snapshot, inns, Citation(claim="штат 10 человек", source_path="report.baseInfo.staff")
    )
    assert not missing.ok and "пустое" in missing.why
    container = check_citation(
        snapshot, inns, Citation(claim="2 открытых дела", source_path="report.arbitrationByStatus")
    )
    assert container.ok  # контейнер: достаточно адреса
    nowhere = check_citation(
        snapshot, inns, Citation(claim="x", source_path="report.baseInfo.headcount")
    )
    assert not nowhere.ok
    unknown_inn = check_citation(
        snapshot, ["0000000000"], Citation(claim="x", source_path="report.baseInfo.inn")
    )
    assert not unknown_inn.ok


def test_extract_inline_citations() -> None:
    from contractor_agent.agent.citations import extract_inline_citations, is_meta_path

    text = (
        "- Светофор: зелёный [report.baseInfo.riskLevel, report.zskRiskLevel]\n"
        "- Убыток за 2024 год — 26 249 000 ₽ (report.finReports[1].common.profit); "
        "блокировка счетов [report.reputationalRisks.negative[3].code]\n"
        "- Вывод: не рекомендуем [report.verdict_ru]\n"
        "- Чего нет: численность (source_path: report.baseInfo.staff)\n"
        "Светофор: зелёный [report.baseInfo.riskLevel]; ЗСК: зелёный [report.zskRiskLevel]"
    )
    found = extract_inline_citations(text)
    assert [(c.claim, c.source_path) for c in found] == [
        ("Светофор: зелёный", "report.baseInfo.riskLevel"),
        ("Светофор: зелёный", "report.zskRiskLevel"),
        ("Убыток за 2024 год — 26 249 000 ₽", "report.finReports[1].common.profit"),
        ("блокировка счетов", "report.reputationalRisks.negative[3].code"),
        ("Вывод: не рекомендуем", "report.verdict_ru"),
        ("Чего нет: численность", "report.baseInfo.staff"),
        ("Светофор: зелёный", "report.baseInfo.riskLevel"),
        ("ЗСК: зелёный", "report.zskRiskLevel"),
    ]
    assert is_meta_path("report.verdict_ru") and not is_meta_path("report.baseInfo.staff")
