from decimal import Decimal

import pytest

from contractor_agent.data.loader import Snapshot
from contractor_agent.signals import finance
from contractor_agent.signals.model import GapReason, Severity, check_paths
from contractor_agent.signals.text import rub


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (26_249_000, "26,2 млн ₽"),
        (-26_249_000, "-26,2 млн ₽"),
        (1_814_055_000, "1,8 млрд ₽"),
        (Decimal("517235.54"), "517 235,54 ₽"),
        (-14_000, "-14 000 ₽"),
        (25, "25 ₽"),
        (1_000_000, "1 млн ₽"),
    ],
)
def test_rub(value, expected: str) -> None:
    assert rub(value) == expected


def _codes(res) -> dict[str, Severity]:
    return {s.code: s.severity for s in res.signals}


def test_gdk_loss_and_liquidity_come_from_2024_row(snapshot: Snapshot) -> None:
    res = finance.run(snapshot.get("6165169320"))
    codes = _codes(res)
    assert codes == {"fin_loss": Severity.MODERATE, "fin_low_liquidity": Severity.MODERATE}
    loss = next(s for s in res.signals if s.code == "fin_loss")
    assert loss.year == 2024 and loss.value == {"profit": -26_249_000, "year": 2024}
    assert (
        loss.source_path.endswith(".common.profit") and "убыток 26,2 млн ₽" in loss.explanation_ru
    )
    liq = next(s for s in res.signals if s.code == "fin_low_liquidity")
    assert liq.value["ratio"] == 0.49 and liq.year == 2024
    assert res.gaps == []  # 2025 капитал положительный — сигнала нет, но и пробела нет


def test_rado_negative_equity_is_critical(snapshot: Snapshot) -> None:
    res = finance.run(snapshot.get("2100006761"))
    eq = next(s for s in res.signals if s.code == "fin_negative_equity")
    assert eq.severity is Severity.CRITICAL and eq.value == {"capitals": -14_000, "year": 2025}
    assert eq.source_path == "report.finReports[0].liabilities.capitals"


def test_techprof_clean_but_profit_line_missing_is_a_gap(snapshot: Snapshot) -> None:
    report = snapshot.get("1684017097")
    assert [r.year for r in finance.real_rows(report)] == [2025, 2024]  # 2023 — до регистрации
    res = finance.run(report)
    assert res.signals == []
    assert {(g.criterion, g.reason) for g in res.gaps} == {
        ("прибыль или убыток", GapReason.FIELD_ABSENT),
        ("текущая ликвидность", GapReason.FIELD_ABSENT),
    }


def test_sport_three_loss_years(snapshot: Snapshot) -> None:
    res = finance.run(snapshot.get("9705152496"))
    streak = next(s for s in res.signals if s.code == "fin_loss_streak")
    assert streak.severity is Severity.INFO
    assert streak.value == {"loss_years": [2023, 2024, 2025], "years_in_report": 3}
    assert len(streak.source_paths) == 3


def test_dom_na_moike_and_maxmarket(snapshot: Snapshot) -> None:
    dom = _codes(finance.run(snapshot.get("7826131151")))
    assert dom == {"fin_negative_equity": Severity.CRITICAL, "fin_low_liquidity": Severity.MODERATE}
    maxmarket = finance.run(snapshot.get("5032257375"))
    assert _codes(maxmarket) == {"fin_stale": Severity.INFO}
    assert maxmarket.signals[0].value == {"latest_year": 2023, "expected_year": 2025}


def test_statement_gaps_by_company_kind(snapshot: Snapshot) -> None:
    (kasatkin,) = finance.run(snapshot.get("772377037026")).gaps
    assert kasatkin.reason is GapReason.NOT_APPLICABLE and not kasatkin.floor_check
    assert "индивидуальный предприниматель" in kasatkin.text_ru
    (le_monlid,) = finance.run(snapshot.get("5029069967")).gaps
    assert le_monlid.reason is GapReason.SECTION_EMPTY and le_monlid.floor_check
    assert le_monlid.source_path == "report.finReports"


def test_net_assets_for_debt_comparison(snapshot: Snapshot) -> None:
    picked = finance.net_assets(snapshot.get("8622002583"))  # САМЗА
    assert picked and (picked.value, picked.year) == (2_316_000, 2025)
    assert finance.net_assets(snapshot.get("5029069967")) is None


def test_loss_rule_equals_bank_profit_flag_and_paths_resolve(snapshot: Snapshot) -> None:
    bank, ours = set(), set()
    for rec in snapshot:
        res = finance.run(rec.report)
        for item in (*res.signals, *res.gaps):
            check_paths(rec.report, item)
        if any(f.code == "profit" for f in rec.report.reputational_risks.negative):
            bank.add(rec.inn)
        if "fin_loss" in {s.code for s in res.signals}:
            ours.add(rec.inn)
    assert bank == ours and len(ours) == 10
