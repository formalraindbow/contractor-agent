from decimal import Decimal

import pytest

from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import Report
from contractor_agent.signals import enforcement
from contractor_agent.signals.model import GapReason, Severity, check_paths
from contractor_agent.signals.text import plural, share


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, "1 дело"), (2, "2 дела"), (5, "5 дел"), (11, "11 дел"), (21, "21 дело"), (114, "114 дел")],
)
def test_plural(n: int, expected: str) -> None:
    assert plural(n, "дело", "дела", "дел") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0.000004"), "меньше 0,1 %"),
        (Decimal("0.011"), "1,1 %"),
        (Decimal("0.25"), "25 %"),
        (Decimal("1.2"), "больше"),
        (Decimal("18.08"), "в 18 раз больше"),
    ],
)
def test_share(value: Decimal, expected: str) -> None:
    assert share(value) == expected


def _active(snapshot: Snapshot, inn: str):
    res = enforcement.run(snapshot.get(inn))
    return res, next(s for s in res.signals if s.code == "enforcement_active")


def test_le_monlid_aggregates_not_lists(snapshot: Snapshot) -> None:
    res, active = _active(snapshot, "5029069967")
    assert active.severity is Severity.CRITICAL  # 45 ≥ 10
    v = active.value
    assert (v["active_count"], v["unknown_amount_count"], v["sum_is_lower_bound"]) == (45, 33, True)
    assert v["known_sum"] == pytest.approx(1_571_231.0, abs=1)
    assert v["net_assets"] is None and "соразмерность долга оценить нельзя" in active.explanation_ru
    assert "не менее 1,6 млн ₽ (у 33 из 45 сумма не указана)" in active.explanation_ru
    assert len(active.source_paths) <= 12  # топ-10 + опоры, не 1744
    finished = next(s for s in res.signals if s.code == "enforcement_finished")
    assert finished.severity is Severity.INFO and finished.value["finished_count"] == 1699
    assert "к действующим долгам не относятся" in finished.explanation_ru
    stale = next(s for s in res.signals if s.code == "enforcement_stale_active")
    assert stale.value["count"] == 2 and "29.11.2021" in stale.explanation_ru


def test_severity_is_relative_to_net_assets(snapshot: Snapshot) -> None:
    _, igraplast = _active(snapshot, "7725497512")  # 180 тыс. долга при капитале 10 тыс.
    assert igraplast.severity is Severity.CRITICAL
    assert "в 18 раз больше чистых активов" in igraplast.explanation_ru
    _, aprel = _active(snapshot, "7813664770")  # 24,63 ₽ при капитале 6,8 млн
    assert aprel.severity is Severity.INFO
    assert (
        "1 действующее исполнительное производство на сумму 24,63 ₽; это меньше 0,1 %"
        in aprel.explanation_ru
    )
    _, maxmarket = _active(snapshot, "5032257375")  # 54 штуки — критично по числу, хоть и 1,1 %
    assert maxmarket.severity is Severity.CRITICAL
    assert maxmarket.value["share_of_net_assets"] == pytest.approx(0.0109, abs=1e-3)


def test_unknown_sums_are_never_zero(snapshot: Snapshot) -> None:
    res, build_yug = _active(snapshot, "2311304742")
    assert build_yug.severity is Severity.MODERATE
    assert (
        "сумма в отчёте не указана" in build_yug.explanation_ru
        and "0 ₽" not in build_yug.explanation_ru
    )
    (gap,) = res.gaps
    assert gap.reason is GapReason.FIELD_ABSENT and gap.source_path.endswith(".amount")
    res, ortus = _active(snapshot, "2016003057")
    assert ortus.severity is Severity.MODERATE and ortus.value["known_sum"] == 0
    assert res.gaps[0].criterion == "суммы исполнительных производств"


def test_absent_section_is_a_gap_and_flag_mismatch_is_info(snapshot: Snapshot) -> None:
    techprof = snapshot.get("1684017097").model_dump()
    techprof.pop("executionProceedings")
    res = enforcement.run(Report.model_validate(techprof))
    assert res.signals == [] and res.gaps[0].reason is GapReason.SECTION_ABSENT
    with_debt = snapshot.get("1684017097").model_dump()
    with_debt["executionProceedings"] = [{"number": "1", "active": True, "amount": "100"}]
    res = enforcement.run(Report.model_validate(with_debt))
    assert {s.code for s in res.signals} == {"enforcement_active", "enforcement_flag_mismatch"}


def test_active_rule_equals_bank_flag_and_paths_resolve(snapshot: Snapshot) -> None:
    bank, ours, severities = set(), set(), []
    for rec in snapshot:
        res = enforcement.run(rec.report)
        for item in (*res.signals, *res.gaps):
            check_paths(rec.report, item)
        if any(f.code == "executionProceedings" for f in rec.report.reputational_risks.negative):
            bank.add(rec.inn)
        for s in res.signals:
            if s.code == "enforcement_active":
                ours.add(rec.inn)
                severities.append(s.severity)
    assert bank == ours and len(ours) == 33
    assert not any(
        s.code == "enforcement_flag_mismatch"
        for r in snapshot
        for s in enforcement.run(r.report).signals
    )
    assert sorted(severities.count(s) for s in Severity) == [4, 12, 17]
