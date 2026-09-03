from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from contractor_agent.data.loader import Snapshot
from contractor_agent.signals.model import (
    Gap,
    GapReason,
    Severity,
    Signal,
    SignalPathError,
    SignalSet,
    Verdict,
    check_paths,
)


def _signal(code: str, severity: Severity, **kw) -> Signal:
    base = dict(
        code=code,
        severity=severity,
        title_ru="t",
        source_path="report.status.reasonName",
        explanation_ru="e",
    )
    return Signal(**{**base, **kw})


def test_severity_order_and_verdict_values() -> None:
    assert Severity.CRITICAL.rank < Severity.MODERATE.rank < Severity.INFO.rank
    assert {v.value for v in Verdict} == {"ok", "check", "not_recommended"}


def test_terminal_must_be_critical() -> None:
    with pytest.raises(ValidationError):
        _signal("x", Severity.MODERATE, terminal=True)
    assert _signal("x", Severity.CRITICAL, terminal=True).terminal


def test_source_paths_are_capped_and_models_are_frozen() -> None:
    with pytest.raises(ValidationError):
        _signal("x", Severity.INFO, source_paths=[f"report.phones[{i}]" for i in range(21)])
    s = _signal("x", Severity.INFO)
    with pytest.raises(ValidationError):
        s.severity = Severity.CRITICAL  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Signal(**s.model_dump(), extra_field=1)  # type: ignore[arg-type]


def test_signal_set_sorts_by_severity_then_code() -> None:
    signals = [
        _signal("b_info", Severity.INFO),
        _signal("z_crit", Severity.CRITICAL),
        _signal("a_mod", Severity.MODERATE),
        _signal("a_crit", Severity.CRITICAL),
    ]
    ss = SignalSet(
        inn="1", report_date=date(2026, 8, 1), signals=signals, verdict=Verdict.CHECK, score=7
    )
    assert [s.code for s in ss.signals] == ["a_crit", "z_crit", "a_mod", "b_info"]
    assert [s.code for s in ss.by_severity(Severity.CRITICAL)] == ["a_crit", "z_crit"]
    assert ss.codes == {"a_crit", "z_crit", "a_mod", "b_info"}


def test_json_round_trip_keeps_decimal_and_date() -> None:
    s = _signal(
        "enforcement_active",
        Severity.CRITICAL,
        value={"sum": Decimal("1571231.50"), "count": Decimal("45"), "since": date(2020, 9, 1)},
    )
    assert s.value == {"sum": 1571231.5, "count": 45, "since": "2020-09-01"}
    ss = SignalSet(
        inn="5029069967",
        report_date=date(2026, 7, 30),
        signals=[s],
        gaps=[
            Gap(
                criterion="финансовая отчётность",
                reason=GapReason.SECTION_EMPTY,
                source_path="report.finReports",
                text_ru="раздел есть, данных нет",
            )
        ],
        verdict=Verdict.NOT_RECOMMENDED,
        score=3,
    )
    back = SignalSet.model_validate_json(ss.model_dump_json())
    assert back == ss
    assert back.signals[0].value["sum"] == 1571231.5


def test_check_paths_against_real_report(snapshot: Snapshot) -> None:
    le_monlid = snapshot.get("5029069967")
    ok = _signal(
        "enforcement_active",
        Severity.CRITICAL,
        source_path="report.executionProceedings",
        source_paths=["report.executionProceedings[0].amount", "report.executionProceedings[1743]"],
    )
    check_paths(le_monlid, ok)  # не бросает
    gap = Gap(
        criterion="филиалы",
        reason=GapReason.SECTION_ABSENT,
        source_path="report.branchesInfo",  # отсутствует → None, это валидный адрес
        text_ru="…",
    )
    check_paths(le_monlid, gap)
    bad = _signal("x", Severity.INFO, source_paths=["report.executionProceedings[1744].amount"])
    with pytest.raises(SignalPathError):
        check_paths(le_monlid, bad)
    deep_under_absent = Gap(
        criterion="филиалы",
        reason=GapReason.SECTION_ABSENT,
        source_path="report.branchesInfo.branchesCount",  # под отсутствующей секцией — ошибка
        text_ru="…",
    )
    with pytest.raises(SignalPathError):
        check_paths(le_monlid, deep_under_absent)
