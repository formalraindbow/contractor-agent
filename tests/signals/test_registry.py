from datetime import date

from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import Report
from contractor_agent.signals import registry
from contractor_agent.signals.finance import months_between
from contractor_agent.signals.model import GapReason, Severity


def _by_code(res) -> dict:
    return {s.code: s for s in res.signals}


def test_maxmarket_bankruptcy_is_terminal_twice(snapshot: Snapshot) -> None:
    res = _by_code(registry.run(snapshot.get("5032257375")))
    assert res["status_reason"].terminal and res["status_reason"].title_ru == "Банкротство"
    assert (
        res["bankruptcy_trustee"].terminal
        and "09.02.2026" in res["bankruptcy_trustee"].explanation_ru
    )
    assert "director_recent" not in res  # управляющий — не «смена директора»


def test_procedural_reasons_are_moderate(snapshot: Snapshot) -> None:
    build_yug = _by_code(registry.run(snapshot.get("2311304742")))["status_reason"]
    assert build_yug.severity is Severity.MODERATE and not build_yug.terminal
    assert (
        build_yug.title_ru == "Решение о смене адреса"
        and "(обновлён 02.08.2026)" in build_yug.explanation_ru
    )
    agro = _by_code(registry.run(snapshot.get("0232016850")))["status_reason"]
    assert agro.title_ru == "Уменьшение уставного капитала"
    rado = _by_code(registry.run(snapshot.get("2100006761")))["status_reason"]
    assert rado.terminal and rado.title_ru == "Предстоящее исключение из ЕГРЮЛ"


def test_status_closed_is_terminal_on_synthetic_report(snapshot: Snapshot) -> None:
    raw = snapshot.get("1684017097").model_dump()
    raw["status"]["status"] = "CLOSED"
    res = _by_code(registry.run(Report.model_validate(raw)))
    assert res["status_not_current"].terminal


def test_director_change_excludes_founding_director(snapshot: Snapshot) -> None:
    verny = _by_code(registry.run(snapshot.get("5003135950")))["director_recent"]
    assert verny.severity is Severity.MODERATE and verny.value["months"] == 4
    counts = {"moderate": 0, "info": 0}
    for rec in snapshot:
        s = _by_code(registry.run(rec.report)).get("director_recent")
        if s:
            counts[s.severity.value] += 1
            person = rec.report.founders_info.auth_person
            assert person.position_date != rec.report.base_info.registration_info.registration_date
    assert counts == {
        "moderate": 11,
        "info": 15,
    }  # 12-й — конкурсный управляющий, он отдельный сигнал


def test_young_company_and_staff_gap(snapshot: Snapshot) -> None:
    young = [r for r in snapshot if "young_company" in _by_code(registry.run(r.report))]
    assert len(young) == 18
    assert all(r.report.base_info.registration_info.years_from_registration == 0 for r in young)
    gaps = registry.run(snapshot.get("1684017097")).gaps
    assert [(g.criterion, g.reason) for g in gaps] == [
        ("численность персонала", GapReason.FIELD_ABSENT)
    ]


def test_expired_license_and_violation_inspection(snapshot: Snapshot) -> None:
    res = _by_code(registry.run(snapshot.get("7718083574")))
    assert res["license_expired"].value["count"] == 2
    assert res["license_expired"].source_path == "report.licenses[0].status" or res[
        "license_expired"
    ].source_path.startswith("report.licenses[")
    assert res["inspection_violation"].severity is Severity.MODERATE
    capped = [
        r.inn
        for r in snapshot
        if any(g.criterion == "свежие проверки" for g in registry.run(r.report).gaps)
    ]
    assert set(capped) == {"5029069967", "7816085851"}


def test_months_between() -> None:
    assert months_between(date(2026, 2, 9), date(2026, 7, 31)) == 5
    assert months_between(date(2026, 3, 18), date(2026, 7, 31)) == 4
    assert months_between(date(2025, 8, 1), date(2026, 8, 1)) == 12


def test_liquidator_and_external_manager_are_terminal(snapshot: Snapshot) -> None:
    for position in ("ЛИКВИДАТОР", "Внешний управляющий", "ПРЕДСЕДАТЕЛЬ ЛИКВИДАЦИОННОЙ КОМИССИИ"):
        raw = snapshot.get("1684017097").model_dump()
        raw["foundersInfo"]["authPerson"]["positionName"] = position
        res = _by_code(registry.run(Report.model_validate(raw)))
        assert res["bankruptcy_trustee"].terminal, position
        assert registry.is_terminal_status(Report.model_validate(raw))
