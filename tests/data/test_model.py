from collections import Counter
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from contractor_agent.data.model import (
    SECTIONS,
    ExecutionProceeding,
    Report,
    RiskFlag,
    parse_moscow_date,
)

MINIMAL: dict[str, Any] = {
    "reportDate": "2026-08-27T21:00:00.000Z",
    "baseInfo": {"inn": "052500690823", "ogrn": "1", "shortName": "ИП", "riskLevel": "HIGH"},
    "status": {"status": "CURRENT"},
    "zskRiskLevel": "GREEN",
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-08-27T21:00:00.000Z", date(2026, 8, 28)),  # московская полночь → 28.08
        ("2011-07-05T20:00:00.000Z", date(2011, 7, 6)),  # UTC+4 в 2011: тоже полночь
        ("2023-11-17", date(2023, 11, 17)),  # плоская дата inspections
        (date(2020, 1, 1), date(2020, 1, 1)),
    ],
)
def test_moscow_date(raw: Any, expected: date) -> None:
    assert parse_moscow_date(raw) == expected


def test_moscow_date_rejects_other_formats() -> None:
    with pytest.raises(ValueError):
        parse_moscow_date("17.11.2023")


def test_minimal_report_and_aliases() -> None:
    report = Report.model_validate(MINIMAL)
    assert report.inn == "052500690823"  # ведущий ноль цел
    assert report.report_date == date(2026, 8, 28)
    assert report.fin_reports is None and report.licenses is None
    dumped = report.model_dump()
    assert dumped["baseInfo"]["inn"] == "052500690823"  # ключи — как в отчёте
    assert dumped["zskRiskLevel"] == "GREEN"


def test_required_sections() -> None:
    with pytest.raises(ValidationError):
        Report.model_validate({k: v for k, v in MINIMAL.items() if k != "baseInfo"})


def test_csv_strings_are_typed_by_field() -> None:
    p = ExecutionProceeding.model_validate(
        {
            "number": "85117/22/02006-ИП",
            "date": "2022-05-04T21:00:00.000Z",
            "amount": "517235.54",
            "active": "true",
        }
    )
    assert p.active is True
    assert p.amount == Decimal("517235.54")
    assert p.date == date(2022, 5, 5)
    assert p.number == "85117/22/02006-ИП"
    # нет суммы — None, а не 0
    assert ExecutionProceeding.model_validate({"active": "true"}).amount is None


def test_cyrillic_code_is_normalized() -> None:
    flag = RiskFlag.model_validate({"code": "аrbitrationDefendant", "name": "x", "chapter": "y"})
    assert flag.code == "arbitrationDefendant"
    assert flag.code.isascii()


def test_empty_string_and_whitespace() -> None:
    raw = dict(MINIMAL, baseInfo=dict(MINIMAL["baseInfo"], address="", email="  a@b.ru  "))
    report = Report.model_validate(raw)
    assert report.base_info.address is None
    assert report.base_info.email == "a@b.ru"


def test_section_state_absent_empty_present() -> None:
    raw = dict(
        MINIMAL,
        finReports=[],
        executionProceedings=[{"active": "false"}],
        arbitrationByStatus={"plaintiffArbitration": {"plaintiffArbitrationFinished": {}}},
        reputationalRisks={"negative": [], "positive": [{"code": "fnsBlocking"}]},
    )
    report = Report.model_validate(raw)
    assert report.section_state("licenses") == "absent"
    assert report.section_state("finReports") == "empty"
    assert report.section_state("executionProceedings") == "present"
    assert report.section_state("arbitrationByStatus") == "empty"  # «пустышка» без commonCount
    assert report.section_state("reputationalRisks") == "present"
    assert len(SECTIONS) == 18 and "reportDate" not in SECTIONS


def test_spec_only_fields_default_to_none() -> None:
    """Поля из спецификации банка, которых нет в снапшоте: модель их знает, данные — нет."""
    report = Report.model_validate(dict(MINIMAL, baseInfo=dict(MINIMAL["baseInfo"], staff="11–50")))
    assert report.base_info.staff == "11–50"
    assert Report.model_validate(MINIMAL).base_info.staff is None
    assert "staff" in Report.model_validate(MINIMAL).model_dump()["baseInfo"]


def test_unknown_fields_are_kept_not_dropped() -> None:
    report = Report.model_validate(dict(MINIMAL, newSection={"x": 1}))
    assert report.model_extra == {"newSection": {"x": 1}}


# --- контракт на снапшот -------------------------------------------------------


def _extras(model: BaseModel, path: str, out: Counter[str]) -> None:
    for key in model.model_extra or {}:
        out[f"{path}.{key}"] += 1
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        alias = field.alias or name
        if isinstance(value, BaseModel):
            _extras(value, f"{path}.{alias}", out)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, BaseModel):
                    _extras(item, f"{path}[].{alias}", out)


def test_all_200_validate_without_unknown_fields(plain_reports: list[tuple[str, dict]]) -> None:
    reports = [Report.model_validate(r) for _, r in plain_reports]
    assert len(reports) == 200
    assert len({r.inn for r in reports}) == 200
    extras: Counter[str] = Counter()
    for r in reports:
        _extras(r, "report", extras)
    assert not extras, extras  # схема поплыла — добавь поля в модель

    states = Counter(
        (src, sec, rep.section_state(sec))
        for (src, _), rep in zip(plain_reports, reports, strict=True)
        for sec in SECTIONS
    )
    assert states[("json", "finReports", "absent")] == 25
    assert states[("json", "finReports", "empty")] == 8
    assert states[("json", "arbitrationByStatus", "empty")] == 52
    assert states[("csv", "executionProceedings", "empty")] == 57  # политика присутствия
    assert states[("csv", "licenses", "present")] == 13


def test_demo_cards(plain_reports: list[tuple[str, dict]]) -> None:
    by_inn = {r["baseInfo"]["inn"]: Report.model_validate(r) for _, r in plain_reports}

    sport = by_inn["9705152496"]  # $numberLong, включая liabilities.capitals
    assert sport.fin_reports and sport.fin_reports[2].liabilities.capitals == 2540684000

    gdk = by_inn["6165169320"]  # дата с 20:00Z
    assert gdk.base_info.registration_info.registration_date == date(2011, 7, 6)
    assert "fnsBlocking" in {f.code for f in gdk.reputational_risks.negative}

    le_monlid = by_inn["5029069967"]
    assert le_monlid.section_state("finReports") == "empty"
    active = [p for p in le_monlid.execution_proceedings if p.active]
    assert len(le_monlid.execution_proceedings) == 1744 and len(active) == 45
    assert sum(p.amount is None for p in active) == 33
    assert le_monlid.arbitration_by_status.common_amount == 4534783044

    kasatkin = by_inn["772377037026"]
    assert kasatkin.founders_info is None and kasatkin.fin_reports is None
    assert kasatkin.base_info.kpp is None and kasatkin.base_info.address is None

    build_yug = by_inn["2311304742"]
    assert build_yug.status.reason_name.startswith("Юридическим лицом принято решение")
    assert [p.amount for p in build_yug.execution_proceedings if p.active] == [None]

    techprof = by_inn["1684017097"]
    assert techprof.reputational_risks.negative == []
    assert techprof.section_state("arbitrationByStatus") == "empty"

    ucgn = by_inn["0277985654"]  # CSV
    assert ucgn.branches_info.branches_count == 12 and len(ucgn.branches_info.branches) == 12
    verny = by_inn["5003135950"]  # CSV, ячеек производств нет → политика присутствия
    assert verny.section_state("executionProceedings") == "empty"
    assert verny.status.reason_name and verny.section_state("licenses") == "absent"

    positive_codes = {f.code for r in by_inn.values() for f in r.reputational_risks.positive}
    assert all(c.isascii() for c in positive_codes)
