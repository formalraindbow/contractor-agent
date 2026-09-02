from datetime import date
from decimal import Decimal

import pytest

from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import ExecutionProceeding, Report
from contractor_agent.data.paths import PathNotFoundError, iter_leaf_paths, parse_path, resolve


def test_parse_path_requires_report_prefix() -> None:
    assert parse_path("report.executionProceedings[12].amount") == (
        "executionProceedings",
        12,
        "amount",
    )
    for bad in ("executionProceedings[0]", "report..x", "report.a[b]", ""):
        with pytest.raises(PathNotFoundError):
            parse_path(bad)


def test_resolve_on_demo_cards(snapshot: Snapshot) -> None:
    le_monlid = snapshot.get("5029069967")
    assert resolve(le_monlid, "report.arbitrationByStatus.commonAmount") == 4534783044
    assert resolve(le_monlid, "report.baseInfo.inn") == "5029069967"
    assert resolve(le_monlid, "report.finReports") == []  # пусто — это значение, не ошибка
    assert resolve(le_monlid, "report.branchesInfo") is None  # нет — тоже значение
    first = resolve(le_monlid, "report.executionProceedings[0]")
    assert isinstance(first, ExecutionProceeding)
    assert isinstance(resolve(le_monlid, "report.executionProceedings[0].date"), date)
    assert resolve(le_monlid, "report.executionProceedings[1743].active") in (True, False)

    gdk = snapshot.get("6165169320")
    assert resolve(gdk, "report.baseInfo.registrationInfo.registrationDate") == date(2011, 7, 6)
    assert resolve(gdk, "report.reputationalRisks.negative[0].code")

    with pytest.raises(PathNotFoundError):
        resolve(le_monlid, "report.executionProceedings[1744].amount")  # за границей
    with pytest.raises(PathNotFoundError):
        resolve(le_monlid, "report.baseInfo.headcount")  # такого поля нет
    with pytest.raises(PathNotFoundError):
        resolve(le_monlid, "report.baseInfo.inn.digits")  # скаляр — не контейнер


def test_resolve_reads_unknown_fields_too() -> None:
    report = Report.model_validate(
        {
            "reportDate": "2026-08-01T21:00:00.000Z",
            "baseInfo": {"inn": "1", "ogrn": "2", "shortName": "x", "riskLevel": "LOW"},
            "status": {"status": "CURRENT"},
            "zskRiskLevel": "GREEN",
            "headcount": {"total": 12},
        }
    )
    assert resolve(report, "report.headcount.total") == 12


def test_every_leaf_path_round_trips(snapshot: Snapshot) -> None:
    leaves = 0
    for record in snapshot:
        for path, value in iter_leaf_paths(record.report):
            assert resolve(record.report, path) == value, path
            leaves += 1
    assert leaves > 60_000  # из них ~7 тысяч — один ЛЕ МОНЛИД


def test_leaf_values_are_normalized(snapshot: Snapshot) -> None:
    kinds = {type(v) for _, v in iter_leaf_paths(snapshot.get("5029069967"))}
    assert kinds <= {str, int, bool, date, Decimal, type(None)}
