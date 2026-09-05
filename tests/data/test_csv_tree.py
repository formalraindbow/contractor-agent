import json
from pathlib import Path
from typing import Any

import pytest

from contractor_agent.data.csv_tree import (
    ALWAYS_PRESENT_DICTS,
    ALWAYS_PRESENT_LISTS,
    ColumnNameError,
    IndexGapError,
    apply_presence_policy,
    iter_rows,
    load_csv_records,
    parse_column,
    row_to_tree,
)
from contractor_agent.data.mongo import unwrap


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("report.baseInfo.inn", ("report", "baseInfo", "inn")),
        (
            "report.executionProceedings[458].amount",
            ("report", "executionProceedings", 458, "amount"),
        ),
        (
            "report.relatedCompanies[2].parentOrganizations[0].inn",
            ("report", "relatedCompanies", 2, "parentOrganizations", 0, "inn"),
        ),
        ("_id.ogrn", ("_id", "ogrn")),
    ],
)
def test_parse_column(name: str, expected: tuple[str | int, ...]) -> None:
    assert parse_column(name) == expected


@pytest.mark.parametrize("bad", ["", "report..inn", "report.items[x]", "report.items[0"])
def test_parse_column_rejects_garbage(bad: str) -> None:
    with pytest.raises(ColumnNameError):
        parse_column(bad)


def test_row_to_tree_is_driven_by_names_not_positions() -> None:
    # колонки нарочно в «неправильном» порядке: [1] раньше [0], поле за полем
    row = {
        "report.cofounders[1].name": "Б",
        "report.cofounders[0].name": "А",
        "report.cofounders[1].inn": "0278949271",
        "report.cofounders[0].inn": "",
        "report.baseInfo.inn": "052500690823",
        "report.kinds[0].code": "31.0",
    }
    assert row_to_tree(row) == {
        "report": {
            "cofounders": [{"name": "А"}, {"name": "Б", "inn": "0278949271"}],
            "baseInfo": {"inn": "052500690823"},
            "kinds": [{"code": "31.0"}],
        }
    }


def test_row_to_tree_fails_on_index_gap() -> None:
    with pytest.raises(IndexGapError):
        row_to_tree({"report.items[0].a": "x", "report.items[2].a": "y"})


def test_presence_policy_adds_only_always_present_sections() -> None:
    out = apply_presence_policy({"baseInfo": {"inn": "1"}})
    assert out["executionProceedings"] == []
    assert out["phones"] == []
    assert out["procurements"] == []
    assert out["arbitrationByStatus"] == {}
    assert out["reputationalRisks"] == {"negative": [], "positive": []}
    assert "licenses" not in out and "finReports" not in out
    # существующее не трогаем
    kept = apply_presence_policy({"reputationalRisks": {"positive": [{"code": "x"}]}})
    assert kept["reputationalRisks"] == {"positive": [{"code": "x"}], "negative": []}


def _collapsed_paths(value: Any, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            paths |= _collapsed_paths(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(value, list):
        for v in value:
            paths |= _collapsed_paths(v, prefix + "[]")
    else:
        paths.add(prefix)
    return paths


def test_snapshot_csv_unflattens_into_json_shaped_records(data_dir: Path) -> None:
    csv_path = data_dir / "contractors_audit.snapshot_C12613591.csv"
    rows = list(iter_rows(csv_path))
    assert len(rows) == 100
    assert all(len(r) == 2654 for r in rows)

    records = load_csv_records(csv_path)
    json_path = data_dir / "contractors_audit.snapshot.json"
    json_records = unwrap(json.loads(json_path.read_text("utf-8")))
    json_paths = _collapsed_paths(json_records)
    csv_paths = _collapsed_paths(records)
    assert csv_paths <= json_paths, csv_paths - json_paths

    for rec in records:
        assert set(rec) == {"_id", "report"}
        report = rec["report"]
        assert set(report) >= ALWAYS_PRESENT_LISTS | ALWAYS_PRESENT_DICTS
        assert {"negative", "positive"} <= set(report["reputationalRisks"])

    by_inn = {r["report"]["baseInfo"]["inn"]: r["report"] for r in records}
    assert len(by_inn) == 100
    assert "0277985654" in by_inn  # ведущий ноль пережил разворот
    esm = by_inn["7718083574"]  # АО «ЭНЕРГОСПЕЦМОНТАЖ»
    proceedings = esm["executionProceedings"]
    assert len(proceedings) == 459
    assert sum(p["active"] == "true" for p in proceedings) == 1
    assert all(isinstance(p["number"], str) for p in proceedings)
    # без ячеек в CSV → политика присутствия, а не «секции нет»
    verny = by_inn["5003135950"]  # ООО «ВЕРНЫЙ»: производств нет, reasonName есть
    assert verny["executionProceedings"] == []
    assert verny["status"]["reasonName"].startswith("Юридическим лицом принято решение")
