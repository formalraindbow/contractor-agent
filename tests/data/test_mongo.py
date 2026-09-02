import json
from pathlib import Path
from typing import Any

import pytest

from contractor_agent.data.mongo import UnknownWrapperError, unwrap


def test_unwraps_number_long_and_date_at_any_depth() -> None:
    raw = {
        "a": {"$numberLong": "4534783044"},
        "b": [{"$date": "2011-07-05T20:00:00.000Z"}, 7, "31.0"],
        "c": {"d": {"e": {"$numberLong": "-5"}}},
        "empty": {},
        "flag": True,
    }
    assert unwrap(raw) == {
        "a": 4534783044,
        "b": ["2011-07-05T20:00:00.000Z", 7, "31.0"],
        "c": {"d": {"e": -5}},
        "empty": {},
        "flag": True,
    }


def test_input_is_not_mutated() -> None:
    raw = {"a": {"$numberLong": "1"}}
    unwrap(raw)
    assert raw == {"a": {"$numberLong": "1"}}


@pytest.mark.parametrize("bad", [{"$oid": "abc"}, {"$date": {"$numberLong": "-1"}}])
def test_unknown_wrapper_fails_loudly(bad: dict[str, Any]) -> None:
    with pytest.raises(UnknownWrapperError):
        unwrap({"x": bad})


def _count_wrappers(value: Any, key: str) -> int:
    if isinstance(value, dict):
        if list(value) == [key]:
            return 1
        return sum(_count_wrappers(v, key) for v in value.values())
    if isinstance(value, list):
        return sum(_count_wrappers(v, key) for v in value)
    return 0


def _has_dollar_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(k.startswith("$") or _has_dollar_key(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_dollar_key(v) for v in value)
    return False


def test_snapshot_json_has_only_known_wrappers_and_all_are_removed(data_dir: Path) -> None:
    records = json.loads((data_dir / "contractors_audit.snapshot.json").read_text("utf-8"))
    assert len(records) == 100
    # контракт на снапшот: ровно столько обёрток, и никаких других
    assert _count_wrappers(records, "$numberLong") == 67
    assert _count_wrappers(records, "$date") == 4672
    plain = unwrap(records)
    assert not _has_dollar_key(plain)

    by_inn = {r["report"]["baseInfo"]["inn"]: r["report"] for r in plain}
    le_monlid = by_inn["5029069967"]
    assert le_monlid["arbitrationByStatus"]["commonAmount"] == 4534783044
    sport = by_inn["9705152496"]
    assert sport["finReports"][2]["liabilities"]["capitals"] == 2540684000
    # дата осталась строкой — её разбирает модель
    assert le_monlid["reportDate"] == "2026-07-29T21:00:00.000Z"
