from pathlib import Path

import pytest

from contractor_agent.data.index import SqliteSource, build_index
from contractor_agent.data.loader import ReportSource, Snapshot


def test_index_has_all_companies_and_reports_round_trip(
    snapshot: Snapshot, index_path: Path
) -> None:
    source = SqliteSource(index_path)
    assert isinstance(source, ReportSource)
    assert len(source) == 200
    for record in snapshot:
        assert source.get(record.inn) == record.report, record.inn  # Decimal и даты пережили JSON
    assert source.get("0000000000") is None


def test_index_keeps_section_states(snapshot: Snapshot, index_path: Path) -> None:
    source = SqliteSource(index_path)
    assert source.sections("5029069967") == snapshot.get("5029069967").sections()
    assert source.sections("5029069967")["finReports"] == "empty"
    assert source.sections("772377037026")["finReports"] == "absent"


@pytest.mark.parametrize(
    "query",
    ["монлид", "ЛЗСО", "омега", "спорт", "5029069967", "052500690823", "ооо", "нет такой", "  "],
)
def test_search_matches_in_memory_snapshot(
    snapshot: Snapshot, index_path: Path, query: str
) -> None:
    assert SqliteSource(index_path).search(query) == snapshot.search(query)


def test_search_escapes_like_wildcards(index_path: Path) -> None:
    assert SqliteSource(index_path).search("%") == []
    assert SqliteSource(index_path).search("_") == []


def test_rebuild_is_idempotent_and_missing_index_is_loud(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    path = tmp_path / "x.sqlite"
    with pytest.raises(FileNotFoundError):
        SqliteSource(path)
    assert build_index(snapshot, path) == 200
    assert build_index(snapshot, path) == 200
    assert len(SqliteSource(path)) == 200
    assert not path.with_suffix(".sqlite.tmp").exists()


def test_sqlite_search_tolerates_declension(index_path):
    from contractor_agent.data.index import SqliteSource

    hits = SqliteSource(index_path).search("ИП Янполова")
    assert hits and hits[0].inn == "052500690823"
