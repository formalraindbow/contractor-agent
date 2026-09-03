from datetime import date

import pytest

from contractor_agent.data.loader import (
    NO_MATCH,
    CompanyRecord,
    DuplicateInnError,
    ReportSource,
    Snapshot,
    Source,
    normalize_name,
    rank,
    strip_legal_form,
)
from contractor_agent.data.model import Report


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('ООО "ЛЕ МОНЛИД"', "ооо ле монлид"),
        ("ООО «Ёлка»  плюс ", "ооо елка плюс"),
        ("ИП ЯНПОЛОВ И.А.", "ип янполов и.а."),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_strip_legal_form_keeps_last_word() -> None:
    assert strip_legal_form("ооо ле монлид") == "ле монлид"
    assert strip_legal_form("ано дпо уцгн") == "уцгн"
    assert strip_legal_form("ооо") == "ооо"


def test_rank_exact_prefix_contains() -> None:
    assert rank("ооо спорт", "спорт") == 0  # без орг. формы — точное
    assert rank("ооо спортмастер", "спорт") == 1
    assert rank("ооо мир спорта", "спорт") == 2
    assert rank("ооо лзсо", "спорт") == NO_MATCH
    assert rank("ип янполов и.а.", "ип янполова") == 3  # падеж


def test_snapshot_loads_200_from_two_disjoint_files(snapshot: Snapshot) -> None:
    assert len(snapshot) == 200
    assert sum(r.source is Source.JSON for r in snapshot) == 100
    assert sum(r.source is Source.CSV for r in snapshot) == 100
    assert isinstance(snapshot, ReportSource)
    assert snapshot.get("052500690823") is not None  # ведущий ноль
    assert snapshot.get("52500690823") is None  # без нуля — другой ИНН, его нет
    assert snapshot.record("0277985654").source is Source.CSV


def test_duplicate_inn_is_an_error(snapshot: Snapshot) -> None:
    record = next(iter(snapshot))
    with pytest.raises(DuplicateInnError):
        Snapshot([record, CompanyRecord(Source.CSV, record.report)])


def test_search_by_identifier(snapshot: Snapshot) -> None:
    (hit,) = snapshot.search("5029069967")
    assert hit.short_name == 'ООО "ЛЕ МОНЛИД"' and hit.source is Source.JSON
    assert hit.risk_level == "UNKNOWN" and hit.zsk_risk_level == "GREEN"
    assert hit.report_date == date(2026, 7, 30)
    (by_ogrn,) = snapshot.search(hit.ogrn)
    assert by_ogrn.inn == "5029069967"
    assert snapshot.search("0000000000") == []
    assert snapshot.search("   ") == []


def test_search_by_name(snapshot: Snapshot) -> None:
    assert [h.inn for h in snapshot.search("монлид")] == ["5029069967"]
    assert [h.inn for h in snapshot.search('ООО "ле монлид"')] == ["5029069967"]
    lzso = snapshot.search("ЛЗСО")
    assert {h.inn for h in lzso} == {"7805327192", "4720028039"}  # два «ЛЗСО» в JSON
    omega = snapshot.search("омега")
    assert {h.source for h in omega} == {Source.JSON, Source.CSV}  # по одной в каждом файле
    assert snapshot.search("спорт")[0].inn == "9705152496"  # точное — раньше подстроки
    assert len(snapshot.search("ооо", limit=3)) == 3


def test_report_source_protocol_is_structural() -> None:
    class Fake:
        def get(self, inn: str) -> Report | None:
            return None

        def search(self, query: str, limit: int = 5) -> list:
            return []

    assert isinstance(Fake(), ReportSource)


def test_search_tolerates_declension(snapshot):
    hits = snapshot.search("ИП Янполова")
    assert hits and hits[0].inn == "052500690823"
