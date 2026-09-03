from collections import Counter

import pytest

from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import Report
from contractor_agent.signals.engine import compute, decide
from contractor_agent.signals.model import Severity, Verdict


@pytest.mark.parametrize(
    ("terminal", "critical", "moderate", "risk", "floor", "expected"),
    [
        (True, False, False, "LOW", False, Verdict.NOT_RECOMMENDED),
        (False, True, False, "LOW", False, Verdict.NOT_RECOMMENDED),  # любой критичный → условия
        (False, False, True, "LOW", False, Verdict.CHECK),
        (False, False, False, "LOW", False, Verdict.OK),
        (False, False, False, "HIGH", False, Verdict.CHECK),  # пол по светофору
        (False, False, False, "LOW", True, Verdict.CHECK),  # пол по пробелу
    ],
)
def test_decide(terminal, critical, moderate, risk, floor, expected) -> None:
    assert (
        decide(
            terminal=terminal,
            critical=critical,
            moderate=moderate,
            risk_level=risk,
            floor_check=floor,
        )
        is expected
    )


DEMO = {
    "5032257375": Verdict.NOT_RECOMMENDED,  # МАКСМАРКЕТ — банкрот при LOW/GREEN
    "2100006761": Verdict.NOT_RECOMMENDED,  # РАДО — исключение из ЕГРЮЛ, негативов ноль
    "2308177290": Verdict.NOT_RECOMMENDED,  # ГЛАВСВЯЗЬСТРОЙ (CSV) — исключение
    "5029069967": Verdict.NOT_RECOMMENDED,  # ЛЕ МОНЛИД — 45 производств, 20 исков
    "6165169320": Verdict.NOT_RECOMMENDED,  # ГДК — иски в 5 раз больше активов, блокировка
    "2311304742": Verdict.NOT_RECOMMENDED,  # БИЛД-ЮГ — 6 признаков при LOW/GREEN
    "052500690823": Verdict.CHECK,  # ЯНПОЛОВ — HIGH/GREEN, одна блокировка
    "7722608440": Verdict.CHECK,  # РЕМСТРОЙ М — пример Дениса «должник ФНС»
    "1684017097": Verdict.OK,  # ТЕХПРОФ — чистая
    "772377037026": Verdict.OK,  # КАСАТКИН — ИП без данных, пробелы
    "0277985654": Verdict.OK,  # УЦГН (CSV) — только завершённые производства
}


def test_demo_cards_land_where_expected(snapshot: Snapshot) -> None:
    got = {inn: compute(snapshot.get(inn)).verdict for inn in DEMO}
    mismatches = {inn: (got[inn], DEMO[inn]) for inn in DEMO if got[inn] is not DEMO[inn]}
    assert not mismatches, mismatches


def test_maxmarket_is_terminal_and_sorted(snapshot: Snapshot) -> None:
    ss = compute(snapshot.get("5032257375"))
    assert ss.terminal and ss.verdict is Verdict.NOT_RECOMMENDED
    assert [s.severity for s in ss.signals] == sorted(
        (s.severity for s in ss.signals), key=lambda x: x.rank
    )
    assert ss.by_severity(Severity.CRITICAL)[0].terminal


def test_label_floor_only_on_svetofor(snapshot: Snapshot) -> None:
    raw = snapshot.get("1684017097").model_dump()
    raw["zskRiskLevel"] = "RED"
    assert compute(Report.model_validate(raw)).verdict is Verdict.OK  # ЗСК не участвует
    raw["baseInfo"]["riskLevel"] = "HIGH"
    assert compute(Report.model_validate(raw)).verdict is Verdict.CHECK


def test_distribution_over_200_is_stable(snapshot: Snapshot) -> None:
    verdicts = Counter(compute(r.report).verdict for r in snapshot)
    assert sum(verdicts.values()) == 200
    assert verdicts[Verdict.NOT_RECOMMENDED] < verdicts[Verdict.CHECK]
    assert verdicts[Verdict.OK] >= 60  # большинство зелёных карточек не должны тонуть в «проверить»
    assert verdicts[Verdict.NOT_RECOMMENDED] >= 29  # любой критичный → условия
