from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import Report
from contractor_agent.signals import flags
from contractor_agent.signals.model import GapReason, Origin, Severity


def test_build_yug_flags_with_bank_text_as_data(snapshot: Snapshot) -> None:
    res = flags.run(snapshot.get("2311304742"))
    by_code = {s.code: s for s in res.signals}
    assert {"flag_invalidAddress", "flag_invalidRegistrationData", "flag_fnsBlocking"} <= set(
        by_code
    )
    assert "flag_executionProceedings" not in by_code and "flag_arbitrationDefendant" not in by_code
    bad_address = by_code["flag_invalidAddress"]
    assert bad_address.severity is Severity.CRITICAL and bad_address.origin is Origin.BANK_FLAG
    assert "Не рекомендуется" in bad_address.bank_text  # текст банка — данные
    assert "Не рекомендуется" not in bad_address.explanation_ru  # наше объяснение — мягкое
    assert bad_address.source_path.startswith("report.reputationalRisks.negative[")


def test_severity_map_matches_case_owner(snapshot: Snapshot) -> None:
    yanpolov = {s.code: s for s in flags.run(snapshot.get("052500690823")).signals}
    assert yanpolov["flag_fnsBlocking"].severity is Severity.MODERATE
    assert "снята ли" in yanpolov["flag_fnsBlocking"].explanation_ru
    remstroy = {s.code: s for s in flags.run(snapshot.get("7722608440")).signals}
    assert remstroy["flag_taxArrears"].severity is Severity.MODERATE
    maxmarket = {s.code: s for s in flags.run(snapshot.get("5032257375")).signals}
    assert "flag_liquidationStatus" not in maxmarket  # тот же факт уже есть из реестра
    raw = snapshot.get("5032257375").model_dump()
    raw["status"].pop("reasonName")
    raw["foundersInfo"]["authPerson"]["positionName"] = "ДИРЕКТОР"
    alone = {s.code: s for s in flags.run(Report.model_validate(raw)).signals}
    assert alone["flag_liquidationStatus"].terminal  # без сырой опоры флаг остаётся
    counts = {}
    for rec in snapshot:
        for s in flags.run(rec.report).signals:
            counts[s.code] = counts.get(s.code, 0) + 1
    assert counts["flag_massOkved"] == 45 and counts["flag_fnsBlocking"] == 18
    assert counts["flag_massAddress"] == 24 and counts["flag_invalidAddress"] == 5


def test_unknown_code_is_moderate_and_absent_section_is_a_gap(snapshot: Snapshot) -> None:
    raw = snapshot.get("1684017097").model_dump()
    raw["reputationalRisks"]["negative"] = [{"code": "brandNewCode", "name": "x", "chapter": "y"}]
    (unknown,) = flags.run(Report.model_validate(raw)).signals
    assert unknown.code == "flag_brandNewCode" and unknown.severity is Severity.MODERATE
    raw.pop("reputationalRisks")
    res = flags.run(Report.model_validate(raw))
    assert res.signals == [] and res.gaps[0].reason is GapReason.SECTION_ABSENT
    assert res.gaps[0].criterion == "реестры ФНС"


def test_negative_flags_survive_incomplete_raw_sections(snapshot):
    raw = snapshot.get("1684017097").model_dump()
    raw["reputationalRisks"]["negative"] = [
        {"code": "profit", "name": "Убыток по признаку банка"},
        {"code": "arbitrationDefendant", "name": "Иски к компании"},
    ]
    for row in raw["finReports"]:
        if row.get("common"):
            row["common"]["profit"] = None
    raw["arbitrationByStatus"] = {"commonCount": 0}
    raw["arbitrationCases"] = []
    report = Report.model_validate(raw)
    codes = {s.code for s in flags.run(report).signals}
    assert {"flag_profit", "flag_arbitrationDefendant"} <= codes
