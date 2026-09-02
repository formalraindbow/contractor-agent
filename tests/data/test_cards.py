"""Карточки из docs/DATA_ANALYSIS.md §6 — контракт на факты, которые увидят сигналы и эвалы."""

from datetime import date

from contractor_agent.data.loader import Snapshot, Source


def test_maxmarket_bankrupt_with_green_labels(snapshot: Snapshot) -> None:
    r = snapshot.get("5032257375")
    assert (r.base_info.risk_level, r.zsk_risk_level, r.status.status) == (
        "LOW",
        "GREEN",
        "CURRENT",
    )
    assert "банкрот" in r.status.reason_name
    assert len(r.execution_proceedings) == 507
    assert sum(p.active for p in r.execution_proceedings) == 54
    assert r.arbitration_by_status.common_count == 278
    assert r.arbitration_cases and any(
        c.defendant_amount == 2589790444 for c in r.arbitration_cases
    )


def test_rado_alatyr_exclusion_with_zero_negatives(snapshot: Snapshot) -> None:
    r = snapshot.get("2100006761")
    assert (r.base_info.risk_level, r.zsk_risk_level) == ("LOW", "GREEN")
    assert r.reputational_risks.negative == []
    assert "исключении" in r.status.reason_name


def test_yanpolov_labels_contradict(snapshot: Snapshot) -> None:
    r = snapshot.get("052500690823")
    assert (r.base_info.risk_level, r.zsk_risk_level) == ("HIGH", "GREEN")
    assert r.base_info.short_name.startswith("ИП")
    assert r.founders_info is None and r.base_info.kpp is None


def test_taifun_branches_and_skv_licenses(snapshot: Snapshot) -> None:
    taifun = snapshot.get("5261137785")
    assert taifun.branches_info and taifun.branches_info.branches
    assert taifun.section_state("branchesInfo") == "present"
    skv = snapshot.get("7816085851")
    assert (
        skv.licenses and skv.licenses[0].issue_date and skv.section_state("licenses") == "present"
    )


def test_report_dates_are_moscow_calendar(snapshot: Snapshot) -> None:
    dates = {r.report.report_date for r in snapshot}
    assert min(dates) == date(2026, 7, 30) and max(dates) == date(2026, 8, 28)
    assert len(dates) == 28


def test_six_reason_names_all_current(snapshot: Snapshot) -> None:
    flagged = [r for r in snapshot if r.report.status.reason_name]
    assert len(flagged) == 6
    assert all(r.report.status.status == "CURRENT" for r in flagged)
    assert sum(r.report.zsk_risk_level == "GREEN" for r in flagged) == 5
    assert {r.source for r in flagged} == {Source.JSON, Source.CSV}


def test_csv_cards(snapshot: Snapshot) -> None:
    esm = snapshot.get("7718083574")  # АО «ЭНЕРГОСПЕЦМОНТАЖ»
    assert (esm.base_info.risk_level, esm.zsk_risk_level) == ("UNKNOWN", "GREEN")
    assert len(esm.execution_proceedings) == 459
    assert sum(p.active for p in esm.execution_proceedings) == 1
    assert "inspectionWithViolation" in {f.code for f in esm.reputational_risks.negative}

    gss = snapshot.get("2308177290")  # ООО «ГЛАВСВЯЗЬСТРОЙ»
    assert (gss.base_info.risk_level, gss.zsk_risk_level) == ("LOW", "GREEN")
    assert "исключении" in gss.status.reason_name
    assert "fnsBlocking" in {f.code for f in gss.reputational_risks.negative}
    assert sum(p.active for p in gss.execution_proceedings) == 9


def test_negative_codes_vocabulary_is_open(snapshot: Snapshot) -> None:
    negative = {f.code for r in snapshot for f in r.report.reputational_risks.negative}
    assert len(negative) == 15
    assert {"dishonestProvider", "taxArrears", "inspectionWithViolation"} <= negative
    positive = {f.code for r in snapshot for f in r.report.reputational_risks.positive}
    assert "arbitrationDefendant" in positive and all(c.isascii() for c in positive)
