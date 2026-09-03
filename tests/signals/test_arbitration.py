from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import Report
from contractor_agent.signals import arbitration
from contractor_agent.signals.model import GapReason, Severity, check_paths


def _by_code(res) -> dict:
    return {s.code: s for s in res.signals}


def test_gdk_open_claims_exceed_net_assets(snapshot: Snapshot) -> None:
    res = _by_code(arbitration.run(snapshot.get("6165169320")))
    open_ = res["arbitration_defendant_open"]
    assert open_.severity is Severity.CRITICAL
    assert open_.value["open_count"] == 2 and open_.value["known_sum"] == 18_894_669
    assert "в 5 раз больше чистых активов" in open_.explanation_ru
    assert open_.source_path.endswith("defandantArbitrationPending.dpCount")  # адрес — как в отчёте
    assert res["arbitration_defendant_finished"].severity is Severity.INFO


def test_dsa_defendant_visible_only_in_yearly_breakdown(snapshot: Snapshot) -> None:
    res = _by_code(arbitration.run(snapshot.get("7838104498")))
    assert set(res) == {"arbitration_defendant_by_years", "arbitration_mismatch_roles"}
    by_years = res["arbitration_defendant_by_years"]
    assert by_years.severity is Severity.MODERATE and by_years.value["defendant_count"] == 1
    assert "за 2026 год" in by_years.explanation_ru
    assert by_years.source_paths == ["report.arbitrationCases[0].defendantCount"]


def test_marba_breakdown_without_aggregate_is_only_a_note(snapshot: Snapshot) -> None:
    res = _by_code(arbitration.run(snapshot.get("2312263270")))
    assert set(res) == {"arbitration_mismatch_no_summary"}
    assert res["arbitration_mismatch_no_summary"].severity is Severity.INFO


def test_le_monlid_counts_not_lists_and_window_is_not_a_mismatch(snapshot: Snapshot) -> None:
    res = _by_code(arbitration.run(snapshot.get("5029069967")))
    open_ = res["arbitration_defendant_open"]
    assert open_.severity is Severity.CRITICAL  # 20 ≥ 10 при отсутствующей отчётности
    assert (open_.value["pending_count"], open_.value["appealed_count"]) == (17, 3)
    assert "соразмерность долга оценить нельзя" in open_.explanation_ru
    assert "таких дел 20 — много само по себе" in open_.explanation_ru
    assert (
        res["arbitration_defendant_finished"].severity is Severity.INFO
    )  # история, сравнить не с чем
    mismatch = res["arbitration_mismatch_roles"]
    assert mismatch.value == {"common_count": 1525, "with_roles": 1488}
    assert len(open_.source_paths) <= 8


def test_finished_with_positive_bank_flag_and_unknown_open_sum(snapshot: Snapshot) -> None:
    tsk = _by_code(arbitration.run(snapshot.get("8602236576")))
    finished = tsk["arbitration_defendant_finished"]
    assert (
        finished.severity is Severity.INFO
        and "признак банка по этому пункту положительный" in finished.explanation_ru
    )
    assert finished.bank_text and "Не найдены" in finished.bank_text
    build_yug = _by_code(arbitration.run(snapshot.get("2311304742")))
    open_ = build_yug["arbitration_defendant_open"]
    assert (
        open_.severity is Severity.MODERATE and "сумма в отчёте не указана" in open_.explanation_ru
    )
    assert open_.value["known_sum"] == 0 and open_.value["unknown_amount_count"] == 2


def test_absent_sections_are_a_gap(snapshot: Snapshot) -> None:
    raw = snapshot.get("1684017097").model_dump()
    raw.pop("arbitrationByStatus")
    raw.pop("arbitrationCases", None)
    res = arbitration.run(Report.model_validate(raw))
    assert res.signals == [] and res.gaps[0].reason is GapReason.SECTION_ABSENT
    # пустой агрегат — не пробел и не сигнал
    assert arbitration.run(snapshot.get("1684017097")) == ([], [])


def test_bank_flag_is_subset_of_ours_and_paths_resolve(snapshot: Snapshot) -> None:
    bank, ours, severities = set(), set(), []
    for rec in snapshot:
        res = arbitration.run(rec.report)
        for item in (*res.signals, *res.gaps):
            check_paths(rec.report, item)
        if any(f.code == "arbitrationDefendant" for f in rec.report.reputational_risks.negative):
            bank.add(rec.inn)
        for s in res.signals:
            if s.code.startswith("arbitration_defendant"):
                ours.add(rec.inn)
            if s.code == "arbitration_defendant_open":
                severities.append(s.severity)
    assert bank <= ours and len(bank) == 67 and len(ours) == 82  # мы строже банка на 15
    assert sorted(severities.count(s) for s in Severity) == [3, 9, 17]
