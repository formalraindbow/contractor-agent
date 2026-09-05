from contractor_agent.agent.evidence import resolve_evidence
from contractor_agent.data.model import ExecutionProceeding
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.engine import compute


def test_missing_long_term_debt_does_not_become_zero(snapshot):
    rows = Tools(snapshot).get_financials("1684017097").data["years"]
    row = next(r for r in rows if r["year"] == 2025)
    assert row["long_term_duties"] is None
    assert row["sustainability"] is None


def test_no_date_and_unknown_status_do_not_crash_or_claim_empty(snapshot):
    report = snapshot.get("1684017097").model_copy(deep=True)
    report.execution_proceedings = [ExecutionProceeding(active=True, amount=100)]

    class Source:
        def get(self, inn):
            return report

    tools = Tools(Source())
    assert tools.get_enforcement_summary(report.inn).data["active"]["count"] == 1
    report.execution_proceedings = [ExecutionProceeding(amount=100)]
    response = tools.get_enforcement_summary(report.inn)
    assert response.data["unknown_status_count"] == 1
    assert response.note is None


def test_partial_financial_statement_has_resolvable_gaps(snapshot):
    report = snapshot.get("1684017097").model_copy(deep=True)
    for f in report.fin_reports:
        f.liabilities = None
        f.assets = None
    assert compute(report).gaps


def test_every_tool_path_resolves_across_snapshot(snapshot):
    tools = Tools(snapshot)

    def paths(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if k in ("path", "source_path", "count_path", "amount_path") and isinstance(v, str):
                    yield v
                elif k in ("source_paths", "paths"):
                    values = v.values() if isinstance(v, dict) else v
                    for val in values:
                        yield from val if isinstance(val, list) else [val]
                else:
                    yield from paths(v)
        elif isinstance(data, list):
            for value in data:
                yield from paths(value)

    # All reports and every exposed nested metric path, including null-parent gaps.
    assert len(snapshot) == 200
    for hit in snapshot:
        for name in (
            "get_report_summary",
            "get_risk_signals",
            "get_financials",
            "get_enforcement_summary",
            "get_arbitration_summary",
        ):
            response = getattr(tools, name)(hit.inn)
            for path in [*response.source_paths, *paths(response.data)]:
                resolve_evidence(snapshot, hit.inn, path)


def test_terminal_reason_is_not_fabricated(snapshot):
    from datetime import date

    from contractor_agent.signals.model import SignalSet, Verdict, recommendation_text

    ss = SignalSet(
        inn="1684017097",
        report_date=date(2026, 8, 28),
        verdict=Verdict.NOT_RECOMMENDED,
        score=0,
        terminal=True,
    )
    assert "банкротств" not in recommendation_text(ss)


def test_section_fallback_facts_are_independently_resolvable(snapshot):
    from contractor_agent.agent.citations import check_citation
    from contractor_agent.agent.section_answer import render_sections

    tools = Tools(snapshot)
    for inn, question, called in [
        (
            "6165169320",
            "Выручка, суды и приставы",
            {"get_financials", "get_arbitration_summary", "get_enforcement_summary"},
        ),
        ("2311304742", "Если нет проверок, значит не проверяли?", {"get_section"}),
        ("2100006761", "Сколько сотрудников?", {"get_report_summary"}),
    ]:
        text, citations = render_sections(tools, inn, question, called)
        assert text and citations
        assert all(check_citation(snapshot, [inn], c).ok for c in citations), citations


def test_computed_source_shows_original_operands(snapshot):
    from contractor_agent.agent.evidence import evidence_basis

    basis = evidence_basis(snapshot, "1684017097", "computed.financials.years[0].current_liquidity")
    values = {b["path"]: b["value"] for b in basis}
    assert "report.finReports[0].assets.currentAssets.total" in values
    assert "report.finReports[0].liabilities.shortTermLiabilities.total" in values
