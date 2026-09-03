import json

from contractor_agent.data.loader import Snapshot
from contractor_agent.data.model import SECTIONS
from contractor_agent.data.paths import resolve
from contractor_agent.mcp_server.envelope import ITEM_LIMIT, truncate_deep
from contractor_agent.mcp_server.tools import Tools


def test_truncate_deep_marks_long_lists() -> None:
    tree = {"a": list(range(25)), "b": {"c": list(range(3))}, "d": [{"e": list(range(21))}]}
    out = truncate_deep(tree, 20)
    assert out["a"] == {"items": list(range(20)), "total": 25, "truncated": True}
    assert out["b"]["c"] == [0, 1, 2]
    assert out["d"][0]["e"]["total"] == 21


def test_search_and_summary(snapshot: Snapshot) -> None:
    tools = Tools(snapshot)
    hit = tools.search_company("лзсо").data
    assert hit["total"] == 2 and {h["inn"] for h in hit["items"]} == {"7805327192", "4720028039"}
    assert hit["items"][0]["zsk"] in {"зелёный", "серый"}  # жёлтый/красный не раскрываем
    summary = tools.get_report_summary("5029069967")
    assert summary.available and summary.report_date == "2026-07-30"
    assert summary.data["sections"]["finReports"] == "empty"
    assert summary.data["counts"]["execution_proceedings"] == 1744
    assert (
        summary.data["labels"]["svetofor"] == "серый" and summary.data["labels"]["zsk"] == "зелёный"
    )
    assert summary.data["labels"]["svetofor_path"] == "report.baseInfo.riskLevel"
    assert summary.data["staff"] is None  # поле есть в спецификации, в снапшоте нет
    missing = tools.get_report_summary("0000000000")
    assert not missing.available and missing.reason == "not_found"


def test_signals_and_financials(snapshot: Snapshot) -> None:
    tools = Tools(snapshot)
    rado = tools.get_risk_signals("2100006761").data
    assert rado["verdict"] == "not_recommended" and rado["terminal"]
    assert rado["signals"]["critical"][0]["terminal"]
    kasatkin = tools.get_financials("772377037026")
    assert not kasatkin.available and kasatkin.reason == "not_applicable"
    gdk = tools.get_financials("6165169320")
    years = {y["year"]: y for y in gdk.data["years"]}
    assert years[2024]["current_liquidity"] == 0.491 and years[2024]["profitability_pct"] == -106.16
    assert gdk.data["bank_coefficient"]["year"] == 2024
    assert gdk.data["unit"] == "руб."


def test_enforcement_and_arbitration_are_aggregates(snapshot: Snapshot) -> None:
    tools = Tools(snapshot)
    le_monlid = tools.get_enforcement_summary("5029069967")
    assert le_monlid.data["active"]["count"] == 45 and le_monlid.data["finished"]["count"] == 1699
    assert len(le_monlid.data["top_active"]) == 10 and le_monlid.data["total"] == 1744
    assert le_monlid.data["active"]["sum_is_lower_bound"]
    assert json.dumps(le_monlid.model_dump())  # JSON-нативно
    techprof = tools.get_arbitration_summary("1684017097")
    assert techprof.available and "не значит «нет»" in techprof.note
    assert techprof.data["defendant"]["pending"]["count"] == 0
    dsa = tools.get_arbitration_summary("7838104498").data
    assert (
        dsa["by_years"][0]["defendant_count"] == 1
        and dsa["by_years"][0]["path"] == "report.arbitrationCases[0]"
    )
    kasatkin = tools.get_enforcement_summary("772377037026")
    assert kasatkin.available and kasatkin.data["total"] == 0 and "не найдено" in kasatkin.note


def test_get_section_states(snapshot: Snapshot) -> None:
    tools = Tools(snapshot)
    ok = tools.get_section("5029069967", "executionProceedings")
    assert ok.data["content"]["total"] == 1744 and len(ok.data["content"]["items"]) == ITEM_LIMIT
    empty = tools.get_section("5029069967", "finReports")
    assert empty.available and empty.data["state"] == "empty" and empty.note
    absent = tools.get_section("772377037026", "licenses")
    assert not absent.available and absent.reason == "section_absent"
    bogus = tools.get_section("5029069967", "headcount")
    assert not bogus.available and bogus.reason == "unknown_section" and "licenses" in bogus.note
    okved = tools.get_section("5029069967", "kindsOfActivityInfo").data["content"]
    assert okved["otherKindsOfActivity"]["truncated"] is True  # вложенный список тоже обрезан


def test_compare_companies(snapshot: Snapshot) -> None:
    tools = Tools(snapshot)
    res = tools.compare_companies(["5032257375", "1684017097", "5032257375", "0000000000"])
    items = res.data["items"]
    assert [i["inn"] for i in items] == ["5032257375", "1684017097", "0000000000"]  # дубликат снят
    assert items[0]["verdict"] == "not_recommended" and items[0]["critical"]
    assert items[1]["verdict"] == "ok" and items[2]["available"] is False
    assert not tools.compare_companies([]).available


def test_every_tool_on_every_company_is_json_and_paths_resolve(snapshot: Snapshot) -> None:
    tools = Tools(snapshot)
    for rec in snapshot:
        inn = rec.inn
        responses = [
            tools.get_report_summary(inn),
            tools.get_risk_signals(inn),
            tools.get_financials(inn),
            tools.get_enforcement_summary(inn),
            tools.get_arbitration_summary(inn),
            *(tools.get_section(inn, name) for name in SECTIONS),
        ]
        for r in responses:
            json.dumps(r.model_dump())
            if r.available:
                for path in r.source_paths:
                    resolve(rec.report, path)
            else:
                assert r.reason and r.note


def test_financials_rows_carry_resolvable_metric_paths(snapshot):
    from contractor_agent.data.paths import resolve
    from contractor_agent.mcp_server.tools import Tools

    resp = Tools(snapshot).get_financials("2311304742")
    row = resp.data["years"][0]
    assert row["paths"]["proceeds"] == "report.finReports[0].common.proceeds"
    report = snapshot.get("2311304742")
    for value in row["paths"].values():
        for path in value if isinstance(value, list) else [value]:
            resolve(report, path)  # адрес существует
    assert "citing" in resp.data
