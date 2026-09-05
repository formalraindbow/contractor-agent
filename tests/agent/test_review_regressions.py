from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from contractor_agent.agent.citations import check_citation
from contractor_agent.agent.nodes import (
    drop_invalid_lines,
    offtopic_reply,
    tool_subset,
    visible_history,
)
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


@pytest.mark.parametrize(
    "claim,path",
    [
        ("Прибыль 26,2 млн ₽", "report.finReports[1].common.profit"),
        (
            "2 дела на 999 млрд ₽",
            "report.arbitrationByStatus.defandantArbitration.defandantArbitrationPending.dpCount",
        ),
        ("999 дел", "report.arbitrationByStatus"),
        ("Численность 2000 сотрудников", "report.baseInfo.staff"),
    ],
)
def test_rejects_false_numbers(snapshot, claim, path):
    assert not check_citation(snapshot, ["6165169320"], Citation(claim=claim, source_path=path)).ok


def test_comparison_citation_cannot_borrow_another_company(snapshot):
    c = Citation(
        claim="Прибыль 100 ₽", source_path="report.finReports[0].common.profit", inn="5032257375"
    )
    assert not check_citation(snapshot, ["6165169320"], c).ok
    assert not check_citation(
        snapshot, ["6165169320", "1684017097"], c.model_copy(update={"inn": None})
    ).ok


def test_rejects_denial_of_bankruptcy(snapshot):
    c = Citation(claim="Компания не является банкротом", source_path="report.status.reasonName")
    assert not check_citation(snapshot, ["5032257375"], c).ok


def test_rounding_and_exact_counts(snapshot):
    from contractor_agent.agent.citations import number_matches

    assert not number_matches(Decimal("26200000"), Decimal("-26249000"))
    assert check_citation(
        snapshot,
        ["6165169320"],
        Citation(claim="Убыток 26,2 млн ₽", source_path="report.finReports[1].common.profit"),
    ).ok
    assert check_citation(
        snapshot,
        ["5032257375"],
        Citation(
            claim="54 действующих производства", source_path="computed.enforcement.active.count"
        ),
    ).ok
    assert not check_citation(
        snapshot,
        ["5032257375"],
        Citation(
            claim="55 действующих производств", source_path="computed.enforcement.active.count"
        ),
    ).ok


def test_drop_invalid_literal_claim_without_marker():
    c = Citation(claim="Выручка 999 млн ₽", source_path="report.finReports[0].common.proceeds")
    assert "999" not in drop_invalid_lines("Выручка 999 млн ₽\nОстальное подтверждено.", [c])


def test_multi_topic_route_and_short_replies():
    names = tool_subset("Суды, выручка и долги у приставов")
    assert {"get_arbitration_summary", "get_financials", "get_enforcement_summary"} <= set(names)
    assert "get_risk_signals" in tool_subset("Долги по налогам")
    for q in ["ТЕХПРОФ", "Да", "Подробнее"]:
        assert offtopic_reply(q) is None


def test_history_uses_only_visible_answers():
    msgs = [
        HumanMessage("Проверь"),
        AIMessage("Неверный черновик"),
        HumanMessage("почини", additional_kwargs={"repair": True}),
        AIMessage("Проверенный ответ", additional_kwargs={"visible": True}),
        HumanMessage("Подробнее"),
    ]
    assert [m.content for m in visible_history(msgs, "Подробнее")] == [
        "Проверь",
        "Проверенный ответ",
        "Подробнее",
    ]


async def test_failed_new_search_clears_company_and_trace(snapshot, tmp_path):
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "a", inn="1684017097"),
            AIMessage("готово"),
            Draft(kind="answer", lines=["Компания ТЕХПРОФ."], citations=[]),
            tool_call("search_company", "b", query="Рога и копыта"),
            AIMessage("не найдено"),
            Draft(kind="refusal", lines=["Компания в базе не найдена."], citations=[]),
        ]
    )
    async with AgentRuntime(Settings(runs_dir=tmp_path), source=snapshot, llm=llm) as rt:
        first = await rt.ask("ТЕХПРОФ 1684017097", "same")
        second = await rt.ask("Проверь Рога и копыта", "same")
        state = (await rt.graph.aget_state({"configurable": {"thread_id": "same"}})).values
    assert first.report_dates
    assert second.card is None and not second.report_dates and not second.active_inns
    assert state["selected_inns"] == []
    assert [t.name for t in state["trace"]] == ["search_company"]
    assert state["messages"][-1].content == second.text_md


async def test_invalid_plain_number_is_not_published(snapshot, tmp_path):
    bad = Draft(kind="answer", lines=["Выручка 999 млн ₽"], citations=[])
    llm = scripted_llm(
        [
            tool_call("get_financials", "a", inn="1684017097"),
            AIMessage("готово"),
            bad,
            AIMessage("готово"),
            bad,
        ]
    )
    async with AgentRuntime(Settings(runs_dir=tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask("Выручка ТЕХПРОФ 1684017097")
    assert "999" not in answer.text_md
    assert answer.verification["removed_lines"] == 1


async def test_structured_failure_uses_retrieved_enforcement_facts(snapshot, tmp_path):
    llm = scripted_llm(
        [
            tool_call("get_enforcement_summary", "a", inn="2311304742"),
            AIMessage("готово"),
            None,
            None,
        ]
    )
    async with AgentRuntime(Settings(runs_dir=tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask(
            "Сколько БИЛД-ЮГ 2311304742 выплатит по действующему исполнительному производству?"
        )
    assert answer.kind == "answer"
    assert "сумма не указана" in answer.text_md
    assert "Повторите запрос" not in answer.text_md
    assert answer.citations
    assert all(check_citation(snapshot, ["2311304742"], c).ok for c in answer.citations)


def test_presentation_does_not_change_bank_labels_or_source_paths():
    from contractor_agent.agent.nodes import presentation_text

    assert (
        presentation_text("Вердикт банка: стоит проверить дополнительно.")
        == "Рекомендация по отчёту: стоит проверить дополнительно."
    )
    assert "null" not in presentation_text("Указано **null** — численность отсутствует.")
    assert presentation_text("Светофор банка: зелёный.") == "Светофор банка: зелёный."


def test_section_absence_can_use_retrieved_report_summary(snapshot):
    from contractor_agent.agent.section_answer import render_sections
    from contractor_agent.mcp_server.tools import Tools

    text, _ = render_sections(
        Tools(snapshot),
        "2311304742",
        "Нет раздела проверок — значит, не проверяли?",
        {"get_report_summary"},
    )
    assert "не означает" in text
    assert "get_section" not in text


def test_branch_section_fallback_uses_real_schema_name(snapshot):
    from contractor_agent.agent.section_answer import render_sections
    from contractor_agent.mcp_server.tools import Tools

    report = next(r for r in snapshot if r.report.branches_info is not None)
    text, _ = render_sections(Tools(snapshot), report.inn, "Какие есть филиалы?", {"get_section"})
    assert "присутствует" in text
    assert "нет доступных" not in text
