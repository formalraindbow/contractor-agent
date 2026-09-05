import pytest
from langchain_core.messages import AIMessage

from contractor_agent.agent.citations import check_citation
from contractor_agent.agent.nodes import build_card, render_card, tool_subset
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Draft
from contractor_agent.agent.section_answer import render_sections
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


def test_ip_overview_keeps_limitations_in_report_without_automatic_assignments(snapshot):
    tools = Tools(snapshot)
    card = build_card(tools, "772377037026")
    text, _ = render_card(card, tools)
    assert len(card.gaps) == 2  # Still available when opening the full report.
    assert any("численности" in gap for gap in card.gaps)
    assert any("бухгалтерской" in gap for gap in card.gaps)
    assert card.ask_before == []
    assert "Следующий шаг" not in text
    assert "Пробелы данных" not in text
    assert "численности" not in text and "бухгалтерской" not in text


def test_concise_overview_keeps_bankruptcy_and_a_justified_next_step(snapshot):
    tools = Tools(snapshot)
    card = build_card(tools, "5032257375")
    text, citations = render_card(card, tools)
    assert "банкрот" in text
    assert "Следующий шаг: Уточните текущий статус" in text
    assert "численности" not in text and "Пробелы данных" not in text
    assert all(check_citation(snapshot, [card.inn], citation).ok for citation in citations)


def test_gap_that_restricts_recommendation_remains_explained(snapshot):
    tools = Tools(snapshot)
    card = build_card(tools, "5029069967")
    text, citations = render_card(card, tools)
    assert "данных в нём нет" in text
    assert any(c.source_path == "report.finReports" for c in citations)
    assert all(check_citation(snapshot, [card.inn], citation).ok for citation in citations)


def test_direct_questions_keep_the_relevant_limitation_without_extra_advice(snapshot):
    tools = Tools(snapshot)
    staff, citations = render_sections(
        tools, "772377037026", "Сколько у него сотрудников?", {"get_report_summary"}
    )
    assert "нет сведений о численности" in staff
    assert "бухгалтерской" not in staff and "спросите" not in staff.lower()
    assert citations[0].source_path == "report.baseInfo.staff"
    finance, _ = render_sections(tools, "772377037026", "Какая выручка?", {"get_financials"})
    assert "нет бухгалтерской отчётности" in finance
    assert "численности" not in finance and "Запросите" not in finance
    assert tool_subset("Покажи финансовое положение") == [
        "search_company",
        "get_report_summary",
        "get_financials",
    ]


async def test_malformed_model_result_does_not_reintroduce_gap_boilerplate(snapshot, tmp_path):
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "summary", inn="772377037026"),
            tool_call("get_risk_signals", "risks", inn="772377037026"),
            AIMessage(content="Фактов достаточно."),
            {"bad_format": True},
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        answer = await runtime.ask("Проверь контрагента 772377037026")
    assert answer.kind == "card"
    assert len(answer.card.gaps) == 2
    assert "Следующий шаг" not in answer.text_md
    assert "численности" not in answer.text_md and "бухгалтерской" not in answer.text_md


async def test_status_followup_cannot_turn_into_the_same_general_card(snapshot, tmp_path):
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "summary", inn="343703064945"),
            AIMessage(content="Статус получен."),
            Draft(kind="card", lines=["Стоит проверить дополнительно. Суды, штат и отчётность."]),
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        answer = await runtime.ask("Какой у него статус? ИНН 343703064945")
    assert answer.kind == "answer" and answer.card is None
    assert "действующий" in answer.text_md
    assert "26.08.2026" in answer.text_md
    assert "Суды" not in answer.text_md and "штат" not in answer.text_md
    assert answer.invalid_citations == []


def test_status_explanation_takes_priority_over_generic_current_code(snapshot):
    tools = Tools(snapshot)
    text, citations = render_sections(
        tools, "5032257375", "Какой статус компании?", {"get_report_summary"}
    )
    assert "банкрот" in text and "действующий" not in text
    assert citations[0].source_path == "report.status.reasonName"


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["staff", "**staff**"])
async def test_staff_answer_omits_internal_field_names(snapshot, tmp_path, field):
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "summary", inn="343703064945"),
            AIMessage(content="Численность не указана."),
            Draft(kind="answer", lines=[f"В отчёте поле {field} не заполнено."]),
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        answer = await runtime.ask("Сколько у него сотрудников? ИНН 343703064945")
    assert answer.kind == "answer"
    assert "нет сведений о численности" in answer.text_md
    assert "staff" not in answer.text_md and "спросите" not in answer.text_md.lower()
