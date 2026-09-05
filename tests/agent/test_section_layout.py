from langchain_core.messages import AIMessage

from contractor_agent.agent.citations import check_citation
from contractor_agent.agent.nodes import uncovered_numeric_lines
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Draft
from contractor_agent.agent.section_answer import needs_section_layout, render_sections
from contractor_agent.data.loader import CompanyRecord, Snapshot, Source
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call

INN = "343703064945"
QUESTION = "Судебные дела и производства у приставов"
CALLED = {"get_arbitration_summary", "get_enforcement_summary"}


def test_court_and_enforcement_blocks_preserve_every_numeric_source(snapshot):
    text, citations = render_sections(Tools(snapshot), INN, QUESTION, CALLED)
    assert [line for line in text.splitlines() if line.startswith("### ")] == [
        "### Иски к контрагенту (ответчик)",
        "### Иски контрагента (истец)",
        "### Производства у приставов",
    ]
    assert "- В производстве: **4 дела**; сумма требований — **92 992 ₽**." in text
    assert "- Завершены: **57 дел**; сумма требований — **128 млн ₽**." in text
    assert "- Действующие: **0 производств**." in text
    assert (
        "- Завершённые: **3 производства**; известная сумма — **не менее 13 438 ₽**. "
        "В 1 производстве сумма не указана."
    ) in text
    assert len([line for line in text.splitlines() if line.startswith("- ")]) == 7
    assert len(citations) == 14
    assert all(check_citation(snapshot, [INN], c).ok for c in citations)
    assert uncovered_numeric_lines(text, citations) == []


def test_zero_claim_amount_is_not_misreported_as_missing(snapshot):
    report = snapshot.get(INN).model_copy(deep=True)
    defendant = report.arbitration_by_status.defandant_arbitration
    defendant.defandant_arbitration_pending.dp_amount = 0
    defendant.defandant_arbitration_finished.df_amount = None
    source = Snapshot([CompanyRecord(Source.JSON, report)])
    text, citations = render_sections(Tools(source), INN, "Судебные дела", CALLED)
    assert "- В производстве: **4 дела**; сумма требований — **0 ₽**." in text
    assert "- Завершены: **7 дел**; сумма требований не указана." in text
    assert all(check_citation(source, [INN], c).ok for c in citations)
    assert uncovered_numeric_lines(text, citations) == []


def test_financial_blocks_keep_periods_and_loss_sign(snapshot):
    inn = "6165169320"
    text, citations = render_sections(
        Tools(snapshot), inn, "Покажи финансовое положение", {"get_financials"}
    )
    assert "### Финансы" in text
    assert "**2025 год**" in text and "**2024 год**" in text
    assert "- Убыток — **26,2 млн ₽**." in text
    assert all(check_citation(snapshot, [inn], c).ok for c in citations)
    assert uncovered_numeric_lines(text, citations) == []
    assert not needs_section_layout("Выручка за 2025 год — 1 млн ₽.", "Какая выручка?")


async def test_valid_but_fragmented_model_draft_is_grouped_without_another_llm_call(
    snapshot, tmp_path
):
    _, citations = render_sections(Tools(snapshot), INN, QUESTION, CALLED)
    # Every value is correct, but repeating one fact per paragraph is hard to read.
    flat = Draft(kind="answer", lines=[c.claim for c in citations], citations=citations)
    llm = scripted_llm(
        [
            tool_call("get_arbitration_summary", "courts", inn=INN),
            tool_call("get_enforcement_summary", "enforcement", inn=INN),
            AIMessage(content="Факты получены."),
            flat,
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        answer = await runtime.ask(f"{QUESTION}, ИНН {INN}")
    assert answer.kind == "answer"
    assert answer.text_md.count("### ") == 3
    assert answer.text_md.count("Отчёт от 26.08.2026") == 1
    assert "**4 дела**; сумма требований — **92 992 ₽**" in answer.text_md
    assert answer.invalid_citations == []
    assert len(answer.citations) == 14
