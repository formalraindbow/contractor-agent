import pytest
from langchain_core.messages import AIMessage

from contractor_agent.agent.nodes import asks_for_actions, build_card, render_card
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Draft
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Привет расскажи мне про проблемы компании МАКСМАРКЕТ", False),
        ("Проверь компанию перед оплатой", False),
        ("Почему у компании зелёный светофор?", False),
        ("Какие документы запросить?", True),
        ("Что мне делать дальше?", True),
        ("Как поступить с этим контрагентом?", True),
    ],
)
def test_action_request_is_distinct_from_a_fact_question(question, expected):
    assert asks_for_actions(question) is expected


async def test_company_problems_answer_does_not_append_advice_or_a_purpose_question(
    snapshot, tmp_path
):
    llm = scripted_llm(
        [
            tool_call("search_company", "find", query="МАКСМАРКЕТ"),
            tool_call("get_report_summary", "summary", inn="5032257375"),
            tool_call("get_risk_signals", "risks", inn="5032257375"),
            AIMessage(content="Проблемы компании получены."),
            {"bad_format": True},
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        answer = await runtime.ask("Привет расскажи мне про проблемы компании МАКСМАРКЕТ")
        state = await runtime.graph.aget_state({"configurable": {"thread_id": "cli"}})
    assert "банкрот" in answer.text_md
    assert "31.07.2026" in answer.text_md
    assert "Следующий шаг" not in answer.text_md
    assert "Уточните текущий статус" not in answer.text_md
    assert "Для чего нужна проверка" not in answer.text_md
    assert "?" not in answer.text_md
    assert state.values.get("pending_clarification") is None


async def test_old_purpose_prompt_is_retired_without_losing_selected_company(snapshot, tmp_path):
    tools = Tools(snapshot)
    card = build_card(tools, "5032257375")
    text, citations = render_card(card, tools)
    llm = scripted_llm(
        [
            tool_call("get_risk_signals", "risks", inn=card.inn),
            AIMessage(content="Факты получены."),
            Draft(kind="card", lines=text.splitlines(), citations=citations),
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        config = {"configurable": {"thread_id": "old-chat"}}
        await runtime.graph.aupdate_state(
            config,
            {"selected_inns": [card.inn], "pending_clarification": "purpose"},
        )
        answer = await runtime.ask("Какие у компании проблемы?", thread_id="old-chat")
        state = await runtime.graph.aget_state(config)
    assert state.values["selected_inns"] == [card.inn]
    assert state.values["pending_clarification"] is None
    assert "Для чего нужна проверка" not in answer.text_md
    context = llm.models[0].seen[0][0].content
    assert "Выбранные ИНН: 5032257375" in context
    assert "Ожидаемое уточнение: purpose" not in context


async def test_explicit_document_request_keeps_relevant_actions_in_fallback(snapshot, tmp_path):
    llm = scripted_llm(
        [
            tool_call("get_risk_signals", "risks", inn="5032257375"),
            AIMessage(content="Фактов достаточно."),
            {"bad_format": True},
        ]
    )
    async with AgentRuntime(
        Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
    ) as runtime:
        answer = await runtime.ask("Проверь МАКСМАРКЕТ 5032257375: какие документы запросить?")
    assert "### Что уточнить" in answer.text_md
    assert "свежей выписке ЕГРЮЛ" in answer.text_md
    assert "Для чего нужна проверка" not in answer.text_md
