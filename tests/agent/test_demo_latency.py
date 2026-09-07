"""Live-demo regressions: known identity, real evidence, bounded gateway recovery."""

import time

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import RateLimitError

from contractor_agent.agent.interpretation import InterpretationPlan
from contractor_agent.agent.llm import LLM
from contractor_agent.agent.nodes import contextual_decision
from contractor_agent.agent.question import needs_interpretation
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


@pytest.mark.parametrize(
    "q",
    [
        "Стоит ли иметь с ними дело?",
        "Можно с ними сотрудничать?",
        "Стоит с ними иметь дело?",
        "С кем лучше работать?",
        "Кого ты бы выбрал?",
    ],
)
def test_pronominal_decision_keeps_ui_scope(q):
    assert contextual_decision(q + " Речь о компании ООО «МАКСМАРКЕТ», ИНН 5032257375.")
    assert needs_interpretation(q)


@pytest.mark.parametrize(
    "q",
    [
        "Проверь новую компанию",
        "Стоит ли работать с ГДК?",
        "Стоит ли работать с рога и копыта?",
        "Кого выбрать из ТЕХПРОФ и ГДК?",
    ],
)
def test_new_names_still_require_resolution(q):
    assert not contextual_decision(q + " Речь о компании ООО «МАКСМАРКЕТ», ИНН 5032257375.")


async def test_first_card_followup_uses_one_composition_and_real_mcp_reads(snapshot, tmp_path):
    opening = "По данным отчёта я бы не начинал сотрудничество: компания проходит банкротство."
    model = scripted_llm([InterpretationPlan(answer=opening, evidence_ids=["E4"])])
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory", llm_fast_known_company=True),
        source=snapshot,
        llm=model,
    ) as runtime:
        answer = await runtime.ask(
            "Стоит ли иметь с ними дело? Речь о компании ООО «МАКСМАРКЕТ», ИНН 5032257375.",
            "fresh-ui-chat",
        )
        state = await runtime.graph.aget_state({"configurable": {"thread_id": "fresh-ui-chat"}})
    assert answer.kind == "answer" and answer.text_md.startswith(opening)
    assert not answer.invalid_citations
    assert answer.report_dates == {"5032257375": "2026-07-31"}
    assert len(model.models[0].seen) == 1
    assert {t.name for t in state.values["trace"]} == {"get_report_summary", "get_risk_signals"}
    assert state.values["active_inns"] == ["5032257375"]


async def test_known_report_fact_still_validates_without_a_model_roundtrip(snapshot, tmp_path):
    model = scripted_llm([])
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory", llm_fast_known_company=True),
        source=snapshot,
        llm=model,
    ) as runtime:
        answer = await runtime.ask("Кто руководитель 5032257375?", "known-head")
    assert "Москвина Ирина Витальевна" in answer.text_md
    assert answer.citations and not answer.invalid_citations
    assert not model.models[0].seen


async def test_unknown_inn_is_not_answered_from_previous_ui_card(snapshot, tmp_path):
    model = scripted_llm([AIMessage(content="Такого отчёта нет.")])
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory", llm_fast_known_company=True),
        source=snapshot,
        llm=model,
    ) as runtime:
        answer = await runtime.ask(
            "Проверь 0000000000. Речь о компании МАКСМАРКЕТ, ИНН 5032257375.",
            "unknown",
        )
    assert answer.kind == "refusal" and "0000000000" in answer.text_md
    assert "МАКСМАРКЕТ" not in answer.text_md and not answer.report_dates


async def test_name_search_stays_enabled_in_fast_mode(snapshot, tmp_path):
    model = scripted_llm(
        [
            tool_call("search_company", "lookup", query="МАКСМАРКЕТ"),
            AIMessage(content="Компания найдена."),
        ]
    )
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory", llm_fast_known_company=True),
        source=snapshot,
        llm=model,
    ) as runtime:
        answer = await runtime.ask("Проверь МАКСМАРКЕТ", "lookup")
    assert answer.card and answer.card.inn == "5032257375"
    assert len(model.models[0].seen) == 2


@pytest.mark.parametrize("status", [502, 503, 429])
@pytest.mark.anyio
async def test_gateway_retry_does_not_sleep_for_a_minute_or_retry_rate_limits(status):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                status,
                headers={"retry-after": "60"},
                json={
                    "error": {
                        "message": "Temporary gateway error",
                        "type": "server_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Работает",
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        chat = ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key="test",
            max_retries=0,
            http_async_client=client,
        )
        chain = LLM([chat], retry_transport=True).with_tools([])
        started = time.monotonic()
        if status == 429:
            with pytest.raises(RateLimitError):
                await chain.ainvoke([HumanMessage(content="Проверка")])
            assert len(requests) == 1
        else:
            result = await chain.ainvoke([HumanMessage(content="Проверка")])
            assert result.content == "Работает" and len(requests) == 2
        assert time.monotonic() - started < 5
