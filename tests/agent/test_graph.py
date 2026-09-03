# ruff: noqa: E501
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.data.loader import Snapshot
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


def _settings(tmp_path) -> Settings:
    return Settings(runs_dir=tmp_path / "runs", openrouter_api_key="x")


async def test_card_flow_with_one_citation_repair(snapshot: Snapshot, tmp_path) -> None:
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "c1", inn="5032257375"),
            tool_call("get_risk_signals", "c2", inn="5032257375"),
            AIMessage(content="Фактов достаточно."),
            Draft(
                kind="card",
                lines=[
                    "Компания банкрот [report.status.reasonName], 54 производства [report.executionProceedings], штат 100 человек.",
                    "",
                    "Стоит проверить до договора.",
                ],
                citations=[
                    Citation(claim="признана банкротом", source_path="report.status.reasonName"),
                    Citation(
                        claim="54 действующих производства",
                        source_path="report.executionProceedings",
                    ),
                    Citation(claim="штат 100 человек", source_path="report.baseInfo.staff"),
                ],
            ),
            AIMessage(content="Убираю утверждение про штат."),
            Draft(
                kind="card",
                lines=[
                    "Компания банкрот [report.status.reasonName], 54 производства [report.executionProceedings].",
                    "",
                    "Работать только на условиях: предоплата и подтверждающие документы. Отчёт от 31.07.2026.",
                ],
                citations=[
                    Citation(claim="признана банкротом", source_path="report.status.reasonName"),
                    Citation(
                        claim="54 действующих производства",
                        source_path="report.executionProceedings",
                    ),
                ],
            ),
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask("Проверь ООО МАКСМАРКЕТ, ИНН 5032257375", thread_id="t1")
        state = await rt.graph.aget_state({"configurable": {"thread_id": "t1"}})

    assert answer.kind == "card" and answer.card is not None
    assert answer.card.verdict.value == "not_recommended" and answer.card.inn == "5032257375"
    assert (
        answer.card.labels.riskLevel == "зелёный"
        and answer.card.attention[0].severity.value == "critical"
    )
    assert answer.report_dates == {"5032257375": "2026-07-31"}
    assert [c.source_path for c in answer.citations] == [
        "report.status.reasonName",
        "report.executionProceedings",
    ]
    assert answer.invalid_citations == []
    values = state.values
    assert values["citation_retry"] == 1 and values["selected_inns"] == ["5032257375"]
    assert [t.name for t in values["trace"]] == ["get_report_summary", "get_risk_signals"]
    assert all(t.available for t in values["trace"])
    repair = [
        m
        for m in values["messages"]
        if isinstance(m, HumanMessage) and "Проверка цитат" in m.content
    ]
    assert len(repair) == 1 and "report.baseInfo.staff" in repair[0].content
    assert "не совпадает с рекомендацией" in repair[0].content  # «стоит проверить» против карточки
    assert "только на условиях" in answer.text_md.casefold()
    assert "\n\n" in answer.text_md  # строки склеены кодом
    tool_msgs = [m for m in values["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 2 and '"available": true' in tool_msgs[0].content
    runs = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(runs) == 1 and "citation_retry" in runs[0].read_text(encoding="utf-8")


async def test_invalid_citation_is_marked_after_one_retry(snapshot: Snapshot, tmp_path) -> None:
    bad = Draft(
        kind="answer",
        lines=["Выручка 999 млн [report.finReports[0].common.proceeds]"],
        citations=[
            Citation(claim="выручка 999 млн ₽", source_path="report.finReports[0].common.proceeds")
        ],
    )
    llm = scripted_llm(
        [
            tool_call("get_financials", "c1", inn="1684017097"),
            AIMessage(content="ок"),
            bad,
            AIMessage(content="повторяю"),
            bad,
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask("Какая выручка у ТЕХПРОФ 1684017097?")
    assert answer.kind == "answer" and answer.card is None
    assert answer.citations == [] and len(answer.invalid_citations) == 1


async def test_refusal_when_company_not_found(snapshot: Snapshot, tmp_path) -> None:
    llm = scripted_llm(
        [
            tool_call("search_company", "c1", query="Рога и копыта"),
            AIMessage(content="Не найдено."),
            Draft(
                kind="refusal", lines=["Компания «Рога и копыта» в базе не найдена."], citations=[]
            ),
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask("Проверь Рога и копыта")
    assert answer.kind == "refusal" and answer.card is None and answer.report_dates == {}


async def test_memory_within_thread(snapshot: Snapshot, tmp_path) -> None:
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "c1", inn="2100006761"),
            AIMessage(content="есть"),
            Draft(kind="answer", lines=["РАДО АЛАТЫРЬ КС, зелёный светофор."], citations=[]),
            AIMessage(content="из истории: 2100006761"),
            Draft(kind="answer", lines=["Речь про ИНН 2100006761."], citations=[]),
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        await rt.ask("Что за компания 2100006761?", thread_id="s")
        second = await rt.ask("Какой у неё ИНН?", thread_id="s")
        state = await rt.graph.aget_state({"configurable": {"thread_id": "s"}})
    assert second.text_md.endswith("2100006761.")
    assert (
        sum(isinstance(m, HumanMessage) for m in state.values["messages"]) == 2
    )  # история сохранена


def test_enforce_verdict_replaces_and_appends() -> None:
    from contractor_agent.agent.nodes import enforce_verdict
    from contractor_agent.signals.model import Verdict

    text = "Итог: Стоит проверить до договора. Отчёт от 01.08.2026."
    fixed = enforce_verdict(text, Verdict.NOT_RECOMMENDED)
    assert "только на условиях" in fixed and "проверить до договора." not in fixed
    assert enforce_verdict("Без вывода.", Verdict.OK).endswith("**Рекомендация:** можно работать.")


def test_draft_accepts_text_instead_of_lines():
    from contractor_agent.agent.schema import Draft

    draft = Draft.model_validate({"kind": "refusal", "text_md": "а\nб", "citations": []})
    assert draft.lines == ["а", "б"] and draft.text_md == "а\nб"


async def test_comparison_flow_builds_cards_for_each_company(snapshot: Snapshot, tmp_path) -> None:
    llm = scripted_llm(
        [
            tool_call("compare_companies", "c1", inns=["5032257375", "6165169320"]),
            AIMessage(content="Фактов достаточно."),
            Draft(
                kind="comparison",
                lines=[
                    "МАКСМАРКЕТ: признана банкротом [report.status.reasonName].",
                    "ГДК: блокировка счетов на дату отчёта.",
                    "",
                    "Итог: с МАКСМАРКЕТ работать только на условиях: предоплата и подтверждающие документы; ГДК стоит проверить до договора.",
                ],
                citations=[
                    Citation(claim="признана банкротом", source_path="report.status.reasonName")
                ],
            ),
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask(
            "Сравни 5032257375 и 6165169320: с кем лучше работать?", thread_id="cmp"
        )

    assert answer.kind == "comparison" and answer.card is None
    assert [c.inn for c in answer.cards] == ["5032257375", "6165169320"]
    assert set(answer.report_dates) == {"5032257375", "6165169320"}
    assert not answer.invalid_citations
