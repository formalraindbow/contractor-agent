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
                    "Итог: МАКСМАРКЕТ — работать только на условиях: предоплата и подтверждающие документы; ГДК — работать только на условиях: предоплата и подтверждающие документы.",
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


async def test_comparison_verdict_is_enforced_by_code(snapshot: Snapshot, tmp_path) -> None:
    """Текст сравнения спорит с вердиктами: круг исправления, потом итог дописывает код."""
    wrong = Draft(
        kind="comparison",
        lines=["С МАКСМАРКЕТ можно работать; с ГДК можно работать."],
        citations=[],
    )
    llm = scripted_llm(
        [
            tool_call("compare_companies", "c1", inns=["5032257375", "6165169320"]),
            AIMessage(content="Фактов достаточно."),
            wrong,
            AIMessage(content="Оставляю как есть."),
            wrong,
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        answer = await rt.ask("Сравни 5032257375 и 6165169320", thread_id="cmp2")
    assert answer.kind == "comparison" and len(answer.cards) == 2
    assert "По данным отчётов" in answer.text_md
    assert answer.text_md.count("работать только на условиях") == 2


def test_inns_from_args_accepts_string_list() -> None:
    from contractor_agent.agent.nodes import _inns_from_args

    assert _inns_from_args({"inns": "5032257375, 6165169320"}) == ["5032257375", "6165169320"]
    assert _inns_from_args({"inn": "5032257375"}) == ["5032257375"]


def test_forbidden_phrase_is_flagged_and_scrubbed() -> None:
    from contractor_agent.agent.nodes import forbidden_problem, scrub_forbidden

    text = "**МАКСМАРКЕТ — работать нельзя, компания в банкротстве.**\nФакты…"
    assert "работать нельзя" in (forbidden_problem(text) or "")
    scrubbed = scrub_forbidden(
        text, "работать только на условиях: предоплата и подтверждающие документы"
    )
    assert "нельзя" not in scrubbed and "работать только на условиях" in scrubbed
    assert forbidden_problem("Стоит проверить до договора.") is None


def test_question_kind_hint() -> None:
    from contractor_agent.agent.nodes import question_kind_hint

    assert question_kind_hint("Сравни 5032257375 и 6165169320")[0] == "comparison"
    assert question_kind_hint("А сколько у них сейчас висит долгов у приставов?")[0] == "answer"
    assert question_kind_hint("Хочу отгрузить ГДК с отсрочкой. Сколько у них судов?")[0] == "answer"
    assert question_kind_hint("Есть открытые иски. Можно ли давать им отсрочку?")[0] == "card"
    assert question_kind_hint("Какая у ИП выручка? Потянет ли он заказ?")[0] == "card"
    assert question_kind_hint("Проверь ООО «МАКСМАРКЕТ»: можно ли с ней работать?")[0] == "card"
    assert question_kind_hint("У него светофор красный, а ЗСК зелёный — кому верить?")[0] == "card"
    assert question_kind_hint("Привет") == (None, None)


def test_format_problem_flags_summary_for_section_question() -> None:
    from contractor_agent.agent.nodes import format_problem
    from contractor_agent.agent.schema import Answer

    q = "А сколько у них сейчас висит долгов у приставов?"
    card_like = Answer(kind="card", text_md="Оценки банка… На что обратить внимание: …")
    assert format_problem(card_like, q)
    short = Answer(
        kind="answer", text_md="54 действующих производства на 4 млн ₽. Дата отчёта 31.07.2026."
    )
    assert format_problem(short, q) is None
    assert format_problem(card_like, "Проверь компанию") is None


def test_verdict_codes_are_replaced_with_phrases() -> None:
    from contractor_agent.agent.nodes import replace_verdict_codes

    out = replace_verdict_codes(
        "Итог: ГДК – not_recommended, ТЕХПРОФ – ok. Поле check_id не трогаем."
    )
    assert "not_recommended" not in out and " ok" not in out
    assert "работать только на условиях" in out and "можно работать" in out
    assert "check_id" in out


def test_visible_history_drops_old_tool_traffic_but_keeps_current_turn() -> None:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from contractor_agent.agent.nodes import visible_history

    old_call = AIMessage(
        content="", tool_calls=[{"name": "get_report_summary", "args": {"inn": "1"}, "id": "a"}]
    )
    msgs = [
        HumanMessage(content="Проверь X"),
        old_call,
        ToolMessage(content="{}", tool_call_id="a", name="get_report_summary"),
        AIMessage(content="Карточка X: … Дата отчёта 31.07.2026."),
        HumanMessage(content="А суды?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_arbitration_summary", "args": {"inn": "1"}, "id": "b"}],
        ),
        ToolMessage(content="{}", tool_call_id="b", name="get_arbitration_summary"),
    ]
    seen = visible_history(msgs, "А суды?")
    assert [type(m).__name__ for m in seen] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert seen[1].content.startswith("Карточка X")


def test_tool_subset_by_question() -> None:
    from contractor_agent.agent.nodes import tool_subset

    assert tool_subset("А сколько у них сейчас висит долгов у приставов?") == [
        "search_company",
        "get_report_summary",
        "get_enforcement_summary",
    ]
    assert tool_subset("Сравни 5032257375 и 6165169320") == ["search_company", "compare_companies"]
    assert tool_subset("Проверь ООО «МАКСМАРКЕТ»: можно ли с ней работать?") is None


def test_forbidden_phrase_with_words_between() -> None:
    from contractor_agent.agent.nodes import forbidden_problem, scrub_forbidden

    text = "**Работать с ООО «МАКСМАРКЕТ» нельзя — компания в процедуре банкротства.**"
    assert forbidden_problem(text)
    assert "нельзя" not in scrub_forbidden(
        text, "работать только на условиях: предоплата и подтверждающие документы"
    )


def test_empty_answer_triggers_repair() -> None:
    from contractor_agent.agent.nodes import empty_problem
    from contractor_agent.agent.schema import Answer

    assert empty_problem(Answer(kind="answer", text_md=""))
    assert empty_problem(Answer(kind="answer", text_md="| | **—**"))
    assert empty_problem(Answer(kind="answer", text_md="54 производства.")) is None
    assert (
        empty_problem(
            Answer(
                kind="answer",
                text_md="54 действующих производства на 4 млн ₽ по отчёту от 31.07.2026.",
            )
        )
        is None
    )


async def test_comparison_after_card_uses_only_this_turn_companies(
    snapshot: Snapshot, tmp_path
) -> None:
    """Карточка МАКСМАРКЕТ, затем сравнение двух других: модель по привычке пишет card —
    код переводит в comparison по компаниям текущего хода, старая компания не попадает."""
    llm = scripted_llm(
        [
            tool_call("get_risk_signals", "c1", inn="5032257375"),
            AIMessage(content="ок"),
            Draft(
                kind="card",
                lines=[
                    "МАКСМАРКЕТ: работать только на условиях: предоплата и подтверждающие документы. Отчёт от 31.07.2026."
                ],
                citations=[],
            ),
            tool_call("compare_companies", "c2", inns=["6165169320", "1684017097"]),
            AIMessage(content="ок"),
            Draft(kind="card", lines=["ГДК и ТЕХПРОФ: см. ниже."], citations=[]),
            AIMessage(content="ок"),
            Draft(kind="card", lines=["ГДК и ТЕХПРОФ: см. ниже."], citations=[]),
        ]
    )
    async with AgentRuntime(_settings(tmp_path), source=snapshot, llm=llm) as rt:
        await rt.ask("Проверь 5032257375", thread_id="t")
        answer = await rt.ask(
            "Сравни 6165169320 и 1684017097: с кем лучше работать?", thread_id="t"
        )
    assert answer.kind == "comparison"
    assert [c.inn for c in answer.cards] == ["6165169320", "1684017097"]
    assert "По данным отчётов" in answer.text_md


def test_tidy_text_removes_field_names_and_repeats() -> None:
    from contractor_agent.agent.nodes import tidy_text

    raw = (
        "- ООО «ГДК» – оценка: зелёный (svetofor) – verdict_ru: работать только на условиях\n"
        "- claim: Открытые иски к компании – 2 дела source_path: report.arbitrationByStatus.x\n"
        "- Адрес признан недостоверным (Адрес признан недостоверным) [report.a]\n"
        "- Bank ratings: зелёный"
    )
    out = tidy_text(raw)
    assert "verdict_ru" not in out and "(svetofor)" not in out and "claim:" not in out
    assert "- Открытые иски к компании – 2 дела" in out and "report." not in out
    assert "недостоверным (Адрес" not in out and "Оценки банка" in out
    assert "[report" not in out


def test_drop_invalid_lines_removes_unverified_numbers() -> None:
    from contractor_agent.agent.nodes import drop_invalid_lines
    from contractor_agent.agent.schema import Citation

    text = (
        "- 258 завершённых дел на 2,6 млрд ₽ [report.arbitrationByStatus.x.dfAmount]\n"
        "- 7 152 200 % от чистых активов [report.arbitrationByStatus.x.dfAmount]\n"
        "- Дата отчёта: 31.07.2026"
    )
    bad = Citation(
        claim="7 152 200 % от чистых активов", source_path="report.arbitrationByStatus.x.dfAmount"
    )
    out = drop_invalid_lines(text, [bad])
    assert "7 152 200" not in out and "258 завершённых" in out and "Дата отчёта" in out
    assert "Убрано утверждений" in out


def test_tidy_text_strips_field_paths_and_inn() -> None:
    from contractor_agent.agent.nodes import tidy_text

    raw = "Капитал 22,8 млн ₽ [report.finReports[0].liabilities.capitals][report.finReports[0].common.proceeds] inn 1684017097 – можно работать"
    out = tidy_text(raw)
    assert "report." not in out and "1684017097" not in out and "]" not in out
    assert out.startswith("Капитал 22,8 млн ₽") and "можно работать" in out
