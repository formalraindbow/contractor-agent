from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

from contractor_agent.agent.nodes import build_card, question_kind_hint, tool_subset
from contractor_agent.agent.presentation import has_internal_text, public_text
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Draft
from contractor_agent.api.evidence import evidence
from contractor_agent.data.paths import PathNotFoundError
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


@pytest.mark.parametrize(
    "text",
    [
        "ЗСК — зелёный (поле labels.zsk, путь report.zskRiskLevel).",
        "Капитал — 365 млн ₽ [report.finReports[0].liabilities.capitals].",
        "Действующих — 54 [source_path: report.executionProceedings].",
        "Можно работать (verdict_ru = «можно работать»).",
        "Сотрудники не указаны (source_path = .",
        "ЗСК — зелёный (значение берётся из поля ).",
        "Статус — CURRENT; причина — банкротство.",
    ],
)
def test_public_text_removes_whole_technical_annotations(text):
    result = public_text(text)
    assert not has_internal_text(result)
    assert result.count("(") == result.count(")")
    assert result.count("[") == result.count("]")
    assert "из поля" not in result


def test_public_text_preserves_economic_meaning():
    text = 'Капитал и резервы (2023 год) — 365,1 млн ₽. ООО "МАКСМАРКЕТ".'
    assert public_text(text) == text.replace('"МАКСМАРКЕТ"', "«МАКСМАРКЕТ»")


def test_routing_all_parts_and_goal_context():
    question = "Финансы. Речь о компании МАКСМАРКЕТ, ИНН 5032257375. Плачу по счёту."
    assert question_kind_hint(question)[0] == "answer"
    assert "get_financials" in tool_subset(question)
    assert {"get_arbitration_summary", "get_enforcement_summary"} <= set(
        tool_subset("Есть ли у них суды и долги у приставов?")
    )
    assert {"get_financials", "get_report_summary"} <= set(
        tool_subset("Кто руководитель и что с финансами?")
    )
    assert "get_financials" in tool_subset("Какие документы запросить?")
    assert question_kind_hint("Можно им платить?")[0] == "card"
    assert question_kind_hint("Объясни рекомендацию")[0] == "answer"


def test_cards_preserve_develop_verdicts_and_use_relevant_requests(snapshot):
    tools = Tools(snapshot)
    maxmarket = build_card(tools, "5032257375")
    techprof = build_card(tools, "1684017097")
    gdk = build_card(tools, "6165169320")
    assert maxmarket.verdict == gdk.verdict == "not_recommended"
    assert techprof.verdict == "ok"
    assert any("финансовых результатах" in text for text in techprof.ask_before)
    assert not any("штат" in text for text in techprof.ask_before)
    assert any("управляющего" in text for text in maxmarket.ask_before)
    for card in [maxmarket, techprof, gdk]:
        for fact in card.attention:
            result = evidence(tools, card.inn, fact.source_path, fact.code)
            assert result["report_date"] == card.report_date.isoformat()
            assert result["calculation"]["description"]
            assert result["basis"]


def test_evidence_keeps_missing_values_and_separate_enforcement_groups(snapshot):
    tools = Tools(snapshot)
    absent = evidence(tools, "5032257375", "report.baseInfo.staff")
    assert absent["value"] is None
    result = evidence(tools, "5032257375", "report.executionProceedings")
    groups = result["calculation"]["values"]
    assert groups["active"]["count"] == 54
    assert groups["finished"]["count"] == 453
    assert Decimal(str(groups["active"]["known_sum"])) == Decimal("3995486.53")
    with pytest.raises(PathNotFoundError):
        evidence(tools, "5032257375", "report.baseInfo.staff", "enforcement_active")
    with pytest.raises(PathNotFoundError):
        evidence(tools, "5032257375", "../../.env")


async def test_conversation_survives_runtime_restart(snapshot, tmp_path):
    settings = Settings(
        session_store="sqlite",
        session_db_path=tmp_path / "sessions.sqlite",
        runs_dir=tmp_path / "runs",
        llm_fallback_models="",
    )
    script = [
        tool_call("get_report_summary", "s1", inn="5032257375"),
        AIMessage(content="Данные получены."),
        Draft(kind="answer", lines=["Сотрудники в отчёте не указаны. Отчёт от 31.07.2026."]),
    ]
    async with AgentRuntime(settings, source=snapshot, llm=scripted_llm(script)) as runtime:
        await runtime.ask("Сколько сотрудников у МАКСМАРКЕТ 5032257375?", "persistent")
    second = scripted_llm(
        [
            AIMessage(content="Продолжаю."),
            Draft(kind="answer", lines=["В отчёте нет сведений о сотрудниках."]),
        ]
    )
    async with AgentRuntime(settings, source=snapshot, llm=second) as runtime:
        state = await runtime.graph.aget_state({"configurable": {"thread_id": "persistent"}})
        assert state.values["selected_inns"] == ["5032257375"]
        assert state.values["answer"].text_md
        await runtime.ask("Сколько их?", "persistent")
        assert any("МАКСМАРКЕТ" in str(m.content) for m in second.models[0].seen[0])


def test_finalization_does_not_copy_old_answers_or_current_draft():
    from langchain_core.messages import HumanMessage, ToolMessage

    from contractor_agent.agent.nodes import finalization_history

    question = HumanMessage(content="Кто руководит?")
    call = tool_call("get_report_summary", "new", inn="5032257375")
    evidence = ToolMessage(content="current fields", tool_call_id="new")
    history = [
        HumanMessage(content="Финансы?"),
        AIMessage(content="OLD CARD"),
        question,
        call,
        evidence,
        AIMessage(content="UNVERIFIED DRAFT"),
    ]
    assert finalization_history(history, question.content) == [question, call, evidence]


def test_label_payment_and_documents_are_grounded(snapshot):
    from contractor_agent.agent.scoped_answers import scoped_answer

    tools = Tools(snapshot)
    answer = scoped_answer(
        tools,
        ["6165169320"],
        "Почему метка банка зелёная, если счета заблокированы? Можно им платить?",
    )
    assert "не раскрывает" in answer.text_md
    assert "подтвердить" in answer.text_md and "18,9 млн" in answer.text_md
    assert "26,2 млн" in answer.text_md
    assert "Вывод помощника" in answer.text_md
    assert "не учитывает" not in answer.text_md
    docs = scoped_answer(tools, ["5032257375"], "Какие документы запросить перед оплатой?")
    assert "полномочий" in docs.text_md and "ограничений" in docs.text_md
    assert "действующих обязательств" in docs.text_md
    assert "2023" not in docs.text_md and "штат" not in docs.text_md
    assert scoped_answer(tools, ["5032257375"], "Что с ЗСК и сколько сотрудников?") is None
    assert "https://www.cbr.ru/" in public_text(answer.text_md)


async def test_missing_structured_output_uses_schema_fallback():
    llm = scripted_llm([None, Draft(kind="answer", lines=["Проверенный ответ"])])
    answer = await llm.structured(Draft).ainvoke([])
    assert answer.text_md == "Проверенный ответ"


def test_empty_litigation_and_deferred_payment_have_relevant_limits(snapshot):
    from contractor_agent.agent.scoped_answers import scoped_answer

    tools = Tools(snapshot)
    from contractor_agent.agent.section_answers import section_answer

    empty = section_answer(tools, ["1684017097"], "Есть ли у них суды и долги у приставов?")
    assert "не найдено" in empty.text_md and "не доказывает" in empty.text_md
    assert "ответчик" in empty.text_md and "истец" in empty.text_md
    assert "0 руб" not in empty.text_md and "производств нет" not in empty.text_md
    deferred = scoped_answer(tools, ["1684017097"], "Можно давать им отсрочку на 60 дней?")
    assert "Капитал и резервы" in deferred.text_md and "два последних года" in deferred.text_md
    assert "2024" in deferred.text_md and "2025" in deferred.text_md
    assert "штат" not in deferred.text_md and "сотрудник" not in deferred.text_md


def test_date_cleanup_keeps_words_intact():
    from contractor_agent.agent.nodes import tidy_text

    assert tidy_text("Срок — 31 марта 2026 года.") == "Срок — 31.03.2026."


def test_internal_annotations_do_not_leave_empty_bold_markup():
    text = "Строка прибыли отсутствует. **(report.finReports[0].common.profit)**"
    assert public_text(text) == "Строка прибыли отсутствует."
    assert public_text("**Важный факт**\n\n****\n\nПродолжение") == (
        "**Важный факт**\n\n****\n\nПродолжение"
    )


def test_comparison_removes_only_report_date_paragraphs():
    from contractor_agent.agent.presentation import comparison_text

    fact = "Руководитель назначен 09.02.2026. Отчётность за 2025 год неполная."
    text = fact + "\n\nОтчёт по ИНН 6165169320 от 25.08.2026.\n\nОтчёт от 28.08.2026."
    assert comparison_text(text) == fact


def test_obsolete_verdict_does_not_prescribe_payment_terms():
    from contractor_agent.signals.model import VERDICT_RU, Verdict

    old = "Работать только на условиях: предоплата и подтверждающие документы"
    assert public_text(old) == VERDICT_RU[Verdict.NOT_RECOMMENDED]
    assert "предоплат" not in " ".join(VERDICT_RU.values())


def test_default_recommendation_does_not_assume_a_contract(snapshot):
    from contractor_agent.agent.nodes import enforce_verdict, verdict_present
    from contractor_agent.signals.model import VERDICT_RU, Verdict

    card = build_card(Tools(snapshot), "9705152496")
    assert card.verdict == Verdict.CHECK
    text = enforce_verdict("Есть факты, которые нужно уточнить.", card.verdict)
    assert VERDICT_RU[Verdict.CHECK] in text and "договор" not in text
    assert verdict_present(text, Verdict.CHECK)
    assert public_text("Стоит проверить до договора.") == "нужна дополнительная проверка."


async def test_comparison_documents_do_not_append_verdicts_or_report_lines(snapshot, tmp_path):
    text = (
        "ГДК: подтверждение текущих ограничений по счетам. ТЕХПРОФ: отчёт о финансовых результатах."
    )
    llm = scripted_llm(
        [
            tool_call("compare_companies", "cmp", inns=["6165169320", "1684017097"]),
            AIMessage(content="Данные получены."),
            Draft(kind="comparison", lines=[text, "Отчёт по ИНН 6165169320 от 25.08.2026."]),
        ]
    )
    settings = Settings(session_db_path=tmp_path / "s.sqlite", runs_dir=tmp_path / "runs")
    async with AgentRuntime(settings, source=snapshot, llm=llm) as runtime:
        answer = await runtime.ask(
            "Какие документы запросить у ГДК 6165169320 и ТЕХПРОФ 1684017097?", "cmp-docs"
        )
    assert "ограничений" in answer.text_md and "финансовых результатах" in answer.text_md
    assert "По данным отчётов:" not in answer.text_md
    assert "Отчёт по ИНН" not in answer.text_md and "предоплат" not in answer.text_md
    assert answer.report_dates == {"6165169320": "2026-08-25", "1684017097": "2026-08-28"}
    assert len(answer.cards) == 2


def test_concise_choice_uses_current_cards_without_repeating_verdicts(snapshot):
    from contractor_agent.agent.citations import validate_citations
    from contractor_agent.agent.comparison_answers import comparison_followup

    tools = Tools(snapshot)
    inns = ["6165169320", "1684017097", "2311304742"]
    cards = [build_card(tools, inn) for inn in inns]
    answer = comparison_followup(cards, "С кем лучше работать?")
    assert "ТЕХПРОФ" in answer.lines[0]
    assert sum(line.startswith("- ") for line in answer.lines) == 3
    assert "предоплат" not in answer.text_md and "Вердикт:" not in answer.text_md
    assert "18,9 млн" in answer.text_md and "недостоверным" in answer.text_md
    assert all(c.ok for c in validate_citations(snapshot, inns, answer.citations))
    assert comparison_followup(cards, "Что с финансами?") is None
    assert comparison_followup(cards, "С кем лучше работать и какие документы запросить?") is None
    assert comparison_followup(cards, "С кем лучше работать с отсрочкой?") is None


def test_court_year_does_not_invent_status_or_missing_blocking(snapshot):
    from contractor_agent.agent.section_answers import section_answer

    tools = Tools(snapshot)
    result = section_answer(
        tools,
        ["5032257375"],
        "С кем они судились в 2025 году и за что? "
        "Речь о компании МАКСМАРКЕТ, ИНН 5032257375. Плачу по счёту.",
    )
    assert "21 685 219" in result.text_md and "предметы" in result.text_md
    assert "не указывает, завершены" in result.text_md
    assert "блокиров" not in result.text_md
    assert "0 ₽" not in result.text_md
    total = section_answer(tools, ["5029069967"], "Сколько у них судебных дел?")
    assert "1525" in total.text_md and "1488" in total.text_md and "37" in total.text_md
    assert "69 605" in total.text_md or "69 6" in total.text_md


@pytest.mark.parametrize("inn", ["5032257375", "1684017097", "5029069967", "6165169320"])
@pytest.mark.parametrize(
    "question", ["Что с судами?", "Что с долгами у приставов?", "С кем судились в 2025 году?"]
)
def test_section_citations_resolve_for_both_roles_and_known_amounts(snapshot, inn, question):
    from contractor_agent.agent.citations import validate_citations
    from contractor_agent.agent.section_answers import section_answer

    result = section_answer(Tools(snapshot), [inn], question)
    checks = validate_citations(snapshot, [inn], result.citations)
    assert checks and all(c.ok for c in checks), [
        (c.citation.claim, c.why) for c in checks if not c.ok
    ]
    assert "54 54" not in result.text_md and "1 1 дело" not in result.text_md


def test_named_company_narrows_comparison_context(snapshot):
    from contractor_agent.agent.nodes import requested_inns

    inns = ["6165169320", "1684017097", "2311304742"]
    context = " Компании: ГДК (ИНН 6165169320), ТЕХПРОФ (ИНН 1684017097), БИЛД-ЮГ (ИНН 2311304742)."
    question = "почему у ТЕХПРОФ не найдено сигналов?" + context
    assert requested_inns(question, inns, snapshot) == ["1684017097"]
    assert question_kind_hint(question)[0] == "answer"
    assert "get_risk_signals" in tool_subset(question)
    assert "compare_companies" not in tool_subset(question)
    assert requested_inns("Что с судами у 1684017097?" + context, inns, snapshot) == ["1684017097"]
    assert requested_inns("Сравни ГДК и ТЕХПРОФ" + context, inns, snapshot) == inns[:2]
    assert requested_inns("Сравни ТЕХПРОФ с остальными" + context, inns, snapshot) == []


def test_no_signal_explanation_distinguishes_checked_and_missing_data(snapshot):
    from contractor_agent.agent.citations import validate_citations
    from contractor_agent.agent.scoped_answers import scoped_answer

    tools = Tools(snapshot)
    q = "Почему у ТЕХПРОФ не найдено сигналов?"
    draft = scoped_answer(tools, ["1684017097"], q)
    assert "помощник" in draft.text_md
    assert "Компания указана как действующая" in draft.text_md
    assert "нет записей о делах" in draft.text_md
    assert "Что оценить не удалось" in draft.text_md
    assert "не считаются положительными результатами" in draft.text_md
    assert "вердикт банка" not in draft.text_md.lower() and "signals" not in draft.text_md
    assert all(c.ok for c in validate_citations(snapshot, ["1684017097"], draft.citations))
    assert scoped_answer(tools, ["6165169320"], q) is None
    assert scoped_answer(tools, ["1684017097"], q + " Покажи суды.") is None


async def test_single_company_followup_keeps_comparison_memory(snapshot, tmp_path):
    inns = ["6165169320", "1684017097", "2311304742"]
    llm = scripted_llm(
        [
            tool_call("compare_companies", "all", inns=inns),
            AIMessage(content="Данные получены."),
            Draft(
                kind="comparison",
                lines=["У ООО «ТЕХПРОФ» по данным отчёта не выделены существенные факторы риска."],
            ),
        ]
    )
    settings = Settings(session_db_path=tmp_path / "s.sqlite", runs_dir=tmp_path / "runs")
    async with AgentRuntime(settings, source=snapshot, llm=llm) as runtime:
        answer = await runtime.ask(
            "Почему у ТЕХПРОФ не найдено сигналов? Компании: ГДК (ИНН 6165169320), "
            "ТЕХПРОФ (ИНН 1684017097), БИЛД-ЮГ (ИНН 2311304742).",
            "focused",
        )
        state = await runtime.graph.aget_state({"configurable": {"thread_id": "focused"}})
    assert answer.kind == "answer"
    assert answer.cards == []
    assert answer.report_dates == {"1684017097": "2026-08-28"}
    assert "ГДК" not in answer.text_md and "БИЛД" not in answer.text_md
    assert state.values["selected_inns"] == inns
    assert state.values["turn_inns"] == ["1684017097"]


@pytest.mark.parametrize(
    "question",
    [
        "Недостоверный адрес у 2311304742 что это значит",
        "Недостоверный адрес у БИЛД-ЮГ — что это означает?",
        "Поясни отметку о блокировке счетов у 2311304742",
    ],
)
def test_fact_explanations_do_not_request_full_comparisons(snapshot, question):
    from contractor_agent.agent.nodes import comparison_needs_verdict, requested_inns

    context = " Компании: ГДК (ИНН 6165169320), ТЕХПРОФ (ИНН 1684017097), БИЛД-ЮГ (ИНН 2311304742)."
    q = question + context
    assert question_kind_hint(q)[0] == "answer"
    assert "get_risk_signals" in tool_subset(q)
    assert "compare_companies" not in tool_subset(q)
    assert not comparison_needs_verdict(q)
    assert requested_inns(q, ["6165169320", "1684017097", "2311304742"], snapshot) == ["2311304742"]
    assert question_kind_hint(question + ". Можно ли им платить?")[0] == "card"


async def test_fact_explanation_repairs_unrequested_company_summaries(snapshot, tmp_path):
    from contractor_agent.agent.schema import Citation

    inns = ["6165169320", "1684017097", "2311304742"]
    concise = "У ООО «БИЛД-ЮГ» в отчёте есть отметка о недостоверных регистрационных данных."
    llm = scripted_llm(
        [
            tool_call("compare_companies", "previous", inns=inns),
            AIMessage(content="Данные получены."),
            AIMessage(content="Продолжаю."),
            Draft(
                kind="answer",
                lines=[
                    concise,
                    "ООО «ГДК»: есть существенные риски. ООО «ТЕХПРОФ»: можно работать.",
                ],
            ),
            tool_call("get_risk_signals", "focused", inn="2311304742"),
            AIMessage(content="Данные получены."),
            Draft(
                kind="answer",
                lines=[concise],
                citations=[
                    Citation(
                        claim="Отмечены недостоверные регистрационные данные",
                        source_path="report.reputationalRisks.negative[4].code",
                        inn="2311304742",
                    )
                ],
            ),
        ]
    )
    settings = Settings(session_db_path=tmp_path / "scope.sqlite", runs_dir=tmp_path / "runs")
    context = " Компании: ГДК (ИНН 6165169320), ТЕХПРОФ (ИНН 1684017097), БИЛД-ЮГ (ИНН 2311304742)."
    async with AgentRuntime(settings, source=snapshot, llm=llm) as runtime:
        await runtime.ask("С кем лучше работать?" + context, "address")
        answer = await runtime.ask(
            "Недостоверные регистрационные данные у 2311304742 что это значит" + context, "address"
        )
        state = await runtime.graph.aget_state({"configurable": {"thread_id": "address"}})
    assert state.values["citation_retry"] == 1
    assert state.values["selected_inns"] == inns
    assert answer.kind == "answer" and not answer.card and not answer.cards
    assert set(answer.report_dates) == {"2311304742"}
    assert not answer.invalid_citations
    assert "ГДК" not in answer.text_md and "ТЕХПРОФ" not in answer.text_md
    assert "По данным отчётов" not in answer.text_md
    assert "недостоверных регистрационных данных" in answer.text_md


@pytest.mark.parametrize(
    ("question", "path"),
    [
        (
            "Недостоверный адрес у 2311304742 что это значит",
            "report.reputationalRisks.negative[2].code",
        ),
        (
            "Блокировка счетов у БИЛД-ЮГ что это значит",
            "report.reputationalRisks.negative[5].code",
        ),
    ],
)
def test_flag_meaning_uses_confirmed_fact_without_other_risks(snapshot, question, path):
    from contractor_agent.agent.citations import validate_citations
    from contractor_agent.agent.scoped_answers import scoped_answer

    result = scoped_answer(Tools(snapshot), ["2311304742"], question)
    assert result and result.kind == "answer"
    assert len(result.text_md) < 650
    assert "ГДК" not in result.text_md and "ТЕХПРОФ" not in result.text_md
    assert not any(t in result.text_md for t in ["скрыть", "фиктивн", "вердикт", "flag_"])
    assert [c.source_path for c in result.citations] == [path]
    assert all(c.ok for c in validate_citations(snapshot, ["2311304742"], result.citations))
    assert scoped_answer(Tools(snapshot), ["1684017097"], question) is None
    assert scoped_answer(Tools(snapshot), ["2311304742"], question + " И что с судами?") is None
