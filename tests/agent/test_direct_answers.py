"""Product regressions: the model answers the question, factual guards stay active."""

import pytest
from langchain_core.messages import AIMessage

from contractor_agent.agent.interpretation import (
    InterpretationPlan,
    evidence_context,
    interpretation_problem,
    plan_draft,
)
from contractor_agent.agent.nodes import build_card, offtopic_reply
from contractor_agent.agent.presentation import public_text
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Answer, Citation
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from tests.agent.fakes import scripted_llm, tool_call


@pytest.mark.parametrize(
    "question",
    [
        "на что может повлиять блокировка счетов",
        "чем грозят недостовренные данные ЕГРЮЛ",
        "стоит с ними иметь дело",
        "Кого из этих трёх порекомендуешь?",
        "Кого ты бы выбрал",
        "На что влияет блокировка счетов",
    ],
)
def test_real_followups_without_question_mark_are_not_offtopic(question):
    assert offtopic_reply(question) is None


@pytest.mark.parametrize("question", ["Кого ты бы выбрал?", "Кого из этих троих выбрал бы?"])
def test_choice_paraphrases_reach_interpretation(question):
    from contractor_agent.agent.question import needs_interpretation

    assert needs_interpretation(question)


def test_choice_followup_refreshes_with_one_batch_and_keeps_specialist_reads():
    from contractor_agent.agent.evidence import missing_reads

    inns = ["6165169320", "1684017097"]
    calls = missing_reads("Кого ты бы выбрал?", inns, [])
    assert [(c["name"], c["args"]) for c in calls] == [("compare_companies", {"inns": inns})]
    calls = missing_reads("Кого выбрать с учётом финансов?", inns, [])
    assert [c["name"] for c in calls] == ["compare_companies", "get_financials", "get_financials"]


async def test_recommendation_is_model_composed_and_keeps_checked_evidence(snapshot, tmp_path):
    opening = "По данным отчёта я бы не начинал сотрудничество: компания проходит банкротство."
    model = scripted_llm(
        [
            tool_call("get_risk_signals", "risk", inn="5032257375"),
            AIMessage(content="Данные получены."),
            InterpretationPlan(answer=opening, evidence_ids=["E4"]),
        ]
    )
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory"), source=snapshot, llm=model
    ) as rt:
        answer = await rt.ask("Стоит с МАКСМАРКЕТ 5032257375 иметь дело?", "rec")
    assert answer.kind == "answer" and answer.card is None
    assert answer.text_md.startswith(opening)
    assert not answer.invalid_citations
    assert "открыто конкурсное производство" in answer.text_md.lower()
    assert "Пробел" not in answer.text_md and "Критический факт" not in answer.text_md


def test_fewer_signals_do_not_justify_recommending_a_critical_company(snapshot):
    cards = [build_card(Tools(snapshot), inn) for inn in ["5032257375", "6165169320", "2311304742"]]
    answer = Answer(
        kind="comparison",
        cards=cards,
        text_md="Рекомендую ГДК: у неё меньше критичных сигналов.",
        citations=[Citation(claim="Зелёный", source_path="report.zskRiskLevel")],
    )
    assert interpretation_problem(answer, "Кого порекомендуешь для сотрудничества?", cards)


@pytest.mark.parametrize(
    "question,opening,should_repair",
    [
        (
            "Стоит сотрудничать с ТЕХПРОФ?",
            "Я не рекомендую сотрудничать с ТЕХПРОФ, поскольку отсутствуют данные о прибыли "
            "и текущей ликвидности.",
            True,
        ),
        (
            "Можно дать ТЕХПРОФ отсрочку?",
            "Я не рекомендую отсрочку, поскольку отсутствуют данные о прибыли и ликвидности.",
            False,
        ),
        (
            "Можно безопасно сотрудничать с ТЕХПРОФ по предоплате?",
            "Я не рекомендую сотрудничать по предоплате: нет данных о платёжеспособности.",
            False,
        ),
        (
            "Стоит сотрудничать с ТЕХПРОФ?",
            "По данным отчёта рекомендую рассматривать ТЕХПРОФ для сотрудничества; "
            "о прибыли и ликвидности сведений недостаточно.",
            False,
        ),
    ],
)
def test_missing_fields_are_not_a_blanket_cooperation_refusal(
    snapshot, question, opening, should_repair
):
    cards = [build_card(Tools(snapshot), "1684017097")]
    answer = Answer(
        kind="answer",
        text_md=opening,
        citations=[Citation(claim="Действующая", source_path="report.status.status")],
    )
    assert bool(interpretation_problem(answer, question, cards)) == should_repair


@pytest.mark.parametrize("separator", ["\n\n", " "])
def test_choice_is_not_rejected_for_explaining_why_other_companies_were_not_chosen(
    snapshot, separator
):
    cards = [build_card(Tools(snapshot), inn) for inn in ["6165169320", "1684017097"]]
    answer = Answer(
        kind="comparison",
        cards=cards,
        text_md="По данным отчётов я бы выбрал ТЕХПРОФ." + separator + "У ГДК открытые иски.",
        citations=[Citation(claim="Зелёный", source_path="report.zskRiskLevel")],
    )
    assert interpretation_problem(answer, "Кого порекомендуешь для сотрудничества?", cards) is None


def test_saved_paragraph_labels_become_readable_bullets_without_losing_facts():
    original = (
        "Критический факт: Компания признана банкротом.\n\n"
        "**Умеренный факт:** Блокировка счетов.\n\n"
        "ℹ Информация: 453 завершённых производства.\n\n"
        "Пробел: Нет сведений о прибыли."
    )
    text = public_text(original)
    assert text == (
        "- Компания признана банкротом.\n\n- Блокировка счетов.\n\n"
        "- 453 завершённых производства.\n\n- Нет сведений о прибыли."
    )
    assert public_text(text) == text


def test_negated_critical_facts_and_other_positive_indicators_are_not_a_false_repair(snapshot):
    cards = [build_card(Tools(snapshot), inn) for inn in ["6165169320", "1684017097"]]
    answer = Answer(
        kind="comparison",
        cards=cards,
        text_md="Рекомендую ТЕХПРОФ: нет критических факторов, все доступные метки зелёные.",
        citations=[Citation(claim="Зелёный", source_path="report.zskRiskLevel")],
    )
    assert interpretation_problem(answer, "Кого порекомендуешь?", cards) is None


def test_blanket_critical_claim_is_repaired_when_one_candidate_has_none(snapshot):
    cards = [build_card(Tools(snapshot), inn) for inn in ["6165169320", "1684017097"]]
    answer = Answer(
        kind="comparison",
        cards=cards,
        text_md="Никого не рекомендую, поскольку у каждой компании есть критичные факты.",
        citations=[Citation(claim="Зелёный", source_path="report.zskRiskLevel")],
    )
    assert interpretation_problem(answer, "Кого порекомендуешь?", cards)


def test_proof_keeps_court_role_when_selected_outside_the_original_section():
    from contractor_agent.agent.interpretation import contextual_citations
    from contractor_agent.agent.schema import Draft

    draft = Draft(
        kind="answer",
        lines=["**Иски компании — истец**", "- Завершены: 20 дел."],
        citations=[Citation(claim="Завершены: 20 дел.", source_path="report.arbitrationByStatus")],
    )
    assert "истец" in contextual_citations(draft)[0].claim


def test_model_plan_cannot_invent_evidence_or_numbers(snapshot):
    from contractor_agent.agent.interpretation import evidence_context, plan_draft

    cards = [build_card(Tools(snapshot), "5032257375")]
    _, sources = evidence_context(cards, "Стоит сотрудничать?")
    with pytest.raises(ValueError, match="Unknown evidence"):
        plan_draft(
            InterpretationPlan(answer="Не рекомендую сотрудничать.", evidence_ids=["E999"]),
            sources,
            cards,
            "Стоит сотрудничать?",
        )
    with pytest.raises(ValueError, match="numbers"):
        plan_draft(
            InterpretationPlan(answer="Компания должна 99 миллионов.", evidence_ids=["E4"]),
            sources,
            cards,
            "Стоит сотрудничать?",
        )


def test_known_evidence_markers_are_removed_without_hiding_unknown_sources(snapshot):
    cards = [build_card(Tools(snapshot), "5032257375")]
    _, sources = evidence_context(cards, "Стоит сотрудничать?")
    plan = InterpretationPlan(
        answer="Не рекомендую сотрудничать: компания банкрот (E4).", evidence_ids=["E4"]
    )
    assert plan_draft(plan, sources, cards).lines[0] == (
        "Не рекомендую сотрудничать: компания банкрот."
    )
    for text in ("Компания банкрот (E999).", "Долг 99 миллионов (E4)."):
        with pytest.raises(ValueError, match="numbers"):
            plan_draft(plan.model_copy(update={"answer": text}), sources, cards)


@pytest.mark.parametrize(
    "text,bad",
    [
        ("Отсутствие дел подтверждает отсутствие текущих юридических рисков.", True),
        ("Компания не имеет негативных судебных или репутационных рисков.", True),
        ("Отсутствие записей не означает, что компания не имеет рисков.", False),
        ("Зелёные метки не подтверждают отсутствие текущих юридических рисков.", False),
        (
            "Зелёные метки не подтверждают безопасность. Отсутствие дел гарантирует надёжность.",
            True,
        ),
    ],
)
def test_guarantee_with_modifiers_and_separate_negated_sentence(text, bad):
    answer = Answer(kind="answer", text_md=text)
    assert bool(interpretation_problem(answer, "Объясни отсутствие сигналов")) == bad


def test_historical_court_evidence_does_not_support_generic_present_consequences():
    answer = Answer(
        kind="answer",
        text_md="Большое число судебных дел может создавать риски для ликвидности и репутации.",
        citations=[
            Citation(
                claim="Всего в сводке за всё время: 278 дел.", source_path="report.arbitration"
            ),
            Citation(
                claim="258 завершённых дел, ответчик.",
                source_path="report.arbitrationByStatus.defandantArbitrationFinished",
            ),
        ],
    )
    assert interpretation_problem(answer, "На что влияет количество судебных дел?")
    safe = answer.model_copy(
        update={"text_md": "Завершённые дела не подтверждают текущие риски для ликвидности."}
    )
    assert interpretation_problem(safe, "На что влияет количество судебных дел?") is None


@pytest.mark.parametrize(
    "text,bad",
    [
        ("Строки нет, что может быть связано с особенностями учёта или неполным раскрытием.", True),
        ("Причина отсутствия строки неизвестна. Определить прибыль или убыток нельзя.", False),
    ],
)
def test_missing_financial_field_has_no_invented_explanation(text, bad):
    answer = Answer(kind="answer", text_md=text)
    assert bool(interpretation_problem(answer, "Объясни, почему в отчётности нет прибыли?")) == bad


def test_user_proposed_duration_is_allowed_but_new_report_numbers_are_not(snapshot):
    from contractor_agent.agent.interpretation import evidence_context, plan_draft

    cards = [build_card(Tools(snapshot), "5032257375")]
    question = "Можно отгружать с отсрочкой 30 дней?"
    _, sources = evidence_context(cards, question)
    plan = InterpretationPlan(answer="Не рекомендую отсрочку 30 дней.", evidence_ids=["E3"])
    assert plan_draft(plan, sources, cards, question).lines[0] == plan.answer
    with pytest.raises(ValueError, match="numbers"):
        plan_draft(
            plan.model_copy(update={"answer": "Долг компании 99 рублей."}), sources, cards, question
        )


def test_positive_recommendation_cannot_claim_proven_reliability(snapshot):
    cards = [build_card(Tools(snapshot), "1684017097")]
    answer = Answer(
        kind="answer",
        text_md="Рекомендую ТЕХПРОФ: зелёные метки подтверждают надёжность компании.",
        citations=[Citation(claim="Зелёный", source_path="report.zskRiskLevel")],
    )
    assert interpretation_problem(answer, "Стоит сотрудничать?", cards)


def test_unknown_company_keeps_useful_clarification():
    answer = Answer(kind="refusal", text_md="В доступной базе нет отчёта по этому ИНН.")
    assert interpretation_problem(answer, "Стоит сотрудничать?") is None


def test_meaning_question_does_not_repeat_cooperation_recommendation(snapshot):
    cards = [build_card(Tools(snapshot), "5032257375")]
    answer = Answer(kind="answer", text_md="Я не рекомендую начинать сотрудничество.")
    assert interpretation_problem(answer, "На что влияет блокировка счетов?", cards)


def test_court_summary_and_its_nested_count_are_not_duplicate_evidence(snapshot):
    from contractor_agent.agent.interpretation import evidence_context

    cards = [build_card(Tools(snapshot), "5032257375")]
    parent_path = "report.arbitrationByStatus.defandantArbitration.defandantArbitrationFinished"
    extra = Citation(
        claim="Иски к компании — ответчик: Завершены: 258 дел.",
        source_path=parent_path,
        inn=cards[0].inn,
    )
    _, sources = evidence_context(cards, "На что влияет количество судебных дел?", [extra])
    assert sum(c.source_path.startswith(parent_path) for c in sources.values()) == 1


@pytest.mark.parametrize(
    "opening, problem",
    [
        ("Я рекомендую ТЕХПРОФ, так как у него нет судебных дел.", True),
        (
            "Я бы выбрал ТЕХПРОФ, так как у него нет критических факторов, судов "
            "и исполнительных производств.",
            True,
        ),
        (
            "Рекомендую ТЕХПРОФ: в отчёте нет критических факторов, судов "
            "и исполнительных производств.",
            False,
        ),
        ("По этим отчётам я бы выбрал ТЕХПРОФ: записей о судебных делах нет.", False),
        ("Рекомендую ТЕХПРОФ: в отчёте не найдено судебных дел.", False),
    ],
)
def test_recommendation_preserves_scope_of_missing_records(snapshot, opening, problem):
    cards = [build_card(Tools(snapshot), "1684017097")]
    answer = Answer(
        kind="answer",
        text_md=opening,
        citations=[
            Citation(claim="Нет записей о делах.", source_path="report.arbitrationByStatus")
        ],
    )
    assert bool(interpretation_problem(answer, "Кого порекомендуешь?", cards)) == problem


@pytest.mark.parametrize(
    "opening, problem",
    [
        ("По указанному адресу компании, скорее всего, её нет.", True),
        (
            "Сведения об адресе признаны недостоверными: компании может не быть по этому адресу.",
            False,
        ),
    ],
)
def test_address_mark_does_not_establish_probability_of_absence(opening, problem):
    answer = Answer(kind="answer", text_md=opening)
    assert bool(interpretation_problem(answer, "Недостоверный адрес: что это значит?")) == problem


@pytest.mark.parametrize(
    "text,bad",
    [
        ("По адресу может не существовать физического объекта.", True),
        ("Регистрационные сведения и учредительные документы могут быть неверными.", True),
        ("Это затрудняет подтверждение юридической личности и правовой чистоты.", True),
        ("Отметка не означает, что учредительные документы недействительны.", False),
        ("Компанию может быть труднее найти по указанному адресу.", False),
    ],
)
def test_address_explanation_preserves_subject_of_the_record(text, bad):
    answer = Answer(kind="answer", text_md=text)
    assert bool(interpretation_problem(answer, "Недостоверные данные ЕГРЮЛ: объясни")) == bad


def test_missing_profit_is_not_repeated_after_aggregate_years(snapshot):
    from contractor_agent.agent.interpretation import plan_draft

    cards = [build_card(Tools(snapshot), "1684017097")]
    sources = {
        "E1": Citation(
            claim="В отчётности за 2024–2025 годы нет строки «прибыль (убыток)».",
            source_path="report.finReports[0].common.profit",
            inn=cards[0].inn,
        ),
        "E2": Citation(
            claim="Прибыль / убыток: строка в отчёте отсутствует.",
            source_path="report.finReports[1].common.profit",
            inn=cards[0].inn,
        ),
    }
    draft = plan_draft(
        InterpretationPlan(answer="Знак результата неизвестен.", evidence_ids=["E1", "E2"]),
        sources,
        cards,
        "Почему отсутствие прибыли не означает убыток?",
    )
    assert len(draft.citations) == 1
    assert "2024–2025" in " ".join(draft.lines)


async def test_choice_keeps_both_companies_even_when_model_fetches_only_first(snapshot, tmp_path):
    llm = scripted_llm(
        [
            tool_call("compare_companies", "group", inns=["5032257375", "1684017097"]),
            AIMessage(content="Данные получены."),
            tool_call("get_risk_signals", "first-only", inn="5032257375"),
            AIMessage(content="Данные получены."),
            InterpretationPlan(answer="По этим отчётам я бы выбрал ТЕХПРОФ.", evidence_ids=["E1"]),
        ]
    )
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory"), source=snapshot, llm=llm
    ) as rt:
        await rt.ask("Сравни МАКСМАРКЕТ 5032257375 и ТЕХПРОФ 1684017097", "choice")
        answer = await rt.ask("Кого ты бы выбрал?", "choice")
    assert answer.kind == "comparison"
    assert [c.inn for c in answer.cards] == ["5032257375", "1684017097"]
    assert "ТЕХПРОФ" in answer.text_md.split("\n\n")[0]


def test_deferred_payment_request_requires_a_position_but_context_alone_does_not():
    from contractor_agent.agent.question import needs_interpretation

    assert needs_interpretation(
        "РЕМСТРОЙ М 7722608440: можно давать отсрочку или лучше по предоплате? "
        "Что проверить до договора?"
    )
    assert not needs_interpretation(
        "Хочу отгрузить ГДК материалы с отсрочкой. Сколько у них судов и кто с кем?"
    )


async def test_deferred_payment_answer_is_not_replaced_by_financial_template(snapshot, tmp_path):
    text = (
        "Я пока не рекомендую давать отсрочку: сначала уточните отмеченную налоговую задолженность."
    )
    model = scripted_llm(
        [
            tool_call("get_risk_signals", "risk", inn="7722608440"),
            AIMessage(content="Данные получены."),
            InterpretationPlan(answer=text, evidence_ids=["E3"]),
        ]
    )
    async with AgentRuntime(
        Settings(runs_dir=tmp_path, session_store="memory"), source=snapshot, llm=model
    ) as rt:
        answer = await rt.ask("РЕМСТРОЙ М 7722608440: можно давать отсрочку?", "deferral")
    assert answer.text_md.startswith(text)
    assert "### Финансы за" not in answer.text_md
    assert not answer.invalid_citations


def test_plaintiff_claim_is_not_its_own_payment_obligation(snapshot):
    cards = [build_card(Tools(snapshot), "5032257375")]
    text = (
        "Наличие завершённых исков как истца свидетельствует о возможных обязательствах "
        "по выплатам, что также может повлиять на финансовое состояние."
    )
    assert interpretation_problem(
        Answer(kind="answer", text_md=text), "На что влияет количество судебных дел?", cards
    )
    correct = (
        "Как истец компания предъявляла требования другим сторонам. Завершённые дела "
        "не подтверждают её текущую обязанность платить или наличие непогашенного долга."
    )
    assert (
        interpretation_problem(
            Answer(kind="answer", text_md=correct), "На что влияет количество судебных дел?", cards
        )
        is None
    )


def test_comparison_must_not_attribute_bankruptcy_to_every_candidate(snapshot):
    cards = [build_card(Tools(snapshot), inn) for inn in ["5032257375", "6165169320", "2311304742"]]
    text = "Я не рекомендую ни одну: у каждой из них имеются признаки банкротства."
    assert interpretation_problem(
        Answer(kind="comparison", text_md=text), "Кого рекомендуешь?", cards
    )


def test_explicit_bank_question_keeps_both_labels(snapshot):
    card = build_card(Tools(snapshot), "5032257375")
    _, sources = evidence_context([card], "Почему светофор и ЗСК зелёные, можно работать?")
    primary = next(k for k, c in sources.items() if c.source_path == "report.status.reasonName")
    plan = InterpretationPlan(
        answer="По отчёту я не рекомендую сотрудничество из-за банкротства.", evidence_ids=[primary]
    )
    draft = plan_draft(plan, sources, [card], "Почему светофор и ЗСК зелёные, можно работать?")
    assert {"report.baseInfo.riskLevel", "report.zskRiskLevel"} <= {
        c.source_path for c in draft.citations
    }


@pytest.mark.parametrize(
    "text",
    [
        "Я рекомендую Павличука, поскольку у него нет открытых судебных дел.",
        "У него нет текущих исполнительных производств.",
        "Нет незакрытых арбитражных дел.",
    ],
)
def test_absent_records_must_not_become_absent_open_events(text):
    answer = Answer(kind="answer", text_md=text)
    assert interpretation_problem(answer, "Что значит отсутствие дел в отчёте?")


def test_report_scoped_absence_is_allowed():
    answer = Answer(kind="answer", text_md="В отчёте нет открытых судебных дел.")
    assert interpretation_problem(answer, "Что значит отсутствие дел в отчёте?") is None


def test_report_year_is_not_silently_replaced_by_last_year():
    answer = Answer(
        kind="answer", text_md="Я не рекомендую отсрочку, поскольку в прошлом году был убыток."
    )
    assert interpretation_problem(answer, "Можно отсрочку?")


def test_later_financial_report_word_does_not_scope_earlier_absence():
    answer = Answer(
        kind="answer",
        text_md="Рекомендую А: нет судебных дел и долгов, однако в отчётности нет прибыли.",
    )
    assert interpretation_problem(answer, "Кого рекомендуешь?")
