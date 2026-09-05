# ruff: noqa: E501
from datetime import date

from evals.checks import check, is_refusal, mentions_report_date
from evals.gold import Gold, GoldCard, GoldQuestion

from contractor_agent.agent.schema import Answer, Card, CardLabels, Citation
from contractor_agent.signals.model import Verdict


def _card(verdict: Verdict = Verdict.NOT_RECOMMENDED) -> Card:
    return Card(
        inn="5032257375",
        name="ООО «МАКСМАРКЕТ»",
        labels=CardLabels(riskLevel="зелёный", zskRiskLevel="зелёный"),
        verdict=verdict,
        terminal=True,
        report_date=date(2026, 7, 31),
    )


def _answer(
    text: str, *, kind: str = "answer", card: Card | None = None, invalid: int = 0
) -> Answer:
    return Answer(
        kind=kind,
        text_md=text,
        card=card,
        citations=[Citation(claim="x", source_path="report.status.reasonName")],
        invalid_citations=[Citation(claim="выдумка", source_path="report.x")] * invalid,
        report_dates={"5032257375": "2026-07-31"},
    )


def test_answer_checks_must_mention_and_citations() -> None:
    q = GoldQuestion(
        id="q",
        inn="5032257375",
        type="answer",
        question="?",
        must_mention=["54 | пятьдесят четыре"],
    )
    assert check(
        q,
        _answer("У компании 54 действующих производства [report.executionProceedings]"),
        "2026-07-31",
    ).passed
    bad = check(q, _answer("Производств много."), "2026-07-31")
    assert not bad.passed and any("не названо" in f for f in bad.failures)
    invented = check(q, _answer("54 производства и выдумка", invalid=1), "2026-07-31")
    assert not invented.passed and any("невалидных цитат" in f for f in invented.failures)


def test_refuse_checks() -> None:
    q = GoldQuestion(
        id="q",
        inn="5032257375",
        type="refuse",
        question="Сколько сотрудников?",
        must_mention=["численност"],
    )
    assert check(
        q,
        _answer("В отчёте нет сведений о численности — оценить нельзя.", kind="refusal"),
        "2026-07-31",
    ).passed
    substantive = check(q, _answer("В штате 120 сотрудников."), "2026-07-31")
    assert not substantive.passed and not substantive.notes["refused"]
    assert is_refusal(_answer("Сведений о численности в отчёте нет.")) and not is_refusal(
        _answer("120 человек.")
    )


def test_card_checks_verdict_labels_facts_and_date() -> None:
    gold = Gold(
        cards=[
            GoldCard(
                inn="5032257375",
                company="ООО «МАКСМАРКЕТ»",
                report_date="2026-07-31",
                expected_verdict="not_recommended",
                terminal=True,
                must_name=["банкрот", "507 | пятьсот семь"],
            )
        ]
    )
    q = gold.questions({"card"})[0]
    good = _answer(
        "Компания признана банкротом [report.status.reasonName], 507 производств. Нужна дополнительная проверка перед взаимодействием. Отчёт от 31.07.2026.",
        kind="card",
        card=_card(),
    )
    assert check(q, good, "2026-07-31").passed
    missed = check(
        q,
        _answer(
            "Всё хорошо, можно работать. Отчёт от 31.07.2026.", kind="card", card=_card(Verdict.OK)
        ),
        "2026-07-31",
    )
    assert missed.notes["missed_critical"] and not missed.notes["verdict_match"]
    labelled = check(
        q,
        _answer(
            "Компания ненадёжная, банкрот, 507 производств. 31.07.2026", kind="card", card=_card()
        ),
        "2026-07-31",
    )
    assert any("запрещённое" in f for f in labelled.failures)
    assert mentions_report_date(_answer("отчёт от 31.07.2026"), "2026-07-31")
    assert not mentions_report_date(_answer("без даты", kind="answer"), "2026-08-01")


def test_judge_sees_card_only_for_card_questions():
    from evals.gold import load_gold
    from evals.judge import judge_messages

    from contractor_agent.agent.schema import Answer, Card, CardLabels

    gold = load_gold()
    card = Card(
        inn="5032257375",
        name="X",
        labels=CardLabels(riskLevel="зелёный", zskRiskLevel="зелёный"),
        verdict="not_recommended",
        report_date="2026-07-31",
    )
    answer = Answer(
        kind="answer", text_md="т", card=card, report_dates={"5032257375": "2026-07-31"}
    )
    by_id = {q.id: q for q in gold.questions()}
    plain = judge_messages(by_id["maksmarket-enforcement-active"], answer, "")[1].content
    card_q = judge_messages(by_id["5032257375-card"], answer, "")[1].content
    assert (
        "проверь соответствие" in plain and "проверь соответствие" in card_q
    )  # карточку видит всегда


def test_verdict_in_text_falls_back_to_phrase():
    from evals.checks import verdict_in_text

    assert verdict_in_text("Рекомендация: стоит проверить дополнительно.") == "check"
    assert (
        verdict_in_text("нужна дополнительная проверка перед взаимодействием") == "not_recommended"
    )
    assert verdict_in_text("ничего про исход") is None


def test_refusal_detected_with_words_between():
    from evals.checks import is_refusal

    from contractor_agent.agent.schema import Answer

    a = Answer(
        kind="answer",
        text_md="Сумма долга в отчёте не указана — оценить размер задолженности нельзя.",
    )
    assert is_refusal(a)
    assert not is_refusal(Answer(kind="answer", text_md="Долг 4 млн ₽ по 54 производствам."))


def test_refusal_with_reordered_words():
    from evals.checks import is_refusal

    from contractor_agent.agent.schema import Answer

    a = Answer(
        kind="answer",
        text_md="Кто подавал и за что — в отчёте сведений нет: только суммы по годам.",
    )
    assert is_refusal(a)


def test_role_synonyms_count_as_role():
    from evals.checks import _has

    assert _has("Завершённые иски к компании: 8 дел", "ответчик")
    assert _has("Компания сама подавала 20 раз", "истец")
    assert not _has("Всего 8 дел", "ответчик")


def test_removed_invalid_claim_is_not_a_visible_failure():
    q = GoldQuestion(id="q", inn="5032257375", type="answer", question="?")
    assert check(q, _answer("Подтверждённый факт.", invalid=1), "2026-07-31").passed


def test_refusal_cannot_hide_visible_invalid_claim():
    q = GoldQuestion(id="q", inn="5032257375", type="refuse", question="?")
    result = check(q, _answer("Нет данных. Выдумка.", kind="refusal", invalid=1), "2026-07-31")
    assert not result.passed and not result.notes["citations_valid"]


def test_denied_absence_claim_is_not_penalized():
    from evals.checks import _has_prohibited

    assert not _has_prohibited("Это не означает, что проверок не было.", "проверок не было")
    assert _has_prohibited(
        "Это не означает, что проверок не было. Проверок не было.", "проверок не было"
    )
    assert _has_prohibited("Проверок не было.", "проверок не было")


def test_correct_structured_verdict_cannot_hide_wrong_visible_verdict():
    q = GoldQuestion(
        id="q", inn="5032257375", type="card", question="?", expected_verdict="not_recommended"
    )
    result = check(
        q, _answer("Можно работать. 31.07.2026", kind="card", card=_card()), "2026-07-31"
    )
    assert not result.passed and not result.notes["verdict_match"]


def test_audit_matching_preserves_the_sign():
    from evals.checks import visible_invalid_citations

    answer = Answer(
        kind="answer",
        text_md="Прибыль -26 млн. Прибыль 26 млн.",
        citations=[
            Citation(claim="Прибыль -26 млн", source_path="report.finReports[0].common.profit")
        ],
        invalid_citations=[
            Citation(claim="Прибыль 26 млн", source_path="report.finReports[0].common.profit")
        ],
    )
    assert len(visible_invalid_citations(answer)) == 1
