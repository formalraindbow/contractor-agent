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
    invented = check(q, _answer("54 производства", invalid=1), "2026-07-31")
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
        "Компания признана банкротом [report.status.reasonName], 507 производств. Только на условиях. Отчёт от 31.07.2026.",
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
