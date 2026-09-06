"""Concise follow-ups reuse the verified cards instead of repeating a full comparison."""

import re

from contractor_agent.agent.schema import Card, Citation, Draft
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import VERDICT_RU, Severity, Verdict

_CHOICE = re.compile(r"с кем.{0,25}(?:лучше|работать)|кого выбрать|кто из них.{0,20}лучше", re.I)
_DOCS = re.compile(r"какие документы|что запросить", re.I)


def report_comparison(cards: list[Card]) -> Draft:
    """A general comparison uses the very same attributed facts as the visible cards."""
    lines, citations = [], []
    for card in cards:
        lines += [
            "",
            f"### {card.name} · ИНН {card.inn}",
            f"Отчёт от {card.report_date.strftime('%d.%m.%Y')}.",
            "",
            "**По данным отчёта:** " + VERDICT_RU[card.verdict] + ".",
            f"Светофор банка: {card.labels.riskLevel}. ЗСК: {card.labels.zskRiskLevel}.",
        ]
        citations += [
            Citation(
                claim="Светофор: " + card.labels.riskLevel,
                source_path="report.baseInfo.riskLevel",
                inn=card.inn,
            ),
            Citation(
                claim="ЗСК: " + card.labels.zskRiskLevel,
                source_path="report.zskRiskLevel",
                inn=card.inn,
            ),
        ]
        facts = [f for f in card.attention if f.severity == Severity.CRITICAL]
        facts += [f for f in card.attention if f.severity == Severity.MODERATE][:3]
        if facts:
            lines += ["", "**Главное:**"]
            for fact in facts:
                lines.append("- " + fact.claim)
                citations.append(
                    Citation(claim=fact.claim, source_path=fact.source_path, inn=card.inn)
                )
        else:
            lines += ["", "Критических и умеренных сигналов по доступным сведениям не найдено."]
            if card.gaps:
                lines.append("При этом часть сведений в отчёте отсутствует: " + " ".join(card.gaps))
    return Draft(kind="comparison", lines=lines, citations=citations)


def comparison_followup(
    cards: list[Card], question: str, tools: Tools | None = None
) -> Draft | None:
    if len(cards) < 2:
        return None
    question = re.split(r"(?:Контекст сравнения|Компании):|Речь о компании", question, maxsplit=1)[
        0
    ]
    documents = bool(_DOCS.search(question))
    choice = bool(_CHOICE.search(question))
    explain = re.search(r"почему|как так|что.{0,20}общего", question, re.I) and re.search(
        r"одинак|разн|вывод|рекомендац|итог", question, re.I
    )
    if (
        tools
        and re.search(r"отсроч|постоплат", question, re.I)
        and re.search(r"кому|с кем|сравни|из (?:двух|них)|кто", question, re.I)
        and not re.search(r"документ|лиценз|руковод|учредител|телефон", question, re.I)
    ):
        from contractor_agent.agent.scoped_answers import scoped_answer

        lines = [
            "Отчёт не подтверждает оплату в срок. Для решения об отсрочке "
            "важны финансы и текущие обязательства каждой компании."
        ]
        citations = []
        for card in cards:
            q = "Можно дать отсрочку?"
            if re.search(r"суд|арбитраж", question, re.I):
                q += " Что известно о судах?"
            draft = scoped_answer(tools, [card.inn], q)
            if draft is None:
                return None
            lines += ["", *draft.lines, f"Отчёт от {card.report_date.strftime('%d.%m.%Y')}."]
            citations += draft.citations
        return Draft(kind="comparison", lines=lines, citations=citations)
    general = re.search(r"сравни|остальн|проверь|проверить", question, re.I) and not re.search(
        r"финанс|выручк|прибыл|убыт|ликвид|капитал|суд|арбитраж|пристав|долг|"
        r"телефон|руковод|лиценз|адрес|учредител|документ|зск|светофор|почему|"
        r"отсроч|предоплат|плат[её]ж|что запросить|рнп|закупк|связанн|владел"
        r"|бенефициар|контакт|филиал|оквэд|деятельност",
        question,
        re.I,
    )
    if general and not choice:
        return report_comparison(cards)
    if explain:
        facts_by_card = [
            [f for f in card.attention if f.severity != Severity.INFO] for card in cards
        ]
        common = set.intersection(*({f.code for f in facts} for facts in facts_by_card))
        lines = [
            "Помощник применяет одинаковые правила к фактам каждого отчёта. "
            "Размер компании и выручка сами по себе не отменяют обнаруженных факторов."
        ]
        if re.search(r"светофор|зск|метк|красн|зел[её]н", question, re.I):
            lines += [
                "Оценки банка приводятся отдельно и не пересчитываются помощником. "
                "Причины назначения этих оценок в отчёте не раскрыты."
            ]
        citations = []
        for card, facts in zip(cards, facts_by_card, strict=True):
            lines += ["", f"### {card.name}", VERDICT_RU[card.verdict] + "."]
            lines.append(
                f"Светофор банка: {card.labels.riskLevel}. ЗСК: {card.labels.zskRiskLevel}."
            )
            citations += [
                Citation(
                    claim="Светофор: " + card.labels.riskLevel,
                    source_path="report.baseInfo.riskLevel",
                    inn=card.inn,
                ),
                Citation(
                    claim="ЗСК: " + card.labels.zskRiskLevel,
                    source_path="report.zskRiskLevel",
                    inn=card.inn,
                ),
            ]
            # Different conclusions must explain the differences, not only shared factors.
            same_verdict = len({c.verdict for c in cards}) == 1
            selected = (
                [f for f in facts if f.code in common]
                if common and same_verdict
                else [f for f in facts if f.severity == Severity.CRITICAL]
                + [f for f in facts if f.severity != Severity.CRITICAL][:3]
            )
            for fact in selected:
                lines.append("- " + fact.claim)
                citations.append(
                    Citation(claim=fact.claim, source_path=fact.source_path, inn=card.inn)
                )
            if not selected:
                lines.append("Дополнительных факторов по доступным сведениям не выделено.")
        return Draft(kind="comparison", lines=lines, citations=citations)
    # Mixed questions still need the model to cover their other parts.
    if choice and (
        documents
        or re.search(r"финанс|суд|арбитраж|долг|лиценз|отсроч|предоплат|услови", question, re.I)
    ):
        return None
    if not (documents or choice) or re.search(
        r"сколько|покажи|перечисли|что с финанс", question, re.I
    ):
        return None
    citations = []
    if documents:
        lines = []
        for card in cards:
            lines += [f"### {card.name}"]
            lines += ["- " + document for document in card.ask_before] or [
                "В отчёте не выделены факты, требующие отдельного списка документов."
            ]
            citations += [
                Citation(claim=f.claim, source_path=f.source_path, inn=card.inn)
                for f in card.attention
                if f.severity != Severity.INFO
            ]
        return Draft(kind="comparison", lines=lines, citations=citations)

    rank = {Verdict.OK: 0, Verdict.CHECK: 1, Verdict.NOT_RECOMMENDED: 2}
    ordered = sorted(cards, key=lambda c: rank[c.verdict])
    best = [c for c in ordered if c.verdict == ordered[0].verdict]
    if len(best) == 1 and best[0].verdict == Verdict.OK:
        # вывод — только штатной фразой: «не выявлено факторов риска» звучит как гарантия
        lines = [f"С **{best[0].name}** по отчёту {VERDICT_RU[Verdict.OK]}."]
    else:
        lines = ["Однозначного выбора по этим отчётам нет. Ключевые различия:"]
    for card in ordered:
        facts = [f for f in card.attention if f.severity != Severity.INFO]
        if facts:
            selected = facts[:1]
            second = next((f for f in facts[1:] if "fnsBlocking" in f.code), None)
            if second or len(facts) > 1:
                selected.append(second or facts[1])
            detail = " ".join(f.claim.split("Такие блокировки", 1)[0].strip() for f in selected)
            citations += [
                Citation(claim=f.claim, source_path=f.source_path, inn=card.inn) for f in selected
            ]
        else:
            detail = "Критических и умеренных сигналов в отчёте не выделено."
        if any(re.search(r"прибыл|отчётност|ликвидност", gap, re.I) for gap in card.gaps):
            detail += " Финансовые данные неполные."
        lines += ["", f"- **{card.name}:** {detail}"]
    return Draft(kind="comparison", lines=lines, citations=citations)
