"""Concise follow-ups reuse the verified cards instead of repeating a full comparison."""

import re

from contractor_agent.agent.schema import Card, Citation, Draft
from contractor_agent.signals.model import VERDICT_RU, Severity, Verdict

_CHOICE = re.compile(r"с кем.{0,25}(?:лучше|работать)|кого выбрать|кто из них.{0,20}лучше", re.I)
_DOCS = re.compile(r"какие документы|что запросить", re.I)


def comparison_followup(cards: list[Card], question: str) -> Draft | None:
    if len(cards) < 2:
        return None
    question = re.split(r"(?:Контекст сравнения|Компании):|Речь о компании", question, maxsplit=1)[
        0
    ]
    documents = bool(_DOCS.search(question))
    choice = bool(_CHOICE.search(question))
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
