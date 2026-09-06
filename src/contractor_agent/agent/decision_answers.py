"""Business decisions use the report's computed facts, not improvised sums or guarantees."""

import re

from contractor_agent.agent.question import COURT_QUESTION, DECISION, subject
from contractor_agent.agent.schema import Card, Citation, Draft
from contractor_agent.agent.section_answers import section_answer
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import VERDICT_RU, Severity


def decision_answer(cards: list[Card], question: str, tools: Tools | None = None) -> Draft | None:
    q = subject(question)
    if len(cards) != 1 or not DECISION.search(q):
        return None
    # Mixed requests for precise fields still need their own evidence and composition.
    if re.search(r"покажи|перечисли|сколько|кто руковод|номер|лиценз|образовательн", q, re.I):
        return None
    card = cards[0]
    collection = bool(re.search(r"претензи|взыск|взыщу|возврат.{0,15}денег", q, re.I))
    lines = []
    if re.search(r"отзыв|клиент\w*.{0,15}довольн", q, re.I):
        lines += [
            "В отчёте нет отзывов клиентов и сведений об их удовлетворённости. "
            "По этому отчёту нельзя оценить качество товара или обслуживания.",
            "",
        ]
    if collection:
        lines += [
            "По отчёту нельзя предсказать исход суда или подтвердить, что деньги удастся взыскать. "
            "В нём нет документов по вашей сделке "
            "и данных о фактическом исполнении будущего решения.",
            "",
        ]
    lines += [
        f"**{card.name} · ИНН {card.inn}**",
        "",
        "**По данным отчёта:** " + VERDICT_RU[card.verdict] + ".",
        f"Светофор банка: {card.labels.riskLevel}. ЗСК: {card.labels.zskRiskLevel}.",
    ]
    citations = [
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
    if card.inn.startswith("0") and re.search(r"нул|начинается\s+с\s+0", q, re.I):
        claim = (
            f"В отчёте указан {len(card.inn)}-значный ИНН {card.inn}. "
            "Начальный ноль — часть идентификатора; при поиске он сохранён. "
            "Сам по себе начальный ноль не является признаком риска."
        )
        lines += ["", claim]
        citations.append(Citation(claim=claim, source_path="report.baseInfo.inn", inn=card.inn))
    sections = []
    if COURT_QUESTION.search(q):
        sections.append("суды")
    if re.search(r"пристав|исполнительн", q, re.I):
        sections.append("исполнительные производства")
    if tools and sections:
        section = section_answer(tools, [card.inn], " и ".join(sections))
        if section:
            lines += ["", *[line for line in section.lines if not line.startswith("Отчёт от ")]]
            citations += section.citations
            if COURT_QUESTION.search(q):
                lines.append(
                    "Само количество дел не показывает размер текущих требований к компании: "
                    "важно различать истца и ответчика, открытые и завершённые дела."
                )
    facts = [f for f in card.attention if f.severity != Severity.INFO]
    if facts:
        lines += ["", "### Факты для вашего решения"]
        for fact in facts:
            lines.append("- " + fact.claim)
            citations.append(Citation(claim=fact.claim, source_path=fact.source_path, inn=card.inn))
    else:
        lines += ["", "Критических и умеренных сигналов по доступным сведениям не найдено."]
        if card.gaps:
            lines.append("При этом проверка ограничена данными: " + " ".join(card.gaps))
    if collection:
        lines += [
            "",
            "Для решения по вашей претензии нужны договор, документы о поставке или работах "
            "и подтверждение задолженности. Суммы судебных требований из отчёта "
            "не равны сумме уже взысканных или фактически выплаченных денег.",
        ]
    return Draft(kind="answer", lines=lines, citations=citations)
