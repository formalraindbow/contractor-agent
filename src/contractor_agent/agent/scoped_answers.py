"""Narrow report policies: no invented bank methodology or arbitrary document lists.

These cover explicit questions about bank labels, documents, and a missing staff field.
Mixed questions outside those scopes continue through the conversational model.
"""

import re
from decimal import Decimal

from contractor_agent.agent.requests import requested_documents
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import VERDICT_RU, Verdict
from contractor_agent.signals.text import rub

LABEL = re.compile(r"зск|светофор|метк[аиу]|оценк[аиу] банка", re.I)
DOCUMENTS = re.compile(r"что (?:запросить|уточнить)|какие документы|список документов", re.I)
STAFF = re.compile(r"сотрудник|численност|сколько.{0,10}работает|сколько.{0,10}персонал", re.I)
PAYMENT = re.compile(r"плат[её]ж|платить|оплат|переводить", re.I)
ZSK_EXPLANATION = (
    "ЗСК отражает риск вовлечённости в подозрительные операции, "
    "а не способность компании исполнить договор. "
    "[О платформе ЗСК — Банк России](https://www.cbr.ru/counteraction_m_ter/platform_zsk). "
    "Почему именно этой компании присвоена такая метка, отчёт не раскрывает. "
    "Метки приводим как есть и не пересчитываем."
)


def scoped_answer(tools: Tools, inns: list[str], question: str) -> Draft | None:
    if len(inns) != 1:
        return None
    q = question.split("Речь о компании", 1)[0]
    inn = inns[0]
    response = tools.get_report_summary(inn)
    if not response.available:
        return None
    summary = response.data
    lines: list[str] = []
    citations: list[Citation] = []
    kind = "answer"

    def cite(claim, path):
        citations.append(Citation(claim=claim, source_path=path, inn=inn))

    if STAFF.search(q) and not re.search(r"финанс|суд|пристав|руковод|зск|светофор", q, re.I):
        if summary["staff"] is not None:
            return None
        lines = ["В отчёте нет сведений о численности сотрудников — оценить штат нельзя."]
        cite(lines[0], "report.baseInfo.staff")
    elif (
        re.search(r"отсроч", q, re.I)
        and re.search(r"можно|давать|дать|предостав", q, re.I)
        and not DOCUMENTS.search(q)
    ):
        risk_response = tools.get_risk_signals(inn)
        financials = tools.get_financials(inn)
        if not risk_response.available:
            return None
        risks = risk_response.data
        kind = "card"
        lines = [
            "### Отсрочка платежа",
            "Вывод помощника по отчёту: " + VERDICT_RU[Verdict(risks["verdict"])] + ".",
            "Светофор банка: "
            + summary["labels"]["svetofor"]
            + ". ЗСК: "
            + summary["labels"]["zsk"]
            + ".",
            "Это общий вывод; он не подтверждает, что компания погасит долг в запрошенный срок.",
        ]
        if financials.available:
            for year in financials.data["years"]:
                lines += ["", f"### Финансы за {year['year']} год"]
                for field, label in [
                    ("proceeds", "Выручка"),
                    ("profit", "Прибыль / убыток"),
                    ("capitals", "Капитал и резервы"),
                ]:
                    value = year[field]
                    line = f"{label}: " + (
                        rub(Decimal(str(value)))
                        if value is not None
                        else "строка в отчёте отсутствует"
                    )
                    lines.append("- " + line)
                    cite(line, year["paths"][field])
                liquidity = year["current_liquidity"]
                line = "Текущая ликвидность: " + (
                    str(liquidity).replace(".", ",")
                    if liquidity is not None
                    else "не рассчитать — нет необходимых данных об оборотных активах "
                    "или краткосрочных обязательствах"
                )
                lines.append("- " + line)
                cite(line, year["path"])
        else:
            lines += [
                "",
                financials.note
                or "Финансовых сведений в отчёте нет — оценить возможность погашения нельзя.",
            ]
        lines += ["", "### Обязательства"]
        facts = [
            f
            for group in risks["signals"].values()
            for f in group
            if f["code"] in {"arbitration_defendant_open", "enforcement_active", "flag_fnsBlocking"}
        ]
        for fact in facts:
            lines.append("- " + fact["explanation"])
            cite(fact["explanation"], fact["source_path"])
        if not facts:
            lines.append(
                "Сигналов о текущих исках и производствах в отчёте не выделено. "
                "Это не подтверждает отсутствие обязательств."
            )
        lines += [
            "",
            "Перед решением об отсрочке запросите баланс и отчёт о финансовых результатах "
            "за два последних года. Без достаточных сведений о прибыли и ликвидности "
            "по этому отчёту нельзя оценить способность погасить долг в срок.",
        ]
    elif LABEL.search(q) and not re.search(r"сотруд|руковод|сколько лет|финанс|отсроч", q, re.I):
        risk_response = tools.get_risk_signals(inn)
        if not risk_response.available:
            return None
        risks = risk_response.data
        label = f"Светофор банка: {summary['labels']['svetofor']}. ЗСК: {summary['labels']['zsk']}."
        lines = ["### Метки банка", label, "", ZSK_EXPLANATION]
        cite(f"Светофор банка — {summary['labels']['svetofor']}", "report.baseInfo.riskLevel")
        cite(f"ЗСК — {summary['labels']['zsk']}", "report.zskRiskLevel")
        signals = [s for group in risks["signals"].values() for s in group]
        blocking = next((s for s in signals if s["code"] == "flag_fnsBlocking"), None)
        if blocking:
            lines += ["", "### Ограничения по счетам", blocking["explanation"]]
            cite(blocking["explanation"], blocking["source_path"])
        if PAYMENT.search(q):
            kind = "card"
            lines += [
                "",
                "### Можно ли платить?",
                "По отчёту нельзя подтвердить, что конкретный платёж пройдёт. "
                "Перед оплатой нужны актуальный статус ограничений, "
                "реквизиты и основание платежа.",
                "",
                "Что ещё влияет на решение:",
            ]
            facts = [
                s
                for s in signals
                if s["code"]
                in {
                    "arbitration_defendant_open",
                    "fin_loss",
                    "bankruptcy_trustee",
                    "status_reason",
                    "enforcement_active",
                }
            ]
            for fact in facts[:4]:
                lines.append("- " + fact["explanation"])
                cite(fact["explanation"], fact["source_path"])
            lines += [
                "",
                "Вывод помощника по отчёту: " + VERDICT_RU[Verdict(risks["verdict"])] + ".",
            ]
    elif DOCUMENTS.search(q) and not re.search(r"сколько|кто руковод|почему", q, re.I):
        risk_response = tools.get_risk_signals(inn)
        if not risk_response.available:
            return None
        risks = risk_response.data
        docs = requested_documents(risks, tools.get_financials(inn))
        if PAYMENT.search(question):
            docs = [d for d in docs if not d.startswith("Баланс")]
        lines = ["### Документы, которые стоит запросить"]
        lines += (
            ["- " + d for d in docs]
            if docs
            else [
                "В отчёте не выделены обстоятельства, требующие отдельного списка документов. "
                "Состав документов зависит от цели проверки."
            ]
        )
        for group in risks["signals"].values():
            for fact in group:
                if fact["severity"] != "info":
                    cite(fact["explanation"], fact["source_path"])
    else:
        return None
    lines += ["", "Отчёт от " + ".".join(reversed(response.report_date.split("-"))) + "."]
    return Draft(kind=kind, lines=lines, citations=citations)
