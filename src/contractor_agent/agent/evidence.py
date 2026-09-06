"""Minimum evidence required by the question, independent of model tool choices.

This is an evidence gate, not an alternative data source: missing reads are executed
by the same MCP ToolNode and included in the ordinary trace and finalizer context.
"""

import re
from uuid import uuid4

from contractor_agent.agent.question import (
    BANK_LABEL,
    COURT_QUESTION,
    is_full_review,
    limitation,
    needs_risk_review,
    subject,
)
from contractor_agent.agent.state import ToolCallTrace


def missing_reads(question: str, inns: list[str], trace: list[ToolCallTrace]) -> list[dict]:
    q = subject(question)
    reads = [("get_report_summary", {})]
    if COURT_QUESTION.search(q) or re.search(r"претензи|взыск|взыщу", q, re.I):
        reads.append(("get_arbitration_summary", {}))
    if (
        is_full_review(q)
        or needs_risk_review(q)
        or limitation(q) == "payment_execution"
        or BANK_LABEL.search(q)
        or re.search(r"соотнош|соразмер", q, re.I)
        or re.search(
            r"сравни|что запросить|какие документы|объясни.{0,20}(?:вывод|рекомендац)"
            r"|почему.{0,40}сигнал",
            q,
            re.I,
        )
    ):
        reads.append(("get_risk_signals", {}))
    for pattern, name in [
        (r"финанс|выручк|прибыл|ликвидн|отсроч|постоплат", "get_financials"),
        (r"пристав|исполнительн|отсроч|претензи|взыск|взыщу", "get_enforcement_summary"),
    ]:
        if re.search(pattern, q, re.I):
            reads.append((name, {}))
    for pattern, section in [
        (r"лиценз|обуч|образоват|удостоверен", "licenses"),
        (r"телефон|позвон|контакт", "phones"),
        (
            r"учредител|руководител|директор|бенефициар|владел|уставн|кому принадлежит",
            "foundersInfo",
        ),
        (r"связанн", "relatedCompanies"),
        (r"закуп(?:к|ок)|госконтракт|тендер|госзаказ", "procurements"),
        (r"филиал", "branchesInfo"),
        (r"оквэд|вид\w* деятельн|профиль|занима[ею]|торгуют|обуч|образоват", "kindsOfActivityInfo"),
        (r"проверк\w* (?:гос|орган)|проверял|инспекц|нарушен", "inspections"),
    ]:
        if re.search(pattern, q, re.I):
            reads.append(("get_section", {"name": section}))

    def covered(name, args):
        for call in trace:
            if call.reason == "tool_error":
                continue
            if call.name == name and call.args == args:
                return True
            if (
                call.name == "compare_companies"
                and call.available
                and args["inn"] in call.args.get("inns", [])
                and name in ("get_report_summary", "get_risk_signals")
            ):
                return True
        return False

    return [
        {"name": name, "args": args, "id": "evidence-" + uuid4().hex, "type": "tool_call"}
        for inn in dict.fromkeys(inns)
        for name, extra in reads
        if not covered(name, args := {"inn": inn, **extra})
    ]
