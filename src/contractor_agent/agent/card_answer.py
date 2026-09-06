"""A full report overview preserves the exact facts used by the company card.

The model still selects companies and answers open questions. It does not need to
rewrite a completed full-check card and risk dropping or changing its evidence.
"""

from contractor_agent.agent.question import is_full_review
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.mcp_server.tools import Tools


def card_answer(tools: Tools, inns: list[str], question: str) -> Draft | None:
    if len(inns) != 1 or not is_full_review(question):
        return None
    inn = inns[0]
    summary = tools.get_report_summary(inn)
    risk = tools.get_risk_signals(inn)
    if not summary.available or not risk.available:
        return None
    info, data = summary.data, risk.data
    labels = data["labels"]
    lines = [
        f"## {info['short_name']} · ИНН {inn}",
        "",
        f"**По данным отчёта:** {data['verdict_ru']}.",
        "",
        f"Оценки банка: светофор — {labels['svetofor']}, ЗСК — {labels['zsk']}.",
    ]
    citations = [
        Citation(
            claim=f"Светофор — {labels['svetofor']}",
            source_path="report.baseInfo.riskLevel",
            inn=inn,
        ),
        Citation(claim=f"ЗСК — {labels['zsk']}", source_path="report.zskRiskLevel", inn=inn),
    ]
    for severity, title in [
        ("critical", "Факты, требующие особого внимания"),
        ("moderate", "На что ещё обратить внимание"),
        ("info", "Дополнительные сведения"),
    ]:
        facts = data["signals"][severity]
        if not facts:
            continue
        lines += ["", f"### {title}"]
        for fact in facts:
            lines.append("- " + fact["explanation"])
            citations.append(
                Citation(claim=fact["explanation"], source_path=fact["source_path"], inn=inn)
            )
    if not any(data["signals"].values()):
        lines += ["", "По доступным полям отчёта сигналов не найдено."]
        if info.get("registered"):
            day = ".".join(reversed(str(info["registered"]).split("-")))
            claim = "Дата регистрации: " + day + "."
            lines += ["", claim]
            citations.append(
                Citation(claim=claim, source_path=info["paths"]["registered"], inn=inn)
            )
        if info.get("main_activity"):
            activity = info["main_activity"]
            claim = (
                f"Основной вид деятельности: {activity['description']} (ОКВЭД {activity['code']})."
            )
            lines.append(claim)
            citations.append(
                Citation(claim=claim, source_path=info["paths"]["main_activity"], inn=inn)
            )
    if data["gaps"]:
        lines += ["", "### Чего нет в отчёте"]
        for gap in data["gaps"]:
            lines.append("- " + gap["text"])
            citations.append(Citation(claim=gap["text"], source_path=gap["source_path"], inn=inn))
    return Draft(kind="card", lines=lines, citations=citations)
