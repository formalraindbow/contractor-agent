"""Render factual court/enforcement sections without asking the model to recalculate them."""

import re
from decimal import Decimal

from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.text import plural


def money(value) -> str:
    if value is None:
        return "сумма в отчёте не указана"
    text = format(Decimal(str(value)), ",.2f").replace(",", " ").replace(".", ",")
    return text.rstrip("0").rstrip(",") + " ₽"


def section_answer(tools: Tools, inns: list[str], question: str) -> Draft | None:
    q = question.split("Речь о компании", 1)[0]
    # «иск» только как слово: «выписка» и «поиск» — не про суды
    court_q = bool(re.search(r"\bсуд|арбитраж|\bиск(?:и|ов|ах|ам|ами|е|а|у)?\b", q, re.I))
    bailiff_q = bool(re.search(r"пристав|исполнительн", q, re.I))
    if (
        len(inns) != 1
        or not (court_q or bailiff_q)
        or re.search(
            r"можно|отсроч|метк|зск|почему|документ|запрос|финанс|руковод|объясни", q, re.I
        )
    ):
        return None
    inn = inns[0]
    report = tools.source.get(inn)
    if report is None:
        return None
    lines, citations = [], []

    def fact(text, path):
        lines.append(text)
        citations.append(Citation(claim=text.lstrip("- "), source_path=path, inn=inn))

    if court_q:
        courts = tools.get_arbitration_summary(inn)
        lines.append("### Арбитражные дела")
        if not courts.available:
            fact(
                "В отчёте нет сведений об арбитражных делах — оценить нельзя.",
                "report.arbitrationByStatus",
            )
        else:
            data = courts.data
            years = set(map(int, re.findall(r"\b20\d{2}\b", q)))
            by_year = bool(years) or bool(re.search(r"по годам", q, re.I))
            if by_year:
                selected = [r for r in data["by_years"] if not years or r["year"] in years]
                for row in selected:
                    lines += ["", f"**{row['year']} год**"]
                    for role, label in [("defendant", "Ответчик"), ("plaintiff", "Истец")]:
                        count = row[role + "_count"]
                        if count:
                            fact(
                                f"- {label}: {plural(count, 'дело', 'дела', 'дел')}; "
                                f"сумма требований — {money(row[role + '_amount'])}.",
                                row["path"],
                            )
                        else:
                            fact(
                                f"- {label}: записей по этой роли в разбивке за год не найдено.",
                                row["path"] + "." + role + "Count",
                            )
                if not selected:
                    fact(
                        "Записей за запрошенный период в разбивке отчёта не найдено.",
                        "report.arbitrationCases",
                    )
                lines += [
                    "",
                    "Разбивка за год не указывает, завершены ли эти дела. "
                    "Стороны и предметы споров (с кем и за что судились) "
                    "в отчёте не раскрыты.",
                ]
            else:
                if data["common_count"] is not None:
                    fact(
                        "Всего в сводке за всё время: "
                        f"{plural(data['common_count'], 'дело', 'дела', 'дел')}.",
                        "report.arbitrationByStatus.commonCount",
                    )
                for role, label in [
                    ("defendant", "Иски к компании — ответчик"),
                    ("plaintiff", "Иски компании — истец"),
                ]:
                    lines += ["", f"**{label}**"]
                    found = False
                    for status, name in [
                        ("pending", "В производстве"),
                        ("appealed", "Обжалуются"),
                        ("finished", "Завершены"),
                    ]:
                        bucket = data[role][status]
                        if not bucket["count"]:
                            continue
                        found = True
                        count = bucket["count"]
                        fact(
                            f"- {name}: {plural(count, 'дело', 'дела', 'дел')}; "
                            f"сумма требований — {money(bucket['amount'])}.",
                            bucket["path"].rsplit(".", 1)[0],
                        )
                    if not found:
                        fact(
                            "В сводке по статусам записей по этой роли не найдено.",
                            "report.arbitrationByStatus."
                            + (
                                "defandantArbitration"
                                if role == "defendant"
                                else "plaintiffArbitration"
                            ),
                        )
                for signal in data["signals"]:
                    if "mismatch" in signal["code"]:
                        fact(signal["explanation"], signal["source_path"])
                lines += [
                    "",
                    "Завершённые дела не являются текущими претензиями. "
                    "Отдельного перечня сторон и предметов споров в отчёте нет.",
                ]
    if bailiff_q:
        result = tools.get_enforcement_summary(inn)
        lines += ["", "### Производства у приставов"]
        if not result.available:
            fact(
                "В отчёте нет сведений об исполнительных производствах — оценить нельзя.",
                "report.executionProceedings",
            )
        else:
            for key, label in [("active", "Действующие"), ("finished", "Завершённые")]:
                group = result.data[key]
                if not group["count"]:
                    fact(f"- {label}: записей в отчёте не найдено.", "report.executionProceedings")
                    continue
                count, unknown = group["count"], group["unknown_amount_count"]
                amount = money(group["known_sum"])
                if unknown == count:
                    amount = "суммы в отчёте не указаны"
                elif unknown:
                    amount = "не менее " + amount + f"; у {unknown} записей сумма не указана"
                fact(
                    f"- {label}: "
                    f"{plural(count, 'производство', 'производства', 'производств')}; "
                    f"{amount}.",
                    "report.executionProceedings",
                )
            if result.data["unknown_status_count"]:
                fact(
                    f"У {result.data['unknown_status_count']} записей статус не указан.",
                    "report.executionProceedings",
                )
            lines += ["", "Завершённые производства не относятся к текущим долгам."]
    lines += [
        "",
        "Отсутствие записей в отчёте не доказывает отсутствие обязательств.",
        "",
        "Отчёт от " + report.report_date.strftime("%d.%m.%Y") + ".",
    ]
    return Draft(kind="answer", lines=lines, citations=citations)
