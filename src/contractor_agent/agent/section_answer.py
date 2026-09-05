"""Резервный ответ по запрошенным разделам из воспроизводимых расчётов.

Используется после неудачной проверки черновика; не меняет исходы и не додумывает данные.
"""

from __future__ import annotations

import re

from contractor_agent.agent.schema import Citation
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.text import plural, rub


def render_sections(
    tools: Tools, inn: str, question: str, called: set[str]
) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []

    def fact(text, path):
        lines.append(text)
        citations.append(Citation(inn=inn, claim=text, source_path=path))

    if re.search(r"сотрудник|штат|численност", question, re.I) and called & {
        "get_report_summary",
        "get_section",
    }:
        response = tools.get_report_summary(inn)
        if response.available:
            value = response.data["staff"]
            if value is None:
                fact("В отчёте нет сведений о численности сотрудников.", "report.baseInfo.staff")
            else:
                fact(f"Численность сотрудников: {value}.", "report.baseInfo.staff")
    if (
        re.search(r"выручк|прибыл|убыт|финанс|капитал|ликвидн|заработ", question, re.I)
        and "get_financials" in called
    ):
        response = tools.get_financials(inn)
        if response.available:
            for row in response.data["years"][:2]:
                year = row["year"]
                for key, label in [
                    ("proceeds", "Выручка"),
                    ("profit", "Прибыль"),
                    ("capitals", "Капитал"),
                ]:
                    if key == "capitals" and not re.search(r"капитал|финанс", question, re.I):
                        continue
                    val = row[key]
                    if val is None:
                        continue
                    if key == "profit" and val < 0:
                        label, val = "Убыток", abs(val)
                    fact(f"{label} за {year} год: {rub(val)}.", row["paths"][key])
                if re.search(r"ликвидн|финанс", question, re.I):
                    val = row["current_liquidity"]
                    if val is not None:
                        fact(
                            f"Текущая ликвидность за {year} год: {str(val).replace('.', ',')}.",
                            row["paths"]["current_liquidity"],
                        )
        else:
            lines.append(response.note or "Финансовые данные недоступны.")
    if (
        re.search(r"суд|арбитраж|иск|ответчик|истец|дел", question, re.I)
        and "get_arbitration_summary" in called
    ):
        response = tools.get_arbitration_summary(inn)
        if response.available:
            for role, label in [("defendant", "Ответчик"), ("plaintiff", "Истец")]:
                present = False
                for status, description in [
                    ("pending", "в производстве"),
                    ("appealed", "обжалуются"),
                    ("finished", "завершены"),
                ]:
                    bucket = response.data[role][status]
                    if not bucket.get("path"):
                        continue
                    present = True
                    base = f"computed.arbitration.{role}.{status}"
                    count_text = plural(bucket["count"], "дело", "дела", "дел")
                    fact(
                        f"{label}, {description}: {count_text}.",
                        base + ".count",
                    )
                    if bucket["amount"]:
                        fact(
                            f"{label}, {description}: сумма требований {rub(bucket['amount'])}.",
                            base + ".amount",
                        )
                    else:
                        lines.append(f"{label}, {description}: сумма требований не указана.")
                if not present:
                    lines.append(f"{label}: записей по этой роли в сводке отчёта не найдено.")
            lines.append(
                "Сводка — за всё время; отдельного перечня дел с предметами споров в отчёте нет."
            )
        else:
            lines.append(response.note or "Сведения о судах недоступны.")
    if re.search(r"пристав|исполнительн", question, re.I) and "get_enforcement_summary" in called:
        response = tools.get_enforcement_summary(inn)
        if response.available:
            for group, label in [
                ("active", "Действующие производства"),
                ("finished", "Завершённые производства"),
            ]:
                item = response.data[group]
                count_text = plural(item["count"], "запись", "записи", "записей")
                fact(
                    f"{label}: {count_text} в отчёте.",
                    f"computed.enforcement.{group}.count",
                )
                if item["count"] and item["known_sum"]:
                    prefix = "не менее " if item["unknown_amount_count"] else ""
                    fact(
                        f"{label}: известная сумма {prefix}{rub(item['known_sum'])}.",
                        f"computed.enforcement.{group}.known_sum",
                    )
                if item["unknown_amount_count"]:
                    count_text = plural(
                        item["unknown_amount_count"], "записи", "записей", "записей"
                    )
                    fact(
                        f"{label}: у {count_text} сумма не указана.",
                        f"computed.enforcement.{group}.unknown_amount_count",
                    )
            if response.data["unknown_status_count"]:
                fact(
                    f"Неизвестен статус {response.data['unknown_status_count']} производств.",
                    "computed.enforcement.unknown_status_count",
                )
            lines.append(
                "Завершённые производства не относятся к текущим долгам. "
                "Отсутствие записей не доказывает отсутствие обязательств."
            )
        else:
            lines.append(response.note or "Сведения о производствах недоступны.")
    if {"get_section", "get_report_summary"} & called:
        for pattern, section, label in [
            (r"проверок|проверки|проверяли|инспекц", "inspections", "проверках госорганов"),
            (r"лиценз", "licenses", "лицензиях"),
            (r"филиал", "branchesInfo", "филиалах"),
        ]:
            if not re.search(pattern, question, re.I):
                continue
            response = tools.get_section(inn, section)
            if response.available:
                value = tools.source.get(inn)
                from contractor_agent.data.paths import resolve

                items = resolve(value, "report." + section)
                if isinstance(items, list):
                    fact(
                        f"Сведения о {label}: в отчёте {len(items)} записей.",
                        f"computed.sections.{section}.count",
                    )
                else:
                    lines.append(f"Раздел со сведениями о {label} в отчёте присутствует.")
            else:
                lines.append(f"В отчёте нет доступных сведений о {label}.")
            lines.append(
                "Отсутствие сведений в отчёте не означает, "
                "что таких событий или документов не было."
            )
    return "\n\n".join(lines), citations
