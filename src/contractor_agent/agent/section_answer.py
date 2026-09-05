"""Резервный ответ по запрошенным разделам из воспроизводимых расчётов.

Используется после неудачной проверки черновика; не меняет исходы и не додумывает данные.
"""

from __future__ import annotations

import re

from contractor_agent.agent.schema import Citation
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.text import plural, rub


def asks_registration_status(question: str) -> bool:
    return bool(
        re.search(r"\bстатус|действует ли|закрыт[ао]? ли", question, re.I)
        and not re.search(r"суд|арбитраж|иск|производств|лиценз|сч[её]т|плат[её]ж", question, re.I)
    )


def needs_section_layout(text: str, question: str) -> bool:
    """Long numerical answers need topic blocks, including valid model drafts."""
    if not re.search(r"суд|арбитраж|пристав|исполнительн|финанс|выручк|прибыл", question, re.I):
        return False
    if re.search(r"^(?:#{1,3}\s|\*\*[^*\n]+\*\*:?[ \t]*$)", text, re.M):
        return False
    return sum(bool(re.search(r"\d", line)) for line in text.splitlines()) >= 3


def render_sections(
    tools: Tools, inn: str, question: str, called: set[str]
) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []

    def cite(text, path):
        citations.append(Citation(inn=inn, claim=text, source_path=path))
        return text

    def fact(text, path):
        lines.append(cite(text, path))

    def add_section(title, rows):
        lines.append(f"### {title}\n\n" + "\n".join(rows))

    if asks_registration_status(question) and "get_report_summary" in called:
        response = tools.get_report_summary(inn)
        if response.available:
            if response.data["status_reason"]:
                fact(
                    "Статус в отчёте: «" + response.data["status_reason"] + "».",
                    "report.status.reasonName",
                )
            elif response.data["status"] == "CURRENT":
                fact("В отчёте контрагент указан как действующий.", "report.status.status")
            else:
                lines.append("В отчёте нет пояснения статуса контрагента.")
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
            rows = []
            for row in response.data["years"][:2]:
                year = row["year"]
                metrics = []
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
                    metrics.append("- " + cite(f"{label} — **{rub(val)}**.", row["paths"][key]))
                if re.search(r"ликвидн|финанс", question, re.I):
                    val = row["current_liquidity"]
                    if val is not None:
                        metrics.append(
                            "- "
                            + cite(
                                f"Текущая ликвидность — **{str(val).replace('.', ',')}**.",
                                row["paths"]["current_liquidity"],
                            )
                        )
                if metrics:
                    rows.extend([f"**{year} год**", *metrics, ""])
            add_section(
                "Финансы", rows or ["Запрошенные финансовые показатели в отчёте не указаны."]
            )
        else:
            add_section("Финансы", [response.note or "Финансовые данные недоступны."])
    if (
        re.search(r"суд|арбитраж|иск|ответчик|истец|дел", question, re.I)
        and "get_arbitration_summary" in called
    ):
        response = tools.get_arbitration_summary(inn)
        if response.available:
            for role, title in [
                ("defendant", "Иски к контрагенту (ответчик)"),
                ("plaintiff", "Иски контрагента (истец)"),
            ]:
                rows = []
                for status, description in [
                    ("pending", "В производстве"),
                    ("appealed", "Обжалуются"),
                    ("finished", "Завершены"),
                ]:
                    bucket = response.data[role][status]
                    if not bucket.get("path"):
                        continue
                    base = f"computed.arbitration.{role}.{status}"
                    count_text = plural(bucket["count"], "дело", "дела", "дел")
                    count = cite(
                        f"{description}: **{count_text}**",
                        base + ".count",
                    )
                    if bucket["amount"] is not None:
                        amount = cite(
                            f"сумма требований — **{rub(bucket['amount'])}**",
                            base + ".amount",
                        )
                        rows.append(f"- {count}; {amount}.")
                    elif bucket["count"]:
                        rows.append(f"- {count}; сумма требований не указана.")
                    else:
                        rows.append(f"- {count}.")
                add_section(title, rows or ["Записей по этой роли в сводке отчёта не найдено."])
            lines.append(
                "Сводка — за всё время; отдельного перечня дел с предметами споров в отчёте нет."
            )
        else:
            add_section("Судебные дела", [response.note or "Сведения о судах недоступны."])
    if re.search(r"пристав|исполнительн", question, re.I) and "get_enforcement_summary" in called:
        response = tools.get_enforcement_summary(inn)
        if response.available:
            rows = []
            for group, label in [
                ("active", "Действующие"),
                ("finished", "Завершённые"),
            ]:
                item = response.data[group]
                count_text = plural(item["count"], "производство", "производства", "производств")
                count = cite(
                    f"{label}: **{count_text}**",
                    f"computed.enforcement.{group}.count",
                )
                parts = [count]
                if item["count"] and (item["known_sum"] or not item["unknown_amount_count"]):
                    prefix = "не менее " if item["unknown_amount_count"] else ""
                    parts.append(
                        cite(
                            f"известная сумма — **{prefix}{rub(item['known_sum'])}**",
                            f"computed.enforcement.{group}.known_sum",
                        )
                    )
                row = "- " + "; ".join(parts) + "."
                if item["unknown_amount_count"]:
                    count_text = plural(
                        item["unknown_amount_count"],
                        "производстве",
                        "производствах",
                        "производствах",
                    )
                    row += " " + cite(
                        f"В {count_text} сумма не указана.",
                        f"computed.enforcement.{group}.unknown_amount_count",
                    )
                rows.append(row)
            if response.data["unknown_status_count"]:
                count_text = plural(
                    response.data["unknown_status_count"],
                    "производства",
                    "производств",
                    "производств",
                )
                rows.append(
                    "- "
                    + cite(
                        f"Статус {count_text} не указан.",
                        "computed.enforcement.unknown_status_count",
                    )
                )
            add_section("Производства у приставов", rows)
            lines.append(
                "Завершённые производства не относятся к текущим долгам. "
                "Отсутствие записей не доказывает отсутствие обязательств."
            )
        else:
            add_section(
                "Производства у приставов",
                [response.note or "Сведения о производствах недоступны."],
            )
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
