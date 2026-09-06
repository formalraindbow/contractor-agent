"""Render factual court/enforcement sections without asking the model to recalculate them."""

import re
from decimal import Decimal

from contractor_agent.agent.question import COURT_QUESTION, DECISION
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
    q = re.sub(r"финансов\w*\s+требован\w*", "требования", q, flags=re.I)
    # «иск» только как слово: «выписка» и «поиск» — не про суды
    court_q = bool(COURT_QUESTION.search(q))
    bailiff_q = bool(re.search(r"пристав|исполнительн", q, re.I))
    proportion = bool(bailiff_q and re.search(r"соотнош|соразмер|сравн.{0,20}капитал", q, re.I))
    reconcile = proportion or bool(
        re.search(
            r"сложил|сходит|сошл|различ|на самом деле|почему.{0,55}(?:числ|сводк|по годам|сумм)",
            q,
            re.I,
        )
    )
    if (
        len(inns) != 1
        or not (court_q or bailiff_q)
        or DECISION.search(q)
        or re.search(r"можно|отсроч|метк|зск|документ|запрос|финанс|руковод", q, re.I)
        or (re.search(r"почему|объясни", q, re.I) and not reconcile)
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
            if re.search(r"сумм.{0,35}(?:не указан|нет|неизвест)|требован.{0,15}нет", q, re.I):
                lines += [
                    "Если сумма требований не указана, это не означает отсутствие требования. "
                    "Его размер по этим данным определить нельзя.",
                    "",
                ]
            years = set(map(int, re.findall(r"\b20\d{2}\b", q)))
            by_year = bool(years) or bool(re.search(r"по годам", q, re.I)) or reconcile
            details = bool(
                re.search(
                    r"кто.{0,25}(?:истц|подал)|с кем|за что|предмет|шанс|проигр|выигр", q, re.I
                )
            )
            if details:
                lines += [
                    "В отчёте нет перечня сторон и предметов отдельных споров. "
                    "Назвать истцов по конкретным делам или оценить вероятность исхода нельзя.",
                    "",
                    "Ниже — только доступная сводка по ролям и статусам.",
                ]
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
            if not by_year or reconcile:
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
                    if re.search(r"открыт|не закрыт|незакрыт|сейчас", q, re.I):
                        buckets = [data[role][key] for key in ("pending", "appealed")]
                        opened = sum(b["count"] or 0 for b in buckets)
                        if opened:
                            known = [
                                b["amount"]
                                for b in buckets
                                if b["count"] and b["amount"] is not None
                            ]
                            unknown = any(b["count"] and b["amount"] is None for b in buckets)
                            total = money(sum(known)) if known else "сумма не указана"
                            if unknown and known:
                                total = "не менее " + total
                            fact(
                                f"- Открытые дела: {plural(opened, 'дело', 'дела', 'дел')}; "
                                f"сумма требований — {total}.",
                                "report.arbitrationByStatus."
                                + (
                                    "defandantArbitration"
                                    if role == "defendant"
                                    else "plaintiffArbitration"
                                ),
                            )
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
                if reconcile and data["by_years"]:
                    total = sum(
                        (r.get("defendant_count") or 0) + (r.get("plaintiff_count") or 0)
                        for r in data["by_years"]
                    )
                    all_years = [r["year"] for r in data["by_years"]]
                    fact(
                        f"В разбивке за {min(all_years)}–{max(all_years)} годы — "
                        f"{plural(total, 'дело', 'дела', 'дел')}. "
                        "Это ограниченное окно по годам. Общая сводка относится ко всему времени. "
                        "Эти итоги нельзя складывать или считать взаимозаменяемыми.",
                        "report.arbitrationCases",
                    )
    if bailiff_q:
        result = tools.get_enforcement_summary(inn)
        lines += ["", "### Производства у приставов"]
        if re.search(r"кому|взыскател|кредитор|за что|предмет|в пользу", q, re.I):
            lines += [
                "В отчёте нет сведений о взыскателях и предмете долга по каждому производству. "
                "Определить, кому и за что компания должна — налоговой, "
                "поставщику или другому лицу — нельзя.",
                "",
                "Доступна только сводка по статусам и указанным суммам.",
            ]
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
            if proportion:
                risks = tools.get_risk_signals(inn)
                if risks.available:
                    active = next(
                        (
                            f
                            for group in risks.data["signals"].values()
                            for f in group
                            if f["code"] == "enforcement_active"
                        ),
                        None,
                    )
                    if active:
                        lines += ["", "### Соотношение с масштабом компании"]
                        fact(active["explanation"], active["source_path"])
                        lines.append(
                            "Капитал — показатель баланса, а не остаток денег на счёте. "
                            "Это сравнение размеров, а не доказательство того, "
                            "что компания сможет или не сможет погасить долг."
                        )
    if any("не найдено" in line or "нет сведений" in line for line in lines):
        lines += ["", "Отсутствие записей в отчёте не доказывает отсутствие обязательств."]
    lines += ["", "Отчёт от " + report.report_date.strftime("%d.%m.%Y") + "."]
    if re.search(r"\b(?:\d{13}|\d{15})\b", q):
        lines = [f"**{report.base_info.short_name} · ИНН {inn}**", "", *lines]
    return Draft(kind="answer", lines=lines, citations=citations)
