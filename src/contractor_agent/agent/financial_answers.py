"""Reported financial figures, with explicit periods and zero/missing semantics."""

import re

from contractor_agent.agent.question import COURT_QUESTION, DECISION, subject
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.agent.section_answers import money
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.finance import is_placeholder


def financial_answer(tools: Tools, inns: list[str], question: str) -> Draft | None:
    q = subject(question)
    if len(inns) != 1 or not re.search(
        r"выручк|прибыл|убыт|ликвидн|финанс|отч[её]тност|капитал|рентабельн", q, re.I
    ):
        return None
    if (
        DECISION.search(q)
        or COURT_QUESTION.search(q)
        or re.search(
            r"суд|пристав|метк|зск|светофор|руковод|документ|рекоменд|сигнал|"
            r"что значит|что означает|почему|объясни|устойчив|плат[её]жеспособ",
            q,
            re.I,
        )
    ):
        return None
    inn = inns[0]
    report = tools.source.get(inn)
    if report is None:
        return None
    response = tools.get_financials(inn)
    if not response.available:
        text = response.note or "В отчёте нет финансовой отчётности — оценить финансы нельзя."
        return Draft(
            kind="refusal",
            lines=[text],
            citations=[Citation(claim=text, source_path="report.finReports", inn=inn)],
        )
    requested = set(map(int, re.findall(r"\b20\d{2}\b", q)))
    all_rows = response.data["years"]
    recent = re.search(r"последни[ех]\s+(\d+|три|два|пять)\s+(?:лет|год)", q, re.I)
    if not requested and recent:
        amount = recent[1].lower()
        count = int(amount) if amount.isdigit() else {"три": 3, "два": 2, "пять": 5}[amount]
        latest = max(
            [row.common.year for row in report.fin_reports or [] if row.common and row.common.year]
            or [row["year"] for row in all_rows]
        )
        requested = set(range(latest - min(count, 10) + 1, latest + 1))
    rows = [row for row in all_rows if not requested or row["year"] in requested]
    explicit = []
    for pattern, field, label in [
        (r"выручк", "proceeds", "Выручка"),
        (r"прибыл|убыт", "profit", "Прибыль / убыток"),
        (r"капитал|резерв", "capitals", "Капитал и резервы"),
        (r"оборотн", "current_assets", "Оборотные активы"),
        (r"краткосроч", "short_term_liabilities", "Краткосрочные обязательства"),
        (r"ликвидн", "current_liquidity", "Текущая ликвидность"),
        (
            r"рентабельн|дол[яюи].{0,20}прибыл|прибыл.{0,30}(?:процент|%|рубль выручки)",
            "profitability_pct",
            "Доля прибыли в выручке",
        ),
    ]:
        if re.search(pattern, q, re.I):
            explicit.append((field, label))
    fields = explicit or [
        ("proceeds", "Выручка"),
        ("profit", "Прибыль / убыток"),
        ("capitals", "Капитал и резервы"),
        ("current_liquidity", "Текущая ликвидность"),
    ]
    if any(field == "profitability_pct" for field, _ in fields):
        for field, label in [("proceeds", "Выручка"), ("profit", "Прибыль / убыток")]:
            if all(field != key for key, _ in fields):
                fields.insert(0, (field, label))
    if any(field == "current_liquidity" for field, _ in fields):
        for field, label in [
            ("current_assets", "Оборотные активы"),
            ("short_term_liabilities", "Краткосрочные обязательства"),
        ]:
            if all(field != key for key, _ in fields):
                fields.insert(-1, (field, label))
    lines, citations = [], []
    for row in rows:
        lines += ["", f"### Финансовые данные за {row['year']} год"]
        for field, label in fields:
            value = row[field]
            if field == "profitability_pct":
                rendered = (
                    str(value).replace(".", ",") + " % (прибыль / выручка × 100)"
                    if value is not None
                    else "не рассчитать: нет прибыли или выручки либо выручка равна нулю"
                )
                path = row["path"]
            elif field == "current_liquidity":
                rendered = (
                    str(value).replace(".", ",")
                    if value is not None
                    else (
                        "не рассчитывается при нулевых краткосрочных обязательствах"
                        if row["short_term_liabilities"] == 0
                        else "не рассчитать: нет необходимых данных об оборотных активах "
                        "или обязательствах"
                    )
                )
                path = row["path"]
            else:
                rendered = money(value) if value is not None else "строка в отчёте отсутствует"
                path = row["paths"][field]
            text = label + ": " + rendered + "."
            lines.append("- " + text)
            citations.append(Citation(claim=text, source_path=path, inn=inn))
        if any(key == "current_liquidity" for key, _ in fields):
            liquidity = row["current_liquidity"]
            if liquidity is not None and liquidity < 1:
                lines.append(
                    "Оборотных активов меньше, чем краткосрочных обязательств на конец этого года."
                )
    if re.search(r"нул|считать.{0,10}0", q, re.I) and any(
        row.get(field) is None for row in rows for field, _ in fields
    ):
        lines += [
            "",
            "Отсутствующее значение не равно нулю. "
            "По пустой строке нельзя определить размер показателя.",
        ]
    reg = report.base_info.registration_info
    for year in sorted(requested - {row["year"] for row in rows}):
        raw_row = next(
            (row for row in report.fin_reports or [] if row.common and row.common.year == year),
            None,
        )
        if reg and reg.registration_date and year < reg.registration_date.year:
            text = (
                f"Компания зарегистрирована {reg.registration_date.strftime('%d.%m.%Y')}. "
                f"{year} год — до регистрации, поэтому он не является периодом её деятельности."
            )
            lines += ["", text]
            citations.append(
                Citation(
                    claim=text,
                    source_path="report.baseInfo.registrationInfo.registrationDate",
                    inn=inn,
                )
            )
            if raw_row and is_placeholder(raw_row):
                lines.append(
                    f"В исходных данных за {year} есть пустая строка с нулями; "
                    "её нельзя использовать как годовую отчётность компании."
                )
        else:
            claim = f"За {year} год в отчёте нет содержательной финансовой отчётности."
            lines += ["", claim]
            citations.append(Citation(claim=claim, source_path="report.finReports", inn=inn))
    if re.search(r"динамик|раст[её]т|рост|сниз|измен|сравн", q, re.I):
        if len(rows) < 2:
            lines += ["", "Для сравнения динамики нужны сопоставимые значения хотя бы за два года."]
        else:
            newer, older = rows[0], rows[-1]
            for field, label in fields:
                a, b = newer[field], older[field]
                if a is None or b is None:
                    lines += [
                        "",
                        f"{label}: динамику между {older['year']} и {newer['year']} "
                        "оценить нельзя — значение за один из периодов отсутствует.",
                    ]
                else:
                    direction = "выросла" if a > b else "снизилась" if a < b else "не изменилась"
                    lines += [
                        "",
                        f"{label}: величина {direction} между "
                        f"{older['year']} и {newer['year']} годами.",
                    ]
    return Draft(kind="answer", lines=lines, citations=citations)


def financial_explanation(tools: Tools, inns: list[str], question: str) -> Draft | None:
    q = subject(question)
    if len(inns) != 1 or not re.search(r"капитал|чист\w* актив", q, re.I):
        return None
    if COURT_QUESTION.search(q) or re.search(r"пристав|документ|отсроч|руковод", q, re.I):
        return None
    same_as_loss = bool(
        re.search(r"прибыл|убыт", q, re.I)
        and re.search(r"то же|одно и то же|разниц|отлич|равно", q, re.I)
    )
    assets_contrast = bool(
        re.search(r"актив", q, re.I) and re.search(r"почему|не отмен|разниц|означа|значит", q, re.I)
    )
    if not (same_as_loss or assets_contrast):
        return None
    data = tools.get_financials(inns[0])
    if not data.available or not data.data["years"]:
        return None
    requested = set(map(int, re.findall(r"\b20\d{2}\b", q)))
    row = next((r for r in data.data["years"] if not requested or r["year"] in requested), None)
    if row is None:
        return None
    lines = [
        "Это разные показатели. Прибыль или убыток — результат за период. "
        "Капитал и резервы — показатель баланса на конец периода. "
        "Прибыль за один год может сочетаться с отрицательным капиталом."
        if same_as_loss
        else "Объём активов сам по себе не показывает размер собственного капитала: "
        "активам могут соответствовать обязательства. Отрицательный капитал означает, "
        "что на дату баланса обязательства превышают активы.",
        "",
        f"### Что указано за {row['year']} год",
    ]
    citations = []
    fields = [("capitals", "Капитал и резервы"), ("profit", "Прибыль / убыток")]
    if assets_contrast:
        fields = [("total_assets", "Всего активов"), ("capitals", "Капитал и резервы")]
    for field, title in fields:
        value = row[field]
        claim = title + ": " + (money(value) if value is not None else "в отчёте не указано") + "."
        lines.append("- " + claim)
        citations.append(Citation(claim=claim, source_path=row["paths"][field], inn=inns[0]))
    if row["capitals"] is not None and row["capitals"] < 0:
        lines += [
            "",
            "Здесь капитал отрицательный. Это не означает, что он равен "
            "убытку за показанный год или сумме денежных средств на счетах.",
        ]
    return Draft(kind="answer", lines=lines, citations=citations)


def financial_comparison(tools: Tools, inns: list[str], question: str) -> Draft | None:
    if len(inns) < 2:
        return None
    parts = [financial_answer(tools, [inn], question) for inn in inns]
    if any(part is None for part in parts):
        return None
    lines, citations = [], []
    for inn, part in zip(inns, parts, strict=True):
        report = tools.source.get(inn)
        lines += ["", f"## {report.base_info.short_name} · ИНН {inn}", *part.lines]
        citations += part.citations
    return Draft(kind="comparison", lines=lines, citations=citations)
