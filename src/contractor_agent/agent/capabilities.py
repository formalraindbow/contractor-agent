"""Truthful boundary for live registry requests; dates always come from the snapshot."""

import re

from contractor_agent.agent.schema import Answer, Citation
from contractor_agent.data.loader import ReportSource


def asks_live_registry(question: str) -> bool:
    return bool(
        re.search(r"егрюл|выписк|статус", question, re.I)
        and re.search(r"свеж|текущ|актуаль|сегодня|сейчас|получ|скача|обнов", question, re.I)
        and not re.search(r"выруч|прибыл|финанс|пристав|судеб|арбитраж", question, re.I)
    )


def registry_answer(source: ReportSource, inns: list[str]) -> Answer:
    lines = [
        "Получить свежую выписку ЕГРЮЛ или проверить изменения в реестре сейчас я не могу: "
        "у меня есть только загруженные отчёты банка."
    ]
    citations, dates = [], {}
    for inn in inns:
        report = source.get(inn)
        if report is None:
            continue
        lines += ["", f"**{report.base_info.short_name} · ИНН {inn}**"]
        if report.status.reason_name:
            claim = report.status.reason_name
            lines.append(f"Статус в имеющемся отчёте: «{claim}».")
            citations.append(Citation(claim=claim, source_path="report.status.reasonName", inn=inn))
        else:
            lines.append("В отчёте нет описания статуса в реестре.")
        lines.append(f"Отчёт от {report.report_date:%d.%m.%Y}.")
        dates[inn] = report.report_date.isoformat()
    lines += [
        "",
        "Чтобы узнать статус на сегодня, получите свежую выписку через сервис ФНС "
        "или запросите её у контрагента и проверьте раздел о состоянии юридического лица.",
    ]
    if not dates:
        lines.append("Если нужен статус из имеющегося отчёта, укажите компанию или ИНН.")
    return Answer(kind="answer", text_md="\n".join(lines), citations=citations, report_dates=dates)
