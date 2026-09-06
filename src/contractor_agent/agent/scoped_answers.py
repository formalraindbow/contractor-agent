"""Narrow report policies: no invented bank methodology or arbitrary document lists.

These cover explicit questions about bank labels, documents, and a missing staff field.
Mixed questions outside those scopes continue through the conversational model.
"""

import re
from decimal import Decimal

from contractor_agent.agent.question import (
    BANK_LABEL,
    COURT_QUESTION,
    HEAD_IDENTITY,
    STAFF_QUESTION,
    limitation,
    needs_risk_review,
)
from contractor_agent.agent.requests import requested_documents
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import VERDICT_RU, Verdict
from contractor_agent.signals.text import rub

LABEL = BANK_LABEL
DOCUMENTS = re.compile(
    r"что\s+(?:(?:мне|нам|у них)\s+)?(?:запросить|уточнить)|какие документы|список документов", re.I
)
STAFF = STAFF_QUESTION
PAYMENT = re.compile(r"плат[её]ж|платить|оплат|переводить", re.I)
FLAG_MEANINGS = (
    (
        re.compile(r"массов\w*\s+адрес|адрес.{0,25}массов", re.I),
        "flag_massAddress",
        "Массовый адрес регистрации",
        "Это означает, что по адресу зарегистрировано много организаций. "
        "Отметка не равнозначна недостоверному адресу и сама по себе не доказывает "
        "фиктивность компании, нарушение или сокрытие владельцев.",
    ),
    (
        re.compile(r"недостоверн.{0,40}регистрационн|регистрационн.{0,40}недостоверн", re.I),
        "flag_invalidRegistrationData",
        "Недостоверные регистрационные сведения",
        "Отметка указывает на сомнение в указанных сведениях, но сама по себе не означает, "
        "что компания не существует. Какие именно сведения вызвали отметку и почему, "
        "можно установить только по дополнительным документам.",
    ),
    (
        re.compile(
            r"недостоверн.{0,40}(?:руководител|директор)|(?:руководител|директор).{0,40}недостоверн",
            re.I,
        ),
        "flag_invalidAuthpersonsData",
        "Недостоверные сведения о руководителе",
        "В отчёте отмечена недостоверность сведений о руководителе. "
        "Достоверность сведений и полномочия на подписание конкретного документа "
        "по одному этому отчёту подтвердить нельзя.",
    ),
    (
        re.compile(r"недостоверн.{0,20}адрес|адрес.{0,20}недостоверн", re.I),
        "flag_invalidAddress",
        "Недостоверный адрес",
        "Отметка касается адресных сведений. Она сама по себе не подтверждает, "
        "что компания не существует или прекратила деятельность.",
    ),
    (
        re.compile(r"блокиров.{0,20}сч[её]т|сч[её]т.{0,20}блокиров", re.I),
        "flag_fnsBlocking",
        "Блокировка счетов",
        "Это отметка об ограничениях по счетам. Какие именно операции ограничены "
        "и действует ли блокировка сейчас, по этому отчёту определить нельзя.",
    ),
)
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
    q = re.split(r"Речь о компании|(?:Контекст сравнения|Компании):", question, maxsplit=1)[0]
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

    meanings = [entry for entry in FLAG_MEANINGS if entry[0].search(q)]
    if not meanings and re.search(r"отметк.{0,25}адрес|адрес.{0,25}отметк", q, re.I):
        risk = tools.get_risk_signals(inn)
        codes = (
            {f["code"] for group in risk.data["signals"].values() for f in group}
            if risk.available
            else set()
        )
        if "flag_invalidAddress" in codes:
            meanings = [entry for entry in FLAG_MEANINGS if entry[1] == "flag_invalidAddress"]
    head_identity = HEAD_IDENTITY.search(q)
    head_date = re.search(r"когда.{0,20}назначен|дат[ауы].{0,15}назначен", q, re.I)
    head_concern = (
        re.search(r"руководител|директор", q, re.I)
        and re.search(r"подпис|полномоч|нормально|в порядке|достоверн", q, re.I)
    ) or (re.search(r"полномочи", q, re.I) and re.search(r"подпис|подтвержд", q, re.I))
    unavailable = limitation(q)
    if unavailable:
        kind = "refusal"
        if unavailable == "external_data":
            lines = [
                "Я не могу открыть внешний сайт или получить обновления из интернета. "
                "Доступен только сохранённый отчёт банка. Сведения за текущий месяц "
                "или изменения после даты отчёта по нему подтвердить нельзя."
            ]
        elif unavailable == "financial_period":
            lines = [
                "В доступном отчёте нет помесячной или квартальной финансовой отчётности. "
                "Годовые показатели нельзя выдать за результат отдельного месяца или квартала."
            ]
        elif unavailable in {"fresh_extract", "fresh_status"}:
            lines = (
                [
                    "Свежей выписки ЕГРЮЛ в доступных данных нет. Я не могу получить или скачать "
                    "её: у меня есть только сохранённый отчёт банка.",
                ]
                if unavailable == "fresh_extract"
                else [
                    "У меня нет обновлений на сегодня. Я не могу подтвердить текущий статус "
                    "или проверить, изменились ли сведения "
                    "после даты сохранённого отчёта банка."
                ]
            )
            if summary.get("status_reason"):
                claim = "Статус на дату отчёта: «" + summary["status_reason"] + "»."
                lines += ["", claim]
                cite(claim, "report.status.reasonName")
            else:
                status = (
                    "действующая"
                    if summary.get("status") == "CURRENT"
                    else "указан в сохранённом отчёте"
                )
                claim = f"В сохранённом отчёте {summary['short_name']} — {status}."
                lines += ["", claim]
                cite(claim, "report.status.status")
            if summary.get("ogrn"):
                claim = "ОГРН: " + summary["ogrn"] + "."
                lines.append(claim)
                cite(claim, "report.baseInfo.ogrn")
        elif unavailable == "payment_history":
            lines = [
                "В отчёте нет сведений о платёжной дисциплине: платит ли компания "
                "поставщикам вовремя и были ли просрочки по конкретным счетам. "
                "По этим данным нельзя подтвердить оплату в срок. "
                "Суды, финансовая отчётность и метки банка не заменяют историю платежей.",
            ]
        elif unavailable == "payment_execution":
            lines = [
                "По этому отчёту нельзя подтвердить прохождение конкретного платежа. "
                "Он не показывает текущее состояние банковского счёта или результат перевода."
            ]
            risk = tools.get_risk_signals(inn)
            signals = (
                [f for group in risk.data["signals"].values() for f in group]
                if risk.available
                else []
            )
            blocked = next((f for f in signals if f["code"] == "flag_fnsBlocking"), None)
            if blocked:
                claim = "На дату отчёта отмечена блокировка счетов по решению ФНС."
                lines += ["", claim, "Уточните действующие ограничения у банка или контрагента."]
                cite(claim, blocked["source_path"])
            else:
                lines += [
                    "",
                    "Отсутствие отметки о блокировке в сохранённом отчёте "
                    "не подтверждает, что перевод пройдёт сейчас.",
                ]
        else:
            lines = [
                "В отчёте нет сведений об адресе проживания руководителя. "
                "Юридический адрес организации — это другие сведения."
            ]
            if head_identity:
                head = summary.get("head") or {}
                if head.get("name"):
                    claim = f"Руководитель на дату отчёта — {head['name'].title()}"
                    if head.get("position"):
                        claim += ", " + head["position"].lower()
                    if head.get("since"):
                        claim += "; назначение с " + ".".join(
                            reversed(str(head["since"]).split("-"))
                        )
                    claim += "."
                    lines = [claim, "", *lines]
                    cite(claim, summary["paths"]["head"])
                    kind = "answer"
    elif re.search(
        r"какой.{0,18}статус|проверь.{0,12}статус|действует.{0,12}ли|закрыта.{0,12}ли", q, re.I
    ) and not re.search(
        r"суд|пристав|лиценз|финанс|руководител|директор|оценк|светофор|зск", q, re.I
    ):
        if summary.get("status_reason"):
            claim = "Статус на дату отчёта: «" + summary["status_reason"] + "»."
            lines = [claim]
            cite(claim, "report.status.reasonName")
        elif summary.get("status") == "CURRENT":
            claim = "В отчёте компания указана как действующая."
            lines = [claim]
            cite(claim, "report.status.status")
        else:
            lines = ["Состояние компании нельзя однозначно определить по доступному статусу."]
    elif (head_identity or head_date or head_concern) and not re.search(
        r"финанс|суд|адрес|телефон|возраст|сколько лет|почему|документ|учредител", q, re.I
    ):
        head = summary.get("head") or {}
        lines = ["### Руководитель компании"]
        fields = (
            [("since", "Дата назначения")]
            if head_date and not head_identity and not re.search(r"зовут|имя|фио|\bкто\b", q, re.I)
            else [("name", "ФИО"), ("position", "Должность"), ("since", "Дата назначения")]
        )
        for field, label in fields:
            value = head.get(field)
            if value and field == "since":
                value = ".".join(reversed(str(value).split("-")))
            elif value and field == "name":
                value = value.title()
            elif value:
                value = value.capitalize()
            claim = f"{label}: {value or 'в отчёте не указано'}."
            lines.append("- " + claim)
            cite(claim, summary["paths"]["head"])
        if re.search(r"исключ|банкрот|ликвидир", q, re.I):
            report = tools.source.get(inn)
            status = report.status
            lines += ["", "### Статус в реестре"]
            if status.reason_name:
                claim = "В отчёте: «" + status.reason_name + "»."
                lines.append(claim)
                cite(claim, "report.status.reasonName")
            if status.date:
                claim = "Дата обновления статуса: " + status.date.strftime("%d.%m.%Y") + "."
                lines.append(claim)
                cite(claim, "report.status.date")
            lines.append(
                "Точная дата исключения из реестра в отчёте не указана. "
                "Дата обновления статуса не является датой будущего исключения."
            )
        if head_concern or re.search(r"подпис|договор", q, re.I):
            risk_response = tools.get_risk_signals(inn)
            if risk_response.available:
                for group in risk_response.data["signals"].values():
                    for fact in group:
                        if fact["code"] in {
                            "status_reason",
                            "status_not_current",
                            "bankruptcy_trustee",
                            "flag_invalidAuthpersonsData",
                            "flag_invalidRegistrationData",
                        } or (head_concern and fact["code"] == "flag_fnsBlocking"):
                            lines += ["", fact["explanation"]]
                            cite(fact["explanation"], fact["source_path"])
            lines += [
                "",
                "По этому отчёту нельзя подтвердить действующие полномочия "
                "на подписание конкретного договора.",
            ]
            if head_concern and risk_response.available:
                data = risk_response.data
                lines += ["", "**По данным отчёта:** " + data["verdict_ru"] + "."]
    elif (
        len(meanings) == 1
        and re.search(
            r"что\s+(?:это\s+)?(?:вообще\s+)?(?:значит|означает)|понимать|поясни|объясни|"
            r"причин|налогов\w*\s+долг|доказ|значит",
            q,
            re.I,
        )
        and not re.search(r"суд|финанс|выруч|сколько|документ|оплат|можно|с кем|отсроч", q, re.I)
    ):
        risk_response = tools.get_risk_signals(inn)
        if not risk_response.available:
            return None
        _, code, title, meaning = meanings[0]
        fact = next(
            (
                f
                for group in risk_response.data["signals"].values()
                for f in group
                if f["code"] == code
            ),
            None,
        )
        if fact is None:
            return None
        claim = fact["explanation"].split("Такие блокировки", 1)[0].strip()
        lines = [f"### {title} у {summary['short_name']}", claim, "", meaning]
        cite(claim, fact["source_path"])
    elif (
        re.search(r"почему|что значит|что означает", q, re.I)
        and re.search(r"(?:не найден|не выявлен|нет).{0,25}(?:сигнал|риск)", q, re.I)
        and not re.search(r"покажи|сколько|документ|что запросить|оплат|отсроч", q, re.I)
    ):
        risk_response = tools.get_risk_signals(inn)
        if not risk_response.available or any(risk_response.data["signals"].values()):
            return None
        lines = [
            f"### Почему у {summary['short_name']} не найдено сигналов",
            "По доступным сведениям отчёта помощник не выявил факторов, "
            "которые срабатывают при проверке рисков.",
            "",
            "### Что есть в отчёте",
        ]
        facts = []
        if summary["status"] == "CURRENT":
            facts.append(("Компания указана как действующая.", "report.status.status"))
        if summary["sections"].get("arbitrationByStatus") == "empty":
            facts.append(("В судебной сводке нет записей о делах.", "report.arbitrationByStatus"))
        if summary["sections"].get("executionProceedings") == "empty":
            facts.append(
                ("Нет записей об исполнительных производствах.", "report.executionProceedings")
            )
        if (
            summary["sections"].get("reputationalRisks") == "present"
            and summary["counts"]["negative_flags"] == 0
        ):
            facts.append(
                (
                    "В разделе репутационных рисков нет отмеченных негативных факторов.",
                    "report.reputationalRisks",
                )
            )
        for claim, path in facts:
            lines.append("- " + claim)
            cite(claim, path)
        if not facts:
            lines.append("Доступные сведения не дали оснований выделить отдельные факторы риска.")
        gaps = risk_response.data["gaps"]
        if gaps:
            lines += ["", "### Что оценить не удалось"]
            for gap in gaps:
                lines.append("- " + gap["text"])
                cite(gap["text"], gap["source_path"])
            lines += [
                "",
                "Пропущенные данные не считаются положительными результатами проверки. "
                "Поэтому отсутствие выявленных сигналов не означает, что компания "
                "проверена по всем критериям.",
            ]
    elif (
        not COURT_QUESTION.search(q)
        and re.search(r"финанс|выручк|прибыл|ликвид|отч[её]тност", q, re.I)
        and not re.search(
            r"суд|пристав|руковод|зск|светофор|документ|оплат|отсроч|почему|что значит", q, re.I
        )
    ):
        financials = tools.get_financials(inn)
        if financials.available:
            if not re.search(r"финанс", q, re.I) or re.search(
                r"динамик|измен|почему|сравни|означает|значит|поясни|объясни", q, re.I
            ):
                return None
            requested_years = set(re.findall(r"\b20\d{2}\b", q))
            years = [
                row
                for row in financials.data["years"]
                if not requested_years or str(row["year"]) in requested_years
            ]
            for row in years:
                lines += ["", f"### {row['year']} год"]
                for field, label in [
                    ("proceeds", "Выручка"),
                    ("profit", "Прибыль / убыток"),
                    ("capitals", "Капитал и резервы"),
                ]:
                    value = row[field]
                    if field == "profit" and value is not None and Decimal(str(value)) < 0:
                        label, value = "Убыток", abs(Decimal(str(value)))
                    claim = (
                        label
                        + ": "
                        + (
                            rub(Decimal(str(value)))
                            if value is not None
                            else "строка в отчёте отсутствует"
                        )
                    )
                    lines.append("- " + claim)
                    cite(claim, row["paths"][field])
                liquidity = row["current_liquidity"]
                if liquidity is None:
                    missing = [
                        label
                        for key, label in [
                            ("current_assets", "итога оборотных активов"),
                            ("short_term_liabilities", "краткосрочных обязательств"),
                        ]
                        if row[key] is None
                    ]
                    value = (
                        "не рассчитать — в отчёте нет " + " и ".join(missing)
                        if missing
                        else "не рассчитана: краткосрочные обязательства равны нулю"
                    )
                else:
                    value = str(liquidity).replace(".", ",")
                claim = "Текущая ликвидность: " + value
                lines.append("- " + claim)
                cite(claim, row["path"])
            for year in sorted(requested_years - {str(row["year"]) for row in years}):
                lines += ["", f"В отчёте нет финансовой отчётности за {year} год."]
            if not requested_years:
                for fact in financials.data["signals"]:
                    if fact["code"] == "fin_stale":
                        lines += ["", fact["explanation"]]
                        cite(fact["explanation"], fact["source_path"])
        else:
            lines = [
                financials.note or "В отчёте нет финансовой отчётности — оценить финансы нельзя."
            ]
            cite(lines[0], "report.finReports")
    elif STAFF.search(q) and not re.search(
        r"финанс|суд|пристав|руковод|зск|светофор|лиценз|образовательн", q, re.I
    ):
        if summary["staff"] is not None:
            return None
        lines = ["В отчёте нет сведений о численности сотрудников — оценить штат нельзя."]
        cite(lines[0], "report.baseInfo.staff")
    elif (
        re.search(r"отсроч|постоплат", q, re.I)
        and re.search(
            r"можно|могу|можем|давать|дать|предостав|отгруж|отгруз|безопасно|рискован|просит|стоит|брать",
            q,
            re.I,
        )
        and not DOCUMENTS.search(q)
    ):
        risk_response = tools.get_risk_signals(inn)
        financials = tools.get_financials(inn)
        if not risk_response.available:
            return None
        risks = risk_response.data
        kind = "card"
        lines = [
            f"### Отсрочка платежа · {summary['short_name']} · ИНН {inn}",
            "Что важно: " + VERDICT_RU[Verdict(risks["verdict"])] + ".",
            "Светофор банка: "
            + summary["labels"]["svetofor"]
            + ". ЗСК: "
            + summary["labels"]["zsk"]
            + ".",
            "Это общий вывод; он не подтверждает, что компания погасит долг в запрошенный срок.",
        ]
        # The formatter overrides the model's draft. Keep decisive facts in the visible
        # answer as well as the typed card; prompt instructions cannot repair this omission.
        critical = risks["signals"].get("critical", [])
        presented_codes = {fact["code"] for fact in critical}
        if COURT_QUESTION.search(q):
            from contractor_agent.agent.section_answers import section_answer

            court_section = section_answer(tools, [inn], "Суды: роли и статусы")
            if court_section:
                lines += [
                    "",
                    *[line for line in court_section.lines if not line.startswith("Отчёт от ")],
                ]
                citations += court_section.citations
        if critical:
            lines += ["", "### Факты, важные для решения"]
            for fact in critical:
                lines.append("- " + fact["explanation"])
                cite(fact["explanation"], fact["source_path"])
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
        facts = [
            f
            for group in risks["signals"].values()
            for f in group
            if f["code"] in {"arbitration_defendant_open", "enforcement_active", "flag_fnsBlocking"}
            and f["code"] not in presented_codes
        ]
        if facts:
            lines += ["", "### Обязательства"]
        for fact in facts:
            lines.append("- " + fact["explanation"])
            cite(fact["explanation"], fact["source_path"])
        if not facts and not presented_codes.intersection(
            {"arbitration_defendant_open", "enforcement_active"}
        ):
            lines += ["", "### Обязательства"]
            lines.append(
                "В отчёте не найдено сведений о текущих исках к компании "
                "и действующих исполнительных производствах. "
                "Это не подтверждает отсутствие обязательств."
            )
            courts = tools.get_arbitration_summary(inn)
            if courts.available and courts.data["plaintiff"]["open"]["count"]:
                count = courts.data["plaintiff"]["open"]["count"]
                claim = f"Открытых дел, где компания — истец: {count}. "
                claim += "Это требования самой компании, а не иски к ней."
                lines.append(claim)
                cite(claim, "report.arbitrationByStatus.plaintiffArbitration")
        other = [
            f
            for f in risks["signals"]["moderate"]
            if f["code"] not in presented_codes | {f["code"] for f in facts}
            and not f["code"].startswith("fin_")
        ]
        if other:
            lines += ["", "### Другие факты, влияющие на вывод"]
            for fact in other:
                lines.append("- " + fact["explanation"])
                cite(fact["explanation"], fact["source_path"])
        if any(re.search(r"прибыл|ликвидн|отчётност", gap["text"], re.I) for gap in risks["gaps"]):
            lines += [
                "",
                "Для решения об отсрочке не хватает финансовых сведений. "
                "Можно запросить баланс и отчёт о финансовых результатах за два последних года. "
                "По неполным данным нельзя оценить способность погасить долг в срок.",
            ]
    elif (
        LABEL.search(q) or re.search(r"банк.{0,20}рекоменд|рекоменд.{0,30}банк", q, re.I)
    ) and not re.search(r"сотруд|руковод|сколько лет|финанс|отсроч", q, re.I):
        risk_response = tools.get_risk_signals(inn)
        if not risk_response.available:
            return None
        risks = risk_response.data
        label = f"Светофор банка: {summary['labels']['svetofor']}. ЗСК: {summary['labels']['zsk']}."
        lines = ["### Метки банка", label, "", ZSK_EXPLANATION]
        if re.search(r"банк.{0,20}рекоменд|рекоменд.{0,30}банк", q, re.I):
            lines = [
                "В отчёте приведены оценки банка, но готового решения о работе с компанией нет. "
                "Вывод помощника формируется отдельно по фактам отчёта.",
                "",
                label,
            ]
        cite(f"Светофор банка — {summary['labels']['svetofor']}", "report.baseInfo.riskLevel")
        cite(f"ЗСК — {summary['labels']['zsk']}", "report.zskRiskLevel")
        signals = [s for group in risks["signals"].values() for s in group]
        if needs_risk_review(q) or re.search(
            r"банкрот|конкурс|исключ|ликвидир|почему|противореч|отменя", q, re.I
        ):
            decisive = [s for s in signals if s["severity"] == "critical"]
            if decisive:
                lines += ["", "### Факты из отчёта"]
                for fact in decisive:
                    lines.append("- " + fact["explanation"])
                    cite(fact["explanation"], fact["source_path"])
                lines += [
                    "",
                    "Метки банка и эти сведения приводятся отдельно. "
                    "Зелёная метка не отменяет записи в отчёте. "
                    "Причину сочетания этих данных отчёт не раскрывает.",
                ]
            if needs_risk_review(q):
                lines += ["", "**По данным отчёта:** " + risks["verdict_ru"] + "."]
        blocking = next((s for s in signals if s["code"] == "flag_fnsBlocking"), None)
        if blocking:
            lines += ["", "### Ограничения по счетам", blocking["explanation"]]
            cite(blocking["explanation"], blocking["source_path"])
        if re.search(r"что.{0,20}(?:плохого|нашл|выяв)|что.{0,15}не так|рисков", q, re.I):
            lines += ["", "### Что есть в доступных сведениях"]
            relevant = [
                s
                for s in signals
                if s["severity"] in {"critical", "moderate"} and s is not blocking
            ]
            for fact in relevant:
                lines.append("- " + fact["explanation"])
                cite(fact["explanation"], fact["source_path"])
            if not relevant:
                lines.append(
                    "В доступных полях отчёта дополнительных негативных фактов не найдено. "
                    "Это не объясняет и не отменяет оценку банка."
                )
            if summary.get("registered"):
                claim = (
                    "Дата регистрации: "
                    + ".".join(reversed(str(summary["registered"]).split("-")))
                    + "."
                )
                lines.append(claim)
                cite(claim, summary["paths"]["registered"])
            lines += ["", "**По данным отчёта:** " + risks["verdict_ru"] + "."]
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
                "Что важно: " + VERDICT_RU[Verdict(risks["verdict"])] + ".",
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
