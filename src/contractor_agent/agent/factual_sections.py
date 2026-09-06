"""Exact renderers for contact details, licences and procurement figures.

These data are identifiers and reported amounts: copying them through a language
model brings no benefit. Mixed analytical questions remain with the model.
"""

import re

from contractor_agent.agent.question import DECISION, EMAIL_QUESTION, needs_risk_review, subject
from contractor_agent.agent.schema import Citation, Draft
from contractor_agent.agent.section_answers import money
from contractor_agent.mcp_server.tools import Tools


def factual_sections(tools: Tools, inns: list[str], question: str) -> Draft | None:
    if len(inns) != 1:
        return None
    q = subject(question)
    q = re.sub(r"(?:а|но)\s+не\s+(?:директор\w*|руководител\w*)", "", q, flags=re.I)
    # A business goal can mention a site, phone or licence incidentally.
    # Do not replace the requested decision with a contact directory.
    if DECISION.search(q) and not re.search(r"закупк|госконтракт|тендер|госзаказ", q, re.I):
        return None
    report = tools.source.get(inns[0])
    if report is None:
        return None
    inn = inns[0]
    lines, citations = [], []
    kind = "answer"

    def fact(text, path):
        lines.append(text)
        citations.append(Citation(claim=text.lstrip("- "), source_path=path, inn=inn))

    if re.search(
        r"возраст|сколько\s+лет|когда\s+(?:был[аи]?\s+)?(?:зарегистр|откр)|дата\s+регистрац",
        q,
        re.I,
    ) and not re.search(r"руководител|директор|суд|лиценз|прибыл|выручк", q, re.I):
        info = report.base_info.registration_info
        registered = info.registration_date if info else None
        if registered:
            fact(
                "Дата регистрации: " + registered.strftime("%d.%m.%Y") + ".",
                "report.baseInfo.registrationInfo.registrationDate",
            )
            months = (
                (report.report_date.year - registered.year) * 12
                + report.report_date.month
                - registered.month
                - (report.report_date.day < registered.day)
            )
            if months >= 0:
                age = (
                    f"меньше года ({months} полных месяцев)"
                    if months < 12
                    else f"{months // 12} полных лет"
                )
                fact(
                    "Возраст на дату отчёта: " + age + ".",
                    "report.baseInfo.registrationInfo.registrationDate",
                )
                if months < 12:
                    lines.append("Ноль полных лет означает, что с регистрации ещё не прошёл год.")
            else:
                lines.append(
                    "Дата регистрации указана позже даты отчёта. Эти сведения требуют уточнения."
                )
        else:
            fact("Дата регистрации в отчёте не указана.", "report.baseInfo.registrationInfo")
    elif EMAIL_QUESTION.search(q) and not re.search(
        r"телефон|позвон|руководител|директор|суд|прибыл|выручк", q, re.I
    ):
        if report.base_info.email:
            fact("Электронная почта: " + report.base_info.email + ".", "report.baseInfo.email")
        else:
            fact("Электронная почта в отчёте не указана.", "report.baseInfo.email")
            kind = "refusal"
    elif re.search(
        r"проверк\w* (?:гос|орган)|кто.{0,15}проверял|проверял[аи]?\s+ли|"
        r"проверял.{0,15}орган|инспекц",
        q,
        re.I,
    ):
        lines = ["### Проверки государственных органов"]
        inspections = report.inspections or []
        if not inspections:
            fact("В отчёте нет сведений о проверках государственных органов.", "report.inspections")
        else:
            statuses = {
                "InspectionsViolationDetected": "Выявлены нарушения",
                "InspectionsViolationNotDetected": "Нарушений не выявлено",
                "InspectionsCanceled": "Проверка отменена",
                "InspectionsUnknownResult": "Результат не указан",
            }
            fact(f"В отчёте записей о проверках: {len(inspections)}.", "report.inspections")
            for status, label in statuses.items():
                count = sum(i.inspection_status == status for i in inspections)
                if count:
                    fact(f"- {label}: {count}.", "report.inspections")
            selected = sorted(
                enumerate(inspections), key=lambda pair: str(pair[1].start_date or ""), reverse=True
            )
            if not re.search(r"все|полный|целиком", q, re.I):
                # Include every violation even when the most recent records are neutral.
                selected = [
                    pair
                    for n, pair in enumerate(selected)
                    if n < 5 or pair[1].inspection_status == "InspectionsViolationDetected"
                ]
            lines += ["", "### Органы и даты из отчёта"]
            for i, inspection in selected:
                day = (
                    inspection.start_date.strftime("%d.%m.%Y")
                    if inspection.start_date
                    else "дата не указана"
                )
                parts = [
                    day,
                    inspection.authority_name or "орган не указан",
                    statuses.get(inspection.inspection_status, "Результат не раскрыт"),
                ]
                if inspection.form:
                    parts.append(inspection.form)
                fact("- " + "; ".join(parts) + ".", f"report.inspections[{i}]")
            if len(selected) < len(inspections):
                has_violations = any(
                    i.inspection_status == "InspectionsViolationDetected" for i in inspections
                )
                lines.append(
                    "Приведены последние пять записей"
                    + (" и все записи с выявленными нарушениями" if has_violations else "")
                    + " из отчёта."
                )
            if any(
                i.inspection_status == "InspectionsViolationDetected" for i in inspections
            ):
                lines.append(
                    "Содержание выявленных нарушений и их устранение в этих записях не раскрыты."
                )
            elif any(i.inspection_status == "InspectionsUnknownResult" for i in inspections):
                lines.append(
                    "По записям без результата нельзя установить, были ли выявлены нарушения."
                )
            if len(inspections) >= 100:
                lines.append(
                    "Отчёт содержит не более 100 записей. Полнота перечня "
                    "и наличие более новых проверок не подтверждены."
                )
    elif re.search(r"филиал", q, re.I) and not re.search(r"руковод|суд|финанс|учредит", q, re.I):
        info = report.branches_info
        lines = ["### Филиалы компании"]
        if info and info.branches_count is not None:
            fact(
                f"Число филиалов по отчёту: {info.branches_count}.",
                "report.branchesInfo.branchesCount",
            )
        if info and info.branches:
            for i, branch in enumerate(info.branches):
                fact(
                    f"- {branch.name or 'Название не указано'}: "
                    f"{branch.address or 'адрес не указан'}.",
                    f"report.branchesInfo.branches[{i}]",
                )
        else:
            fact(
                "В отчёте нет перечня филиалов с названиями и адресами.",
                "report.branchesInfo.branches" if info else "report.branchesInfo",
            )
    elif re.search(r"телефон|позвонить|номер\s+для\s+связи", q, re.I) and not re.search(
        r"руковод|директор|лиценз|финанс|учредит|адрес|суд", q, re.I
    ):
        lines = ["### Контактные телефоны"]
        phones = [(i, p) for i, p in enumerate(report.phones or []) if p.phone_number]
        if not phones:
            fact("В отчёте не указан телефон компании — номера для звонка нет.", "report.phones")
            kind = "refusal"
        else:
            for i, phone in phones:
                number = " ".join(str(v) for v in (phone.phone_code, phone.phone_number) if v)
                fact("- " + number, f"report.phones[{i}]")
    elif re.search(
        r"учредител|бенефициар|владел|кому\s+принадлежит|уставн\w*\s+капитал", q, re.I
    ) and not re.search(r"руковод|директор|телефон|суд|пристав|финанс", q, re.I):
        info = report.founders_info
        lines = ["### Учредители и владельцы"]
        founders = info.cofounders if info else None
        if not founders:
            fact(
                "В отчёте нет сведений об учредителях. Руководитель организации "
                "не обязательно является её учредителем или владельцем.",
                "report.foundersInfo.cofounders" if info else "report.foundersInfo",
            )
            kind = "refusal"
        else:
            seen_founders = set()
            for i, founder in enumerate(founders):
                identity = founder.model_dump_json()
                if identity in seen_founders:
                    continue
                seen_founders.add(identity)
                parts = [founder.name or "Имя не указано"]
                if founder.share is not None:
                    parts.append(f"доля {founder.share} %")
                if founder.amount is not None:
                    parts.append("вклад " + money(founder.amount))
                if founder.date_from:
                    parts.append("запись с " + founder.date_from.strftime("%d.%m.%Y"))
                if founder.active is False or founder.is_active is False:
                    parts.append("запись отмечена как недействующая")
                fact("- " + "; ".join(parts) + ".", f"report.foundersInfo.cofounders[{i}]")
        if re.search(r"бенефициар|кому\s+принадлежит", q, re.I):
            lines += [
                "",
                "Конечные бенефициары отдельно в отчёте не раскрыты. "
                "Записи об учредителях не заменяют подтверждение конечного владения.",
            ]
        if re.search(r"уставн", q, re.I):
            amount = info.share_capital if info else None
            fact(
                "Уставный капитал: " + money(amount) + ".",
                "report.foundersInfo.shareCapital" if info else "report.foundersInfo",
            )
        if re.search(r"связанн", q, re.I):
            related = report.related_companies or []
            lines += ["", "### Связанные компании"]
            fact(
                f"В отчёте указано связанных компаний: {len(related)}."
                if related
                else "В отчёте нет сведений о связанных компаниях.",
                "report.relatedCompanies",
            )
            for i, company in enumerate(related):
                fact(
                    f"- {company.name or 'Название не указано'} · "
                    f"ИНН {company.inn or 'не указан'}.",
                    f"report.relatedCompanies[{i}]",
                )
            if related:
                lines.append("Такая связь сама по себе не подтверждает владение или контроль.")
    elif re.search(r"сайт|соцсет", q, re.I) and not re.search(
        r"финанс|суд|руковод|долг|лиценз", q, re.I
    ):
        lines = ["### Сайт и страницы компании"]
        if report.base_info.website:
            fact("Сайт из отчёта: " + report.base_info.website + ".", "report.baseInfo.website")
        else:
            fact("В отчёте не указан сайт компании.", "report.baseInfo.website")
            kind = "refusal"
        if re.search(r"соцсет", q, re.I):
            lines.append("Официальные страницы в социальных сетях в доступных данных не приведены.")
        if re.search(r"торгуют|деятельност|занима", q, re.I):
            info = report.kinds_of_activity_info
            activity = info.main_kind_of_activity if info else None
            if activity:
                fact(
                    f"Основной вид деятельности по отчёту — ОКВЭД {activity.code}: "
                    f"{activity.description}.",
                    "report.kindsOfActivityInfo.mainKindOfActivity",
                )
                kind = "answer"
    elif re.search(r"лиценз", q, re.I) and not re.search(
        r"суд|пристав|финанс|выручк|учредит|телефон|рнп|недобросовест|"
        r"какие.{0,12}(?:нужны|нужна|требуются)|почему|что значит|можно.{0,15}работ",
        q,
        re.I,
    ):
        lines = ["### Лицензии из отчёта"]
        if not report.licenses:
            fact(
                "В отчёте нет сведений о лицензиях. Подтвердить их наличие по этим данным нельзя.",
                "report.licenses",
            )
            kind = "refusal"
            if re.search(r"доказ|незакон|запрещ|нельзя", q, re.I):
                lines += [
                    "",
                    "Отсутствие сведений о лицензиях в отчёте не доказывает "
                    "незаконность деятельности. По этому пропуску нельзя установить, "
                    "нужна ли лицензия для конкретной работы и есть ли она у компании.",
                ]
        else:
            for i, licence in enumerate(report.licenses):
                parts = [f"**№ {licence.number or 'не указан'}**"]
                if licence.name:
                    parts.append(licence.name)
                if licence.issuing_authority:
                    parts.append("Выдавший орган: " + licence.issuing_authority)
                if licence.issue_date:
                    parts.append("Выдана " + licence.issue_date.strftime("%d.%m.%Y"))
                if licence.status == "INDEFINITE":
                    parts.append("По отчёту — бессрочная")
                elif licence.status == "EXPIRED":
                    parts.append("По отчёту — срок действия истёк")
                elif licence.status == "ACTIVE":
                    parts.append("По отчёту — действует на дату отчёта")
                elif licence.end_date:
                    parts.append("Срок по отчёту — до " + licence.end_date.strftime("%d.%m.%Y"))
                elif licence.status:
                    parts.append("Срок действия в доступных данных не раскрыт")
                if licence.end_date and licence.status in {"EXPIRED", "ACTIVE"}:
                    parts.append("Дата окончания: " + licence.end_date.strftime("%d.%m.%Y"))
                fact("- " + ". ".join(parts) + ".", f"report.licenses[{i}]")
            if re.search(r"обуч|образоват|\bкурс", q, re.I) and not any(
                re.search(r"образоват|обуч", licence.name or "", re.I)
                for licence in report.licenses
            ):
                fact(
                    "В отчёте не указана лицензия на образовательную деятельность. "
                    "Перечисленные лицензии относятся к другим видам деятельности "
                    "и не подтверждают право проводить обучение.",
                    "report.licenses",
                )
    elif re.search(r"закуп(?:к|ок)|госконтракт|тендер|госзаказ", q, re.I) and not re.search(
        r"лиценз|выручк|прибыл|финанс|сколько.{0,10}суд|телефон|учредит", q, re.I
    ):
        lines = ["### Опыт государственных закупок"]
        years = set(map(int, re.findall(r"\b20\d{2}\b", q)))
        rows = [
            (i, p)
            for i, p in enumerate(report.procurements or [])
            if not years or p.procurements_year in years
        ]
        if not rows:
            fact(
                "В отчёте не найдено сведений о закупках за запрошенный период."
                if years
                else "В отчёте нет сведений об участии в государственных закупках.",
                "report.procurements",
            )
        for i, row in rows:
            lines += [
                "",
                f"**{row.procurements_year or 'Год не указан'} · "
                f"{row.federal_law_code or 'Закон не указан'}**",
            ]
            for field, title in [
                ("tender_winner_cnt", "Выигранные тендеры"),
                ("contract_signed_cnt", "Подписанные контракты"),
            ]:
                value = getattr(row, field)
                alias = "tenderWinnerCnt" if field == "tender_winner_cnt" else "contractSignedCnt"
                fact(
                    f"- {title}: {value if value is not None else 'число не указано'}.",
                    f"report.procurements[{i}].{alias}",
                )
            fact(
                "- Сумма подписанных контрактов: " + money(row.contract_signed_amt) + ".",
                f"report.procurements[{i}].contractSignedAmt",
            )
        if needs_risk_review(q) or re.search(r"субподряд|взять|работать", q, re.I):
            data = tools.get_risk_signals(inn).data
            facts = [f for group in data["signals"].values() for f in group]
            rnp = next((f for f in facts if f["code"] == "flag_dishonestProvider"), None)
            lines += ["", "### Реестр недобросовестных поставщиков"]
            if rnp:
                fact(rnp["explanation"], rnp["source_path"])
                lines.append("Основание и дата включения в этом отчёте не раскрыты.")
            else:
                lines.append(
                    "В доступных полях отчёта отметка о включении в РНП не найдена. "
                    "Текущий статус в реестре по этому снимку подтвердить нельзя."
                )
            others = [f for f in data["signals"]["critical"] if f is not rnp]
            if others:
                lines += ["", "### Другие существенные факты"]
                for f in others:
                    fact("- " + f["explanation"], f["source_path"])
            lines += ["", "**По данным отчёта:** " + data["verdict_ru"] + "."]
    else:
        return None
    return Draft(kind=kind, lines=lines, citations=citations)
