"""Реестровые сигналы: статус и его причина, руководитель, возраст, лицензии, проверки.

* ``status.status`` ≠ CURRENT — компания закрыта: критично и терминально.
* ``status.reasonName`` по спецификации — «причина закрытия»; у действующих
  компаний это идущий процесс. По тексту: банкротство, предстоящее исключение
  из ЕГРЮЛ, ликвидация — критично и терминально; смена адреса, уменьшение
  уставного капитала — «проверить» (кейсодатель: «маркеры грядущих изменений»);
  незнакомый текст — «проверить» с цитатой.
* Руководитель — конкурсный управляющий: банкротство, видное по должности, —
  критично и терминально (независимо от reasonName).
* Компания моложе года — «проверить». Директор вступил в должность меньше
  полугода назад — «проверить», 6–11 месяцев — для сведения; учредительный
  директор (дата должности = дата регистрации) сменой не считается.
* Лицензия с истёкшим сроком, проверка с нарушениями — «проверить».
* Численность персонала: поле есть в спецификации, в снапшоте его нет —
  пробел, тот самый пример кейсодателя про отказ.

Метки банка сигналами не становятся: их не объясняем и не пересчитываем,
светофор влияет только на нижнюю границу рекомендации в ``engine``.
"""

from __future__ import annotations

from contractor_agent.data.model import Report
from contractor_agent.signals.finance import company_age_years, months_between
from contractor_agent.signals.model import Gap, GapReason, RuleResult, Severity, Signal
from contractor_agent.signals.text import date_ru, plural

YOUNG_COMPANY_YEARS = 1
DIRECTOR_RECENT_MONTHS = 6
DIRECTOR_INFO_MONTHS = 12
INSPECTIONS_CAP = 100
VIOLATION_STATUS = "InspectionsViolationDetected"

TERMINAL_REASONS = {
    "банкрот": "Банкротство",
    "несостоятельн": "Банкротство",
    "исключени": "Предстоящее исключение из ЕГРЮЛ",
    "ликвидац": "Ликвидация",
}
MODERATE_REASONS = {
    "места нахождения": "Решение о смене адреса",
    "уменьшени": "Уменьшение уставного капитала",
}
# Должность руководителя, по которой видна процедура банкротства или ликвидации.
INSOLVENCY_POSITIONS = {
    "конкурсн": "Руководит конкурсный управляющий",
    "внешн": "Руководит внешний управляющий",
    "административн": "Руководит административный управляющий",
    "ликвидатор": "Руководит ликвидатор",
    "ликвидационн": "Руководит ликвидационная комиссия",
}


def is_terminal_status(report: Report) -> bool:
    """Банкротство или ликвидация видны по причине статуса или по должности руководителя."""
    reason = (report.status.reason_name or "").casefold()
    if any(k in reason for k in TERMINAL_REASONS):
        return True
    return _insolvency_title(report) is not None


def _insolvency_title(report: Report) -> str | None:
    founders = report.founders_info
    person = founders.auth_person if founders else None
    position = (person.position_name or "").casefold() if person else ""
    if not position:
        return None
    for key, title in INSOLVENCY_POSITIONS.items():
        if key in position and ("управляющ" in position or "ликвид" in position):
            return title
    return None


def run(report: Report) -> RuleResult:
    signals: list[Signal] = []
    gaps: list[Gap] = []
    signals.extend(_status(report))
    signals.extend(_auth_person(report, gaps))
    signals.extend(_age(report))
    signals.extend(_licenses(report))
    signals.extend(_inspections(report, gaps))
    if report.base_info.staff is None:
        gaps.append(
            Gap(
                criterion="численность персонала",
                reason=GapReason.FIELD_ABSENT,
                source_path="report.baseInfo.staff",
                text_ru="В отчёте нет сведений о численности сотрудников.",
            )
        )
    return RuleResult(signals, gaps)


def _status(report: Report) -> list[Signal]:
    out: list[Signal] = []
    status = report.status
    if status.status != "CURRENT":
        out.append(
            Signal(
                code="status_not_current",
                severity=Severity.CRITICAL,
                title_ru="Компания не действует",
                value={"status": status.status},
                source_path="report.status.status",
                explanation_ru=f"Статус в реестре — {status.status}: компания не действует.",
                terminal=True,
            )
        )
    reason = status.reason_name
    if reason:
        lowered = reason.casefold()
        title = next((t for k, t in TERMINAL_REASONS.items() if k in lowered), None)
        terminal = title is not None
        if title is None:
            title = next(
                (t for k, t in MODERATE_REASONS.items() if k in lowered), "Запись о статусе"
            )
        when = f" (обновлён {date_ru(status.date)})" if status.date else ""
        out.append(
            Signal(
                code="status_reason",
                severity=Severity.CRITICAL if terminal else Severity.MODERATE,
                title_ru=title,
                value={"reason": reason, "status_date": status.date},
                source_path="report.status.reasonName",
                source_paths=["report.status.date"] if status.date else [],
                explanation_ru=f"Статус в реестре{when}: «{reason}».",
                terminal=terminal,
            )
        )
    return out


def _auth_person(report: Report, gaps: list[Gap]) -> list[Signal]:
    out: list[Signal] = []
    founders = report.founders_info
    if founders is None:
        if not _is_sole_proprietor(report):
            gaps.append(
                Gap(
                    criterion="руководитель и учредители",
                    reason=GapReason.SECTION_ABSENT,
                    source_path="report.foundersInfo",
                    text_ru=(
                        "В отчёте нет раздела о руководителе и учредителях — "
                        "оценить по этому критерию нельзя."
                    ),
                )
            )
        return out
    person = founders.auth_person
    if person is None:
        return out
    title = _insolvency_title(report)
    if title:
        since = f" с {date_ru(person.position_date)}" if person.position_date else ""
        out.append(
            Signal(
                code="bankruptcy_trustee",
                severity=Severity.CRITICAL,
                title_ru=title,
                value={"position": person.position_name, "since": person.position_date},
                source_path="report.foundersInfo.authPerson.positionName",
                source_paths=["report.foundersInfo.authPerson.positionDate"]
                if person.position_date
                else [],
                explanation_ru=(
                    f"Руководитель по данным реестра — {person.position_name}{since}: "
                    f"это признак процедуры банкротства или ликвидации."
                ),
                terminal=True,
            )
        )
    if out:
        return out  # конкурсный управляющий — не «смена директора», а банкротство
    info = report.base_info.registration_info
    registered = info.registration_date if info else None
    if person.position_date and person.position_date != registered:
        months = months_between(person.position_date, report.report_date)
        if months < DIRECTOR_INFO_MONTHS:
            recent = months < DIRECTOR_RECENT_MONTHS
            out.append(
                Signal(
                    code="director_recent",
                    severity=Severity.MODERATE if recent else Severity.INFO,
                    title_ru="Руководитель в должности меньше года",
                    value={"months": months, "since": person.position_date, "name": person.name},
                    source_path="report.foundersInfo.authPerson.positionDate",
                    source_paths=["report.foundersInfo.authPerson.name"],
                    explanation_ru=(
                        f"Руководитель{(' ' + person.name) if person.name else ''} в должности с "
                        f"{date_ru(person.position_date)} — "
                        f"{plural(months, 'месяц', 'месяца', 'месяцев')} на дату отчёта."
                    ),
                )
            )
    return out


def _age(report: Report) -> list[Signal]:
    info = report.base_info.registration_info
    if info is None or info.registration_date is None:
        return []
    years = company_age_years(report)
    if years is None or years >= YOUNG_COMPANY_YEARS:
        return []
    months = months_between(info.registration_date, report.report_date)
    return [
        Signal(
            code="young_company",
            severity=Severity.MODERATE,
            title_ru="Компания моложе года",
            value={"months": months, "registered": info.registration_date},
            source_path="report.baseInfo.registrationInfo.registrationDate",
            source_paths=["report.baseInfo.registrationInfo.yearsFromRegistration"],
            explanation_ru=(
                f"Зарегистрирована {date_ru(info.registration_date)} — "
                f"{plural(months, 'месяц', 'месяца', 'месяцев')} на дату отчёта; "
                f"истории работы ещё нет."
            ),
        )
    ]


def _licenses(report: Report) -> list[Signal]:
    expired = [
        (i, lic)
        for i, lic in enumerate(report.licenses or [])
        if lic.status == "EXPIRED" or (lic.end_date and lic.end_date < report.report_date)
    ]
    if not expired:
        return []
    names = "; ".join(lic.name or lic.number or "без названия" for _, lic in expired)
    return [
        Signal(
            code="license_expired",
            severity=Severity.MODERATE,
            title_ru="Истёкшие лицензии",
            value={"count": len(expired), "names": [lic.name for _, lic in expired]},
            source_path=f"report.licenses[{expired[0][0]}].status",
            source_paths=[f"report.licenses[{i}].endDate" for i, _ in expired[:10]],
            explanation_ru=(
                f"{plural(len(expired), 'лицензия', 'лицензии', 'лицензий')} с истёкшим сроком: "
                f"{names}."
            ),
        )
    ]


def _inspections(report: Report, gaps: list[Gap]) -> list[Signal]:
    inspections = report.inspections or []
    out: list[Signal] = []
    violations = [
        (i, x) for i, x in enumerate(inspections) if x.inspection_status == VIOLATION_STATUS
    ]
    if violations:
        i, first = violations[0]
        who = f" ({first.authority_name})" if first.authority_name else ""
        out.append(
            Signal(
                code="inspection_violation",
                severity=Severity.MODERATE,
                title_ru="Проверка выявила нарушения",
                value={"count": len(violations), "authority": first.authority_name},
                source_path=f"report.inspections[{i}].inspectionStatus",
                source_paths=[
                    f"report.inspections[{j}].inspectionStatus" for j, _ in violations[1:10]
                ],
                explanation_ru=(
                    f"{plural(len(violations), 'проверка', 'проверки', 'проверок')} "
                    f"с выявленными нарушениями{who}."
                ),
            )
        )
    if len(inspections) >= INSPECTIONS_CAP:
        gaps.append(
            Gap(
                criterion="свежие проверки",
                reason=GapReason.FIELD_ABSENT,
                source_path="report.inspections",
                text_ru=(
                    f"Список проверок в отчёте усечён до {INSPECTIONS_CAP} самых старых — "
                    f"свежих проверок в нём может не быть."
                ),
            )
        )
    return out


def _is_sole_proprietor(report: Report) -> bool:
    return len(report.base_info.inn) == 12
