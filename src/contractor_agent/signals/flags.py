"""Признаки банка из ``reputationalRisks.negative`` — «по данным реестров ФНС».

Большинство этих фактов больше нигде в отчёте нет (блокировка счетов, массовый
адрес, недостоверные данные, должник по налогам): признак банка — единственный
источник, поэтому берём его как есть (``origin=bank_flag``), а тяжесть задаём
сами по кейсодателю. Коды с сырой опорой — производства, арбитраж, убыток —
здесь пропускаем: их считают свои модули, а флаг там — сверка.

Текст банка хранится как данные (``bank_text``), объяснение — своё: два текста
банка категоричны («Не рекомендуется сотрудничать…»), а рекомендация должна быть
мягкой. Словарь кодов открытый: неизвестный негативный код — «проверить».
Секции нет — пробел «реестры ФНС»: без неё «нарушений нет» сказать нельзя.
"""

from __future__ import annotations

from dataclasses import dataclass

from contractor_agent.data.model import Report
from contractor_agent.signals.finance import real_rows
from contractor_agent.signals.model import Gap, GapReason, Origin, RuleResult, Severity, Signal
from contractor_agent.signals.registry import VIOLATION_STATUS, is_terminal_status

SECTION_PATH = "report.reputationalRisks"


@dataclass(frozen=True)
class FlagRule:
    severity: Severity
    title_ru: str
    explanation_ru: str
    terminal: bool = False


RULES: dict[str, FlagRule] = {
    "liquidationStatus": FlagRule(
        Severity.CRITICAL,
        "В процессе ликвидации или банкротства",
        "По данным реестров ФНС компания находится в процессе ликвидации или банкротства.",
        terminal=True,
    ),
    "invalidAddress": FlagRule(
        Severity.CRITICAL,
        "Недостоверный адрес",
        "Адрес компании признан недостоверным по данным реестров ФНС — "
        "по этому адресу её может не быть.",
    ),
    "invalidAuthpersonsData": FlagRule(
        Severity.CRITICAL,
        "Недостоверные сведения о руководителе",
        "Сведения о руководителе признаны недостоверными по данным реестров ФНС.",
    ),
    "disqualifiedAuthpersons": FlagRule(
        Severity.CRITICAL,
        "Дисквалифицированный руководитель",
        "Руководитель числится в реестре дисквалифицированных лиц.",
    ),
    "fnsBlocking": FlagRule(
        Severity.MODERATE,
        "Блокировка счетов по решению ФНС",
        "На дату отчёта есть блокировка банковских счетов по постановлению налоговой. "
        "Такие блокировки часто краткосрочные — уточните, снята ли она.",
    ),
    "taxArrears": FlagRule(
        Severity.MODERATE,
        "Задолженность по налогам",
        "Компания числится в реестре должников ФНС.",
    ),
    "taxReporting": FlagRule(
        Severity.MODERATE,
        "Не сдаёт налоговую отчётность",
        "По данным ФНС компания не представляет налоговую отчётность.",
    ),
    "dishonestProvider": FlagRule(
        Severity.MODERATE,
        "Реестр недобросовестных поставщиков",
        "Компания включена в реестр недобросовестных поставщиков по госзакупкам.",
    ),
    "invalidRegistrationData": FlagRule(
        Severity.MODERATE,
        "Недостоверные регистрационные данные",
        "В ЕГРЮЛ есть отметка о недостоверности регистрационных данных.",
    ),
    "massAddress": FlagRule(
        Severity.MODERATE,
        "Массовый адрес регистрации",
        "Адрес регистрации числится как массовый — по нему зарегистрировано много компаний.",
    ),
    "massAuthpersons": FlagRule(
        Severity.MODERATE,
        "Массовый руководитель",
        "Руководитель числится как массовый — возглавляет много компаний.",
    ),
    "inspectionWithViolation": FlagRule(
        Severity.MODERATE,
        "Проверка с нарушениями (признак банка)",
        "По данным реестров у компании есть проверка, завершённая с нарушениями.",
    ),
    "currentAssets": FlagRule(
        Severity.MODERATE,
        "Оборотные активы равны нулю",
        "По последней отчётности оборотные активы компании равны нулю.",
    ),
    "massOkved": FlagRule(
        Severity.INFO,
        "ОКВЭД, характерные для однодневок",
        "Среди видов деятельности есть коды, которые по статистике ФНС часто выбирают "
        "фирмы-однодневки; сам по себе это повод присмотреться, не более.",
    ),
}

RAW_FALLBACK: dict[str, FlagRule] = {
    "executionProceedings": FlagRule(
        Severity.MODERATE,
        "Действующие производства (признак банка)",
        "По признаку банка есть действующие исполнительные производства; самого списка "
        "в отчёте нет.",
    ),
    "arbitrationDefendant": FlagRule(
        Severity.MODERATE,
        "Иски к компании (признак банка)",
        "По признаку банка есть арбитражные дела, где компания — ответчик; сырые данные "
        "отчёта не подтверждают этот признак полностью.",
    ),
    "profit": FlagRule(
        Severity.MODERATE,
        "Убыток (признак банка)",
        "По признаку банка последний год закрыт с убытком; "
        "сырые финансовые данные не подтверждают этот признак полностью.",
    ),
}
"""Коды с сырой опорой: их считают свои модули, флаг — сверка. Берём флаг, только если
сырой секции в отчёте нет — иначе факт потерялся бы целиком."""


def _raw_present(report: Report, code: str) -> bool:
    if code == "executionProceedings":
        return report.section_state("executionProceedings") != "absent"
    if code == "arbitrationDefendant":
        from contractor_agent.signals.arbitration import defendant_roles

        return bool(defendant_roles(report.arbitration_by_status).total) or any(
            (c.defendant_count or 0) > 0 for c in report.arbitration_cases or []
        )
    if code == "profit":
        rows = real_rows(report)
        return any(
            r.data.common and r.data.common.profit is not None and r.data.common.profit < 0
            for r in rows[:2]
        )
    if code == "inspectionWithViolation":
        return any(i.inspection_status == VIOLATION_STATUS for i in report.inspections or [])
    if code == "liquidationStatus":
        return is_terminal_status(report)
    return False


UNKNOWN = FlagRule(
    Severity.MODERATE,
    "Негативный признак банка",
    "Банк отметил негативный признак, которого нет в нашем справочнике — см. текст банка.",
)


def run(report: Report) -> RuleResult:
    risks = report.reputational_risks
    if risks is None:
        return RuleResult(
            [],
            [
                Gap(
                    criterion="реестры ФНС",
                    reason=GapReason.SECTION_ABSENT,
                    source_path=SECTION_PATH,
                    text_ru=(
                        "В отчёте нет раздела с признаками из реестров ФНС (блокировки счетов, "
                        "массовый адрес, недостоверные данные, долги по налогам) — по этим "
                        "критериям оценить нельзя."
                    ),
                    ask_ru="Проверьте контрагента в сервисах ФНС «Прозрачный бизнес».",
                )
            ],
        )
    signals: list[Signal] = []
    for i, flag in enumerate(risks.negative or []):
        if flag.code in RAW_FALLBACK:
            if _raw_present(report, flag.code):
                continue  # посчитано своим модулем по сырым данным
            rule = RAW_FALLBACK[flag.code]
        elif flag.code in ("inspectionWithViolation", "liquidationStatus") and _raw_present(
            report, flag.code
        ):
            continue  # тот же факт уже есть сигналом из реестра
        else:
            rule = RULES.get(flag.code, UNKNOWN)
        signals.append(
            Signal(
                code=f"flag_{flag.code}",
                severity=rule.severity,
                title_ru=rule.title_ru,
                value={"code": flag.code, "chapter": flag.chapter},
                source_path=f"{SECTION_PATH}.negative[{i}].code",
                source_paths=[f"{SECTION_PATH}.negative[{i}].name"],
                explanation_ru=rule.explanation_ru,
                origin=Origin.BANK_FLAG,
                terminal=rule.terminal,
                bank_text=flag.name,
            )
        )
    return RuleResult(signals, [])
