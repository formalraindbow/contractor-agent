"""`Report` — контракт на схему отчёта о контрагенте.

Одна pydantic-модель для обоих форматов снапшота: на вход — «плоское дерево»
(JSON после ``mongo.unwrap`` или CSV после ``csv_tree``), на выходе — типизированный
отчёт. Типизация идёт по пути поля, не по виду значения: ИНН ``0277985654`` — строка,
``executionProceedings[].amount`` ``"517235.54"`` — ``Decimal``, ``"true"`` — ``bool``.

Правила модели:

* ``None`` — «в отчёте нет», ``[]`` — «есть и пусто». Это разные состояния, оба
  реальны в источнике; ``section_state`` возвращает ``absent | empty | present``.
* Все секции, кроме ``reportDate``, ``baseInfo``, ``status``, ``zskRiskLevel``,
  опциональны; подполя опциональны поодиночке (``*Count`` бывает без ``*Amount``).
* Имена полей в Python — snake_case, алиасы — camelCase отчёта (``to_camel``),
  ``model_dump()`` отдаёт дерево с адресами как в источнике — это ``source_path``.
  Опечатка ``defandantArbitration`` сохранена: это адрес поля, а не наша ошибка.
* Даты — ``MoscowDate``: ``$date`` в источнике — московская полночь, у дат до 2014
  года в строке ``20:00:00Z`` (UTC+4), поэтому только ``zoneinfo``, не «+3 часа».
* ``extra="allow"``: неизвестное поле плавающей схемы не роняет загрузку и не
  теряется; тест на снапшоте утверждает, что таких полей нет.
* Пустая строка — «данных нет» → ``None`` (в источнике так пустой адрес у ИП).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, BeforeValidator, ConfigDict, field_validator
from pydantic.alias_generators import to_camel

MOSCOW = ZoneInfo("Europe/Moscow")

SectionState = Literal["absent", "empty", "present"]


def parse_moscow_date(value: Any) -> Any:
    """ISO-строка с ``Z`` → московская календарная дата; ``YYYY-MM-DD`` → как есть."""
    if not isinstance(value, str):
        return value  # date / datetime — дальше проверит pydantic
    if len(value) == 10:
        return date.fromisoformat(value)
    if value.endswith("Z"):
        moment = datetime.fromisoformat(value[:-1]).replace(tzinfo=UTC)
        return moment.astimezone(MOSCOW).date()
    raise ValueError(f"дата не в формате снапшота: {value!r}")


MoscowDate = Annotated[date, BeforeValidator(parse_moscow_date)]
"""Дата отчёта в московском календаре — единственный тип дат в модели."""

# Кириллические буквы, неотличимые от латинских: в источнике код
# ``аrbitrationDefendant`` начинается с кириллической «а» (U+0430).
_HOMOGLYPHS = str.maketrans("аеорсху", "aeopcxy")


class ReportModel(BaseModel):
    """База для всех узлов отчёта: алиасы camelCase, лишние поля сохраняем."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="allow",
        str_strip_whitespace=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def _empty_string_is_absent(cls, value: Any) -> Any:
        return None if value == "" else value


# --- baseInfo ------------------------------------------------------------------


class RegistrationInfo(ReportModel):
    registration_date: MoscowDate | None = None
    years_from_registration: int | None = None


class BaseInfo(ReportModel):
    inn: str
    ogrn: str
    short_name: str
    full_name: str | None = None
    okpo: str | None = None
    kpp: str | None = None
    registration_info: RegistrationInfo | None = None
    risk_level: str  # LOW | MEDIUM | HIGH | UNKNOWN — метка банка, приводим как есть
    address: str | None = None
    company_size: str | None = None
    staff: str | None = None  # диапазон численности — есть в спецификации, нет в снапшоте
    email: str | None = None
    website: str | None = None


class Status(ReportModel):
    status: str  # в снапшоте всегда CURRENT
    date: MoscowDate | None = None
    reason_name: str | None = None  # банкротство, исключение из ЕГРЮЛ — читать обязательно


# --- reputationalRisks --------------------------------------------------------


class RiskFlag(ReportModel):
    code: str
    name: str | None = None
    chapter: str | None = None

    @field_validator("code")
    @classmethod
    def _latin_code(cls, value: str) -> str:
        return value.translate(_HOMOGLYPHS)


class ReputationalRisks(ReportModel):
    negative: list[RiskFlag] | None = None
    positive: list[RiskFlag] | None = None


# --- kindsOfActivityInfo --------------------------------------------------------


class Okved(ReportModel):
    code: str
    description: str | None = None


class KindsOfActivityInfo(ReportModel):
    main_kind_of_activity: Okved | None = None
    other_kinds_of_activity: list[Okved] | None = None


# --- foundersInfo ---------------------------------------------------------------


class AuthPerson(ReportModel):
    inn: str | None = None
    name: str | None = None
    position_name: str | None = None
    position_date: MoscowDate | None = None


class Cofounder(ReportModel):
    inn: str | None = None
    name: str | None = None
    amount: int | None = None
    share: int | None = None
    date_from: MoscowDate | None = None
    active: bool | None = None
    is_active: bool | None = None  # так поле названо в спецификации; в снапшоте — ``active``


class ParentOrganization(ReportModel):
    inn: str | None = None
    ogrn: str | None = None
    full_name: str | None = None
    parent_date: MoscowDate | None = None


class FoundersInfo(ReportModel):
    share_capital: int | None = None
    auth_person: AuthPerson | None = None
    cofounders: list[Cofounder] | None = None
    parent_organizations: list[ParentOrganization] | None = None  # по спецификации; в снапшоте нет


# --- finReports / coefficient ---------------------------------------------------


class FinCommon(ReportModel):
    year: int | None = None
    proceeds: int | None = None  # рубли
    profit: int | None = None


class CurrentAssets(ReportModel):
    total: int | None = None
    receivables: int | None = None
    stocks: int | None = None
    bankroll: int | None = None


class UncurrentAssets(ReportModel):
    total: int | None = None
    fixed_assets: int | None = None


class Assets(ReportModel):
    total_assets: int | None = None
    current_assets: CurrentAssets | None = None
    uncurrent_assets: UncurrentAssets | None = None


class ShortTermLiabilities(ReportModel):
    total: int | None = None
    accounts_payable: int | None = None
    borrowed_funds: int | None = None


class LongTermDuties(ReportModel):
    total: int | None = None
    others: int | None = None


class Liabilities(ReportModel):
    capitals: int | None = None  # бывает отрицательным
    total_liabilities: int | None = None
    short_term_liabilities: ShortTermLiabilities | None = None
    long_term_duties: LongTermDuties | None = None


class FinReport(ReportModel):
    common: FinCommon | None = None
    assets: Assets | None = None
    liabilities: Liabilities | None = None


class Coefficient(ReportModel):
    year: int | None = None
    profitability: Decimal | None = None
    solvency: Decimal | None = None
    sustainability: Decimal | None = None


# --- arbitrationByStatus / arbitrationCases -------------------------------------
# Имена счётчиков разные в каждом узле (pfCount, dfAmount…) — это адреса полей
# в отчёте, поэтому шесть маленьких моделей, а не одна общая.


class PlaintiffFinished(ReportModel):
    pf_count: int | None = None
    pf_amount: int | None = None


class PlaintiffPending(ReportModel):
    pp_count: int | None = None
    pp_amount: int | None = None


class PlaintiffAppealed(ReportModel):
    pa_count: int | None = None
    pa_amount: int | None = None


class PlaintiffArbitration(ReportModel):
    plaintiff_arbitration_finished: PlaintiffFinished | None = None
    plaintiff_arbitration_pending: PlaintiffPending | None = None
    plaintiff_arbitration_appealed: PlaintiffAppealed | None = None


class DefandantFinished(ReportModel):
    df_count: int | None = None
    df_amount: int | None = None


class DefandantPending(ReportModel):
    dp_count: int | None = None
    dp_amount: int | None = None


class DefandantAppealed(ReportModel):
    da_count: int | None = None
    da_amount: int | None = None


class DefandantArbitration(ReportModel):
    defandant_arbitration_finished: DefandantFinished | None = None
    defandant_arbitration_pending: DefandantPending | None = None
    defandant_arbitration_appealed: DefandantAppealed | None = None


class ArbitrationByStatus(ReportModel):
    common_count: int | None = None
    common_amount: int | None = None
    plaintiff_arbitration: PlaintiffArbitration | None = None
    defandant_arbitration: DefandantArbitration | None = None  # опечатка — из источника


class ArbitrationCase(ReportModel):
    """Не дело, а разбивка по году."""

    year: int | None = None
    plaintiff_count: int | None = None
    plaintiff_amount: int | None = None
    defendant_count: int | None = None
    defendant_amount: int | None = None


# --- остальные списки -----------------------------------------------------------


class ExecutionProceeding(ReportModel):
    number: str | None = None
    date: MoscowDate | None = None
    amount: Decimal | None = None  # нет у 770 из 3873 — это «нет данных», не 0
    active: bool | None = None


class Inspection(ReportModel):
    authority_name: str | None = None
    form: str | None = None
    inspection_status: str | None = None
    start_date: MoscowDate | None = None  # плоская строка YYYY-MM-DD в источнике
    end_date: MoscowDate | None = None
    erp_id: str | None = None  # 20 цифр, в int64 не влезает
    type: str | None = None


class License(ReportModel):
    number: str | None = None
    name: str | None = None
    issuing_authority: str | None = None
    status: str | None = None
    issue_date: MoscowDate | None = None
    end_date: MoscowDate | None = None


class RelatedCompany(ReportModel):
    inn: str | None = None
    ogrn: str | None = None
    name: str | None = None
    auth_person_name: str | None = None
    auth_person_position: str | None = None
    registration_date: MoscowDate | None = None
    parent_organizations: list[ParentOrganization] | None = None


class Branch(ReportModel):
    name: str | None = None
    address: str | None = None


class BranchesInfo(ReportModel):
    branches_count: int | None = None
    branches: list[Branch] | None = None


class Procurement(ReportModel):
    procurements_year: int | None = None
    federal_law_code: str | None = None  # «ФЗ94» / «ФЗ223» — текст
    tender_admitted_cnt: int | None = None  # по спецификации; в снапшоте нет
    tender_winner_cnt: int | None = None
    contract_signed_cnt: int | None = None
    contract_signed_amt: int | None = None


class TaxSystem(ReportModel):
    short_name: str | None = None
    full_name: str | None = None


class Phone(ReportModel):
    phone_type: str | None = None  # по спецификации; в снапшоте нет
    phone_code: str | None = None
    phone_number: str | None = None


# --- report ---------------------------------------------------------------------


class Report(ReportModel):
    """Отчёт о контрагенте. Ключ — ``base_info.inn`` (строка)."""

    report_date: MoscowDate
    base_info: BaseInfo
    status: Status
    zsk_risk_level: str  # GREEN | YELLOW | RED — вторая метка, приводим как есть
    kinds_of_activity_info: KindsOfActivityInfo | None = None
    reputational_risks: ReputationalRisks | None = None
    arbitration_by_status: ArbitrationByStatus | None = None
    execution_proceedings: list[ExecutionProceeding] | None = None
    procurements: list[Procurement] | None = None
    phones: list[Phone] | None = None
    founders_info: FoundersInfo | None = None
    tax_system: list[TaxSystem] | None = None
    fin_reports: list[FinReport] | None = None
    related_companies: list[RelatedCompany] | None = None
    arbitration_cases: list[ArbitrationCase] | None = None
    inspections: list[Inspection] | None = None
    coefficient: Coefficient | None = None
    licenses: list[License] | None = None
    branches_info: BranchesInfo | None = None

    @property
    def inn(self) -> str:
        return self.base_info.inn

    def section_state(self, section: str) -> SectionState:
        """Состояние секции по её имени в отчёте (``finReports``, ``licenses``…)."""
        field = _FIELD_BY_ALIAS[section]
        return _state(getattr(self, field))

    def sections(self) -> dict[str, SectionState]:
        """Состояние всех секций отчёта, кроме ``reportDate``."""
        return {alias: self.section_state(alias) for alias in SECTIONS}


SECTIONS: tuple[str, ...] = tuple(
    field.alias or name for name, field in Report.model_fields.items() if name != "report_date"
)
_FIELD_BY_ALIAS: dict[str, str] = {
    (field.alias or name): name for name, field in Report.model_fields.items()
}


def _state(value: Any) -> SectionState:
    if value is None:
        return "absent"
    if isinstance(value, list):
        return "empty" if not value else "present"
    if isinstance(value, BaseModel):
        return "empty" if _is_hollow(value) else "present"
    return "present"


def _is_hollow(model: BaseModel) -> bool:
    """Узел без единого значения: ``arbitrationByStatus`` у 52 компаний — такой."""
    values = [getattr(model, name) for name in type(model).model_fields]
    values.extend((model.model_extra or {}).values())
    for value in values:
        if value is None or (isinstance(value, list) and not value):
            continue
        if isinstance(value, BaseModel) and _is_hollow(value):
            continue
        return False
    return True
