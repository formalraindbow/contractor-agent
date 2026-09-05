"""Валидатор цитат — защита от галлюцинаций без второй модели.

Каждое утверждение ответа несёт ``source_path``. Проверяем программно: адрес
существует в отчёте (``paths.resolve``), а если утверждение содержит число и поле
числовое — число совпадает с полем (с поправкой на «тыс.», «млн», «млрд», проценты
и округление). Невалидная цитата — не факт: один круг на исправление, потом
утверждение помечается и не показывается как факт.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from contractor_agent.agent.evidence import resolve_evidence
from contractor_agent.agent.schema import Citation
from contractor_agent.data.loader import ReportSource
from contractor_agent.data.paths import PathNotFoundError

_NUMBER = re.compile(
    r"(?<![\w.])([−-]?\d[\d \u00a0\u202f]*(?:[.,]\d+)?)\s*(тыс\.?|млн|млрд|%)?", re.IGNORECASE
)
_SCALE = {
    "тыс": Decimal(1_000),
    "тыс.": Decimal(1_000),
    "млн": Decimal(1_000_000),
    "млрд": Decimal(1_000_000_000),
}
RELATIVE_TOLERANCE = Decimal("0.002")


@dataclass(frozen=True)
class CitationCheck:
    citation: Citation
    ok: bool
    why: str | None = None
    value: Any = None
    inn: str | None = None


_ABSENCE = re.compile(r"\bнет\b|\bне\s|отсутству|пуст", re.I)  # утверждение об отсутствии
_PATH = re.compile(r"(?:report|computed)\.[A-Za-z_]\w*(?:\[\d+\])?(?:\.[A-Za-z_]\w*(?:\[\d+\])?)*")
_CLAIM_TAIL = re.compile(
    r"[\s\[(«\"'`:;,\-—•*]*(?:source_path|адрес|путь)?[\s\[(«\"'`:;,\-—•*]*$", re.IGNORECASE
)
META_PATHS = frozenset(
    {"report.verdict", "report.verdict_ru", "report.report_date", "report.reportDate"}
)
"""Псевдоадреса, которые модель ставит к выводу и дате: не поля отчёта, но и не выдумка."""


def extract_inline_citations(text: str) -> list[Citation]:
    """Цитаты из адресов в тексте в любом оформлении: в квадратных или круглых скобках,
    с префиксом «source_path:» или без — утверждение берётся из текста перед адресом."""
    out: list[Citation] = []
    for line in text.splitlines():
        cursor = 0
        claim = ""
        for m in _PATH.finditer(line):
            before = line[cursor : m.start()]
            candidate = _CLAIM_TAIL.sub("", before).strip(" -•*:;,.][()«»\"'`")
            if candidate:
                claim = candidate  # несколько адресов подряд относятся к одному утверждению
            cursor = m.end()
            if claim:
                out.append(Citation(claim=claim, source_path=m.group(0)))
    return out


def is_meta_path(path: str) -> bool:
    norm = normalize_path(path)
    return norm in META_PATHS or "verdict_ru" in norm or norm.startswith("report.riskSignals")


def normalize_path(path: str) -> str:
    """Адрес из цитаты модели: снять кавычки и обрамляющие скобки, но не «]», закрывающую индекс."""
    path = path.strip().strip("«»\"'` ").lstrip("[")
    while path.endswith("]") and not re.search(r"\[\d+\]$", path):
        path = path[:-1]
    path = path.rstrip("«»\"'` ")
    if not path.startswith(("report.", "computed.")):
        path = f"report.{path}"
    return path


def numbers_in(text: str) -> list[Decimal]:
    """Числа из текста с учётом «26,2 млн», «180 809,59», «1,1 %» (проценты → доля)."""
    found: list[Decimal] = []
    for raw, unit in _NUMBER.findall(text):
        cleaned = (
            re.sub(r"\s", "", raw).replace("−", "-").replace(",", ".")
        )  # любые пробелы, включая U+202F
        try:
            value = Decimal(cleaned)
        except Exception:
            continue
        unit = (unit or "").lower()
        if unit in _SCALE:
            value *= _SCALE[unit]
        elif unit == "%":
            found.append(value / 100)
        found.append(value)
    return found


def number_matches(claimed: Decimal, actual: Decimal) -> bool:
    """Совпадение знака и числа; точность отображения проверяется отдельно в _check_value."""
    if actual == 0:
        return claimed == 0
    return abs(claimed - actual) / abs(actual) <= RELATIVE_TOLERANCE


def check_citation(source: ReportSource, inns: list[str], citation: Citation) -> CitationCheck:
    if re.search(
        r"[,;]\s*report\.", citation.source_path
    ):  # два адреса в одной цитате — проверяем оба
        parts = [x.strip() for x in re.split(r"[,;]", citation.source_path) if x.strip()]
        checks = [
            check_citation(source, inns, citation.model_copy(update={"source_path": x}))
            for x in parts
        ]
        bad = next((c for c in checks if not c.ok), None)
        return bad or CitationCheck(citation, True, None, checks[0].value, checks[0].inn)
    path = normalize_path(citation.source_path)
    last_error = "адреса нет в отчёте"
    if citation.inn and citation.inn not in inns:
        return CitationCheck(citation, False, "ИНН цитаты не относится к текущему ответу")
    if not citation.inn and len(inns) != 1:
        return CitationCheck(citation, False, "При нескольких компаниях укажи ИНН цитаты")
    scope = [citation.inn] if citation.inn else inns
    for inn in scope:  # своя компания, иначе последняя выбранная — первой
        report = source.get(inn)
        if report is None:
            continue
        try:
            value = resolve_evidence(source, inn, path)
        except PathNotFoundError as e:
            last_error = str(e)
            continue
        return _check_value(citation, path, value, inn)
    return CitationCheck(citation, False, last_error)


_ABSENCE_STRICT = re.compile(
    r"нет сведений|сведений нет|не найдено|нет данных|данных нет|отсутству|нет ни одного", re.I
)


def _positive_counts(value: Any) -> bool:
    """В сводке (объект с счётчиками) есть число больше нуля — раздел не пустой."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return any(_positive_counts(v) for v in value.values())
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float | Decimal):
        return value > 0
    return False


def _is_year(n: Decimal) -> bool:
    return n == n.to_integral_value() and 1990 <= n <= 2035


def claim_measurements(text: str) -> list[tuple[Decimal, Decimal, str]]:
    """Значение, допуск по последнему отображённому разряду, единица.

    Годы убираем только в явном контексте периода. Счётчики всегда точные.
    """
    text = re.sub(r"\bИНН\s*[:№]?\s*\d{10,12}\b", "", text, flags=re.I)
    text = re.sub(r"\b\d{2}\.\d{2}\.\d{4}\b|\b\d{4}-\d{2}-\d{2}\b", "", text)
    text = re.sub(r"(?<!\d)(?:19|20)\d{2}\s*[–—-]\s*(?:19|20)\d{2}", "", text)
    text = re.sub(r"\b(?:за|на конец|в|от)\s+(?:19|20)\d{2}\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:19|20)\d{2}(?=\s*г(?:од|\.))", "", text, flags=re.I)
    found = []
    for raw, unit in _NUMBER.findall(text):
        raw = re.sub(r"\s", "", raw).replace("−", "-").replace(",", ".")
        unit = unit.lower()
        number = Decimal(raw)
        factor = _SCALE.get(unit, Decimal(1))
        precision = len(raw.split(".")[1]) if "." in raw else 0
        tolerance = (
            factor * Decimal(10) ** -precision / 2 if unit in _SCALE or precision else Decimal(0)
        )
        found.append((number * factor, tolerance, unit))
    return found


def _check_value(citation: Citation, path: str, value: Any, inn: str) -> CitationCheck:
    def result(ok: bool, why: str | None = None) -> CitationCheck:
        return CitationCheck(citation, ok, why, value, inn)

    claim = citation.claim
    measures = claim_measurements(claim)
    if value is None:
        return result(
            bool(_ABSENCE.search(claim)), "поле пустое; можно сообщить только отсутствие данных"
        )
    # A generated explanation is accepted only verbatim: its calculations ran in Python.
    if path.startswith("computed.risks.") and isinstance(value, str):
        return result(
            claim.strip().casefold().rstrip(".") == value.strip().casefold().rstrip("."),
            "Фраза не совпадает с рассчитанным основанием",
        )
    if isinstance(value, bool):
        return result(not measures, "Логическое поле не подтверждает число")
    if isinstance(value, int | float | Decimal):
        if _ABSENCE_STRICT.search(claim) and value != 0:
            return result(False, "утверждение об отсутствии, а поле заполнено")
        if not measures:
            return result(True)
        actual = Decimal(str(value))
        count = bool(re.search(r"count|staff|yearsFromRegistration", path, re.I))
        for number, tolerance, unit in measures:
            if re.search(r"убыт|отрицательн", claim, re.I) and number > 0:
                number = -number
            if unit == "%" and not path.endswith(("profitability", "profitability_pct")):
                number, tolerance = number / 100, tolerance / 100
            if count and unit:
                return result(False, "Количество нельзя округлять или выражать денежной единицей")
            if abs(number - actual) > (Decimal(0) if count else tolerance):
                return result(False, f"число не совпадает: в отчёте {value}")
        return result(True)
    if isinstance(value, date):
        return result(
            value.isoformat() in claim or value.strftime("%d.%m.%Y") in claim, "Дата не совпадает"
        )
    if isinstance(value, str):
        if measures:
            # INN/OKVED and other literal numeric strings are checked as strings.
            if value not in claim:
                return result(False, "Текстовое поле не подтверждает указанное число")
            stripped = claim.replace(value, "")
            if claim_measurements(stripped):
                return result(False, "В утверждении есть число, которого нет в поле")
        if (
            path.endswith("reasonName")
            and re.search(r"банкрот", value, re.I)
            and re.search(r"не\s+(?:является\s+)?банкрот|банкротств[ао]\s+нет", claim, re.I)
        ):
            return result(False, "Утверждение отрицает указанный в отчёте статус")
        return result(True)
    if measures:
        return result(
            False, "Объект или список не подтверждает число; укажи точное поле или computed.*"
        )
    if (
        _ABSENCE_STRICT.search(claim)
        and path.startswith("report.arbitration")
        and _positive_counts(value)
    ):
        return result(False, "утверждение об отсутствии, а поле заполнено")
    return result(True)


def validate_citations(
    source: ReportSource, inns: list[str], citations: list[Citation]
) -> list[CitationCheck]:
    return [check_citation(source, inns, c) for c in citations]


def describe_value(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
