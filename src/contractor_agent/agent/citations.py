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

from contractor_agent.agent.schema import Citation
from contractor_agent.data.loader import ReportSource
from contractor_agent.data.paths import PathNotFoundError, resolve

_NUMBER = re.compile(r"(?<![\w.])(-?\d[\d\s ]*(?:[.,]\d+)?)\s*(тыс\.?|млн|млрд|%)?", re.IGNORECASE)
_SCALE = {
    "тыс": Decimal(1_000),
    "тыс.": Decimal(1_000),
    "млн": Decimal(1_000_000),
    "млрд": Decimal(1_000_000_000),
}
RELATIVE_TOLERANCE = Decimal("0.05")


@dataclass(frozen=True)
class CitationCheck:
    citation: Citation
    ok: bool
    why: str | None = None
    value: Any = None
    inn: str | None = None


_ABSENCE = re.compile(r"\bнет\b|\bне\s|отсутству|пуст", re.I)  # утверждение об отсутствии
_PATH = re.compile(r"report\.[A-Za-z_]\w*(?:\[\d+\])?(?:\.[A-Za-z_]\w*(?:\[\d+\])?)*")
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
    return normalize_path(path) in META_PATHS


def normalize_path(path: str) -> str:
    """Адрес из цитаты модели: снять кавычки и обрамляющие скобки, но не «]», закрывающую индекс."""
    path = path.strip().strip("«»\"'` ").lstrip("[")
    while path.endswith("]") and not re.search(r"\[\d+\]$", path):
        path = path[:-1]
    path = path.rstrip("«»\"'` ")
    if not path.startswith("report."):
        path = f"report.{path}"
    return path


def numbers_in(text: str) -> list[Decimal]:
    """Числа из текста с учётом «26,2 млн», «180 809,59», «1,1 %» (проценты → доля)."""
    found: list[Decimal] = []
    for raw, unit in _NUMBER.findall(text):
        cleaned = re.sub(r"\s", "", raw).replace(",", ".")  # любые пробелы, включая U+202F
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
    """Совпадение с допуском на округление; знак не учитываем («убыток 26,2 млн» ↔ −26 249 000)."""
    if actual == 0:
        return claimed == 0
    return abs(abs(claimed) - abs(actual)) / abs(actual) <= RELATIVE_TOLERANCE


def check_citation(source: ReportSource, inns: list[str], citation: Citation) -> CitationCheck:
    path = normalize_path(citation.source_path)
    last_error = "адреса нет в отчёте"
    scope = [citation.inn] if citation.inn and citation.inn in inns else list(reversed(inns))
    for inn in scope:  # своя компания, иначе последняя выбранная — первой
        report = source.get(inn)
        if report is None:
            continue
        try:
            value = resolve(report, path)
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


def _check_value(citation: Citation, path: str, value: Any, inn: str) -> CitationCheck:
    claimed = numbers_in(citation.claim)
    if _ABSENCE_STRICT.search(citation.claim):
        # «сведений о судах нет» со ссылкой на сводку, где счётчики > 0, — выдуманное отсутствие
        filled = (
            isinstance(value, int | float | Decimal) and not isinstance(value, bool) and value > 0
        ) or (not isinstance(value, list | str) and _positive_counts(value))
        if filled:
            return CitationCheck(
                citation, False, "утверждение об отсутствии, а поле заполнено", value, inn
            )
    if isinstance(value, bool) or not isinstance(value, int | float | Decimal):
        if value is None and claimed and not _ABSENCE.search(citation.claim):
            # «нет сведений о прибыли за 2024–2025» с пустым полем — верная цитата: годы не значения
            return CitationCheck(
                citation, False, "поле пустое, а утверждение содержит число", value, inn
            )
        return CitationCheck(citation, True, None, value, inn)  # не число: достаточно адреса
    if not claimed:
        return CitationCheck(citation, True, None, value, inn)
    actual = Decimal(str(value))
    if any(number_matches(c, actual) for c in claimed):
        return CitationCheck(citation, True, None, value, inn)
    return CitationCheck(citation, False, f"число не совпадает: в отчёте {value}", value, inn)


def validate_citations(
    source: ReportSource, inns: list[str], citations: list[Citation]
) -> list[CitationCheck]:
    return [check_citation(source, inns, c) for c in citations]


def describe_value(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
