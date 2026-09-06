"""Загрузка снапшота и протокол источника отчётов.

``ReportSource`` — граница между данными и остальной системой: ``signals`` и
``mcp_server`` зависят только от двух методов, ``get(inn)`` и ``search(query)``.
Сейчас источник — снапшот из двух файлов (``Snapshot`` в памяти и ``SqliteSource``
в ``index.py``), в банке — HTTP-клиент к их эндпоинту «ИНН → отчёт». Формат ответа
тот же, поэтому вторая реализация — это ещё один класс, а не правки в агенте.

Снапшот: JSON и CSV — разные компании, по 100 в каждом, пересечение по ИНН — ноль.
Ключ записи — ИНН строкой (у ИП 12 цифр, бывает ведущий ноль).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from contractor_agent.data.csv_tree import load_csv_records
from contractor_agent.data.model import Report
from contractor_agent.data.mongo import unwrap

JSON_FILE = "contractors_audit.snapshot.json"
CSV_FILE = "contractors_audit.snapshot_C12613591.csv"


class Source(StrEnum):
    JSON = "json"
    CSV = "csv"


@dataclass(frozen=True, slots=True)
class CompanyHit:
    """Строка результата поиска — то, что нужно, чтобы выбрать компанию."""

    inn: str
    ogrn: str
    short_name: str
    address: str | None
    risk_level: str
    zsk_risk_level: str
    report_date: date
    source: Source


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    source: Source
    report: Report

    @property
    def inn(self) -> str:
        return self.report.inn

    def hit(self) -> CompanyHit:
        info = self.report.base_info
        return CompanyHit(
            inn=info.inn,
            ogrn=info.ogrn,
            short_name=info.short_name,
            address=info.address,
            risk_level=info.risk_level,
            zsk_risk_level=self.report.zsk_risk_level,
            report_date=self.report.report_date,
            source=self.source,
        )


@runtime_checkable
class ReportSource(Protocol):
    """Единственное, что остальная система знает об источнике отчётов."""

    def get(self, inn: str) -> Report | None: ...

    def search(self, query: str, limit: int = 5) -> list[CompanyHit]: ...


class DuplicateInnError(ValueError):
    """Один ИНН встретился дважды — так не должно быть ни в файле, ни между файлами."""


# --- поиск по названию ----------------------------------------------------------
# SQLite не умеет casefold для кириллицы (LOWER/LIKE — только ASCII), поэтому
# нормализуем в Python и храним нормализованное имя отдельной колонкой.

_QUOTES = re.compile(r"[\"'«»„“”`]")
_SPACES = re.compile(r"\s+")
_LEGAL_FORMS = frozenset({"ооо", "ао", "пао", "зао", "оао", "ип", "ано", "нко", "дпо", "тд"})


def normalize_name(name: str) -> str:
    """``ООО «ЛЕ МОНЛИД»`` → ``ооо ле монлид``: регистр, ё, кавычки, пробелы."""
    return _SPACES.sub(" ", _QUOTES.sub("", name.casefold().replace("ё", "е"))).strip()


def strip_legal_form(name_norm: str) -> str:
    """``ооо ле монлид`` → ``ле монлид``: пользователь ищет без организационной формы."""
    words = name_norm.split(" ")
    while len(words) > 1 and words[0] in _LEGAL_FORMS:
        words = words[1:]
    return " ".join(words)


NO_MATCH = 4


def query_stems(query_norm: str) -> list[str]:
    """Основы слов запроса без орг. формы: «ип янполова» → ["янпол"] — чтобы падеж не мешал.
    Короткие слова («гдк», «псг») участвуют целиком: выкинуть их — значит искать «ГДК материалы»
    по одному слову «материал» и найти чужую компанию."""
    words = [w for w in strip_legal_form(query_norm).split(" ") if w]
    return [w[:-2] if len(w) >= 6 else (w[:-1] if len(w) >= 4 else w) for w in words]


def rank(name_norm: str, query_norm: str) -> int:
    """0 — точное совпадение, 1 — начало названия, 2 — подстрока,
    3 — все основы слов запроса начинают слова названия (склонение), 4 — нет."""
    best = NO_MATCH
    for candidate in (name_norm, strip_legal_form(name_norm)):
        for query in (query_norm, strip_legal_form(query_norm)):
            if candidate == query:
                return 0
            if candidate.startswith(query):
                best = min(best, 1)
            elif query in candidate:
                best = min(best, 2)
    if best == NO_MATCH:
        stems, words = query_stems(query_norm), name_norm.split(" ")
        if stems and all(any(w.startswith(stem) for w in words) for stem in stems):
            best = 3
        elif stems and strip_legal_form(name_norm) in stems:
            # «ГДК материалы»: модель склеила название с товаром — точное короткое название
            # среди слов запроса ценнее, чем пустая выдача (ранг ниже частичных совпадений)
            best = 3
    return best


# --- Snapshot -------------------------------------------------------------------


class Snapshot:
    """Все записи снапшота в памяти; реализация ``ReportSource`` для тестов и сборки индекса."""

    def __init__(self, records: Iterable[CompanyRecord]) -> None:
        self._records: dict[str, CompanyRecord] = {}
        for record in records:
            if record.inn in self._records:
                raise DuplicateInnError(record.inn)
            self._records[record.inn] = record

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CompanyRecord]:
        return iter(self._records.values())

    def record(self, inn: str) -> CompanyRecord | None:
        return self._records.get(inn)

    def get(self, inn: str) -> Report | None:
        record = self._records.get(inn)
        return record.report if record else None

    def search(self, query: str, limit: int = 5) -> list[CompanyHit]:
        query = query.strip()
        if not query:
            return []
        if query.isdigit():
            hits = [r.hit() for r in self if r.inn == query or r.report.base_info.ogrn == query]
            return hits[:limit]
        query_norm = normalize_name(query)
        scored: list[tuple[int, str, CompanyHit]] = []
        for record in self:
            info = record.report.base_info
            score = min(
                rank(normalize_name(info.short_name), query_norm),
                rank(normalize_name(info.full_name or ""), query_norm),
            )
            if score < NO_MATCH:
                scored.append((score, info.short_name, record.hit()))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [hit for _, _, hit in scored[:limit]]


# --- загрузка файлов -------------------------------------------------------------


def load_json(path: Path) -> list[CompanyRecord]:
    """JSON-файл снапшота: массив ``{_id, report}`` в Extended JSON."""
    records = unwrap(json.loads(path.read_text(encoding="utf-8")))
    return [CompanyRecord(Source.JSON, Report.model_validate(r["report"])) for r in records]


def load_csv(path: Path) -> list[CompanyRecord]:
    """CSV-файл снапшота: плоская таблица → дерево → ``Report``."""
    records = load_csv_records(path)
    return [CompanyRecord(Source.CSV, Report.model_validate(r["report"])) for r in records]


def load_snapshot(data_dir: Path) -> Snapshot:
    """Оба файла как один набор. Дубликат ИНН между файлами — ошибка."""
    return Snapshot([*load_json(data_dir / JSON_FILE), *load_csv(data_dir / CSV_FILE)])
