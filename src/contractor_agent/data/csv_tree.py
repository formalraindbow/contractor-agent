"""Разворот CSV-снапшота обратно в дерево.

CSV — та же схема, что в JSON, но сплющенная: путь поля склеен в имя колонки
(``report.foundersInfo.cofounders[0].name``). Заголовок — разреженное объединение
путей в порядке первого появления, сгруппированное по полю, а не по элементу,
поэтому разбираем только имя колонки, никогда позицию.

Все ячейки — строки; обёрток Mongo в CSV нет. Типизацию здесь не делаем:
по виду значения ОКВЭД ``31.0`` стал бы числом, а ИНН ``0277985654`` потерял бы
ноль. Типизирует модель ``Report`` — по пути поля. Пустая ячейка = поля нет.

Политика присутствия: CSV не отличает «секции нет» от «секция есть, но пуста».
Секции, которые источник отдаёт всегда (в JSON — у всех 100 записей), при пустых
ячейках получают ``[]`` / ``{}``; остальные остаются отсутствующими. Так
CSV-компания выглядит как JSON-компания, и одна модель обслуживает обе.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PathToken = str | int
"""Сегмент пути: имя поля или индекс списка."""

_SEGMENT = re.compile(r"^([^.\[\]]+)((?:\[\d+\])*)$")
_INDEX = re.compile(r"\[(\d+)\]")

ALWAYS_PRESENT_LISTS: frozenset[str] = frozenset({"executionProceedings", "phones", "procurements"})
ALWAYS_PRESENT_DICTS: frozenset[str] = frozenset({"arbitrationByStatus", "reputationalRisks"})
"""Секции, которые источник отдаёт всегда (у всех 100 JSON-записей), даже пустыми."""


class ColumnNameError(ValueError):
    """Имя колонки не разбирается в путь."""


class IndexGapError(ValueError):
    """В списке есть дыра: заполнены индексы 0 и 2, а 1 — нет."""


def parse_column(name: str) -> tuple[PathToken, ...]:
    """``"report.cofounders[2].inn"`` → ``("report", "cofounders", 2, "inn")``."""
    tokens: list[PathToken] = []
    for segment in name.split("."):
        m = _SEGMENT.match(segment)
        if not m:
            raise ColumnNameError(f"не разбирается имя колонки: {name!r}")
        tokens.append(m.group(1))
        tokens.extend(int(i) for i in _INDEX.findall(m.group(2)))
    return tuple(tokens)


def _insert(root: dict[str, Any], path: tuple[PathToken, ...], value: str) -> None:
    """Положить значение по пути, создавая словари и списки по типу следующего токена."""
    node: Any = root
    for i, token in enumerate(path[:-1]):
        nxt = path[i + 1]
        child: Any = [] if isinstance(nxt, int) else {}
        if isinstance(token, int):
            while len(node) <= token:
                node.append(None)
            if node[token] is None:
                node[token] = child
            node = node[token]
        else:
            node = node.setdefault(token, child)
    last = path[-1]
    if isinstance(last, int):
        while len(node) <= last:
            node.append(None)
        node[last] = value
    else:
        node[last] = value


def _check_no_gaps(node: Any, where: str) -> None:
    if isinstance(node, list):
        for i, item in enumerate(node):
            if item is None:
                raise IndexGapError(f"{where}[{i}] не заполнен, а следующие индексы есть")
            _check_no_gaps(item, f"{where}[{i}]")
    elif isinstance(node, dict):
        for k, v in node.items():
            _check_no_gaps(v, f"{where}.{k}" if where else k)


def row_to_tree(row: dict[str, str]) -> dict[str, Any]:
    """Одна строка CSV → дерево ``{"_id": …, "report": …}`` той же формы, что запись JSON."""
    tree: dict[str, Any] = {}
    for column, cell in row.items():
        if cell == "":
            continue
        _insert(tree, parse_column(column), cell)
    _check_no_gaps(tree, "")
    return tree


def apply_presence_policy(report: dict[str, Any]) -> dict[str, Any]:
    """Добавить секции, которые источник отдаёт всегда, если в CSV для них не было ячеек."""
    out = dict(report)
    for key in ALWAYS_PRESENT_LISTS:
        out.setdefault(key, [])
    for key in ALWAYS_PRESENT_DICTS:
        out.setdefault(key, {})
    risks = dict(out["reputationalRisks"])
    risks.setdefault("negative", [])
    risks.setdefault("positive", [])
    out["reputationalRisks"] = risks
    return out


def iter_rows(path: Path) -> Iterator[dict[str, str]]:
    """Строки CSV как словари «имя колонки → ячейка»; все значения — строки."""
    csv.field_size_limit(10**9)
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    """Все записи CSV в виде плоских деревьев, после политики присутствия."""
    records = []
    for row in iter_rows(path):
        tree = row_to_tree(row)
        tree["report"] = apply_presence_policy(tree["report"])
        records.append(tree)
    return records
