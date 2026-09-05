"""Адрес поля в отчёте — ``source_path``.

Каждое утверждение агента ссылается на поле отчёта строкой вида
``report.executionProceedings[12].amount``. Синтаксис тот же, что у имён колонок
CSV, поэтому разбор общий (``csv_tree.parse_column``). Этим модулем пользуются
``signals`` (откуда взят сигнал), инструменты MCP (``source_paths`` в ответе)
и валидатор цитат (существует ли путь, совпадает ли значение).

Путь резолвится по модели, не по сырому JSON: значения уже нормализованы
(``Decimal``, московская ``date``), и именно их увидит модель в ответах инструментов.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from typing import Any

from pydantic import BaseModel

from contractor_agent.data.csv_tree import ColumnNameError, PathToken, parse_column
from contractor_agent.data.model import Report

ROOT = "report"


class PathNotFoundError(LookupError):
    """Пути нет в отчёте: нет такого поля, индекс за границей списка, или это не путь."""


def parse_path(path: str) -> tuple[PathToken, ...]:
    """``report.a[1].b`` → ``("a", 1, "b")``; префикс ``report.`` обязателен."""
    try:
        tokens = parse_column(path)
    except ColumnNameError as e:
        raise PathNotFoundError(str(e)) from e
    if not tokens or tokens[0] != ROOT:
        raise PathNotFoundError(f"путь должен начинаться с {ROOT!r}: {path!r}")
    return tokens[1:]


@cache
def _field_by_alias(cls: type[BaseModel]) -> dict[str, str]:
    return {(f.alias or name): name for name, f in cls.model_fields.items()}


def resolve(report: Report, path: str) -> Any:
    """Значение по адресу или ``PathNotFoundError``.

    Отсутствующее поле (``None``) — валидный результат: путь есть, данных нет.
    """
    node: Any = report
    for token in parse_path(path):
        if isinstance(token, int):
            if not isinstance(node, list) or not 0 <= token < len(node):
                raise PathNotFoundError(f"{path}: индекс [{token}] вне списка")
            node = node[token]
        elif isinstance(node, BaseModel):
            field = _field_by_alias(type(node)).get(token)
            if field is not None:
                node = getattr(node, field)
            elif node.model_extra and token in node.model_extra:
                node = node.model_extra[token]
            else:
                raise PathNotFoundError(f"{path}: нет поля {token!r}")
        elif isinstance(node, dict) and token in node:
            node = node[token]
        else:
            raise PathNotFoundError(f"{path}: {token!r} — не поле")
    return node


def iter_leaf_paths(report: Report) -> Iterator[tuple[str, Any]]:
    """Все листовые адреса отчёта с их значениями, в терминах отчёта (camelCase)."""
    yield from _walk(report.model_dump(), ROOT)


def _walk(node: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{prefix}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{prefix}[{i}]")
    else:
        yield prefix, node


def nearest_path(report: Report, path: str) -> str:
    """Адрес существующего узла для отсутствующего поля; не создаёт значения."""
    import re

    while path != ROOT:
        try:
            resolve(report, path)
            return path
        except PathNotFoundError:
            path = re.sub(r"(?:\.[^.\[]+|\[\d+\])$", "", path)
    return ROOT
