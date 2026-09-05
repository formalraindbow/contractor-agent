"""Разворот MongoDB Extended JSON в плоское дерево.

JSON-снапшот — выгрузка из MongoDB: целые, не помещающиеся в int32, обёрнуты
в ``{"$numberLong": "4534783044"}``, даты — в ``{"$date": "2024-01-14T21:00:00.000Z"}``.
Обёртка появляется у любого целого ≥ 2³¹ (в снапшоте — 16 путей, 67 значений),
поэтому разворачиваем по всему дереву, а не по списку полей.

Результат — «плоское дерево»: те же словари и списки, но числа — ``int``,
а даты — строки ISO. Ровно такие же строки лежат в ячейках CSV, так что дальше
оба формата проходят через одну и ту же модель ``Report``; разбор дат — там.
"""

from __future__ import annotations

from typing import Any

NUMBER_LONG = "$numberLong"
DATE = "$date"


class UnknownWrapperError(ValueError):
    """Встретилась обёртка Extended JSON, которую мы не умеем разворачивать."""


def unwrap(value: Any) -> Any:
    """Рекурсивно снять обёртки ``$numberLong`` и ``$date``; остальное вернуть как есть.

    Возвращает новое дерево, входное не меняет.
    """
    if isinstance(value, dict):
        if len(value) == 1:
            ((key, inner),) = value.items()
            if key == NUMBER_LONG:
                return int(inner)
            if key == DATE:
                if not isinstance(inner, str):
                    raise UnknownWrapperError(f"{DATE} с нестроковым значением: {inner!r}")
                return inner
            if key.startswith("$"):
                raise UnknownWrapperError(f"неизвестная обёртка Extended JSON: {key}")
        return {k: unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwrap(v) for v in value]
    return value
