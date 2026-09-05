"""Конверт ответа инструмента: ``{available, data, source_paths, report_date}``.

Один формат на все инструменты, чтобы модель и валидатор цитат не гадали:
``available: false`` с причиной — секции нет или компания не найдена, никогда
не пустой список; длинные списки обрезаются с указанием полного размера.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contractor_agent.signals.model import jsonable

ITEM_LIMIT = 20


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    data: Any = None
    source_paths: list[str] = Field(default_factory=list)
    report_date: str | None = Field(
        default=None, description="дата отчёта ISO — у каждой компании своя"
    )
    reason: str | None = Field(
        default=None,
        description="почему недоступно: not_found | section_absent | unknown_section | bad_request",
    )
    note: str | None = Field(
        default=None, description="оговорка для модели, например «не найдено ≠ нет»"
    )

    @field_validator("data", mode="before")
    @classmethod
    def _json_native(cls, value: Any) -> Any:
        return jsonable(value)


def ok(
    data: Any,
    *,
    source_paths: list[str] | None = None,
    report_date: date | None = None,
    note: str | None = None,
) -> ToolResponse:
    return ToolResponse(
        available=True,
        data=data,
        source_paths=list(source_paths or []),
        report_date=report_date.isoformat() if report_date else None,
        note=note,
    )


def unavailable(
    reason: str, *, note: str | None = None, report_date: date | None = None
) -> ToolResponse:
    return ToolResponse(
        available=False,
        data=None,
        reason=reason,
        note=note,
        report_date=report_date.isoformat() if report_date else None,
    )


def truncate(items: list[Any], limit: int = ITEM_LIMIT) -> dict[str, Any]:
    """Список для модели: первые ``limit`` элементов и полный размер."""
    return {"items": items[:limit], "total": len(items), "truncated": len(items) > limit}


def truncate_deep(value: Any, limit: int = ITEM_LIMIT) -> Any:
    """Обрезать все списки длиннее ``limit`` внутри произвольного дерева."""
    if isinstance(value, dict):
        return {k: truncate_deep(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        items = [truncate_deep(v, limit) for v in value[:limit]]
        if len(value) > limit:
            return {"items": items, "total": len(value), "truncated": True}
        return items
    return value
