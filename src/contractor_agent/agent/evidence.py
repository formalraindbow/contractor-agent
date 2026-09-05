"""Читаемые основания: поле отчёта или воспроизводимый расчёт инструмента."""

from __future__ import annotations

from typing import Any

from contractor_agent.data.loader import ReportSource
from contractor_agent.data.paths import PathNotFoundError, parse_path, resolve


def resolve_evidence(source: ReportSource, inn: str, path: str) -> Any:
    report = source.get(inn)
    if report is None:
        raise PathNotFoundError("Компания не найдена")
    if not path.startswith("computed."):
        return resolve(report, path)
    from contractor_agent.mcp_server.tools import Tools

    family, _, tail = path.removeprefix("computed.").partition(".")
    if family == "sections":
        from contractor_agent.data.model import SECTIONS

        section, _, metric = tail.partition(".")
        if section not in SECTIONS or metric != "count":
            raise PathNotFoundError("Неизвестный счётчик раздела")
        value = resolve(report, "report." + section)
        return len(value) if isinstance(value, list) else None
    methods = {
        "financials": "get_financials",
        "enforcement": "get_enforcement_summary",
        "arbitration": "get_arbitration_summary",
        "risks": "get_risk_signals",
    }
    if family not in methods or not tail:
        raise PathNotFoundError("Неизвестный расчёт")
    response = getattr(Tools(source), methods[family])(inn)
    if not response.available:
        raise PathNotFoundError("Нет данных для расчёта")
    node = response.data
    for token in parse_path("report." + tail):
        try:
            node = node[token]
        except (TypeError, KeyError, IndexError) as e:
            raise PathNotFoundError(f"Нет поля расчёта: {path}") from e
    return node


def evidence_basis(source: ReportSource, inn: str, path: str) -> list[dict[str, Any]]:
    """Исходные поля для расчёта, а не только повтор текста результата."""
    import re

    from contractor_agent.data.paths import nearest_path
    from contractor_agent.signals.model import jsonable

    report = source.get(inn)
    paths = []
    if path.startswith("computed.sections."):
        paths = ["report." + path.split(".")[2]]
    elif path.startswith("computed.risks.signals."):
        item = resolve_evidence(source, inn, path.rsplit(".", 1)[0])
        paths = [item["source_path"], *item.get("source_paths", [])]
    elif path.startswith("computed.enforcement."):
        paths = ["report.executionProceedings"]
    elif path.startswith("computed.arbitration."):
        base = path.rsplit(".", 1)[0]
        try:
            item = resolve_evidence(source, inn, base)
            paths = [item[k] for k in ("count_path", "amount_path") if item.get(k)]
        except PathNotFoundError:
            paths = ["report.arbitrationByStatus"]
    elif path.startswith("computed.financials.years["):
        match = re.match(r"computed.financials.years\[(\d+)\]", path)
        item = resolve_evidence(source, inn, match.group())
        paths = [
            item["paths"][key]
            for key in (
                "current_assets",
                "short_term_liabilities",
                "profit",
                "proceeds",
                "capitals",
                "long_term_duties",
                "total_assets",
            )
        ]
    result = []
    for p in dict.fromkeys(paths):
        value = resolve(report, nearest_path(report, p))
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True)
        elif isinstance(value, list):
            value = [
                v.model_dump(mode="json", by_alias=True) if hasattr(v, "model_dump") else v
                for v in value
            ]
        result.append({"path": p, "value": jsonable(value)})
    return result
