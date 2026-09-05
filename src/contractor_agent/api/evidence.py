"""Report fields and reproducible aggregates for the evidence drawer."""

from pydantic import BaseModel

from contractor_agent.data.paths import PathNotFoundError, resolve
from contractor_agent.mcp_server.envelope import truncate_deep
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import jsonable


def native(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, list):
        return [native(v) for v in value]
    return jsonable(value)


def evidence(tools: Tools, inn: str, path: str, signal: str | None = None) -> dict:
    report = tools.source.get(inn)
    if report is None:
        raise PathNotFoundError("Компания не найдена")
    value = resolve(report, path)
    basis, calculation = [], None
    if signal:
        groups = tools.get_risk_signals(inn).data["signals"]
        item = next((s for group in groups.values() for s in group if s["code"] == signal), None)
        if item is None or item["source_path"] != path:
            raise PathNotFoundError("Основание не найдено")
        calculation = {"description": item["explanation"], "values": item["value"]}
        basis = [
            {"path": p, "value": native(resolve(report, p))}
            for p in dict.fromkeys([path, *item["source_paths"]])
        ]
    elif path.startswith("report.executionProceedings"):
        response = tools.get_enforcement_summary(inn)
        if response.available:
            calculation = {
                "description": "Действующие и завершённые производства посчитаны отдельно. "
                "Суммируются только указанные положительные суммы; "
                "неизвестные не заменяются нулями.",
                "values": {
                    k: response.data[k] for k in ("active", "finished", "unknown_status_count")
                },
            }
    elif path.startswith("report.finReports"):
        response = tools.get_financials(inn)
        if response.available:
            years = [
                r
                for r in response.data["years"]
                if path == "report.finReports" or path.startswith(r["path"])
            ]
            calculation = {
                "description": "Текущая ликвидность = оборотные активы / "
                "краткосрочные обязательства. "
                "Капитал и резервы используются как оценка чистых активов. Пропуски — не нули.",
                "values": years,
            }
    return {
        "inn": inn,
        "path": path,
        "report_date": report.report_date.isoformat(),
        "value": truncate_deep(native(value)),
        "basis": truncate_deep(basis),
        "calculation": truncate_deep(calculation),
    }
