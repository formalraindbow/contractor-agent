"""Инструменты MCP как чистые функции над ``ReportSource`` и ``signals``.

Без зависимости от MCP: тестируются напрямую, а ``server.py`` только регистрирует
их. Инструменты агрегируют, а не зеркалят единственный поинт «ИНН → отчёт»
(инвариант 10): модель получает готовые своды и ровно нужные срезы, а карточка
ЛЕ МОНЛИД с 1744 производствами в неё не попадает целиком.

Каждый ответ — ``ToolResponse``: ``available``, данные, адреса полей, дата отчёта.
Отсутствие секции — ``available: false`` с причиной, никогда не пустой список.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from contractor_agent.data.loader import ReportSource
from contractor_agent.data.model import SECTIONS
from contractor_agent.data.paths import resolve
from contractor_agent.labels import svetofor_ru, zsk_ru
from contractor_agent.mcp_server.envelope import (
    ITEM_LIMIT,
    ToolResponse,
    ok,
    truncate_deep,
    unavailable,
)
from contractor_agent.signals import arbitration, enforcement, finance
from contractor_agent.signals.engine import compute
from contractor_agent.signals.model import TERMINAL_RU, VERDICT_RU, Severity, SignalSet

NOT_FOUND_NOTE = "Компании с таким ИНН в базе нет — ответить по ней нельзя."
NOT_FOUND_TEXT_NOTE = "«Не найдено» — не значит «нет»: данные могли не найтись."
MAX_COMPARE = 10
SEARCH_LIMIT_MAX = 20


@dataclass
class Tools:
    source: ReportSource

    # --- поиск и сводка ---------------------------------------------------------

    def search_company(self, query: str, limit: int = 5) -> ToolResponse:
        limit = max(1, min(limit, SEARCH_LIMIT_MAX))
        hits = self.source.search(query, limit)
        items = [
            {
                "inn": h.inn,
                "ogrn": h.ogrn,
                "name": h.short_name,
                "address": h.address,
                "svetofor": svetofor_ru(h.risk_level),
                "zsk": zsk_ru(h.zsk_risk_level),
                "report_date": h.report_date,
            }
            for h in hits
        ]
        return ok({"items": items, "total": len(items), "query": query})

    def get_report_summary(self, inn: str) -> ToolResponse:
        report = self.source.get(inn)
        if report is None:
            return unavailable("not_found", note=NOT_FOUND_NOTE)
        info = report.base_info
        reg = info.registration_info
        okved = (
            report.kinds_of_activity_info.main_kind_of_activity
            if report.kinds_of_activity_info
            else None
        )
        other = (
            report.kinds_of_activity_info.other_kinds_of_activity
            if report.kinds_of_activity_info
            else None
        )
        person = report.founders_info.auth_person if report.founders_info else None
        data = {
            "inn": info.inn,
            "ogrn": info.ogrn,
            "kpp": info.kpp,
            "short_name": info.short_name,
            "full_name": info.full_name,
            "sole_proprietor": finance.is_sole_proprietor(report),
            "address": info.address,
            "registered": reg.registration_date if reg else None,
            "age_years": finance.company_age_years(report),
            "company_size": info.company_size,
            "staff": info.staff,
            "status": report.status.status,
            "status_reason": report.status.reason_name,
            "labels": {
                "svetofor": svetofor_ru(info.risk_level),
                "svetofor_raw": info.risk_level,
                "svetofor_path": "report.baseInfo.riskLevel",
                "zsk": zsk_ru(report.zsk_risk_level),
                "zsk_path": "report.zskRiskLevel",
            },
            "main_activity": {"code": okved.code, "description": okved.description}
            if okved
            else None,
            "other_activities_count": len(other or []),
            "head": {
                "name": person.name,
                "position": person.position_name,
                "since": person.position_date,
            }
            if person
            else None,
            "sections": report.sections(),
            "paths": {  # адреса для цитат — ключи этого JSON адресами не являются
                "registered": "report.baseInfo.registrationInfo.registrationDate",
                "age_years": "report.baseInfo.registrationInfo.yearsFromRegistration",
                "company_size": "report.baseInfo.companySize",
                "staff": "report.baseInfo.staff",
                "address": "report.baseInfo.address",
                "status": "report.status.status",
                "status_reason": "report.status.reasonName",
                "main_activity": "report.kindsOfActivityInfo.mainKindOfActivity",
                "other_activities": "report.kindsOfActivityInfo.otherKindsOfActivity",
                "head": "report.foundersInfo.authPerson",
                "svetofor": "report.baseInfo.riskLevel",
                "zsk": "report.zskRiskLevel",
            },
            "counts": {
                "execution_proceedings": len(report.execution_proceedings or []),
                "licenses": len(report.licenses or []),
                "inspections": len(report.inspections or []),
                "related_companies": len(report.related_companies or []),
                "negative_flags": len(report.reputational_risks.negative or [])
                if report.reputational_risks
                else None,
            },
        }
        return ok(
            data,
            source_paths=[
                "report.baseInfo",
                "report.status",
                "report.zskRiskLevel",
                "report.kindsOfActivityInfo.mainKindOfActivity",
            ],
            report_date=report.report_date,
        )

    # --- сигналы -----------------------------------------------------------------

    def get_risk_signals(self, inn: str) -> ToolResponse:
        report = self.source.get(inn)
        if report is None:
            return unavailable("not_found", note=NOT_FOUND_NOTE)
        signal_set = compute(report)
        data = _signal_set_data(signal_set)
        data["labels"] = {
            "svetofor": svetofor_ru(report.base_info.risk_level),
            "svetofor_path": "report.baseInfo.riskLevel",
            "zsk": zsk_ru(report.zsk_risk_level),
            "zsk_path": "report.zskRiskLevel",
        }
        paths = [s.source_path for s in signal_set.signals] + [
            g.source_path for g in signal_set.gaps
        ]
        return ok(data, source_paths=_dedupe(paths)[:ITEM_LIMIT], report_date=report.report_date)

    # --- финансы -----------------------------------------------------------------

    def get_financials(self, inn: str) -> ToolResponse:
        report = self.source.get(inn)
        if report is None:
            return unavailable("not_found", note=NOT_FOUND_NOTE)
        result = finance.run(report)
        rows = finance.real_rows(report)
        if not rows:
            gap = result.gaps[0]
            return unavailable(gap.reason.value, note=gap.text_ru, report_date=report.report_date)
        years = []
        for row in rows:
            f = row.data
            common, assets, liabilities = f.common, f.assets, f.liabilities
            current = assets.current_assets.total if assets and assets.current_assets else None
            short = (
                liabilities.short_term_liabilities.total
                if liabilities and liabilities.short_term_liabilities
                else None
            )
            long = (
                liabilities.long_term_duties.total
                if liabilities and liabilities.long_term_duties
                else None
            )
            capitals = liabilities.capitals if liabilities else None
            total_assets = assets.total_assets if assets else None
            proceeds = common.proceeds if common else None
            profit = common.profit if common else None
            years.append(
                {
                    "year": row.year,
                    "proceeds": proceeds,
                    "profit": profit,
                    "total_assets": total_assets,
                    "current_assets": current,
                    "short_term_liabilities": short,
                    "long_term_duties": long,
                    "capitals": capitals,
                    "current_liquidity": _ratio(current, short),
                    "profitability_pct": _ratio(profit, proceeds, 100),
                    "sustainability": _ratio((capitals or 0) + (long or 0), total_assets)
                    if capitals is not None
                    else None,
                    "path": row.path,
                }
            )
        coef = report.coefficient
        bank = (
            {
                "year": coef.year,
                "profitability_pct": coef.profitability,
                "solvency": coef.solvency,
                "sustainability": coef.sustainability,
                "path": "report.coefficient",
                "meaning": (
                    "рентабельность продаж %, общая платёжеспособность, "
                    "финансовая устойчивость — расчёт банка"
                ),
            }
            if coef
            else None
        )
        net = finance.net_assets(report)
        data = {
            "unit": "руб.",
            "years": years,
            "net_assets": {"value": net.value, "year": net.year, "path": net.path} if net else None,
            "bank_coefficient": bank,
            "signals": [_signal_data(s) for s in result.signals],
            "gaps": [_gap_data(g) for g in result.gaps],
        }
        paths = [r.path for r in rows] + (["report.coefficient"] if coef else [])
        return ok(data, source_paths=paths, report_date=report.report_date)

    # --- долги и суды ------------------------------------------------------------

    def get_enforcement_summary(self, inn: str) -> ToolResponse:
        report = self.source.get(inn)
        if report is None:
            return unavailable("not_found", note=NOT_FOUND_NOTE)
        if report.section_state("executionProceedings") == "absent":
            return unavailable(
                "section_absent",
                note="В отчёте нет раздела об исполнительных производствах — оценить нельзя.",
                report_date=report.report_date,
            )
        active, finished, unknown = enforcement.split(report)
        result = enforcement.run(report)
        flag = enforcement._bank_flag(report)
        data = {
            "active": _aggregate_data(active),
            "finished": _aggregate_data(finished),
            "unknown_status_count": len(unknown),
            "top_active": [_item_data(i) for i in active.top()],
            "recent_active": [
                _item_data(i)
                for i in sorted(
                    active.items, key=lambda i: i.data.date or i.data.date.min, reverse=True
                )[:5]
            ],
            "total": active.count + finished.count + len(unknown),
            "bank_flag": {"negative": flag[1], "text": flag[2], "path": flag[0]} if flag else None,
            "signals": [_signal_data(s) for s in result.signals],
            "gaps": [_gap_data(g) for g in result.gaps],
        }
        note = None
        if active.count == 0 and finished.count == 0:
            note = "Исполнительных производств в отчёте не найдено. " + NOT_FOUND_TEXT_NOTE
        paths = ["report.executionProceedings"] + [i.path for i in active.top()]
        if flag:
            paths.append(flag[0])
        return ok(data, source_paths=paths, report_date=report.report_date, note=note)

    def get_arbitration_summary(self, inn: str) -> ToolResponse:
        report = self.source.get(inn)
        if report is None:
            return unavailable("not_found", note=NOT_FOUND_NOTE)
        agg = report.arbitration_by_status
        cases = report.arbitration_cases
        if agg is None and not cases:
            return unavailable(
                "section_absent",
                note="В отчёте нет раздела об арбитражных делах — оценить нельзя.",
                report_date=report.report_date,
            )
        defendant = arbitration.defendant_roles(agg)
        plaintiff = arbitration.plaintiff_roles(agg)
        result = arbitration.run(report)
        flag = arbitration._bank_flag(report)
        data = {
            "common_count": agg.common_count if agg else None,
            "common_amount": agg.common_amount if agg else None,
            "defendant": _roles_data(defendant),
            "plaintiff": _roles_data(plaintiff),
            "by_years": [
                {
                    "year": c.year,
                    "plaintiff_count": c.plaintiff_count,
                    "plaintiff_amount": c.plaintiff_amount,
                    "defendant_count": c.defendant_count,
                    "defendant_amount": c.defendant_amount,
                    "path": f"report.arbitrationCases[{i}]",
                }
                for i, c in enumerate(cases or [])
            ],
            "by_years_window": (
                "разбивка по годам покрывает только 2023–2026; сводка — за всё время"
            ),
            "bank_flag": {"negative": flag[1], "text": flag[2], "path": flag[0]} if flag else None,
            "signals": [_signal_data(s) for s in result.signals],
        }
        found = (
            (agg is not None and bool(agg.common_count))
            or defendant.total
            or plaintiff.total
            or bool(cases)
        )
        note = None if found else "Арбитражных дел в отчёте не найдено. " + NOT_FOUND_TEXT_NOTE
        paths = ["report.arbitrationByStatus"]
        if cases:
            paths.append("report.arbitrationCases")
        if flag:
            paths.append(flag[0])
        return ok(data, source_paths=paths, report_date=report.report_date, note=note)

    # --- сырые секции и сравнение -----------------------------------------------

    def get_section(self, inn: str, name: str) -> ToolResponse:
        report = self.source.get(inn)
        if report is None:
            return unavailable("not_found", note=NOT_FOUND_NOTE)
        if name not in SECTIONS:
            return unavailable(
                "unknown_section", note=f"Секции «{name}» нет. Доступные: {', '.join(SECTIONS)}."
            )
        state = report.section_state(name)
        if state == "absent":
            return unavailable(
                "section_absent",
                note=f"Раздела «{name}» в отчёте нет — оценить по нему нельзя.",
                report_date=report.report_date,
            )
        value = resolve(report, f"report.{name}")
        raw = (
            value.model_dump()
            if hasattr(value, "model_dump")
            else [v.model_dump() for v in value]
            if isinstance(value, list)
            else value
        )
        note = "Раздел есть, но данных в нём нет." if state == "empty" else None
        return ok(
            {"section": name, "state": state, "content": truncate_deep(raw)},
            source_paths=[f"report.{name}"],
            report_date=report.report_date,
            note=note,
        )

    def compare_companies(self, inns: list[str]) -> ToolResponse:
        if not inns:
            return unavailable("bad_request", note="Нужен хотя бы один ИНН.")
        inns = list(dict.fromkeys(inns))[:MAX_COMPARE]
        items = []
        paths = []
        for inn in inns:
            report = self.source.get(inn)
            if report is None:
                items.append({"inn": inn, "available": False, "reason": "not_found"})
                continue
            signal_set = compute(report)
            net = finance.net_assets(report)
            proceeds = finance.pick(finance.real_rows(report), finance._proceeds, "common.proceeds")
            items.append(
                {
                    "inn": inn,
                    "available": True,
                    "name": report.base_info.short_name,
                    "report_date": report.report_date,
                    "labels": {
                        "svetofor": svetofor_ru(report.base_info.risk_level),
                        "zsk": zsk_ru(report.zsk_risk_level),
                    },
                    "verdict": signal_set.verdict.value,
                    "terminal": signal_set.terminal,
                    "score": signal_set.score,
                    "signal_counts": {s.value: len(signal_set.by_severity(s)) for s in Severity},
                    "critical": [
                        {
                            "title": s.title_ru,
                            "explanation": s.explanation_ru,
                            "source_path": s.source_path,
                        }
                        for s in signal_set.by_severity(Severity.CRITICAL)
                    ],
                    "moderate_titles": [
                        s.title_ru for s in signal_set.by_severity(Severity.MODERATE)
                    ],
                    "net_assets": {"value": net.value, "year": net.year} if net else None,
                    "proceeds": {"value": proceeds.value, "year": proceeds.year}
                    if proceeds
                    else None,
                    "gaps": [g.criterion for g in signal_set.gaps],
                }
            )
            paths.append("report.baseInfo.inn")
        return ok({"items": items, "total": len(items)}, source_paths=_dedupe(paths))


# --- вспомогательное ----------------------------------------------------------------


def _ratio(numerator: int | None, denominator: int | None, scale: int = 1) -> float | None:
    if numerator is None or not denominator:
        return None
    return round(float(Decimal(numerator) * scale / Decimal(denominator)), 3)


def _dedupe(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(paths))


def _signal_data(s) -> dict[str, Any]:
    return {
        "code": s.code,
        "severity": s.severity.value,
        "title": s.title_ru,
        "explanation": s.explanation_ru,
        "value": s.value,
        "source_path": s.source_path,
        "source_paths": s.source_paths,
        "terminal": s.terminal,
        "origin": s.origin.value,
        "year": s.year,
        "bank_text": s.bank_text,
    }


def _gap_data(g) -> dict[str, Any]:
    return {
        "criterion": g.criterion,
        "reason": g.reason.value,
        "text": g.text_ru,
        "ask": g.ask_ru,
        "source_path": g.source_path,
    }


def _signal_set_data(ss: SignalSet) -> dict[str, Any]:
    return {
        "verdict": ss.verdict.value,
        "verdict_ru": TERMINAL_RU if ss.terminal else VERDICT_RU[ss.verdict],
        "terminal": ss.terminal,
        "score": ss.score,
        "signals": {s.value: [_signal_data(x) for x in ss.by_severity(s)] for s in Severity},
        "gaps": [_gap_data(g) for g in ss.gaps],
    }


def _aggregate_data(agg: enforcement.Aggregate) -> dict[str, Any]:
    return {
        "count": agg.count,
        "known_sum": agg.known_sum,
        "unknown_amount_count": agg.unknown_count,
        "sum_is_lower_bound": agg.unknown_count > 0,
        "recent_12m_count": agg.recent_count,
        "older_3y_count": len(agg.stale),
        "earliest": agg.earliest,
        "latest": agg.latest,
    }


def _item_data(item: enforcement.Item) -> dict[str, Any]:
    return {
        "number": item.data.number,
        "date": item.data.date,
        "amount": item.data.amount,
        "active": item.data.active,
        "path": item.path,
    }


def _roles_data(roles: arbitration.Roles) -> dict[str, Any]:
    def bucket(b):
        return (
            {"count": b.count, "amount": b.amount, "path": b.count_path}
            if b
            else {"count": 0, "amount": None}
        )

    return {
        "finished": bucket(roles.finished),
        "pending": bucket(roles.pending),
        "appealed": bucket(roles.appealed),
    }
