"""Documents worth requesting, tied to the company's report rather than a fixed next step."""

from contractor_agent.mcp_server.envelope import ToolResponse


def requested_documents(risks: dict, financials: ToolResponse) -> list[str]:
    signals = [s for group in risks["signals"].values() for s in group]
    codes = {s["code"] for s in signals}
    result = []
    if risks.get("terminal"):
        result.append(
            "Свежая выписка ЕГРЮЛ и подтверждение полномочий управляющего или ликвидатора."
        )
    if "flag_fnsBlocking" in codes:
        result.append("Подтверждение текущего статуса ограничений по банковским счетам.")
    if "enforcement_active" in codes:
        result.append(
            "Документы о погашении или текущем состоянии действующих обязательств у приставов."
        )
    if "arbitration_defendant_open" in codes:
        result.append("Документы о текущем состоянии открытых арбитражных требований к компании.")
    if "flag_invalidAuthpersonsData" in codes:
        result.append("Актуальные сведения о руководителе и подтверждение его полномочий.")
    if "flag_invalidAddress" in codes or "flag_invalidRegistrationData" in codes:
        result.append("Подтверждение актуального адреса и регистрационных данных.")
    if financials.available:
        years = financials.data.get("years", [])
        if (
            any(r.get("profit") is None or r.get("current_liquidity") is None for r in years)
            or "fin_stale" in codes
        ):
            result.append("Баланс и отчёт о финансовых результатах за два последних года.")
    else:
        result.extend(
            g["ask"] for g in risks["gaps"] if g.get("ask") and "финанс" in g["criterion"]
        )
    return list(dict.fromkeys(result))
