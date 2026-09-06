"""Exhaust every local report through supported deterministic factual renderers.

This checks provenance/amount invariants, not the semantic quality of arbitrary prose.
Run with PYTHONPATH=src python -m evals.canonical --out runs/canonical.json.
"""

import argparse
import json
from pathlib import Path

from contractor_agent.agent.card_answer import card_answer
from contractor_agent.agent.citations import validate_citations
from contractor_agent.agent.factual_sections import factual_sections
from contractor_agent.agent.financial_answers import financial_answer
from contractor_agent.agent.money_guard import label_violations, monetary_violations
from contractor_agent.agent.section_answers import section_answer
from contractor_agent.data.loader import load_snapshot
from contractor_agent.mcp_server.tools import Tools


def audit(data_dir: Path) -> dict:
    snapshot = load_snapshot(data_dir)
    tools = Tools(snapshot)
    checks, failures = 0, []
    for record in snapshot:
        inn = record.report.inn
        renderers = [
            ("Проверь компанию", lambda inn=inn: card_answer(tools, [inn], "Проверь компанию")),
            (
                "Покажи финансовое положение",
                lambda inn=inn: financial_answer(tools, [inn], "Финансы"),
            ),
            ("Суды", lambda inn=inn: section_answer(tools, [inn], "Суды")),
            ("Открытые иски", lambda inn=inn: section_answer(tools, [inn], "Открытые иски")),
            ("Суды за 2025 год", lambda inn=inn: section_answer(tools, [inn], "Суды за 2025 год")),
            (
                "Долги у приставов",
                lambda inn=inn: section_answer(tools, [inn], "Долги у приставов"),
            ),
        ]
        for question in (
            "Телефон",
            "Лицензии",
            "Учредители",
            "Кто проверял компанию?",
            "Госзакупки",
        ):
            renderers.append(
                (question, lambda q=question, inn=inn: factual_sections(tools, [inn], q))
            )
        for question, render in renderers:
            checks += 1
            answer = render()
            if answer is None:
                failures.append({"inn": inn, "question": question, "error": "No answer"})
                continue
            invalid = [
                item
                for item in validate_citations(snapshot, [inn], answer.citations)
                if not item.ok
            ]
            invalid += monetary_violations(tools, [inn], answer.text_md, question)
            invalid += label_violations(tools, [inn], answer.text_md)
            for item in invalid:
                failures.append(
                    {
                        "inn": inn,
                        "question": question,
                        "claim": item.citation.claim,
                        "reason": item.why,
                    }
                )
    return {"checks": checks, "failures": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"checks": result["checks"], "failures": len(result["failures"])}))
    raise SystemExit(bool(result["failures"]))
