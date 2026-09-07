"""Export every regression into Alan's evaluators YAML format (hard/high).

References come from versioned questions and the report oracle, never from a
candidate model's answers. Existing must-mention rules are kept for audit but
are not converted into mandatory wording or a guaranteed positive score.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

from contractor_agent.data.loader import load_snapshot, normalize_name, strip_legal_form
from contractor_agent.mcp_server.tools import Tools
from evals.gold import GOLD_PATH, load_gold

POLICY = "contractor-reference-v2-high"


def expected_calls(question, fallback, known, *, names, spec=None):
    """Reviewed reference policy, independent of runtime routing/evidence gates.

    Identity/report date first, then only sources needed for this question.
    Explicit last-turn targets come from the gold case, not candidate output.
    Independent read ordering is canonical; the hard judge handles minor reorderings.
    """
    ids = list(dict.fromkeys(re.findall(r"\b(?:\d{12}|\d{10})\b", question)))
    normal = normalize_name(question)
    named = [inn for inn, name in names.items() if strip_legal_form(normalize_name(name)) in normal]
    selected = ids or named or known or [fallback]
    if spec:
        # Some questions mention an old company only to explicitly exclude it.
        selected = (
            [spec.inn, *spec.extra_inns]
            if spec.type == "comparison"
            else (ids if ids and spec.inn not in ids else [spec.inn])
        )
    if (
        spec
        and len(known) > 1
        and not ids
        and re.search(r"кого|кто из|из них|выбра|выбер", question, re.I)
    ):
        selected = known
    selected = list(dict.fromkeys(selected))
    if spec and spec.id in {"edge-grupp-search-ambiguous", "edge-omega-same-name-ask-inn"}:
        query = "ГРУПП" if "grupp" in spec.id else "ОМЕГА"
        return [("search_company", {"query": query})], []
    calls = []
    ogrn = re.search(r"\b\d{13}\b", question)
    if not ids and (not known or ogrn):
        for inn in selected:
            query = ogrn.group() if ogrn else strip_legal_form(names.get(inn, question))
            calls.append(("search_company", {"query": query}))
    comparison = len(selected) > 1
    if comparison:
        calls.append(("compare_companies", {"inns": selected}))
    else:
        calls.append(("get_report_summary", {"inn": selected[0]}))
    if spec and spec.id in {"edge-techprof-inn-not-found", "direct-v8-11"}:
        return calls, selected

    topic = spec.topic if spec else ""
    source = (spec.source_path or "") if spec else ""
    decision = bool(
        re.search(
            r"(?:можно|могу|стоит|совет|рекоменд|выбр|лучше|безопасн|что.*запросить)"
            r"|(?:отсроч|постоплат)",
            question,
            re.I,
        )
    )
    risk = (
        bool(
            re.search(
                r"проверь|риски|проблемы|сигнал|недостовер|недостовр|блокиров|метк|светофор|ЗСК",
                question,
                re.I,
            )
        )
        or decision
        or source.startswith("report.reputationalRisks")
    )
    reads = []
    if risk and not comparison:
        reads.append(("get_risk_signals", {}))
    if (
        topic == "courts"
        or source.startswith("report.arbitration")
        or re.search(r"суд|арбитраж|иск(?:и|ов|а|\b)|истец|истц", question, re.I)
    ):
        reads.append(("get_arbitration_summary", {}))
    if (
        topic == "enforcement"
        or source.startswith("report.executionProceedings")
        or re.search(r"пристав|исполнительн|отсроч", question, re.I)
    ):
        reads.append(("get_enforcement_summary", {}))
    if (
        topic == "finance"
        or source.startswith("report.finReports")
        or re.search(r"финанс|выручк|прибыл|ликвид|актив|капитал|отсроч|постоплат", question, re.I)
    ):
        reads.append(("get_financials", {}))
    sections = [
        (r"лиценз|допуск|удостоверен", "licenses"),
        (r"телефон|позвон", "phones"),
        (r"учред|руковод|директор|управляющ|владел|бенефициар", "foundersInfo"),
        (r"проверял|проверк.*(?:гос|орган)|инспекц|нарушен", "inspections"),
        (r"закуп|госконтракт|тендер", "procurements"),
        (r"филиал", "branchesInfo"),
        (r"связан", "relatedCompanies"),
        (r"оквэд|профиль|занима|торгуют|обуч|образоват", "kindsOfActivityInfo"),
        (r"налогов.*режим|систем.*налого", "taxSystem"),
    ]
    for pattern, section in sections:
        if source.startswith("report." + section) or re.search(pattern, question, re.I):
            reads.append(("get_section", {"name": section}))
    # A comparison tool already includes aggregated courts and enforcement.
    if comparison:
        reads = [
            (n, a)
            for n, a in reads
            if n not in {"get_arbitration_summary", "get_enforcement_summary"}
        ]
    for inn in selected:
        calls.extend((name, {"inn": inn, **extra}) for name, extra in reads)
    return calls, selected


def export(out: Path):
    tools = Tools(load_snapshot(Path("data")))
    gold = load_gold()
    direct = load_gold(Path("evals/suites/v8_direct_answers.yaml"))
    names = {c.inn: c.company for c in [*gold.cards, *direct.cards]}
    questions = {q.id: (q, "regression") for q in gold.questions()}
    for q in direct.questions():
        if q.type != "card":
            questions[q.id] = (q, "direct")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "reference_policy": POLICY,
        "judge": "z-ai/glm-5.3-flash",
        "agent": "openai/gpt-oss-20b",
        "strictness": "high",
        "target_score_100": 85,
        "reference_status": "Reviewed source-based canonical reads; no candidate answers used",
        "gold_sha256": hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest(),
        "count": len(questions),
        "scenarios": [],
    }
    for q, group in questions.values():
        queries = list(q.prior_questions)
        if q.follow_up and not queries:
            queries.append(f"Проверь {names[q.inn]}, ИНН {q.inn}")
        queries.append(q.question)
        known = []
        reference = []
        for turn, query in enumerate(queries):
            reference.append({"role": "user", "content": query})
            calls, selected = (
                ([], known)
                if q.type == "guard" or q.expect_tools is False
                else expected_calls(
                    query,
                    q.inn,
                    known,
                    names=names,
                    spec=q if turn == len(queries) - 1 else None,
                )
            )
            if calls:
                reference.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": f"ref-{turn}-{i}", "function": {"name": name, "arguments": args}}
                            for i, (name, args) in enumerate(calls)
                        ],
                    }
                )
                for name, args in calls:
                    oracle = getattr(tools, name)(**args)
                    reference.append(
                        {
                            "role": "tool",
                            "content": json.dumps(
                                {
                                    "tool": name,
                                    "arguments": args,
                                    "report_oracle": oracle.model_dump(mode="json"),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
            known = selected or known
            if turn == len(queries) - 1:
                target = str(
                    q.expected_value
                    or "Дать ответ на поставленный вопрос по сведениям отчёта. "
                    "Если данных нет — прямо сказать об этом."
                )
                if q.type == "card":
                    target += "\nВ полной карточке обязательны факты: " + "; ".join(q.must_name)
                reference.append(
                    {
                        "role": "assistant",
                        "content": "Ожидаемый смысл ответа (формулировки свободные): " + target,
                    }
                )
            else:
                reference.append(
                    {
                        "role": "assistant",
                        "content": "Ответить на этот вопрос по данным инструментов, "
                        "сохранив компанию/группу для следующего хода.",
                    }
                )
        spec = {
            "context": {
                "source_question_id": q.id,
                "source_group": group,
                "reference_policy": POLICY,
                "legacy_assertions": json.dumps(q.model_dump(), ensure_ascii=False),
            },
            "agent": {
                "endpoint_url": "http://127.0.0.1:18102/evaluate",
                "timeout": 150,
                "request_template": {
                    "user_message": "{{ user_message }}",
                    "thread_id": "{{ thread_id }}",
                },
                "response_paths": {
                    "final_answer": "$.result",
                    "trajectory": "$.trajectory",
                    "session_id": "$.session_id",
                },
            },
            "simulation": {
                "initial_query": queries[0],
                "flow": "Заданные реплики без перефразирования.",
                "fixed_queries": queries,
                "max_turns": len(queries),
                "reference_outputs": reference,
            },
            "judge": [
                {"type": "trajectory_match", "parameters": {"strictness": "high"}},
                {"type": "hallucination"},
            ],
        }
        path = out / group / (q.id + ".yml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest["scenarios"].append(
            {
                "id": q.id,
                "path": str(path.relative_to(out)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Exported", len(questions), "scenarios; high strictness; fixed user questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    export(parser.parse_args().out)
