# ruff: noqa: E501
"""Paired SSE benchmark, including real multi-turn conversations.

uv run python scripts/compare_branches.py --repeats 1
--gold runs the original 50 questions with the same new evaluator for both versions.
Raw answers are immutable within each output directory. No judge calls by default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import math
import re
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]


def evaluate(turn, answer, tools):
    if not answer:
        return ["нет завершённого ответа"]
    failures = []
    text = answer.get("text_md", "")
    for pattern in turn.get("must", []):
        if not re.search(pattern, text, re.I):
            failures.append("не найдено: " + pattern)
    for pattern in turn.get("forbidden", []):
        if re.search(pattern, text, re.I):
            failures.append("запрещённое: " + pattern)
    cards = answer.get("cards") or ([answer["card"]] if answer.get("card") else [])
    if turn.get("no_card") and cards:
        failures.append("подставлена карточка")
    dates = set(answer.get("report_dates") or {})
    if turn.get("no_dates") and dates:
        failures.append("подставлена дата старой компании")
    if turn.get("inns") and dates != set(turn["inns"]):
        failures.append("набор ИНН ответа не совпал")
    for name in turn.get("tools", []):
        if name not in tools:
            failures.append("не вызван " + name)
    for inn, verdict in turn.get("verdicts", {}).items():
        card = next((c for c in cards if c["inn"] == inn), None)
        if not card or card["verdict"] != verdict:
            failures.append("неверный вывод по " + inn)
    if turn.get("gold"):
        from evals.checks import check
        from evals.gold import GoldQuestion

        from contractor_agent.agent.schema import Answer

        q = GoldQuestion.model_validate(turn["gold"])
        result = check(q, Answer.model_validate(answer), turn["date"])
        failures.extend(result.failures)
    return failures


async def request(client, url, thread, question):
    began = time.perf_counter()
    events = []
    answer = None
    error = None
    runtime = None
    try:
        async with client.stream(
            "POST",
            url + "/v1/runs/stream",
            json={"thread_id": thread, "input": {"question": question}},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:])
                events.append(event)
                if event["type"] == "end":
                    answer = event["data"]["output"]
                if event["type"] == "error":
                    error = event["data"].get("message")
                    runtime = event["data"].get("runtime")
    except (httpx.HTTPError, ValueError) as e:
        error = type(e).__name__
    return {
        "question": question,
        "answer": answer,
        "events": events,
        "error": error,
        "runtime": runtime or (answer or {}).get("runtime"),
        "duration_s": round(time.perf_counter() - began, 3),
    }


def report(out, records, manifest):
    summaries = {}
    for label in ("develop", "review-v4"):
        rows = [r for r in records if r["label"] == label]
        dur = sorted(r["duration_s"] for r in rows)
        runtimes = [r.get("runtime") or (r.get("answer") or {}).get("runtime") or {} for r in rows]
        summaries[label] = {
            "n": len(rows),
            "passed": sum(not r["failures"] and not r["error"] for r in rows),
            "errors": sum(bool(r["error"]) or not r["answer"] for r in rows),
            "rejected_citations": sum(
                len((r.get("answer") or {}).get("invalid_citations") or []) for r in rows
            ),
            "median_s": round(statistics.median(dur), 2) if dur else None,
            "p95_s": dur[math.ceil(len(dur) * 0.95) - 1] if dur else None,
            "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in runtimes),
            "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in runtimes),
            "llm_calls": sum(r.get("llm_calls", 0) for r in runtimes),
            "usage_incomplete_runs": sum(
                not runtime or runtime.get("usage_complete") is not True for runtime in runtimes
            ),
        }
    (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    lines = [
        "# Сравнение develop и review-v4",
        "",
        f"Модель: `{manifest['develop']['parameters']['model']}`.",
        "Одинаковые модель, endpoint, temperature=0, reasoning=low, max_tokens=4096; резервов нет.",
        f"Подготовительные ходы: {manifest.get('context_turns_per_version', 0)} на версию; сохраняются отдельно и не входят в метрики целевых вопросов.",
        "Проверки сценариев — диагностические. Это не оценка полного качества и не LLM-судья. Ошибки остаются в знаменателе.",
        "",
        "| Версия | Проверки | Ошибки | Медиана, с | p95, с | Входные токены | Выходные токены | Вызовы LLM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in summaries.items():
        lines.append(
            f"| {label} | {s['passed']}/{s['n']} | {s['errors']} | {s['median_s']} | {s['p95_s']} | {s['input_tokens']} | {s['output_tokens']} | {s['llm_calls']} |"
        )
    lines += ["", "## Ответы и провалы", ""]
    for r in records:
        if r["failures"] or r["error"]:
            lines += [
                f"- {r['label']} / {r['case']} / ход {r['turn'] + 1}: "
                + "; ".join(r["failures"] + [r["error"]] if r["error"] else r["failures"])
            ]
    (out / "report.md").write_text("\n".join(lines))
    esc = html.escape
    pairs = {}
    for r in records:
        pairs.setdefault((r["case"], r["repeat"], r["turn"]), {})[r["label"]] = r
    rows = []
    for (case, _rep, turn), pair in pairs.items():
        q = next(iter(pair.values()))["question"]
        cols = []
        for label in ("develop", "review-v4"):
            r = pair.get(label, {})
            text = (r.get("answer") or {}).get("text_md") or r.get("error") or ""
            failures = "; ".join(r.get("failures") or [])
            cols.append(
                "<article><h3>"
                + label
                + '</h3><p class="meta">'
                + str(r.get("duration_s", ""))
                + " c · "
                + esc(failures or "Проверки сценария пройдены")
                + "</p><pre>"
                + esc(text)
                + "</pre></article>"
            )
        rows.append(
            "<section><h2>"
            + esc(case)
            + f" · повтор {_rep + 1} · ход {turn + 1}</h2><p>"
            + esc(q)
            + '</p><div class="pair">'
            + "".join(cols)
            + "</div></section>"
        )
    (out / "report.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Сравнение версий</title><style>body{max-width:1440px;margin:40px auto;padding:0 28px;background:#f3f5f7;color:#172331;font:15px/1.6 Arial}h1{font-size:32px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:white;border:1px solid #dce1e7;padding:22px;border-radius:10px}pre{white-space:pre-wrap;font:14px/1.7 Arial;overflow-wrap:anywhere}.meta{font-size:12px;color:#647181}section{margin:35px 0}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>develop / review-v4</h1><p>Одна модель. Одинаковые сценарии. Ответы без редактирования.</p><pre>'
        + esc("\n".join(lines[:12]))
        + "</pre>"
        + "".join(rows)
        + "</html>"
    )


async def main(args):
    out = Path(args.output or ROOT / "runs" / ("ab-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    out.mkdir(parents=True, exist_ok=False)
    cases = yaml.safe_load((ROOT / "evals/comparison.yaml").read_text())["cases"]
    if args.gold:
        from evals.gold import load_gold

        gold = load_gold()
        cases = [
            {
                "id": q.id,
                "context_question": (
                    f"Проверь {gold.card(q.inn).company}, ИНН {q.inn}" if q.follow_up else None
                ),
                "turns": [
                    {
                        "question": q.question,
                        "gold": q.model_dump(),
                        "date": gold.card(q.inn).report_date,
                    }
                ],
            }
            for q in gold.questions()
        ]
    if args.limit:
        cases = cases[: args.limit]
    urls = {"develop": args.baseline, "review-v4": args.candidate}
    records = []
    async with httpx.AsyncClient(timeout=200) as client:
        manifest = {
            label: (await client.get(url + "/v1/comparison/config")).json()
            for label, url in urls.items()
        }
        if manifest["develop"]["parameters"] != manifest["review-v4"]["parameters"]:
            raise SystemExit("Different model settings")
        for m in manifest.values():
            if m["fallback_models"]:
                raise SystemExit("Fallback models must be disabled")
        manifest["cases_sha256"] = hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest()
        manifest["repeats"] = args.repeats
        manifest["started_at"] = datetime.now().astimezone().isoformat()
        manifest["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest["context_turns_per_version"] = (
            sum(bool(c.get("context_question")) for c in cases) * args.repeats
        )
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print("Results:", out, flush=True)
        for i, case in enumerate(cases):
            for rep in range(args.repeats):
                threads = {label: uuid.uuid4().hex for label in urls}
                order = list(urls) if (i + rep) % 2 == 0 else list(reversed(urls))
                if case.get("context_question"):
                    # Seed with a real completed dialogue, equally for both versions.
                    setup = await asyncio.gather(
                        *(
                            request(client, urls[label], threads[label], case["context_question"])
                            for label in order
                        )
                    )
                    for label, record in zip(order, setup, strict=True):
                        (out / f"setup-{case['id']}-{rep}-{label}.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2)
                        )
                for turn_index, turn in enumerate(case["turns"]):
                    # Independent local services, at most two simultaneous cloud requests.
                    batch = await asyncio.gather(
                        *(
                            request(client, urls[label], threads[label], turn["question"])
                            for label in order
                        )
                    )
                    for label, record in zip(order, batch, strict=True):
                        tools = [e["data"]["name"] for e in record["events"] if e["type"] == "tool"]
                        record.update(
                            label=label,
                            case=case["id"],
                            repeat=rep,
                            turn=turn_index,
                            failures=evaluate(turn, record["answer"], tools),
                        )
                        records.append(record)
                        (out / f"{case['id']}-{rep}-{turn_index}-{label}.json").write_text(
                            json.dumps(record, ensure_ascii=False, indent=2)
                        )
                        print(
                            label,
                            case["id"],
                            turn_index,
                            record["duration_s"],
                            "PASS" if not record["failures"] and not record["error"] else "FAIL",
                            flush=True,
                        )
                    report(out, records, manifest)
        for label, url in urls.items():
            end = (await client.get(url + "/v1/comparison/config")).json()
            if end != manifest[label]:
                raise SystemExit("Server build changed during benchmark")
    print(json.dumps(json.loads((out / "summary.json").read_text()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="http://127.0.0.1:8081")
    p.add_argument("--candidate", default="http://127.0.0.1:8082")
    p.add_argument("--output")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--gold", action="store_true")
    asyncio.run(main(p.parse_args()))
