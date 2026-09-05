# ruff: noqa: E501
"""``python -m evals run|report``.

run    --model X [--judge-model Y | --no-judge] [--types card,answer] [--limit N] [--repeat K] [--refresh]
report [--models X,Y]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from contractor_agent.agent.llm import make_judge_llm, make_llm
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.settings import Settings
from evals.gold import load_gold
from evals.metrics import compute_metrics
from evals.report import render
from evals.runner import EvalRunner, RunRecord, model_slug

CACHE = Path("runs/evals")


async def _run(args: argparse.Namespace) -> int:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings()
    gold = load_gold()
    types = set(args.types.split(",")) if args.types else None
    questions = gold.questions(types)
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q.id in wanted or q.inn in wanted]
    if args.limit:
        questions = questions[: args.limit]
    model = args.model or settings.llm_model
    agent_llm = make_llm(settings, model)
    if args.tag:  # отдельный кэш и строка отчёта: промпт v2 против v1 на той же модели
        model = f"{model} [{args.tag}]"  # только имя для кэша и отчёта, не для API
    judge_llm = None if args.no_judge else make_judge_llm(settings, args.judge_model)
    async with AgentRuntime(settings, llm=agent_llm) as runtime:
        runner = EvalRunner(
            gold,
            runtime,
            cache_dir=CACHE,
            model_name=model,
            judge_llm=judge_llm,
            delay_s=args.delay,
        )
        print(
            f"модель {model}, судья {args.judge_model or settings.judge_model if judge_llm else 'нет'}, вопросов {len(questions)} × {args.repeat}",
            file=sys.stderr,
        )
        records = []
        for i, question in enumerate(questions, 1):
            batch = await runner.run([question], repeats=args.repeat, refresh=args.refresh)
            records.extend(batch)
            r = batch[-1]
            status = "ошибка" if r.error else ("ок" if r.checks_passed else "провал")
            judge = f" судья {r.judge.score}" if r.judge else ""
            print(
                f"  {i:3}/{len(questions)} {question.id:40} {status}{judge}  {r.duration_s}s",
                file=sys.stderr,
            )
    metrics = compute_metrics(records)
    print(render([metrics], {model: records}))
    return 0


def _report(args: argparse.Namespace) -> int:
    models = (
        args.models.split(",")
        if args.models
        else sorted(p.name for p in CACHE.iterdir() if p.is_dir())
    )
    all_metrics, all_records = [], {}
    for slug in models:
        folder = CACHE / model_slug(slug)
        records = [
            RunRecord.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(folder.rglob("*.json"))
        ]
        if not records:
            continue
        m = compute_metrics(records)
        all_metrics.append(m)
        all_records[m.model] = records
    text = render(all_metrics, all_records)
    (CACHE / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model")
    run.add_argument("--judge-model")
    run.add_argument("--tag", help="метка прогона (например prompt-v2): свой кэш и строка отчёта")
    run.add_argument("--no-judge", action="store_true")
    run.add_argument("--types", help="card,answer,refuse,infer")
    run.add_argument("--only", help="id вопросов или ИНН через запятую")
    run.add_argument("--limit", type=int)
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--delay", type=float, default=0.5)
    run.add_argument("--refresh", action="store_true", help="не читать кэш")
    report = sub.add_parser("report")
    report.add_argument("--models")
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    return _report(args)


if __name__ == "__main__":
    sys.exit(main())
