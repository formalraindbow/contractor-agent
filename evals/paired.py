"""Frozen, bounded A/B runs. Never rewrites historical eval caches or uses an LLM judge.

python -m evals.paired --env-file /path/to/.env --models MODEL_A MODEL_B --out runs/paired/NAME
Add --execute to make API calls. Default is a dry run. See docs/EVAL_READINESS_2026-09-06.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import random
import subprocess
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from contractor_agent.agent.llm import make_llm
from contractor_agent.agent.prompt import PROMPT_VERSION, prompt_fingerprint
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.settings import Settings, make_source
from evals.gold import GOLD_PATH, Gold, load_gold
from evals.runner import EvalRunner, RunRecord, model_slug

SUITES = Path(__file__).parent / "suites/readiness.json"


def select_suite(gold: Gold, name: str) -> list:
    ids = json.loads(SUITES.read_text())[name]
    questions = {q.id: q for q in gold.questions()}
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs in suite")
    missing = set(ids) - questions.keys()
    if missing:
        raise ValueError(f"Unknown question IDs: {sorted(missing)}")
    return [questions[qid] for qid in ids]


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Usage(BaseCallbackHandler):
    """Includes agent, formatter and repair calls, even when not stored in graph messages."""

    def __init__(self):
        self.calls = []
        self.failed_callbacks = 0

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                metadata = getattr(message, "response_metadata", {}) or {}
                self.calls.append(
                    {
                        "usage": getattr(message, "usage_metadata", None),
                        "reported_model": metadata.get("model_name"),
                        "finish_reason": metadata.get("finish_reason"),
                    }
                )

    def on_llm_error(self, error, **kwargs):
        # SDK retries may happen inside a callback: this is not a count of HTTP requests.
        self.failed_callbacks += 1


class ObservedRuntime(AgentRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns = []

    async def ask(self, question, thread_id="cli", *, history=()):
        started = time.perf_counter()
        turn = {"question": question}
        try:
            answer = await super().ask(question, thread_id, history=history)
            turn["answer"] = answer.model_dump(mode="json")
            return answer
        except BaseException as exc:
            turn["error_type"] = type(exc).__name__
            raise
        finally:
            turn["duration_s"] = round(time.perf_counter() - started, 3)
            self.turns.append(turn)


def quantile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def summarize(records: list[dict], models: list[str], planned: int, repeats: int = 2) -> dict:
    result = {"planned_scenarios": planned, "completed_scenarios": len(records), "models": {}}
    for model in models:
        rows = [r for r in records if r["record"]["model"] == model]
        completed = [r for r in rows if not r["record"]["error"]]
        turns = [t["duration_s"] for r in completed for t in r["turns"] if "answer" in t]
        groups = defaultdict(list)
        for row in rows:
            groups[row["record"]["question_id"]].append(row)
        repeat_groups = [g for g in groups.values() if repeats > 1 and len(g) == repeats]
        calls = [c for r in rows for c in r["calls"]]
        usage = [c["usage"] for c in calls if c["usage"] is not None]
        result["models"][model] = {
            "scenarios": len(rows),
            "agent_errors": len(rows) - len(completed),
            "code_passes": sum(
                r["record"]["checks_passed"] and not r["record"]["error"] for r in rows
            ),
            "by_type": {
                kind: {
                    "n": sum(r["record"]["type"] == kind for r in rows),
                    "passed": sum(
                        r["record"]["type"] == kind
                        and r["record"]["checks_passed"]
                        and not r["record"]["error"]
                        for r in rows
                    ),
                }
                for kind in sorted({r["record"]["type"] for r in rows})
            },
            "case_duration_p50_s": quantile([r["record"]["duration_s"] for r in rows], 0.5),
            "case_duration_p95_s": quantile([r["record"]["duration_s"] for r in rows], 0.95),
            "successful_turn_duration_p50_s": quantile(turns, 0.5),
            "successful_turn_duration_p95_s": quantile(turns, 0.95),
            "repeat_groups": len(repeat_groups),
            "planned_case_groups": planned // (len(models) * repeats),
            "groups_all_repeats_pass": sum(
                all(r["record"]["checks_passed"] and not r["record"]["error"] for r in g)
                for g in repeat_groups
            ),
            "observed_llm_completions": len(calls),
            "completions_with_usage": len(usage),
            "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
            "failed_llm_callbacks": sum(r["failed_callbacks"] for r in rows),
            "judge_coverage": 0,
            "semantic_correctness": None,
        }
    paired = defaultdict(dict)
    for row in records:
        r = row["record"]
        paired[(r["question_id"], r["repeat"])][r["model"]] = bool(
            r["checks_passed"] and not r["error"]
        )
    pairs = [p for p in paired.values() if all(m in p for m in models)]
    a, b = models
    result["paired_code_checks"] = {
        "matched_pairs": len(pairs),
        "both_pass": sum(p[a] and p[b] for p in pairs),
        "only_first_pass": sum(p[a] and not p[b] for p in pairs),
        "only_second_pass": sum(p[b] and not p[a] for p in pairs),
        "both_fail": sum(not p[a] and not p[b] for p in pairs),
    }
    result["limitations"] = [
        "Diagnostic selection of known cases; not a random sample or unseen holdout.",
        "Code passes are not semantic correctness; no hallucination rate measured.",
        "Dialogue turns are saved in full, but legacy checks assess only the last turn.",
        "SDK retries and failed-request tokens may be unobservable; sums are reported usage.",
        "Successful-turn latency excludes failed turns; see errors and scenario durations.",
    ]
    return result


async def run(args):
    gold = load_gold()
    questions = select_suite(gold, args.suite)
    if len(set(args.models)) != 2 or not 1 <= args.repeats <= 5:
        raise ValueError("Exactly two different models and 1–5 repeats required")
    if args.out.exists():
        raise ValueError("Output directory already exists. Use a new name; runs are immutable.")
    settings = Settings(
        _env_file=args.env_file,
        session_store="memory",
        llm_base_url=args.base_url,
        llm_fallback_models="",
        llm_reasoning_effort="low",
        llm_max_tokens=8192,
        llm_timeout=90,
        report_source="snapshot",
        data_dir=Path("data"),
        runs_dir=args.out / "traces",
    )
    jobs = []
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        ordered = questions[:]
        rng.shuffle(ordered)
        for q in ordered:
            models = args.models[:]
            rng.shuffle(models)
            jobs.extend((q, repeat, m) for m in models)
    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    source_paths = {
        Path(p)
        for p in tracked
        if p
        and Path(p).is_file()
        and (p.startswith(("src/", "evals/")) or p in ("uv.lock", "pyproject.toml"))
    }
    source_paths.update(
        p for p in Path("evals").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".yaml")
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "source_sha256": {str(p): digest(p) for p in sorted(source_paths)},
        "gold_sha256": digest(GOLD_PATH),
        "suite_sha256": digest(SUITES),
        "snapshot_sha256": digest(settings.data_dir / "contractors_audit.snapshot.json"),
        "prompt_version": PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint(),
        "base_url": args.base_url,
        "models": args.models,
        "fallback_models": [],
        "temperature": 0,
        "reasoning_effort": "low",
        "max_tokens": 8192,
        "sdk_retries": 2,
        "per_turn_timeout_s": 120,
        "scenario_timeout_s": 360,
        "repeats": args.repeats,
        "seed": args.seed,
        "concurrency": 2,
        "suite": args.suite,
        "question_ids": [q.id for q in questions],
        "type_counts": dict(Counter(q.type for q in questions)),
        "planned_scenarios": len(jobs),
        "job_order": [{"id": q.id, "repeat": rep, "model": m} for q, rep, m in jobs],
        "judge": None,
        "holdout": False,
    }
    if not args.execute:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    api_key = settings.api_key_for(args.base_url)
    if not api_key:
        raise ValueError("Missing API key for provider; no requests made")
    args.out.mkdir(parents=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    source = make_source(settings)
    records = []
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    stop = asyncio.Event()

    async def worker():
        while not queue.empty() and not stop.is_set():
            q, repeat, model = queue.get_nowait()
            usage = Usage()
            llm = make_llm(settings, model, api_key=api_key, fallbacks=[])
            for underlying in llm.models:
                underlying.callbacks = [usage]
            async with ObservedRuntime(settings, source=source, llm=llm) as runtime:
                runner = EvalRunner(
                    gold, runtime, cache_dir=args.out, model_name=model, agent_timeout_s=120
                )
                try:
                    record = await asyncio.wait_for(runner.run_one(q, repeat), timeout=360)
                except TimeoutError:
                    record = RunRecord(
                        question_id=q.id,
                        inn=q.inn,
                        type=q.type,
                        model=model,
                        repeat=repeat,
                        question=q.question,
                        duration_s=360,
                        error="ScenarioTimeout: 360s",
                    )
                if record.error and any(
                    marker in record.error
                    for marker in (
                        "401",
                        "403",
                        "insufficient_quota",
                        "insufficient funds",
                        "RESOURCE_EXHAUSTED",
                    )
                ):
                    stop.set()
                if record.error:
                    record.error = record.error.replace(api_key, "[redacted]")
                row = {
                    "record": record.model_dump(mode="json"),
                    "turns": runtime.turns,
                    "calls": usage.calls,
                    "failed_callbacks": usage.failed_callbacks,
                }
            path = args.out / f"{model_slug(model)}--{q.id}--{repeat}.json"
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
            records.append(row)
            summary = summarize(records, args.models, len(jobs), args.repeats)
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(
                f"{len(records)}/{len(jobs)} {model.split('/')[-2:]} {q.id} "
                f"repeat={repeat} pass={record.checks_passed} error={bool(record.error)} "
                f"{record.duration_s}s",
                flush=True,
            )

    await asyncio.gather(worker(), worker())
    print(
        json.dumps(summarize(records, args.models, len(jobs), args.repeats), indent=2), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--models", nargs=2, required=True)
    parser.add_argument("--base-url", default="https://llm.api.cloud.yandex.net/v1")
    parser.add_argument(
        "--suite", choices=["model_probe24", "smoke48", "regression_fix4"], default="model_probe24"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
