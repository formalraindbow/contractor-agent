"""Два изолированных checkout на одинаковом Yandex endpoint/model/parameters.

uv run python scripts/serve_comparison.py start --baseline ../contractor-agent-baseline-6c8a10a
uv run python scripts/serve_comparison.py stop
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from contextvars import ContextVar
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "comparison-servers"


def fingerprint(checkout):
    h = hashlib.sha256()
    for p in sorted((checkout / "src").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".html"):
            h.update(str(p.relative_to(checkout)).encode())
            h.update(p.read_bytes())
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip(),
        "source_sha256": h.hexdigest(),
    }


def worker(args):
    checkout = Path(args.checkout).resolve()
    sys.path[:0] = [str(checkout / "src"), str(checkout)]
    from dotenv import dotenv_values

    config = dotenv_values(ROOT / ".env")
    # Do not load the original checkout's .env; both processes share this exact profile.
    os.chdir(checkout)
    os.environ["WEB_PASSWORD"] = ""
    import uvicorn
    from langchain_openai import ChatOpenAI

    import contractor_agent.api.app as api
    from contractor_agent.agent.llm import LLM
    from contractor_agent.agent.runtime import AgentRuntime
    from contractor_agent.settings import Settings

    spec = importlib.util.spec_from_file_location(
        "comparison_metrics", ROOT / "src/contractor_agent/agent/telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = ContextVar("comparison_metrics", default=None)
    model = config["LLM_MODEL"]
    parameters = {
        "model": model,
        "base_url": config["LLM_BASE_URL"],
        "temperature": 0,
        "reasoning_effort": config.get("LLM_REASONING_EFFORT") or None,
        "max_tokens": 4096,
        "timeout": 60,
        "max_retries": 1,
    }
    manifest = {
        "label": args.label,
        **fingerprint(checkout),
        "parameters": parameters,
        "fallback_models": [],
        "checkout": str(checkout),
    }
    settings = Settings(
        _env_file=None,
        data_dir=checkout / "data",
        runs_dir=STATE / args.label,
        llm_model=model,
        llm_base_url=parameters["base_url"],
        llm_fallback_models="",
        yandex_api_key=config["YANDEX_API_KEY"],
        llm_reasoning_effort=parameters["reasoning_effort"],
        llm_max_tokens=4096,
        run_timeout_s=180,
        web_password=None,
    )

    class ObservedGraph:
        """Observability-only adapter; no changes to prompts, tools or generated answers."""

        def __init__(self, graph):
            self.graph = graph

        def __getattr__(self, name):
            return getattr(self.graph, name)

        async def astream(self, *args, config=None, **kwargs):
            cfg = dict(config or {})
            cfg["callbacks"] = [*(cfg.get("callbacks") or []), current.get()]
            async for item in self.graph.astream(*args, config=cfg, **kwargs):
                yield item

    class Runtime(AgentRuntime):
        async def __aenter__(self):
            await super().__aenter__()
            self.graph = ObservedGraph(self.graph)
            return self

    original_stream = api.stream_run

    async def stream(runtime, *args, **kwargs):
        import asyncio

        metrics = module.RunMetrics()
        token = current.set(metrics)
        try:
            async with asyncio.timeout(180):
                async for event in original_stream(runtime, *args, **kwargs):
                    if event.type == "end":
                        event.data["output"]["runtime"] = {**metrics.result(), "build": manifest}
                        event.data["usage"] = metrics.result()["usage"]
                    yield event
        finally:
            current.reset(token)

    api.stream_run = stream
    app = api.create_app(
        lambda: Runtime(
            settings, llm=LLM([ChatOpenAI(api_key=config["YANDEX_API_KEY"], **parameters)])
        )
    )

    @app.get("/v1/comparison/config")
    def comparison_config():
        return manifest

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def start(args):
    import httpx

    STATE.mkdir(parents=True, exist_ok=True)
    for port in (8081, 8082):
        import socket

        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"Port {port} is occupied. Stop existing comparison first.")
    entries = []
    for label, checkout, port in [
        ("develop", Path(args.baseline).resolve(), 8081),
        ("review-v4", ROOT, 8082),
    ]:
        with (STATE / f"{label}.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--checkout",
                    str(checkout),
                    "--label",
                    label,
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        entries.append(
            {"label": label, "pid": process.pid, "port": port, "checkout": str(checkout)}
        )
    (STATE / "pids.json").write_text(json.dumps(entries, indent=2))
    for item in entries:
        for _ in range(100):
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{item['port']}/v1/comparison/config", timeout=1
                )
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        else:
            raise SystemExit(f"{item['label']} failed to start; see {STATE}")
        print(f"{item['label']}: http://localhost:{item['port']} (PID {item['pid']})")


def stop():
    p = STATE / "pids.json"
    if not p.exists():
        return
    for item in json.loads(p.read_text()):
        try:
            command = subprocess.check_output(
                ["ps", "-p", str(item["pid"]), "-o", "command="], text=True
            )
            if "serve_comparison.py worker" in command:
                os.kill(item["pid"], signal.SIGTERM)
        except (ProcessLookupError, subprocess.CalledProcessError):
            pass
    p.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("start")
    p.add_argument("--baseline", required=True)
    sub.add_parser("stop")
    p = sub.add_parser("worker")
    p.add_argument("--checkout")
    p.add_argument("--label")
    p.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
    elif args.mode == "start":
        start(args)
    else:
        stop()
