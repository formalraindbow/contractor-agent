"""Свой сервис с графом внутри — точка входа за API-gateway банка.

Контракт: ``POST /v1/runs/stream`` → SSE-кадры ``event: <type>\\ndata: <envelope>``
(``token | tool | end | error``, ``interrupt`` зарезервирован); ``GET /v1/threads/{id}/state``
— история сессии; ``GET /v1/health`` — версия контракта. Справочные ручки для
виджетов: ``/companies/search``, ``/report/{inn}/card``, ``/report/{inn}/financials``.
Фронт рисует по ``Answer``/``Card``, не по свободному тексту.
"""

from __future__ import annotations

import base64
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from evals.metrics import compute_metrics
from evals.runner import RunRecord
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from contractor_agent import __version__
from contractor_agent.agent.evidence import evidence_basis, resolve_evidence
from contractor_agent.agent.nodes import build_card
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.stream import CONTRACT_VERSION, new_run_id, stream_run
from contractor_agent.data.paths import PathNotFoundError
from contractor_agent.mcp_server.server import DESCRIPTIONS
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from contractor_agent.signals.model import jsonable

STATIC = Path(__file__).parent / "static"
QUALITY_NAMES = {  # папки кэша прогонов → понятные имена для страницы
    "gpt_b1gir8dkimq5j60i6ajf_deepseek-v4-flash_latest": (
        "deepseek-v4-flash — сильная модель, для сравнения"
    ),
    "gpt_b1gir8dkimq5j60i6ajf_gpt-oss-20b_latest": (
        "gpt-oss-20b — целевая, первая версия подсказок"
    ),
    "gpt_b1gir8dkimq5j60i6ajf_gpt-oss-20b_latest_prompt-v2_": (
        "gpt-oss-20b — целевая, текущая версия"
    ),
}


class RunInput(BaseModel):
    """Тело ``POST /v1/runs/stream``: сессия, прогон и вход — сообщения или ИНН с вопросом."""

    thread_id: str = Field(min_length=1)
    run_id: str | None = None
    input: dict[str, Any]

    def question(self) -> str:
        messages = self.input.get("messages")
        if messages:
            last = messages[-1]
            content = last.get("content") if isinstance(last, dict) else str(last)
            if content:
                return str(content)
        inn = self.input.get("inn")
        question = self.input.get("question")
        if inn and question:
            return f"{question} (ИНН {inn})"
        if inn:
            return f"Проверь компанию с ИНН {inn}: можно ли с ней работать?"
        if question:
            return str(question)
        raise HTTPException(400, "input должен содержать messages или inn/question")


def create_app(runtime_factory: Callable[[], AgentRuntime] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory() if runtime_factory else AgentRuntime()
        async with runtime:
            app.state.runtime = runtime
            app.state.tools = Tools(runtime.source)
            app.state.active_threads = set()
            yield

    app = FastAPI(title="kontragent-agent", version=__version__, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # без кэша: страницу правим по ходу демо, у коллег не должно остаться старой версии
        return HTMLResponse(
            (STATIC / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    web_password = Settings().web_password
    if web_password:  # публичная ссылка: в данных ИНН физлиц, закрываем паролем
        expected_basic = "Basic " + base64.b64encode(f"alfa:{web_password}".encode()).decode()

        @app.middleware("http")
        async def password_gate(request: Request, call_next):
            """Пускаем по ключу в ссылке (?k=…, дальше cookie) или по паре логин-пароль."""
            key = request.query_params.get("k")
            if key and secrets.compare_digest(key, web_password):
                response = RedirectResponse(
                    str(request.url.remove_query_params("k")), status_code=303
                )
                response.set_cookie("kontragent_key", web_password, httponly=True, max_age=86400)
                return response
            cookie = request.cookies.get("kontragent_key", "")
            header = request.headers.get("authorization", "")
            if secrets.compare_digest(cookie, web_password) or secrets.compare_digest(
                header, expected_basic
            ):
                return await call_next(request)
            return Response(
                "Нужен пароль: откройте ссылку, которую вам прислали, целиком.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="kontragent"'},
            )

    @app.get("/v1/health")
    def health(request: Request) -> dict[str, Any]:
        runtime: AgentRuntime = request.app.state.runtime
        return {
            "status": "ok",
            "version": __version__,
            "contract_version": CONTRACT_VERSION,
            "model": runtime.llm.name,
            "tools": [t.name for t in runtime.tools],
            "build": runtime.fingerprint,
            "fallback_models": runtime.settings.fallback_models,
            "provider": runtime.settings.llm_base_url,
        }

    @app.post("/v1/runs/stream")
    async def run_stream(body: RunInput, request: Request) -> EventSourceResponse:
        runtime: AgentRuntime = request.app.state.runtime
        question = body.question()
        run_id = body.run_id or new_run_id()

        if body.thread_id in request.app.state.active_threads:
            raise HTTPException(409, "Предыдущий запрос в этом разговоре ещё выполняется")
        request.app.state.active_threads.add(body.thread_id)

        async def frames():
            try:
                async for envelope in stream_run(runtime, question, body.thread_id, run_id):
                    yield {"event": envelope.type, "data": envelope.model_dump_json()}
            finally:
                request.app.state.active_threads.discard(body.thread_id)

        return EventSourceResponse(frames())

    @app.get("/v1/threads/{thread_id}/state")
    async def thread_state(thread_id: str, request: Request) -> dict[str, Any]:
        runtime: AgentRuntime = request.app.state.runtime
        state = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = state.values or {}
        messages = []
        for m in values.get("messages") or []:
            if isinstance(m, HumanMessage) and not m.additional_kwargs.get("repair"):
                messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage) and m.additional_kwargs.get("visible"):
                messages.append({"role": "assistant", "content": m.content})
        answer = values.get("answer")
        return {
            "thread_id": thread_id,
            "messages": messages,
            "selected_inns": values.get("selected_inns") or [],
            "purpose": values.get("purpose"),
            "pending_clarification": values.get("pending_clarification"),
            "report_dates": values.get("report_dates") or {},
            "answer": answer.model_dump(mode="json") if answer else None,
        }

    @app.get("/v1/mcp/info")
    def mcp_info() -> dict[str, Any]:
        """Как подключить наши инструменты к любому агенту по MCP: описания и готовый конфиг."""
        project = str(Path(__file__).resolve().parents[3])
        args = ["run", "--directory", project, "kontragent-mcp", "--source", "snapshot"]
        return {
            "server": "kontragent",
            "transport": ["stdio", "streamable-http"],
            "command": "uv " + " ".join(args),
            "tools": [{"name": n, "description": d} for n, d in DESCRIPTIONS.items()],
            "envelope": {
                "available": "есть ли данные в отчёте",
                "data": "сам ответ",
                "source_paths": "адреса полей отчёта под каждым фактом",
                "report_date": "дата отчёта, у каждой компании своя",
                "note": "оговорка, если раздел пуст или данные обрезаны",
            },
            "client_config": {"mcpServers": {"kontragent": {"command": "uv", "args": args}}},
        }

    @app.get("/v1/quality")
    def quality() -> dict[str, Any]:
        """Как агента проверяли: эталон, доли и оценка судьи по каждому прогону."""
        cache = Settings().runs_dir / "evals"
        rows = []
        folders = sorted({p.parent for p in cache.rglob("*.json")}) if cache.exists() else []
        for folder in folders:
            records = [
                RunRecord.model_validate_json(f.read_text(encoding="utf-8"))
                for f in sorted(folder.glob("*.json"))
            ]
            if not records:
                continue
            m = compute_metrics(records)
            rows.append(
                {
                    "model": QUALITY_NAMES.get(folder.name, m.model),
                    "build": records[0].manifest,
                    "errors": m.errors,
                    "total": m.total,
                    "grounded": m.grounded_share,
                    "refusal": m.refusal_share,
                    "invented": m.invented_share,
                    "bad_citation": m.bad_citation_share,
                    "missed_critical": m.missed_critical_share,
                    "judge": m.judge_mean,
                    "seconds": m.mean_duration_s,
                }
            )
        return {"rows": rows, "questions": 50, "companies": 12}

    @app.get("/companies/search")
    def companies_search(
        request: Request, q: str = Query(min_length=1), limit: int = 5
    ) -> dict[str, Any]:
        tools: Tools = request.app.state.tools
        return tools.search_company(q, limit).model_dump(mode="json")

    @app.get("/report/{inn}/card")
    def report_card(inn: str, request: Request) -> dict[str, Any]:
        tools: Tools = request.app.state.tools
        card = build_card(tools, inn)
        if card is None:
            raise HTTPException(404, f"нет компании с ИНН {inn}")
        return card.model_dump(mode="json")

    @app.get("/report/{inn}/financials")
    def report_financials(inn: str, request: Request) -> dict[str, Any]:
        tools: Tools = request.app.state.tools
        return tools.get_financials(inn).model_dump(mode="json")

    @app.get("/report/{inn}/source")
    def report_source(inn: str, request: Request, path: str = Query(max_length=300)):
        source = request.app.state.runtime.source
        try:
            value = resolve_evidence(source, inn, path)
        except PathNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True)
        elif isinstance(value, list):
            value = [
                v.model_dump(mode="json", by_alias=True) if hasattr(v, "model_dump") else v
                for v in value
            ]
        from contractor_agent.mcp_server.envelope import truncate_deep

        return {
            "inn": inn,
            "path": path,
            "value": truncate_deep(jsonable(value)),
            "basis": truncate_deep(evidence_basis(source, inn, path)),
            "report_date": source.get(inn).report_date.isoformat(),
            "kind": "calculation" if path.startswith("computed.") else "report",
        }

    return app


app = create_app()
