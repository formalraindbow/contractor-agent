"""Свой сервис с графом внутри — точка входа за API-gateway банка.

Контракт: ``POST /v1/runs/stream`` → SSE-кадры ``event: <type>\\ndata: <envelope>``
(``token | tool | end | error``, ``interrupt`` зарезервирован); ``GET /v1/threads/{id}/state``
— история сессии; ``GET /v1/health`` — версия контракта. Справочные ручки для
виджетов: ``/companies/search``, ``/report/{inn}/card``, ``/report/{inn}/financials``.
Фронт рисует по ``Answer``/``Card``, не по свободному тексту.
"""

from __future__ import annotations

import base64
import json
import re
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from evals.metrics import compute_metrics
from evals.runner import RunRecord
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from contractor_agent import __version__
from contractor_agent.agent.nodes import build_card
from contractor_agent.agent.presentation import (
    PUBLIC_ANNOTATIONS,
    PUBLIC_LABELS,
    PUBLIC_PHRASES,
    PUBLIC_TERMS,
    public_text,
)
from contractor_agent.agent.prompt import PROMPT_VERSION
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.stream import CONTRACT_VERSION, new_run_id, stream_run
from contractor_agent.api.evidence import evidence
from contractor_agent.data.paths import PathNotFoundError
from contractor_agent.mcp_server.server import DESCRIPTIONS
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.settings import Settings
from contractor_agent.signals.model import LEGACY_VERDICT_RU, VERDICT_RU

STATIC = Path(__file__).parent / "static"


def quality_version(model: str, folder: str, prompt_version: str = "") -> str | None:
    value = prompt_version or model + " " + folder
    match = re.search(r"(?:prompt[-_ ]?|^\s*)v(\d+)", value, re.I)
    return "v" + match[1] if match else None


def quality_name(model: str, folder: str, prompt_version: str = "") -> str:
    """No provider paths or raw model URIs in the presentation table."""
    value = (model + " " + folder).lower()
    family = next(
        (
            name
            for token, name in (
                ("gpt-oss-20b", "GPT-OSS-20B"),
                ("gpt-oss-120b", "GPT-OSS-120B"),
                ("deepseek-v4-flash", "DeepSeek V4 Flash"),
                ("glm-5.3-flash", "GLM 5.3 Flash"),
            )
            if token in value
        ),
        "Другая модель",
    )
    version = quality_version(model, folder, prompt_version) or "версия не указана"
    return f"{family} · {version}"


def quality_role(
    model: str, folder: str, *, active_model: str = "", prompt_version: str = ""
) -> str:
    value = (model + " " + folder).lower()
    if (
        active_model
        and model.split()[0].removesuffix("/latest") == active_model.removesuffix("/latest")
        and quality_version(model, folder, prompt_version) == PROMPT_VERSION
    ):
        return "current"
    if "deepseek-v4-flash" in value and "v3" not in value:
        return "control"
    if "gpt-oss-20b" in value:
        return "first" if quality_version(model, folder, prompt_version) is None else "other"
    return "other"


class RunInput(BaseModel):
    """Тело ``POST /v1/runs/stream``: сессия, прогон и вход — сообщения или ИНН с вопросом."""

    thread_id: str = Field(min_length=1, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)
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


def create_app(
    runtime_factory: Callable[[], AgentRuntime] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory() if runtime_factory else AgentRuntime()
        async with runtime:
            app.state.runtime = runtime
            app.state.tools = Tools(runtime.source)
            yield

    app = FastAPI(title="kontragent-agent", version=__version__, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # без кэша: страницу правим по ходу демо, у коллег не должно остаться старой версии
        return HTMLResponse(
            (STATIC / "index.html")
            .read_text(encoding="utf-8")
            .replace(
                "__PUBLIC_TEXT_RULES__",
                json.dumps(
                    {
                        "terms": PUBLIC_TERMS,
                        "labels": PUBLIC_LABELS,
                        "phrases": PUBLIC_PHRASES,
                        "annotations": PUBLIC_ANNOTATIONS,
                        "verdicts": VERDICT_RU,
                        "verdict_phrases": {
                            old: VERDICT_RU[verdict] for old, verdict in LEGACY_VERDICT_RU.items()
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    web_password = settings.web_password
    if web_password:  # публичная ссылка: в данных ИНН физлиц, закрываем паролем
        expected_basic = "Basic " + base64.b64encode(f"alfa:{web_password}".encode()).decode()

        @app.middleware("http")
        async def password_gate(request: Request, call_next):
            """Пускаем по ключу в ссылке (?k=…, дальше cookie) или по паре логин-пароль."""
            key = request.query_params.get("k")
            if key and secrets.compare_digest(key, web_password):
                # Keep the browser's HTTPS origin even when a tunnel forwards plain HTTP.
                target = request.url.remove_query_params("k")
                path = "/" + target.path.lstrip("/")
                location = path + (f"?{target.query}" if target.query else "")
                response = RedirectResponse(location, status_code=303)
                response.set_cookie(
                    "kontragent_key",
                    web_password,
                    httponly=True,
                    secure=request.url.scheme == "https",
                    max_age=86400,
                )
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
        }

    @app.post("/v1/runs/stream")
    async def run_stream(body: RunInput, request: Request) -> EventSourceResponse:
        runtime: AgentRuntime = request.app.state.runtime
        question = body.question()
        if not question.strip() or len(question) > 8000:
            raise HTTPException(400, "Вопрос должен содержать от 1 до 8000 символов.")
        run_id = body.run_id or new_run_id()

        async def frames():
            async for envelope in stream_run(runtime, question, body.thread_id, run_id):
                yield {"event": envelope.type, "data": envelope.model_dump_json()}

        return EventSourceResponse(frames())

    @app.get("/v1/threads/{thread_id}/state")
    async def thread_state(thread_id: str, request: Request) -> dict[str, Any]:
        runtime: AgentRuntime = request.app.state.runtime
        state = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = state.values or {}
        messages = []
        turns = []
        for m in values.get("messages") or []:
            if isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": m.content})
                if not m.additional_kwargs.get("repair"):
                    turns.append({"who": "user", "text": m.content})
            elif isinstance(m, AIMessage):
                if m.additional_kwargs.get("final_answer"):
                    turns.append({"who": "agent", "text": public_text(m.content)})
                messages.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [c["name"] for c in m.tool_calls],
                    }
                )
            elif isinstance(m, ToolMessage):
                messages.append({"role": "tool", "name": m.name, "chars": len(str(m.content))})
        answer = values.get("answer")
        public_answer = (
            answer.model_copy(update={"text_md": public_text(answer.text_md)}).model_dump(
                mode="json"
            )
            if answer
            else None
        )
        if public_answer and turns:
            if turns[-1]["who"] == "agent":
                turns[-1]["answer"] = public_answer
            else:
                turns.append(
                    {"who": "agent", "text": public_answer["text_md"], "answer": public_answer}
                )
        return {
            "thread_id": thread_id,
            "messages": messages,
            "turns": turns,
            "selected_inns": values.get("selected_inns") or [],
            "report_dates": values.get("report_dates") or {},
            "answer": public_answer,
        }

    @app.get("/v1/mcp/info")
    def mcp_info() -> dict[str, Any]:
        """Как подключить наши инструменты к любому агенту по MCP: описания и готовый конфиг."""
        args = ["run", "kontragent-mcp", "--source", "snapshot"]
        return {
            "server": "kontragent",
            "transport": ["stdio", "streamable-http"],
            "setup_note": "Подключение по stdio на машине разработчика. Запускайте клиент "
            "из каталога установленного проекта с доступом к отчётам. "
            "Публичный MCP-адрес не настроен.",
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
        cache = settings.runs_dir / "evals"
        rows = []
        questions, companies = set(), set()
        for folder in sorted(p for p in cache.iterdir() if p.is_dir()) if cache.exists() else []:
            records = []
            for f in sorted(folder.glob("*.json")):
                try:
                    records.append(RunRecord.model_validate_json(f.read_text(encoding="utf-8")))
                except (ValueError, OSError):
                    continue
            if not records:
                continue
            questions.update(r.question_id for r in records)
            companies.update(r.inn for r in records if r.inn)
            m = compute_metrics(records)
            rows.append(
                {
                    "model": quality_name(m.model, folder.name, m.prompt_version),
                    "role": quality_role(
                        m.model,
                        folder.name,
                        active_model=settings.llm_model,
                        prompt_version=m.prompt_version,
                    ),
                    "questions": len({r.question_id for r in records}),
                    "total": m.total,
                    "errors": m.errors,
                    "grounded": m.grounded_share,
                    "refusal": m.refusal_share,
                    "invented": m.invented_share,
                    "bad_citation": m.bad_citation_share,
                    "missed_critical": m.missed_critical_share,
                    "judge": m.judge_mean,
                    "seconds": None,  # old durations include the judge, not only the answer
                }
            )
        return {"rows": rows, "questions": len(questions), "companies": len(companies)}

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

    @app.get("/report/{inn}/summary")
    def report_summary(inn: str, request: Request) -> dict[str, Any]:
        return request.app.state.tools.get_report_summary(inn).model_dump(mode="json")

    @app.get("/report/{inn}/source")
    def report_evidence(
        inn: str,
        request: Request,
        path: str = Query(max_length=300),
        signal: str | None = Query(default=None, max_length=100),
    ) -> dict[str, Any]:
        try:
            return evidence(request.app.state.tools, inn, path, signal)
        except PathNotFoundError as e:
            raise HTTPException(404, "Поле отчёта или расчёт не найден") from e

    return app


app = create_app()
