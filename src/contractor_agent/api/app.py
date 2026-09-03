"""Свой сервис с графом внутри — точка входа за API-gateway банка.

Контракт: ``POST /v1/runs/stream`` → SSE-кадры ``event: <type>\\ndata: <envelope>``
(``token | tool | end | error``, ``interrupt`` зарезервирован); ``GET /v1/threads/{id}/state``
— история сессии; ``GET /v1/health`` — версия контракта. Справочные ручки для
виджетов: ``/companies/search``, ``/report/{inn}/card``, ``/report/{inn}/financials``.
Фронт рисует по ``Answer``/``Card``, не по свободному тексту.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from contractor_agent import __version__
from contractor_agent.agent.nodes import build_card
from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.stream import CONTRACT_VERSION, new_run_id, stream_run
from contractor_agent.mcp_server.tools import Tools

STATIC = Path(__file__).parent / "static"


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
            yield

    app = FastAPI(title="kontragent-agent", version=__version__, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

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
        for m in values.get("messages") or []:
            if isinstance(m, HumanMessage):
                messages.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
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
        return {
            "thread_id": thread_id,
            "messages": messages,
            "selected_inns": values.get("selected_inns") or [],
            "report_dates": values.get("report_dates") or {},
            "answer": answer.model_dump(mode="json") if answer else None,
        }

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

    return app


app = create_app()
