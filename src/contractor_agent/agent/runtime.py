"""Сборка агента целиком: источник отчётов → MCP-сервер → инструменты → граф.

``AgentRuntime`` держит MCP-клиент открытым на время сессии: по умолчанию сервер
поднимается в памяти (тот же код, что и по stdio), для демонстрации транспорта —
подпроцессом ``kontragent-mcp``. Каждый прогон пишет строку в ``runs/`` — это
минимальная трасса: что спросили, какие инструменты вызвали, что ответили.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

from contractor_agent.agent.graph import build_graph, initial_state
from contractor_agent.agent.llm import LLM, make_llm
from contractor_agent.agent.schema import Answer
from contractor_agent.agent.tools import load_tools
from contractor_agent.data.loader import ReportSource
from contractor_agent.mcp_server.server import build_server
from contractor_agent.settings import Settings, make_source


class AgentRuntime:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        source: ReportSource | None = None,
        llm: LLM | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        mcp_stdio: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.source = source or make_source(self.settings)
        self.llm = llm or make_llm(self.settings)
        self.checkpointer = checkpointer
        self.mcp_stdio = mcp_stdio
        self.client: Client | None = None
        self.graph = None
        self.tools: list[Any] = []

    async def __aenter__(self) -> AgentRuntime:
        if self.mcp_stdio:
            params = StdioServerParameters(
                command="uv",
                args=["run", "kontragent-mcp", "--source", self.settings.report_source],
            )
            self.client = Client(params)
        else:
            self.client = Client(build_server(self.source))
        await self.client.__aenter__()
        self.tools = await load_tools(self.client)
        self.graph = build_graph(self.llm, self.tools, self.source, self.checkpointer)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self.client is not None:
            await self.client.__aexit__(*exc)

    async def ask(
        self, question: str, thread_id: str = "cli", *, history: Sequence[BaseMessage] = ()
    ) -> Answer:
        assert self.graph is not None, "используй `async with AgentRuntime(...)`"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.recursion_limit,
        }
        result = await self.graph.ainvoke(initial_state(question, history), config=config)
        answer: Answer = result["answer"]
        self.record_trace(thread_id, question, result, answer)
        return answer

    def record_trace(
        self, thread_id: str, question: str, result: dict[str, Any], answer: Answer
    ) -> None:
        """Строка JSONL в ``runs/`` — минимальная трасса прогона."""
        runs = self.settings.runs_dir
        try:
            runs.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "thread_id": thread_id,
                "model": self.llm.name,
                "question": question,
                "kind": answer.kind,
                "verdict": answer.card.verdict.value if answer.card else None,
                "tool_calls": [t.model_dump() for t in result.get("trace") or []],
                "citations": len(answer.citations),
                "invalid_citations": [c.model_dump() for c in answer.invalid_citations],
                "citation_retry": result.get("citation_retry"),
            }
            with (runs / f"{datetime.now(UTC):%Y-%m-%d}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
