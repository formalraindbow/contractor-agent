"""Граф LangGraph, собранный явно из ``StateGraph``.

          ┌──────────┐   посторонний ввод, ругань, попытка сменить правила
 START ─┬▶ │  guard   │ ──▶ END (короткий ответ без модели)
        │  └──────────┘
        │ ┌──────────┐   tool_calls   ┌─────────┐
        └▶│  agent   │ ─────────────▶ │  tools  │ ──┐
          │ (LLM +   │ ◀───────────── │ToolNode │   │  ToolMessage[] в state
          │  tools)  │                └─────────┘   │
          └────┬─────┘ ◀────────────────────────────┘
               │ нет tool_calls
               ▼
          ┌──────────┐    invalid (≤1 раз)   ┌──────────┐
          │ finalize │ ────────────────────▶ │ validate │ ──▶ END
          └──────────┘                       └──────────┘
                ▲   HumanMessage «цитата не найдена» — назад в agent
                └──────────────────────────────────────┘

Prebuilt ``create_agent`` делает внутри ровно то же для первых двух узлов;
``finalize`` и ``validate`` — наша добавка ради структурированного ответа
и программной проверки цитат.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from contractor_agent.agent.llm import LLM
from contractor_agent.agent.nodes import (
    make_nodes,
    route_after_agent,
    route_after_validate,
    route_input,
)
from contractor_agent.agent.state import AgentState
from contractor_agent.data.loader import ReportSource


def build_graph(
    llm: LLM,
    tools: list[Any],
    source: ReportSource,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    nodes = make_nodes(llm, tools, source)
    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.add_conditional_edges(START, route_input, {"guard": "guard", "agent": "agent"})
    graph.add_edge("guard", END)
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "evidence"}
    )
    graph.add_conditional_edges(
        "tools",
        lambda state: "evidence" if state.get("agent_rounds", 0) >= 3 else "agent",
        {"agent": "agent", "evidence": "evidence"},
    )
    graph.add_edge("evidence", "finalize")
    graph.add_edge("finalize", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"agent": "agent", END: END})
    return graph.compile(checkpointer=checkpointer or memory_saver())


ALLOWED_STATE_TYPES = (
    [
        ("contractor_agent.agent.schema", name)
        for name in ("Answer", "Draft", "Card", "CardLabels", "Attention", "Citation")
    ]
    + [("contractor_agent.agent.state", "ToolCallTrace")]
    + [("contractor_agent.signals.model", name) for name in ("Verdict", "Severity")]
)


def sqlite_saver(path: Path) -> AbstractAsyncContextManager[BaseCheckpointSaver]:
    """Чекпоинтер в файле: диалог не теряется при обновлении страницы и перезапуске сервиса.
    В контуре банка тот же интерфейс даёт ``PostgresSaver`` — меняется только строка подключения."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return _sqlite_saver(path)


@asynccontextmanager
async def _sqlite_saver(path: Path) -> AsyncIterator[BaseCheckpointSaver]:
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        saver.serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)
        yield saver


def memory_saver() -> InMemorySaver:
    """Чекпоинтер в памяти, которому явно разрешены наши pydantic-типы в состоянии."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES))


def initial_state(question: str, history: Sequence[BaseMessage] = ()) -> dict[str, Any]:
    """Вход графа: вопрос плюс, при необходимости, предыдущие реплики (память сессии в эвалах)."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [*history, HumanMessage(content=question)],
        "question": question,
        "turn_inns": [],
        # selected_inns и report_dates не сбрасываем: память сессии — уточняющий вопрос
        # по компании из прошлого хода проверяется по её отчёту
        "trace": [],
        "draft": None,
        "answer": None,
        "citation_retry": 0,
        "agent_rounds": 0,
    }
