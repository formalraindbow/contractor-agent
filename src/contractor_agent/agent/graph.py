"""Граф LangGraph, собранный явно из ``StateGraph``.

          ┌──────────┐   tool_calls   ┌─────────┐
 START ─▶ │  agent   │ ─────────────▶ │  tools  │ ──┐
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

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from contractor_agent.agent.llm import LLM
from contractor_agent.agent.nodes import make_nodes, route_after_agent, route_after_validate
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
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "agent")
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


def memory_saver() -> InMemorySaver:
    """Чекпоинтер в памяти, которому явно разрешены наши pydantic-типы в состоянии."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES))


def initial_state(question: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=question)],
        "selected_inns": [],
        "report_dates": {},
        "trace": [],
        "draft": None,
        "answer": None,
        "citation_retry": 0,
    }
