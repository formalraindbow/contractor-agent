"""Local evaluation transport for the same AgentRuntime; never a public API.

Run with PYTHONPATH=src python -m evals.framework_bridge. Exposes only real
executed tool messages for the current turn, plus the user-visible final answer.
Intermediate drafts/repair instructions are not user-visible agent responses.
"""

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.settings import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings(session_store="memory", runs_dir="runs/framework-bridge")
    async with AgentRuntime(settings) as runtime:
        app.state.runtime = runtime
        yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    user_message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=160)


def visible_tool_trajectory(messages: list[Any]) -> list[dict[str, Any]]:
    start = next(
        (
            i + 1
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage)
            and not messages[i].additional_kwargs.get("repair")
        ),
        0,
    )
    trajectory = []
    for message in messages[start:]:
        if isinstance(message, AIMessage) and message.tool_calls:
            trajectory.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"]},
                        }
                        for c in message.tool_calls
                    ],
                }
            )
        elif isinstance(message, ToolMessage):
            trajectory.append(
                {
                    "role": "tool",
                    "name": message.name,
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
    return trajectory


@app.post("/evaluate")
async def evaluate(body: Query, request: Request):
    runtime = request.app.state.runtime
    answer = await runtime.ask(body.user_message, thread_id=body.thread_id)
    state = await runtime.graph.aget_state({"configurable": {"thread_id": body.thread_id}})
    return {
        "result": answer.text_md,
        "trajectory": visible_tool_trajectory(state.values.get("messages", [])),
        "session_id": body.thread_id,
        "answer": answer.model_dump(),
    }


@app.get("/health")
async def health(request: Request):
    return {"model": request.app.state.runtime.llm.name, "transport": "evaluation-only"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=18102)
