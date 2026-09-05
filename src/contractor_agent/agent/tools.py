"""Инструменты MCP как инструменты LangChain — свой тонкий адаптер.

``langchain-mcp-adapters`` не собирается с ``mcp 2.x``, а нужного здесь мало:
взять список инструментов у клиента SDK и обернуть каждый в ``StructuredTool``
с асинхронным вызовом ``client.call_tool``. Схема аргументов берётся из
``input_schema`` сервера, результат отдаётся модели как JSON-строка — это и есть
``ToolMessage``. Клиент может быть в памяти (тесты, CLI) или по stdio / HTTP
(тот же сервер, что для Claude Desktop).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from mcp.client.client import Client


async def load_tools(client: Client) -> list[StructuredTool]:
    """Все инструменты подключённого MCP-клиента как инструменты LangChain."""
    listed = await client.list_tools()
    return [_wrap(client, t.name, t.description or "", t.input_schema) for t in listed.tools]


def _wrap(
    client: Client, name: str, description: str, input_schema: dict[str, Any]
) -> StructuredTool:
    async def call(**kwargs: Any) -> str:
        result = await client.call_tool(name, kwargs)
        if result.structured_content is not None:
            payload: Any = result.structured_content
        else:
            payload = [c.model_dump(mode="json") for c in result.content]
        if result.is_error:
            payload = {"error": payload}
        return json.dumps(payload, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=call,
        name=name,
        description=description,
        args_schema=input_schema,
    )
