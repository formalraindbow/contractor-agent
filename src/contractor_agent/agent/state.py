"""Состояние графа — то, что передаётся между узлами и живёт в чекпоинтере.

``messages`` — история диалога с редьюсером ``add_messages`` (новые сообщения
дописываются, не заменяют). Остальные поля перезаписываются узлами, кроме
``trace`` — она накапливается.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState
from pydantic import BaseModel

from contractor_agent.agent.schema import Answer, Draft


class ToolCallTrace(BaseModel):
    """Один вызов инструмента — для события ``tool`` в SSE и для трасс в ``runs/``."""

    name: str
    args: dict[str, Any]
    available: bool | None = None
    reason: str | None = None
    result_chars: int = 0


class AgentState(MessagesState):
    selected_inns: list[str]
    report_dates: dict[str, str]
    trace: Annotated[list[ToolCallTrace], operator.add]
    draft: Draft | None
    answer: Answer | None
    citation_retry: int
    question: str  # вопрос текущего хода — для подсказки вида ответа и проверки формы
    turn_inns: list[str]  # компании из вызовов инструментов текущего хода (для карточек и цитат)
