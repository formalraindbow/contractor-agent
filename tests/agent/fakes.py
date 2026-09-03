"""Подставная модель для тестов графа: отдаёт заранее заданные ответы по очереди."""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from contractor_agent.agent.llm import LLM


class ScriptedChatModel(BaseChatModel):
    """``responses`` — очередь AIMessage (для узла agent) и объектов Draft (для finalize)."""

    responses: list[Any]
    seen: list[list[BaseMessage]] = []

    def __init__(self, responses: list[Any]) -> None:
        super().__init__(responses=list(responses))
        self._queue = deque(self.responses)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self) -> Any:
        if not self._queue:
            raise AssertionError("сценарий подставной модели закончился")
        return self._queue.popleft()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        item = self._next()
        assert isinstance(item, AIMessage), f"ожидался AIMessage, в сценарии {type(item).__name__}"
        return ChatResult(generations=[ChatGeneration(message=item)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def produce(messages: Any) -> Any:
            self.seen.append(list(messages))
            item = self._next()
            assert not isinstance(item, AIMessage), "ожидался Draft, в сценарии AIMessage"
            return item

        return RunnableLambda(produce)


def scripted_llm(responses: list[Any]) -> LLM:
    return LLM([ScriptedChatModel(responses)])


def tool_call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )
