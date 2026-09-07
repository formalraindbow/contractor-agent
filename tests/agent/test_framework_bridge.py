from evals.framework_bridge import visible_tool_trajectory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_bridge_keeps_actual_arguments_results_and_repairs_only_in_current_turn():
    messages = [
        HumanMessage(content="старый вопрос"),
        AIMessage(content="", tool_calls=[{"name": "old", "args": {}, "id": "old"}]),
        ToolMessage(content="old", tool_call_id="old", name="old"),
        HumanMessage(content="новый вопрос"),
        AIMessage(content="внутренний черновик"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_financials", "args": {"inn": "1684017097"}, "id": "new"}],
        ),
        ToolMessage(content='{"profit":null}', tool_call_id="new", name="get_financials"),
        HumanMessage(content="исправь цитату", additional_kwargs={"repair": True}),
        AIMessage(content="исправленный черновик"),
    ]
    result = visible_tool_trajectory(messages)
    assert len(result) == 2
    assert result[0]["tool_calls"][0]["function"] == {
        "name": "get_financials",
        "arguments": {"inn": "1684017097"},
    }
    assert result[1]["content"] == '{"profit":null}'
    assert result[1]["tool_call_id"] == "new"
    assert "черновик" not in str(result)


def test_guard_turn_cannot_leak_previous_tools():
    assert (
        visible_tool_trajectory(
            [
                AIMessage(content="", tool_calls=[{"name": "old", "args": {}, "id": "old"}]),
                HumanMessage(content="погода"),
                AIMessage(content="Не могу ответить"),
            ]
        )
        == []
    )
