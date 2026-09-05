from langchain_core.runnables import RunnableLambda

from contractor_agent.agent.llm import LLM
from contractor_agent.agent.schema import Draft


async def test_empty_structured_result_tries_other_format_on_same_model():
    attempts = []

    class EmptyFirstModel:
        def with_structured_output(self, schema, *, method):
            def respond(_):
                attempts.append(method)
                return None if method == "json_schema" else {"kind": "answer", "lines": ["Факт."]}

            return RunnableLambda(respond)

    result = await LLM([EmptyFirstModel()]).structured(Draft).ainvoke([])
    assert isinstance(result, Draft)
    assert result.text_md == "Факт."
    assert attempts == ["json_schema", "function_calling"]
