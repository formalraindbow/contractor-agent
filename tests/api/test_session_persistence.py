import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from tests.agent.fakes import scripted_llm, tool_call
from tests.api.test_api import CARD_SCRIPT, _sse

from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.sessions import initialize_postgres
from contractor_agent.api.app import create_app
from contractor_agent.settings import Settings


@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
def test_conversation_survives_runtime_restart_and_keeps_company(
    snapshot, tmp_path, backend, request
):
    settings = Settings(_env_file=None, runs_dir=tmp_path)
    if backend == "postgres":
        settings = request.getfixturevalue("postgres_settings")
        asyncio.run(initialize_postgres(settings))

    def app(script):
        return create_app(
            lambda script=script: AgentRuntime(settings, source=snapshot, llm=scripted_llm(script))
        )

    thread = "same-browser-chat"
    url = f"/v1/threads/{thread}/state"
    first_question = "Проверь МАКСМАРКЕТ, ИНН 5032257375. Планирую оплату."
    with TestClient(app(CARD_SCRIPT)) as client:
        response = client.post(
            "/v1/runs/stream", json={"thread_id": thread, "input": {"question": first_question}}
        )
        assert _sse(response.iter_lines())[-1][0] == "end"
        before = client.get(url).json()
        assert before["selected_inns"] == ["5032257375"]
        assert before["purpose"]

    # A new app and MCP client, a new LLM, a newly opened database connection.
    if backend == "postgres":
        # A replacement pod has none of the previous pod's local files.
        settings = settings.model_copy(update={"runs_dir": tmp_path / "replacement-pod"})
    llm = scripted_llm(
        [
            tool_call("get_report_summary", "status", inn="5032257375"),
            AIMessage(content="Уточните ИНН, чтобы я получил свежую выписку и подготовил Draft."),
        ]
    )
    restarted = create_app(lambda: AgentRuntime(settings, source=snapshot, llm=llm))
    with TestClient(restarted) as client:
        assert client.get(url).json() == before
        assert client.get("/v1/threads/another-chat/state").json()["messages"] == []
        response = client.post(
            "/v1/runs/stream",
            json={
                "thread_id": thread,
                "input": {"question": "текущий статус компании по свежей выписке ЕГРЮЛ ?"},
            },
        )
        frames = _sse(response.iter_lines())
        assert frames[-1][0] == "end", frames[-1]
        answer = frames[-1][1]["data"]["output"]
        assert "5032257375" in answer["text_md"]
        assert "несостоятельным" in answer["text_md"]
        assert "31.07.2026" in answer["text_md"]
        assert "не могу" in answer["text_md"]
        assert "Draft" not in answer["text_md"]
        assert "уточните" not in answer["text_md"].lower()
        assert answer["invalid_citations"] == []
        state = client.get(url).json()
        assert state["messages"][0]["content"] == first_question
        assert len(state["messages"]) == 4
        assert state["messages"][-1]["output"]["active_inns"] == ["5032257375"]
        assert state["purpose"] == before["purpose"]
        assert state["selected_inns"] == ["5032257375"]
        # Company and visible conversation really reached the model after restart.
        seen = "\n".join(str(m.content) for m in llm.models[0].seen[0])
        assert first_question in seen
        assert "Выбранные ИНН: 5032257375" in seen


def test_memory_mode_is_explicitly_ephemeral(snapshot, tmp_path):
    settings = Settings(_env_file=None, runs_dir=tmp_path, session_store="memory")
    for script in (CARD_SCRIPT, []):
        with TestClient(
            create_app(
                lambda script=script: AgentRuntime(
                    settings, source=snapshot, llm=scripted_llm(script)
                )
            )
        ) as client:
            assert client.get("/v1/threads/t/state").json()["messages"] == []
            if script:
                client.post(
                    "/v1/runs/stream", json={"thread_id": "t", "input": {"inn": "5032257375"}}
                )
    assert not settings.session_database.exists()


def test_unknown_new_company_does_not_reuse_registry_status(snapshot, tmp_path):
    llm = scripted_llm(
        [
            *CARD_SCRIPT,
            tool_call("search_company", "missing", query="НЕСУЩЕСТВУЮЩАЯФИРМА"),
            AIMessage(content="Компания не найдена."),
        ]
    )
    with TestClient(
        create_app(
            lambda: AgentRuntime(
                Settings(_env_file=None, runs_dir=tmp_path), source=snapshot, llm=llm
            )
        )
    ) as client:
        client.post("/v1/runs/stream", json={"thread_id": "t", "input": {"inn": "5032257375"}})
        response = client.post(
            "/v1/runs/stream",
            json={
                "thread_id": "t",
                "input": {"question": "Свежая выписка ЕГРЮЛ НЕСУЩЕСТВУЮЩАЯФИРМА"},
            },
        )
        event = _sse(response.iter_lines())[-1]
        assert event[0] == "end"
        answer = event[1]["data"]["output"]
        assert answer["active_inns"] == []
        assert "МАКСМАРКЕТ" not in answer["text_md"]
        assert answer["report_dates"] == {}
