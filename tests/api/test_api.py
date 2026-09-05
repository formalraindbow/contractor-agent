# ruff: noqa: E501
import json
from collections.abc import Iterable

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from tests.agent.fakes import scripted_llm, tool_call

from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.schema import Draft
from contractor_agent.api.app import create_app
from contractor_agent.data.loader import Snapshot
from contractor_agent.settings import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        runs_dir=tmp_path / "runs",
        openrouter_api_key="x",
        session_db_path=tmp_path / "sessions.sqlite",  # свои диалоги у каждого теста
        web_password=None,  # пароль стенда не мешает тестам
    )


def _runtime(snapshot: Snapshot, tmp_path, responses: list) -> AgentRuntime:
    return AgentRuntime(_settings(tmp_path), source=snapshot, llm=scripted_llm(responses))


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_link_login_preserves_browser_origin_and_chat(snapshot, tmp_path, scheme):
    settings = _settings(tmp_path).model_copy(update={"web_password": "test-stand-key"})
    app = create_app(lambda: _runtime(snapshot, tmp_path, []), settings=settings)
    # HTTP also models TLS termination at a proxy that omits X-Forwarded-Proto.
    with TestClient(app, base_url=f"{scheme}://stand.example") as client:
        assert client.get("/").status_code == 401
        assert client.get("/?k=wrong", follow_redirects=False).status_code == 401
        response = client.get("/?k=test-stand-key&chat=existing-chat", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/?chat=existing-chat"
        assert "HttpOnly" in response.headers["set-cookie"]
        assert ("Secure" in response.headers["set-cookie"]) == (scheme == "https")
        page = client.get(response.headers["location"])
        assert page.status_code == 200
        assert str(page.url) == f"{scheme}://stand.example/?chat=existing-chat"
        assert "Проверка контрагента" in page.text
        response = client.get("/?k=test-stand-key", follow_redirects=False)
        assert response.headers["location"] == "/"


def _sse(lines: Iterable[str]) -> list[tuple[str, dict]]:
    frames, event, data = [], "", ""
    for line in list(lines) + [""]:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data += line[5:].strip()
        elif line == "" and data:
            frames.append((event, json.loads(data)))
            event, data = "", ""
    return frames


CARD_SCRIPT = [
    tool_call("get_report_summary", "c1", inn="5032257375"),
    tool_call("get_risk_signals", "c2", inn="5032257375"),
    AIMessage(content="Фактов достаточно."),
    Draft(
        kind="card",
        lines=[
            "Компания признана банкротом [report.status.reasonName].",
            "",
            "Есть существенные риски. Отчёт от 31.07.2026.",
        ],
        citations=[],
    ),
]


def test_stream_contract_and_reference_endpoints(snapshot: Snapshot, tmp_path) -> None:
    app = create_app(
        lambda: _runtime(snapshot, tmp_path, CARD_SCRIPT), settings=_settings(tmp_path)
    )
    with TestClient(app) as client:
        health = client.get("/v1/health").json()
        assert (
            health["status"] == "ok"
            and health["contract_version"] == "1.0"
            and len(health["tools"]) == 8
        )

        body = {
            "thread_id": "t1",
            "run_id": "r1",
            "input": {"messages": [{"role": "user", "content": "Проверь МАКСМАРКЕТ 5032257375"}]},
        }
        with client.stream("POST", "/v1/runs/stream", json=body) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            frames = _sse(response.iter_lines())
        types = [t for t, _ in frames]
        assert types == ["tool", "tool", "token", "end"]
        envelopes = [e for _, e in frames]
        assert [e["seq"] for e in envelopes] == [1, 2, 3, 4]
        assert {e["run_id"] for e in envelopes} == {"r1"} and {
            e["thread_id"] for e in envelopes
        } == {"t1"}
        assert all(e["contract_version"] == "1.0" for e in envelopes)
        assert (
            envelopes[0]["data"]["name"] == "get_report_summary"
            and envelopes[0]["data"]["available"] is True
        )
        assert envelopes[2]["data"]["text"] == "Фактов достаточно."
        end = envelopes[3]["data"]
        assert (
            end["output"]["kind"] == "card"
            and end["output"]["card"]["verdict"] == "not_recommended"
        )
        assert end["output"]["card"]["labels"] == {
            "riskLevel": "зелёный",
            "zskRiskLevel": "зелёный",
        }
        assert end["output"]["citations"][0]["source_path"] == "report.status.reasonName"
        assert set(end["usage"]) == {"input_tokens", "output_tokens", "total_tokens"}

        state = client.get("/v1/threads/t1/state").json()
        assert state["selected_inns"] == ["5032257375"] and state["report_dates"] == {
            "5032257375": "2026-07-31"
        }
        assert [m["role"] for m in state["messages"]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
            "assistant",
        ]
        assert state["answer"]["card"]["inn"] == "5032257375"
        assert [t["who"] for t in state["turns"]] == ["user", "agent"]
        assert state["turns"][-1]["answer"] == state["answer"]
        assert "Фактов достаточно" not in json.dumps(state["turns"], ensure_ascii=False)

        assert (
            client.get("/companies/search", params={"q": "монлид"}).json()["data"]["items"][0][
                "inn"
            ]
            == "5029069967"
        )
        card = client.get("/report/5032257375/card").json()
        assert (
            card["verdict"] == "not_recommended" and card["attention"][0]["severity"] == "critical"
        )
        assert client.get("/report/0000000000/card").status_code == 404
        assert client.get("/report/772377037026/financials").json()["available"] is False
        assert "Проверка контрагента" in client.get("/").text
        assert (
            client.post("/v1/runs/stream", json={"thread_id": "t1", "input": {}}).status_code == 400
        )


def test_inn_input_and_error_event(snapshot: Snapshot, tmp_path) -> None:
    app = create_app(
        lambda: _runtime(snapshot, tmp_path, []), settings=_settings(tmp_path)
    )  # сценарий пуст → модель падает
    with TestClient(app) as client:
        body = {"thread_id": "t2", "input": {"inn": "1684017097"}}
        with client.stream("POST", "/v1/runs/stream", json=body) as response:
            frames = _sse(response.iter_lines())
        assert [t for t, _ in frames] == ["error"]
        assert "сценарий" in frames[0][1]["data"]["message"]


def test_public_history_restores_after_restart_without_model_drafts(snapshot, tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_STORE", "sqlite")
    script = [
        tool_call("get_report_summary", "head", inn="5032257375"),
        AIMessage(content="Руководитель — Ирина Владимировна Москвина."),
    ]
    app = create_app(lambda: _runtime(snapshot, tmp_path, script), settings=_settings(tmp_path))
    with TestClient(app) as client:
        body = {"thread_id": "restored", "input": {"question": "Кто руководитель МАКСМАРКЕТ?"}}
        frames = _sse(client.post("/v1/runs/stream", json=body).iter_lines())
        assert frames[-1][0] == "end"

    restarted = create_app(lambda: _runtime(snapshot, tmp_path, []), settings=_settings(tmp_path))
    with TestClient(restarted) as client:
        state = client.get("/v1/threads/restored/state").json()
        assert [t["who"] for t in state["turns"]] == ["user", "agent"]
        assert "Витальевна" in state["turns"][1]["text"]
        assert "Владимировна" not in json.dumps(state["turns"], ensure_ascii=False)
        assert state["selected_inns"] == ["5032257375"]
        assert client.get("/v1/threads/missing/state").json()["turns"] == []


def test_mcp_info_lists_tools_and_config(snapshot: Snapshot, tmp_path) -> None:
    """Вкладка «Для агентов банка» берёт отсюда описания функций и готовый конфиг."""
    app = create_app(lambda: _runtime(snapshot, tmp_path, []), settings=_settings(tmp_path))
    with TestClient(app) as client:
        info = client.get("/v1/mcp/info").json()
    names = [t["name"] for t in info["tools"]]
    assert "get_risk_signals" in names and len(names) == 8
    assert all(t["description"] for t in info["tools"])
    assert info["client_config"]["mcpServers"]["kontragent"]["command"] == "uv"
    assert "kontragent-mcp" in info["command"]


def test_report_evidence_and_portable_mcp(snapshot, tmp_path):
    settings = _settings(tmp_path)
    app = create_app(lambda: _runtime(snapshot, tmp_path, []), settings=settings)
    with TestClient(app) as client:
        summary = client.get("/report/5032257375/summary").json()
        assert summary["data"]["head"]["position"] == "КОНКУРСНЫЙ УПРАВЛЯЮЩИЙ"
        response = client.get("/report/5032257375/source", params={"path": "report.baseInfo.staff"})
        assert response.status_code == 200 and response.json()["value"] is None
        response = client.get("/report/5032257375/source", params={"path": "../../.env"})
        assert response.status_code == 404
        response = client.get("/report/0000000000/source", params={"path": "report.baseInfo.staff"})
        assert response.status_code == 404
        info = client.get("/v1/mcp/info").json()
        assert len(info["tools"]) == 8
        assert "Публичный MCP-адрес не настроен" in info["setup_note"]
        assert "/Users/" not in json.dumps(info)


def test_quality_names_do_not_expose_provider_paths():
    from contractor_agent.api.app import quality_name, quality_role

    uri = "gpt://test-folder/gpt-oss-20b/latest"
    assert "gpt://" not in quality_name(uri, "prompt-v3")
    assert "GPT-OSS-20B" in quality_name(uri, "prompt-v3")
    assert quality_role(uri, "prompt-v3") == "current"
    assert quality_role(uri, "original") == "first"
