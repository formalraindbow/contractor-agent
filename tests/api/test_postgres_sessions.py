import asyncio
import os
import subprocess
import sys
from contextlib import AsyncExitStack

import anyio
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.conninfo import conninfo_to_dict
from tests.agent.fakes import scripted_llm
from tests.api.test_api import CARD_SCRIPT, _sse

from contractor_agent.agent.runtime import AgentRuntime
from contractor_agent.agent.sessions import (
    SessionStorage,
    SessionStoreUnavailable,
    ThreadBusy,
    initialize_postgres,
)
from contractor_agent.api.app import create_app

pytestmark = pytest.mark.postgres


async def test_schema_is_initialized_explicitly_and_concurrent_init_is_safe(postgres_settings):
    async with AsyncExitStack() as stack:
        with pytest.raises(SessionStoreUnavailable, match="kontragent-db-init"):
            await SessionStorage(postgres_settings).open(stack)
    await asyncio.gather(
        initialize_postgres(postgres_settings), initialize_postgres(postgres_settings)
    )
    async with AsyncExitStack() as stack:
        store = SessionStorage(postgres_settings)
        await store.open(stack)
        assert await store.ready()


async def test_lock_spans_instances_and_cancel_releases_it(postgres_settings):
    await initialize_postgres(postgres_settings)
    async with AsyncExitStack() as stack:
        first, second = SessionStorage(postgres_settings), SessionStorage(postgres_settings)
        await first.open(stack)
        await second.open(stack)
        with anyio.CancelScope() as scope:
            async with await first.acquire("same-chat"):
                with pytest.raises(ThreadBusy):
                    await second.acquire("same-chat")
                async with await second.acquire("different-chat"):
                    assert await second.ready()
                scope.cancel()
                await anyio.sleep(0)
        async with await second.acquire("same-chat"):
            assert second.active == {"same-chat"}
        assert not first.active and not second.active


def test_api_returns_conflict_across_replicas_and_then_accepts_retry(postgres_settings, snapshot):
    asyncio.run(initialize_postgres(postgres_settings))
    first = create_app(
        lambda: AgentRuntime(postgres_settings, source=snapshot, llm=scripted_llm([]))
    )
    second = create_app(
        lambda: AgentRuntime(postgres_settings, source=snapshot, llm=scripted_llm(CARD_SCRIPT))
    )
    with TestClient(first) as a, TestClient(second) as b:
        lease = a.portal.call(first.state.runtime.sessions.acquire, "same-chat")
        body = {"thread_id": "same-chat", "input": {"question": "Проверь МАКСМАРКЕТ 5032257375"}}
        try:
            assert b.post("/v1/runs/stream", json=body).status_code == 409
        finally:
            a.portal.call(lease.release)
        response = b.post("/v1/runs/stream", json=body)
        assert _sse(response.iter_lines())[-1][0] == "end"
        assert (
            a.get("/v1/threads/same-chat/state").json()
            == b.get("/v1/threads/same-chat/state").json()
        )
        assert a.get("/v1/ready").status_code == 200
        assert a.get("/v1/health").json()["session_store"] == "postgres"


def test_committed_chat_survives_sigkill_and_releases_database_lock(postgres_settings, snapshot):
    asyncio.run(initialize_postgres(postgres_settings))
    # Independent process: kill it with an open runtime and a held conversation lock.
    # No close(), shutdown hook or local files may be needed to recover the conversation.
    script = """
import asyncio
from contractor_agent.settings import Settings
from contractor_agent.agent.runtime import AgentRuntime
from tests.agent.fakes import scripted_llm
from tests.api.test_api import CARD_SCRIPT
async def main():
    async with AgentRuntime(Settings(), llm=scripted_llm(CARD_SCRIPT)) as runtime:
        await runtime.ask("Проверь МАКСМАРКЕТ 5032257375. Планирую оплату.", "killed-pod")
        async with await runtime.sessions.acquire("killed-pod"):
            print("CHECKPOINT_COMMITTED", flush=True)
            await asyncio.Event().wait()
asyncio.run(main())
"""
    env = {
        **os.environ,
        "SESSION_STORE": "postgres",
        "SESSION_DATABASE_URL": postgres_settings.session_database_url.get_secret_value(),
        "RUNS_DIR": str(postgres_settings.runs_dir),
    }
    process = subprocess.Popen(
        [sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    try:
        import select

        assert select.select([process.stdout], [], [], 40)[0], "Worker did not commit its turn"
        assert process.stdout.readline().strip() == b"CHECKPOINT_COMMITTED"
    finally:
        process.kill()
        process.wait(timeout=10)
        process.stdout.close()
    settings = postgres_settings.model_copy(
        update={"runs_dir": postgres_settings.runs_dir.parent / "replacement-pod"}
    )
    app = create_app(lambda: AgentRuntime(settings, source=snapshot, llm=scripted_llm([])))
    with TestClient(app) as client:
        state = client.get("/v1/threads/killed-pod/state").json()
        assert len(state["messages"]) == 2
        assert state["selected_inns"] == ["5032257375"]
        assert state["purpose"]
        assert state["answer"]["card"]["inn"] == "5032257375"
        lease = client.portal.call(app.state.runtime.sessions.acquire, "killed-pod")
        client.portal.call(lease.release)
    assert not settings.session_database.exists()


def test_database_outage_reports_not_ready_and_never_exposes_credentials(
    postgres_settings, snapshot
):
    asyncio.run(initialize_postgres(postgres_settings))
    settings = postgres_settings.model_copy(update={"session_connect_timeout_s": 1})
    app = create_app(lambda: AgentRuntime(settings, source=snapshot, llm=scripted_llm([])))
    dsn = settings.session_database_url.get_secret_value()
    with (
        TestClient(app) as client,
        psycopg.connect(os.environ["TEST_POSTGRES_URL"], autocommit=True) as admin,
    ):
        # Close app connections and reject its next connections until the assertions finish.
        database = conninfo_to_dict(dsn)["dbname"]
        admin.execute(
            psycopg.sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS false").format(
                psycopg.sql.Identifier(database)
            )
        )
        try:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database,),
            )
            assert client.get("/v1/ready").status_code == 503
            assert client.get("/v1/live").status_code == 200
            response = client.post(
                "/v1/runs/stream", json={"thread_id": "outage", "input": {"question": "ИНН?"}}
            )
            assert response.status_code == 503
            assert dsn not in response.text
        finally:
            admin.execute(
                psycopg.sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS true").format(
                    psycopg.sql.Identifier(database)
                )
            )
    assert not settings.session_database.exists()
