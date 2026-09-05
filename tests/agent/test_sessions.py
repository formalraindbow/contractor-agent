import asyncio
from contextlib import AsyncExitStack

import anyio
import pytest
from sse_starlette.sse import EventSourceResponse

from contractor_agent.agent.sessions import SessionStorage, SessionStoreUnavailable, ThreadBusy
from contractor_agent.api.app import LeasedEventSourceResponse
from contractor_agent.settings import Settings


async def test_postgres_requires_explicit_dsn_and_does_not_fall_back(tmp_path):
    store = SessionStorage(Settings(_env_file=None, session_store="postgres", runs_dir=tmp_path))
    async with AsyncExitStack() as stack:
        with pytest.raises(SessionStoreUnavailable, match="SESSION_DATABASE_URL"):
            await store.open(stack)
    assert not (tmp_path / "sessions.sqlite3").exists()


async def test_same_chat_excludes_concurrent_writer_and_releases_on_cancel():
    store = SessionStorage(Settings(_env_file=None, session_store="memory"))
    with anyio.CancelScope() as scope:
        async with await store.acquire("chat"):
            with pytest.raises(ThreadBusy):
                await store.acquire("chat")
            scope.cancel()
            await anyio.sleep(0)
    async with await store.acquire("chat"):
        assert store.active == {"chat"}
    assert not store.active


def test_session_database_credentials_are_hidden():
    settings = Settings(
        _env_file=None,
        session_store="postgres",
        session_database_url="postgresql://user:not-for-logs@localhost/sessions",
    )
    assert "not-for-logs" not in repr(settings)


async def test_disconnect_before_first_stream_frame_releases_lease(monkeypatch):
    store = SessionStorage(Settings(_env_file=None, session_store="memory"))

    async def disconnected(*args):
        raise asyncio.CancelledError

    async def frames():
        raise AssertionError("The client disconnected before the generator started")
        yield

    monkeypatch.setattr(EventSourceResponse, "__call__", disconnected)
    response = LeasedEventSourceResponse(frames(), await store.acquire("early-disconnect"))
    with pytest.raises(asyncio.CancelledError):
        await response({}, None, None)
    assert not store.active
