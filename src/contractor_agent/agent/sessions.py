"""Durable graph checkpoints and one writer per conversation, including across replicas."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import aiosqlite
import anyio
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from contractor_agent.agent.graph import ALLOWED_STATE_TYPES
from contractor_agent.settings import Settings


class SessionStoreUnavailable(RuntimeError):
    """Public error deliberately excludes credentials and database connection details."""


class ThreadBusy(RuntimeError):
    pass


def _dsn(settings: Settings) -> str:
    if not settings.session_database_url or not settings.session_database_url.get_secret_value():
        raise SessionStoreUnavailable("Для SESSION_STORE=postgres нужен SESSION_DATABASE_URL")
    return settings.session_database_url.get_secret_value()


async def _connection(settings: Settings) -> AsyncConnection:
    return await AsyncConnection.connect(
        _dsn(settings),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=settings.session_connect_timeout_s,
        # Abandoned session locks must also expire after an ungraceful network loss.
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )


async def initialize_postgres(settings: Settings) -> None:
    """Run once before rolling out API replicas. Concurrent initializers are serialized."""
    try:
        async with await _connection(settings) as conn:
            # Session lock: migrations include CREATE INDEX CONCURRENTLY and cannot
            # run inside a transaction. Closing this dedicated connection releases it.
            await conn.execute("SET statement_timeout = '120s'")
            # Do not wait in pg_advisory_lock(): its statement snapshot can
            # deadlock CREATE INDEX CONCURRENTLY in the migration holding the lock.
            async with asyncio.timeout(120):
                while True:
                    cursor = await conn.execute(
                        "SELECT pg_try_advisory_lock(1472698901, 1) AS locked"
                    )
                    if (await cursor.fetchone())["locked"]:
                        break
                    await asyncio.sleep(0.2)
            await AsyncPostgresSaver(conn).setup()
    except SessionStoreUnavailable:
        raise
    except Exception:
        raise SessionStoreUnavailable("Не удалось подготовить PostgreSQL для сессий") from None


class ThreadLease:
    def __init__(
        self, active: set[str], thread_id: str, connection: AsyncConnection | None
    ) -> None:
        self.active = active
        self.thread_id = thread_id
        self.connection = connection
        self.released = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            # SSE disconnect cancels its task scope. Cleanup still has to close the
            # dedicated connection; it must never go back to a pool with a lock held.
            with anyio.CancelScope(shield=True):
                if self.connection is not None:
                    await self.connection.close()
        finally:
            self.active.discard(self.thread_id)

    async def __aenter__(self) -> ThreadLease:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.release()


class SessionStorage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool: AsyncConnectionPool | None = None
        self.active: set[str] = set()
        self.postgres = False

    async def open(
        self, stack: AsyncExitStack, injected: BaseCheckpointSaver | None = None
    ) -> BaseCheckpointSaver | None:
        if injected is not None or self.settings.session_store == "memory":
            return injected
        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_STATE_TYPES)
        if self.settings.session_store == "sqlite":
            path = self.settings.session_database
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(mode=0o600, exist_ok=True)
            conn = await stack.enter_async_context(aiosqlite.connect(path))
            saver = AsyncSqliteSaver(conn, serde=serde)
            await saver.setup()
            return saver

        dsn = _dsn(self.settings)
        try:
            self.pool = AsyncConnectionPool(
                dsn,
                min_size=1,
                max_size=self.settings.session_pool_size,
                timeout=self.settings.session_connect_timeout_s,
                open=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "connect_timeout": self.settings.session_connect_timeout_s,
                },
            )
            stack.push_async_callback(self.pool.close)
            await self.pool.open(wait=True, timeout=self.settings.session_connect_timeout_s)
            async with asyncio.timeout(self.settings.session_connect_timeout_s):
                async with self.pool.connection() as conn:
                    cursor = await conn.execute("SELECT max(v) AS v FROM checkpoint_migrations")
                    version = (await cursor.fetchone())["v"]
            if version != len(AsyncPostgresSaver.MIGRATIONS) - 1:
                raise SessionStoreUnavailable("Обновите схему сессий: kontragent-db-init")
        except SessionStoreUnavailable:
            raise
        except Exception:
            raise SessionStoreUnavailable(
                "PostgreSQL сессий недоступен или не подготовлен; выполните kontragent-db-init"
            ) from None
        self.postgres = True
        return AsyncPostgresSaver(self.pool, serde=serde)

    async def ready(self) -> bool:
        if self.pool is None:
            return True
        try:
            async with asyncio.timeout(self.settings.session_connect_timeout_s):
                async with self.pool.connection() as conn:
                    await conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    async def acquire(self, thread_id: str) -> ThreadLease:
        if thread_id in self.active:
            raise ThreadBusy("Предыдущий запрос в этом разговоре ещё выполняется")
        self.active.add(thread_id)
        lease = ThreadLease(self.active, thread_id, None)
        try:
            if self.postgres:
                async with asyncio.timeout(self.settings.session_connect_timeout_s):
                    # Separate connection from the checkpoint pool: long LLM calls
                    # must not occupy its slots or deadlock pending checkpoint writes.
                    lease.connection = await _connection(self.settings)
                    cursor = await lease.connection.execute(
                        "SELECT pg_try_advisory_lock(hashtextextended(%s, 1472698902)) AS locked",
                        (thread_id,),
                    )
                    if not (await cursor.fetchone())["locked"]:
                        raise ThreadBusy("Предыдущий запрос в этом разговоре ещё выполняется")
            return lease
        except BaseException as error:
            await lease.release()
            if isinstance(error, Exception) and not isinstance(error, ThreadBusy):
                raise SessionStoreUnavailable(
                    "Хранилище чатов временно недоступно. Повторите запрос позже."
                ) from None
            raise


def main() -> int:
    """Explicit schema initialization; never prints the DSN."""
    try:
        asyncio.run(initialize_postgres(Settings()))
    except SessionStoreUnavailable as error:
        print(str(error))
        return 1
    print("Схема PostgreSQL для сессий готова")
    return 0
