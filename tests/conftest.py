import json
import os
import uuid
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from contractor_agent.data.csv_tree import load_csv_records
from contractor_agent.data.index import build_index
from contractor_agent.data.loader import Snapshot, load_snapshot
from contractor_agent.data.mongo import unwrap
from contractor_agent.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JSON_PATH = DATA_DIR / "contractors_audit.snapshot.json"
CSV_PATH = DATA_DIR / "contractors_audit.snapshot_C12613591.csv"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Каталог со снапшотом; только чтение."""
    return DATA_DIR


@pytest.fixture(scope="session")
def plain_reports() -> list[tuple[str, dict[str, Any]]]:
    """Все 200 отчётов как плоские деревья: (источник, report)."""
    json_records = unwrap(json.loads(JSON_PATH.read_text("utf-8")))
    csv_records = load_csv_records(CSV_PATH)
    return [("json", r["report"]) for r in json_records] + [
        ("csv", r["report"]) for r in csv_records
    ]


@pytest.fixture(scope="session")
def snapshot() -> Snapshot:
    """Все 200 компаний как ``Snapshot`` (реализация ``ReportSource`` в памяти)."""
    return load_snapshot(DATA_DIR)


@pytest.fixture(scope="session")
def index_path(snapshot: Snapshot, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """SQLite-индекс, собранный из снапшота во временный каталог."""
    path = tmp_path_factory.mktemp("index") / "snapshot.sqlite"
    build_index(snapshot, path)
    return path


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def postgres_settings(tmp_path):
    """A disposable database, never tables in the caller's database."""
    dsn = os.environ.get("TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("Set TEST_POSTGRES_URL to run real PostgreSQL persistence checks")
    database = "contractor_test_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(database))
        )
    try:
        yield Settings(
            _env_file=None,
            session_store="postgres",
            session_database_url=make_conninfo(dsn, dbname=database),
            runs_dir=tmp_path / "first-pod",
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database)))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Все async-тесты — через anyio на asyncio, без пометки в каждом файле."""
    import inspect

    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)
