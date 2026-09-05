import importlib.util
import sqlite3
import stat
from contextlib import closing
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "session_backup", Path(__file__).parents[1] / "deploy" / "backup_sessions.py"
)
backup_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup_module)


def test_backup_includes_committed_wal_without_changing_live_database(tmp_path):
    source = tmp_path / "live.sqlite"
    with closing(sqlite3.connect(source)) as live:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE messages (body TEXT)")
        live.commit()
        live.execute("INSERT INTO messages VALUES ('saved conversation')")
        live.commit()
        live.execute("INSERT INTO messages VALUES ('unfinished transaction')")
        result = backup_module.backup(source, tmp_path / "backups")
        with closing(sqlite3.connect(result)) as restored:
            assert restored.execute("SELECT * FROM messages").fetchall() == [
                ("saved conversation",)
            ]
            assert restored.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert live.execute("SELECT count(*) FROM messages").fetchone() == (2,)
        assert stat.S_IMODE(result.stat().st_mode) == 0o600


def test_missing_source_does_not_create_an_empty_database(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        backup_module.backup(missing, tmp_path / "backups")
    assert not missing.exists()
