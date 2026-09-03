"""SQLite-индекс снапшота: поиск компаний и хранилище нормализованных отчётов.

Зачем, если 200 записей помещаются в память: MCP-сервер — отдельный процесс,
и ему нужен один файл, а не два формата и нормализатор; поиск по названию, ИНН,
ОГРН — запрос, а не перебор; а в банке на этом месте будет их база.

Две таблицы: ``companies`` — поля для поиска и карточки списка плюс состояние
секций; ``reports`` — весь нормализованный отчёт в JSON (``Decimal`` и даты
сериализованы pydantic, обратно — той же моделью). Сборка идемпотентна: пишем
во временный файл и подменяем атомарно.

Кириллица: ``LOWER``/``LIKE`` в SQLite работают только для ASCII, поэтому
нормализованное имя считается в Python (``loader.normalize_name``) и хранится
колонкой ``name_norm``; ранжирование — тоже в Python по найденным кандидатам.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from contractor_agent.data.loader import (
    NO_MATCH,
    CompanyHit,
    Snapshot,
    Source,
    normalize_name,
    query_stems,
    rank,
)
from contractor_agent.data.model import Report, SectionState

DEFAULT_INDEX_PATH = Path(".cache/snapshot.sqlite")

SCHEMA = """
CREATE TABLE companies (
    inn            TEXT PRIMARY KEY,
    ogrn           TEXT NOT NULL,
    short_name     TEXT NOT NULL,
    full_name      TEXT,
    name_norm      TEXT NOT NULL,
    full_name_norm TEXT NOT NULL,
    address        TEXT,
    okved_main     TEXT,
    risk_level     TEXT NOT NULL,
    zsk_risk_level TEXT NOT NULL,
    report_date    TEXT NOT NULL,
    source         TEXT NOT NULL,
    sections_json  TEXT NOT NULL
);
CREATE INDEX companies_ogrn ON companies (ogrn);
CREATE INDEX companies_name_norm ON companies (name_norm);
CREATE TABLE reports (
    inn         TEXT PRIMARY KEY REFERENCES companies (inn),
    report_json TEXT NOT NULL
);
"""


def build_index(snapshot: Snapshot, path: Path = DEFAULT_INDEX_PATH) -> int:
    """Собрать индекс заново из снапшота; вернуть число компаний."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as conn:
        conn.executescript(SCHEMA)
        for record in snapshot:
            report, info = record.report, record.report.base_info
            okved = (
                report.kinds_of_activity_info
                and report.kinds_of_activity_info.main_kind_of_activity
            )
            conn.execute(
                "INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.inn,
                    info.ogrn,
                    info.short_name,
                    info.full_name,
                    normalize_name(info.short_name),
                    normalize_name(info.full_name or ""),
                    info.address,
                    okved.code if okved else None,
                    info.risk_level,
                    report.zsk_risk_level,
                    report.report_date.isoformat(),
                    record.source.value,
                    json.dumps(report.sections()),
                ),
            )
            conn.execute("INSERT INTO reports VALUES (?, ?)", (info.inn, report.model_dump_json()))
        conn.commit()
        (count,) = conn.execute("SELECT count(*) FROM companies").fetchone()
    os.replace(tmp, path)
    return count


def _fetch_like(conn: sqlite3.Connection, pattern: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM companies WHERE name_norm LIKE ? ESCAPE '\\' "
        "OR full_name_norm LIKE ? ESCAPE '\\'",
        (pattern, pattern),
    ).fetchall()


def _like_pattern(query_norm: str) -> str:
    escaped = query_norm.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class SqliteSource:
    """``ReportSource`` поверх собранного индекса."""

    def __init__(self, path: Path = DEFAULT_INDEX_PATH) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"индекс не собран: {path}")
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def __len__(self) -> int:
        with closing(self._connect()) as conn:
            (count,) = conn.execute("SELECT count(*) FROM companies").fetchone()
        return count

    def get(self, inn: str) -> Report | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT report_json FROM reports WHERE inn = ?", (inn,)).fetchone()
        return Report.model_validate_json(row["report_json"]) if row else None

    def sections(self, inn: str) -> dict[str, SectionState] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT sections_json FROM companies WHERE inn = ?", (inn,)
            ).fetchone()
        return json.loads(row["sections_json"]) if row else None

    def search(self, query: str, limit: int = 5) -> list[CompanyHit]:
        query = query.strip()
        if not query:
            return []
        with closing(self._connect()) as conn:
            if query.isdigit():
                rows = conn.execute(
                    "SELECT * FROM companies WHERE inn = ? OR ogrn = ? LIMIT ?",
                    (query, query, limit),
                ).fetchall()
                return [_hit(row) for row in rows]
            query_norm = normalize_name(query)
            rows = _fetch_like(conn, _like_pattern(query_norm))
            if not rows:  # склонение: «янполова» → по основе «янпол»
                stems = query_stems(query_norm)
                if stems:
                    rows = _fetch_like(conn, _like_pattern(max(stems, key=len)))
        scored = [
            (min(rank(row["name_norm"], query_norm), rank(row["full_name_norm"], query_norm)), row)
            for row in rows
        ]
        scored = [(score, row) for score, row in scored if score < NO_MATCH]
        scored.sort(key=lambda item: (item[0], item[1]["short_name"]))
        return [_hit(row) for _, row in scored[:limit]]


def _hit(row: sqlite3.Row) -> CompanyHit:
    return CompanyHit(
        inn=row["inn"],
        ogrn=row["ogrn"],
        short_name=row["short_name"],
        address=row["address"],
        risk_level=row["risk_level"],
        zsk_risk_level=row["zsk_risk_level"],
        report_date=date.fromisoformat(row["report_date"]),
        source=Source(row["source"]),
    )
