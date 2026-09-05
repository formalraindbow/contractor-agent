"""Consistent SQLite backup, including committed data in the WAL file."""

from __future__ import annotations

import argparse
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def backup(source: Path, directory: Path) -> Path:
    source = source.resolve(strict=True)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = directory / f"sessions-{timestamp}.sqlite"
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with (
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(target)) as dest,
        ):
            src.backup(dest)
            if dest.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise RuntimeError("SQLite backup failed integrity check")
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(backup(args.source, args.directory))
