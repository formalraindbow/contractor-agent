"""Мини-CLI слоя данных: ``python -m contractor_agent.data <команда>``.

stats                 сколько загрузилось, заполненность секций по источникам
show <inn> [секция]   нормализованный отчёт (или одна секция) в JSON
search <запрос>       поиск по названию / ИНН / ОГРН
build-index [--out]   собрать SQLite-индекс из снапшота
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from contractor_agent.data.index import DEFAULT_INDEX_PATH, build_index
from contractor_agent.data.loader import Source, load_snapshot
from contractor_agent.data.model import SECTIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m contractor_agent.data")
    parser.add_argument("--data", type=Path, default=Path("data"), help="каталог снапшота")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    show = sub.add_parser("show")
    show.add_argument("inn")
    show.add_argument("section", nargs="?")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    build = sub.add_parser("build-index")
    build.add_argument("--out", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args(argv)

    snapshot = load_snapshot(args.data)

    if args.command == "stats":
        by_source = Counter(r.source for r in snapshot)
        print(
            f"компаний: {len(snapshot)} "
            f"(json {by_source[Source.JSON]}, csv {by_source[Source.CSV]})"
        )
        states = Counter(
            (r.source, section, state)
            for r in snapshot
            for section, state in r.report.sections().items()
        )
        print(f"{'секция':22} {'json: нет/пусто/есть':>22} {'csv: нет/пусто/есть':>22}")
        for section in SECTIONS:
            cells = [
                "/".join(str(states[(src, section, st)]) for st in ("absent", "empty", "present"))
                for src in (Source.JSON, Source.CSV)
            ]
            print(f"{section:22} {cells[0]:>22} {cells[1]:>22}")
        return 0

    if args.command == "show":
        report = snapshot.get(args.inn)
        if report is None:
            print(f"нет компании с ИНН {args.inn}", file=sys.stderr)
            return 1
        data = report.model_dump(mode="json")
        if args.section:
            data = data.get(
                args.section, {"available": False, "state": report.section_state(args.section)}
            )
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "search":
        for hit in snapshot.search(args.query, args.limit):
            labels = f"{hit.risk_level}/{hit.zsk_risk_level}"
            print(f"{hit.inn:>13}  {labels:15} {hit.short_name}  [{hit.source}]")
        return 0

    if args.command == "build-index":
        count = build_index(snapshot, args.out)
        print(f"индекс {args.out}: {count} компаний")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
