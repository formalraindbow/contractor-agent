"""``kontragent-api`` — запуск FastAPI-сервиса через uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from contractor_agent.settings import Settings


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="kontragent-api")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    uvicorn.run("contractor_agent.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0
