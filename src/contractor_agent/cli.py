"""``kontragent`` — консольные команды проекта.

kontragent signals <ИНН> [--json]   сигналы, пробелы и рекомендация по компании
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from contractor_agent.data.loader import load_snapshot
from contractor_agent.signals.engine import compute
from contractor_agent.signals.model import Severity, SignalSet, Verdict

VERDICT_RU = {
    Verdict.OK: "можно работать",
    Verdict.CHECK: "стоит проверить дополнительно",
    Verdict.NOT_RECOMMENDED: "не рекомендуем без дополнительной проверки",
}
SEVERITY_RU = {
    Severity.CRITICAL: "КРИТИЧНО",
    Severity.MODERATE: "ПРОВЕРИТЬ",
    Severity.INFO: "для сведения",
}
from contractor_agent.labels import svetofor_ru, zsk_ru  # noqa: E402


def render(signal_set: SignalSet, name: str, risk_level: str, zsk: str) -> str:
    lines = [
        f"{name} — ИНН {signal_set.inn}, отчёт от {signal_set.report_date:%d.%m.%Y}",
        f"Светофор банка: {svetofor_ru(risk_level)} · ЗСК: {zsk_ru(zsk)}",
        f"Рекомендация: {VERDICT_RU[signal_set.verdict].upper()} "
        f"(баллы {signal_set.score}{', терминальный факт' if signal_set.terminal else ''})",
        "",
    ]
    if signal_set.signals:
        lines.append("Сигналы:")
        for s in signal_set.signals:
            lines.append(f"  [{SEVERITY_RU[s.severity]}] {s.title_ru}")
            lines.append(f"      {s.explanation_ru}")
            lines.append(f"      ← {s.source_path}")
    else:
        lines.append("Сигналов нет.")
    if signal_set.gaps:
        lines.append("")
        lines.append("Не удалось оценить (данных в отчёте нет):")
        for g in signal_set.gaps:
            lines.append(f"  · {g.criterion}: {g.text_ru}")
            if g.ask_ru:
                lines.append(f"      → {g.ask_ru}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kontragent")
    parser.add_argument("--data", type=Path, default=Path("data"))
    sub = parser.add_subparsers(dest="command", required=True)
    signals = sub.add_parser("signals", help="сигналы и рекомендация по ИНН")
    signals.add_argument("inn")
    signals.add_argument("--json", action="store_true", help="вывести SignalSet как JSON")
    ask = sub.add_parser("ask", help="один вопрос агенту")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="вывести Answer как JSON")
    ask.add_argument("--model", default=None, help="модель вместо LLM_MODEL из .env")
    ask.add_argument(
        "--mcp-stdio", action="store_true", help="MCP-сервер подпроцессом, а не в памяти"
    )
    chat = sub.add_parser("chat", help="диалог с агентом")
    chat.add_argument("--thread", default="cli")
    chat.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    if args.command in ("ask", "chat"):
        return asyncio.run(_agent_command(args))

    snapshot = load_snapshot(args.data)
    report = snapshot.get(args.inn)
    if report is None:
        print(f"нет компании с ИНН {args.inn}", file=sys.stderr)
        return 1
    signal_set = compute(report)
    if args.json:
        print(signal_set.model_dump_json(indent=2))
    else:
        print(
            render(
                signal_set,
                report.base_info.short_name,
                report.base_info.risk_level,
                report.zsk_risk_level,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


async def _agent_command(args: argparse.Namespace) -> int:
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)  # запросы к модели — не в вывод CLI
    from contractor_agent.agent.llm import make_llm
    from contractor_agent.agent.runtime import AgentRuntime
    from contractor_agent.settings import Settings

    settings = Settings(data_dir=args.data)
    llm = make_llm(settings, args.model)
    async with AgentRuntime(settings, llm=llm, mcp_stdio=getattr(args, "mcp_stdio", False)) as rt:
        if args.command == "ask":
            answer = await rt.ask(args.question)
            print(answer.model_dump_json(indent=2) if args.json else render_answer(answer))
            return 0
        print(f"Модель: {llm.name}. Пустая строка — выход.")
        while True:
            try:
                question = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            answer = await rt.ask(question, thread_id=args.thread)
            print("\nАгент:", render_answer(answer))
    return 0


def render_answer(answer) -> str:
    lines = [answer.text_md.strip()]
    if answer.card:
        c = answer.card
        lines.append("")
        lines.append(
            f"[карточка] {c.name} · светофор {c.labels.riskLevel} · ЗСК {c.labels.zskRiskLevel} · "
            f"{VERDICT_RU[c.verdict]} · отчёт от {c.report_date:%d.%m.%Y}"
        )
    if answer.citations:
        lines.append("")
        lines.append(
            "Цитаты проверены: "
            + "; ".join(f"{c.claim} ← {c.source_path}" for c in answer.citations)
        )
    if answer.invalid_citations:
        lines.append(
            "Не подтверждены отчётом (не факты): "
            + "; ".join(c.claim for c in answer.invalid_citations)
        )
    return "\n".join(lines)
