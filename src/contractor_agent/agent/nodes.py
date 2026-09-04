# ruff: noqa: E501
"""Узлы графа. Каждый — функция «состояние → частичное обновление состояния».

agent     вход: messages (история + новые ToolMessage)      выход: AIMessage — текст или tool_calls
tools     вход: последний AIMessage с tool_calls            выход: ToolMessage[], selected_inns, report_dates, trace
finalize  вход: вся история                                 выход: draft (текст модели), answer (карточки — кодом)
validate  вход: answer, selected_inns                        выход: answer с помеченными цитатами — или HumanMessage
                                                                    «цитата не найдена» и citation_retry+1 (один круг)
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from contractor_agent.agent.citations import (
    extract_inline_citations,
    is_meta_path,
    validate_citations,
)
from contractor_agent.agent.llm import LLM
from contractor_agent.agent.prompt import FINALIZE_PROMPT, SYSTEM_PROMPT, citation_repair_prompt
from contractor_agent.agent.schema import Answer, Attention, Card, CardLabels, Draft
from contractor_agent.agent.state import AgentState, ToolCallTrace
from contractor_agent.data.loader import ReportSource
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import VERDICT_RU, Severity, Verdict

MAX_CITATION_RETRIES = 1
Node = Callable[[AgentState], Awaitable[dict[str, Any]]]


def make_nodes(llm: LLM, tools: list[Any], source: ReportSource) -> dict[str, Node]:
    with_tools = llm.with_tools(tools)
    structured = llm.structured(Draft)
    tool_node = ToolNode(tools, handle_tool_errors=True)
    tools_layer = Tools(source)

    async def agent(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tools_(state: AgentState) -> dict[str, Any]:
        result = await tool_node.ainvoke(state)
        tool_messages: list[ToolMessage] = result["messages"]
        last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        args_by_id = {c["id"]: (c["name"], c["args"]) for c in last_ai.tool_calls}
        inns = list(state.get("selected_inns") or [])
        dates = dict(state.get("report_dates") or {})
        trace: list[ToolCallTrace] = []
        for msg in tool_messages:
            name, args = args_by_id.get(msg.tool_call_id, (msg.name or "?", {}))
            payload = _parse(msg.content)
            available = payload.get("available") if isinstance(payload, dict) else None
            reason = payload.get("reason") if isinstance(payload, dict) else None
            if isinstance(payload, dict) and "error" in payload:
                reason = "tool_error"  # инструмент отверг аргументы — компании в состояние не пишем
            item_dates = _item_dates(payload)  # compare_companies: дата у каждой компании своя
            for inn in _inns_from_args(args) if reason != "tool_error" else []:
                if inn not in inns:
                    inns.append(inn)
                report_date = payload.get("report_date") if isinstance(payload, dict) else None
                if item_dates.get(inn) or report_date:
                    dates[inn] = item_dates.get(inn) or report_date
            trace.append(
                ToolCallTrace(
                    name=name,
                    args=args,
                    available=available,
                    reason=reason,
                    result_chars=len(msg.content) if isinstance(msg.content, str) else 0,
                )
            )
        return {
            "messages": tool_messages,
            "selected_inns": inns,
            "report_dates": dates,
            "trace": trace,
        }

    async def finalize(state: AgentState) -> dict[str, Any]:
        _, hint = question_kind_hint(str(state.get("question") or ""))
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
            HumanMessage(content=f"{FINALIZE_PROMPT}\n\n{hint}" if hint else FINALIZE_PROMPT),
        ]
        try:
            draft = await structured.ainvoke(messages)
            if not isinstance(draft, Draft):
                draft = Draft.model_validate(draft)
        except Exception as e:  # модель не смогла выдать структуру — честный отказ, не падение
            last = next(
                (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), ""
            )
            draft = Draft(
                kind="refusal",
                text_md=str(last) or f"Не удалось собрать ответ: {type(e).__name__}.",
                citations=[],
            )
        citations = list(draft.citations)
        seen_paths = {c.source_path for c in citations}
        for extra in extract_inline_citations(
            draft.text_md
        ):  # адреса из текста, если модель их не перечислила
            if extra.source_path not in seen_paths:
                citations.append(extra)
                seen_paths.add(extra.source_path)
        inns = state.get("selected_inns") or []
        cards = [c for c in (build_card(tools_layer, inn) for inn in inns) if c]
        answer = Answer(
            kind=draft.kind,
            text_md=draft.text_md,
            card=cards[0] if draft.kind == "card" and cards else None,
            cards=cards if draft.kind == "comparison" else [],
            citations=citations,
            report_dates=dict(state.get("report_dates") or {}),
        )
        return {"draft": draft, "answer": answer}

    async def validate(state: AgentState) -> dict[str, Any]:
        answer = state["answer"]
        assert answer is not None
        real = [c for c in answer.citations if not is_meta_path(c.source_path)]
        checks = validate_citations(source, state.get("selected_inns") or [], real)
        invalid = [c for c in checks if not c.ok]
        verdict_problem = (
            verdict_mismatch(answer)
            or forbidden_problem(answer.text_md)
            or format_problem(answer, str(state.get("question") or ""))
        )
        retry = state.get("citation_retry") or 0
        if (invalid or verdict_problem) and retry < MAX_CITATION_RETRIES:
            repair = citation_repair_prompt(
                [(c.citation.claim, c.citation.source_path, c.why or "") for c in invalid],
                verdict_problem,
            )
            return {
                "messages": [HumanMessage(content=repair)],
                "citation_retry": retry + 1,
                "answer": None,
            }
        if answer.card:
            text = enforce_verdict(
                scrub_forbidden(
                    replace_verdict_codes(answer.text_md), VERDICT_RU[answer.card.verdict]
                ),
                answer.card.verdict,
            )
        elif answer.cards:
            text = enforce_comparison(scrub_forbidden(answer.text_md, None), answer.cards)
        else:
            text = scrub_forbidden(answer.text_md, None)
        checked = answer.model_copy(
            update={
                "text_md": text,
                "citations": [c.citation for c in checks if c.ok],
                "invalid_citations": [c.citation for c in invalid],
            }
        )
        return {"answer": checked}

    return {"agent": agent, "tools": tools_, "finalize": finalize, "validate": validate}


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalize"


def route_after_validate(state: AgentState) -> str:
    return END if state.get("answer") is not None else "agent"


def verdict_mismatch(answer: Answer) -> str | None:
    """Текст модели не должен спорить с вердиктом карточки, посчитанным по сигналам.
    При сравнении — с вердиктом каждой компании."""
    if answer.card is not None:
        expected = VERDICT_RU[answer.card.verdict]
        lowered = answer.text_md.casefold()
        if expected in lowered:
            return None
        others = [v for v in VERDICT_RU.values() if v != expected and v in lowered]
        if others:
            return (
                f"Вывод в тексте («{others[0]}») не совпадает с рекомендацией по сигналам. "
                f"Рекомендация должна быть буквально: «{expected}»."
            )
        return f"В тексте нет рекомендации. Добавь буквально: «{expected}»."
    if answer.cards:
        missing = [
            c for c in answer.cards if VERDICT_RU[c.verdict] not in answer.text_md.casefold()
        ]
        if missing:
            wanted = "; ".join(
                f"{c.name} (ИНН {c.inn}) — «{VERDICT_RU[c.verdict]}»" for c in missing
            )
            return f"Для каждой компании вывод должен быть буквально по verdict_ru: {wanted}."
    return None


_SECTION_HINTS = (  # слова вопроса → раздел отчёта: подсказка виду ответа, чтобы слабая модель не давала сводку
    ("долги у приставов", re.compile(r"пристав|исполнительн|долг", re.I)),
    ("суды", re.compile(r"\bсуд|\bиск|арбитраж", re.I)),
    ("финансы", re.compile(r"выручк|прибыл|убыт|актив|капитал|ликвидн|отчётност|оборот", re.I)),
    ("лицензии", re.compile(r"лиценз", re.I)),
    ("филиалы", re.compile(r"филиал", re.I)),
    ("проверки госорганов", re.compile(r"проверк[аи]\b|проверял|инспекц|надзор", re.I)),
    ("виды деятельности", re.compile(r"оквэд|вид\w* деятельн", re.I)),
    ("госзакупки", re.compile(r"закупк|тендер|госзаказ", re.I)),
    ("численность", re.compile(r"сотрудник|численност|\bштат|персонал", re.I)),
)
_CARD_RE = re.compile(
    r"провер(ь|ить|ка)\b|что можешь сказать|можно ли .{0,40}работ|стоит ли .{0,40}работ|отсрочк|предоплат"
    r"|заключ\w* договор|кому верить|надёжн|стоит ли связыв",
    re.I,
)
_COMPARE_RE = re.compile(r"сравни|с кем лучше|кого выбрать|кто из них", re.I)


def question_kind_hint(question: str) -> tuple[str | None, str | None]:
    """Вид ответа по словам вопроса и подсказка модели: несколько компаний → comparison,
    вопрос про раздел → answer, «можно ли работать» → card. Решает модель, но с якорем."""
    if _COMPARE_RE.search(question) or len(re.findall(r"\b\d{10,12}\b", question)) >= 2:
        return "comparison", "Подсказка: в вопросе несколько компаний — kind «comparison»."
    for name, rx in _SECTION_HINTS:
        if rx.search(question):
            return "answer", (
                f"Подсказка: вопрос про один раздел ({name}) — kind «answer»: сначала ответ по "
                "существу с числами, карточку не повторяй; если спрашивают, можно ли работать "
                "или давать отсрочку — добавь вывод verdict_ru одной строкой."
            )
    if _CARD_RE.search(question):
        return "card", "Подсказка: просят оценить, можно ли работать с компанией — kind «card»."
    return None, None


def format_problem(answer: Answer, question: str) -> str | None:
    """Вопрос про один раздел, а в ответе — сводка по компании: повод для круга исправления."""
    kind, _ = question_kind_hint(question)
    if kind != "answer":
        return None
    text = answer.text_md.casefold()
    summary_like = (
        answer.kind == "card" or "обратить внимание" in text or len(text.splitlines()) > 10
    )
    if not summary_like:
        return None
    return (
        "Это вопрос про один раздел отчёта. Перепиши как kind «answer»: сначала ответ по существу "
        "с числами (одна-три строки), потом не больше одной строки контекста. Полную карточку "
        "и список «на что обратить внимание» не повторяй."
    )


FORBIDDEN_RU = (  # характеристика вместо действия — CRITERIA §2, инвариант 5
    "работать нельзя",
    "нельзя работать",
    "не рекомендуем",
    "не связывайтесь",
    "ненадёжн",
    "сомнительн",
    "однодневк",
)


def forbidden_problem(text: str) -> str | None:
    """Категоричная характеристика в тексте — повод для круга исправления."""
    lowered = text.casefold()
    found = [f for f in FORBIDDEN_RU if f in lowered]
    if not found:
        return None
    return (
        f"Убери формулировку «{found[0]}»: рекомендация — это действие для пользователя, "
        "одна из трёх штатных фраз, а не характеристика компании."
    )


def scrub_forbidden(text: str, replacement: str | None) -> str:
    """После неудачного круга исправления категоричные фразы вычищает код."""
    out = text
    for f in FORBIDDEN_RU:
        pattern = re.compile(rf"(?:компания\s+)?{re.escape(f)}\w*", re.IGNORECASE)
        out = pattern.sub(replacement or "", out)
    return re.sub(r"[ \t]{2,}", " ", out)


_VERDICT_CODE_RE = re.compile(r"(?<![\w/])(not_recommended|check|ok)(?![\w/])")


def replace_verdict_codes(text: str) -> str:
    """Код исхода в тексте («ГДК — not_recommended») → штатная фраза."""
    return _VERDICT_CODE_RE.sub(lambda m: VERDICT_RU[Verdict(m.group(1))], text)


def enforce_comparison(text: str, cards: list[Card]) -> str:
    """После неудачного круга исправления итог сравнения дописывается кодом."""
    text = replace_verdict_codes(text)
    lowered = text.casefold()
    missing = [c for c in cards if VERDICT_RU[c.verdict] not in lowered]
    if not missing:
        return text
    lines = [f"- {c.name} (ИНН {c.inn}): {VERDICT_RU[c.verdict]}" for c in missing]
    return f"{text.rstrip()}\n\n**По данным отчётов:**\n" + "\n".join(lines)


def enforce_verdict(text: str, verdict: Verdict) -> str:
    """После неудачного круга исправления вывод в тексте заменяется кодом."""
    expected = VERDICT_RU[verdict]
    out = text
    for other in VERDICT_RU.values():
        if other == expected:
            continue
        out = re.compile(re.escape(other), re.IGNORECASE).sub(expected, out)
    if expected not in out.casefold():
        out = f"{out.rstrip()}\n\n**Рекомендация:** {expected}."
    return out


def build_card(tools: Tools, inn: str) -> Card | None:
    """Карточка — кодом из сигналов: вердикт и пробелы модель испортить не может."""
    signals = tools.get_risk_signals(inn)
    summary = tools.get_report_summary(inn)
    if not signals.available or not summary.available:
        return None
    data, info = signals.data, summary.data
    attention = [
        Attention(
            claim=s["explanation"], severity=Severity(s["severity"]), source_path=s["source_path"]
        )
        for severity in (Severity.CRITICAL, Severity.MODERATE)
        for s in data["signals"][severity.value]
    ]
    asks = list(dict.fromkeys(g["ask"] for g in data["gaps"] if g.get("ask")))
    return Card(
        inn=inn,
        name=info["short_name"],
        labels=CardLabels(riskLevel=data["labels"]["svetofor"], zskRiskLevel=data["labels"]["zsk"]),
        verdict=Verdict(data["verdict"]),
        terminal=bool(data.get("terminal")),
        attention=attention,
        ask_before=asks,
        gaps=[g["text"] for g in data["gaps"]],
        report_date=signals.report_date,
    )


def _parse(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return content
    return content


def _item_dates(payload: Any) -> dict[str, str]:
    """ИНН → дата отчёта из списка компаний в данных инструмента (сравнение)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    return {
        str(i["inn"]): str(i["report_date"])
        for i in items
        if isinstance(i, dict) and i.get("inn") and i.get("report_date")
    }


def _inns_from_args(args: dict[str, Any]) -> list[str]:
    inn = args.get("inn")
    inns = args.get("inns") or []
    if isinstance(inns, str):  # слабая модель передаёт список строкой
        inns = re.findall(r"\d{10,12}", inns)
    out = [str(inn)] if inn else []
    out.extend(str(i) for i in inns)
    return out
