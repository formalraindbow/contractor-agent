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
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from contractor_agent.agent.citations import (
    extract_inline_citations,
    is_meta_path,
    normalize_path,
    numbers_in,
    validate_citations,
)
from contractor_agent.agent.comparison_answers import comparison_followup
from contractor_agent.agent.llm import LLM
from contractor_agent.agent.presentation import comparison_text, public_text
from contractor_agent.agent.prompt import FINALIZE_PROMPT, SYSTEM_PROMPT, citation_repair_prompt
from contractor_agent.agent.schema import Answer, Attention, Card, CardLabels, Citation, Draft
from contractor_agent.agent.scoped_answers import scoped_answer
from contractor_agent.agent.section_answers import section_answer
from contractor_agent.agent.state import AgentState, ToolCallTrace
from contractor_agent.data.loader import ReportSource, normalize_name, strip_legal_form
from contractor_agent.mcp_server.tools import Tools
from contractor_agent.signals.model import VERDICT_RU, Severity, Verdict

MAX_CITATION_RETRIES = 1
Node = Callable[[AgentState], Awaitable[dict[str, Any]]]


def make_nodes(llm: LLM, tools: list[Any], source: ReportSource) -> dict[str, Node]:
    with_tools = llm.with_tools(tools)
    by_name = {t.name: t for t in tools}

    def tools_for(question: str) -> Any:
        """Вопрос про раздел — модели даём только инструменты этого раздела: без get_risk_signals
        она не соберёт карточку вместо ответа. Сравнение — только поиск и compare_companies."""
        names = tool_subset(question)
        subset = [by_name[n] for n in names if n in by_name] if names else []
        return llm.with_tools(subset) if subset else with_tools

    structured = llm.structured(Draft)
    tool_node = ToolNode(tools, handle_tool_errors=True)
    tools_layer = Tools(source)

    def answer_inns(state: AgentState) -> list[str]:
        question = str(state.get("question") or "")
        candidates = list(
            dict.fromkeys(
                [
                    *(state.get("turn_inns") or []),
                    *(state.get("selected_inns") or []),
                    *re.findall(r"\b\d{10,12}\b", question),
                ]
            )
        )
        return requested_inns(question, candidates, source) or list(
            state.get("turn_inns") or state.get("selected_inns") or []
        )

    async def guard(state: AgentState) -> dict[str, Any]:
        """Короткий ответ без модели и инструментов: посторонний ввод не повод собирать карточку."""
        reply = offtopic_reply(str(state.get("question") or "")) or _CAPABILITIES
        return {
            "answer": Answer(
                kind="refusal",
                text_md=reply,
                citations=[],
                report_dates={},
            )
        }

    async def agent(state: AgentState) -> dict[str, Any]:
        question = str(state.get("question") or "")
        history = visible_history(state["messages"], question)
        messages = [SystemMessage(content=SYSTEM_PROMPT), *history]
        focus = requested_inns(question, answer_inns(state), source)
        if focus:
            messages.append(
                SystemMessage(
                    content="Компании текущего вопроса (ИНН): "
                    + ", ".join(focus)
                    + ". Отвечай о них; общий список прошлого сравнения не расширяет этот вопрос."
                )
            )
        response = await tools_for(question).ainvoke(messages)
        return {"messages": [response]}

    async def tools_(state: AgentState) -> dict[str, Any]:
        result = await tool_node.ainvoke(state)
        tool_messages: list[ToolMessage] = result["messages"]
        last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        args_by_id = {c["id"]: (c["name"], c["args"]) for c in last_ai.tool_calls}
        inns = list(state.get("selected_inns") or [])
        turn_inns = list(state.get("turn_inns") or [])
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
                if inn not in turn_inns:
                    turn_inns.append(inn)
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
            "turn_inns": turn_inns,
            "report_dates": dates,
            "trace": trace,
        }

    async def finalize(state: AgentState) -> dict[str, Any]:
        _, hint = question_kind_hint(str(state.get("question") or ""))
        if is_follow_up(state["messages"]) and not _FULL_CHECK_RE.search(
            str(state.get("question") or "")
        ):
            hint = ((hint + " ") if hint else "") + FOLLOW_UP_HINT
        question = str(state.get("question") or "")
        inns = answer_inns(state)
        cards = [c for c in (build_card(tools_layer, inn) for inn in inns) if c]
        identities = ", ".join(inns)
        history = finalization_history(state["messages"], question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history,
            HumanMessage(
                content=f"{FINALIZE_PROMPT}\n\n{hint or ''}\n\n"
                f"Компании текущего вопроса (ИНН): {identities}. Отвечай только о них.\n"
                f"Текущий вопрос, на который нужно ответить: {question}"
            ),
        ]
        try:
            # Narrow policy answers still use current report evidence and the same validator.
            turn_inns = inns
            draft = (
                comparison_followup(cards, question)
                or section_answer(tools_layer, turn_inns, question)
                or scoped_answer(tools_layer, turn_inns, question)
            )
            if draft is None:
                draft = await structured.ainvoke(messages)
            if not isinstance(draft, Draft):
                draft = Draft.model_validate(draft)
        except Exception as e:
            logging.getLogger(__name__).exception("Structured answer failed")
            # Tool reasoning and incomplete drafts must never become a public answer.
            raise RuntimeError("Не удалось собрать проверяемый ответ") from e
        citations = list(draft.citations)
        seen_paths = {c.source_path for c in citations}
        for extra in extract_inline_citations(
            draft.text_md
        ):  # адреса из текста, если модель их не перечислила
            if extra.source_path not in seen_paths:
                citations.append(extra)
                seen_paths.add(extra.source_path)
        question = str(state.get("question") or "")
        kind_hint, _ = question_kind_hint(question)
        if kind_hint == "comparison" and len(cards) >= 2 and draft.kind != "comparison":
            draft = draft.model_copy(
                update={"kind": "comparison"}
            )  # несколько компаний — сравнение
        elif (kind_hint == "answer" and draft.kind == "card") or (
            len(inns) == 1 and draft.kind == "comparison"
        ):
            draft = draft.model_copy(update={"kind": "answer"})
        elif kind_hint == "card" and cards and draft.kind in ("refusal", "answer"):
            # просили проверить компанию, карточка собрана кодом: пробелы в отчёте не повод
            # отдавать ответ без рекомендации — иначе интерфейс теряет вывод и вид ответа
            draft = draft.model_copy(update={"kind": "card"})
        answer = Answer(
            kind=draft.kind,
            text_md=draft.text_md,
            card=cards[0] if draft.kind == "card" and cards else None,
            cards=cards if draft.kind == "comparison" else [],
            citations=citations,
            report_dates={
                inn: day for inn, day in (state.get("report_dates") or {}).items() if inn in inns
            },
        )
        return {"draft": draft, "answer": answer, "turn_inns": inns}

    async def validate(state: AgentState) -> dict[str, Any]:
        answer = state["answer"]
        assert answer is not None
        real = [c for c in answer.citations if not is_meta_path(c.source_path)]
        checks = validate_citations(
            source, list(state.get("turn_inns") or state.get("selected_inns") or []), real
        )
        invalid = [c for c in checks if not c.ok]
        require_comparison = comparison_needs_verdict(str(state.get("question") or "")) and (
            comparison_followup(answer.cards, str(state.get("question") or "")) is None
        )
        verdict_problem = (
            scope_problem(
                answer,
                str(state.get("question") or ""),
                list(state.get("selected_inns") or []),
                source,
            )
            or refusal_problem(answer, state.get("trace") or [])
            or empty_problem(answer)
            or verdict_mismatch(answer, require_comparison=require_comparison)
            or forbidden_problem(answer.text_md)
            or format_problem(answer, str(state.get("question") or ""))
            or details_problem(answer, str(state.get("question") or ""))
            or absence_problem(answer.text_md)
        )
        retry = state.get("citation_retry") or 0
        if (invalid or verdict_problem) and retry < MAX_CITATION_RETRIES:
            repair = citation_repair_prompt(
                [(c.citation.claim, c.citation.source_path, c.why or "") for c in invalid],
                verdict_problem,
            )
            return {
                "messages": [HumanMessage(content=repair, additional_kwargs={"repair": True})],
                "citation_retry": retry + 1,
                "answer": None,
            }
        base = public_text(
            tidy_text(drop_invalid_lines(answer.text_md, [c.citation for c in invalid]))
        )
        if answer.card:
            text = enforce_verdict(
                scrub_forbidden(replace_verdict_codes(base), VERDICT_RU[answer.card.verdict]),
                answer.card.verdict,
            )
        elif answer.cards and require_comparison:
            text = enforce_comparison(scrub_forbidden(base, None), answer.cards)
        else:
            text = scrub_forbidden(base, None)
        text = public_text(tidy_text(text))
        dates = answer.report_dates
        if len(dates) == 1 and re.search(
            r"финанс|выручк|ликвид|отч[её]тност",
            question_subject(str(state.get("question") or "")),
            re.I,
        ):
            inn = next(iter(dates))
            financials = tools_layer.get_financials(inn)
            if financials.available:
                stale = next(
                    (s for s in financials.data.get("signals", []) if s["code"] == "fin_stale"),
                    None,
                )
                if stale and not re.search(r"31[.]03[.]|31 марта", text):
                    note = stale["explanation"]
                    start = re.search(r"(?m)^Отч[её]т от ", text)
                    at = start.start() if start else len(text)
                    text = text[:at].rstrip() + "\n\n" + note + "\n\n" + text[at:]
        subject = question_subject(str(state.get("question") or ""))
        if len(dates) == 1 and _COURT_Q.search(subject) and not re.search(r"20\d{2}", subject):
            inn = next(iter(dates))
            courts = tools_layer.get_arbitration_summary(inn)
            if courts.available:
                for signal in courts.data.get("signals", []):
                    if "mismatch" in signal["code"] and signal["explanation"] not in text:
                        text += "\n\n" + signal["explanation"]
        if len(dates) > 1:
            text = comparison_text(text)
        else:
            for day in dates.values():
                ru = ".".join(reversed(day.split("-")))
                if ru not in text and day not in text:
                    text += f"\n\nОтчёт от {ru}."
        text = public_text(text)
        checked = answer.model_copy(
            update={
                "text_md": text,
                "citations": [
                    c.citation.model_copy(
                        update={
                            "claim": public_text(c.citation.claim),
                            "source_path": normalize_path(path),
                            "inn": c.inn,
                        }
                    )
                    for c in checks
                    if c.ok
                    for path in re.split(r"[,;]", c.citation.source_path)
                    if path.strip()
                ],
                "invalid_citations": [c.citation for c in invalid],
            }
        )
        return {
            "answer": checked,
            "messages": [
                AIMessage(content=checked.text_md, additional_kwargs={"final_answer": True})
            ],
        }

    return {
        "guard": guard,
        "agent": agent,
        "tools": tools_,
        "finalize": finalize,
        "validate": validate,
    }


def route_input(state: AgentState) -> str:
    """Первая развилка: посторонний ввод не тратит модель и инструменты."""
    return "guard" if offtopic_reply(str(state.get("question") or "")) else "agent"


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalize"


def route_after_validate(state: AgentState) -> str:
    return END if state.get("answer") is not None else "agent"


def verdict_mismatch(answer: Answer, *, require_comparison: bool = True) -> str | None:
    """Текст модели не должен спорить с вердиктом карточки, посчитанным по сигналам.
    При сравнении — с вердиктом каждой компании."""
    if answer.card is not None:
        expected = VERDICT_RU[answer.card.verdict]
        lowered = answer.text_md.casefold()
        if verdict_present(answer.text_md, answer.card.verdict):
            return None
        others = [v for v in VERDICT_RU.values() if v != expected and v in lowered]
        if others:
            return (
                f"Вывод в тексте («{others[0]}») не совпадает с рекомендацией по сигналам. "
                f"Рекомендация должна быть буквально: «{expected}»."
            )
        return f"В тексте нет рекомендации. Добавь буквально: «{expected}»."
    if answer.cards and require_comparison:
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
    (
        "финансы",
        re.compile(r"финанс|выручк|прибыл|убыт|актив|капитал|ликвидн|отч[её]тност|оборот", re.I),
    ),
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
_DECISION_RE = re.compile(
    r"можно (?:ли|им|давать|платить|работать)|стоит ли|давать ли|кому верить|или лучше|"
    r"безопасно ли|рискованно ли|надо ли|потянет ли",
    re.I,
)
_DOCUMENTS_Q = re.compile(r"что (?:запросить|уточнить)|какие документы|список документов", re.I)
_EXPLAIN_Q = re.compile(
    r"почему.{0,70}(?:вывод|сигнал|риск)|объясни.{0,20}(?:рекомендац|вывод)", re.I
)
_MEANING_Q = re.compile(
    r"что\s+(?:это\s+)?(?:значит|означает)|как\s+(?:это\s+)?понимать|поясни|объясни", re.I
)
_HEAD_Q = re.compile(r"руковод|управляющ|директор|сколько лет|возраст компан", re.I)
_LABEL_Q = re.compile(r"зск|светофор|метк[аиу]|оценк[аиу] банка", re.I)


def question_subject(question: str) -> str:
    # The UI appends company identity and goal; those are context, not another question.
    return re.split(r"Речь о компании|(?:Контекст сравнения|Компании):", question, maxsplit=1)[
        0
    ].strip()


def requested_inns(question: str, candidates: list[str], source: ReportSource) -> list[str]:
    """Named companies narrow this turn; the comparison remains in session memory."""
    subject = normalize_name(question_subject(question))
    if re.search(r"остальн|другими|всеми|у всех|каждой|обоих|из них|с кем", subject):
        return []
    return mentioned_inns(subject, candidates, source)


def mentioned_inns(text: str, candidates: list[str], source: ReportSource) -> list[str]:
    subject = normalize_name(text)
    mentioned = []
    for inn in dict.fromkeys(candidates):
        report = source.get(inn)
        if report is None:
            continue
        name = strip_legal_form(normalize_name(report.base_info.short_name))
        if re.search(rf"\b{re.escape(inn)}\b", subject) or (
            name and re.search(rf"\b{re.escape(name)}(?:а|у|ом|е)?\b", subject)
        ):
            mentioned.append(inn)
    return mentioned


def scope_problem(
    answer: Answer, question: str, candidates: list[str], source: ReportSource
) -> str | None:
    focus = requested_inns(question, candidates, source)
    if not focus:
        return None
    outside = [inn for inn in candidates if inn not in focus]
    if mentioned_inns(answer.text_md, outside, source) or any(
        citation.inn in outside for citation in answer.citations
    ):
        return (
            "Ответ вышел за рамки текущего вопроса. Объясни только запрошенный факт для ИНН "
            + ", ".join(focus)
            + ". Убери остальные компании, их выводы и цитаты; прошлое сравнение повторять не нужно."
        )
    return None


def comparison_needs_verdict(question: str) -> bool:
    """A question about specific sections does not need the general verdict repeated."""
    question = question_subject(question)
    if re.search(r"с кем|кого выбрать|кто из них|можно.{0,20}работать", question, re.I):
        return True
    return not (
        _DOCUMENTS_Q.search(question)
        or _MEANING_Q.search(question)
        or any(rx.search(question) for _, rx in _SECTION_HINTS)
    )


_SECTION_TOOLS = {
    "долги у приставов": ["get_enforcement_summary"],
    "суды": ["get_arbitration_summary"],
    "финансы": ["get_financials"],
    "лицензии": ["get_section"],
    "филиалы": ["get_section"],
    "проверки госорганов": ["get_section"],
    "виды деятельности": ["get_section"],
    "госзакупки": ["get_section"],
    "численность": ["get_section", "get_report_summary"],
}


def tool_subset(question: str) -> list[str] | None:
    """Имена инструментов под вид вопроса; None — все (карточка и неопределённые вопросы)."""
    question = question_subject(question)
    kind, _ = question_kind_hint(question)
    if kind == "comparison":
        return ["search_company", "compare_companies"]
    if kind != "answer":
        return None
    selected = ["search_company", "get_report_summary"]
    if _DOCUMENTS_Q.search(question) or _EXPLAIN_Q.search(question):
        selected.extend(["get_risk_signals", "get_financials"])
    if _MEANING_Q.search(question):
        selected.append("get_risk_signals")
    if _HEAD_Q.search(question) or _LABEL_Q.search(question):
        selected.append("get_risk_signals")
    for name, rx in _SECTION_HINTS:
        if rx.search(question):
            selected.extend(_SECTION_TOOLS[name])
    return list(dict.fromkeys(selected)) if len(selected) > 2 else None


_ABUSE_RE = re.compile(
    r"\b(?:на|по|о|за|в)?\s*(?:бл[яa]д|ху[йёеяю]|пизд|[еёe]б[аоу]|сук[аиу]\b|мудак|долбо|"
    r"гандон|пид[оа]р|чмо\b|уёб|охуе|нахуй|шлюх|дебил|идиот|тварь)",
    re.I,
)
_INJECTION_RE = re.compile(
    r"(?:игнорир\w*|забудь|отмени|не\s+следуй)[^.\n]{0,30}(?:инструкц|правил|промпт|систем)"
    r"|(?:покажи|выведи|напечатай|повтори)[^.\n]{0,20}(?:систем\w*\s*промпт|свои\s+инструкц)"
    r"|ты\s+(?:теперь|больше\s+не)\b|представь,?\s+что\s+ты\b|веди\s+себя\s+как\b"
    r"|ignore\s+(?:all\s+)?(?:previous|prior)|disregard\s+(?:all\s+)?(?:previous|your)"
    r"|system\s+prompt|jailbreak|developer\s+mode|act\s+as\s+(?:a\s+)?(?:dan|different)",
    re.I,
)
_TOPIC_RE = re.compile(
    r"\d{10,12}|компан|контрагент|фирм|ооо|ип\b|инн|суд|иск|приста|долг|финанс|выручк|прибыл|"
    r"убыт|отчёт|отчет|светофор|зск|лиценз|адрес|директор|учредит|банкрот|реестр|закупк|проверк|"
    r"работать|отсрочк|предоплат|сделк|договор|поставк|надёжн|надежн|риск|что\s+ты\s+умеешь|"
    r"чем\s+(?:ты\s+)?поможешь|привет|здравств|спасибо|помог",
    re.I,
)
_OFFTOPIC_RE = re.compile(
    r"(?:какая|какую|скажи|расскажи|покажи|узнай|прогноз)[^.!?\n]{0,30}погод"
    r"|\bпогод[ауы]\s+(?:в\s|на\s|сегодня|завтра)"
    r"|(?:расскажи|придумай)\s+анекдот"
    r"|(?:напиши|сочини)\s+(?:стих|песн|сказк)"
    r"|(?:дай|подскажи|напиши)\s+рецепт",
    re.I,
)
_CAPABILITIES = (
    "Я отвечаю по отчёту банка о контрагенте: оценки банка, суды, долги у приставов, финансы, "
    "лицензии, проверки, виды деятельности — и говорю, на каких условиях с компанией работать. "
    "Назовите компанию или ИНН."
)
_INJECTION_REPLY = (
    "Правила проверки я не меняю и инструкции не показываю: отвечаю только фактами из отчёта "
    "банка. Спросите про компанию — суды, долги у приставов, финансы, оценки банка."
)
_OFFTOPIC_REPLY = (
    "Я могу помочь с вопросами по банковскому отчёту о контрагенте: "
    "финансами, судами, долгами и оценками банка. На этот вопрос в отчёте ответа нет."
)


def offtopic_reply(question: str) -> str | None:
    """Не всё, что написал пользователь, — вопрос про контрагента. Ругань, попытку сменить
    правила и болтовню разбираем кодом: до модели и инструментов это не доходит."""
    text = question.strip()
    if not text:
        return "Напишите вопрос: название компании, ИНН или что посмотреть в отчёте."
    if _INJECTION_RE.search(text):
        return _INJECTION_REPLY
    if _ABUSE_RE.search(text):
        return "Давайте по делу. " + _CAPABILITIES
    # Company identity appended by the UI does not turn an unrelated request into
    # a report question. Keep the full text above for injection and abuse checks.
    if _OFFTOPIC_RE.search(question_subject(text)):
        return _OFFTOPIC_REPLY
    if len(text) <= 60 and not _TOPIC_RE.search(text) and not re.search(r"\?$", text):
        return _CAPABILITIES
    return None


def question_kind_hint(question: str) -> tuple[str | None, str | None]:
    """Вид ответа по словам вопроса и подсказка модели: несколько компаний → comparison,
    вопрос про раздел → answer, «можно ли работать» → card. Решает модель, но с якорем."""
    question = question_subject(question)
    if _COMPARE_RE.search(question) or len(re.findall(r"\b\d{10,12}\b", question)) >= 2:
        return "comparison", "Подсказка: в вопросе несколько компаний — kind «comparison»."
    if _DOCUMENTS_Q.search(question):
        return (
            "answer",
            "Нужен список документов с причиной каждого запроса по фактам компании; не пересказ судов или приставов.",
        )
    if _EXPLAIN_Q.search(question):
        return (
            "answer",
            "Ответь, почему получился такой вывод, по get_risk_signals. Если сигналов не выделено, объясни результат проверки доступных сведений и существенные ограничения; пропуски данных не означают благополучия. Не заполняй ответ выдуманными фактами. Это вывод помощника, не рекомендация банка.",
        )
    if _DECISION_RE.search(
        question
    ):  # просят решение — карточка с выводом, даже если назван раздел
        return "card", (
            "Подсказка: просят решение (можно ли работать, давать отсрочку, кому верить) — "
            "kind «card» с выводом из verdict_ru; факты по разделу из вопроса — первыми."
        )
    if _MEANING_Q.search(question):
        return (
            "answer",
            "Объясни только запрошенный факт или выражение: что оно означает и что подтверждено "
            "в отчёте. Для отметок ФНС, адреса и блокировок обязательно используй get_risk_signals. "
            "Наличие адреса в реквизитах не опровергает отметку о его недостоверности. "
            "Не добавляй общий вердикт, перечень других рисков и выводы по остальным компаниям.",
        )
    if _HEAD_Q.search(question):
        return (
            "answer",
            "Назови руководителя, должность, дату назначения и возраст, если их спросили. Не перерисовывай карточку.",
        )
    for name, rx in _SECTION_HINTS:
        if rx.search(question):
            return "answer", (
                f"Подсказка: вопрос касается раздела ({name}) — kind «answer»: ответь на все части "
                "вопроса по существу с числами, карточку не повторяй; если спрашивают, можно ли работать "
                "или давать отсрочку — добавь вывод verdict_ru одной строкой."
            )
    if _CARD_RE.search(question):
        return "card", "Подсказка: просят оценить, можно ли работать с компанией — kind «card»."
    return None, None


_COURT_Q = re.compile(r"\bсуд|\bиск|арбитраж|ответчик|истец", re.I)
_BAILIFF_Q = re.compile(r"пристав|исполнительн", re.I)
_ROLE_WORDS = re.compile(r"ответчик|истец|истц|подавал|к компании|против компании", re.I)
_FINISHED_WORDS = re.compile(r"заверш|закрыт|окончен", re.I)
_ACTIVE_WORDS = re.compile(r"действующ|активн|открыт|текущ|непогашен", re.I)


def details_problem(answer: Answer, question: str) -> str | None:
    """Ответ про суды без ролей и про приставов без разделения — самая частая потеря смысла.
    Пользователю нужно знать, кто на кого подавал и что из долгов ещё висит."""
    text = answer.text_md
    if _COURT_Q.search(question) and not _ROLE_WORDS.search(text):
        return (
            "В ответе про суды не указана роль. Назови сначала общее число дел из ответа "
            "инструмента, затем разбивку: сколько дел, где компания ответчик, и сколько, где истец; "
            "если по роли дел не найдено — так и скажи. Общее число не убирай."
        )
    if _BAILIFF_Q.search(question) and not (
        _ACTIVE_WORDS.search(text) and _FINISHED_WORDS.search(text)
    ):
        return (
            "В ответе про приставов назови общее число производств, а затем отдельно действующие "
            "(сколько и на какую сумму) и завершённые (сколько), не складывая их."
        )
    return None


def format_problem(answer: Answer, question: str) -> str | None:
    """Вопрос про один раздел, а в ответе — сводка по компании: повод для круга исправления."""
    kind, _ = question_kind_hint(question)
    if kind != "answer":
        return None
    text = answer.text_md.casefold()
    summary_like = (
        answer.kind == "card" or "обратить внимание" in text or "выводы по компаниям" in text
    )
    if not summary_like:
        return None
    return (
        "Это вопрос про один раздел отчёта. Перепиши как kind «answer»: сначала ответ по существу "
        "с числами, по смысловым блокам, затем необходимый контекст. Полную карточку "
        "и список «на что обратить внимание» не повторяй."
    )


FORBIDDEN_RU = (  # характеристика вместо действия — CRITERIA §2, инвариант 5 (регулярные выражения)
    r"работать[^.\n]{0,40}?нельзя",
    r"нельзя[^.\n]{0,20}?работать",
    r"не рекомендуем",
    r"не связывайтесь",
    r"ненадёжн\w*",
    r"сомнительн\w*",
    r"однодневк\w*",
)
_FORBIDDEN_RES = [re.compile(f, re.IGNORECASE) for f in FORBIDDEN_RU]


_ABSENCE_NOUNS = r"судов|дел|исков|производств|долгов|нарушений|проверок|лицензий|штрафов|претензий"
_ABSENCE_CLAIM = re.compile(
    rf"\b(?:{_ABSENCE_NOUNS})\s+(?:у\s+\w+\s+)?нет\b"
    r"|\bникаких\s+[\w\s]{0,40}?(?:нет|не\s+зафиксирован\w*|не\s+имеется|отсутству\w+)\b"
    rf"|\b(?:{_ABSENCE_NOUNS})[\w\s]{{0,20}}?(?:не\s+зафиксирован\w*|не\s+имеется)\b"
    r"|\bне\s+имеет\s+(?:судов|долгов|дел)\b",
    re.I,
)


def absence_problem(text: str) -> str | None:
    """«Судов нет» — вывод, которого отчёт не даёт: в нём нет записей, а не гарантия отсутствия.
    Правило 3 промпта, самая частая подмена смысла у слабой модели."""
    m = _ABSENCE_CLAIM.search(text)
    if not m:
        return None
    return (
        f"Формулировка «{m.group(0)}» утверждает больше, чем есть в отчёте. Пиши «в отчёте не "
        "найдено» с оговоркой, что это не значит отсутствия, — и только про тот раздел, который "
        "инструмент действительно вернул."
    )


def refusal_problem(answer: Answer, trace: list[ToolCallTrace]) -> str | None:
    """Отказ, когда инструмент вернул данные, — ошибка модели, а не пробел в отчёте."""
    if answer.kind != "refusal":
        return None
    with_data = [t.name for t in trace if t.available and t.result_chars > 400]
    if not with_data:
        return None
    return (
        f"Данные получены ({', '.join(dict.fromkeys(with_data))}), отказываться нельзя. "
        "Перепиши ответ по существу: назови числа из ответа инструмента и дату отчёта; "
        "«нет сведений» пиши только про то, чего в отчёте действительно нет."
    )


def empty_problem(answer: Answer) -> str | None:
    """Пустой или обрывочный ответ (gpt-oss-20b отдавал lines=[]) — круг исправления."""
    if len(re.sub(r"[\s*#|_\-—–]+", "", answer.text_md)) >= 12:
        return None
    return (
        "Ответ пустой. Напиши ответ пользователю в lines: по фактам из ответов инструментов, "
        "с числами и адресами полей, с датой отчёта."
    )


def forbidden_problem(text: str) -> str | None:
    """Категоричная характеристика в тексте — повод для круга исправления."""
    for rx in _FORBIDDEN_RES:
        m = rx.search(text)
        if m:
            return (
                f"Убери формулировку «{m.group(0)}»: рекомендация — это действие для пользователя, "
                "одна из трёх штатных фраз, а не характеристика компании."
            )
    return None


def scrub_forbidden(text: str, replacement: str | None) -> str:
    """После неудачного круга исправления категоричные фразы вычищает код."""
    out = text
    for rx in _FORBIDDEN_RES:
        out = rx.sub(replacement or "", out)
    return re.sub(r"[ \t]{2,}", " ", out)


_VERDICT_CODE_RE = re.compile(r"(?<![\w/])(not_recommended|check|ok)(?![\w/])")


def replace_verdict_codes(text: str) -> str:
    """Код исхода в тексте («ГДК — not_recommended») → штатная фраза."""
    return _VERDICT_CODE_RE.sub(lambda m: VERDICT_RU[Verdict(m.group(1))], text)


_MONTHS = ["янв", "фев", "мар", "апр", "ма", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
_TIDY = (  # слабая модель протаскивает в текст имена полей и повторы — чистим кодом
    (
        re.compile(r"^(\s*[-•]\s*)?claim:\s*(.+?)\s+source_path:\s*(report\.\S+)\s*$", re.M),
        r"\1\2 [\3]",
    ),
    (re.compile(r"\bverdict_ru:\s*", re.I), ""),
    (re.compile(r"\s*\((svetofor|zsk|labels?|verdict)\)", re.I), ""),
    (re.compile(r"^(\s*[-•]\s*)?(.{12,}?)\s+\(\2\)", re.M), r"\1\2"),  # «X (X)» → «X»
    (re.compile(r"\bBank ratings\b", re.I), "Оценки банка"),
    (re.compile(r"\bVerdict\b:?", re.I), "Вывод:"),
    # адреса полей живут в citations, в тексте пользователю их не показываем
    (re.compile(r"\s*\[\s*report\.[A-Za-z0-9_.]+(?:\[\d+\][A-Za-z0-9_.]*)*\s*\]"), ""),
    (re.compile(r"\s*\[\s*report\.[^\]\n]*$", re.M), ""),  # незакрытый адрес в конце строки
    # служебные ключи в скобках: [verdict_ru], [dfCount], [svetofor]
    (re.compile(r"\s*\[[A-Za-z_][A-Za-z0-9_.]*(?:\[\d+\][A-Za-z0-9_.]*)*\]"), ""),
    (re.compile(r"\s*\binn\s*[:=]?\s*\d{10,12}\b", re.I), ""),
    (re.compile(r"\bZSK\b"), "ЗСК"),
    # сильная модель подписывает каждую строку тяжестью сигнала — в интерфейсе это плашка
    (
        re.compile(
            r"^(\s*[-•*]?\s*)(?:критическ\w+|умеренн\w+|информационн\w+|значим\w+)\s+сигнал\s*[—–:-]\s*",
            re.M | re.I,
        ),
        r"\1",
    ),
    (re.compile(r"^(\s*[-•*]?\s*)(?:critical|moderate|info)\s*[—–:-]\s*", re.M | re.I), r"\1"),
    # эмодзи и значки: deepseek и подобные любят 🟢🔴 в тексте — в банковском ответе им не место
    (re.compile(r"[\U0001F300-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\uFE0F]"), ""),
    (
        re.compile(
            r"\b(\d{4})[\u2010\u2011\u2012\u2013-](\d{2})[\u2010\u2011\u2012\u2013-](\d{2})\b"
        ),
        r"\3.\2.\1",
    ),  # ISO → ДД.ММ.ГГГГ
    (  # «25 авг 2026 г.» → «25.08.2026»
        re.compile(
            r"\b(\d{1,2})\s+(янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек)\w*\s+(\d{4})(?:\s*(?:года|год|г\.?))?",
            re.I,
        ),
        lambda m: (
            f"{int(m.group(1)):02d}.{_MONTHS.index(m.group(2).lower()[:3].replace('май', 'ма').replace('мая', 'ма')) + 1:02d}.{m.group(3)}"
        ),
    ),
    (
        re.compile(
            r"\b(\d{2})[\u2010\u2011\u2012\u2013](\d{2})[\u2010\u2011\u2012\u2013](\d{4})\b"
        ),
        r"\1.\2.\3",
    ),
    (re.compile(r"\b(svetofor|traffic light)\b", re.I), "светофор"),
    # подсказка «зачем проверяете» живёт кнопками в интерфейсе — в тексте это дубль
    (re.compile(r"^.*Скажите, зачем проверяете.*$\n?", re.M | re.I), ""),
    # ниже — только после всех чисток, иначе схлопывать нечего
    (re.compile(r"[ \t]{2,}"), " "),  # двойные пробелы после удалённых кусков
    (re.compile(r"\s+([.,;:])"), r"\1"),  # пробел перед знаком
    (re.compile(r"([,;:])(?:\s*[,;:])+"), r"\1"),  # «нет данных,,» → «нет данных,»
    (re.compile(r",\s*\."), "."),
    (re.compile(r"\n{3,}"), "\n\n"),
    # замена запрещённой фразы штатной рекомендацией даёт повтор: «…документы только на условиях…»
    (re.compile(r"(\b[^\n]{25,}?)\s*[;,]?\s+\1"), r"\1"),
    (
        re.compile(
            r"(работать только на условиях: предоплата и подтверждающие документы)\s+только на условиях: предоплата и подтверждающие документы",
            re.I,
        ),
        r"\1",
    ),
)


_LINE_START = re.compile(
    r"^([\s>]*(?:[-•*]\s+|\d+[.)]\s+|#{1,6}\s+)?(?:\*\*|__)?)([а-яёa-z])", re.M
)


def capitalize_lines(text: str) -> str:
    """Модель пишет пункты вперемешку: «Запросить у контрагента…» и «работать только на
    условиях…». Первая буква строки — заглавная, маркеры списка и разметка не считаются."""
    return _LINE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)


def tidy_text(text: str) -> str:
    out = text
    for rx, repl in _TIDY:
        out = rx.sub(repl, out)
    for verdict in VERDICT_RU.values():
        phrase = re.escape(verdict)
        out = re.sub(rf"({phrase})(?:[\s;,.]+{phrase})+", r"\1", out, flags=re.I)
    return capitalize_lines(out.strip())


def drop_invalid_lines(text: str, invalid: list[Citation]) -> str:
    """После неудачных кругов исправления строки с неподтверждёнными числами убираются из текста:
    пользователь не должен видеть «7 152 200 %», которых нет в отчёте."""
    if not invalid:
        return text
    lines = text.split("\n")
    keep: list[str] = []
    for line in lines:
        hit = False
        for c in invalid:
            nums = {str(n) for n in numbers_in(c.claim)}
            inline = f"[{c.source_path}]" in line
            by_number = bool(nums) and bool(nums & {str(n) for n in numbers_in(line)})
            if (
                inline
                and (by_number or not nums)
                or (by_number and c.source_path.split(".")[-1] in line)
            ):
                hit = True
                break
        if hit and line.strip():
            continue
        keep.append(line)
    return "\n".join(keep)


def enforce_comparison(text: str, cards: list[Card]) -> str:
    """После неудачного круга исправления итог сравнения дописывается кодом."""
    text = replace_verdict_codes(text)
    lowered = text.casefold()
    missing = [c for c in cards if VERDICT_RU[c.verdict] not in lowered]
    if not missing:
        return text
    lines = [f"- {c.name} (ИНН {c.inn}): {VERDICT_RU[c.verdict]}" for c in missing]
    return f"{text.rstrip()}\n\n**По данным отчётов:**\n" + "\n".join(lines)


_VERDICT_CORE = {
    Verdict.OK: re.compile(r"можно работать|работать можно", re.I),
    Verdict.CHECK: re.compile(r"нужна дополнительная проверка", re.I),
    Verdict.NOT_RECOMMENDED: re.compile(r"(?:есть |выявлены )?существенные риски", re.I),
}


def verdict_present(text: str, verdict: Verdict) -> bool:
    """Вывод в тексте уже есть, даже если модель вплела его в предложение."""
    return bool(_VERDICT_CORE[verdict].search(text))


def enforce_verdict(text: str, verdict: Verdict) -> str:
    """После неудачного круга исправления вывод в тексте заменяется кодом."""
    expected = VERDICT_RU[verdict]
    out = text
    for other_verdict, other in VERDICT_RU.items():
        if other_verdict is verdict:
            continue
        if _VERDICT_CORE[other_verdict].search(out):
            out = re.compile(re.escape(other), re.IGNORECASE).sub(expected, out)
    if not verdict_present(out, verdict):
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
            claim=public_text(s["explanation"]),
            severity=Severity(s["severity"]),
            source_path=s["source_path"],
            code=s["code"],
            source_paths=s.get("source_paths", []),
        )
        # info тоже показываем: у компаний без рисков это единственное содержание карточки
        for severity in (Severity.CRITICAL, Severity.MODERATE, Severity.INFO)
        for s in data["signals"][severity.value]
    ]
    from contractor_agent.agent.requests import requested_documents

    asks = requested_documents(data, tools.get_financials(inn))
    return Card(
        inn=inn,
        name=public_text(info["short_name"]),
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


_FULL_CHECK_RE = re.compile(
    r"проверь\s+(?:ещё|еще|заново|полностью|целиком)|полн\w+\s+проверк|всю\s+карточк|"
    r"карточк\w*\s+(?:целиком|заново)|повтори\s+проверк",
    re.I,
)
FOLLOW_UP_HINT = (
    "Это продолжение разговора: пользователь уже видел проверку компании. Отвечай репликой на "
    "заданный вопрос, карточку целиком не повторяй. Если спрашивают о рекомендации или решении, "
    "назови вывод и объясни его фактами, которые относятся к вопросу. Переспроси только если "
    "без уточнения невозможно определить компанию или понять вопрос."
)


def is_follow_up(messages: list[BaseMessage]) -> bool:
    """Второй и следующие вопросы в сессии: пользователь уже получил ответ по компании.
    Служебные сообщения круга исправления за ход пользователя не считаются."""
    asked = [
        m for m in messages if isinstance(m, HumanMessage) and not m.additional_kwargs.get("repair")
    ]
    return len(asked) > 1


def visible_history(messages: list[BaseMessage], question: str) -> list[BaseMessage]:
    """Что видит модель: прошлые ходы — только текстом (без вызовов инструментов и их выдачи),
    текущий ход — целиком. Иначе на уточняющий вопрос модель копирует прошлую карточку
    и берёт числа из старых выдач, а правило «данные прошлых ходов не считаются» не работает."""
    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage) and messages[i].content == question
        ),
        0,
    )
    has_checked = any(m.additional_kwargs.get("final_answer") for m in messages[:start])
    past = [
        m
        for m in messages[:start]
        if isinstance(m, HumanMessage)
        or (
            isinstance(m, AIMessage)
            and not m.tool_calls
            and m.content
            and (not has_checked or m.additional_kwargs.get("final_answer"))
        )
    ]
    return [*past, *messages[start:]]


def finalization_history(messages: list[BaseMessage], question: str) -> list[BaseMessage]:
    """Finalize from current tool evidence, not previous answers or unverified drafts."""
    start = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if isinstance(messages[i], HumanMessage) and messages[i].content == question
        ),
        len(messages),
    )
    return [m for m in messages[start:] if not isinstance(m, AIMessage) or m.tool_calls]


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
