# ruff: noqa: E501
"""LLM-судья по рубрике `docs/CRITERIA.md` §5.

Провалы обнуляют: пропущен критичный факт · назван факт, которого нет в данных ·
ответ по существу, когда данных нет. Снижения: числа по судам без роли, поля
и периода (−2) · метка пересчитана или объяснена (−2) · нет даты отчёта (−1) ·
вердикт вместо условий (−1) · ложная тревога (−1) · ярлык на контрагента (−1).
Судья видит вопрос, эталон и то, что агент получил от инструментов, — иначе
он не может отличить выдумку от факта.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from contractor_agent.agent.llm import LLM
from contractor_agent.agent.schema import Answer
from evals.gold import GoldQuestion

Failure = Literal["missed_critical", "invented_fact", "answered_without_data"]
Deduction = Literal[
    "courts_without_role_period_field",
    "label_recomputed_or_explained",
    "no_report_date",
    "verdict_instead_of_conditions",
    "false_alarm",
    "label_on_counterparty",
]


class JudgeVerdict(BaseModel):
    score: int = Field(ge=0, le=5)
    failures: list[Failure] = Field(default_factory=list)
    deductions: list[Deduction] = Field(default_factory=list)
    invented_facts: list[str] = Field(
        default_factory=list, description="утверждения без опоры в данных"
    )
    comment: str = Field(description="одно-два предложения по-русски")


JUDGE_PROMPT = """Ты судья ответов агента, который проверяет контрагента по отчёту банка.
Оцени ОДИН ответ по рубрике. Начинай с 5 баллов.

Провалы — оценка 0, независимо от остального:
- missed_critical: в ответе нет критичного факта — из эталона (expected_value / must_name; варианты через «или» — достаточно любого одного, в любой форме, синонимом и округлении: «ЕГРЮЛ» = «реестр юридических лиц», «конкурсное производство» = «банкротство») или из данных инструментов (банкротство, ликвидация, исключение из реестра, блокировка счетов, недостоверность сведений), когда вопрос про компанию в целом;
- invented_fact: в тексте ответа назван факт или число, которого нет ни в данных инструментов, ни в эталоне, ни в карточке;
- answered_without_data: вопрос типа refuse (данных нет), а агент ответил по существу вместо «в отчёте нет сведений — оценить нельзя».

Снижения (если провалов нет):
- courts_without_role_period_field (−2): названо число судебных дел БЕЗ роли (истец/ответчик) И БЕЗ статуса или периода («открытые», «завершённые», год — это статус/период) И без поля-источника (в тексте или в списке цитат); если роль и статус названы — снижения нет;
- label_recomputed_or_explained (−2): агент объяснил, ПОЧЕМУ метка банка такая, спорил с ней или пересчитал её. Назвать метку как есть, без объяснения, — правильно, снижения нет (расшифровать шкалу — тоже можно);
- no_report_date (−1): в ответе нет даты отчёта;
- verdict_instead_of_conditions (−1): категоричная оценка компании или готовое разрешение — «работать нельзя», «не рекомендуем», «надёжная/ненадёжная компания», «можно работать», «есть существенные риски». Формулировки о доступных сведениях снижением НЕ считаются: «существенных факторов риска в отчёте не выявлено», «нужна дополнительная проверка», «в отчёте есть факты, требующие особого внимания»;
- false_alarm (−1): названы риски, которых нет в данных инструментов;
- label_on_counterparty (−1): «ненадёжная», «сомнительная», «однодневка», «не рекомендуем работать».

Общие прогнозы об оспаривании сделок и неисполнении обязательств не заменяют конкретный статус из отчёта. Не оценивай тон, вежливость и длину. Числа считай совпавшими при разумном округлении (26,2 млн ₽ = 26 249 000). Снижение применяй только если нарушение есть в тексте буквально; сомневаешься — не снижай. Комментарий пиши по-русски."""


def judge_messages(question: GoldQuestion, answer: Answer, tool_outputs: str) -> list:
    # must_mention / must_not_mention — для проверок кодом; судье их не показываем,
    # иначе он обнуляет за форму слова («истец» vs «подавала»)
    gold = question.model_dump(
        exclude={"evidence", "must_mention", "must_not_mention"}, exclude_none=True
    )
    for key in ("must_name",):
        if key in gold:  # « | » — варианты для проверки кодом; судье показываем словами
            gold[key] = [" или ".join(v.strip() for v in item.split(" | ")) for item in gold[key]]
    card_line = (
        "Карточка (собрана кодом из полей отчёта, факты в ней подтверждены программно — "
        f"выдумкой не считать): {answer.card.model_dump(mode='json')}\n"
        if answer.card
        else ""
    )
    user = (
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question.question}\n\n"
        f"ЭТАЛОН (JSON):\n{gold}\n\n"
        f"ЧТО АГЕНТ ПОЛУЧИЛ ОТ ИНСТРУМЕНТОВ (обрезано):\n{tool_outputs[:40000]}\n\n"
        f"ОТВЕТ АГЕНТА:\n{answer.text_md}\n\n"
        f"{card_line}"
        f"Цитаты, прошедшие проверку кодом (утверждение → поле отчёта): "
        f"{[c.model_dump() for c in answer.citations]}\n"
        f"Цитаты, не прошедшие проверку: {[c.model_dump() for c in answer.invalid_citations]}"
    )
    return [SystemMessage(content=JUDGE_PROMPT), HumanMessage(content=user)]


async def judge(
    llm: LLM, question: GoldQuestion, answer: Answer, tool_outputs: str
) -> JudgeVerdict:
    chain = llm.structured(JudgeVerdict)
    result = await chain.ainvoke(judge_messages(question, answer, tool_outputs))
    return result if isinstance(result, JudgeVerdict) else JudgeVerdict.model_validate(result)
