"""Эталон: карточки и вопросы трёх типов (ответ есть · ответа нет · нужен вывод).

Формат — `evals/gold.yaml`. Каждый вопрос знает, какой провал из `CRITERIA.md` §5
он ловит, какие подстроки обязан содержать правильный ответ и какие — не должен.
Для каждой карточки дополнительно генерируется вопрос-карточка «Проверь компанию»:
по нему считается пропуск критичного факта и совпадение вывода с разметкой.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal["answer", "refuse", "infer", "card", "comparison", "guard", "dialog"]
FinalType = Literal["answer", "refuse", "infer", "card"]
RANKING_WORDS = ["рейтинг", "балл", "первое место", "второе место", "третье место", "скоринг"]
Verdict = Literal["ok", "check", "not_recommended"]

GOLD_PATH = Path(__file__).parent / "gold.yaml"
FORBIDDEN_LABELS = [
    "ненадёжн",
    "сомнительн",
    "однодневк",
    "не рекомендуем работать",
    "не связывайтесь",
]
REFUSAL_MARKERS = [
    "нет сведений",
    "в отчёте нет",
    "оценить нельзя",
    "не содержит",
    "отсутству",
    "нет данных",
]


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    inn: str
    type: QuestionType
    question: str
    topic: str = "other"
    expected_value: str | float | int | None = None
    source_path: str | None = None
    must_mention: list[str] = Field(
        default_factory=list, description="подстроки; варианты через « | »"
    )
    must_not_mention: list[str] = Field(default_factory=list)
    expected_verdict: Verdict | None = None
    must_name: list[str] = Field(
        default_factory=list, description="для card: факты, которые обязан назвать"
    )
    catches: str = ""
    evidence: str = ""
    follow_up: bool = Field(
        default=False,
        description="уточняющий вопрос («у них…»): раннер засевает сессию репликой о компании",
    )
    category: str = Field(
        default="",
        description="раздел регресса: courts · enforcement · finance · registry · sections · "
        "refusal · decision · comparison · dialog · guard · edge · consistency",
    )
    extra_inns: list[str] = Field(
        default_factory=list, description="comparison: остальные компании (у каждой своя карточка)"
    )
    prior_questions: list[str] = Field(
        default_factory=list,
        description="dialog: предыдущие вопросы той же сессии, раннер задаёт их живьём по очереди",
    )
    final_type: FinalType | None = Field(
        default=None, description="dialog: как проверять последний ответ (answer/refuse/infer/card)"
    )
    expect_tools: bool | None = Field(
        default=None,
        description="guard: False — агент не должен вызывать инструменты (посторонний ввод)",
    )

    @property
    def effective_type(self) -> str:
        """Каким набором проверок мерить ответ: для dialog — типом последнего вопроса."""
        if self.type == "dialog":
            return self.final_type or "answer"
        return self.type

    @property
    def all_inns(self) -> list[str]:
        return [self.inn, *self.extra_inns]


class GoldCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inn: str
    company: str
    report_date: str
    expected_verdict: Verdict
    terminal: bool = False
    must_name: list[str] = Field(default_factory=list)
    questions: list[GoldQuestion] = Field(default_factory=list)

    def card_question(self) -> GoldQuestion:
        return GoldQuestion(
            id=f"{self.inn}-card",
            inn=self.inn,
            type="card",
            question=f"Проверь {self.company}, ИНН {self.inn}: можно ли с ней работать?",
            topic="card",
            expected_verdict=self.expected_verdict,
            must_name=self.must_name,
            must_not_mention=list(FORBIDDEN_LABELS),
            catches="missed_critical | label_on_counterparty | no_report_date",
        )


class Gold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cards: list[GoldCard]

    def questions(self, types: set[str] | None = None) -> list[GoldQuestion]:
        out: list[GoldQuestion] = []
        for card in self.cards:
            out.append(card.card_question())
            out.extend(q.model_copy(update={"inn": card.inn}) for q in card.questions)
        if types:
            out = [q for q in out if q.type in types]
        return out

    def card(self, inn: str) -> GoldCard:
        return next(c for c in self.cards if c.inn == inn)


def load_gold(path: Path = GOLD_PATH) -> Gold:
    return Gold.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
