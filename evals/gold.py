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

QuestionType = Literal["answer", "refuse", "infer", "card"]
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
