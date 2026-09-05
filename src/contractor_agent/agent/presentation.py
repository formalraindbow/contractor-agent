"""Presentation rules shared by API answers and cards; evidence stays in citations."""

import re

_INTERNAL = re.compile(
    r"\b(?:report|labels|data|computed)\.[A-Za-z_]"
    r"|\b(?:source_paths?|verdict_ru|signal_counts|riskLevel|zskRiskLevel|gaps?|signals)\b"
    r"|\b[a-zA-Z]+(?:_[a-zA-Z]+)+\b",
    re.I,
)
_BRACKET = re.compile(r"\[(?:[^\[\]\n]|\[\d+\])*\]")
_PAREN = re.compile(r"\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)")
_PATH = re.compile(
    r"`?(?:report|labels|data|computed)\.[A-Za-z_][A-Za-z0-9_.]*(?:\[\d+\][A-Za-z0-9_.]*)*`?"
)
_BANNER = re.compile(
    r"^.*(?:Убрано утверждений|Часть утверждений не прошла проверку|"
    r"Часть запрошенных сведений не удалось подтвердить|готово за \d+).*$",
    re.M | re.I,
)
_REPORT_DATE_LINE = re.compile(
    r"(?im)^[ \t]*(?:[-•] )?(?:\*\*)?(?:Отч[её]т|Дата отч[её]та)"
    r"(?: по ИНН \d{10,12})?\s*(?:от|:)\s*"
    r"(?:\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})[.]?(?:\*\*)?[ \t]*$"
)


def comparison_text(text: str) -> str:
    """Report dates are structured metadata, not repeated paragraphs in a comparison."""
    return re.sub(r"\n{3,}", "\n\n", _REPORT_DATE_LINE.sub("", text)).strip()


def public_text(text: str) -> str:
    """Remove technical annotations as a whole, including nested array indices.

    Run after extracting and checking citations, so presentation cannot erase evidence.
    Ordinary brackets (dates, company names, explanations) remain intact.
    """
    text = _BANNER.sub("", text)
    text = re.sub(r"стоит проверить до договора", "нужна дополнительная проверка", text, flags=re.I)
    text = re.sub(r"`(?:report|critical|moderate|info)`", "", text)
    text = re.sub(
        r"работать только на условиях: предоплата и подтверждающие документы",
        "есть существенные риски",
        text,
        flags=re.I,
    )
    text = re.sub(r"(?m)[ \t]*\\[ \t]*$", "", text)
    text = re.sub(r"(?i)\b(вывод|рекомендация) банка\b", r"\1 помощника", text)
    # Remove malformed, unclosed annotations before deleting their keywords.
    text = re.sub(
        r"\([^\n)]*(?:source_paths?|verdict_ru|значение берётся из поля)[^\n)]*\)?", "", text
    )
    for pattern in (_PAREN, _BRACKET):
        text = pattern.sub(
            lambda m: "" if _INTERNAL.search(m[0]) and not m[0].startswith("(https://") else m[0],
            text,
        )
    text = re.sub(r"【[^】\n]*】", "", text)
    text = _PATH.sub("", text)
    # Removing a bold internal annotation can leave empty emphasis after a fact.
    text = re.sub(r"(?<=\S)[ \t]+\*{4}(?=[ \t]*(?:\n|$))", "", text)
    text = re.sub(r"\s*\((?:defendant|plaintiff)\)", "", text, flags=re.I)
    text = re.sub(r"\b(?:source_paths?|verdict_ru|signal_counts)\s*[:=]?", "", text)
    text = re.sub(r"\b[ZЗ](?:SK|СК)\b", "ЗСК", text, flags=re.I)
    text = re.sub(r"(?:Статус\s*(?:компании)?\s*[—:=-]\s*)?\bCURRENT\b[;,.]?", "", text)
    text = re.sub(r"\bFINISHED\b", "завершено", text)
    text = re.sub(r"\bIN_PROGRESS\b", "в производстве", text)
    text = re.sub(r'"([^"\n]+)"', r"«\1»", text)
    # Fragments left by malformed annotations must not become visible prose.
    text = re.sub(r"\b(?:путь|поле)\s*`?\s*`?(?=[).,;]|$)", "", text, flags=re.M | re.I)
    text = re.sub(r"\(\s*[.,;:=]*\s*\)|\[\s*\]|``", "", text)
    text = re.sub(
        r"(?m)^[-•] \*\*Количество:\*\* ([^\n]+)\n[-•] \*\*Сумма:\*\* ([^\n]+)",
        r"- \1; сумма — \2",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_internal_text(text: str) -> bool:
    return bool(_INTERNAL.search(text) or re.search(r"\bCURRENT\b", text))
