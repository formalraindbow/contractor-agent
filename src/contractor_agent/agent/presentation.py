"""Presentation rules shared by API answers and cards; evidence stays in citations."""

import re

from contractor_agent.data.model import Report
from contractor_agent.signals.flags import RULES as FLAG_RULES

PUBLIC_LABELS = {
    "finReports": "финансовая отчётность",
    "baseInfo": "сведения о компании",
    "executionProceedings": "исполнительные производства",
    "arbitrationByStatus": "сводка судебных дел",
    "reputationalRisks": "отмеченные риски",
    "currentAssets": "оборотные активы",
    "shortTermLiabilities": "краткосрочные обязательства",
    "riskLevel": "светофор банка",
    "zskRiskLevel": "ЗСК",
    **{key: rule.title_ru for key, rule in FLAG_RULES.items()},
    **{"flag_" + key: rule.title_ru for key, rule in FLAG_RULES.items()},
}
_MISSING_FINANCIALS = (
    "В отчёте нет финансовой отчётности: выручку, прибыль, капитал и ликвидность оценить нельзя."
)
PUBLIC_PHRASES = {
    "Вердикт:": "Рекомендация помощника:",
    (
        "Раздел «Финансовая отчётность» в отчёте есть, но данных в нём нет — "
        "оценить финансы по отчёту нельзя."
    ): _MISSING_FINANCIALS,
    (
        "В отчёте нет раздела «Финансовая отчётность» — оценить выручку, прибыль "
        "и устойчивость по этому критерию нельзя."
    ): _MISSING_FINANCIALS,
    "Раздел «Финансовая отчётность» пуст": "Финансовой отчётности в отчёте нет",
}
PUBLIC_ANNOTATIONS = ["HIGH", "MEDIUM", "LOW", "UNKNOWN", "critical", "moderate", "info"]


def _field_names(value: object) -> set[str]:
    if isinstance(value, dict):
        own = {key for key in value.get("properties", {}) if re.search(r"[A-Z_]", key)}
        return own.union(*(_field_names(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_field_names(child) for child in value))
    return set()


PUBLIC_TERMS = sorted(
    _field_names(Report.model_json_schema())
    | PUBLIC_LABELS.keys()
    | {"source_path", "source_paths", "verdict_ru", "signal_counts", "signals", "gaps"},
    key=lambda word: (-len(word), word),
)
_TERM_PATTERN = r"\b(?:" + "|".join(map(re.escape, PUBLIC_TERMS)) + r"|[a-zA-Z]+(?:_[a-zA-Z]+)+)\b"
_TERM = re.compile(_TERM_PATTERN, re.I)
_LABELS = {key.casefold(): value for key, value in PUBLIC_LABELS.items()}
_URL = re.compile(r"https?://[^\s<>)]*", re.I)
_INTERNAL = re.compile(
    r"\b(?:report|labels|data|computed)\.[A-Za-z_]"
    r"|\b(?:source_paths?|verdict_ru|signal_counts|riskLevel|zskRiskLevel|gaps?|signals)\b"
    r"|\b(?:" + "|".join(PUBLIC_ANNOTATIONS) + r")\b"
    r"|" + _TERM_PATTERN,
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
    urls = []

    def keep_url(match):
        urls.append(match[0])
        return f"⟪{len(urls) - 1}⟫"

    text = _URL.sub(keep_url, text)
    for old, new in PUBLIC_PHRASES.items():
        text = text.replace(old, new)
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
    text = re.sub(
        r"`?\b(?:"
        + "|".join(map(re.escape, PUBLIC_TERMS))
        + r")(?:\[\d+\])?(?:\.[A-Za-z_]\w*(?:\[\d+\])?)+`?",
        "",
        text,
    )
    text = _TERM.sub(lambda match: _LABELS.get(match[0].casefold(), ""), text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Removing a bold internal annotation can leave empty emphasis after a fact.
    text = re.sub(r"(?<=\S)[ \t]+\*{4}(?=[ \t]*(?:\n|$))", "", text)
    text = re.sub(r"\s*\((?:defendant|plaintiff)\)", "", text, flags=re.I)
    text = re.sub(r"\b(?:source_paths?|verdict_ru|signal_counts)\s*[:=]?", "", text)
    text = re.sub(r"\b[ZЗ](?:SK|СК)\b", "ЗСК", text, flags=re.I)
    text = re.sub(r"(?:Статус\s*(?:компании)?\s*[—:=-]\s*)?\bCURRENT\b[;,.]?", "", text)
    text = re.sub(r"\bFINISHED\b", "завершено", text)
    text = re.sub(r"\bIN_PROGRESS\b", "в производстве", text)
    text = re.sub(r"\bо оборотных\b", "об оборотных", text, flags=re.I)
    text = re.sub(r"(?<=[а-яёА-ЯЁ])[ \t]+[-‐‑‒–][ \t]+(?=\S)", " — ", text)
    text = re.sub(r'"([^"\n]+)"', r"«\1»", text)
    # Fragments left by malformed annotations must not become visible prose.
    text = re.sub(r"\b(?:путь|поле)\s*`?\s*`?(?=[).,;]|$)", "", text, flags=re.M | re.I)
    text = re.sub(r"\([\s.,;:=]*\)|\[[\s.,;:=]*\]|``", "", text)
    text = re.sub(
        r"(?m)^[-•] \*\*Количество:\*\* ([^\n]+)\n[-•] \*\*Сумма:\*\* ([^\n]+)",
        r"- \1; сумма — \2",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    for i, url in enumerate(urls):
        text = text.replace(f"⟪{i}⟫", url)
    return text.strip()


def has_internal_text(text: str) -> bool:
    text = _URL.sub("", text)
    return bool(_INTERNAL.search(text) or re.search(r"\bCURRENT\b", text))
