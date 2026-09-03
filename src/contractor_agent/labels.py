"""Метки банка словами — как их показывает сам банк (спецификация 3.09).

Светофор: LOW / MEDIUM / HIGH / UNKNOWN → зелёный / жёлтый / красный / серый.
ЗСК: клиенту показывают только зелёный и серый; жёлтый и красный не раскрываются.
Метки приводим как есть, не объясняем и не пересчитываем (инвариант 3).
"""

SVETOFOR_RU = {"LOW": "зелёный", "MEDIUM": "жёлтый", "HIGH": "красный", "UNKNOWN": "серый"}
ZSK_RU = {"GREEN": "зелёный"}


def svetofor_ru(level: str | None) -> str:
    return SVETOFOR_RU.get((level or "").upper(), "серый")


def zsk_ru(level: str | None) -> str:
    return ZSK_RU.get((level or "").upper(), "серый")
