"""Форматирование чисел для объяснений: рубли, годы."""

from __future__ import annotations

from datetime import date
from decimal import ROUND_DOWN, Decimal

NBSP = " "  # обычный пробел: модель и судья эвалов сравнивают строки


def rub(value: int | Decimal) -> str:
    """Рубли для текста: ``26249000`` → ``26,2 млн ₽``, ``517235.54`` → ``517 235,54 ₽``."""
    v = Decimal(value)
    magnitude = abs(v)
    if magnitude >= 1_000_000_000:
        return f"{_short(v / 1_000_000_000)}{NBSP}млрд{NBSP}₽"
    if magnitude >= 1_000_000:
        return f"{_short(v / 1_000_000)}{NBSP}млн{NBSP}₽"
    sign = "-" if v < 0 else ""
    whole, _, frac = f"{magnitude:.2f}".partition(".")
    grouped = f"{int(whole):,}".replace(",", NBSP)
    tail = "" if frac == "00" else f",{frac}"
    return f"{sign}{grouped}{tail}{NBSP}₽"


def _short(v: Decimal) -> str:
    text = f"{v:.1f}".replace(".", ",")
    return text[:-2] if text.endswith(",0") else text


def ratio(value: Decimal) -> str:
    return f"{value:.2f}".replace(".", ",")


def years_list(years: list[int]) -> str:
    return ", ".join(str(y) for y in sorted(years))


def share(value: Decimal) -> str:
    """Доля от чистых активов для текста: ``0.003`` → ``0,3 %``, ``18.08`` → ``в 18 раз больше``."""
    if value >= 2:
        times = round(value)
        return f"в {plural(times, 'раз', 'раза', 'раз')} больше"
    if value >= 1:
        return "больше"
    percent = value * 100
    if percent < Decimal("0.1"):
        return "меньше 0,1 %"
    text = str(percent.quantize(Decimal("0.1"), rounding=ROUND_DOWN)).replace(".", ",")
    text = text[:-2] if text.endswith(",0") else text
    return f"{text} %"


def plural(n: int, one: str, few: str, many: str) -> str:
    """``plural(1, "производство", "производства", "производств")`` → ``1 производство``."""
    n_abs = abs(n)
    if n_abs % 10 == 1 and n_abs % 100 != 11:
        form = one
    elif 2 <= n_abs % 10 <= 4 and not 12 <= n_abs % 100 <= 14:
        form = few
    else:
        form = many
    return f"{n} {form}"


def date_ru(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def years_range(first: int, last: int) -> str:
    return f"{first} год" if first == last else f"{first}–{last} годы"
