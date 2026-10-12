"""Text formatting for the report templates (passed to templates as `fmt`)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

MINUS = chr(0x2212)  # true minus sign


def pct(value: Decimal | None) -> str:
    return "—" if value is None else f"{value}%"


def count(value: int) -> str:
    return f"{value:,}"


def signed(value: Decimal | None, unit: str = " pts") -> str:
    """'+3.2 pts' / '-8.0 pts' with U+2212 (true minus sign); the sign is always printed."""
    if value is None:
        return "no ref."
    sign = "+" if value >= 0 else MINUS
    return f"{sign}{abs(value)}{unit}"


def signed_whole(value: Decimal | None) -> str:
    if value is None:
        return "—"
    rounded = int(value.quantize(Decimal(1)))
    return f"+{rounded}" if rounded >= 0 else f"{MINUS}{abs(rounded)}"


def day(value: date) -> str:
    return value.strftime("%d %b %Y")


def month_name(value: date) -> str:
    return value.strftime("%b %Y")
