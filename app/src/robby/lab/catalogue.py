"""The tests the lab can be asked for, their names on screen, and how a result is read from the
lab technician's form and written back in words. A result is stored as text (encrypted):
the proteinuria option value of the admission form, mmol/L with at most one decimal for
glucose, positive / negative / indeterminate for HIV."""

from __future__ import annotations

from decimal import Decimal

from robby.clinical import labels
from robby.clinical.form import DECIMAL_1, GLUCOSE
from robby.models import LabTest

TEST_NAMES: dict[LabTest, str] = {
    LabTest.URINE_PROTEIN: "Urine protein (dipstick)",
    LabTest.BLOOD_GLUCOSE: "Blood glucose",
    LabTest.HIV_RAPID: "HIV rapid test",
}
TEST_ORDER = (LabTest.URINE_PROTEIN, LabTest.BLOOD_GLUCOSE, LabTest.HIV_RAPID)
HIV_OPTIONS = (
    ("negative", "Negative"),
    ("positive", "Positive"),
    ("indeterminate", "Indeterminate"),
)
# The readiness model's input an HIV result fills in, and the result that means "yes" / "no".
HIV_INPUT = "hiv_positive"
HIV_ANSWER = {"positive": True, "negative": False}
# Which screening line of the patient page a result completes (HIV is not a screening test).
SCREENING_KEY: dict[LabTest, str] = {
    LabTest.URINE_PROTEIN: "protein",
    LabTest.BLOOD_GLUCOSE: "glucose",
}
NOTE_MAX = 200
REASON_MAX = 200


def test_name(test: LabTest) -> str:
    return TEST_NAMES[test]


def options_of(test: LabTest) -> tuple[tuple[str, str], ...]:
    """The choices for a test answered by choosing (none for glucose, which is typed)."""
    if test is LabTest.URINE_PROTEIN:
        return labels.PROTEINURIA_OPTIONS
    if test is LabTest.HIV_RAPID:
        return HIV_OPTIONS
    return ()


def parse_result(test: LabTest, raw: str) -> tuple[str | None, str | None]:
    """(stored value, None) or (None, the error to show next to the field)."""
    text = raw.strip()
    if test is LabTest.BLOOD_GLUCOSE:
        low, high = GLUCOSE
        if not text:
            return None, "Enter the glucose result in mmol/L."
        if not DECIMAL_1.fullmatch(text) or not low <= Decimal(text) <= high:
            return (
                None,
                f"Glucose must be between {low} and {high} mmol/L, with at most one decimal.",
            )
        return str(Decimal(text)), None
    if text not in dict(options_of(test)):
        what = "urine protein" if test is LabTest.URINE_PROTEIN else "HIV test"
        return None, f"Choose the {what} result."
    return text, None


def result_text(test: LabTest, value: str) -> str:
    """A stored result in words, with its unit."""
    if test is LabTest.BLOOD_GLUCOSE:
        return f"{value} mmol/L"
    return dict(options_of(test)).get(value, value)
