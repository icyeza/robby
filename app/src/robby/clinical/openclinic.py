"""Reading an export from OpenClinic (the clinics' current patient record system).

ASSUMPTION: no real OpenClinic export was available when this was written. The column names
below (ALIASES) and the value spellings (the *_VALUES tables) are an informed guess at what a
maternity admission export looks like. When a clinic supplies a real export, correct those tables
and nothing else needs to change.

An OpenClinic sheet is translated into the app's own template layout and then goes through the
same checks as every other upload. Columns the app does not use are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime

from robby.clinical.xlsx_template import Column

# template field -> accepted OpenClinic headers (compared lowercase, spaces collapsed)
ALIASES: dict[str, tuple[str, ...]] = {
    "patient_number": ("patient id", "personid", "patient number", "file number"),
    "last_name": ("last name", "surname", "family name"),
    "first_name": ("first name", "given name"),
    "full_name": ("patient name", "name", "full name"),
    "admitted_at": ("admission date", "date of admission", "admission date/time"),
    "age": ("age", "age (years)", "maternal age"),
    "birth_date": ("date of birth", "birthdate", "dob"),
    "gestational_age_weeks": ("gestational age", "gestational age (weeks)", "ga (weeks)", "ga"),
    "parity": ("para", "parity"),
    "previous_cs_count": (
        "previous c-sections",
        "previous cesareans",
        "previous caesareans",
        "number of previous c-sections",
    ),
    "plurality": ("number of fetuses", "fetuses", "pregnancy type", "plurality"),
    "fetal_presentation": ("presentation", "fetal presentation", "foetal presentation"),
    "onset_of_labour": ("labour onset", "labor onset", "onset of labour", "onset of labor"),
    "bp": ("blood pressure", "bp"),
    "bp_sys": ("systolic", "systolic bp", "sbp"),
    "bp_dia": ("diastolic", "diastolic bp", "dbp"),
    "proteinuria": ("urine protein", "proteinuria", "urine dipstick protein"),
    "glucose": ("glucose", "blood glucose", "glucose (mmol/l)"),
    "glucose_mg": ("glucose (mg/dl)", "blood glucose (mg/dl)"),
}

PLURALITY_VALUES = {"1": "1", "singleton": "1", "single": "1", "2": "2", "twins": "2",
                    "twin": "2", "multiple": "2", "triplets": "2"}  # fmt: skip
PRESENTATION_VALUES = {
    "cephalic": "cephalic", "vertex": "cephalic", "head": "cephalic",
    "breech": "breech", "transverse": "transverse", "oblique": "oblique",
    "shoulder": "transverse",
}  # fmt: skip
ONSET_VALUES = {
    "spontaneous": "spontaneous", "natural": "spontaneous",
    "induced": "induced", "induction": "induced",
    "elective c-section": "prelabour_cs", "elective cesarean": "prelabour_cs",
    "no labour": "prelabour_cs", "no labor": "prelabour_cs", "prelabour cs": "prelabour_cs",
}  # fmt: skip
PROTEIN_VALUES = {
    "neg": "negative", "negative": "negative", "nil": "negative", "0": "negative",
    "trace": "trace", "+": "+", "1+": "+", "++": "++", "2+": "++", "+++": "+++", "3+": "+++",
}  # fmt: skip

MG_DL_PER_MMOL = 18.016
DATE_FORMATS = (
    "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%d-%m-%Y %H:%M", "%d-%m-%Y",
    "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
)  # fmt: skip
Problem = tuple[int, str, str]  # (Excel row, column, message), as importer.RowProblem


def _key(value: object) -> str:
    return " ".join(str(value).split()).lower()


def _sources(header: Sequence[object]) -> dict[str, int]:
    """Template field -> index of the OpenClinic column that supplies it."""
    wanted = {alias: field for field, aliases in ALIASES.items() for alias in aliases}
    found: dict[str, int] = {}
    for i, raw in enumerate(header):
        if raw is None:
            continue
        field = wanted.get(_key(raw))
        if field and field not in found:
            found[field] = i
    return found


def looks_like_openclinic(header: Sequence[object], cols: list[Column]) -> bool:
    """True when the header is not the app's template but matches several OpenClinic columns."""
    own = {_key(c.label) for c in cols}
    if any(_key(h) in own for h in header if h is not None):
        return False
    return len(_sources(header)) >= 4


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:  # any other ISO form, e.g. 2026-10-18T09:30:00.250 or with a UTC offset
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        return None


def _gestation(text: str) -> str | None:
    """Completed weeks from '39', '39.5', '39+2', '39w2d' or '39 weeks 2 days' (days dropped)."""
    m = re.fullmatch(
        r"(\d{1,2})(?:\s*(?:\+|w|wk|wks|weeks?)\s*(\d)?\s*(?:d|days?)?)?", text.lower()
    )
    if m:
        return m.group(1)
    try:
        return str(int(float(text)))
    except ValueError:
        return None


def _whole_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def convert(
    table: list[tuple[object, ...]], cols: list[Column]
) -> tuple[list[tuple[object, ...]], list[Problem]]:
    """The OpenClinic sheet as rows in the app's template layout, and any problems found.
    Row i of the result is row i of the input, so Excel row numbers in messages stay right."""
    header = table[0]
    src = _sources(header)
    shown = {f: " ".join(str(header[i]).split()) for f, i in src.items()}
    problems: list[Problem] = []
    need = [("patient_number", "Patient ID"), ("admitted_at", "Admission date")]
    for field, label in need:
        if field not in src:
            problems.append((1, label, f"The OpenClinic column “{label}” is missing."))
    if "full_name" not in src and not ("last_name" in src and "first_name" in src):
        problems.append((1, "Last name / First name", "The patient's name columns are missing."))
    if problems:
        return [], problems

    out_header = tuple(c.label for c in cols)
    position = {c.field: i for i, c in enumerate(cols)}
    out: list[tuple[object, ...]] = [out_header]
    for n, cells in enumerate(table[1:], start=2):
        if not any(_whole_text(c) for c in cells):
            out.append(tuple(None for _ in out_header))
            continue

        def get(field: str, cells: tuple[object, ...] = cells) -> str:
            i = src.get(field)
            return _whole_text(cells[i]) if i is not None and i < len(cells) else ""

        row: list[object] = [None] * len(out_header)

        def put(field: str, value: object, row: list[object] = row) -> None:
            row[position[field]] = value

        def bad(field: str, message: str, n: int = n) -> None:
            problems.append((n, shown.get(field, field), message))

        put("patient_number", get("patient_number"))
        name = get("full_name") or f"{get('first_name')} {get('last_name')}".strip()
        put("patient_name", name)

        when = _parse_datetime(cells[src["admitted_at"]]) if get("admitted_at") else None
        if get("admitted_at") and when is None:
            bad("admitted_at", "Not a date. Use day/month/year, e.g. 18/10/2026 09:30.")
        elif when is not None:
            put("admitted_at", when.strftime("%Y-%m-%d %H:%M"))

        if get("age"):
            put("maternal_age", get("age"))
        elif get("birth_date") and when is not None:
            born = _parse_datetime(cells[src["birth_date"]])
            if born is None:
                bad("birth_date", "Not a date. Use day/month/year.")
            else:
                before_birthday = (when.month, when.day) < (born.month, born.day)
                put("maternal_age", when.year - born.year - before_birthday)

        if get("gestational_age_weeks"):
            ga = _gestation(get("gestational_age_weeks"))
            if ga is None:
                bad("gestational_age_weeks", "Use weeks, e.g. 39, 39+2 or 39w2d.")
            else:
                put("gestational_age_weeks", ga)
        put("parity", get("parity"))
        put("previous_cs_count", get("previous_cs_count"))

        for field, values, what in (
            ("plurality", PLURALITY_VALUES, "number of fetuses"),
            ("fetal_presentation", PRESENTATION_VALUES, "presentation"),
            ("onset_of_labour", ONSET_VALUES, "onset of labour"),
        ):
            text = get(field)
            if not text:
                continue
            if _key(text) in values:
                stored = values[_key(text)]
                options = next(c.options for c in cols if c.field == field) or ()
                put(field, next((shown_ for v, shown_ in options if v == stored), text))
            else:
                bad(field, f"“{text}” is not a recognised {what}.")

        if get("bp"):
            m = re.fullmatch(r"(\d{2,3})\s*/\s*(\d{2,3})", get("bp"))
            if m:
                put("bp_sys", int(m.group(1)))
                put("bp_dia", int(m.group(2)))
            else:
                bad("bp", "Use systolic/diastolic, e.g. 120/80.")
        else:
            put("bp_sys", get("bp_sys"))
            put("bp_dia", get("bp_dia"))

        protein = get("proteinuria")
        if protein:
            if _key(protein) in PROTEIN_VALUES:
                stored = PROTEIN_VALUES[_key(protein)]
                options = next(c.options for c in cols if c.field == "proteinuria") or ()
                put("proteinuria", next((s for v, s in options if v == stored), protein))
            else:
                bad("proteinuria", f"“{protein}” is not a recognised dipstick result.")

        if get("glucose"):
            put("glucose", get("glucose"))
        elif get("glucose_mg"):
            try:
                put("glucose", round(float(get("glucose_mg")) / MG_DL_PER_MMOL, 1))
            except ValueError:
                bad("glucose_mg", "Not a number.")

        # The app requires a value or a reason for each screening test. Where OpenClinic holds
        # no value, record "Other" so the row can be imported; the clinician sees it on the result.
        for blank, reason_field, reason in (
            (not (row[position["bp_sys"]] and row[position["bp_dia"]]), "bp_reason", "Other"),
            (not row[position["proteinuria"]], "protein_reason", "Other"),
            (not row[position["glucose"]], "glucose_reason", "Other"),
        ):
            if blank:
                put(reason_field, reason)
                if reason_field == "bp_reason":
                    put("bp_sys", None)
                    put("bp_dia", None)
        out.append(tuple(row))
    return out, problems
