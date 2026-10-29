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
from dataclasses import dataclass
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
UNMEASURED_REASON = "Other"  # the "not measured" reason recorded when the export has no value


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


def _missing_columns(src: dict[str, int]) -> list[Problem]:
    """Row-1 problems: the columns no admission can be read without."""
    problems: list[Problem] = []
    for field, label in (("patient_number", "Patient ID"), ("admitted_at", "Admission date")):
        if field not in src:
            problems.append((1, label, f"The OpenClinic column “{label}” is missing."))
    if "full_name" not in src and not ("last_name" in src and "first_name" in src):
        problems.append((1, "Last name / First name", "The patient's name columns are missing."))
    return problems


@dataclass(frozen=True)
class _Sheet:
    """Where each template field comes from in the OpenClinic sheet and goes to in the output."""

    src: dict[str, int]  # template field -> OpenClinic column index
    shown: dict[str, str]  # template field -> the OpenClinic header, as problems name it
    position: dict[str, int]  # template field -> output column index
    options: dict[str, tuple[tuple[str, str], ...]]  # template field -> (value, label) choices
    problems: list[Problem]  # filled while the rows are translated


class _Row:
    """One OpenClinic row being written into the template layout."""

    def __init__(self, sheet: _Sheet, n: int, cells: tuple[object, ...]) -> None:
        self.sheet = sheet
        self.n = n
        self.cells = cells
        self.out: list[object] = [None] * len(sheet.position)

    def raw(self, field: str) -> object:
        i = self.sheet.src.get(field)
        return self.cells[i] if i is not None and i < len(self.cells) else None

    def get(self, field: str) -> str:
        return _whole_text(self.raw(field))

    def put(self, field: str, value: object) -> None:
        self.out[self.sheet.position[field]] = value

    def written(self, field: str) -> object:
        return self.out[self.sheet.position[field]]

    def bad(self, field: str, message: str) -> None:
        self.sheet.problems.append((self.n, self.sheet.shown.get(field, field), message))

    def put_choice(self, field: str, stored: str, text: str) -> None:
        """The option's label for a stored value (the template sheet holds labels)."""
        label = next((s for v, s in self.sheet.options[field] if v == stored), text)
        self.put(field, label)


def _admission_time(row: _Row) -> datetime | None:
    if not row.get("admitted_at"):
        return None
    when = _parse_datetime(row.raw("admitted_at"))
    if when is None:
        row.bad("admitted_at", "Not a date. Use day/month/year, e.g. 18/10/2026 09:30.")
    else:
        row.put("admitted_at", when.strftime("%Y-%m-%d %H:%M"))
    return when


def _maternal_age(row: _Row, when: datetime | None) -> None:
    """The age column, else the age on the admission date from the date of birth."""
    if row.get("age"):
        row.put("maternal_age", row.get("age"))
    elif row.get("birth_date") and when is not None:
        born = _parse_datetime(row.raw("birth_date"))
        if born is None:
            row.bad("birth_date", "Not a date. Use day/month/year.")
        else:
            before_birthday = (when.month, when.day) < (born.month, born.day)
            row.put("maternal_age", when.year - born.year - before_birthday)


def _gestational_age(row: _Row) -> None:
    if not row.get("gestational_age_weeks"):
        return
    ga = _gestation(row.get("gestational_age_weeks"))
    if ga is None:
        row.bad("gestational_age_weeks", "Use weeks, e.g. 39, 39+2 or 39w2d.")
    else:
        row.put("gestational_age_weeks", ga)


def _coded_choices(row: _Row) -> None:
    """Fetuses, presentation and onset, from OpenClinic's spellings."""
    for field, values, what in (
        ("plurality", PLURALITY_VALUES, "number of fetuses"),
        ("fetal_presentation", PRESENTATION_VALUES, "presentation"),
        ("onset_of_labour", ONSET_VALUES, "onset of labour"),
    ):
        text = row.get(field)
        if not text:
            continue
        if _key(text) in values:
            row.put_choice(field, values[_key(text)], text)
        else:
            row.bad(field, f"“{text}” is not a recognised {what}.")


def _blood_pressure(row: _Row) -> None:
    """One "120/80" column, or separate systolic and diastolic columns."""
    if not row.get("bp"):
        row.put("bp_sys", row.get("bp_sys"))
        row.put("bp_dia", row.get("bp_dia"))
        return
    m = re.fullmatch(r"(\d{2,3})\s*/\s*(\d{2,3})", row.get("bp"))
    if m:
        row.put("bp_sys", int(m.group(1)))
        row.put("bp_dia", int(m.group(2)))
    else:
        row.bad("bp", "Use systolic/diastolic, e.g. 120/80.")


def _proteinuria(row: _Row) -> None:
    protein = row.get("proteinuria")
    if not protein:
        return
    if _key(protein) in PROTEIN_VALUES:
        row.put_choice("proteinuria", PROTEIN_VALUES[_key(protein)], protein)
    else:
        row.bad("proteinuria", f"“{protein}” is not a recognised dipstick result.")


def _glucose(row: _Row) -> None:
    """mmol/L as given, else converted from mg/dL."""
    if row.get("glucose"):
        row.put("glucose", row.get("glucose"))
    elif row.get("glucose_mg"):
        try:
            row.put("glucose", round(float(row.get("glucose_mg")) / MG_DL_PER_MMOL, 1))
        except ValueError:
            row.bad("glucose_mg", "Not a number.")


def _unmeasured_screening(row: _Row) -> None:
    """The app requires a value or a reason for each screening test. Where OpenClinic holds no
    value, record "Other" so the row can be imported; the clinician sees it on the result."""
    if not (row.written("bp_sys") and row.written("bp_dia")):
        row.put("bp_reason", UNMEASURED_REASON)
        row.put("bp_sys", None)
        row.put("bp_dia", None)
    if not row.written("proteinuria"):
        row.put("protein_reason", UNMEASURED_REASON)
    if not row.written("glucose"):
        row.put("glucose_reason", UNMEASURED_REASON)


def _translate(row: _Row) -> tuple[object, ...]:
    row.put("patient_number", row.get("patient_number"))
    name = row.get("full_name") or f"{row.get('first_name')} {row.get('last_name')}".strip()
    row.put("patient_name", name)
    when = _admission_time(row)
    _maternal_age(row, when)
    _gestational_age(row)
    row.put("parity", row.get("parity"))
    row.put("previous_cs_count", row.get("previous_cs_count"))
    _coded_choices(row)
    _blood_pressure(row)
    _proteinuria(row)
    _glucose(row)
    _unmeasured_screening(row)
    return tuple(row.out)


def convert(
    table: list[tuple[object, ...]], cols: list[Column]
) -> tuple[list[tuple[object, ...]], list[Problem]]:
    """The OpenClinic sheet as rows in the app's template layout, and any problems found.
    Row i of the result is row i of the input, so Excel row numbers in messages stay right."""
    header = table[0]
    src = _sources(header)
    problems = _missing_columns(src)
    if problems:
        return [], problems
    sheet = _Sheet(
        src=src,
        shown={f: " ".join(str(header[i]).split()) for f, i in src.items()},
        position={c.field: i for i, c in enumerate(cols)},
        options={c.field: c.options or () for c in cols},
        problems=problems,
    )
    out_header = tuple(c.label for c in cols)
    out: list[tuple[object, ...]] = [out_header]
    for n, cells in enumerate(table[1:], start=2):
        if not any(_whole_text(c) for c in cells):
            out.append(tuple(None for _ in out_header))
        else:
            out.append(_translate(_Row(sheet, n, cells)))
    return out, problems
