"""Reading the UR-CMHS research export (KoboToolbox "labels" download, one row per delivery).

The export covers several hospitals and past deliveries, so it is read differently from a ward
upload: only the rows of the uploader's hospital are kept (matched on the facility's name or its
model facility key), the delivery date stands in for the admission time (the export has none), and
the mode of delivery is returned so the outcome can be recorded with the admission. The column
mapping follows ML/configs/mapping_ur_cmhs.yaml.

The export holds a national patient identifier (UPI) and no name. The UPI is never stored: the
patient number is a one-way hash of it ("UR-" + 10 hex characters), so the same woman gets the
same number on every upload, and the name is "Research record".
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time

from robby.clinical import openclinic
from robby.clinical.xlsx_template import Column

Problem = tuple[int, str, str]  # (Excel row, column, message), as importer.RowProblem
Outcome = tuple[str, datetime]  # (DeliveryMode value, delivered at, local time)

UPI = "patient upi"
HOSPITAL = "hospital name"
DELIVERY_DATE = "date of delivery"
DELIVERY_TIME = "time of delivery"
SIGNATURE = (UPI, HOSPITAL, DELIVERY_DATE, "labor onset", "mode of delivery")
COLUMNS = {
    "age": "maternal age",
    "vaginal": "number of previous vaginal delivery",
    "cs": "number of previous c-sections",
    "multiple": "multiple pregnancy",
    "malpresentation": "fetal malpresentation",
    "onset": "labor onset",
    "ga": "actual gestational week at delivery",
    "mode": "mode of delivery",
    "instrumental": "instrumental delivery",
    "height": "maternal height",
    "weight": "maternal weight",
}
TOP_CODES = {"4 or more": 4, "4 and above": 4}
ONSET = {"spontaneous": "spontaneous", "induced": "induced", "planned c-section": "prelabour_cs"}
NAME = "Research record"
GA_BAND = "gestational age at time of labor"
# Used only when the exact week is missing. "35 - 37 Weeks" straddles the 37-week Robson cut, so
# it is left unknown; the others are on one side of it.
GA_BANDS = {"less than 34 weeks": 33, "38 - 40 weeks": 39, "over 40 weeks": 41}
GA_WEEKS = (20, 45)  # plausible completed weeks at delivery; anything else is left unknown
HEIGHT_CM = (120, 200)
WEIGHT_KG = (30, 200)


def _key(value: object) -> str:
    return " ".join(str(value).split()).lower() if value is not None else ""


def looks_like_research_export(header: Sequence[object]) -> bool:
    keys = {_key(h) for h in header}
    return all(s in keys for s in SIGNATURE)


def _count(text: str) -> int | None:
    if text in TOP_CODES:
        return TOP_CODES[text]
    try:
        return int(float(text))
    except ValueError:
        return None


def _weeks(text: str) -> int | None:
    """Completed weeks from free text such as '39', '39weeks, 2days' or '39+2'."""
    m = re.match(r"\s*(\d{1,2})(?!\d)", text)
    weeks = int(m.group(1)) if m else None
    low, high = GA_WEEKS
    return weeks if weeks is not None and low <= weeks <= high else None


def _when(day: object, clock: object) -> datetime | None:
    if isinstance(day, datetime):
        day = day.date()
    if not isinstance(day, date):
        parsed = openclinic._parse_datetime(day) if day else None
        if parsed is None:
            return None
        day = parsed.date()
    at = time(0, 0)
    if isinstance(clock, time):
        at = clock
    elif clock:
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p"):
            try:
                at = datetime.strptime(str(clock).strip(), fmt).time()
                break
            except ValueError:
                continue
    return datetime.combine(day, at.replace(second=0, microsecond=0))


@dataclass(frozen=True)
class _Layout:
    """Where each export column is, and where each template field goes."""

    at: dict[str, int]  # export header (compared lowercase) -> column index
    position: dict[str, int]  # template field -> output column index
    labels: dict[str, dict[str, str]]  # template field -> stored value -> label
    details: set[str]  # the optional model inputs the template has


class _Record:
    """One delivery of the export being written into the template layout."""

    def __init__(self, layout: _Layout, cells: tuple[object, ...]) -> None:
        self.layout = layout
        self.cells = cells
        self.out: list[object] = [None] * len(layout.position)

    def raw(self, name: str) -> object:
        i = self.layout.at.get(name)
        return self.cells[i] if i is not None and i < len(self.cells) else None

    def text(self, name: str) -> str:
        return _key(self.raw(name))

    def put(self, field: str, value: object) -> None:
        self.out[self.layout.position[field]] = value

    def put_choice(self, field: str, stored: str) -> None:
        self.put(field, self.layout.labels[field].get(stored, stored))


def _robson_inputs(record: _Record) -> None:
    """Gestational age, previous births, fetuses, presentation and onset, where the export says."""
    ga = _weeks(record.text(COLUMNS["ga"])) or GA_BANDS.get(record.text(GA_BAND))
    if ga is not None:
        record.put("gestational_age_weeks", ga)
    cs = _count(record.text(COLUMNS["cs"]))
    vaginal = _count(record.text(COLUMNS["vaginal"]))
    if cs is not None:
        record.put("previous_cs_count", cs)
    if cs is not None and vaginal is not None:
        record.put("parity", cs + vaginal)
    multiple = record.text(COLUMNS["multiple"])
    if multiple in ("yes", "no"):
        record.put_choice("plurality", "2" if multiple == "yes" else "1")
    if record.text(COLUMNS["malpresentation"]) == "no":  # "yes" gives no type: left unknown
        record.put_choice("fetal_presentation", "cephalic")
    onset = ONSET.get(record.text(COLUMNS["onset"]))
    if onset:
        record.put_choice("onset_of_labour", onset)


def _body_size(record: _Record) -> None:
    """Height and weight, when the template asks for them; an implausible value is left out."""
    for name, field, (low, high) in (
        ("height", "d_height_cm", HEIGHT_CM),
        ("weight", "d_weight_kg", WEIGHT_KG),
    ):
        text = openclinic._whole_text(record.raw(COLUMNS[name]))
        try:
            value = float(text)
        except ValueError:
            continue
        if field in record.layout.details and low <= value <= high:
            record.put(field, text)


def _outcome(record: _Record, when: datetime) -> Outcome | None:
    mode = record.text(COLUMNS["mode"])
    if mode == "c-section":
        return ("cesarean", when)
    if mode == "vaginal":
        assisted = record.text(COLUMNS["instrumental"]) == "yes"
        return ("vaginal_assisted" if assisted else "vaginal_spontaneous", when)
    return None


def _translate(
    record: _Record, n: int
) -> tuple[tuple[object, ...], None, Outcome | None] | tuple[None, Problem, None]:
    """The delivery as a template row and its outcome, or the problem that leaves it out."""
    upi = str(record.raw(UPI) or "").strip()
    if not upi:
        return None, (n, "Patient UPI", "No patient UPI."), None
    record.put("patient_number", "UR-" + hashlib.sha256(upi.encode()).hexdigest()[:10].upper())
    record.put("patient_name", NAME)
    when = _when(record.raw(DELIVERY_DATE), record.raw(DELIVERY_TIME))
    if when is None:
        return None, (n, "Date of Delivery", "Not a date."), None
    record.put("admitted_at", when.strftime("%Y-%m-%d %H:%M"))
    record.put("maternal_age", openclinic._whole_text(record.raw(COLUMNS["age"])))
    _robson_inputs(record)
    _body_size(record)
    # The export holds no blood pressure, proteinuria or glucose.
    for reason_field in ("bp_reason", "protein_reason", "glucose_reason"):
        record.put(reason_field, openclinic.UNMEASURED_REASON)
    return tuple(record.out), None, _outcome(record, when)


def convert(
    table: list[tuple[object, ...]], cols: list[Column], facility_keys: set[str]
) -> tuple[list[tuple[object, ...]], list[Problem], dict[int, Outcome], int]:
    """The rows of the uploader's hospital in the template layout, the problems found, the
    outcome of each kept row (by Excel row number) and the number of other hospitals' rows left
    out. Rows of other hospitals become blank rows so row numbers in messages stay right."""
    layout = _Layout(
        at={_key(h): i for i, h in reversed(list(enumerate(table[0]))) if h is not None},
        position={c.field: i for i, c in enumerate(cols)},
        labels={c.field: dict(c.options or ()) for c in cols},
        details={c.field for c in cols if c.field.startswith("d_")},
    )
    wanted = {_key(k) for k in facility_keys if k}
    blank = tuple(None for _ in cols)
    out: list[tuple[object, ...]] = [tuple(c.label for c in cols)]
    problems: list[Problem] = []
    outcomes: dict[int, Outcome] = {}
    other = 0
    for n, cells in enumerate(table[1:], start=2):
        record = _Record(layout, cells)
        if not any(c is not None and str(c).strip() for c in cells):
            out.append(blank)
            continue
        if record.text(HOSPITAL) not in wanted:
            other += 1
            out.append(blank)
            continue
        row, problem, outcome = _translate(record, n)
        if problem is not None:
            problems.append(problem)
        if outcome is not None:
            outcomes[n] = outcome
        out.append(blank if row is None else row)
    return out, problems, outcomes, other
