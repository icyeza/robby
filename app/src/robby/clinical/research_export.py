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
    return weeks if weeks is not None and 20 <= weeks <= 45 else None


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


def convert(
    table: list[tuple[object, ...]], cols: list[Column], facility_keys: set[str]
) -> tuple[list[tuple[object, ...]], list[Problem], dict[int, Outcome], int]:
    """The rows of the uploader's hospital in the template layout, the problems found, the
    outcome of each kept row (by Excel row number) and the number of other hospitals' rows left
    out. Rows of other hospitals become blank rows so row numbers in messages stay right."""
    header = table[0]
    at = {_key(h): i for i, h in reversed(list(enumerate(header))) if h is not None}
    wanted = {_key(k) for k in facility_keys if k}
    out_header = tuple(c.label for c in cols)
    position = {c.field: i for i, c in enumerate(cols)}
    labels = {c.field: dict(c.options or ()) for c in cols}
    details = {c.field for c in cols if c.field.startswith("d_")}
    out: list[tuple[object, ...]] = [out_header]
    problems: list[Problem] = []
    outcomes: dict[int, Outcome] = {}
    other = 0
    for n, cells in enumerate(table[1:], start=2):

        def raw(name: str, cells: tuple[object, ...] = cells) -> object:
            i = at.get(name)
            return cells[i] if i is not None and i < len(cells) else None

        def text(name: str) -> str:
            return _key(raw(name))

        blank = tuple(None for _ in out_header)
        if not any(c is not None and str(c).strip() for c in cells):
            out.append(blank)
            continue
        if text(HOSPITAL) not in wanted:
            other += 1
            out.append(blank)
            continue
        row: list[object] = [None] * len(out_header)

        def put(field: str, value: object, row: list[object] = row) -> None:
            row[position[field]] = value

        def shown(field: str, stored: str) -> str:
            return labels[field].get(stored, stored)

        upi = str(raw(UPI) or "").strip()
        if not upi:
            problems.append((n, "Patient UPI", "No patient UPI."))
            out.append(blank)
            continue
        put("patient_number", "UR-" + hashlib.sha256(upi.encode()).hexdigest()[:10].upper())
        put("patient_name", NAME)
        when = _when(raw(DELIVERY_DATE), raw(DELIVERY_TIME))
        if when is None:
            problems.append((n, "Date of Delivery", "Not a date."))
            out.append(blank)
            continue
        put("admitted_at", when.strftime("%Y-%m-%d %H:%M"))
        put("maternal_age", openclinic._whole_text(raw(COLUMNS["age"])))
        ga = _weeks(text(COLUMNS["ga"])) or GA_BANDS.get(text(GA_BAND))
        if ga is not None:
            put("gestational_age_weeks", ga)
        cs = _count(text(COLUMNS["cs"]))
        vaginal = _count(text(COLUMNS["vaginal"]))
        if cs is not None:
            put("previous_cs_count", cs)
        if cs is not None and vaginal is not None:
            put("parity", cs + vaginal)
        multiple = text(COLUMNS["multiple"])
        if multiple in ("yes", "no"):
            put("plurality", shown("plurality", "2" if multiple == "yes" else "1"))
        if text(COLUMNS["malpresentation"]) == "no":  # "yes" gives no type: left unknown
            put("fetal_presentation", shown("fetal_presentation", "cephalic"))
        onset = ONSET.get(text(COLUMNS["onset"]))
        if onset:
            put("onset_of_labour", shown("onset_of_labour", onset))
        for name, field, (low, high) in (
            ("height", "d_height_cm", (120, 200)),
            ("weight", "d_weight_kg", (30, 200)),
        ):
            try:
                value = float(openclinic._whole_text(raw(COLUMNS[name])))
            except ValueError:
                continue
            if field in details and low <= value <= high:  # an implausible value is left out
                put(field, openclinic._whole_text(raw(COLUMNS[name])))
        # The export holds no blood pressure, proteinuria or glucose.
        for reason_field in ("bp_reason", "protein_reason", "glucose_reason"):
            put(reason_field, "Other")
        mode = text(COLUMNS["mode"])
        if mode == "c-section":
            outcomes[n] = ("cesarean", when)
        elif mode == "vaginal":
            assisted = text(COLUMNS["instrumental"]) == "yes"
            outcomes[n] = ("vaginal_assisted" if assisted else "vaginal_spontaneous", when)
        out.append(tuple(row))
    return out, problems, outcomes, other
