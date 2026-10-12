"""Writer for the monthly aggregate export (FR10), format v1.

IMPORTANT: the national submission format has not been confirmed. The proposal only says the
export must be compatible with the monthly aggregate submissions facilities already make to the
national health information system. v1 is Robby's own documented layout and MUST be compared with
the facility's current submission (columns, codes, period notation, group naming) before it is
relied on. To change the layout edit this module and bump EXPORT_FORMAT_VERSION.

Layout (UTF-8, CRLF line ends, header row first):
    format_version,facility_code,period_month,robson_group,births,cesareans
One row per Robson group 1-10 (zeros are written), then one row with robson_group
`unclassified` for births without a resolved group. period_month is YYYY-MM. Counts only: no
patient identifiers, no rates."""

from __future__ import annotations

import csv
import io
from datetime import date

from robby.audit.counts import Counts, GroupCount

EXPORT_FORMAT_VERSION = "v1"
HEADER = ("format_version", "facility_code", "period_month", "robson_group", "births", "cesareans")


def filename(facility_code: str, month: date) -> str:
    return f"robby-monthly-{EXPORT_FORMAT_VERSION}_{facility_code}_{month:%Y-%m}.csv"


def render_monthly_csv(facility_code: str, month: date, counts: Counts) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(HEADER)
    period = f"{month:%Y-%m}"
    for key in [*range(1, 11), None]:
        gc = counts.get(key, GroupCount())
        label = "unclassified" if key is None else str(key)
        writer.writerow(
            [EXPORT_FORMAT_VERSION, facility_code, period, label, gc.births, gc.cesareans]
        )
    return out.getvalue().encode("utf-8")
