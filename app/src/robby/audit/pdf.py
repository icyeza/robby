"""PDF versions of the hospital report and the national overview, for reading, printing and
sending on. Built from the same context as the web pages (totals only, never an admission).
The CSV exports stay the format for the national monthly submission."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from robby.audit import formatting as fmt
from robby.audit.measures import MIN_BIRTHS_FOR_RATE_FLAG, RATE_FLAG_POINTS, Report
from robby.audit.plain import in_100

BRAND = colors.HexColor("#7b1e3a")
BRAND_TINT = colors.HexColor("#f8eef1")
INK_3 = colors.HexColor("#77707a")
LINE = colors.HexColor("#e4e1e3")
TRACK = colors.HexColor("#f1eef0")
VERDICT_COLOURS = {
    "high": colors.HexColor("#7b1e3a"),
    "low": colors.HexColor("#4d3373"),
    "same": colors.HexColor("#2e6b3f"),
    "muted": INK_3,
}
WIDTH = A4[0] - 30 * mm

_base = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=_base["Title"], fontSize=20, leading=24, alignment=0)
H2 = ParagraphStyle("h2", parent=_base["Heading2"], fontSize=13, textColor=BRAND, spaceBefore=10)
BODY = ParagraphStyle("b", parent=_base["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10.5, textColor=INK_3)
HEAD = ParagraphStyle("hd", parent=BODY, fontSize=12, leading=16)
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.5, leading=11)
CELL_R = ParagraphStyle("cr", parent=CELL, alignment=TA_RIGHT)


def _p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def _e(text: object) -> str:
    return escape(str(text))


def _bar(width_pct: int, size: float = 70 * mm) -> Drawing:
    d = Drawing(size, 8)
    d.add(Rect(0, 0, size, 8, fillColor=TRACK, strokeColor=None))
    d.add(Rect(0, 0, size * max(width_pct, 0) / 100, 8, fillColor=BRAND, strokeColor=None))
    return d


def _verdict(v: Any) -> Paragraph:
    colour = VERDICT_COLOURS.get(v.css, INK_3).hexval()[2:]
    return _p(f'<font color="#{colour}"><b>{_e(v.label)}</b></font>', CELL)


def _table(rows: list[list[Any]], widths: list[float]) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("TEXTCOLOR", (0, 0), (-1, 0), INK_3),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _tiles(items: list[tuple[str, str, str]]) -> Table:
    cells = [
        [_p(f'<font size="8" color="#77707a">{_e(k)}</font>', BODY) for k, _, _ in items],
        [_p(f'<font size="18"><b>{_e(v)}</b></font>', HEAD) for _, v, _ in items],
        [_p(_e(s), SMALL) for _, _, s in items],
    ]
    t = Table(cells, colWidths=[WIDTH / len(items)] * len(items))
    t.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, -1), 0.5, LINE),
                ("BOX", (1, 0), (1, -1), 0.5, LINE),
                ("BOX", (2, 0), (2, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def _stands_out(items: list[tuple[str, str]]) -> list[Any]:
    bullets = ListFlowable(
        [ListItem(_p(f"<b>{_e(lead)}</b>{_e(rest)}"), leftIndent=10) for lead, rest in items],
        bulletType="bullet",
        start="•",
        leftIndent=10,
    )
    box = Table([[bullets]], colWidths=[WIDTH])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_TINT),
                ("LINEBEFORE", (0, 0), (0, -1), 3, BRAND),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return [_p("What stands out", H2), box]


def _header(title: str, ctx: dict[str, Any], demo: bool, zone: Any) -> list[Any]:
    period = ctx["period"]
    ref = ctx.get("reference")
    made = datetime.now(zone).strftime("%d %b %Y, %H:%M")
    parts = [_p(_e(title), TITLE)]
    line = f"{fmt.day(period.start)} to {fmt.day(period.end)}"
    if ref is not None:
        line += f" · compared with {_e(ref.name)} ({_e(ref.citation)})"
    parts.append(_p(line, SMALL))
    parts.append(_p(f"Made by Robson Readiness on {made}. Totals only; no patient details.", SMALL))
    if demo:
        parts.append(
            _p("<b>Example data:</b> every figure comes from synthetic admissions.", SMALL)
        )
    parts.append(Spacer(1, 6))
    return parts


def _build(story: list[Any], title: str) -> bytes:
    out = BytesIO()

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(INK_3)
        canvas.drawString(15 * mm, 10 * mm, title)
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        out,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Robson Readiness",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()


def _hospital_summary(report: Report) -> list[Any]:
    """The headline sentence and the three tiles."""
    story: list[Any] = []
    if report.births:
        story.append(
            _p(
                f"<b>{fmt.count(report.births)}</b> women gave birth here and "
                f"<b>{fmt.count(report.cesareans)}</b> had a cesarean: {in_100(report.cs_rate)}.",
                HEAD,
            )
        )
        story.append(Spacer(1, 6))
    story.append(
        _tiles(
            [
                ("Births", fmt.count(report.births), "With the delivery recorded"),
                ("Cesareans", fmt.count(report.cesareans), f"{in_100(report.cs_rate)} births"),
                ("Waiting for an outcome", fmt.count(report.awaiting), "Not counted yet"),
            ]
        )
    )
    return story


def _sources_table(ctx: dict[str, Any]) -> list[Any]:
    """Where the cesareans come from, largest group first (nothing when there are none)."""
    if not ctx["sources"]:
        return []
    rows: list[list[Any]] = [["Group", "Women in this group", "", "Cesareans"]]
    for s in ctx["sources"]:
        rows.append(
            [
                _p(f"<b>{s.group}</b>", CELL),
                _p(_e(ctx["plain_groups"][s.group]), CELL),
                _bar(s.width, 45 * mm),
                _p(f"<b>{s.cesareans}</b> · {fmt.pct(s.share)} of all", CELL_R),
            ]
        )
    return [
        KeepTogether(
            [
                _p("Where the cesareans come from", H2),
                _table(rows, [14 * mm, WIDTH - 104 * mm, 50 * mm, 40 * mm]),
            ]
        )
    ]


def _groups_table(ctx: dict[str, Any]) -> list[Any]:
    """Each Robson group against the reference, then the births with no group."""
    report = ctx["report"]
    rows = [["Group", "Women in this group", "Births", "Cesareans", "Compared with the reference"]]
    for r in report.rows:
        v = ctx["verdicts"][r.group]
        cs = f"{r.cesareans} of {r.births}<br/>{in_100(r.rate_pct)}" if r.births else "—"
        rows.append(
            [
                _p(f"<b>{r.group}</b>", CELL),
                _p(_e(ctx["plain_groups"][r.group]), CELL),
                _p(fmt.count(r.births), CELL_R),
                _p(cs, CELL),
                _verdict(v),
            ]
        )
    u = report.residual
    reasons = ctx.get("reasons") or []
    why = "No group: details were missing or did not fit together"
    if reasons:
        why += "<br/>Most often: " + ", ".join(f"{_e(lbl.lower())} ({n})" for lbl, n in reasons)
    rows.append(
        [
            _p("—", CELL),
            _p(why, CELL),
            _p(fmt.count(u.births), CELL_R),
            _p(f"{u.cesareans} of {u.births}" if u.births else "—", CELL),
            _p("Check the records" if report.unclassified_high else "Counted in the totals", CELL),
        ]
    )
    return [
        _p("Each Robson group", H2),
        _table(rows, [14 * mm, WIDTH - 104 * mm, 16 * mm, 30 * mm, 44 * mm]),
        Spacer(1, 4),
        _p(
            '"Higher" or "Lower" means the group\'s cesarean rate differs from the reference by '
            f"{int(RATE_FLAG_POINTS)} or more in 100. Groups with fewer than "
            f"{MIN_BIRTHS_FOR_RATE_FLAG} births are not judged.",
            SMALL,
        ),
    ]


def hospital_pdf(ctx: dict[str, Any], *, demo: bool, zone: Any) -> bytes:
    name = ctx["facility"].name
    story = _header(f"{name} · Hospital report", ctx, demo, zone)
    story += _hospital_summary(ctx["report"])
    story += _stands_out(ctx["stands_out"])
    story += _sources_table(ctx)
    story += _groups_table(ctx)
    return _build(story, f"{name} · Hospital report")


def national_pdf(ctx: dict[str, Any], *, demo: bool, zone: Any) -> bytes:
    story = _header("National overview", ctx, demo, zone)
    if ctx["births"]:
        story.append(
            _p(
                f"<b>{fmt.count(ctx['births'])}</b> women gave birth at {ctx['reporting']} "
                f"hospitals and <b>{fmt.count(ctx['cesareans'])}</b> had a cesarean: "
                f"{in_100(ctx['cs_rate'])}.",
                HEAD,
            )
        )
        story.append(Spacer(1, 6))
    story.append(
        _tiles(
            [
                (
                    "Hospitals reporting",
                    f"{ctx['reporting']} of {ctx['total_facilities']}",
                    "With births in this period",
                ),
                ("Births", fmt.count(ctx["births"]), "With the delivery recorded"),
                ("Cesareans", fmt.count(ctx["cesareans"]), f"{in_100(ctx['cs_rate'])} births"),
            ]
        )
    )
    story += _stands_out(ctx["stands_out"])
    rows: list[list[Any]] = [["Hospital", "Births", "Cesareans", "Compared with expected"]]
    for line in ctx["lines"]:
        v = ctx["verdicts"][line.facility.id]
        rep = line.report
        cs = f"{rep.cesareans} of {rep.births}<br/>{in_100(rep.cs_rate)}" if rep.births else "—"
        expected = line.comparison.expected
        verdict = _verdict(v)
        cell: Any = verdict
        if expected is not None:
            cell = [verdict, _p(f"Expected: {in_100(expected)}", SMALL)]
        name = f"<b>{_e(line.facility.name)}</b>"
        if line.unclassified_high:
            name += f"<br/>{fmt.pct(rep.unclassified_pct)} of births have no group"
        rows.append([_p(name, CELL), _p(fmt.count(rep.births), CELL_R), _p(cs, CELL), cell])
    story.append(_p("Each hospital", H2))
    story.append(_table(rows, [WIDTH - 110 * mm, 20 * mm, 40 * mm, 50 * mm]))
    story.append(Spacer(1, 4))
    story.append(
        _p(
            '"Expected" allows for the mix of women each hospital sees. "More" or "Fewer than '
            'expected" means at least 10% away from that rate.',
            SMALL,
        )
    )
    return _build(story, "National overview")
