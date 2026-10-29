# audit

The hospital report (FR9), the monthly aggregate export (FR10) and the national overview. Only
aggregates leave this module, except the facility administrator's own list of unclassified
births.

- `counts.py`: grouped counts in SQL. A birth is a current admission with a delivery outcome in
  the period; its group is that of its newest classification, anything unresolved is the
  residual.
- `measures.py`: the audit arithmetic as pure functions: group size, cesarean rate, absolute
  contribution, share of cesareans, and the expected rate for the case mix against a reference.
- `references.py`, `cli_references.py`: versioned reference populations (Vogel, C-Model) loaded
  from CSV with `robby-admin references load`.
- `narrative.py`, `plain.py`: "what stands out" sentences and plain-language verdicts, from
  fixed rules over the same thresholds as `measures`.
- `service.py`: page context shared by the facility and national pages, the reporting period,
  and the record of each export.
- `status.py`: monthly export status (exported, due, late).
- `export_format.py`, `pdf.py`, `formatting.py`: the CSV export, the PDF reports and number
  formatting for templates.
- `routes.py`: facility administrator pages; `national.py`: national reviewer pages.
