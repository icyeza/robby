"""Automated import: POST /api/v1/admissions/import, for a script that sends a clinic's records
(for example nightly from OpenClinic). It authenticates with a bearer token (see
robby.auth.api_tokens) instead of a browser session, so it has no cookie and no CSRF token.

The body is one of:
  * JSON: {"rows": [{"Patient ID": "...", ...}, ...]} (or the bare list), with the same column
    names as an OpenClinic or template spreadsheet;
  * the .xlsx file itself (Content-Type application/vnd.openxmlformats-officedocument...);
  * multipart/form-data with the file in the field "file".

Checking is all-or-nothing, as for the web upload. A record whose patient number and admission
time were already recorded is skipped, not an error, so sending a batch twice is safe."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

from robby import events
from robby.auth import api_tokens
from robby.auth.dependencies import DbDep
from robby.clinical import classify as cls
from robby.clinical import importer, service
from robby.clinical.routes import XLSX_TYPE, runtime_of
from robby.clinical.routes import _specs as specs_of
from robby.timeutil import utcnow

router = APIRouter(prefix="/api/v1")


def _problems(items: list[importer.RowProblem], shift: int = 0) -> list[dict[str, Any]]:
    return [{"row": max(p.row - shift, 0), "column": p.column, "message": p.message} for p in items]


def _reply(status: int, body: dict[str, Any]) -> JSONResponse:
    return JSONResponse(body, status_code=status)


def _json_table(payload: object) -> list[tuple[object, ...]] | str:
    """The JSON batch as a header row plus rows, or a message saying what is wrong with it."""
    records = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        return 'Send {"rows": [ ... ]} with at least one record.'
    if not all(isinstance(r, dict) for r in records):
        return "Every record must be an object of column name to value."
    header: list[str] = []
    for record in records:
        header += [k for k in record if k not in header]
    return [tuple(header), *[tuple(r.get(k) for k in header) for r in records]]


def _percent(db: Any, admission: Any) -> int | None:
    predicted = service.prediction_of(db, admission)
    return service.percent_of(predicted[0].probability) if predicted else None


@router.post("/admissions/import")
async def import_admissions(request: Request, db: DbDep) -> Response:
    now = utcnow()
    user = api_tokens.authenticate(db, request.headers.get("authorization"), now)
    if user is None:
        return JSONResponse(
            {"error": "Missing, invalid or revoked API token."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    db.commit()  # keeps last_used_at
    assert user.facility_id is not None
    specs = specs_of(request, db)
    zone = request.app.state.settings.zone
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    shift = 0
    if content_type == "application/json":
        data = await request.body()
        try:
            payload = await request.json()
        except ValueError:
            return _reply(400, {"error": "The body is not valid JSON."})
        table = _json_table(payload)
        if isinstance(table, str):
            return _reply(400, {"error": table})
        read = importer.read_table(table, specs, zone, now)
        shift = 1  # JSON records are numbered from 1; the header would be row 1
    elif content_type == "multipart/form-data":
        upload = (await request.form()).get("file")
        if not isinstance(upload, UploadFile):
            return _reply(400, {"error": 'Send the spreadsheet in the form field "file".'})
        data = await upload.read()
        read = importer.read_upload(data, upload.filename or "upload.xlsx", specs, zone, now)
    elif content_type == XLSX_TYPE:
        data = await request.body()
        read = importer.read_upload(data, "api.xlsx", specs, zone, now)
    else:
        return _reply(
            415, {"error": "Send application/json, an .xlsx body or multipart/form-data."}
        )

    facility_code = user.facility.code if user.facility else ""
    if read.problems:
        events.append(
            db,
            user,
            "api_import.rejected",
            ("facility", user.facility_id),
            {"facility": facility_code, "problems": len(read.problems), "channel": "api"},
        )
        db.commit()
        return _reply(422, {"imported": 0, "problems": _problems(read.problems, shift)})

    scan = importer.find_duplicates(db, user.facility_id, read.rows)
    skipped = {p.row for p in scan.in_file} | set(scan.recorded)
    fresh = [(n, parsed) for n, parsed in read.rows if n not in skipped]
    saved: list[service.Saved] = []
    if fresh:
        try:
            saved = service.import_admissions(
                db, runtime_of(request), user, fresh, hashlib.sha256(data).hexdigest(), "api"
            )
            db.commit()
        except cls.NoRuleSetError:
            db.rollback()
            return _reply(503, {"error": "No Robson rule set is active, so nothing can be saved."})
    if not saved:
        events.append(
            db,
            user,
            "api_import.nothing_new",
            ("facility", user.facility_id),
            {"facility": facility_code, "skipped": len(skipped), "channel": "api"},
        )
        db.commit()
    return _reply(
        201 if saved else 200,
        {
            "imported": len(saved),
            "skipped_duplicates": _problems(scan.in_file, shift),
            "already_recorded": [
                {
                    "row": n - shift,
                    "code": a.code,
                    "robson_group": service.classification_of(db, a).robson_group,
                    "readiness_percent": _percent(db, a),
                }
                for n, a in scan.recorded.items()
            ],
            "admissions": [
                {
                    "row": n - shift,
                    "code": s.admission.code,
                    "robson_group": s.classification.robson_group,
                    "readiness_percent": (
                        service.percent_of(s.prediction.probability) if s.prediction else None
                    ),
                }
                for (n, _), s in zip(fresh, saved, strict=True)
            ],
        },
    )
