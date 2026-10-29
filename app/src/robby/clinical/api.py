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
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

from robby import events
from robby.auth import api_tokens
from robby.auth.dependencies import DbDep
from robby.clinical import classify as cls
from robby.clinical import importer, service
from robby.clinical.form import DetailSpec, Parsed
from robby.clinical.routes import XLSX_TYPE, runtime_of
from robby.clinical.routes import _specs as specs_of
from robby.models import AppUser
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


@dataclass(frozen=True)
class _Body:
    read: importer.ReadResult
    data: bytes  # the bytes received, hashed into the import event
    shift: int  # subtracted from Excel row numbers in the reply


async def _read_body(
    request: Request, specs: list[DetailSpec], zone: ZoneInfo, now: datetime
) -> _Body | JSONResponse:
    """The records in whichever of the three accepted bodies was sent, or the error reply."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/json":
        data = await request.body()
        try:
            payload = await request.json()
        except ValueError:
            return _reply(400, {"error": "The body is not valid JSON."})
        table = _json_table(payload)
        if isinstance(table, str):
            return _reply(400, {"error": table})
        # JSON records are numbered from 1; the header would be row 1
        return _Body(importer.read_table(table, specs, zone, now), data, shift=1)
    if content_type == "multipart/form-data":
        upload = (await request.form()).get("file")
        if not isinstance(upload, UploadFile):
            return _reply(400, {"error": 'Send the spreadsheet in the form field "file".'})
        data = await upload.read()
        filename = upload.filename or "upload.xlsx"
        return _Body(importer.read_upload(data, filename, specs, zone, now), data, shift=0)
    if content_type == XLSX_TYPE:
        data = await request.body()
        return _Body(importer.read_upload(data, "api.xlsx", specs, zone, now), data, shift=0)
    return _reply(415, {"error": "Send application/json, an .xlsx body or multipart/form-data."})


def _facility_event(db: DbDep, user: AppUser, event: str, counts: dict[str, int]) -> None:
    """An api_import.* event on the user's facility, committed at once."""
    assert user.facility_id is not None
    facility_code = user.facility.code if user.facility else ""
    payload = {"facility": facility_code, **counts, "channel": "api"}
    events.append(db, user, event, ("facility", user.facility_id), payload)
    db.commit()


def _imported_reply(
    db: DbDep,
    shift: int,
    scan: importer.DuplicateScan,
    fresh: list[tuple[int, Parsed]],
    saved: list[service.Saved],
) -> JSONResponse:
    summaries = service.summaries_of(db, list(scan.recorded.values()))
    return _reply(
        201 if saved else 200,
        {
            "imported": len(saved),
            "skipped_duplicates": _problems(scan.in_file, shift),
            "already_recorded": [
                {
                    "row": n - shift,
                    "code": a.code,
                    "robson_group": summaries[a.id][0],
                    "readiness_percent": summaries[a.id][1],
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
    body = await _read_body(request, specs_of(request, db), request.app.state.settings.zone, now)
    if isinstance(body, JSONResponse):
        return body
    read, shift = body.read, body.shift
    if read.problems:
        _facility_event(db, user, "api_import.rejected", {"problems": len(read.problems)})
        return _reply(422, {"imported": 0, "problems": _problems(read.problems, shift)})

    scan = importer.find_duplicates(db, user.facility_id, read.rows)
    skipped = {p.row for p in scan.in_file} | set(scan.recorded)
    fresh = [(n, parsed) for n, parsed in read.rows if n not in skipped]
    saved: list[service.Saved] = []
    if fresh:
        digest = hashlib.sha256(body.data).hexdigest()
        try:
            saved = service.import_admissions(db, runtime_of(request), user, fresh, digest, "api")
            db.commit()
        except cls.NoRuleSetError:
            db.rollback()
            return _reply(503, {"error": cls.NOTHING_SAVED})
    if not saved:
        _facility_event(db, user, "api_import.nothing_new", {"skipped": len(skipped)})
    return _imported_reply(db, shift, scan, fresh, saved)
