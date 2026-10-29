"""A stand-in hospital FHIR server for trying pull mode. NOT a real FHIR server: a few invented
patients, held in memory, and only the searches the app makes.

    uv run uvicorn --app-dir scripts/fhir_standin server:app --port 8091
    ROBBY_FHIR_BASE_URL=http://127.0.0.1:8091 uv run uvicorn robby.main:create_app --factory

Patient numbers to type on the admission form (see README.md):
  FX-2001  complete record                    FX-2002  glucose not measured, HIV recorded
  FX-2003  some values the app cannot use     FX-2004  discharged: no admission in progress
  SLOW-1   answers after 10 s (timeout)       FAIL-1   answers HTTP 500
Admission times are a few hours before the server started, so the form accepts them."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
ROBBY = "https://icyeza.github.io/robby/fhir/CodeSystem/admission"
UCUM = "http://unitsofmeasure.org"
ABSENT = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
FHIR_JSON = "application/fhir+json"
Json = dict[str, Any]


def _obs(oid: str, enc: str, system: str, code: str, **value: Any) -> Json:
    return {"resourceType": "Observation", "id": oid, "status": "final",
            "code": {"coding": [{"system": system, "code": code}]},
            "encounter": {"reference": f"Encounter/{enc}"}, **value}  # fmt: skip


def _qty(value: float, code: str) -> Json:
    return {"valueQuantity": {"value": value, "system": UCUM, "code": code}}


def _concept(code: str) -> Json:
    return {"valueCodeableConcept": {"coding": [{"system": ROBBY, "code": code}]}}


def _bp(oid: str, enc: str, sys_: int, dia: int) -> Json:
    def part(code: str, v: int) -> Json:
        return {"code": {"coding": [{"system": LOINC, "code": code}]}, **_qty(v, "mm[Hg]")}

    return _obs(oid, enc, LOINC, "85354-9", component=[part("8480-6", sys_), part("8462-4", dia)])


def _patient(pid: str, number: str, given: str, family: str, born: str) -> Json:
    return {"resourceType": "Patient", "id": pid,
            "identifier": [{"system": "urn:example:hospital:patient-number", "value": number}],
            "name": [{"use": "official", "family": family, "given": [given]}],
            "birthDate": born}  # fmt: skip


def _encounter(eid: str, pid: str, start: datetime, status: str = "in-progress") -> Json:
    return {"resourceType": "Encounter", "id": eid, "status": status,
            "subject": {"reference": f"Patient/{pid}"},
            "period": {"start": start.isoformat(timespec="seconds")}}  # fmt: skip


def build(now: datetime) -> list[Json]:
    """The stand-in's records, admitted a few hours before ``now``."""
    ago = lambda hours: now - timedelta(hours=hours)  # noqa: E731
    return [
        _patient("p1", "FX-2001", "Aline", "Uwimana", "1997-05-02"),
        _encounter("e1", "p1", ago(2)),
        _obs("e1-par", "e1", LOINC, "11977-6", valueInteger=0),
        _obs("e1-ga", "e1", LOINC, "49051-6", **_qty(39, "wk")),
        _obs("e1-cs", "e1", ROBBY, "previous-cesareans", valueInteger=0),
        _obs("e1-fet", "e1", ROBBY, "fetuses", valueInteger=1),
        _obs("e1-pres", "e1", ROBBY, "fetal-presentation", **_concept("cephalic")),
        _obs("e1-on", "e1", ROBBY, "labour-onset", **_concept("spontaneous")),
        _bp("e1-bp", "e1", 116, 74),
        _obs("e1-pro", "e1", LOINC, "20454-5", valueString="negative"),
        _obs("e1-glu", "e1", LOINC, "15074-8", **_qty(4.9, "mmol/L")),
        _patient("p2", "FX-2002", "Grace", "Mukamana", "1990-11-20"),
        _encounter("e2", "p2", ago(5)),
        _obs("e2-par", "e2", LOINC, "11977-6", valueInteger=2),
        _obs("e2-ga", "e2", LOINC, "49051-6", **_qty(266, "d")),
        _obs("e2-cs", "e2", ROBBY, "previous-cesareans", valueInteger=1),
        _obs("e2-fet", "e2", ROBBY, "fetuses", valueInteger=1),
        _obs("e2-pres", "e2", ROBBY, "fetal-presentation", **_concept("cephalic")),
        _bp("e2-bp", "e2", 138, 88),
        _obs("e2-pro", "e2", LOINC, "20454-5", valueString="trace"),
        _obs("e2-glu", "e2", LOINC, "15074-8",
             dataAbsentReason={"coding": [{"system": ABSENT, "code": "asked-declined"}]}),  # fmt: skip
        {"resourceType": "Condition", "id": "c2-hiv", "subject": {"reference": "Patient/p2"},
         "code": {"coding": [{"system": SNOMED, "code": "86406008"}]}},  # fmt: skip
        _patient("p3", "FX-2003", "Diane", "Uwase", "2001-01-09"),
        _encounter("e3", "p3", ago(1)),
        _obs("e3-par", "e3", LOINC, "11977-6", valueInteger=1),
        _obs("e3-glu", "e3", LOINC, "15074-8", **_qty(1.2, "g/L")),
        _obs("e3-pres", "e3", ROBBY, "fetal-presentation", valueString="footling"),
        _patient("p4", "FX-2004", "Esther", "Nyiraneza", "1993-07-30"),
        _encounter("e4", "p4", ago(72), status="finished"),
        _patient("p5", "SLOW-1", "Slow", "Server", "1995-01-01"),
        _patient("p6", "FAIL-1", "Failing", "Server", "1995-01-01"),
    ]


RECORDS = build(datetime.now(UTC).astimezone())
app = FastAPI(title="Stand-in hospital FHIR server", docs_url=None, redoc_url=None)


def _searchset(found: list[Json]) -> JSONResponse:
    body = {"resourceType": "Bundle", "type": "searchset", "total": len(found),
            "entry": [{"resource": r, "search": {"mode": "match"}} for r in found]}  # fmt: skip
    return JSONResponse(body, media_type=FHIR_JSON)


def _of(kind: str) -> list[Json]:
    return [r for r in RECORDS if r["resourceType"] == kind]


def _ref(resource: Json, name: str) -> str:
    return str((resource.get(name) or {}).get("reference", ""))


@app.get("/metadata")
def metadata() -> JSONResponse:
    return JSONResponse({"resourceType": "CapabilityStatement", "status": "active",
                         "kind": "instance", "fhirVersion": "4.0.1", "format": ["json"]},
                        media_type=FHIR_JSON)  # fmt: skip


@app.get("/Patient")
async def patients(identifier: str = "") -> JSONResponse:
    number = identifier.split("|")[-1]
    if number == "SLOW-1":
        await asyncio.sleep(10)
    if number == "FAIL-1":
        return JSONResponse({"resourceType": "OperationOutcome"}, status_code=500)
    found = [p for p in _of("Patient") if any(i["value"] == number for i in p["identifier"])]
    return _searchset(found)


@app.get("/Encounter")
def encounters(subject: str = "", status: str = "") -> JSONResponse:
    found = [e for e in _of("Encounter") if _ref(e, "subject") == subject]
    return _searchset([e for e in found if not status or e["status"] == status])


@app.get("/Observation")
def observations(encounter: str = "") -> JSONResponse:
    return _searchset([o for o in _of("Observation") if _ref(o, "encounter") == encounter])


@app.get("/Condition")
def conditions(subject: str = "") -> JSONResponse:
    return _searchset([c for c in _of("Condition") if _ref(c, "subject") == subject])
