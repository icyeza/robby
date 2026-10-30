"""The FHIR resources the app answers with: OperationOutcome, the transaction-response Bundle
and the CapabilityStatement."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from robby.fhir.adapter import Issue
from robby.fhir.intake import Outcome
from robby.fhir.mapping import EXTENSION_BASE

FHIR_JSON = "application/fhir+json"
ROBSON_GROUP = EXTENSION_BASE + "robson-group"
READINESS = EXTENSION_BASE + "readiness-percent"
ADMISSION_CODE = EXTENSION_BASE + "admission-code"
STATUS_LINE = {
    "created": "201 Created",
    "already_recorded": "200 OK",
    "repeated": "200 OK",
}

Json = dict[str, Any]


def operation_outcome(issues: list[Issue]) -> Json:
    return {
        "resourceType": "OperationOutcome",
        "issue": [
            {
                "severity": i.severity,
                "code": i.code,
                "diagnostics": i.message,
                "expression": [i.expression],
            }
            for i in issues
        ],
    }


def problem(message: str, code: str = "invalid", severity: str = "error") -> Json:
    """An OperationOutcome with one issue about the request as a whole."""
    return {
        "resourceType": "OperationOutcome",
        "issue": [{"severity": severity, "code": code, "diagnostics": message}],
    }


def _summary(o: Outcome) -> str:
    group = f"Robson group {o.group}" if o.group else "No Robson group yet (details missing)"
    readiness = f"; readiness signal {o.percent}%" if o.percent is not None else ""
    done = {
        "created": "Saved as",
        "already_recorded": "Already recorded as",
        "repeated": f"Same patient and admission time as {o.repeats}; not saved again. See",
    }[o.state]
    return f"{o.encounter}: {done} {o.admission.code}. {group}{readiness}."


def _entry(o: Outcome) -> Json:
    extension: list[Json] = [{"url": ADMISSION_CODE, "valueString": o.admission.code}]
    if o.group is not None:
        extension.append({"url": ROBSON_GROUP, "valueInteger": o.group})
    if o.percent is not None:
        extension.append({"url": READINESS, "valueInteger": o.percent})
    return {
        "response": {
            "extension": extension,
            "status": STATUS_LINE[o.state],
            "location": f"/admissions/{o.admission.code}",
            "outcome": problem(_summary(o), "informational", "information"),
        }
    }


def transaction_response(outcomes: list[Outcome]) -> Json:
    """One entry per Encounter, in Bundle order."""
    return {
        "resourceType": "Bundle",
        "type": "transaction-response",
        "entry": [_entry(o) for o in outcomes],
    }


def capability_statement(now: datetime) -> Json:
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "date": now.date().isoformat(),
        "kind": "instance",
        "software": {"name": "Robson Readiness"},
        "implementation": {"description": "Admission intake for Robson classification (FR8)"},
        "fhirVersion": "4.0.1",
        "format": [FHIR_JSON, "application/json"],
        "rest": [
            {
                "mode": "server",
                "documentation": (
                    "POST a Bundle (transaction, batch or collection) of Patient, Encounter, "
                    "Observation and Condition to /api/v1/fhir/Bundle with a bearer API token. "
                    "One admission is saved per Encounter; all or nothing."
                ),
                "security": {
                    "service": [{"text": "Bearer API token"}],
                    "description": "Authorization: Bearer <token of a lab technician>",
                },
                "resource": [
                    {"type": "Bundle", "interaction": [{"code": "create"}]},
                    {
                        "type": "ServiceRequest",
                        "interaction": [{"code": "search-type"}],
                        "searchParam": [{"name": "status", "type": "token"}],
                        "documentation": "Open lab requests of the token's facility "
                        "(status=active only).",
                    },
                    {
                        "type": "Observation",
                        "interaction": [{"code": "create"}],
                        "documentation": "The result of one lab request: basedOn its "
                        "ServiceRequest. Also accepted inside an admission Bundle.",
                    },
                    *[
                        {"type": kind, "documentation": "Accepted inside a Bundle only."}
                        for kind in ("Patient", "Encounter", "Condition")
                    ],
                ],
            }
        ],
    }
