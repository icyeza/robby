"""Lab requests over FHIR (robby.lab): open requests as ServiceRequest resources, and a result
sent back as an Observation whose basedOn names the ServiceRequest.

The Observation's value is read with the admission adapter's own readers (glucose in mmol/L or
mg/dL, the dipstick words) and then checked by lab.catalogue.parse_result, exactly as the lab
technician's form is: the API cannot record a value the form would refuse. The codes are in
mapping.LAB_TESTS (HIV is a placeholder code)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from robby.fhir import mapping as m
from robby.fhir.adapter import READERS, Issue
from robby.lab import catalogue
from robby.models import Admission, LabRequest, LabTest

Json = dict[str, Any]
SERVICE_REQUEST_REF = re.compile(r"(?:^|/)ServiceRequest/([0-9a-fA-F-]{36})$")
ADMISSION_CODE_SYSTEM = m.EXTENSION_BASE + "admission-code"
AT = "Observation"


def _code(test: LabTest) -> Json:
    system, code, display = m.LAB_TESTS[test.value]
    return {
        "coding": [{"system": system, "code": code, "display": display}],
        "text": catalogue.test_name(test),
    }


def service_request(
    request: LabRequest, admission: Admission, patient_id_system: str | None
) -> Json:
    """One open request. The subject is the patient number only (the token's lab technician may
    see it, as on the Lab requests page); the name is not sent."""
    subject: Json
    if admission.patient_number:
        identifier: Json = {"value": admission.patient_number}
        if patient_id_system:
            identifier["system"] = patient_id_system
        subject = {"identifier": identifier}
    else:
        subject = {"display": f"Patient of admission {admission.code}"}
    resource: Json = {
        "resourceType": "ServiceRequest",
        "id": str(request.id),
        "status": "active",
        "intent": "order",
        "priority": "urgent" if request.urgent else "routine",
        "code": _code(request.test),
        "subject": subject,
        "encounter": {"identifier": {"system": ADMISSION_CODE_SYSTEM, "value": admission.code}},
        "authoredOn": request.requested_at.isoformat(timespec="seconds"),
    }
    if request.note:
        resource["note"] = [{"text": request.note}]
    return resource


def searchset(rows: list[tuple[LabRequest, Admission]], patient_id_system: str | None) -> Json:
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(rows),
        "entry": [
            {
                "fullUrl": f"ServiceRequest/{request.id}",
                "resource": service_request(request, admission, patient_id_system),
                "search": {"mode": "match"},
            }
            for request, admission in rows
        ],
    }


def based_on(observation: Json) -> uuid.UUID | Issue:
    """The id of the ServiceRequest the Observation answers."""
    references = observation.get("basedOn")
    for i, ref in enumerate(references if isinstance(references, list) else []):
        found = (
            SERVICE_REQUEST_REF.search(str(ref.get("reference", "")))
            if isinstance(ref, dict)
            else None
        )
        if found:
            try:
                return uuid.UUID(found.group(1))
            except ValueError:
                return Issue(f"{AT}.basedOn[{i}].reference", "Not a ServiceRequest id.")
    return Issue(
        f"{AT}.basedOn",
        'Name the request answered: basedOn [{"reference": "ServiceRequest/<id>"}].',
        "required",
    )


@dataclass(frozen=True)
class Read:
    value: str | None  # the stored value, checked like the form's
    issues: list[Issue]


def _words(observation: Json) -> tuple[list[str], str] | None:
    concept = observation.get("valueCodeableConcept")
    if isinstance(concept, dict):
        found = concept.get("coding")
        codings: list[Any] = found if isinstance(found, list) else []
        words = [
            str(c.get(k))
            for c in codings
            if isinstance(c, dict)
            for k in ("code", "display")
            if c.get(k)
        ]
        if isinstance(concept.get("text"), str):
            words.append(concept["text"])
        return words, f"{AT}.valueCodeableConcept"
    if isinstance(observation.get("valueString"), str):
        return [observation["valueString"]], f"{AT}.valueString"
    return None


def _raw_value(test: LabTest, observation: Json) -> tuple[str, str] | Issue:
    """(the value as the form would receive it, the FHIR element it came from)."""
    if test is LabTest.HIV_RAPID:
        found = _words(observation)
        if found is None:
            return Issue(f"{AT}.value[x]", "Give the HIV result as a code or text.", "required")
        words, where = found
        for word in words:
            wanted = " ".join(word.split()).lower()
            for value, shown in catalogue.HIV_OPTIONS:
                if wanted in (value, shown.lower()):
                    return value, where
        return Issue(where, "The HIV result must be positive, negative or indeterminate.")
    system, code, _ = m.LAB_TESTS[test.value]
    rule = m.OBSERVATIONS[(system, code)]
    filled = READERS[rule.reader](observation, AT, rule)
    if isinstance(filled, Issue):
        return filled
    return filled[rule.field]


def read_result(test: LabTest, observation: Json) -> Read:
    """The result of `test` in `observation`, or every issue with it."""
    issues: list[Issue] = []
    status = observation.get("status")
    if status not in m.RESULT_STATUSES:
        issues.append(Issue(f"{AT}.status", "Only a final result is recorded (status final)."))
    system, code, _ = m.LAB_TESTS[test.value]
    concept = observation.get("code")
    codings = concept.get("coding") if isinstance(concept, dict) else None
    sent = {(c.get("system"), c.get("code")) for c in codings or [] if isinstance(c, dict)}
    if (system, code) not in sent:
        message = f"The request is for {catalogue.test_name(test)}: code {system}|{code}."
        issues.append(Issue(f"{AT}.code", message))
    raw = _raw_value(test, observation)
    if isinstance(raw, Issue):
        return Read(None, [*issues, raw])
    text, where = raw
    value, error = catalogue.parse_result(test, text)
    if error is not None:
        issues.append(Issue(where, error))
    return Read(value if not issues else None, issues)


def recorded(result_id: uuid.UUID, request: LabRequest) -> Json:
    """The answer to a recorded result: the Observation's id and what it answered, no value."""
    return {
        "resourceType": "Observation",
        "id": str(result_id),
        "status": "final",
        "basedOn": [{"reference": f"ServiceRequest/{request.id}"}],
        "code": _code(request.test),
    }
