"""Pull mode: fetching a woman's current admission from the hospital's FHIR server by patient
number, for the lab technician to review on the admission form (never saved by itself).

NFR7: the hospital system may be slow, down or wrong. Every way the lookup can fail is a
FetchProblem with a sentence for the form; the whole lookup is bounded by the timeout, and the
manual form keeps working whatever happens here."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import httpx2

from robby.fhir.resources import FHIR_JSON

Json = dict[str, Any]
MANUAL = "Fill in the form by hand."


class FetchProblem(Exception):
    """The lookup gave nothing usable; str(exc) is the plain-words reason for the form."""


class HospitalRecords(Protocol):
    """Where pull mode reads admissions from (a FHIR server; tests give a fake)."""

    async def admission_bundle(self, patient_number: str) -> Json:
        """A collection Bundle: the Patient, the in-progress Encounter, and its Observations and
        Conditions. Raises FetchProblem."""
        ...


def _resources(bundle: object, kind: str) -> list[Json]:
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        raise FetchProblem(f"The hospital system sent an answer that could not be read. {MANUAL}")
    found = []
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource") if isinstance(entry, dict) else None
        if isinstance(resource, dict) and resource.get("resourceType") == kind:
            found.append(resource)
    return found


class FhirClient:
    """A FHIR R4 server reached over HTTP (JSON search API)."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        patient_id_system: str | None = None,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.patient_id_system = patient_id_system
        self.transport = transport

    async def admission_bundle(self, patient_number: str) -> Json:
        seconds = f"{self.timeout:g} second{'s' if self.timeout != 1 else ''}"
        try:
            async with httpx2.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
                headers={"Accept": FHIR_JSON},
            ) as http:
                return await asyncio.wait_for(self._lookup(http, patient_number), self.timeout)
        except (TimeoutError, httpx2.TimeoutException):
            raise FetchProblem(
                f"The hospital system did not answer within {seconds}. {MANUAL}"
            ) from None
        except httpx2.HTTPError:
            raise FetchProblem(f"The hospital system could not be reached. {MANUAL}") from None

    async def _search(
        self, http: httpx2.AsyncClient, kind: str, params: dict[str, str]
    ) -> list[Json]:
        response = await http.get(f"/{kind}", params=params)
        if response.status_code >= 400:
            raise FetchProblem(
                f"The hospital system answered with an error (HTTP {response.status_code}). "
                f"{MANUAL}"
            )
        try:
            body = response.json()
        except ValueError:
            body = None
        return _resources(body, kind)

    async def _lookup(self, http: httpx2.AsyncClient, number: str) -> Json:
        token = f"{self.patient_id_system}|{number}" if self.patient_id_system else number
        patients = await self._search(http, "Patient", {"identifier": token})
        if not patients:
            raise FetchProblem(f"The hospital system has no patient with number {number}.")
        if len(patients) > 1:
            raise FetchProblem(
                f"The hospital system has several patients with number {number}. {MANUAL}"
            )
        patient = patients[0]
        subject = f"Patient/{patient.get('id')}"
        encounters = await self._search(
            http, "Encounter", {"subject": subject, "status": "in-progress"}
        )
        if not encounters:
            raise FetchProblem(
                f"The hospital system has no admission in progress for patient {number}."
            )
        encounter = max(encounters, key=lambda e: str((e.get("period") or {}).get("start", "")))
        reference = f"Encounter/{encounter.get('id')}"
        observations = await self._search(http, "Observation", {"encounter": reference})
        # The woman's conditions: those of this admission, and those tied to no admission.
        conditions = [
            c
            for c in await self._search(http, "Condition", {"subject": subject})
            if (c.get("encounter") or {}).get("reference", reference) == reference
        ]
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"fullUrl": f"{r['resourceType']}/{r.get('id')}", "resource": r}
                for r in [patient, encounter, *observations, *conditions]
            ],
        }
