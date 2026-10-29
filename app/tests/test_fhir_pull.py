"""Pull mode: "Fetch from hospital system" on the admission form (FR8), and NFR7: whatever the
hospital system does, the manual form keeps working."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.fhir.client import FetchProblem, FhirClient
from robby.models import Admission
from tests.test_clinical_flow import activate_model, post_form, saved_code, world  # noqa: F401
from tests.test_fhir_adapter import admission, condition, obs
from tests.test_no_inline_code import assert_no_inline_code

BASE = "http://hospital.test/fhir"
Json = dict[str, Any]
Handler = Callable[[httpx2.Request], httpx2.Response]


def hospital(records: list[Json]) -> Handler:
    """A FHIR server answering the four searches pull mode makes, from ``records``."""

    def ref(resource: Json, name: str) -> str:
        return str((resource.get(name) or {}).get("reference", ""))

    def handle(request: httpx2.Request) -> httpx2.Response:
        kind = request.url.path.rsplit("/", 1)[-1]
        q = request.url.params
        found = [r for r in records if r["resourceType"] == kind]
        if kind == "Patient":
            found = [p for p in found if any(i["value"] == q["identifier"].split("|")[-1]
                                             for i in p["identifier"])]  # fmt: skip
        elif kind == "Encounter":
            found = [e for e in found if ref(e, "subject") == q["subject"]
                     and e["status"] == q["status"]]  # fmt: skip
        elif kind == "Observation":
            found = [o for o in found if ref(o, "encounter") == q["encounter"]]
        elif kind == "Condition":
            found = [c for c in found if ref(c, "subject") == q["subject"]]
        body = {"resourceType": "Bundle", "type": "searchset",
                "entry": [{"resource": r} for r in found]}  # fmt: skip
        return httpx2.Response(200, json=body)

    return handle


def failing(error: Exception | None = None, status: int = 500) -> Handler:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if error is not None:
            raise error
        return httpx2.Response(status, json={"resourceType": "OperationOutcome"})

    return handle


@pytest.fixture
def records(world: dict[str, Any]) -> list[Json]:  # noqa: F811
    start = (datetime.now(world["settings"].zone) - timedelta(hours=2)).isoformat()
    found = admission("p1", "e1", "FX-77", start)
    found[2] = obs("e1-par", "11977-6", enc="e1", valueInteger=1)
    # a later glucose in a unit the app does not know: not filled in, and listed as such
    odd = obs("e1-x", "15074-8", enc="e1", valueQuantity={"value": 3, "code": "g"})
    return [*found, condition("86406008"), odd]


def connect(
    app: FastAPI,
    world: dict[str, Any],  # noqa: F811
    handler: Handler,
    timeout: float = 3.0,
) -> None:
    settings = world["settings"].model_copy(
        update={"fhir_base_url": BASE, "fhir_timeout_seconds": timeout}
    )
    app.state.settings = world["settings"] = settings
    app.state.fhir_transport = httpx2.MockTransport(handler)


def fetch(
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
    number: str,
    **extra: str,
) -> Any:
    data = {"patient_number": number, "csrf_token": world["csrf"], **extra}
    return client.post("/admissions/fetch", data=data, headers={"HX-Request": "true"})


def test_without_a_base_url_there_is_no_button_and_no_fetch(
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
) -> None:
    page = client.get("/admissions/new")
    assert page.status_code == 200 and "Fetch from hospital system" not in page.text
    assert fetch(client, world, "FX-77").status_code == 404


def test_with_a_base_url_the_form_offers_the_button(
    app: FastAPI,
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
    records: list[Json],
) -> None:
    connect(app, world, hospital(records))
    page = client.get("/admissions/new")
    assert "Fetch from hospital system" in page.text
    assert 'hx-post="/admissions/fetch"' in page.text
    assert_no_inline_code(page.text, "admission form with the fetch button")


def test_fetch_prefills_the_form_for_review_and_saves_nothing(
    app: FastAPI,
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
    records: list[Json],
    db: Session,
) -> None:
    activate_model(db)  # so the form asks for the model's optional inputs (HIV)
    connect(app, world, hospital(records))
    response = fetch(client, world, "FX-77")
    assert response.status_code == 200
    html = response.text
    assert "<html" not in html and 'id="adm-form-wrap"' in html  # the form fragment for htmx
    assert "Filled in from the hospital system" in html
    assert 'value="Aline Uwimana"' in html
    assert 'name="parity" type="number" value="1"' in html
    assert 'value="cephalic" checked' in html
    assert 'name="bp_sys" type="number" value="120"' in html
    assert 'name="d_hiv_positive" value="Yes" checked' in html
    assert "Glucose in “g” is not understood" in html  # listed as "not filled in"
    assert_no_inline_code(html, "fetched form")
    assert db.scalar(select(Admission)) is None


def test_fetched_values_save_like_typed_ones(
    app: FastAPI,
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
    records: list[Json],
    db: Session,
) -> None:
    connect(app, world, hospital(records[:-1]))
    html = client.post(
        "/admissions/fetch", data={"patient_number": "FX-77", "csrf_token": world["csrf"]}
    ).text
    assert "<html" in html  # without htmx: the whole page
    saved = post_form(client, world, patient_number="FX-77", patient_name="Aline Uwimana")
    assert db.scalars(select(Admission)).one().code == saved_code(saved)


@pytest.mark.parametrize(
    ("handler", "number", "words"),
    [
        (None, "FX-404", "The hospital system has no patient with number FX-404."),
        (failing(httpx2.ConnectTimeout("slow")), "FX-77", "did not answer within 3 seconds"),
        (failing(httpx2.ConnectError("refused")), "FX-77", "could not be reached"),
        (failing(status=500), "FX-77", "answered with an error (HTTP 500)"),
        (lambda r: httpx2.Response(200, text="<html>"), "FX-77", "could not be read"),
        (None, "", "Enter the patient number first"),
    ],
)
def test_when_the_hospital_system_fails_the_form_says_so_and_still_works(
    app: FastAPI,
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
    records: list[Json],
    db: Session,
    handler: Handler | None,
    number: str,
    words: str,
) -> None:
    connect(app, world, handler or hospital(records))
    response = fetch(client, world, number, maternal_age="31", parity="2")
    assert response.status_code == 200
    assert "Nothing fetched." in response.text and words in response.text
    assert "The form still works as usual." in response.text
    # what was typed is kept, and the form still saves
    assert 'name="maternal_age" type="number" value="31"' in response.text
    assert 'name="parity" type="number" value="2"' in response.text
    saved_code(post_form(client, world))
    assert db.scalar(select(Admission)) is not None


def test_a_patient_without_an_admission_in_progress(
    app: FastAPI,
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
    records: list[Json],
) -> None:
    for r in records:
        if r["resourceType"] == "Encounter":
            r["status"] = "finished"
    connect(app, world, hospital(records))
    assert "has no admission in progress for patient FX-77" in fetch(client, world, "FX-77").text


def test_a_slow_server_is_cut_off_at_the_timeout(
    app: FastAPI,
    client: TestClient,
    world: dict[str, Any],  # noqa: F811
) -> None:
    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(5)
        return httpx2.Response(200, json={})

    connect(app, world, slow, timeout=0.2)  # type: ignore[arg-type]  # an async handler
    started = datetime.now()
    response = fetch(client, world, "FX-77")
    assert "did not answer within 0.2 seconds" in response.text
    assert datetime.now() - started < timedelta(seconds=3)


def test_the_client_uses_the_configured_identifier_system(records: list[Json]) -> None:
    seen: list[str] = []
    inner = hospital(records)

    def spy(request: httpx2.Request) -> httpx2.Response:
        seen.append(str(request.url))
        return inner(request)

    client = FhirClient(BASE, 3, patient_id_system="urn:h:pn", transport=httpx2.MockTransport(spy))
    found = asyncio.run(client.admission_bundle("FX-77"))
    assert seen[0] == f"{BASE}/Patient?identifier=urn%3Ah%3Apn%7CFX-77"
    kinds = [e["resource"]["resourceType"] for e in found["entry"]]
    assert kinds[:2] == ["Patient", "Encounter"] and "Condition" in kinds
    with pytest.raises(FetchProblem, match="no patient with number X-1"):
        asyncio.run(client.admission_bundle("X-1"))
