"""Stress test of a running demo instance with Locust: mixed roles, stepped load until it breaks.

Usage, against a demo instance served over plain http (``ROBBY_SECURE_COOKIES=false``), e.g. the
14,000-admission performance database::

    uv run --with locust locust -f scripts/locustfile.py --host http://127.0.0.1:8171 \\
        --headless --csv stress --html stress.html

Simulated users sign in with the demo accounts of a random hospital (``da`` to ``dd``) and act as:

* lab technicians (weight 3): open the admission form and save an admission, as at the bedside;
* clinicians (weight 3): search a patient by name (the name is encrypted, so the server decrypts
  the facility's admissions to match) and open the admission;
* facility administrators (weight 1): open the hospital report for the whole period.

``StepLoad`` raises the number of users in steps (10, 25, 50, 100, 150, 200; one minute each) so
the stats history shows where the 95th percentile passes 500 ms and where errors start. Uses only
demo accounts and invented patient details, and writes only to the demo database.
"""

from __future__ import annotations

import random
import re
import uuid
from html.parser import HTMLParser

from locust import HttpUser, LoadTestShape, between, task

PASSWORD = "demo-password-123"
HOSPITALS = ("da", "db", "dc", "dd")
ADMISSION = {
    "maternal_age": "29",
    "gestational_age_weeks": "39",
    "parity": "2",
    "previous_cs_count": "1",
    "fetal_presentation": "cephalic",
    "plurality": "1",
    "onset_of_labour": "spontaneous",
    "bp_sys": "128",
    "bp_dia": "84",
    "proteinuria": "negative",
    "glucose": "5.2",
}
CODE = re.compile(r'href="/admissions/([A-Z]{2,6}-\d+)"')
STEPS = (10, 25, 50, 100, 150, 200)
STEP_SECONDS = 60


class _Inputs(HTMLParser):
    """The inputs a browser would send: hidden, prefilled and checked ones."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "input":
            return
        a = dict(attrs)
        unchecked = a.get("type") in ("checkbox", "radio") and "checked" not in a
        if a.get("name") and not unchecked:
            self.values[str(a["name"])] = a.get("value") or ""


def inputs(html: str) -> dict[str, str]:
    parser = _Inputs()
    parser.feed(html)
    return parser.values


class DemoUser(HttpUser):
    """Signs in once as the role's demo account at a random hospital."""

    abstract = True
    role = ""
    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.hospital = random.choice(HOSPITALS)
        form = inputs(self.client.get("/login", name="/login").text)
        username = f"{self.role}.{self.hospital}"
        self.client.post(
            "/login", {**form, "username": username, "password": PASSWORD}, name="/login [POST]"
        )


class LabTechnician(DemoUser):
    role = "lab"
    weight = 3

    @task
    def save_admission(self) -> None:
        form = inputs(self.client.get("/admissions/new", name="/admissions/new").text)
        data = {
            **form,
            **ADMISSION,
            "patient_number": f"LT-{uuid.uuid4().hex[:8].upper()}",
            "patient_name": f"Load Test {random.choice('ABCDEFGH')}",
        }
        with self.client.post(
            "/admissions", data, name="/admissions [save]", catch_response=True
        ) as response:
            if "saved=" not in response.url:
                response.failure("admission not saved")


class Clinician(DemoUser):
    role = "clinician"
    weight = 3

    @task
    def find_and_open(self) -> None:
        page = self.client.get(
            f"/patients?q=Load Test {random.choice('ABCDEFGH')}", name="/patients?q=[name]"
        )
        codes = CODE.findall(page.text)
        if codes:
            self.client.get(f"/admissions/{random.choice(codes)}", name="/admissions/[code]")


class FacilityAdmin(DemoUser):
    role = "admin"
    weight = 1

    @task
    def hospital_report(self) -> None:
        self.client.get("/audit?from=2000-01-01", name="/audit [whole period]")


class StepLoad(LoadTestShape):
    """10, 25, 50, 100, 150 and 200 users, one minute each, then stop."""

    def tick(self) -> tuple[int, float] | None:
        step = int(self.get_run_time() // STEP_SECONDS)
        if step >= len(STEPS):
            return None
        return STEPS[step], 5.0
