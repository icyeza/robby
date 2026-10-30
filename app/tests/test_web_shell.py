"""The page shell: layout, security headers, styled error pages, navigation and helpers."""

import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Connection
from starlette.requests import ClientDisconnect

from robby.config import Settings
from robby.models import AppUser, Facility, Role
from robby.web.nav import NAV, Who, landing_path, nav_for, who_for
from robby.web.security import CLIENT_CLOSED_REQUEST, CSP
from robby.web.templating import format_local
from tests.webutil import make_test_app


def test_static_files_are_served_and_cacheable(client: TestClient) -> None:
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert "no-store" not in response.headers.get("cache-control", "")


def test_unknown_page_is_a_styled_404_with_security_headers(client: TestClient) -> None:
    response = client.get("/no-such-page")
    assert response.status_code == 404
    assert "<title>Page not found · Robson Readiness</title>" in response.text
    for asset in ("/static/styles.css", "/static/app.css", "/static/htmx.min.js", "/static/app.js"):
        assert asset in response.text
    assert response.headers["content-security-policy"] == CSP
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"
    assert "example-banner" not in response.text


def test_csp_forbids_inline_code_and_framing() -> None:
    assert "default-src 'self'" in CSP
    assert "frame-ancestors 'none'" in CSP
    assert "unsafe-inline" not in CSP


def test_demo_mode_shows_the_example_data_banner(
    settings: Settings, connection: Connection
) -> None:
    demo_app = make_test_app(settings.model_copy(update={"demo": True}), connection)
    with TestClient(demo_app) as demo_client:
        assert "example-banner" in demo_client.get("/no-such-page").text


def test_server_errors_are_styled_and_hide_details(app: FastAPI) -> None:
    def boom() -> None:
        raise RuntimeError("secret internals")

    app.add_api_route("/boom", boom)
    with TestClient(app) as test_client:
        response = test_client.get("/boom")
    assert response.status_code == 500
    assert "Something went wrong" in response.text
    assert "secret internals" not in response.text and "Traceback" not in response.text
    assert response.headers["content-security-policy"] == CSP


def test_a_client_that_leaves_mid_request_is_not_a_server_error(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    def gone() -> None:
        raise ClientDisconnect()

    app.add_api_route("/gone", gone, methods=["POST"])
    with TestClient(app) as test_client:
        response = test_client.post("/gone")
    assert response.status_code == CLIENT_CLOSED_REQUEST
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_navigation_matches_the_mockup_extension() -> None:
    labels = {role: [(item.label, item.path) for item in items] for role, items in NAV.items()}
    assert labels == {
        Role.LAB_TECHNICIAN: [
            ("New admission", "/admissions/new"),
            ("Recent entries", "/admissions"),
            ("Lab requests", "/lab-requests"),
        ],
        Role.CLINICIAN: [("Find patient", "/patients")],
        Role.FACILITY_ADMIN: [("Hospital report", "/audit"), ("Users", "/admin/users")],
        Role.MODEL_CUSTODIAN: [
            ("Models", "/models"),
            ("Monitoring", "/monitoring"),
            ("Approval log", "/approval-log"),
        ],
        Role.NATIONAL_REVIEWER: [("National overview", "/national")],
        Role.SYSTEM_ADMIN: [("Facilities", "/admin/facilities"), ("Users", "/admin/users")],
    }


def test_landing_page_per_role() -> None:
    assert {role: landing_path(role) for role in Role} == {
        Role.LAB_TECHNICIAN: "/admissions/new",
        Role.CLINICIAN: "/patients",
        Role.FACILITY_ADMIN: "/audit",
        Role.MODEL_CUSTODIAN: "/models",
        Role.NATIONAL_REVIEWER: "/national",
        Role.SYSTEM_ADMIN: "/admin/facilities",
    }


def active(role: Role, path: str) -> list[str]:
    return [link.label for link in nav_for(role, path) if link.active]


def test_the_active_item_is_the_longest_matching_path() -> None:
    assert active(Role.LAB_TECHNICIAN, "/admissions/new") == ["New admission"]
    assert active(Role.LAB_TECHNICIAN, "/admissions") == ["Recent entries"]
    assert active(Role.CLINICIAN, "/patients") == ["Find patient"]
    assert active(Role.CLINICIAN, "/admissions/DA-0001") == ["Find patient"]
    assert active(Role.CLINICIAN, "/admissions/DA-0001/outcome") == ["Find patient"]
    assert active(Role.FACILITY_ADMIN, "/audit") == ["Hospital report"]
    assert active(Role.SYSTEM_ADMIN, "/admin/users/123/reset-password") == ["Users"]
    assert active(Role.CLINICIAN, "/account/password") == []


def test_header_identity_lines() -> None:
    facility = Facility(name="Facility A", code="FA", district="District 1", active=True)

    def who(role: Role, with_facility: bool = False) -> Who:
        user = AppUser(username="u", password_hash="h", role=role)
        if with_facility:
            user.facility = facility
        return who_for(user)

    assert who(Role.CLINICIAN, True) == Who("Facility A", "Clinician")
    assert who(Role.FACILITY_ADMIN, True) == Who("Facility A", "Facility administrator")
    assert who(Role.MODEL_CUSTODIAN) == Who("All facilities", "Model custodian")
    assert who(Role.NATIONAL_REVIEWER) == Who("All facilities · National", "National reviewer")
    assert who(Role.SYSTEM_ADMIN) == Who("All facilities", "System administrator")


def test_times_are_shown_in_the_configured_zone() -> None:
    kigali = ZoneInfo("Africa/Kigali")
    assert format_local(datetime(2026, 10, 5, 12, 20, tzinfo=UTC), kigali) == "05 Oct 2026, 14:20"
    assert format_local(None, kigali) == "Never"


def test_segmented_control_is_made_of_real_radio_inputs(app: FastAPI) -> None:
    template = app.state.templates.env.from_string(
        '{% import "partials/_macros.html" as ui %}'
        '{{ ui.segmented("mode", "Mode", [("a", "First"), ("b", "Second")], "b") }}'
    )
    html = template.render()
    assert html.count('type="radio"') == 2
    assert 'value="b" checked' in html
    assert 'role="radiogroup"' in html
