"""Every page against every role: allowed roles get 200, others 403, visitors the sign-in page.

Tasks 17 and 18 add the Facilities and Users pages and their POST actions to these tables."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from robby.config import Settings
from robby.models import Role
from robby.web.nav import NAV, landing_path
from robby.web.placeholders import PLACEHOLDER_PAGES
from tests.factories import make_facility, make_user
from tests.webutil import csrf_for, sign_in_client

PAGES: list[tuple[str, frozenset[Role]]] = [
    (path, frozenset({role})) for path, _title, role in PLACEHOLDER_PAGES
]
PAGES.append(("/models", frozenset({Role.MODEL_CUSTODIAN})))
PAGES.append(("/approval-log", frozenset({Role.MODEL_CUSTODIAN})))
PAGES.append(("/monitoring", frozenset({Role.MODEL_CUSTODIAN})))
PAGES.append(("/admin/facilities", frozenset({Role.SYSTEM_ADMIN})))
PAGES.append(("/admin/users", frozenset({Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN})))
PAGES.append(("/admissions/new", frozenset({Role.LAB_TECHNICIAN})))
PAGES.append(("/admissions", frozenset({Role.LAB_TECHNICIAN})))
PAGES.append(("/patients", frozenset({Role.CLINICIAN})))
PAGES.append(("/admissions/import", frozenset({Role.FACILITY_ADMIN})))
PAGES.append(("/admissions/import/template.xlsx", frozenset({Role.FACILITY_ADMIN})))
PAGES.append(("/lab-requests", frozenset({Role.LAB_TECHNICIAN})))
PAGES.append(("/lab-requests/count", frozenset({Role.LAB_TECHNICIAN})))
# Pages under an admission code or a lab request id: test_lab_requests.test_request_pages_by_role


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(("path", "allowed"), PAGES, ids=[path for path, _ in PAGES])
def test_page_access_by_role(
    client: TestClient,
    db: Session,
    settings: Settings,
    role: Role,
    path: str,
    allowed: frozenset[Role],
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=role))
    response = client.get(path)
    assert response.status_code == (200 if role in allowed else 403)
    if role not in allowed:
        assert "Not allowed" in response.text


@pytest.mark.parametrize("path", [path for path, _ in PAGES] + ["/", "/account/password"])
def test_every_page_needs_a_session(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 303 and response.headers["location"] == "/login"


@pytest.mark.parametrize("role", list(Role))
def test_landing_redirects_to_the_role_start_page(
    client: TestClient, db: Session, settings: Settings, role: Role
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=role))
    response = client.get("/")
    assert response.status_code == 303 and response.headers["location"] == landing_path(role)


@pytest.mark.parametrize("role", list(Role))
def test_every_role_can_change_its_password(
    client: TestClient, db: Session, settings: Settings, role: Role
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=role))
    assert client.get("/account/password").status_code == 200


def test_placeholder_pages_say_not_available_and_show_no_data(
    client: TestClient, db: Session, settings: Settings
) -> None:
    if not PLACEHOLDER_PAGES:
        pytest.skip("no placeholder pages left")
    sign_in_client(client, db, settings, make_user(db, role=Role.MODEL_CUSTODIAN))
    path, title, _role = PLACEHOLDER_PAGES[0]
    response = client.get(path)
    assert f"<title>{title} · Robson Readiness</title>" in response.text
    assert "Not available yet" in response.text
    assert f'<a class="on" aria-current="page" href="{path}">{title}</a>' in response.text
    assert "<table" not in response.text


POST_ACTIONS: list[tuple[str, frozenset[Role]]] = [
    ("/admin/facilities", frozenset({Role.SYSTEM_ADMIN})),
    ("/admin/facilities/{facility}/deactivate", frozenset({Role.SYSTEM_ADMIN})),
    ("/admin/users", frozenset({Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN})),
    ("/admin/users/{user}/reset-password", frozenset({Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN})),
    ("/admin/users/{user}/deactivate", frozenset({Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN})),
    ("/admin/users/{user}/reactivate", frozenset({Role.SYSTEM_ADMIN, Role.FACILITY_ADMIN})),
    ("/admissions/import", frozenset({Role.FACILITY_ADMIN})),
    ("/admissions", frozenset({Role.LAB_TECHNICIAN})),
    ("/admissions/preview", frozenset({Role.LAB_TECHNICIAN})),
    ("/admissions/ZZ-0001/lab-requests", frozenset({Role.CLINICIAN})),
    (
        "/admissions/ZZ-0001/lab-requests/00000000-0000-4000-8000-000000000001/cancel",
        frozenset({Role.CLINICIAN}),
    ),
    ("/lab-requests/00000000-0000-4000-8000-000000000001", frozenset({Role.LAB_TECHNICIAN})),
]


def forbidden_posts() -> list[tuple[str, Role]]:
    return [(path, role) for path, allowed in POST_ACTIONS for role in Role if role not in allowed]


@pytest.mark.parametrize(("path", "role"), forbidden_posts(), ids=lambda value: str(value))
def test_actions_are_refused_to_other_roles(
    client: TestClient, db: Session, settings: Settings, path: str, role: Role
) -> None:
    facility = make_facility(db)
    target = make_user(db, role=Role.CLINICIAN, facility=facility)
    row = sign_in_client(client, db, settings, make_user(db, role=role))
    url = path.format(facility=facility.id, user=target.id)
    response = client.post(
        url,
        data={
            "csrf_token": csrf_for(settings, row),
            "name": "Facility X",
            "code": "FX",
            "district": "D",
            "username": "nobody.new",
            "role": "clinician",
        },
    )
    assert response.status_code == 403
    db.refresh(facility)
    db.refresh(target)
    assert facility.active and target.active


@pytest.mark.parametrize("role", list(Role))
def test_every_nav_link_of_every_role_opens_a_page(
    client: TestClient, db: Session, settings: Settings, role: Role
) -> None:
    sign_in_client(client, db, settings, make_user(db, role=role))
    for item in NAV[role]:
        assert client.get(item.path).status_code == 200, item.path
