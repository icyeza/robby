"""Every page against every role: allowed roles get 200, others 403, visitors the sign-in page.

Tasks 17 and 18 add the Facilities and Users pages and their POST actions to these tables."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from robby.config import Settings
from robby.models import Role
from robby.web.nav import landing_path
from robby.web.placeholders import PLACEHOLDER_PAGES
from tests.factories import make_user
from tests.webutil import sign_in_client

PAGES: list[tuple[str, frozenset[Role]]] = [
    (path, frozenset({role})) for path, _title, role in PLACEHOLDER_PAGES
]


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize(("path", "allowed"), PAGES, ids=[path for path, _ in PAGES])
def test_page_access_by_role(
    client: TestClient, db: Session, settings: Settings, role: Role, path: str,
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
    sign_in_client(client, db, settings, make_user(db, role=Role.CLINICIAN))
    response = client.get("/admissions/new")
    assert "<title>New admission · Robson Readiness</title>" in response.text
    assert "Not available yet" in response.text
    assert '<a class="on" aria-current="page" href="/admissions/new">New admission</a>' in response.text
    assert "<table" not in response.text
