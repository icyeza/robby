"""Sign-in, sign-out, session end, forced password change and CSRF through the real routes."""

from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from robby.auth.dependencies import SESSION_COOKIE
from robby.config import Settings
from robby.models import Role
from robby.timeutil import utcnow
from robby.web.csrf import LOGIN_NONCE_COOKIE
from robby.web.nav import landing_path
from tests.factories import DEFAULT_PASSWORD, events_of, make_user
from tests.webutil import csrf_for, form_token, log_in, make_test_app, set_cookie, sign_in_client

GENERIC = "Username or password not recognised."


def test_login_page(client: TestClient) -> None:
    response = client.get("/login")
    assert response.status_code == 200
    assert "<title>Sign in · Robson Readiness</title>" in response.text
    assert form_token(response.text)
    assert 'autocomplete="current-password"' in response.text
    assert "example-banner" not in response.text
    nonce_cookie = set_cookie(response, LOGIN_NONCE_COOKIE)
    assert "httponly" in nonce_cookie and "path=/login" in nonce_cookie


def test_sign_in_sets_a_strict_http_only_cookie_and_lands_by_role(
    client: TestClient, db: Session
) -> None:
    user = make_user(db, role=Role.SYSTEM_ADMIN)
    response = log_in(client, user.username)
    assert response.status_code == 303 and response.headers["location"] == "/"
    cookie = set_cookie(response, SESSION_COOKIE)
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/" in cookie
    assert "secure" not in cookie
    landing = client.get("/")
    assert landing.status_code == 303 and landing.headers["location"] == "/admin/facilities"


def test_session_cookie_is_secure_when_configured(
    settings: Settings, connection: Connection, db: Session
) -> None:
    secure_app = make_test_app(settings.model_copy(update={"secure_cookies": True}), connection)
    user = make_user(db)
    with TestClient(secure_app, base_url="https://testserver", follow_redirects=False) as https:
        response = log_in(https, user.username)
    assert response.status_code == 303
    assert "secure" in set_cookie(response, SESSION_COOKIE)


def test_wrong_password_shows_the_generic_message(client: TestClient, db: Session) -> None:
    user = make_user(db)
    response = log_in(client, user.username, "wrong-password")
    assert response.status_code == 400
    assert GENERIC in response.text
    assert all(not h.startswith(SESSION_COOKIE) for h in response.headers.get_list("set-cookie"))
    db.refresh(user)
    assert user.failed_logins == 1


def test_unknown_user_gets_the_same_message(client: TestClient) -> None:
    response = log_in(client, "nobody.here", "whatever-password")
    assert response.status_code == 400 and GENERIC in response.text


def test_locked_account_gets_the_same_message(client: TestClient, db: Session) -> None:
    user = make_user(db)
    for _ in range(5):
        log_in(client, user.username, "wrong-password")
    response = log_in(client, user.username)
    assert response.status_code == 400 and GENERIC in response.text
    db.refresh(user)
    assert user.locked_until is not None


def test_sign_ins_are_logged(client: TestClient, db: Session) -> None:
    user = make_user(db)
    log_in(client, user.username, "wrong-password")
    log_in(client, user.username)
    assert len(events_of(db, "session.login_failed")) == 1
    assert len(events_of(db, "session.login")) == 1


def test_sign_in_needs_a_genuine_form_token(client: TestClient, db: Session) -> None:
    user = make_user(db)
    client.get("/login")
    response = client.post(
        "/login",
        data={"username": user.username, "password": DEFAULT_PASSWORD, "csrf_token": "forged"},
    )
    assert response.status_code == 400
    assert "The sign-in form had expired" in response.text


def test_sign_in_needs_the_matching_cookie(client: TestClient, db: Session) -> None:
    user = make_user(db)
    token = form_token(client.get("/login").text)
    client.cookies.clear()
    response = client.post(
        "/login",
        data={"username": user.username, "password": DEFAULT_PASSWORD, "csrf_token": token},
    )
    assert response.status_code == 400


def test_signed_in_visitor_to_the_login_page_goes_home(
    client: TestClient, db: Session, settings: Settings
) -> None:
    sign_in_client(client, db, settings, make_user(db))
    response = client.get("/login")
    assert response.status_code == 303 and response.headers["location"] == "/"


def test_pages_need_a_session(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 303 and response.headers["location"] == "/login"


def test_sign_out_needs_csrf_then_revokes_the_session(
    client: TestClient, db: Session, settings: Settings
) -> None:
    row = sign_in_client(client, db, settings, make_user(db))
    refused = client.post("/logout")
    assert refused.status_code == 403 and "This form has expired" in refused.text
    response = client.post("/logout", data={"csrf_token": csrf_for(settings, row)})
    assert response.status_code == 303 and response.headers["location"] == "/login"
    assert "max-age=0" in set_cookie(response, SESSION_COOKIE)
    db.refresh(row)
    assert row.revoked_at is not None
    client.cookies.set(SESSION_COOKIE, "stale")
    assert client.get("/").headers["location"] == "/login"


def test_csrf_token_is_accepted_from_the_htmx_header(
    client: TestClient, db: Session, settings: Settings
) -> None:
    row = sign_in_client(client, db, settings, make_user(db))
    response = client.post("/logout", headers={"X-CSRF-Token": csrf_for(settings, row)})
    assert response.status_code == 303


def test_csrf_token_of_another_session_is_refused(
    client: TestClient, db: Session, settings: Settings
) -> None:
    other = sign_in_client(client, db, settings, make_user(db))
    sign_in_client(client, db, settings, make_user(db))
    response = client.post("/logout", data={"csrf_token": csrf_for(settings, other)})
    assert response.status_code == 403


def test_idle_session_ends_on_the_session_ended_page(
    client: TestClient, db: Session, settings: Settings
) -> None:
    row = sign_in_client(client, db, settings, make_user(db))
    row.last_seen_at = utcnow() - timedelta(minutes=31)
    db.commit()
    response = client.get("/")
    assert response.status_code == 303
    assert response.headers["location"] == "/session-ended?reason=idle"
    assert "max-age=0" in set_cookie(response, SESSION_COOKIE)
    page = client.get(response.headers["location"])
    assert page.status_code == 200
    assert "<title>Session ended · Robson Readiness</title>" in page.text
    assert "Your session ended after 30 minutes without activity." in page.text
    assert form_token(page.text)


def test_session_ends_after_its_absolute_lifetime(
    client: TestClient, db: Session, settings: Settings
) -> None:
    row = sign_in_client(client, db, settings, make_user(db))
    row.expires_at = utcnow() - timedelta(seconds=1)
    db.commit()
    response = client.get("/")
    assert response.headers["location"] == "/session-ended?reason=absolute"
    assert "Your session ended after 12 hours." in client.get(response.headers["location"]).text


def test_htmx_requests_are_redirected_with_a_header(
    client: TestClient, db: Session, settings: Settings
) -> None:
    row = sign_in_client(client, db, settings, make_user(db))
    row.last_seen_at = utcnow() - timedelta(minutes=31)
    db.commit()
    response = client.get("/", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["HX-Redirect"] == "/session-ended?reason=idle"


def test_an_active_session_slides(client: TestClient, db: Session, settings: Settings) -> None:
    row = sign_in_client(client, db, settings, make_user(db))
    before = utcnow() - timedelta(minutes=5)
    row.last_seen_at = before
    db.commit()
    client.get("/")
    db.refresh(row)
    assert row.last_seen_at > before + timedelta(minutes=4)


def test_a_deactivated_account_is_signed_out(
    client: TestClient, db: Session, settings: Settings
) -> None:
    user = make_user(db)
    sign_in_client(client, db, settings, user)
    user.active = False
    db.commit()
    assert client.get("/").headers["location"] == "/login"


def test_forced_password_change_comes_before_anything_else(
    client: TestClient, db: Session, settings: Settings
) -> None:
    user = make_user(db, must_change=True)
    row = sign_in_client(client, db, settings, user)
    assert client.get("/").headers["location"] == "/account/password"
    page = client.get("/account/password")
    assert page.status_code == 200
    assert "Choose your own password before you continue." in page.text
    assert "<nav></nav>" in page.text
    token = csrf_for(settings, row)
    rejected = client.post(
        "/account/password",
        data={
            "csrf_token": token,
            "current_password": DEFAULT_PASSWORD,
            "new_password": "short",
            "confirm_password": "short",
        },
    )
    assert rejected.status_code == 400 and "Use at least 10 characters." in rejected.text
    accepted = client.post(
        "/account/password",
        data={
            "csrf_token": token,
            "current_password": DEFAULT_PASSWORD,
            "new_password": "brand-new-password",
            "confirm_password": "brand-new-password",
        },
    )
    assert accepted.status_code == 303 and accepted.headers["location"] == "/"
    db.refresh(user)
    assert user.must_change_password is False
    assert client.get("/").headers["location"] == landing_path(user.role)


def test_password_change_needs_csrf(client: TestClient, db: Session, settings: Settings) -> None:
    sign_in_client(client, db, settings, make_user(db))
    response = client.post(
        "/account/password",
        data={
            "current_password": DEFAULT_PASSWORD,
            "new_password": "brand-new-password",
            "confirm_password": "brand-new-password",
        },
    )
    assert response.status_code == 403
