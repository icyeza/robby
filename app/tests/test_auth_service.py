"""Sign-in with lockout, session resolution with idle and absolute expiry, password change."""

import hashlib
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from robby.auth.passwords import verify_password
from robby.auth.service import (
    LOCKOUT,
    LoginFailed,
    PasswordChangeError,
    SessionState,
    authenticate,
    change_password,
    logout,
    resolve_session,
    revoke_sessions,
    start_session,
)
from robby.config import Settings
from robby.models import Role, UserSession
from robby.timeutil import utcnow
from tests.factories import DEFAULT_PASSWORD, events_of, make_facility, make_user


def sign_in(
    db: Session, settings: Settings, username: str, password: str, now: datetime | None = None
) -> tuple[UserSession, str]:
    return authenticate(
        db, settings, username, password, ip="10.0.0.1", user_agent="pytest", now=now or utcnow()
    )


def test_successful_sign_in_creates_a_session_storing_only_the_token_hash(
    db: Session, settings: Settings
) -> None:
    user = make_user(db, username="mw.a014")
    now = utcnow()
    row, token = sign_in(db, settings, "  MW.A014 ", DEFAULT_PASSWORD, now)
    assert row.user_id == user.id
    assert row.token_hash == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert row.last_seen_at == now
    assert row.expires_at == now + timedelta(hours=12)
    assert (row.ip, row.user_agent) == ("10.0.0.1", "pytest")
    assert user.last_login_at == now
    [event] = events_of(db, "session.login")
    assert (event.actor_id, event.entity_type, event.entity_id) == (user.id, "user_session", row.id)


def test_wrong_password_is_counted_and_logged(db: Session, settings: Settings) -> None:
    user = make_user(db)
    with pytest.raises(LoginFailed):
        sign_in(db, settings, user.username, "wrong-password")
    assert user.failed_logins == 1
    [event] = events_of(db, "session.login_failed")
    assert event.actor_id is None
    assert (event.entity_type, event.entity_id) == ("app_user", user.id)
    assert event.payload == {"reason": "bad_password", "locked": False}


def test_unknown_username_is_logged_without_what_was_typed(db: Session, settings: Settings) -> None:
    with pytest.raises(LoginFailed):
        sign_in(db, settings, "nobody.here", "whatever-password")
    [event] = events_of(db, "session.login_failed")
    assert event.payload == {"reason": "unknown_user"}
    assert event.entity_id is None


def test_five_failures_lock_the_account_for_fifteen_minutes(db: Session, settings: Settings) -> None:
    user = make_user(db)
    now = utcnow()
    for _ in range(5):
        with pytest.raises(LoginFailed):
            sign_in(db, settings, user.username, "wrong-password", now)
    assert user.locked_until == now + LOCKOUT
    assert events_of(db, "session.login_failed")[-1].payload == {"reason": "bad_password", "locked": True}
    with pytest.raises(LoginFailed):
        sign_in(db, settings, user.username, DEFAULT_PASSWORD, now + timedelta(minutes=14))
    assert events_of(db, "session.login_failed")[-1].payload == {"reason": "locked"}
    row, _ = sign_in(db, settings, user.username, DEFAULT_PASSWORD, now + timedelta(minutes=15, seconds=1))
    assert row.user_id == user.id
    assert user.failed_logins == 0 and user.locked_until is None


def test_deactivated_accounts_cannot_sign_in(db: Session, settings: Settings) -> None:
    user = make_user(db, active=False)
    with pytest.raises(LoginFailed):
        sign_in(db, settings, user.username, DEFAULT_PASSWORD)
    assert events_of(db, "session.login_failed")[0].payload == {"reason": "inactive"}


def test_accounts_of_a_deactivated_facility_cannot_sign_in(db: Session, settings: Settings) -> None:
    facility = make_facility(db, active=False)
    user = make_user(db, role=Role.CLINICIAN, facility=facility)
    with pytest.raises(LoginFailed):
        sign_in(db, settings, user.username, DEFAULT_PASSWORD)


def test_session_slides_at_most_once_a_minute_and_ends_after_idle_time(
    db: Session, settings: Settings
) -> None:
    user = make_user(db)
    t0 = utcnow()
    row, token = start_session(db, settings, user, ip=None, user_agent=None, now=t0)
    assert resolve_session(db, settings, token, now=t0 + timedelta(seconds=30)).state is SessionState.VALID
    assert row.last_seen_at == t0
    resolved = resolve_session(db, settings, token, now=t0 + timedelta(minutes=5))
    assert resolved.state is SessionState.VALID and resolved.session is row
    assert row.last_seen_at == t0 + timedelta(minutes=5)
    late = t0 + timedelta(minutes=35)
    assert resolve_session(db, settings, token, now=late).state is SessionState.IDLE


def test_session_ends_after_twelve_hours_even_when_active(db: Session, settings: Settings) -> None:
    user = make_user(db)
    t0 = utcnow()
    _, token = start_session(db, settings, user, ip=None, user_agent=None, now=t0)
    t = t0
    while t < t0 + timedelta(hours=11, minutes=40):
        t += timedelta(minutes=20)
        assert resolve_session(db, settings, token, now=t).state is SessionState.VALID
    end = t0 + timedelta(hours=12)
    assert resolve_session(db, settings, token, now=end).state is SessionState.ABSOLUTE


def test_missing_unknown_and_revoked_tokens(db: Session, settings: Settings) -> None:
    user = make_user(db)
    now = utcnow()
    row, token = start_session(db, settings, user, ip=None, user_agent=None, now=now)
    assert resolve_session(db, settings, None, now=now).state is SessionState.MISSING
    assert resolve_session(db, settings, "not-a-token", now=now).state is SessionState.MISSING
    logout(db, row, now=now)
    assert row.revoked_at == now
    assert resolve_session(db, settings, token, now=now).state is SessionState.MISSING
    [event] = events_of(db, "session.logout")
    assert event.actor_id == user.id and event.entity_id == row.id


def test_sessions_of_deactivated_users_stop_working(db: Session, settings: Settings) -> None:
    user = make_user(db)
    now = utcnow()
    _, token = start_session(db, settings, user, ip=None, user_agent=None, now=now)
    user.active = False
    db.flush()
    assert resolve_session(db, settings, token, now=now).state is SessionState.MISSING


def test_revoke_sessions_ends_every_open_session(db: Session, settings: Settings) -> None:
    user = make_user(db)
    now = utcnow()
    first, _ = start_session(db, settings, user, ip=None, user_agent=None, now=now)
    second, _ = start_session(db, settings, user, ip=None, user_agent=None, now=now)
    revoke_sessions(db, user, now=now)
    assert first.revoked_at == now and second.revoked_at == now


@pytest.mark.parametrize(
    ("current", "new", "confirm", "message"),
    [
        ("wrong-password", "brand-new-password", "brand-new-password", "Current password not recognised."),
        (DEFAULT_PASSWORD, "brand-new-password", "different-password", "The new passwords do not match."),
        (DEFAULT_PASSWORD, "short", "short", "Use at least 10 characters."),
        (DEFAULT_PASSWORD, DEFAULT_PASSWORD, DEFAULT_PASSWORD, "Choose a password different from the current one."),
    ],
)
def test_password_change_problems(
    db: Session, current: str, new: str, confirm: str, message: str
) -> None:
    user = make_user(db, must_change=True)
    with pytest.raises(PasswordChangeError, match=message):
        change_password(db, user, current, new, confirm)
    assert user.must_change_password is True


def test_password_change_clears_the_forced_change(db: Session) -> None:
    user = make_user(db, must_change=True)
    change_password(db, user, DEFAULT_PASSWORD, "brand-new-password", "brand-new-password")
    assert user.must_change_password is False
    assert verify_password(user.password_hash, "brand-new-password")
    [event] = events_of(db, "user.password_changed")
    assert event.actor_id == user.id and event.entity_id == user.id and event.payload == {}
