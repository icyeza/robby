"""CSRF tokens (session HMAC and signed sign-in nonce) and the role check dependency."""

import uuid

import pytest
from pydantic import SecretStr

from robby.auth.dependencies import CurrentUser, Forbidden, PasswordChangeRequired, require
from robby.config import Settings
from robby.models import AppUser, Role, UserSession
from robby.web.csrf import check_login_token, new_login_token, session_csrf_token, tokens_match

OTHER_SECRET = SecretStr("another-secret-key-that-is-long-enough-012345")


def test_session_token_is_stable_and_bound_to_the_session_and_the_secret(settings: Settings) -> None:
    first, second = uuid.uuid4(), uuid.uuid4()
    token = session_csrf_token(settings, first)
    assert len(token) == 64
    assert session_csrf_token(settings, first) == token
    assert session_csrf_token(settings, second) != token
    assert session_csrf_token(settings.model_copy(update={"secret_key": OTHER_SECRET}), first) != token


def test_tokens_match_only_identical_strings() -> None:
    assert tokens_match("abc", "abc")
    assert not tokens_match("abc", "abd")
    assert not tokens_match(None, "abc")
    assert not tokens_match(123, "abc")


def test_login_token_round_trip(settings: Settings) -> None:
    nonce, token = new_login_token(settings)
    assert check_login_token(settings, token, nonce)
    assert not check_login_token(settings, token, "another-nonce")
    assert not check_login_token(settings, token, None)
    assert not check_login_token(settings, "", nonce)
    assert not check_login_token(settings, token + "x", nonce)
    other = settings.model_copy(update={"secret_key": OTHER_SECRET})
    assert not check_login_token(other, token, nonce)


def test_login_token_expires(settings: Settings) -> None:
    nonce, token = new_login_token(settings)
    assert not check_login_token(settings, token, nonce, max_age=-1)


def current(role: Role, must_change: bool = False) -> CurrentUser:
    user = AppUser(username="u", password_hash="h", role=role, must_change_password=must_change)
    return CurrentUser(user=user, session=UserSession())


def test_require_admits_listed_roles() -> None:
    signed_in = current(Role.CLINICIAN)
    assert require(Role.CLINICIAN, Role.FACILITY_ADMIN)(signed_in) is signed_in


def test_require_refuses_other_roles() -> None:
    with pytest.raises(Forbidden):
        require(Role.SYSTEM_ADMIN)(current(Role.CLINICIAN))


def test_require_without_roles_admits_everyone() -> None:
    for role in Role:
        require()(current(role))


def test_a_forced_password_change_comes_before_any_role_check() -> None:
    with pytest.raises(PasswordChangeRequired):
        require(Role.SYSTEM_ADMIN)(current(Role.CLINICIAN, must_change=True))
