"""argon2id hashing, temporary passwords, the password rule and session tokens."""

import hashlib
import re

from robby.auth.passwords import (
    TEMPORARY_ALPHABET,
    generate_temporary_password,
    hash_password,
    needs_rehash,
    password_problem,
    verify_password,
)
from robby.auth.tokens import hash_token, new_session_token


def test_hashes_are_argon2id_and_verify() -> None:
    hashed = hash_password("a-long-password")
    assert hashed.startswith("$argon2id$")
    assert verify_password(hashed, "a-long-password")
    assert not verify_password(hashed, "another-password")


def test_hashes_are_salted() -> None:
    assert hash_password("same-password") != hash_password("same-password")


def test_garbage_hash_never_verifies() -> None:
    assert verify_password("not-a-hash", "anything") is False


def test_fresh_hash_needs_no_rehash() -> None:
    assert needs_rehash(hash_password("a-long-password")) is False


def test_temporary_passwords_are_three_groups_of_four_unambiguous_characters() -> None:
    pattern = re.compile(rf"[{TEMPORARY_ALPHABET}]{{4}}(-[{TEMPORARY_ALPHABET}]{{4}}){{2}}")
    passwords = {generate_temporary_password() for _ in range(50)}
    assert len(passwords) == 50
    assert all(pattern.fullmatch(p) for p in passwords)
    assert not set("ilo01") & set(TEMPORARY_ALPHABET)


def test_password_rule() -> None:
    assert password_problem("short", username="mw.a014") == "Use at least 10 characters."
    assert password_problem("MW.A014-xx", username="mw.a014-xx") == (
        "Do not use your username as your password."
    )
    assert password_problem("a-long-password", username="mw.a014") is None


def test_session_tokens_are_random_and_stored_as_sha256() -> None:
    token, token_hash = new_session_token()
    assert len(token) >= 43
    assert token_hash == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert hash_token(token) == token_hash
    assert new_session_token()[0] != token
