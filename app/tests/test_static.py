"""Static assets: the mockups' stylesheet and fonts copied verbatim; htmx vendored and pinned."""

import hashlib
from pathlib import Path

import pytest

import robby

STATIC = Path(robby.__file__).parent / "web" / "static"
MOCKUP_ASSETS = Path(__file__).resolve().parents[2] / "ui-mockups" / "assets"
HTMX_VERSION = "2.0.4"
HTMX_SHA256 = "e209dda5c8235479f3166defc7750e1dbcd5a5c1808b7792fc2e6733768fb447"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_htmx_is_the_pinned_release() -> None:
    assert sha256(STATIC / "htmx.min.js") == HTMX_SHA256


def test_stylesheet_is_the_mockups_stylesheet_unchanged() -> None:
    assert sha256(STATIC / "styles.css") == sha256(MOCKUP_ASSETS / "styles.css")


@pytest.mark.parametrize(
    "name", ["SourceSerif4-Semibold.woff2", "SourceSerif4-Bold.woff2", "OFL.txt"]
)
def test_fonts_and_their_licence_are_copied(name: str) -> None:
    assert sha256(STATIC / "fonts" / name) == sha256(MOCKUP_ASSETS / "fonts" / name)


def test_app_script_sends_the_csrf_header_and_never_evaluates_code() -> None:
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "htmx:configRequest" in script and "X-CSRF-Token" in script
    for forbidden in ("eval(", "new Function", "innerHTML", "document.write"):
        assert forbidden not in script


def test_app_stylesheet_replaces_every_inline_style_the_ported_pages_used() -> None:
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    for selector in (
        ".titlebar.tight",
        ".titlebar .push",
        ".errbox.spaced",
        ".f .fixed",
        ".linkbtn",
        ".seg input:checked + span",
    ):
        assert selector in css
