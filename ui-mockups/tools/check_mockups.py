"""Structural checks for the static UI mockups. Stdlib only.

Run from the repo root:  python ui-mockups/tools/check_mockups.py
Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    "index.html",
    "login.html",
    "admission-new.html",
    "admission-incomplete.html",
    "admission-conflict.html",
    "result.html",
    "result-no-model.html",
    "admissions-recent.html",
    "outcome-record.html",
    "correction.html",
    "audit-report.html",
    "models.html",
    "model-detail.html",
]

# Copy rules (spec section 7) apply inside any element with class "signal".
BANNED_IN_SIGNAL = [
    re.compile(r"\brisk\b", re.I),
    re.compile(r"\bshould\b", re.I),
    re.compile(r"\bindicated\b", re.I),
    re.compile(r"(?<!not a )\brecommend", re.I),
]


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.classes: set[str] = set()
        self.title = ""
        self._in_title = False
        self._stack: list[bool] = []  # per open element: is it inside .signal?
        self.signal_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        self.classes.update(cls)
        for key in ("href", "src"):
            if a.get(key):
                self.links.append(a[key])
        if tag == "title":
            self._in_title = True
        inside = (bool(self._stack) and self._stack[-1]) or "signal" in cls
        if tag not in {"br", "img", "input", "meta", "link", "hr", "use"}:
            self._stack.append(inside)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag not in {"br", "img", "input", "meta", "link", "hr", "use"} and self._stack:
            self._stack.pop()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._stack and self._stack[-1]:
            self.signal_text.append(data)


def is_local(link: str) -> bool:
    return not re.match(r"^(https?:|mailto:|#|data:|javascript:)", link)


def check() -> list[str]:
    errors: list[str] = []
    for name in PAGES:
        path = ROOT / name
        if not path.exists():
            errors.append(f"{name}: missing")
            continue
        p = Page()
        p.feed(path.read_text(encoding="utf-8"))
        if not p.title.strip():
            errors.append(f"{name}: no <title>")
        if "assets/styles.css" not in p.links:
            errors.append(f"{name}: does not link assets/styles.css")
        if name != "index.html" and "example-banner" not in p.classes:
            errors.append(f"{name}: no example-data banner")
        for link in p.links:
            if not is_local(link):
                continue
            target = (ROOT / link.split("#")[0]).resolve()
            if not target.exists():
                errors.append(f"{name}: broken link {link}")
        text = " ".join(p.signal_text)
        for rx in BANNED_IN_SIGNAL:
            m = rx.search(text)
            if m:
                errors.append(f"{name}: banned wording '{m.group(0)}' in readiness signal")
        if name == "index.html":
            for other in PAGES[1:]:
                if other not in p.links:
                    errors.append(f"index.html: does not link {other}")
    return errors


def main() -> int:
    errors = check()
    for e in errors:
        print("FAIL", e)
    print(f"{len(PAGES)} pages checked, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
