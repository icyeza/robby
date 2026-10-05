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
    "result-corrected.html",
    "admissions-recent.html",
    "outcome-record.html",
    "correction.html",
    "audit-report.html",
    "models.html",
    "model-detail.html",
    "national-overview.html",
    "national-facility.html",
    "users.html",
    "facilities.html",
    "audit-log.html",
    "unclassified.html",
    "connection-lost.html",
    "session-expired.html",
    "monitoring.html",
]

# Copy rules (spec section 7) apply inside any element with class "signal" or "signal-cell".
BANNED_IN_SIGNAL = [
    re.compile(r"\brisk(s|y)?\b", re.I),
    re.compile(r"\bshould\b", re.I),
    re.compile(r"\bindicat(ed|es|ion)\b", re.I),
    re.compile(r"(?<!not a )\brecommend", re.I),
]
SIGNAL_CLASSES = {"signal", "signal-cell"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr", "use", "path", "circle"}


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.classes: set[str] = set()
        self.title = ""
        self._in_title = False
        self._stack: list[tuple[str, bool]] = []  # (tag, inside a signal element?)
        self.signal_text: list[str] = []

    def _open(self, tag, attrs, push):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        self.classes.update(cls)
        for key in ("href", "src", "action"):
            if a.get(key):
                self.links.append(a[key])
        if tag == "title":
            self._in_title = True
        if push and tag not in VOID:
            inside = (bool(self._stack) and self._stack[-1][1]) or bool(SIGNAL_CLASSES & set(cls))
            self._stack.append((tag, inside))

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs, True)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, False)

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break
        # no matching open tag: ignore the stray end tag

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._stack and self._stack[-1][1]:
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
    for extra in sorted(ROOT.glob("*.html")):
        if extra.name not in PAGES:
            errors.append(f"{extra.name}: page not in PAGES")
    return errors


def _self_test() -> None:
    def parse(html: str) -> Page:
        p = Page()
        p.feed(html)
        return p

    p = parse('<section class="signal"><div></span><p>fine</p></div><p>you should</p></section>')
    text = " ".join(p.signal_text)
    assert any(rx.search(text) for rx in BANNED_IN_SIGNAL), "stray end tag lost the signal context"
    assert BANNED_IN_SIGNAL[1].search(text), "'should' not detected"
    q = parse('<section class="signal"><p>fine</p></section><p>you should</p>')
    assert "should" not in " ".join(q.signal_text), "text after a closed signal section was collected"
    r = parse('<table><tr><td class="signal-cell">no risk here</td><td>risk</td></tr></table>')
    assert " ".join(r.signal_text) == "no risk here", "signal-cell scoping wrong"
    print("self-test passed")


def main() -> int:
    if "--self-test" in sys.argv:
        _self_test()
        return 0
    errors = check()
    for e in errors:
        print("FAIL", e)
    print(f"{len(PAGES)} pages checked, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
