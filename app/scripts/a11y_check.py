"""Accessibility check (WCAG 2.1 A and AA rules of axe-core) of every screen of a running demo.

Usage, from the app folder, against a demo instance (``sh scripts/demo.sh``, or the deployed
demonstration server through ``ssh -L 8100:127.0.0.1:8170 <server>``)::

    uv run --with playwright python scripts/a11y_check.py [--base http://127.0.0.1:8100]

Signs in with the demo accounts, opens each role's screens in Chrome and runs axe-core on them.
axe is evaluated through the browser's debugging protocol, so the app's strict Content-Security-
Policy stays on (nothing is injected into the page). The script is downloaded once from jsDelivr
at a pinned version and cached in the system temp folder. Prints one Markdown row per rule
violated, with its impact and the screens it was found on; exits 1 on any serious or critical
violation.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

AXE_VERSION = "4.10.2"
AXE_URL = f"https://cdn.jsdelivr.net/npm/axe-core@{AXE_VERSION}/axe.min.js"
PASSWORD = "demo-password-123"
TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]
BLOCKING = {"serious", "critical"}
# (account, screens) per role; {code} is replaced by an admission code of that hospital.
SCREENS: dict[str, list[str]] = {
    "": ["/login"],
    "lab.da": ["/admissions/new", "/admissions", "/lab-requests"],
    "clinician.da": [
        "/patients",
        "/patients?q=a",
        "/admissions/{code}",
        "/admissions/{code}/outcome",
        "/admissions/{code}/lab-requests/new",
    ],
    "admin.da": ["/audit", "/admissions/import", "/admin/users"],
    "custodian": ["/models", "/monitoring", "/approval-log"],
    "reviewer": ["/national"],
    "sysadmin": ["/admin/facilities", "/admin/users"],
}


def axe_source() -> str:
    cached = Path(tempfile.gettempdir()) / f"axe-core-{AXE_VERSION}.min.js"
    if not cached.exists():
        with urllib.request.urlopen(AXE_URL, timeout=30) as response:
            cached.write_bytes(response.read())
    return cached.read_text(encoding="utf-8")


def sign_in(page: Page, base: str, username: str) -> None:
    page.context.clear_cookies()
    page.goto(f"{base}/login")
    if username:
        page.fill('[name="username"]', username)
        page.fill('[name="password"]', PASSWORD)
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")


def an_admission_code(page: Page, base: str) -> str:
    page.goto(f"{base}/patients?q=a")
    link = page.locator('a[href^="/admissions/"]').first
    return (link.get_attribute("href") or "/admissions/none").rsplit("/", 1)[-1]


def audit(page: Page, source: str) -> list[dict[str, object]]:
    page.evaluate(source)
    result = page.evaluate(
        "tags => axe.run(document, {runOnly: {type: 'tag', values: tags}})", TAGS
    )
    return list(result["violations"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8100")
    args = parser.parse_args()
    source = axe_source()
    found: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    checked = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        for account, screens in SCREENS.items():
            sign_in(page, args.base, account)
            code = an_admission_code(page, args.base) if account.startswith("clinician") else ""
            for screen in screens:
                path = screen.replace("{code}", code)
                page.goto(f"{args.base}{path}")
                page.wait_for_load_state("networkidle")
                for v in audit(page, source):
                    key = (str(v["id"]), str(v["impact"]), str(v["help"]))
                    found[key].add(f"{path} ({len(v['nodes'])})")  # type: ignore[arg-type]
                checked += 1
        browser.close()

    print(f"axe-core {AXE_VERSION}, WCAG 2.1 A/AA rules, {checked} screens\n")
    print("| Rule | Impact | What | Screens (elements) |")
    print("|---|---|---|---|")
    for (rule, impact, text), screens in sorted(found.items(), key=lambda x: x[0][1]):
        print(f"| {rule} | {impact} | {text} | {', '.join(sorted(screens))} |")
    if not found:
        print("| none | | no violations | |")
    return 1 if any(impact in BLOCKING for _, impact, _ in found) else 0


if __name__ == "__main__":
    sys.exit(main())
