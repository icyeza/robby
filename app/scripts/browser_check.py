"""Cross-browser check of the clinician flow against a running demo instance.

Usage, from the app folder, with the demo served (``sh scripts/demo.sh``)::

    uv run --with playwright python scripts/browser_check.py [--base http://127.0.0.1:8100]

For every browser (installed Chrome and Edge, and Playwright's Firefox and WebKit when they have
been installed with ``playwright install firefox webkit``) at a desktop (1366x768) and a phone
(390x844) viewport, it signs in as a demo clinician and then:

1. opens the admission form and times the page load;
2. fills the six Robson inputs and checks the live panel resolves to group 5;
3. checks the save button stays disabled while screening is blank (enforced screening);
4. fills screening, saves, and checks the result page shows group 5 and the readiness signal;
5. fails on any console error or Content-Security-Policy violation caused by the app's own
   pages. Two kinds of message are counted separately, not as failures: requests to other
   origins (browser extensions or local filtering software inject these, the app never makes
   them) and WebKit's report that its own built-in ``<select>`` styling is refused by the CSP,
   which a bare page holding only a ``<select>`` and the same policy also produces.

Writes a screenshot per run to ``browser-check/`` and prints a Markdown table. Uses only demo
accounts and invented patient details, and writes only to the demo database.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from pathlib import Path

from playwright.sync_api import Browser, ConsoleMessage, Error, Page, Playwright, sync_playwright

PASSWORD = "demo-password-123"
USER = "clinician.da"
VIEWPORTS = {"desktop": (1366, 768), "phone": (390, 844)}
ROBSON = {
    "maternal_age": "29",
    "gestational_age_weeks": "39",
    "parity": "2",
    "previous_cs_count": "1",
}
CHOICES = {"fetal_presentation": "cephalic", "plurality": "1", "onset_of_labour": "spontaneous"}
OUT = Path("browser-check")
URL = re.compile(r"https?://[^\s\"'»”�)]+")
WEBKIT_SELECT_NOTICE = "Refused to apply a stylesheet because its hash, its nonce"


def launchers(p: Playwright) -> dict[str, object]:
    return {
        "Chrome": lambda: p.chromium.launch(channel="chrome"),
        "Edge": lambda: p.chromium.launch(channel="msedge"),
        "Firefox": p.firefox.launch,
        "WebKit (Safari engine)": p.webkit.launch,
    }


def check(page: Page, base: str, name: str) -> dict[str, str]:
    errors: list[str] = []
    ignored: list[str] = []

    def on_console(message: ConsoleMessage) -> None:
        if message.type != "error":
            return
        urls = URL.findall(message.text)
        foreign = any(not u.startswith(base) for u in urls)
        if WEBKIT_SELECT_NOTICE in message.text or foreign:
            ignored.append(message.text)
        else:
            errors.append(message.text)

    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(f"{base}/login")
    page.fill('[name="username"]', USER)
    page.fill('[name="password"]', PASSWORD)
    page.click('button[type="submit"]')
    page.wait_for_load_state("networkidle")

    t0 = time.perf_counter()
    page.goto(f"{base}/admissions/new", wait_until="load")
    load_ms = (time.perf_counter() - t0) * 1000

    page.fill('[name="patient_number"]', f"BC-{uuid.uuid4().hex[:6].upper()}")
    page.fill('[name="patient_name"]', "Browser Check")
    for field, value in ROBSON.items():
        page.fill(f'[name="{field}"]', value)
    for field, value in CHOICES.items():
        page.locator(f'label:has(input[name="{field}"][value="{value}"])').click()
    page.wait_for_selector("#robson-panel .strip .on", timeout=10_000)
    live_group = page.locator("#robson-panel .strip .on").inner_text().strip()

    button = page.locator("[data-submit]")
    gated = button.is_disabled()

    page.fill('[name="bp_sys"]', "128")
    page.fill('[name="bp_dia"]', "84")
    page.select_option('[name="proteinuria"]', "negative")
    page.fill('[name="glucose"]', "5.2")
    page.wait_for_function("!document.querySelector('[data-submit]').disabled", timeout=5_000)
    button.click()
    page.wait_for_url(f"{base}/admissions/*-*", timeout=15_000)
    page.wait_for_load_state("networkidle")
    text = page.inner_text("main")
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)

    ok = live_group == "5" and gated and "5" in text and "cesarean" in text.lower() and not errors
    return {
        "live": f"group {live_group}",
        "gate": "blocked" if gated else "NOT blocked",
        "saved": page.url.rsplit("/", 1)[-1],
        "load": f"{load_ms:.0f} ms",
        "errors": str(len(errors)) if not errors else "; ".join(errors)[:120],
        "ignored": str(len(ignored)),
        "result": "pass" if ok else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8100")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    with sync_playwright() as p:
        for label, launch in launchers(p).items():
            try:
                browser: Browser = launch()  # type: ignore[operator]
            except Error as exc:
                reason = str(exc).splitlines()[0][:80]
                rows.append(f"| {label} | | not run: {reason} | | | | | | |")
                continue
            for viewport, (width, height) in VIEWPORTS.items():
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    is_mobile=viewport == "phone" if label != "Firefox" else False,
                )
                page = context.new_page()
                name = f"{label.split()[0].lower()}-{viewport}"
                try:
                    r = check(page, args.base, name)
                except Error as exc:
                    r = {
                        "live": "",
                        "gate": "",
                        "saved": "",
                        "load": "",
                        "errors": str(exc).splitlines()[0][:120],
                        "ignored": "",
                        "result": "FAIL",
                    }
                rows.append(
                    f"| {label} {browser.version} | {viewport} {width}x{height} | {r['live']} | "
                    f"{r['gate']} | {r['saved']} | {r['load']} | {r['errors']} | {r['ignored']} | "
                    f"{r['result']} |"
                )
                context.close()
            browser.close()
    print(
        "| Browser | Viewport | Live panel | Blank screening | Saved | Form load | "
        "App errors | Other notices | Result |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    return 0 if all("FAIL" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
