"""Response times of a running demo instance, measured from a client (standard library only).

Usage, against a demo instance (``sh scripts/demo.sh`` or the deployed demonstration server)::

    python scripts/perf_check.py --base http://127.0.0.1:8100 [--hospital da] [--users 8]

It signs in with the demo accounts of one hospital (``--hospital``, the demo code in lower case:
``da`` to ``dd``) and measures, in this order:

1. **Saving an admission** (lab technician, ``POST /admissions``): parse, classify, score with
   the active model, write the rows and the event, the whole request. Requirement NFR1: under
   500 ms server-side; this includes the network as well.
2. **Opening the saved admission** (clinician, ``GET /admissions/{code}``).
3. **The hospital report** (facility administrator, ``GET /audit``) and its PDF for the whole
   period the database covers. Requirement NFR1: one facility-year in under 3 s.
4. **Concurrent use**: ``--users`` simultaneous clients, each opening the admission form and
   the report for ``--seconds``; prints throughput and latency percentiles, and counts errors.

Uses only demo accounts and invented patient details, and writes only to the demo database.
Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser

PASSWORD = "demo-password-123"
ADMISSION = {
    "maternal_age": "29",
    "gestational_age_weeks": "39",
    "parity": "2",
    "previous_cs_count": "1",
    "fetal_presentation": "cephalic",
    "plurality": "1",
    "onset_of_labour": "spontaneous",
    "bp_sys": "128",
    "bp_dia": "84",
    "proteinuria": "negative",
    "glucose": "5.2",
}
REPORT_FROM = "2000-01-01"  # the report covers every admission in the demo database
SAVED = re.compile(r"saved=([A-Z]{2,6}-\d+)")  # the form comes back with ?saved=<code>


class _Inputs(HTMLParser):
    """Value of every ``<input>`` a browser would send: hidden, prefilled, checked."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "input":
            a = dict(attrs)
            unchecked = a.get("type") in ("checkbox", "radio") and "checked" not in a
            if a.get("name") and not unchecked:
                self.values[str(a["name"])] = a.get("value") or ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


class Client:
    """One signed-in browser: a cookie jar, no automatic redirects."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), _NoRedirect()
        )

    def request(self, path: str, data: dict[str, str] | None = None) -> tuple[int, str, str]:
        """Status, body and Location header."""
        body = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(self.base + path, data=body)
        try:
            with self.opener.open(req, timeout=60) as resp:
                return resp.status, resp.read().decode(errors="replace"), ""
        except urllib.error.HTTPError as err:
            return err.code, err.read().decode(errors="replace"), err.headers.get("Location", "")

    def inputs(self, path: str) -> dict[str, str]:
        status, html, _ = self.request(path)
        if status != 200:
            raise RuntimeError(f"GET {path}: HTTP {status}")
        parser = _Inputs()
        parser.feed(html)
        return parser.values

    def sign_in(self, username: str) -> Client:
        form = self.inputs("/login")
        status, _, _ = self.request("/login", {**form, "username": username, "password": PASSWORD})
        if status != 303:
            raise RuntimeError(f"sign-in as {username}: HTTP {status}")
        return self


@dataclass
class Timings:
    label: str
    ms: list[float] = field(default_factory=list)
    errors: int = 0

    def timed(self, call: Callable[[], bool]) -> None:
        start = time.perf_counter()
        ok = call()
        self.ms.append((time.perf_counter() - start) * 1000)
        self.errors += 0 if ok else 1

    def row(self) -> str:
        q = statistics.quantiles(self.ms, n=100) if len(self.ms) > 1 else self.ms * 99
        return (
            f"| {self.label} | {len(self.ms)} | {statistics.median(self.ms):.0f} | {q[94]:.0f} "
            f"| {max(self.ms):.0f} | {self.errors} |"
        )


def save_admissions(lab: Client, n: int) -> tuple[Timings, str]:
    timings, code = Timings("Save an admission (POST /admissions)"), ""
    for _ in range(n):
        form = lab.inputs("/admissions/new")
        data = {
            **form,
            **ADMISSION,
            "patient_number": f"PERF-{uuid.uuid4().hex[:8].upper()}",
            "patient_name": "Perf Check Patient",
        }

        start = time.perf_counter()
        status, _, location = lab.request("/admissions", data)
        timings.ms.append((time.perf_counter() - start) * 1000)
        timings.errors += 0 if status == 303 else 1
        found = SAVED.search(location)
        code = found.group(1) if found else code
    return timings, code


def repeat(label: str, client: Client, path: str, n: int) -> Timings:
    timings = Timings(label)
    for _ in range(n):
        timings.timed(lambda: client.request(path)[0] == 200)
    return timings


@dataclass(frozen=True)
class Accounts:
    """The demo accounts of one hospital."""

    lab: str
    clinician: str
    admin: str

    @classmethod
    def of(cls, hospital: str) -> Accounts:
        return cls(f"lab.{hospital}", f"clinician.{hospital}", f"admin.{hospital}")


def concurrent(base: str, who: Accounts, users: int, seconds: float) -> tuple[Timings, float]:
    timings, lock = Timings(f"{users} users at once: form and report pages"), threading.Lock()
    stop = time.perf_counter() + seconds
    report = f"/audit?from={REPORT_FROM}"

    def user(username: str) -> None:
        client = Client(base).sign_in(username)
        paths = ["/admissions/new"] if username == who.lab else [report]
        while time.perf_counter() < stop:
            for path in paths:
                start = time.perf_counter()
                ok = client.request(path)[0] == 200
                elapsed = (time.perf_counter() - start) * 1000
                with lock:
                    timings.ms.append(elapsed)
                    timings.errors += 0 if ok else 1

    names = [who.lab if i % 2 == 0 else who.admin for i in range(users)]
    threads = [threading.Thread(target=user, args=(name,)) for name in names]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return timings, len(timings.ms) / (time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8100")
    parser.add_argument("--hospital", default="da", help="demo hospital code, da to dd")
    parser.add_argument("--saves", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()

    who = Accounts.of(args.hospital)
    lab = Client(args.base).sign_in(who.lab)
    saves, code = save_admissions(lab, args.saves)
    rows = [saves]
    if code:
        clinician = Client(args.base).sign_in(who.clinician)
        rows.append(repeat("Open a saved admission", clinician, f"/admissions/{code}", 50))
    admin = Client(args.base).sign_in(who.admin)
    rows.append(
        repeat("Hospital report, whole period", admin, f"/audit?from={REPORT_FROM}", args.repeats)
    )
    rows.append(
        repeat("Hospital report PDF", admin, f"/audit/report.pdf?from={REPORT_FROM}", args.repeats)
    )
    load, throughput = concurrent(args.base, who, args.users, args.seconds)
    rows.append(load)

    print("| Measure | Requests | p50 ms | p95 ms | max ms | Errors |")
    print("|---|---|---|---|---|---|")
    for timings in rows:
        print(timings.row())
    print(f"\nThroughput with {args.users} users: {throughput:.1f} requests per second.")
    return 1 if any(t.errors for t in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
