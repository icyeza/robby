"""Send a FHIR Bundle of admissions to the Robson Readiness FHIR endpoint. Standard library only,
so it runs on a clinic computer with any Python 3.9+ and nothing installed.

    python fhir_push.py scripts/fhir_examples/two_admissions.json --url https://robby.example.rw
    python fhir_push.py bundle.json --url http://127.0.0.1:8000 --recent

The token comes from the environment (ROBBY_API_TOKEN) or --token. --recent moves every
Encounter's start to the last few hours (keeping their order), so the example files, whose dates
are fixed, are accepted (the app refuses admissions more than 30 days old). Sending the same
Bundle twice is safe: admissions already recorded are skipped. Exit code 0 = sent (even if all
were skipped), 1 = rejected (nothing was saved; the issues are printed), 2 = could not reach the
server."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

FHIR_JSON = "application/fhir+json"


def restamp(bundle: dict, now: datetime) -> dict:
    """Every Encounter's period.start moved so the latest is one hour before now."""
    encounters = [
        e["resource"] for e in bundle.get("entry", [])
        if e.get("resource", {}).get("resourceType") == "Encounter"
    ]  # fmt: skip
    for k, enc in enumerate(encounters):
        start = now - timedelta(hours=len(encounters) - k)
        enc.setdefault("period", {})["start"] = start.isoformat(timespec="seconds")
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", help="FHIR Bundle (JSON)")
    parser.add_argument("--url", required=True, help="Address of the Robson Readiness server")
    parser.add_argument("--token", default=os.environ.get("ROBBY_API_TOKEN"))
    parser.add_argument("--recent", action="store_true", help="move admission times to today")
    args = parser.parse_args()
    if not args.token:
        print("No token: set ROBBY_API_TOKEN or pass --token.", file=sys.stderr)
        return 2
    with open(args.file, encoding="utf-8") as f:
        bundle = json.load(f)
    if args.recent:
        bundle = restamp(bundle, datetime.now(timezone.utc).astimezone())  # noqa: UP017
    request = urllib.request.Request(
        args.url.rstrip("/") + "/api/v1/fhir/Bundle",
        data=json.dumps(bundle).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {args.token}", "Content-Type": FHIR_JSON,
                 "Accept": FHIR_JSON},
    )  # fmt: skip
    try:
        with urllib.request.urlopen(request) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
    except OSError as exc:
        print(f"Could not reach {args.url}: {exc}", file=sys.stderr)
        return 2
    result = json.loads(body or b"{}")
    if status in (200, 201):
        for entry in result.get("entry", []):
            response = entry["response"]
            text = response["outcome"]["issue"][0]["diagnostics"]
            print(f"{response['status']:<12} {text}")
        return 0
    print(f"Rejected (HTTP {status}). Nothing was saved.", file=sys.stderr)
    for issue in result.get("issue", []):
        where = ", ".join(issue.get("expression", [])) or "-"
        print(f"  {where} · {issue.get('diagnostics', '')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
