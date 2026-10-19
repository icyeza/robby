"""Send an OpenClinic export to the Robson Readiness import API. Standard library only, so it runs
on a clinic computer with any Python 3.9+ and nothing installed.

    python openclinic_push.py export.xlsx   --url https://robby.example.rw
    python openclinic_push.py export.csv    --url https://robby.example.rw

The token comes from the environment (ROBBY_API_TOKEN) or --token. Run it from a scheduler (Task
Scheduler on Windows, cron on Linux) after OpenClinic's export. Sending the same file twice is
safe: records already recorded are skipped. Exit code 0 = sent (even if all were skipped),
1 = rejected (nothing was saved; the problems are printed), 2 = could not reach the server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def build_request(path: str, url: str, token: str) -> urllib.request.Request:
    endpoint = url.rstrip("/") + "/api/v1/admissions/import"
    if path.lower().endswith(".xlsx"):
        with open(path, "rb") as f:
            body, kind = f.read(), XLSX
    else:  # CSV: first row is the header, as OpenClinic exports it
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
        body, kind = json.dumps({"rows": rows}).encode("utf-8"), "application/json"
    return urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": kind},
    )  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("file", help="OpenClinic export (.xlsx or .csv)")
    parser.add_argument("--url", required=True, help="Address of the Robson Readiness server")
    parser.add_argument("--token", default=os.environ.get("ROBBY_API_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        print("No token: set ROBBY_API_TOKEN or pass --token.", file=sys.stderr)
        return 2
    try:
        with urllib.request.urlopen(build_request(args.file, args.url, args.token)) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
    except OSError as exc:
        print(f"Could not reach {args.url}: {exc}", file=sys.stderr)
        return 2
    result = json.loads(body or b"{}")
    if status in (200, 201):
        skipped = len(result.get("skipped_duplicates", []))
        print(f"Imported {result['imported']} admissions; {skipped} already recorded.")
        return 0
    print(f"Rejected (HTTP {status}). Nothing was saved.", file=sys.stderr)
    for p in result.get("problems", []):
        print(f"  row {p['row']} · {p['column'] or '-'} · {p['message']}", file=sys.stderr)
    if "error" in result:
        print(f"  {result['error']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
