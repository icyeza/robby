"""Read maternity admissions straight from the clinic's OpenClinic database and send them to the
Robson Readiness import API.

    uv run --with pymysql python connector.py --url http://127.0.0.1:8100 --token rbk_...
    ... --dry-run          show what would be sent, send nothing

Settings come from options or environment: ROBBY_API_TOKEN, OC_DB_HOST (127.0.0.1), OC_DB_PORT
(3307), OC_DB_USER (robby_reader), OC_DB_PASSWORD, OC_DB_NAME (openclinic). The database account
only needs SELECT. The SQL is in query.sql, the one file that knows OpenClinic's tables.

Each run sends everything admitted in the last --days days (default 25; the app accepts admission
times up to 30 days old). Records the app already holds are skipped on its side, so running this
often, or twice, is safe. Exit code: 0 = sent, 1 = the app rejected the batch (nothing saved),
2 = could not reach the database or the app."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pymysql

BATCH = 500  # the app's limit per request


def fetch_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    since = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d %H:%M:%S")
    sql = Path(args.query).read_text(encoding="utf-8").replace(":since", f"'{since}'")
    db = pymysql.connect(
        host=args.db_host, port=args.db_port, user=args.db_user,
        password=args.db_password, database=args.db_name,
        cursorclass=pymysql.cursors.DictCursor,
    )  # fmt: skip
    try:
        with db.cursor() as cur:
            cur.execute(sql)
            return list(cur.fetchall())
    finally:
        db.close()


def send(url: str, token: str, rows: list[dict[str, object]]) -> tuple[int, dict]:
    body = json.dumps({"rows": rows}, default=str).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/api/v1/admissions/import", data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )  # fmt: skip
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True, help="Address of the Robson Readiness server")
    parser.add_argument("--token", default=os.environ.get("ROBBY_API_TOKEN"))
    parser.add_argument("--days", type=int, default=25, help="Lookback window (default 25)")
    parser.add_argument("--query", default=str(Path(__file__).with_name("query.sql")))
    parser.add_argument("--db-host", default=os.environ.get("OC_DB_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("OC_DB_PORT", "3307")))
    parser.add_argument("--db-user", default=os.environ.get("OC_DB_USER", "robby_reader"))
    parser.add_argument("--db-password", default=os.environ.get("OC_DB_PASSWORD", "reader-password-123"))
    parser.add_argument("--db-name", default=os.environ.get("OC_DB_NAME", "openclinic"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.token and not args.dry_run:
        print("No token: set ROBBY_API_TOKEN or pass --token.", file=sys.stderr)
        return 2
    try:
        rows = fetch_rows(args)
    except (pymysql.MySQLError, OSError) as exc:
        print(f"Could not read the OpenClinic database: {exc}", file=sys.stderr)
        return 2
    print(f"Read {len(rows)} maternity admissions from the last {args.days} days.")
    if args.dry_run or not rows:
        if rows:
            print(json.dumps(rows[0], default=str, indent=2))
        return 0
    imported = already = 0
    for start in range(0, len(rows), BATCH):
        try:
            status, result = send(args.url, args.token, rows[start : start + BATCH])
        except OSError as exc:
            print(f"Could not reach {args.url}: {exc}", file=sys.stderr)
            return 2
        if status not in (200, 201):
            print(f"Rejected (HTTP {status}). Nothing from this batch was saved.", file=sys.stderr)
            for p in result.get("problems", []):
                row = start + p["row"]
                print(f"  record {row} ({rows[row - 1].get('Patient ID')}) · {p['column'] or '-'} · {p['message']}", file=sys.stderr)
            if "error" in result:
                print(f"  {result['error']}", file=sys.stderr)
            return 1
        imported += result["imported"]
        already += len(result.get("already_recorded", [])) + len(result.get("skipped_duplicates", []))
    print(f"Imported {imported} new admissions; {already} were already recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
