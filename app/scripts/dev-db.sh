#!/bin/sh
# Start, stop or inspect a private development PostgreSQL cluster (no system service touched).
# Usage: sh scripts/dev-db.sh {init|start|stop|status}
# Settings (environment): ROBBY_PGDATA (data dir), ROBBY_PGPORT (default 55432),
# ROBBY_PGBIN (PostgreSQL bin dir), ROBBY_PGUSER (superuser role, default robby_dev).
PGDATA="${ROBBY_PGDATA:-$HOME/robby-pgdata}"
PGPORT="${ROBBY_PGPORT:-55432}"
PGBIN="${ROBBY_PGBIN:-/c/Program Files/PostgreSQL/16/bin}"
PGUSER_DEV="${ROBBY_PGUSER:-robby_dev}"

case "$1" in
  init)
    [ -f "$PGDATA/PG_VERSION" ] && { echo "already initialised: $PGDATA"; exit 0; }
    "$PGBIN/initdb" -D "$PGDATA" -U "$PGUSER_DEV" --auth=trust -E UTF8 >/dev/null
    "$0" start
    "$PGBIN/createdb" -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER_DEV" robby
    echo "initialised $PGDATA on port $PGPORT"
    ;;
  start)
    "$PGBIN/pg_ctl" -D "$PGDATA" -o "-p $PGPORT -c listen_addresses=127.0.0.1" -l "$PGDATA/server.log" start
    ;;
  stop)
    "$PGBIN/pg_ctl" -D "$PGDATA" stop
    ;;
  status)
    "$PGBIN/pg_ctl" -D "$PGDATA" status
    ;;
  *)
    echo "usage: sh scripts/dev-db.sh {init|start|stop|status}" >&2
    exit 2
    ;;
esac
