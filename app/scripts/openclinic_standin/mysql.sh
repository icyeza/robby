#!/bin/sh
# A private MySQL 8 server that stands in for the clinic's OpenClinic database (no system service
# is touched). Usage: sh scripts/openclinic_standin/mysql.sh {init|start|stop|status|shell}
# Settings (environment): OC_DATA (data dir), OC_PORT (default 3307), OC_BIN (MySQL bin dir).
OC_DATA="${OC_DATA:-$HOME/openclinic-mysql}"
OC_PORT="${OC_PORT:-3307}"
OC_BIN="${OC_BIN:-/c/Program Files/MySQL/MySQL Server 8.0/bin}"
HERE="$(cd "$(dirname "$0")" && pwd)"

wait_ready() {
  for _ in $(seq 1 40); do
    "$OC_BIN/mysqladmin" -h 127.0.0.1 -P "$OC_PORT" -u root ping >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "MySQL did not become ready; see $OC_DATA/server.log" >&2
  return 1
}

case "$1" in
  init)
    [ -d "$OC_DATA/mysql" ] && { echo "already initialised: $OC_DATA"; exit 0; }
    "$OC_BIN/mysqld" --initialize-insecure --datadir="$OC_DATA" --console >"$OC_DATA.init.log" 2>&1 \
      || { cat "$OC_DATA.init.log"; exit 1; }
    "$0" start || exit 1
    "$OC_BIN/mysql" -h 127.0.0.1 -P "$OC_PORT" -u root <"$HERE/schema.sql"
    echo "initialised $OC_DATA on port $OC_PORT (database openclinic, read-only user robby_reader)"
    ;;
  start)
    nohup "$OC_BIN/mysqld" --datadir="$OC_DATA" --port="$OC_PORT" --bind-address=127.0.0.1 \
      --mysqlx=OFF --console >"$OC_DATA.server.log" 2>&1 &
    wait_ready && echo "MySQL stand-in running on 127.0.0.1:$OC_PORT"
    ;;
  stop)
    "$OC_BIN/mysqladmin" -h 127.0.0.1 -P "$OC_PORT" -u root shutdown && echo stopped
    ;;
  status)
    "$OC_BIN/mysqladmin" -h 127.0.0.1 -P "$OC_PORT" -u root ping
    ;;
  shell)
    exec "$OC_BIN/mysql" -h 127.0.0.1 -P "$OC_PORT" -u root openclinic
    ;;
  *)
    echo "usage: sh mysql.sh {init|start|stop|status|shell}" >&2
    exit 2
    ;;
esac
