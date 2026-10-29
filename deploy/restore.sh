#!/bin/sh
# Restore an encrypted Robby backup into a NEW database of the "robby" cluster, as root.
#
#   sh deploy/restore.sh <backup.dump.age> <age identity file> <new database name>
#
# The identity (private key) is brought to the server only for the restore and should be removed
# afterwards. The checksum is verified first, and an existing database is never overwritten:
# restore under a new name, check it, then point ROBBY_DATABASE_URL at it.
set -eu

[ $# -eq 3 ] || { echo "usage: $0 <backup.dump.age> <identity> <new database>" >&2; exit 2; }
backup="$1"; identity="$2"; target="$3"
PG_PORT="${PG_PORT:-5433}"

if [ -f "$backup.sha256" ]; then
    (cd "$(dirname "$backup")" && sha256sum -c --quiet "$(basename "$backup").sha256")
fi
exists="$(sudo -u postgres psql -p "$PG_PORT" -tAc "SELECT 1 FROM pg_database WHERE datname = '$target'")"
[ -z "$exists" ] || { echo "database $target already exists; choose a new name" >&2; exit 1; }

sudo -u postgres createdb -p "$PG_PORT" -O robby "$target"
age -d -i "$identity" "$backup" | sudo -u postgres pg_restore -p "$PG_PORT" -d "$target" --no-owner --role=robby
echo "restored into $target"
