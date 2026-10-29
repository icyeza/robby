#!/bin/sh
# Encrypted backup of Robby's databases, run as root by robby-backup.timer (daily).
#
# Each database of the "robby" PostgreSQL cluster is dumped (pg_dump custom format) and encrypted
# with age to the public key(s) in /etc/robby/backup-recipients.txt before it touches the disk. The
# private key is NOT kept on the server: a stolen disk, snapshot or backup copy cannot be read
# without it. Files land in /srv/robby-backups (root only) with a SHA-256 next to each, and the
# newest KEEP_DAYS days are kept. If ROBBY_BACKUP_REMOTE is set in /etc/robby/backup.env (an rsync
# target such as backup@host:/srv/robby), the new files are also copied off-site; an off-site
# machine can instead pull them with deploy/pull-backups.sh. Everything sent is already encrypted.
#
# The field-encryption key (ROBBY_FIELD_KEY in /etc/robby/robby.env) is not in these backups: keep
# a copy of it apart from them, or the patient identifiers in a restored database are unreadable.
set -eu

PG_PORT="${PG_PORT:-5433}"
OUT=/srv/robby-backups
RECIPIENTS=/etc/robby/backup-recipients.txt
KEEP_DAYS="${KEEP_DAYS:-14}"
ROBBY_BACKUP_REMOTE=""
[ -f /etc/robby/backup.env ] && . /etc/robby/backup.env

[ -s "$RECIPIENTS" ] || { echo "no recipients in $RECIPIENTS" >&2; exit 1; }
umask 077
mkdir -p "$OUT"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

databases="$(sudo -u postgres psql -p "$PG_PORT" -tAc \
    "SELECT datname FROM pg_database WHERE datname LIKE 'robby%' AND NOT datistemplate")"
for db in $databases; do
    file="$OUT/$db-$stamp.dump.age"
    sudo -u postgres pg_dump -p "$PG_PORT" -Fc "$db" | age -R "$RECIPIENTS" -o "$file.part"
    mv "$file.part" "$file"
    (cd "$OUT" && sha256sum "$(basename "$file")" >"$(basename "$file").sha256")
    echo "backed up $db: $(du -h "$file" | cut -f1)"
done

find "$OUT" -name 'robby*.dump.age*' -mtime +"$KEEP_DAYS" -delete

if [ -n "$ROBBY_BACKUP_REMOTE" ]; then
    rsync -a --include='*.age' --include='*.sha256' --exclude='*' "$OUT/" "$ROBBY_BACKUP_REMOTE/"
    echo "copied to $ROBBY_BACKUP_REMOTE"
fi
