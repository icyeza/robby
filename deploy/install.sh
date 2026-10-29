#!/bin/sh
# Install or update Robby on an Ubuntu 24.04 server, as root. Safe to run again: each step checks
# what is already there. It touches nothing outside its own user, cluster, unit and site.
#
#   sudo sh deploy/install.sh            # from a clone, or:
#   curl -LsSf https://raw.githubusercontent.com/icyeza/robby/main/deploy/install.sh | sudo sh
#
# What it sets up:
#   user robby, home /srv/robby, the repository in /srv/robby/robby (branch main)
#   its own PostgreSQL 16 cluster "robby" on 127.0.0.1:5433 (peer authentication for user robby)
#   /etc/robby/robby.env (settings, a fresh secret key; kept on later runs)
#   the demonstration database robby_demo, filled with synthetic data on the first run
#   systemd unit robby.service (Uvicorn, 2 workers, 127.0.0.1:8170) and the nginx site
# TLS afterwards: certbot --nginx -d robby.example.org
set -eu

REPO_URL="${ROBBY_REPO_URL:-https://github.com/icyeza/robby.git}"
HOME_DIR=/srv/robby
SRC="$HOME_DIR/robby"
APP="$SRC/app"
ENV_FILE=/etc/robby/robby.env
PG_PORT=5433
DB=robby_demo
DB_URL="postgresql+psycopg://robby@/$DB?host=/var/run/postgresql&port=$PG_PORT"

say() { printf '\n== %s\n' "$*"; }
as_robby() { sudo -u robby -H sh -c "$1"; }

say "user robby"
if ! id robby >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "$HOME_DIR" --shell /bin/bash robby
fi

say "uv"
if [ ! -x "$HOME_DIR/.local/bin/uv" ]; then
    as_robby 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi
UV="$HOME_DIR/.local/bin/uv"

say "repository"
if [ -d "$SRC/.git" ]; then
    as_robby "git -C $SRC fetch --quiet origin main && git -C $SRC reset --quiet --hard origin/main"
else
    as_robby "git clone --quiet --branch main $REPO_URL $SRC"
fi
as_robby "git -C $SRC log -1 --format='at commit %h %s'"

say "PostgreSQL cluster 16/robby on port $PG_PORT"
if ! pg_lsclusters --no-header | awk '{print $1"/"$2}' | grep -qx '16/robby'; then
    pg_createcluster 16 robby --port "$PG_PORT" --start-conf auto -- --auth-local=peer --auth-host=scram-sha-256
fi
systemctl enable --now postgresql@16-robby >/dev/null
psql_admin() { sudo -u postgres psql -p "$PG_PORT" -tAc "$1"; }
[ "$(psql_admin "SELECT 1 FROM pg_roles WHERE rolname = 'robby'")" = 1 ] || psql_admin "CREATE ROLE robby LOGIN"
[ "$(psql_admin "SELECT 1 FROM pg_database WHERE datname = '$DB'")" = 1 ] ||
    psql_admin "CREATE DATABASE $DB OWNER robby TEMPLATE template0 ENCODING 'UTF8'"

say "settings in $ENV_FILE"
if [ ! -f "$ENV_FILE" ]; then
    mkdir -p /etc/robby
    SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
    cat >"$ENV_FILE" <<EOF
ROBBY_DATABASE_URL="$DB_URL"
ROBBY_SECRET_KEY=$SECRET
ROBBY_SECURE_COOKIES=true
ROBBY_DEMO=true
ROBBY_ARTEFACT_DIR=$APP/artefacts/demo
ROBBY_TIMEZONE=Africa/Kigali
EOF
    chown root:robby "$ENV_FILE"
    chmod 640 "$ENV_FILE"
fi

say "Python environment"
as_robby "cd $APP && $UV sync --quiet --no-dev --frozen"

say "database"
ADMIN="set -a && . $ENV_FILE && set +a && cd $APP && .venv/bin/robby-admin"
if [ "$(sudo -u robby psql -h /var/run/postgresql -p "$PG_PORT" -d "$DB" -tAc \
        "SELECT count(*) FROM pg_tables WHERE schemaname = 'public'")" = 0 ]; then
    # The loader refuses to fill the configured database, which on a real install holds real
    # records; this server serves only the demonstration database, so name an unused one here.
    REAL_URL="postgresql+psycopg://robby@/robby?host=/var/run/postgresql&port=$PG_PORT"
    as_robby "set -a && . $ENV_FILE && set +a && export ROBBY_DATABASE_URL='$REAL_URL' && cd $APP &&
        .venv/bin/robby-admin demo load --database-url '$DB_URL' \
        --model-dir $SRC/ML/artefacts/packaged/readiness-v1.0"
else
    as_robby "$ADMIN db upgrade"
fi

say "systemd unit"
install -m 644 "$SRC/deploy/robby.service" /etc/systemd/system/robby.service
systemctl daemon-reload
systemctl enable robby >/dev/null
systemctl restart robby

say "nginx site"
# Installed once: certbot adds the TLS lines to this file, and a later run must keep them.
if [ ! -f /etc/nginx/sites-available/robby ]; then
    install -m 644 "$SRC/deploy/nginx-robby.conf" /etc/nginx/sites-available/robby
fi
if [ ! -e /etc/nginx/sites-enabled/robby ]; then
    ln -s /etc/nginx/sites-available/robby /etc/nginx/sites-enabled/robby
fi
nginx -t
systemctl reload nginx

say "check"
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS -o /dev/null http://127.0.0.1:8170/login; then
        echo "robby.service answers on 127.0.0.1:8170"
        exit 0
    fi
    sleep 2
done
echo "robby.service did not answer; see: journalctl -u robby -n 50" >&2
exit 1
