# Robson Readiness application

The web application of the Robson Readiness system: Robson classification at admission and a
readiness signal for maternity teams. Server-rendered FastAPI + Jinja2 + htmx over PostgreSQL 16.

This folder is its own uv project. It depends on `../ML/packages/robson_engine` and
`../ML/packages/robson_serving` (editable path dependencies; the app declares both itself) and
never on the ML training code (`robson_ml`) or its stack. `tests/test_serving_contract.py` loads a
synthetic packaged artefact and runs the golden check.

## What is here (sub-project 1)

- Every database table, created by one Alembic migration, with PostgreSQL triggers that make
  clinical rows and the event log impossible to update or delete.
- A hash-chained event log: every state change appends one entry; `robby-admin events verify`
  recomputes the chain.
- Sign-in (argon2id, lockout after 5 failed attempts for 15 minutes), sessions (30 minutes idle,
  12 hours absolute), forced password change, CSRF protection, strict Content-Security-Policy.
- Facilities and Users administration. Other navigation items show "Not available yet" until
  their sub-project is built.
- The `robby-admin` command line (create the first administrator, migrate, load the rule set,
  verify the event log).

## Settings

Set as environment variables or in `app/.env` (git-ignored; start from `.env.example`).
Environment variables win over the file.

| Variable | Default | Meaning |
|---|---|---|
| `ROBBY_DATABASE_URL` | (required) | `postgresql+psycopg://user:password@host:5432/robby` |
| `ROBBY_SECRET_KEY` | (required) | At least 32 bytes; signs CSRF tokens. `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ROBBY_SECURE_COOKIES` | `true` | `false` only for local development over plain http |
| `ROBBY_SESSION_IDLE_MINUTES` | `30` | Session ends after this long without a request |
| `ROBBY_SESSION_ABSOLUTE_HOURS` | `12` | Session ends this long after sign-in regardless |
| `ROBBY_TIMEZONE` | `Africa/Kigali` | Zone used to show times (all times are stored in UTC) |
| `ROBBY_RULE_SET_PATH` | rule set shipped in `robson_engine` | Robson rule set YAML for `robby-admin rules load` |
| `ROBBY_ARTEFACT_DIR` | `artefacts` | Where registered model artefacts live (used from sub-project 2) |
| `ROBBY_DEMO` | `false` | Shows the example-data banner (demo database only) |
| `ROBBY_RECONCILE_INTERVAL_MINUTES` | `15` | How often the app reconciles predictions with recorded outcomes in the background; `0` turns the in-app job off |
| `ROBBY_TEST_DATABASE_URL` | (tests only) | Server for the test suite; the database name is a prefix |

## Development on Windows (Git Bash)

Needs uv and PostgreSQL 16 (`psql` on PATH). Docker is not used. Two ways to get a database.

### Option A: private development cluster (no system service touched)

`scripts/dev-db.sh` runs a PostgreSQL 16 cluster of its own (trust authentication, listening on
127.0.0.1 only, port 55432, role `robby_dev`, data in `~/robby-pgdata`):

```bash
cd app
sh scripts/dev-db.sh init     # once: creates the cluster, the role robby_dev and the database robby
sh scripts/dev-db.sh start    # also: stop, status
```

Environment variables `ROBBY_PGDATA`, `ROBBY_PGPORT`, `ROBBY_PGBIN` and `ROBBY_PGUSER` change the
data directory, port, PostgreSQL `bin` folder and role. Then in `app/.env`:

```
ROBBY_DATABASE_URL=postgresql+psycopg://robby_dev@127.0.0.1:55432/robby
ROBBY_TEST_DATABASE_URL=postgresql+psycopg://robby_dev@127.0.0.1:55432/postgres
```

### Option B: the installed PostgreSQL

Create the role and database once (asks for the `postgres` password chosen at install):

```bash
psql -U postgres -h localhost -d postgres -c "CREATE ROLE robby LOGIN CREATEDB PASSWORD 'robby-dev-only'"
psql -U postgres -h localhost -d postgres -c "CREATE DATABASE robby OWNER robby TEMPLATE template0 ENCODING 'UTF8'"
```

and put `postgresql+psycopg://robby:robby-dev-only@localhost:5432/robby` in `ROBBY_DATABASE_URL`
(and the same server, database `robby_test`, in `ROBBY_TEST_DATABASE_URL`).

### Then, for either option

1. Settings: `cp .env.example .env`, then set the two URLs, a fresh `ROBBY_SECRET_KEY` and
   `ROBBY_SECURE_COOKIES=false`.
2. Install, migrate, load the rule set, create the first account, run:
   ```bash
   cd app
   uv sync
   uv run robby-admin db upgrade
   uv run robby-admin rules load
   uv run robby-admin create-system-admin --username admin
   uv run uvicorn robby.main:create_app --factory --reload
   ```
   Open http://127.0.0.1:8000 and sign in.

## Linux server

Deployment files (systemd, nginx, TLS) come with sub-project 8. Until then:

```bash
sudo apt install postgresql-16          # or the distribution's PostgreSQL 16 package
sudo -u postgres psql -c "CREATE ROLE robby LOGIN PASSWORD '<strong password>'"
sudo -u postgres psql -c "CREATE DATABASE robby OWNER robby TEMPLATE template0 ENCODING 'UTF8'"
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository> robby && cd robby/app
uv sync --no-dev
export ROBBY_DATABASE_URL=postgresql+psycopg://robby:<password>@localhost:5432/robby
export ROBBY_SECRET_KEY=<48 random url-safe characters>
uv run robby-admin db upgrade
uv run robby-admin rules load
uv run robby-admin create-system-admin --username admin
uv run uvicorn robby.main:create_app --factory --host 127.0.0.1 --port 8000
```

Serve it behind a TLS-terminating reverse proxy; keep `ROBBY_SECURE_COOKIES=true` (the default).
The production role does not need `CREATEDB`.

## Commands

| Command | What it does |
|---|---|
| `uv run robby-admin create-system-admin --username NAME` | Creates a system administrator (the first account); asks for the password twice (at least 10 characters). `--password` skips the prompt, but then the password is visible in the shell history |
| `uv run robby-admin db upgrade [REVISION]` | `alembic upgrade` (default `head`) against `ROBBY_DATABASE_URL` |
| `uv run robby-admin rules load [--path FILE]` | Validates a rule set YAML (structure and checksum) and makes it the active rule set; refuses changed content under an existing version label |
| `uv run robby-admin reconcile` | Joins every prediction to the outcome of its admission (following corrections) in `reconciled_outcome`; safe to repeat; logs one `reconciliation.run` event with counts |
| `uv run robby-admin events verify` | Recomputes the event-log hash chain; exit code 1 if it is broken |
| `uv run alembic revision -m "..."` | New migration (hand-written; see `migrations/versions/0001_initial.py`) |

## Reconciliation and monitoring

Monitoring (`/monitoring`) and the calibration figures need each prediction joined to the outcome of
its admission. Two ways to run that job; use one:

- **In the app (default).** Each Uvicorn process starts a background task that reconciles every
  `ROBBY_RECONCILE_INTERVAL_MINUTES` (15). A PostgreSQL advisory lock means several workers never
  run it at once; a failure is logged and the app carries on. Quiet runs write no event.
- **systemd timer.** Set `ROBBY_RECONCILE_INTERVAL_MINUTES=0` and run `robby-admin reconcile` from
  a timer (for example every 15 minutes).

Alerts are computed when the page is opened (no alerts table): calibration drift per facility over
28 days (mean predicted against observed differs by more than 5 points, with at least 50 outcomes
and 10 of each kind), input drift (PSI above 0.20 against the active model's
`reference_distributions.json`, from at least 50 recently scored admissions), and a facility with
outcomes recorded for under 70% of admissions scored 3 or more days ago (at least 20 such
admissions). "Acknowledge" writes a `monitoring.alert_acknowledged` event; the alert key contains
the ISO week, so an acknowledged alert that is still present reappears the next week. The
readiness board (`/readiness`, clinicians and facility administrators) shows a unit-level expected
cesarean count for the facility's admissions of the last 24 hours without an outcome.

## Demo mode

A complete, synthetic example (4 demo hospitals, ~700 admissions over 90 days, outcomes, corrections,
audit and national pages) lives in a SEPARATE database named `<your database>_demo` (for example
`robby_demo`) on the same server. Clinical rows and the event log cannot be deleted, so demo data is
never mixed into the real database; the commands refuse any database whose name does not end in `_demo`.

```bash
uv run robby-admin demo load            # create, migrate, fill (about 2 minutes); --admissions, --seed, --force, --database-url
sh scripts/demo.sh                      # serve it on http://127.0.0.1:8100 with the "Example data" banner
uv run robby-admin demo purge --yes     # drop the demo database
```

Everything is created through the real services (classification, the packaged model for predictions,
events and the hash chain). Outcomes are sampled from the model probability with a per-facility offset.
Sign in with `clinician.da` ... `clinician.dd`, `admin.da` ... `admin.dd`, `custodian`, `reviewer` or
`sysadmin`; the password for all demo accounts is `demo-password-123`. The reference table in the demo
is a labelled placeholder ("DEMO placeholder reference"), not a published source. The model is read from
`../ML/artefacts/packaged/readiness-v1.0` (override with `--model-dir`); without it, admissions are
saved without predictions.

## Tests and checks

The suite needs a running PostgreSQL and a role with `CREATEDB`. Each run creates a throwaway
database named `<prefix>_<random>` from `ROBBY_TEST_DATABASE_URL`, migrates it, runs every test in
a rolled-back transaction, and drops it.

```bash
uv run pytest                      # everything
uv run ruff check src tests        # lint
uv run ruff format --check src tests
uv run mypy                        # types (src/robby)
```

If a run is killed, a `robby_test_*` database may be left behind; list and drop leftovers with
`psql -U <role> -h <host> -p <port> -d postgres -c "\l robby_test_*"` and `DROP DATABASE "<name>"`.

## Security model

Passwords are hashed with argon2id and never logged or stored in events; five failed sign-ins
lock an account for 15 minutes. Session cookies carry a random 32-byte token (only its SHA-256 is
stored), are `HttpOnly`, `SameSite=Strict` and `Secure` unless `ROBBY_SECURE_COOKIES=false`, and
expire after 30 minutes idle or 12 hours in total; signing out or deactivating a user ends their
sessions. Every form carries a signed CSRF token. Every response sends a Content-Security-Policy
that allows only the application's own origin, and rendered pages contain no inline script, style
or event handler (a test enforces it). Access is checked on the server for each role and
facility, never only by hiding links. Every state change appends an entry to a hash-chained event
log, and database triggers refuse updates and deletes on that log and on clinical rows, so
history can be verified with `robby-admin events verify` but not edited.

## Troubleshooting

- **`password authentication failed for user "postgres"`**: ask whoever installed PostgreSQL for
  the `postgres` password; the role and database only need creating once.
- **`ZoneInfoNotFoundError` on Windows**: run `uv sync` (the `tzdata` package provides the zones).
- **Sign-in works but every page sends you back to sign-in, locally**: set
  `ROBBY_SECURE_COOKIES=false` when using plain http.
- **`create-system-admin` seems to hang when its input is piped**: the password prompt reads
  from the terminal; pass `--password` in scripts.
