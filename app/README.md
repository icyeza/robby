# Robson Readiness application

The web application of the Robson Readiness system: Robson classification at admission and a
readiness signal for maternity teams. Server-rendered FastAPI + Jinja2 + htmx over PostgreSQL 16.

This folder is its own uv project. It depends on `../ML/packages/robson_engine` and
`../ML/packages/robson_serving` (editable path dependencies; the app declares both itself) and
never on the ML training code (`robson_ml`) or its stack. `tests/test_serving_contract.py` loads a
synthetic packaged artefact and runs the golden check.

## What is here

- **Clinical flow**: the lab technician records each admission on a form with a live Robson panel
  (partial and conflict states) and enforced screening (value or recorded reason), and sees their
  recent entries. The clinician finds a patient by name or number (`/patients`), reads the
  admission on a read-only page with the Robson explanation and readiness signal, and records the
  delivery outcome. Nothing entered can be edited from the web pages.
- **Lab requests**: the clinician asks the lab for urine protein, blood glucose or an HIV rapid
  test from the patient page; the lab technician sees the open requests of the facility (count in
  the navigation) and records each result. An HIV result updates the readiness signal; see
  [Lab requests](#lab-requests).
- **Import**: Excel template download and upload for the facility administrator (rows with a
  problem are left out and listed), and token-authenticated automated import for the hospital's
  patient record system (tokens from `robby-admin api-token create`).
- **Audit**: hospital report with CSV export and comparison with a reference population, and
  national aggregates for the national reviewer.
- **Model governance**: registry with candidate, shadow and active states, approvals with a named
  approver and reason, approval log; shadow models score every admission without being shown.
- **Monitoring**: reconciliation of predictions with outcomes; discrimination (AUC) and calibration
  per facility and per Robson group; calibration and input-drift alerts.
- **Platform**: every table created by Alembic migrations, with PostgreSQL triggers that make
  clinical rows and the event log impossible to update or delete; a hash-chained event log
  (`robby-admin events verify`); sign-in (argon2id, lockout after 5 failed attempts for 15
  minutes), sessions (30 minutes idle, 12 hours absolute), forced password change, CSRF
  protection, strict Content-Security-Policy; facilities and users administration; demo mode.
- The `robby-admin` command line (first administrator, migrations, rule set, model registration,
  reconciliation, API tokens, demo database, event-log verification).
- Deployment: [`../deploy/`](../deploy/) holds the install script, the systemd unit and the nginx
  site used for the live demonstration server (see [Linux server](#linux-server)).
- **FHIR** (FR8): admissions sent as a FHIR R4 Bundle, or fetched from the hospital's FHIR server
  by patient number on the admission form, through an explicit adapter
  ([`src/robby/fhir/`](src/robby/fhir/README.md)); see [FHIR](#fhir) below.

## Data model

Twenty-one tables (`api_token` is not in the diagram below), created by migrations `0001`–`0008` in `migrations/versions/`. The diagram shows
each table's primary key (PK), foreign keys (FK), unique columns (UK) and main business columns,
not every column. Source: [`docs/diagrams/erd.mmd`](../docs/diagrams/erd.mmd); image:
[`erd.png`](../docs/diagrams/erd.png).

```mermaid
erDiagram
    facility {
        uuid id PK
        text name UK
        text code UK "2-6 capital letters"
        text district
        text model_facility_key "facility level the model knows"
        bool active
    }
    app_user {
        uuid id PK
        text username UK "lowercase"
        text password_hash "argon2id"
        enum role "clinician, facility_admin, model_custodian, national_reviewer, system_admin, lab_technician"
        uuid facility_id FK "set only for clinician and facility_admin"
        bool active
        int failed_logins
    }
    user_session {
        uuid id PK
        uuid user_id FK
        char64 token_hash UK "SHA-256 of the cookie token"
        timestamptz expires_at
        timestamptz revoked_at
    }
    rule_set {
        uuid id PK
        text version_label UK
        char64 checksum
        jsonb definition
        bool is_active "at most one true"
    }
    admission_counter {
        uuid facility_id PK, FK
        int next "next admission code number"
    }
    admission {
        uuid id PK
        uuid facility_id FK
        text code UK "e.g. DA-0042"
        timestamptz admitted_at
        enum source "form, fhir, import"
        uuid entered_by FK "required unless source is fhir"
        uuid supersedes_id FK "set on a correction"
        text correction_reason "required with supersedes_id"
        bytea patient_number "encrypted; required unless source is fhir"
        bytea patient_name "encrypted; required unless source is fhir"
        char patient_number_index "blind index of the number"
        int parity
        int previous_cs_count
        int plurality
        text fetal_presentation
        numeric gestational_age_weeks
        text onset_of_labour
        int maternal_age
        int systolic_bp
        int diastolic_bp
        text proteinuria
        numeric glucose_mmol
        jsonb details "optional model inputs"
    }
    screening_omission {
        uuid admission_id PK, FK
        enum field PK "blood_pressure, proteinuria, glucose"
        enum reason "why it was not measured"
    }
    classification {
        uuid id PK
        uuid admission_id FK "unique with rule_set_id"
        uuid rule_set_id FK
        enum status "resolved, partial, conflict"
        int robson_group "1-10, set when resolved"
        text subgroup
        int_array candidates
        jsonb trace "rules checked"
        uuid supersedes_id FK
    }
    model_version {
        uuid id PK
        text version_label UK
        text algorithm
        text feature_set
        char64 artefact_hash
        text artefact_path
        jsonb manifest
        enum state "candidate, shadow, active, retired; one active"
        uuid registered_by FK
        uuid approved_by FK
    }
    model_transition {
        uuid id PK
        uuid model_id FK
        enum from_state
        enum to_state
        uuid actor_id FK
        text reason "not blank"
        timestamptz at
    }
    prediction {
        uuid id PK
        uuid admission_id FK "unique with model_id"
        uuid model_id FK
        enum mode "active, shadow"
        numeric probability "0 to 1"
        jsonb feature_vector "exact row scored"
        uuid supersedes_id FK, UK "set on a re-score"
        uuid lab_result_id FK, UK "the HIV result that prompted a re-score"
    }
    delivery_outcome {
        uuid id PK
        uuid admission_id FK, UK
        enum mode "vaginal_spontaneous, vaginal_assisted, cesarean"
        timestamptz delivered_at
        uuid recorded_by FK
    }
    reconciled_outcome {
        uuid id PK
        uuid prediction_id FK, UK
        uuid delivery_outcome_id FK
        bool cesarean
        timestamptz reconciled_at
    }
    reference_set {
        uuid id PK
        text name UK "unique with version_label"
        enum kind "vogel, c_model"
        text citation
        text version_label UK
        uuid loaded_by FK
    }
    reference_value {
        uuid reference_set_id PK, FK
        int robson_group PK "1-10"
        numeric group_size_pct
        numeric cs_rate_pct
    }
    event_log {
        bigint seq PK
        uuid id UK
        timestamptz occurred_at
        uuid actor_id FK "null for the system"
        text event_type "e.g. admission.created, admissions.imported"
        text entity_type "polymorphic, no FK"
        uuid entity_id "polymorphic, no FK"
        jsonb payload "never a patient name or number"
        char64 prev_hash
        char64 hash UK "hash chain"
    }
    lab_request {
        uuid id PK
        uuid admission_id FK
        enum test "urine_protein, blood_glucose, hiv_rapid"
        bool urgent
        bytea note "encrypted, optional"
        uuid requested_by FK "a clinician"
        timestamptz requested_at
    }
    lab_result {
        uuid id PK
        uuid request_id FK, UK
        bytea value "encrypted"
        uuid recorded_by FK "a lab technician"
        timestamptz resulted_at
    }
    lab_cancellation {
        uuid id PK
        uuid request_id FK, UK
        bytea reason "encrypted, required"
        uuid cancelled_by FK "a clinician"
        timestamptz cancelled_at
    }
    export_record {
        uuid id PK
        uuid facility_id FK
        date period_month "first day of the month"
        uuid exported_by FK
        char64 file_sha256
    }

    facility |o--o{ app_user : "staff of"
    app_user ||--o{ user_session : "signs in with"
    facility ||--o| admission_counter : "numbers codes with"
    facility ||--o{ admission : "admits"
    app_user |o--o{ admission : "enters"
    admission |o--o| admission : "supersedes"
    admission ||--o{ screening_omission : "has"
    admission ||--o{ classification : "classified by"
    rule_set ||--o{ classification : "applied in"
    classification |o--o| classification : "supersedes"
    app_user |o--o{ model_version : "registers or approves"
    model_version ||--o{ model_transition : "moved by"
    app_user ||--o{ model_transition : "acts in"
    admission ||--o{ prediction : "scored as"
    model_version ||--o{ prediction : "produces"
    prediction |o--o| prediction : "supersedes"
    admission ||--o| delivery_outcome : "ends in"
    app_user ||--o{ delivery_outcome : "records"
    prediction ||--o| reconciled_outcome : "reconciled in"
    delivery_outcome ||--o{ reconciled_outcome : "matched to"
    reference_set ||--o{ reference_value : "contains"
    app_user |o--o{ reference_set : "loads"
    app_user |o--o{ event_log : "acts in"
    admission ||--o{ lab_request : "tests asked for"
    lab_request ||--o| lab_result : "answered by"
    lab_request ||--o| lab_cancellation : "withdrawn by"
    lab_result |o--o| prediction : "prompts a re-score"
    app_user ||--o{ lab_request : "requests"
    facility ||--o{ export_record : "exported for"
    app_user ||--o{ export_record : "exports"
```

How to read it:

- **Clinical rows are insert-only.** `admission`, `screening_omission`, `classification`,
  `prediction`, `delivery_outcome`, `reconciled_outcome`, `lab_request`, `lab_result`,
  `lab_cancellation` and `event_log` have PostgreSQL triggers that reject every `UPDATE`,
  `DELETE` and `TRUNCATE`. A lab request's status is derived (requested, resulted or cancelled),
  and a trigger lets a request get a result or a cancellation only once.
- **A re-score is a new prediction.** After an HIV result the readiness signal is scored again:
  the new `prediction` row names the one it replaces (`supersedes_id`) and the result that
  prompted it (`lab_result_id`); the first prediction stays. Pages, monitoring and the evidence
  report read the current prediction of each admission (the one nothing supersedes).
- **A correction is a new row.** The web pages no longer offer corrections, but the schema keeps
  them: a corrected admission points at the one it replaces through `supersedes_id` and must give a
  `correction_reason`; the original stays as it was. A delivery
  outcome stays on the admission version it was recorded on, and reconciliation follows the
  `supersedes_id` chain to match it to every prediction of that chain.
- **Patient identifiers live only on `admission`, encrypted.** `patient_number` and `patient_name`
  are required for form and Excel-import entries (check `entry_has_patient`; rows saved before
  migration `0003` are not re-checked). Both columns hold AES-256-GCM ciphertext, and
  `patient_number_index` a keyed hash of the number for exact search (see
  [Encryption at rest](#encryption-at-rest)). They are never copied into the event log, the
  feature vector or any export. Lab technicians and clinicians see the name and number, facility
  administrators the number only, and the other roles neither. Lab results, notes and
  cancellation reasons are encrypted the same way.
- `event_log.entity_type` / `entity_id` name the row an event is about (an admission, a model, a
  facility) without a foreign key; each entry's `hash` covers the previous entry's `hash`.
- A partial unique index allows only one `model_version` in state `active` and only one active
  `rule_set`.

## How a prediction is made

Source: [`docs/diagrams/prediction-sequence.mmd`](../docs/diagrams/prediction-sequence.mmd); image:
[`prediction-sequence.png`](../docs/diagrams/prediction-sequence.png). The code is
`clinical/routes.py` (`create_admission`), `clinical/service.py` (`save_admission`) and
`clinical/scoring.py`.

```mermaid
sequenceDiagram
    autonumber
    actor L as Lab technician (browser)
    actor C as Clinician (browser)
    participant W as Robby web (FastAPI)
    participant F as Form parser (form.parse)
    participant R as Robson engine (robson_engine)
    participant M as Model runtime (robson_serving)
    participant DB as PostgreSQL
    participant E as Event log

    L->>W: POST /admissions (form fields, CSRF token)
    W->>F: parse(values)
    F-->>W: Parsed admission, or errors
    alt any error
        W-->>L: form again with the errors (nothing saved)
    else valid
        Note over W,DB: service.save_admission, one transaction
        W->>DB: read the active rule_set
        W->>R: classify the six Robson inputs
        R-->>W: status, group, candidates, trace
        W->>DB: next code from admission_counter
        W->>DB: INSERT admission (facility from the session, patient number and name encrypted)
        W->>DB: INSERT screening_omission rows (any not-measured reasons)
        W->>DB: INSERT classification (rule_set_id, status, group, trace)
        opt group resolved
            W->>M: ModelRuntime.active()
            M->>DB: SELECT the model_version in state active
            Note over M: artefact loaded once per process from artefact_path,<br/>manifest, hash and feature schema checked on load
            W->>M: canonical(form values) then score()
            M-->>W: probability and the exact feature vector
            W->>DB: INSERT prediction (mode active, probability, feature_vector)
            W->>M: score with every shadow model
            W->>DB: INSERT prediction (mode shadow), never shown
        end
        Note over W,M: if scoring fails, the admission and classification are still saved
        W->>E: append admission.created (code, group, model, no identifiers)
        E->>DB: INSERT event_log (hash chained to the previous entry)
        W->>DB: COMMIT
        W-->>L: empty form for the next patient, with the note saved DA-0042, group 5
    end

    Note over L,E: Excel import (facility administrator): POST /admissions/import parses every row.<br/>Rows with a problem are left out and listed, the others go through the same<br/>save_admission (source import) in one transaction, plus one admissions.imported event

    C->>W: GET /patients?q=name or number
    W->>DB: admissions of the user's own facility only (number by blind index, name decrypted)
    C->>W: GET /admissions/{code}
    W-->>C: read-only page: Robson group, explanation, readiness signal
    C->>W: POST /admissions/{code}/outcome (mode of delivery)
    W->>DB: INSERT delivery_outcome
    W->>E: append outcome.recorded
    W->>DB: COMMIT
    loop every 15 minutes, or robby-admin reconcile
        W->>DB: join each prediction to the outcome of its admission chain
        W->>DB: INSERT reconciled_outcome (cesarean yes or no)
        W->>E: append reconciliation.run (counts only)
    end
    Note over W,DB: /monitoring (model custodian): AUC and calibration per facility and per Robson group,<br/>input drift against the training distributions
```

How the data is fetched for a prediction: the model never queries old records. It scores the
admission at the moment it is saved, from the values the clinician has just entered (or the row of
the Excel file), mapped onto the model's inputs by `scoring.canonical()`. The prediction row stores
the exact feature vector that was scored and the `model_version` that scored it, so any prediction
can be reproduced later. The artefact is loaded once per process and checked against its manifest;
each save only re-reads which model is active, so promoting a model needs no restart. Past
predictions are read back by admission code, and only within the signed-in user's own facility.
Patient name and number are stored, encrypted, on the admission but are not model inputs and never
reach the feature vector or the event log.

## Lab requests

Source: `src/robby/lab/` (`catalogue.py` tests and result checks, `service.py`, `routes.py`,
`views.py`), migration `0008`.

| Who | Where | What |
|---|---|---|
| Clinician | patient page `/admissions/{code}`, "Request lab tests" | Chooses one or more tests (urine protein dipstick, blood glucose, HIV rapid test), an optional short note and "urgent". A test waiting for the lab cannot be requested again until it has a result or is cancelled. Blood pressure is bedside, not lab, so it is not offered |
| Clinician | patient page, "Cancel request" | Withdraws an open request; the reason is required and kept |
| Lab technician | "Lab requests" `/lab-requests` (in the navigation, with the number of open requests, refreshed every 30 seconds by htmx) | Open requests of the facility, urgent first, then the oldest: patient name and number, admission code, test, note. "Record result" checks the value (dipstick option; glucose 0.5–60 mmol/L, one decimal; HIV positive, negative or indeterminate) |
| Both | patient page (clinician), the result form (lab technician) | Each request with its status, and the result or cancellation reason |

The lab technician keeps entering admissions exactly as before; facility administrators,
national reviewers, model custodians and system administrators see no lab requests or results.

What a result changes:

- **Robson classification**: never (no lab value is a Robson input).
- **Readiness signal**: of the three, only HIV status is a model input (`hiv_positive` in the
  model's feature schema). A positive or negative result re-scores the admission with the active
  model: the current prediction's stored feature vector with `hiv_positive` set, spelled by the
  model itself, so the row is the one the form would have produced had the status been entered
  at admission (`tests/test_rederivation.py` checks this and re-derives the new probability). The
  new prediction supersedes the old one, which stays on record; the patient page shows the
  current signal and says it was updated after the HIV result. Indeterminate is recorded with no
  re-score. If re-scoring fails, the result is still saved and the earlier signal stays (NFR7).
  Shadow models are not re-scored.
- **Proteinuria and glucose**: shown on the patient page as "Screening completed later" next to
  the "Not measured" entry of the admission; no re-score.

Privacy: results, notes and reasons are encrypted at rest. The events `lab.requested`,
`lab.resulted` and `lab.cancelled` carry the admission code, the facility code and the test names
only, never a value, a note, a reason or a patient identifier. Every route checks the role and
takes the facility from the signed-in user, so another facility's admission or request is a 404.
The badge's polling does not move the session's idle timer, so an unattended screen still signs
out after 30 minutes.

## Settings

Set as environment variables or in `app/.env` (git-ignored; start from `.env.example`).
Environment variables win over the file.

| Variable | Default | Meaning |
|---|---|---|
| `ROBBY_DATABASE_URL` | (required) | `postgresql+psycopg://user:password@host:5432/robby` |
| `ROBBY_SECRET_KEY` | (required) | At least 32 bytes; signs CSRF tokens. `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `ROBBY_FIELD_KEY` | (required) | 32 random bytes, base64url; encrypts patient numbers and names. `uv run robby-admin keys generate`. Never change it on a database that holds patients, and back it up apart from the database ([Encryption at rest](#encryption-at-rest)) |
| `ROBBY_SECURE_COOKIES` | `true` | `false` only for local development over plain http |
| `ROBBY_SESSION_IDLE_MINUTES` | `30` | Session ends after this long without a request |
| `ROBBY_SESSION_ABSOLUTE_HOURS` | `12` | Session ends this long after sign-in regardless |
| `ROBBY_TIMEZONE` | `Africa/Kigali` | Zone used to show times (all times are stored in UTC) |
| `ROBBY_RULE_SET_PATH` | rule set shipped in `robson_engine` | Robson rule set YAML for `robby-admin rules load` |
| `ROBBY_ARTEFACT_DIR` | `artefacts` | Where registered model artefacts live |
| `ROBBY_DEMO` | `false` | Shows the example-data banner (demo database only) |
| `ROBBY_RECONCILE_INTERVAL_MINUTES` | `15` | How often the app reconciles predictions with recorded outcomes in the background; `0` turns the in-app job off |
| `ROBBY_FHIR_BASE_URL` | (unset) | The hospital's FHIR server; when set, the admission form shows "Fetch from hospital system" |
| `ROBBY_FHIR_TIMEOUT_SECONDS` | `3` | Longest the whole fetch may take before the form says the hospital system did not answer |
| `ROBBY_FHIR_PATIENT_ID_SYSTEM` | (unset) | Identifier system of the hospital's patient number; unset: the Patient's first identifier |
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

1. Settings: `cp .env.example .env`, then set the two URLs, a fresh `ROBBY_SECRET_KEY`, a
   `ROBBY_FIELD_KEY` from `uv run robby-admin keys generate` and `ROBBY_SECURE_COOKIES=false`.
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

`../deploy/install.sh` installs everything on Ubuntu 24.04 (its own user, its own PostgreSQL 16
cluster, systemd unit, nginx site) and can be re-run to update; it serves the demonstration
database. The steps it automates, for a manual install against a real database:

```bash
sudo apt install postgresql-16          # or the distribution's PostgreSQL 16 package
sudo -u postgres psql -c "CREATE ROLE robby LOGIN PASSWORD '<strong password>'"
sudo -u postgres psql -c "CREATE DATABASE robby OWNER robby TEMPLATE template0 ENCODING 'UTF8'"
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repository> robby && cd robby/app
uv sync --no-dev
export ROBBY_DATABASE_URL=postgresql+psycopg://robby:<password>@localhost:5432/robby
export ROBBY_SECRET_KEY=<48 random url-safe characters>
export ROBBY_FIELD_KEY=<output of: uv run robby-admin keys generate>   # back it up separately
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
| `uv run robby-admin db upgrade [REVISION]` | `alembic upgrade` (default `head`) against `ROBBY_DATABASE_URL`; migration `0007` needs `ROBBY_FIELD_KEY` |
| `uv run robby-admin keys generate` | Prints a new `ROBBY_FIELD_KEY` (for a new installation only) |
| `uv run robby-admin rules load [--path FILE]` | Validates a rule set YAML (structure and checksum) and makes it the active rule set; refuses changed content under an existing version label |
| `uv run robby-admin reconcile` | Joins every prediction to the outcome of its admission (following corrections) in `reconciled_outcome`; safe to repeat; logs one `reconciliation.run` event with counts |
| `uv run robby-admin events verify` | Recomputes the event-log hash chain; exit code 1 if it is broken |
| `uv run alembic revision -m "..."` | New migration (hand-written; see `migrations/versions/0001_initial.py`) |

## Automated import (OpenClinic and other systems)

A clinic's script can send admissions straight to the app instead of someone uploading a file.
The records go through exactly the same checks as the web upload (all or nothing), are saved as
`import` admissions of the token's facility, and a record that was already recorded (same patient
number and admission time) is skipped, so sending a batch twice is safe.

1. Create a lab technician account for the clinic in the Users page, then a token for it:
   ```bash
   uv run robby-admin api-token create --username lab.muhima --label "Lab system feed"
   uv run robby-admin api-token list
   uv run robby-admin api-token revoke <id>
   ```
   The token (`rbk_...`) is shown once; only its SHA-256 is stored. It acts as that lab
   technician, so it can only ever write to that technician's facility. Create and revoke are in the event log.
2. Send records to `POST /api/v1/admissions/import` with `Authorization: Bearer <token>`:
   - `Content-Type: application/json`, body `{"rows": [{"Patient ID": "...", ...}]}`;
   - or the `.xlsx` file as the body (`Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`);
   - or `multipart/form-data` with the file in the field `file`.

   Column names are those of the app's Excel template, or of an OpenClinic export (see
   `src/robby/clinical/openclinic.py`: the OpenClinic column names there are an assumption until a
   real export is available).
3. Answers: `201` imported (`imported`, `skipped_duplicates`, and per record the admission `code`,
   `robson_group`, `readiness_percent`); `200` nothing new to import; `422` a list of `problems`
   (`row`, `column`, `message`; JSON records are numbered from 1) and nothing saved; `401` bad,
   revoked or missing token; `400`/`415` unreadable body; `503` no active rule set.

`scripts/openclinic_push.py` is a ready client (standard library only; takes an `.xlsx` or `.csv`
export, reads the token from `ROBBY_API_TOKEN`). Run it from Task Scheduler or cron after
OpenClinic's export:

```bash
ROBBY_API_TOKEN=rbk_... python openclinic_push.py export.csv --url https://robby.example.rw
```

Serve the API over TLS only; a token sent over plain http can be read by anyone on the path.
There is no rate limit yet.

## FHIR

Admissions can also arrive as FHIR R4 resources (Patient, Encounter, Observation, Condition). An
explicit adapter maps them onto the admission template, so they pass the same checks as the form
and the Excel import. The codes understood, the units and the placeholder codes are listed in
[`src/robby/fhir/README.md`](src/robby/fhir/README.md).

- **Push**: `POST /api/v1/fhir/Bundle` with the same bearer token as the import API and
  `Content-Type: application/fhir+json` (or `application/json`); one admission per Encounter,
  saved with source `fhir`. Answers: `201`/`200` a `transaction-response` Bundle (per Encounter:
  status, `location` `/admissions/<code>`, Robson group and readiness signal as extensions);
  `422` an `OperationOutcome` naming every problem by FHIR element (for example
  `Observation/obs-3.valueQuantity`), nothing saved; `401`, `400`, `415`, `503` as for the import
  API, as OperationOutcomes. `GET /api/v1/fhir/metadata` is the CapabilityStatement.
- **Pull**: with `ROBBY_FHIR_BASE_URL` set, the lab technician's form has "Fetch from hospital
  system" next to the patient number. It pre-fills the form for review and never saves. If the
  hospital system is down, slow (`ROBBY_FHIR_TIMEOUT_SECONDS`) or answers an error, the form says
  so and works as usual (NFR7).
- **Lab requests**: `GET /api/v1/fhir/ServiceRequest?status=active` lists the facility's open
  lab requests and `POST /api/v1/fhir/Observation` (with `basedOn` the ServiceRequest) records a
  result, checked like the lab technician's form (same token; codes in the FHIR README).

Try it with the examples and the stand-in server (from `app/`):

```bash
ROBBY_API_TOKEN=rbk_... python scripts/fhir_push.py scripts/fhir_examples/two_admissions.json \
  --url http://127.0.0.1:8000 --recent      # --recent moves the example times to today
uv run uvicorn --app-dir scripts/fhir_standin server:app --port 8091     # stand-in hospital
ROBBY_FHIR_BASE_URL=http://127.0.0.1:8091 uv run uvicorn robby.main:create_app --factory
```

`scripts/fhir_standin/README.md` lists the stand-in's patient numbers (complete, not measured,
unusable values, no admission in progress, slow, failing).

## Reconciliation and monitoring

Monitoring (`/monitoring`) and the calibration figures need each prediction joined to the outcome of
its admission. Two ways to run that job; use one:

- **In the app (default).** Each Uvicorn process starts a background task that reconciles every
  `ROBBY_RECONCILE_INTERVAL_MINUTES` (15). A PostgreSQL advisory lock means several workers never
  run it at once; a failure is logged and the app carries on. Quiet runs write no event.
- **systemd timer.** Set `ROBBY_RECONCILE_INTERVAL_MINUTES=0` and run `robby-admin reconcile` from
  a timer (for example every 15 minutes).

The page also shows, per facility and per Robson group, the observed and mean predicted cesarean
rates and the AUC over the last 28 days (a stratum is judged only with at least 50 outcomes and 10 of
each kind).

Alerts are computed when the page is opened (no alerts table): calibration drift per facility over
28 days (mean predicted against observed differs by more than 5 points, with at least 50 outcomes
and 10 of each kind), input drift (PSI above 0.20 against the active model's
`reference_distributions.json`, from at least 50 recently scored admissions), and a facility with
outcomes recorded for under 70% of admissions scored 3 or more days ago (at least 20 such
admissions). "Acknowledge" writes a `monitoring.alert_acknowledged` event; the alert key contains
the ISO week, so an acknowledged alert that is still present reappears the next week.

## Demo mode

A complete, synthetic example (4 demo hospitals, ~700 admissions over 90 days, outcomes, audit and
national pages) lives in a SEPARATE database named `<your database>_demo` (for example
`robby_demo`) on the same server. Clinical rows and the event log cannot be deleted, so demo data is
never mixed into the real database; the commands refuse any database whose name does not end in `_demo`.

```bash
uv run robby-admin demo load            # create, migrate, fill (about 2 minutes); --admissions, --seed, --force, --database-url
sh scripts/demo.sh                      # serve it on http://127.0.0.1:8100 with the "Example data" banner
uv run robby-admin demo purge --yes     # drop the demo database
```

Everything is created through the real services (classification, the packaged model for predictions,
events and the hash chain). Outcomes are sampled from the model probability with a per-facility offset.
Admissions are entered by the demo lab technicians and outcomes recorded by the demo clinicians.
Each demo hospital's four most recent admissions carry lab requests: two waiting for the lab (an
urgent HIV test at Demo Hospital A), a glucose result and a negative HIV result that re-scored
the readiness signal.
Sign in with `lab.da` ... `lab.dd`, `clinician.da` ... `clinician.dd`, `admin.da` ... `admin.dd`,
`reviewer`, or (behind the scenes) `custodian` and `sysadmin`; the password for all demo accounts is `demo-password-123`. The reference table in the demo
is a labelled placeholder ("DEMO placeholder reference"), not a published source. The model is read from
`../ML/artefacts/packaged/readiness-v1.0` (override with `--model-dir`); without it, admissions are
saved without predictions.

The optional "other details" of the demo admissions are sampled from the marginal distributions stored
with the packaged model (`reference_distributions.json`), so the input-drift check starts quiet; about
1.5% of admissions have none (a larger share would itself register as drift against the training data).
As an intentional demo story, antenatal visits are shifted to the low side for the most recent weeks, so
the monitoring page shows exactly one such drift alert. Gestational age is also always flagged: the
reference has about a third of its values missing, which no resolved admission can reproduce. The load also
registers a copy of the packaged model as candidate `readiness-v1.1` (not activated), so the Models page
shows an Active and a Candidate model for the promote and rollback walkthrough.

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
history can be verified with `robby-admin events verify` but not edited. Patient numbers and names
are encrypted in the database (below).

### Encryption at rest

- **What**: `admission.patient_number` and `admission.patient_name`, the only identifiers the
  database holds, and the lab values (`lab_result.value`; an HIV result is the most sensitive
  value stored), with the free text that may name a patient (`lab_request.note`,
  `lab_cancellation.reason`). The rest is clinical data without identifiers (the event log,
  feature vectors and exports never carry them) and is not encrypted by the application;
  database backups and disks still need their own protection.
- **How** (`src/robby/crypto.py`): AES-256-GCM. Each value is stored as `bytea`: version byte
  `0x01`, a random 12-byte nonce, then ciphertext and tag. The column name is the associated
  data, so a value moved to the other column, altered, or written under another key fails to
  decrypt instead of reading as something else. The ORM type `EncryptedText` encrypts on write
  and decrypts on read, so the code works with plain strings; SQL comparisons on these columns
  raise an error rather than silently matching nothing.
- **Keys**: one master key, `ROBBY_FIELD_KEY` (32 random bytes, base64url, from
  `robby-admin keys generate`). HKDF-SHA256 derives two separate subkeys from it, one for
  encryption and one for the blind index. The key lives only in the environment or `app/.env`,
  never in the database.
- **Blind index and search**: `patient_number_index` is HMAC-SHA256 over the facility id and the
  normalised number (trimmed, upper case, inner spaces collapsed), set on every insert and
  enforced by check `patient_number_indexed`. The same number at two hospitals gives different
  indexes, so rows cannot be linked across facilities by it. The trade-off: a number is found
  only when typed whole (any case), never by a part of it. A name is found by any part of it:
  the search decrypts the names of the signed-in user's facility's current admissions (a few
  thousand rows) in the application and matches there. Import duplicate checks compare indexes.
- **Losing the key loses the identifiers.** Without the key, numbers and names cannot be
  recovered by anyone, including from backups. Back the key up separately from the database
  backups (one backup holding both defeats the encryption). There is no key rotation: changing
  the key of a database that holds patients makes their identifiers unreadable.
- **Migration `0007`** encrypts the identifiers already stored and drops the plaintext columns.
  It rewrites existing admission rows, so it disables the `admission_immutable` trigger for that
  rewrite only and enables it again in the same transaction. It stops before changing anything
  when `ROBBY_FIELD_KEY` is not set. Its downgrade decrypts back to text with the same key.

## Troubleshooting

- **`password authentication failed for user "postgres"`**: ask whoever installed PostgreSQL for
  the `postgres` password; the role and database only need creating once.
- **`ZoneInfoNotFoundError` on Windows**: run `uv sync` (the `tzdata` package provides the zones).
- **Sign-in works but every page sends you back to sign-in, locally**: set
  `ROBBY_SECURE_COOKIES=false` when using plain http.
- **`create-system-admin` seems to hang when its input is piped**: the password prompt reads
  from the terminal; pass `--password` in scripts.
