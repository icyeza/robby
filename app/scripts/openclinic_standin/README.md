# Stand-in OpenClinic (for trying the whole flow)

**This is not OpenClinic.** The real OpenClinic GA could not be installed or inspected here (it
needs Java and the vendor installer, and its schema is not documented where we could read it). This
folder is a private MySQL database with an invented, plausible maternity layout, filled with
invented patients, plus a connector that reads it and sends it to the app. It exists so the
integration can be demonstrated end to end. When a clinic supplies its real schema, only
`query.sql` has to change.

```
OpenClinic (MySQL, port 3307)  --connector.py (read-only SQL)-->  POST /api/v1/admissions/import
                                                                  (token) --> app --> Robson group
```

## Run it

From `app/` (Git Bash). `uv run --with pymysql` supplies the MySQL driver without changing the project.

```bash
sh scripts/openclinic_standin/mysql.sh init     # once: private MySQL on 127.0.0.1:3307
sh scripts/openclinic_standin/mysql.sh start    # later sessions (also: stop, status, shell)
uv run --with pymysql python scripts/openclinic_standin/seed.py seed   # 60 maternity admissions + 15 others

# a feed token for a demo clinician (or create one on the Integrations page as admin.da)
ROBBY_DATABASE_URL=postgresql+psycopg://robby_dev@127.0.0.1:55432/robby_demo \
  uv run robby-admin api-token create --username clinician.da --label "OpenClinic connector"

export ROBBY_API_TOKEN=rbk_...                  # PowerShell: $env:ROBBY_API_TOKEN="rbk_..."
uv run --with pymysql python scripts/openclinic_standin/connector.py --url http://127.0.0.1:8100
```

Sign in to the demo as `clinician.da` (password `demo-password-123`) and open Recent admissions;
`admin.da` sees the batches on the Integrations page.

## Things to try

| Do this | See this |
|---|---|
| Run the connector twice | second run: `0 new, 60 already recorded` |
| `seed.py add 3`, then the connector | the 3 new patients arrive; the rest are skipped |
| `seed.py add 1 --bad`, then the connector | rejected, names the record and the field; nothing saved; the Integrations page shows "Rejected" |
| Upload the same patients on the Excel page | they appear as "already recorded" with a link |
| `connector.py --dry-run` | shows the first record without sending |

`seed.py reset` empties the stand-in. Rebuilding the demo database (`robby-admin demo load
--force`) removes the imported admissions on the app side.

## Accounts

| What | Value |
|---|---|
| MySQL admin (setup/seed only) | `root`, no password, 127.0.0.1:3307 |
| Connector's database account | `robby_reader` / `reader-password-123`, SELECT only |
| Database | `openclinic` (tables: patient, encounter, obstetric_record, vitals, lab_result) |

## Pointing the connector at a real OpenClinic

1. Get a read-only database account and the schema (table and column names only).
2. Rewrite the FROM/JOIN/WHERE of `query.sql`; keep the quoted column aliases.
3. Value spellings that the app does not recognise are listed in
   `src/robby/clinical/openclinic.py` (fix them there; the connector reports the record and field).
4. Run with `--db-host/--db-port/--db-user/--db-password/--db-name` (or `OC_DB_*`) and schedule it.
