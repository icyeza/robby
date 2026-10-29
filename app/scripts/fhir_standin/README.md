# Stand-in hospital FHIR server (for trying pull mode)

**This is not a real FHIR server.** It holds a few invented patients in memory and answers only
the four searches the app's "Fetch from hospital system" button makes (see
`src/robby/fhir/README.md`). It exists so pull mode, and what happens when the hospital system
fails, can be shown end to end.

## Run it

From `app/`, in two terminals:

```bash
uv run uvicorn --app-dir scripts/fhir_standin server:app --port 8091
ROBBY_FHIR_BASE_URL=http://127.0.0.1:8091 uv run uvicorn robby.main:create_app --factory --port 8000
```

Sign in as a lab technician, open New admission, type a patient number and press **Fetch from
hospital system**. Admission times are a few hours before the stand-in started.

| Patient number | What the form shows |
|---|---|
| `FX-2001` | Everything filled in: nulliparous, 39 weeks, cephalic, spontaneous labour, screening (group 1) |
| `FX-2002` | Previous cesarean (group 5), gestational age sent in days, glucose not measured (woman declined), HIV recorded |
| `FX-2003` | Some values used; glucose in g/L and "footling" presentation listed as not filled in |
| `FX-2004` | "No admission in progress" (her encounter is finished) |
| `SLOW-1` | Answers after 10 s: "did not answer within 3 seconds", form unchanged |
| `FAIL-1` | HTTP 500: "answered with an error", form unchanged |
| anything else | "has no patient with number ..." |

Stop the stand-in (Ctrl+C) and fetch again to see "could not be reached". Every case leaves the
form fully usable.

`GET http://127.0.0.1:8091/metadata` answers a minimal CapabilityStatement.
