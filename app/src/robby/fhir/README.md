# FHIR interoperability (FR8, NFR7)

The app accepts admission data as FHIR R4 Patient, Encounter, Observation and Condition resources
and maps them, through an explicit adapter, onto the same internal admission representation the
web form uses. Nothing FHIR-specific reaches classification or scoring: the adapter writes the
admission template's fields (the columns of the Excel template, the names the form posts), and
those go through `importer.read_table` and `form.parse` with the very same checks as the form and
the Excel import.

```
Bundle (JSON) --adapter.read_bundle--> template rows + where each value came from
              --intake.check---------> importer.read_table / form.parse   (same checks as the form)
              --intake.save----------> service.save_admission(source=fhir) (same classification,
                                       scoring, events and duplicate skip as the import API)
```

| Module | What it does |
|---|---|
| `mapping.py` | The mapping as data: FHIR code -> template field, units, dataAbsentReason codes |
| `adapter.py` | Reads a Bundle: one admission per Encounter, every problem as an `Issue` on a FHIR element |
| `intake.py` | Template table -> importer checks -> issues by element; saving, all or nothing |
| `resources.py` | OperationOutcome, transaction-response Bundle, CapabilityStatement |
| `client.py` | Pull mode: the hospital's FHIR server (`HospitalRecords` protocol, `FhirClient`) |
| `routes.py` | `POST /api/v1/fhir/Bundle`, `GET /api/v1/fhir/metadata`, `POST /admissions/fetch` |

## What a Bundle must hold

- `Bundle.type` `transaction`, `batch` or `collection`; JSON only.
- Per admission: an **Encounter** (`period.start` = admission time, a dateTime with its offset;
  `subject` -> the Patient) and its **Patient**.
- **Observations** and **Conditions** referencing the Encounter (`encounter`), or only the
  Patient (`subject`) when she has exactly one Encounter in the Bundle.
- References may be `Type/id`, a `fullUrl` (`urn:uuid:...`) or an absolute URL ending in `Type/id`.
- Resources of other types, codes not in the mapping, and resources with status
  `entered-in-error` or `cancelled` are ignored. When several Observations fill one field, the
  latest wins (`effectiveDateTime`, else `issued`, else Bundle order).

## The mapping

Each line is one entry of `OBSERVATIONS` or `CONDITIONS` in `mapping.py`; adding a code (another
LOINC for parity, the hospital's own code for presentation) is a one-line change there.

**Patient**

| FHIR | Template field | Notes |
|---|---|---|
| `identifier[0].value`, or the identifier whose `system` is `ROBBY_FHIR_PATIENT_ID_SYSTEM` | Patient number | |
| `name` (the `official` one, else the first): `text`, else `given` + `family` | Patient name | |
| `birthDate` | Maternal age | Years at the admission date; a partial date (year only) gives no age |

**Observations**

| Code | Meaning | Template field | Value |
|---|---|---|---|
| LOINC `11977-6` | Parity | Parity | `valueInteger` (or a unitless `valueQuantity`, `valueString`) |
| LOINC `11996-6` | Gravidity | Gravidity (optional model input) | as above; used only if the active model asks for it |
| LOINC `49051-6` | Gestational age in weeks | Gestational age | `valueQuantity` in `wk` (or `d`, divided by 7), completed weeks; `valueInteger` = weeks |
| LOINC `85354-9` | Blood pressure panel | Systolic / Diastolic BP | components LOINC `8480-6` systolic and `8462-4` diastolic, `valueQuantity` in `mm[Hg]` |
| LOINC `15074-8` | Glucose [Moles/volume] in Blood | Glucose (mmol/L) | `valueQuantity` in `mmol/L`, or `mg/dL` divided by 18.016; rounded to 0.1 |
| LOINC `20454-5` | Protein [Presence] in Urine by Test strip | Proteinuria | `valueCodeableConcept` (code, display or text) or `valueString`: negative/neg/nil, trace, + / 1+, ++ / 2+, +++ / 3+ |
| `previous-cesareans` * | Previous cesareans | Previous cesareans | `valueInteger` |
| `fetuses` * | Number of fetuses | Number of fetuses | `valueInteger` (2 or more = "Two or more"), or code `1` / `2` |
| `fetal-presentation` * | Fetal presentation | Fetal presentation | code `cephalic`, `breech`, `transverse`, `oblique` |
| `labour-onset` * | Onset of labour | Onset of labour | code `spontaneous`, `induced`, `prelabour_cs` |

\* **Placeholder codes** in the project CodeSystem
`https://icyeza.github.io/robby/fhir/CodeSystem/admission`. No LOINC or SNOMED CT code could be
cited with confidence for these four facts, so none was invented. They are to be replaced by the
codes of the hospital's FHIR profile: change the code in its line of `OBSERVATIONS`. Coded values
may also be sent as `valueString` or a CodeableConcept `text` matching the option's value or label
(e.g. "Breech").

**Conditions** (SNOMED CT) set the matching optional model input to "Yes" when the active model
uses it (otherwise they are ignored). A Condition whose `verificationStatus` is `refuted` sets
"No"; `entered-in-error` is ignored. No Condition means "not given", never "No".

| SNOMED CT | Meaning | Model input |
|---|---|---|
| `398254007` | Pre-eclampsia | Pre-eclampsia recorded |
| `11687002` | Gestational diabetes mellitus | Gestational diabetes recorded |
| `86406008` | HIV infection | HIV positive |

**Screening not measured.** The form needs, for blood pressure, proteinuria and glucose, either a
value or the reason it was not measured. An Observation with the screening code, no value and a
`dataAbsentReason` gives the reason:

| `dataAbsentReason` | App's reason |
|---|---|
| `text` naming one of the field's reasons (value or label, any case), e.g. `equipment_unavailable`, `no_time_before_delivery`, "Reagent strips out of stock" | that reason |
| code `asked-declined` | Woman declined |
| any other code (`not-performed`, `unknown`, ...) | Other |

A screening test with neither a value nor a dataAbsentReason is an issue on the Encounter
(push); in pull mode it is simply left for the lab technician to fill in.

## Push: `POST /api/v1/fhir/Bundle`

Same bearer token as `POST /api/v1/admissions/import` (`robby-admin api-token create`): the
token's lab technician gives the facility. `Content-Type: application/fhir+json` or
`application/json`.

| Status | Body |
|---|---|
| `201` (something saved) / `200` (all already recorded) | `Bundle` of type `transaction-response`, one entry per Encounter in Bundle order: `response.status` (`201 Created`, or `200 OK` for already recorded / repeated in the Bundle), `response.location` `/admissions/<code>`, extensions `.../StructureDefinition/admission-code`, `robson-group`, `readiness-percent` (when there is one), and `response.outcome` with the same in words |
| `422` | `OperationOutcome` with every issue: `expression` is the FHIR element (`Observation/obs-3.valueQuantity`, `Encounter/enc-1.period.start`, `Patient/pat-1.identifier[0].value`); nothing saved |
| `400` | not JSON, or not a Bundle |
| `401` | missing, invalid or revoked token (`WWW-Authenticate: Bearer`) |
| `415` | another content type |
| `503` | no active Robson rule set |

All or nothing, like the import API. An admission whose patient number and admission time are
already recorded for the facility is skipped (status `200 OK`, the existing admission's location),
so a Bundle can be sent twice. Events: `admissions.imported`, `api_import.rejected`,
`api_import.nothing_new`, with `channel: fhir`; no patient identifier in any payload.

`GET /api/v1/fhir/metadata` returns a small CapabilityStatement (no token needed).

## Pull: "Fetch from hospital system"

With `ROBBY_FHIR_BASE_URL` set, the lab technician's admission form shows **Fetch from hospital
system** next to the patient number (htmx; without JavaScript the button posts the form). The app
then asks the hospital's server:

```
GET {base}/Patient?identifier=[{ROBBY_FHIR_PATIENT_ID_SYSTEM}|]<number>
GET {base}/Encounter?subject=Patient/<id>&status=in-progress      (the latest one is used)
GET {base}/Observation?encounter=Encounter/<id>
GET {base}/Condition?subject=Patient/<id>     (kept: this Encounter's, and those of no Encounter)
```

runs the same adapter and pre-fills the form. **It never saves**: the lab technician checks the
values and presses Save admission as usual. Values the adapter could not use are listed under
the notice to be entered by hand.

**Graceful degradation (NFR7).** The whole lookup is limited to `ROBBY_FHIR_TIMEOUT_SECONDS`
(default 3). Timeout, refused connection, an HTTP error, an unreadable answer, no such patient or
no admission in progress: the form says so in plain words ("The hospital system did not answer
within 3 seconds. Fill in the form by hand.") and keeps what was typed. The manual form never
depends on the hospital system.

## Examples and the stand-in

- `app/scripts/fhir_examples/two_admissions.json`: a valid transaction Bundle with two
  admissions (one with glucose in mg/dL, one with glucose not measured, `equipment_unavailable`,
  and a pre-eclampsia Condition). `invalid.json`: one admission with seven problems.
- `app/scripts/fhir_push.py`: standard-library client; `--recent` moves the examples' fixed
  admission times to today (the app refuses admissions older than 30 days).
- `app/scripts/fhir_standin/`: a tiny stand-in hospital FHIR server for pull mode (see its README).

## Not supported (yet)

XML; FHIR search on this server; Bundle `request` semantics (every entry is read, none is executed
as a separate create/update); `Observation.component`-level `dataAbsentReason` (use the
Observation's own); profiles or validation beyond what the mapping reads; SMART on FHIR / OAuth
for pull mode (the hospital server is expected on a trusted network or behind the hospital's
gateway).
