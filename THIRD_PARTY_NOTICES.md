# Third-party notices

Items bundled in, adapted in, or referred to by this repository, with their licences. Where a
licence or its conditions could not be verified, the entry says **UNVERIFIED**.

All Python dependencies are installed unmodified from PyPI at the versions pinned in
`app/uv.lock` and `ML/uv.lock`; each keeps its own licence (mostly MIT, BSD, Apache-2.0).
Only the items below are bundled, adapted, or carry conditions that matter here.

| Item | Where | Licence | Notes |
|---|---|---|---|
| htmx 2.0.4 | `app/src/robby/web/static/htmx.min.js` | 0BSD | Vendored unmodified; a one-line credit comment sits above it. |
| Source Serif 4 (Semibold, Bold) | `app/src/robby/web/static/fonts/` | SIL Open Font License 1.1 | Copyright 2014-2023 Adobe, Reserved Font Name "Source". The licence text is in `fonts/OFL.txt`. The fonts are not modified. |
| WHO Robson Classification Implementation Manual (2017) | rule set `ML/packages/robson_engine/src/robson_engine/rules/robson_v1.0.yaml` | CC BY-NC-SA 3.0 IGO | The rule file is an adaptation (see its header for what was changed) and is licensed CC BY-NC-SA 3.0 IGO: attribution, non-commercial, share-alike. WHO does not endorse it. |
| HL7 FHIR R4 | `app/src/robby/fhir/` (resource shapes and mapping) | CC0 1.0 (the specification) | HL7 states it does not identify patents that may need a licence to implement FHIR. "HL7" and "FHIR" are HL7 trademarks. |
| LOINC codes (8 codes and their long names, in `app/src/robby/fhir/mapping.py`) | FHIR mapping | LOINC licence | **UNVERIFIED**: the licence terms could not be read (the site refused the request). Check the LOINC copyright notice and attribution conditions before any deployment with real data. |
| SNOMED CT concept codes (3 codes in `app/src/robby/fhir/mapping.py`) | FHIR mapping | SNOMED International licence | **UNVERIFIED**: whether Rwanda is a SNOMED member country, and so whether use is free of charge, was not confirmed. The repository holds only three identifiers as mapping keys, no SNOMED release files. |
| psycopg and psycopg-binary 3.3.6 | PostgreSQL driver, installed from PyPI | LGPL-3.0-only | Used unmodified, as a normal installed library. The wheel bundles libpq and OpenSSL libraries. If this software is ever distributed (for example as an installable package or image), the LGPL notice and licence text must accompany it and the library must stay replaceable. Hosting it as a service is not distribution (UNVERIFIED reading of LGPL-3.0 section 4). |
| axe-core 4.10.2 | accessibility check `app/scripts/a11y_check.py` | MPL-2.0 | Loaded from a CDN when the check runs; not bundled in this repository. |

The research dataset used to train the model (a de-identified KoboToolbox export from a UR-CMHS
study, 2024) is held under a data-sharing agreement, is not in this repository, and must not be
redistributed. Its exact terms are not recorded here (**UNVERIFIED**).
