# fhir

Not implemented yet. This package is the planned adapter for FR8: receiving admissions from a
FHIR server instead of the web form, Excel import or JSON API.

The schema is ready for it: `AdmissionSource.FHIR` exists, and an admission from FHIR is the one
kind allowed without an entering user or a patient number and name (see the check constraints on
the `admission` table). Until the adapter exists, nothing writes that source.
