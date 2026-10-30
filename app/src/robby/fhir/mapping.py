"""Which FHIR codes fill which admission field: the mapping as data (FR8).

Every Observation code the adapter understands is one line of OBSERVATIONS, every Condition one
line of CONDITIONS. Adding a code a hospital sends (another LOINC for parity, say) is adding a
line here; nothing else changes. The ``field`` is the admission template's field name (the one
the web form and the Excel import use, see clinical/xlsx_template.py); ``d_<name>`` is an
optional model input and is used only when the active model asks for it.

ASSUMPTION: four admission facts have no LOINC or SNOMED CT code we could cite with confidence
(previous cesareans, number of fetuses, fetal presentation, onset of labour). They use the
project's own CodeSystem, ROBBY_SYSTEM, as placeholders until the hospital's FHIR profile names
the real codes; change the line, keep the reader."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

LOINC = "http://loinc.org"
SNOMED = "http://snomed.info/sct"
UCUM = "http://unitsofmeasure.org"
ROBBY_SYSTEM = "https://icyeza.github.io/robby/fhir/CodeSystem/admission"
DATA_ABSENT_SYSTEM = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
EXTENSION_BASE = "https://icyeza.github.io/robby/fhir/StructureDefinition/"


class Reader(StrEnum):
    """How an Observation's value becomes the field's text (see adapter.READERS)."""

    COUNT = "count"  # whole number: valueInteger, a unitless valueQuantity or valueString
    GESTATION = "gestation"  # valueQuantity in wk (or d), completed weeks
    BLOOD_PRESSURE = "blood_pressure"  # panel with systolic and diastolic components
    GLUCOSE = "glucose"  # valueQuantity in mmol/L, or mg/dL converted
    PROTEINURIA = "proteinuria"  # dipstick result as a coded or string value
    FETUSES = "fetuses"  # a count (2 or more is "Two or more") or the option's code
    CHOICE = "choice"  # valueCodeableConcept or valueString naming one of the app's options


@dataclass(frozen=True)
class ObservationMap:
    field: str
    reader: Reader
    what: str  # how the value is named in messages


R = Reader

# (code system, code) -> where the Observation's value goes. One line per code.
OBSERVATIONS: dict[tuple[str, str], ObservationMap] = {
    (LOINC, "11977-6"): ObservationMap("parity", R.COUNT, "parity"),
    (LOINC, "11996-6"): ObservationMap("d_gravidity", R.COUNT, "gravidity"),
    (LOINC, "49051-6"): ObservationMap("gestational_age_weeks", R.GESTATION, "gestational age"),
    (LOINC, "85354-9"): ObservationMap("bp", R.BLOOD_PRESSURE, "blood pressure"),
    (LOINC, "15074-8"): ObservationMap("glucose", R.GLUCOSE, "glucose"),
    (LOINC, "20454-5"): ObservationMap("proteinuria", R.PROTEINURIA, "urine protein"),
    # Placeholders (see the module docstring).
    (ROBBY_SYSTEM, "previous-cesareans"): ObservationMap(
        "previous_cs_count", R.COUNT, "previous cesareans"
    ),
    (ROBBY_SYSTEM, "fetuses"): ObservationMap("plurality", R.FETUSES, "number of fetuses"),
    (ROBBY_SYSTEM, "fetal-presentation"): ObservationMap(
        "fetal_presentation", R.CHOICE, "fetal presentation"
    ),
    (ROBBY_SYSTEM, "labour-onset"): ObservationMap("onset_of_labour", R.CHOICE, "onset of labour"),
}

# Blood pressure panel components (LOINC).
SYSTOLIC = (LOINC, "8480-6")
DIASTOLIC = (LOINC, "8462-4")

# (code system, code) -> optional model input recorded "Yes" when the Condition is present.
CONDITIONS: dict[tuple[str, str], str] = {
    (SNOMED, "398254007"): "preeclampsia_recorded",  # Pre-eclampsia
    (SNOMED, "11687002"): "gdm_recorded",  # Gestational diabetes mellitus
    (SNOMED, "86406008"): "hiv_positive",  # Human immunodeficiency virus infection
}

# Screening fields whose absence needs a reason, and the reason key used by the form.
SCREENING: dict[str, str] = {"bp": "bp", "proteinuria": "protein", "glucose": "glucose"}

# dataAbsentReason code -> the app's omission reason. A code not listed becomes "other", unless
# dataAbsentReason.text names one of the app's reasons (e.g. "equipment_unavailable").
ABSENT_REASONS: dict[str, str] = {"asked-declined": "declined"}

# Units: UCUM code (or the unit text) -> factor to the app's unit.
GLUCOSE_UNITS: dict[str, float] = {"mmol/l": 1.0, "mg/dl": 1 / 18.016}
GESTATION_UNITS: dict[str, float] = {"wk": 1.0, "weeks": 1.0, "d": 1 / 7, "days": 1 / 7}
BP_UNITS = frozenset({"mm[hg]", "mmhg"})

# Lab requests (robby.lab) as ServiceRequest.code and Observation.code: lab_test value ->
# (system, code, display). ASSUMPTION: no LOINC code for "HIV rapid test" could be cited with
# confidence (several exist, by method and specimen), so it is a placeholder in the project's
# lab-test CodeSystem until the hospital's lab profile names the code it uses.
LAB_SYSTEM = "https://icyeza.github.io/robby/fhir/CodeSystem/lab-test"
LAB_TESTS: dict[str, tuple[str, str, str]] = {
    "urine_protein": (LOINC, "20454-5", "Protein [Presence] in Urine by Test strip"),
    "blood_glucose": (LOINC, "15074-8", "Glucose [Moles/volume] in Blood"),
    "hiv_rapid": (LAB_SYSTEM, "hiv-rapid-test", "HIV rapid test (placeholder code)"),
}
# Observation statuses that are a result to record (preliminary, registered ... are not).
RESULT_STATUSES = frozenset({"final", "amended", "corrected"})

# Statuses that mean the resource does not count.
IGNORED_STATUS = frozenset({"entered-in-error", "cancelled"})
# Condition verification statuses: refuted records "No"; entered-in-error is ignored.
REFUTED = "refuted"
