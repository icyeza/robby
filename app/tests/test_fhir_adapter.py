"""The FHIR adapter (FR8): Bundle -> template rows -> the importer's checks, with every problem
named by its FHIR element. No database."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from robby.clinical.form import DetailSpec
from robby.fhir import adapter, intake, mapping

ZONE = ZoneInfo("Africa/Kigali")
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=ZONE)
START = "2026-10-18T09:30:00+02:00"
LOINC, ROBBY, SNOMED = mapping.LOINC, mapping.ROBBY_SYSTEM, mapping.SNOMED
EXAMPLES = Path(__file__).resolve().parents[1] / "scripts" / "fhir_examples"
HIV = DetailSpec("hiv_positive", "HIV positive", "category", "", None, None, ("No", "Yes"))
GRAVIDITY = DetailSpec("gravidity", "Gravidity", "integer", "", 1, 30, None)
PRESENTATION, ONSET = "fetal_presentation", "onset_of_labour"

Json = dict[str, Any]


# ---------------------------------------------------------------- building Bundles


def patient(pid: str = "p1", number: str = "FX-1", born: str = "1996-03-14") -> Json:
    return {"resourceType": "Patient", "id": pid,
            "identifier": [{"system": "urn:h:pn", "value": number}],
            "name": [{"family": "Uwimana", "given": ["Aline"]}], "birthDate": born}  # fmt: skip


def encounter(eid: str = "e1", pid: str = "p1", start: str | None = START) -> Json:
    enc: Json = {"resourceType": "Encounter", "id": eid, "status": "in-progress",
                 "subject": {"reference": f"Patient/{pid}"}}  # fmt: skip
    if start is not None:
        enc["period"] = {"start": start}
    return enc


def obs(oid: str, code: str, system: str = LOINC, enc: str | None = "e1", **value: Any) -> Json:
    o: Json = {"resourceType": "Observation", "id": oid, "status": "final",
               "code": {"coding": [{"system": system, "code": code}]}, **value}  # fmt: skip
    if enc is not None:
        o["encounter"] = {"reference": f"Encounter/{enc}"}
    return o


def qty(value: float, code: str) -> Json:
    return {"valueQuantity": {"value": value, "system": mapping.UCUM, "code": code}}


def bp(oid: str = "bp", sys_: float = 120, dia: float = 80, enc: str = "e1") -> Json:
    def part(code: str, v: float) -> Json:
        return {"code": {"coding": [{"system": LOINC, "code": code}]}, **qty(v, "mm[Hg]")}

    return obs(oid, "85354-9", enc=enc, component=[part("8480-6", sys_), part("8462-4", dia)])


def concept(code: str) -> Json:
    return {"valueCodeableConcept": {"coding": [{"system": ROBBY, "code": code}]}}


def screening(enc: str = "e1") -> list[Json]:
    return [
        bp(f"{enc}-bp", enc=enc),
        obs(f"{enc}-pro", "20454-5", enc=enc, valueString="negative"),
        obs(f"{enc}-glu", "15074-8", enc=enc, **qty(5.0, "mmol/L")),
    ]


def admission(
    pid: str = "p1", eid: str = "e1", number: str = "FX-1", start: str = START
) -> list[Json]:
    """A complete admission: Patient, Encounter, the Robson facts and the screening."""
    return [
        patient(pid, number),
        encounter(eid, pid, start),
        obs(f"{eid}-par", "11977-6", enc=eid, valueInteger=0),
        obs(f"{eid}-ga", "49051-6", enc=eid, **qty(39, "wk")),
        obs(f"{eid}-cs", "previous-cesareans", ROBBY, enc=eid, valueInteger=0),
        obs(f"{eid}-fet", "fetuses", ROBBY, enc=eid, valueInteger=1),
        obs(f"{eid}-pres", "fetal-presentation", ROBBY, enc=eid, **concept("cephalic")),
        obs(f"{eid}-on", "labour-onset", ROBBY, enc=eid, **concept("spontaneous")),
        *screening(eid),
    ]


def bundle(*resources: Json, kind: str = "collection") -> Json:
    return {"resourceType": "Bundle", "type": kind, "entry": [{"resource": r} for r in resources]}


def read(*resources: Json, specs: list[DetailSpec] | None = None, **kw: Any) -> intake.Checked:
    return intake.check(adapter.read_bundle(bundle(*resources), ZONE, **kw), specs or [], ZONE, NOW)


def values(*resources: Json, **kw: Any) -> dict[str, str]:
    found = adapter.read_bundle(bundle(*resources), ZONE, require_screening=False, **kw)
    assert len(found.admissions) == 1, found.issues
    return found.admissions[0].values


def issues(*resources: Json, **kw: Any) -> list[tuple[str, str]]:
    return [(i.expression, i.message) for i in read(*resources, **kw).issues]


def where(*resources: Json, **kw: Any) -> list[str]:
    return [e for e, _ in issues(*resources, **kw)]


# ---------------------------------------------------------------- mapping


def test_a_complete_admission_maps_onto_the_template_and_parses() -> None:
    checked = read(*admission())
    assert checked.issues == []
    [(row, parsed)] = checked.rows
    assert row == intake.FIRST_ROW
    assert (parsed.patient_number, parsed.patient_name) == ("FX-1", "Aline Uwimana")
    assert parsed.admitted_at == datetime(2026, 10, 18, 9, 30, tzinfo=ZONE)
    assert parsed.maternal_age == 30
    assert (parsed.parity, parsed.previous_cs_count, parsed.plurality) == (0, 0, 1)
    assert (parsed.ga_weeks, parsed.presentation, parsed.onset) == (39, "cephalic", "spontaneous")
    assert (parsed.systolic, parsed.diastolic, parsed.proteinuria) == (120, 80, "negative")
    assert parsed.glucose == Decimal("5.0")


@pytest.mark.parametrize(
    ("resource", "field", "text"),
    [
        (obs("o", "11977-6", valueInteger=3), "parity", "3"),
        (obs("o", "11996-6", valueInteger=4), "d_gravidity", "4"),
        (obs("o", "49051-6", **qty(38.6, "wk")), "gestational_age_weeks", "38"),
        (obs("o", "49051-6", **qty(275, "d")), "gestational_age_weeks", "39"),
        (obs("o", "15074-8", **qty(6.24, "mmol/L")), "glucose", "6.2"),
        (obs("o", "15074-8", **qty(90, "mg/dL")), "glucose", "5.0"),
        (obs("o", "20454-5", valueString="2+"), "proteinuria", "++"),
        (obs("o", "20454-5", valueCodeableConcept={"text": "Trace"}), "proteinuria", "trace"),
        (obs("o", "previous-cesareans", ROBBY, valueInteger=2), "previous_cs_count", "2"),
        (obs("o", "fetuses", ROBBY, valueInteger=3), "plurality", "2"),
        (obs("o", "fetuses", ROBBY, **concept("1")), "plurality", "1"),
        (obs("o", "fetal-presentation", ROBBY, **concept("breech")), PRESENTATION, "breech"),
        (obs("o", "fetal-presentation", ROBBY, valueString="Oblique"), PRESENTATION, "oblique"),
        (obs("o", "labour-onset", ROBBY, **concept("prelabour_cs")), ONSET, "prelabour_cs"),
    ],
)
def test_each_code_fills_its_field(resource: Json, field: str, text: str) -> None:
    assert values(patient(), encounter(), resource)[field] == text


def test_the_blood_pressure_panel_fills_both_pressures() -> None:
    found = values(patient(), encounter(), bp(sys_=131, dia=87))
    assert (found["bp_sys"], found["bp_dia"]) == ("131", "87")


def test_every_mapped_code_is_one_line_of_the_table() -> None:
    fields = {rule.field: system for (system, _), rule in mapping.OBSERVATIONS.items()}
    placeholders = {"previous_cs_count", "plurality", PRESENTATION, ONSET}
    robson = {"parity", "gestational_age_weeks", "bp", "proteinuria", "glucose", *placeholders}
    assert robson <= set(fields)
    assert {fields[f] for f in placeholders} == {ROBBY}


def test_birth_date_gives_the_age_at_admission() -> None:
    assert values(patient(born="1996-10-19"), encounter())["maternal_age"] == "29"
    assert values(patient(born="1996-10-18"), encounter())["maternal_age"] == "30"
    assert "maternal_age" not in values(patient(born="1996"), encounter())


def test_the_admission_time_is_shown_in_the_apps_zone() -> None:
    assert values(patient(), encounter(start="2026-10-18T07:30:00Z"))["admitted_at"] == (
        "2026-10-18 09:30"
    )


def test_a_configured_identifier_system_picks_the_patient_number() -> None:
    p = patient()
    p["identifier"] = [{"system": "urn:national-id", "value": "1199"}, *p["identifier"]]
    assert values(p, encounter())["patient_number"] == "1199"
    assert values(p, encounter(), patient_id_system="urn:h:pn")["patient_number"] == "FX-1"
    p["identifier"] = p["identifier"][:1]
    assert ("Patient/p1.identifier", "The Patient has no identifier with system urn:h:pn.") in (
        issues(p, *admission()[1:], patient_id_system="urn:h:pn")
    )


def test_the_latest_observation_for_a_field_wins() -> None:
    early = obs("a", "11977-6", valueInteger=1, effectiveDateTime="2026-10-18T08:00:00+02:00")
    late = obs("b", "11977-6", valueInteger=2, effectiveDateTime="2026-10-18T09:00:00+02:00")
    assert values(patient(), encounter(), late, early)["parity"] == "2"


def test_observations_without_an_encounter_belong_to_the_patients_only_encounter() -> None:
    o = obs("o", "11977-6", enc=None, valueInteger=2, subject={"reference": "Patient/p1"})
    assert values(patient(), encounter(), o)["parity"] == "2"


def test_unknown_codes_and_entered_in_error_resources_are_ignored() -> None:
    other = obs("x", "8302-2", valueQuantity={"value": 160, "code": "cm"})
    wrong = obs("w", "11977-6", valueInteger=9, status="entered-in-error")
    assert read(*admission(), other, wrong).issues == []


# ---------------------------------------------------------------- conditions


def condition(code: str, cid: str = "c1", **extra: Any) -> Json:
    return {"resourceType": "Condition", "id": cid, "subject": {"reference": "Patient/p1"},
            "code": {"coding": [{"system": SNOMED, "code": code}]}, **extra}  # fmt: skip


def test_conditions_fill_the_models_optional_inputs() -> None:
    checked = read(*admission(), condition("86406008"), specs=[HIV])
    assert checked.issues == []
    assert checked.rows[0][1].details == {"hiv_positive": "Yes"}
    found = values(patient(), encounter(), condition("398254007"), condition("11687002", "c2"))
    assert (found["d_preeclampsia_recorded"], found["d_gdm_recorded"]) == ("Yes", "Yes")


def test_a_refuted_condition_records_no() -> None:
    refuted = {"verificationStatus": {"coding": [{"code": "refuted"}]}}
    checked = read(*admission(), condition("86406008", **refuted), specs=[HIV])
    assert checked.rows[0][1].details == {"hiv_positive": "No"}


def test_inputs_the_active_model_does_not_use_are_left_out() -> None:
    checked = read(*admission(), condition("86406008"), obs("g", "11996-6", valueInteger=2))
    assert checked.issues == [] and checked.rows[0][1].details == {}
    with_model = read(*admission(), obs("g", "11996-6", valueInteger=2), specs=[GRAVIDITY])
    assert with_model.rows[0][1].details == {"gravidity": 2}


# ---------------------------------------------------------------- not measured


def absent(code: str | None, text: str | None = None) -> Json:
    reason: Json = {}
    if code:
        reason["coding"] = [{"system": mapping.DATA_ABSENT_SYSTEM, "code": code}]
    if text:
        reason["text"] = text
    return {"dataAbsentReason": reason}


@pytest.mark.parametrize(
    ("loinc", "reason", "expected"),
    [
        ("15074-8", absent("asked-declined"), "declined"),
        ("15074-8", absent("not-performed"), "other"),
        ("15074-8", absent("not-performed", "equipment_unavailable"), "equipment_unavailable"),
        ("20454-5", absent("unknown", "Reagent strips out of stock"), "supplies_out_of_stock"),
        ("85354-9", absent(None, "no_time_before_delivery"), "no_time_before_delivery"),
    ],
)
def test_data_absent_reason_becomes_the_not_measured_reason(
    loinc: str, reason: Json, expected: str
) -> None:
    key = {"15074-8": "glucose", "20454-5": "protein", "85354-9": "bp"}[loinc]
    rest = [r for r in admission() if not r["id"].endswith(("-bp", "-pro", "-glu"))]
    kept = [r for r in screening() if r["code"]["coding"][0]["code"] != loinc]
    checked = read(*rest, *kept, obs("nm", loinc, **reason))
    assert checked.issues == []
    assert getattr(checked.rows[0][1], f"{key}_reason") == expected


def test_a_reason_text_the_field_does_not_allow_falls_back_to_the_code() -> None:
    rest = [r for r in admission() if not r["id"].endswith("-pro")]
    nm = obs("nm", "20454-5", **absent("not-performed", "equipment_unavailable"))
    checked = read(*rest, nm)
    assert checked.issues == [] and checked.rows[0][1].protein_reason == "other"
    found = adapter.read_bundle(bundle(*rest, nm), ZONE).admissions[0]
    assert found.sources["protein_reason"] == "Observation/nm.dataAbsentReason"


def test_missing_screening_is_one_issue_on_the_encounter() -> None:
    rest = [r for r in admission() if not r["id"].endswith("-glu")]
    [(expression, message)] = issues(*rest)
    assert expression == "Encounter/e1" and "glucose (LOINC 15074-8)" in message


# ---------------------------------------------------------------- problems by element


def test_invalid_values_are_reported_on_their_element() -> None:
    bad = [
        obs("e1-par", "11977-6", valueInteger=45),
        obs("e1-glu", "15074-8", **qty(5.1, "g/L")),
        obs("e1-pro", "20454-5", valueString="a lot"),
        obs("e1-ga", "49051-6", **qty(39, "mo")),
        obs("e1-pres", "fetal-presentation", ROBBY, valueString="footling"),
    ]
    ids = {r["id"] for r in bad}
    found = dict(issues(*[r for r in admission() if r["id"] not in ids], *bad))
    assert found["Observation/e1-par.valueInteger"] == "Parity must be a whole number from 0 to 30."
    assert "“g/L” is not understood" in found["Observation/e1-glu.valueQuantity.code"]
    assert "not a dipstick result" in found["Observation/e1-pro.valueString"]
    assert "“mo” is not understood" in found["Observation/e1-ga.valueQuantity.code"]
    assert "“footling” is not a fetal presentation" in found["Observation/e1-pres.valueString"]
    assert len(found) == 5


def test_blood_pressure_problems_name_the_component() -> None:
    rest = [r for r in admission() if not r["id"].endswith("-bp")]
    assert where(*rest, bp(sys_=70, dia=90)) == ["Observation/bp.component[0].valueQuantity"]
    half = bp()
    half["component"] = half["component"][:1]
    assert where(*rest, half) == ["Observation/bp.component"]


def swap(resources: list[Json], replacement: Json) -> list[Json]:
    """The resources with those of the replacement's type replaced by it."""
    kind = replacement["resourceType"]
    return [replacement if r["resourceType"] == kind else r for r in resources]


def test_missing_and_broken_elements() -> None:
    assert where(*swap(admission(), encounter(start=None))) == ["Encounter/e1.period.start"]
    date_only = swap(admission(), encounter(start="2026-10-18"))
    assert where(*date_only) == ["Encounter/e1.period.start"]
    nameless = swap(admission(), {**patient(), "name": []})
    assert issues(*nameless) == [("Patient/p1.name", "Enter the patient name.")]
    orphan = obs("o", "11977-6", enc="e9", valueInteger=1)
    assert where(*admission(), orphan) == ["Observation/o.encounter"]
    assert where(*swap(admission(), encounter(pid="p9"))) == ["Encounter/e1.subject"]


def test_an_observation_of_a_patient_with_two_encounters_must_name_one() -> None:
    o = obs("o", "11977-6", enc=None, valueInteger=2, subject={"reference": "Patient/p1"})
    second = [r for r in admission(eid="e2") if r["resourceType"] != "Patient"]
    later = swap(second, encounter("e2", start="2026-10-18T10:30:00+02:00"))
    assert "Observation/o.encounter" in where(*admission(), *later, o)


def test_several_encounters_are_several_admissions_and_problems_name_the_right_one() -> None:
    second = admission("p2", "e2", "FX-2", "2026-10-18T10:00:00+02:00")
    second[2] = obs("e2-par", "11977-6", enc="e2", valueInteger=-1)
    checked = read(*admission(), *second)
    assert checked.rows == []  # all or nothing
    assert [i.expression for i in checked.issues] == ["Observation/e2-par.valueInteger"]
    fine = read(*admission(), *admission("p2", "e2", "FX-2", "2026-10-18T10:00:00+02:00"))
    assert [p.patient_number for _, p in fine.rows] == ["FX-1", "FX-2"]
    assert [a.encounter for a in fine.admissions] == ["Encounter/e1", "Encounter/e2"]


def test_references_by_full_url() -> None:
    b = bundle(*admission())
    b["entry"][0]["fullUrl"] = "urn:uuid:aaa"
    b["entry"][1]["resource"]["subject"] = {"reference": "urn:uuid:aaa"}
    assert adapter.read_bundle(b, ZONE).issues == []


def test_bundle_level_problems() -> None:
    assert adapter.not_a_bundle({"resourceType": "Patient"}) is not None
    assert adapter.not_a_bundle(bundle()) is None
    wrong = adapter.read_bundle(bundle(*admission(), kind="document"), ZONE)
    assert [i.expression for i in wrong.issues] == ["Bundle.type"]
    empty = adapter.read_bundle(bundle(patient()), ZONE)
    assert [(i.expression, i.code) for i in empty.issues] == [("Bundle.entry", "required")]


# ---------------------------------------------------------------- the example files


def recent(b: Json) -> Json:
    """The example's Encounters moved to just before NOW."""
    hours = iter(range(9, 12))
    for entry in b["entry"]:
        if entry["resource"]["resourceType"] == "Encounter":
            entry["resource"]["period"]["start"] = f"2026-10-18T{next(hours):02d}:00:00+02:00"
    return b


def test_the_valid_example_gives_two_admissions_one_with_glucose_not_measured() -> None:
    b = recent(json.loads((EXAMPLES / "two_admissions.json").read_text(encoding="utf-8")))
    checked = intake.check(adapter.read_bundle(b, ZONE), [], ZONE, NOW)
    assert checked.issues == []
    first, second = (p for _, p in checked.rows)
    assert (first.patient_number, first.glucose) == ("FX-1001", Decimal("4.8"))
    assert (second.glucose, second.glucose_reason) == (None, "equipment_unavailable")


def test_the_invalid_example_names_every_element() -> None:
    b = json.loads((EXAMPLES / "invalid.json").read_text(encoding="utf-8"))
    checked = intake.check(adapter.read_bundle(b, ZONE), [], ZONE, NOW)
    assert checked.rows == []
    assert sorted(i.expression for i in checked.issues) == [
        "Encounter/enc-9",
        "Encounter/enc-9.period.start",
        "Observation/obs-9-ga.encounter",
        "Observation/obs-9-glucose.valueQuantity.code",
        "Observation/obs-9-parity.valueInteger",
        "Observation/obs-9-protein.valueString",
        "Patient/pat-9.identifier[0].value",
    ]
