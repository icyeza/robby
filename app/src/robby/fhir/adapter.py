"""Reading a FHIR R4 Bundle into admissions (FR8): one admission per Encounter.

The adapter only translates. Each Encounter becomes the field values of one row of the app's
admission template (the same field names the web form posts), together with the FHIR element
each value came from; intake.py then sends those rows through the importer's checks and
form.parse, exactly as an Excel row. What the adapter itself cannot read (a missing reference, an
unknown unit, a dipstick result it does not recognise) is an Issue naming the FHIR element, such
as ``Observation/obs-3.valueQuantity``.

Linking: an Observation or Condition belongs to the Encounter its ``encounter`` names; without
one, to its subject Patient's only Encounter in the Bundle. When several Observations fill the
same field, the latest (effectiveDateTime, else issued, else Bundle order) wins, so a pulled
record with repeated readings gives the last one."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from robby.clinical import labels
from robby.clinical.openclinic import PROTEIN_VALUES
from robby.fhir import mapping as m

BUNDLE_TYPES = frozenset({"transaction", "batch", "collection"})
TIME_TEXT = "%Y-%m-%d %H:%M"  # what the template's "Admitted at" column accepts
CHOICES: dict[str, tuple[tuple[str, str], ...]] = {
    "fetal_presentation": labels.PRESENTATION_OPTIONS,
    "onset_of_labour": labels.ONSET_OPTIONS,
    "plurality": labels.PLURALITY_OPTIONS,
}
# The template fields a screening Observation fills, by mapping field.
SCREENING_FIELDS = {
    "bp": ("bp_sys", "bp_dia", "bp_reason"),
    "proteinuria": ("proteinuria", "protein_reason"),
    "glucose": ("glucose", "glucose_reason"),
}
SCREENING_CODES = {
    "bp": "blood pressure (LOINC 85354-9)",
    "proteinuria": "urine protein (LOINC 20454-5)",
    "glucose": "glucose (LOINC 15074-8)",
}

Json = dict[str, Any]


@dataclass(frozen=True)
class Issue:
    """One problem, as an OperationOutcome.issue: ``expression`` is the FHIR element."""

    expression: str
    message: str
    code: str = "invalid"  # IssueType: invalid | required | not-found | not-supported ...
    severity: str = "error"


@dataclass(frozen=True)
class Mapped:
    """One Encounter as template field values, and where each came from."""

    encounter: str  # "Encounter/enc-1"
    patient: str  # "Patient/pat-1"
    values: dict[str, str]
    sources: dict[str, str]  # field -> FHIR element
    flagged: frozenset[str] = frozenset()  # fields the adapter already reported a problem on

    def source_of(self, field_name: str) -> str:
        defaults = {
            "patient_number": f"{self.patient}.identifier",
            "patient_name": f"{self.patient}.name",
            "admitted_at": f"{self.encounter}.period.start",
        }
        return self.sources.get(field_name) or defaults.get(field_name, self.encounter)


@dataclass(frozen=True)
class BundleRead:
    admissions: list[Mapped]
    issues: list[Issue]


def not_a_bundle(payload: object) -> str | None:
    """Why the body is not a FHIR Bundle at all (an HTTP 400), or None."""
    if not isinstance(payload, dict) or payload.get("resourceType") != "Bundle":
        return 'The body must be a FHIR Bundle ({"resourceType": "Bundle", ...}).'
    return None


# ---------------------------------------------------------------- the Bundle's resources


@dataclass
class _Entry:
    resource: Json
    label: str  # "Type/id", or the entry's position when the resource has no id
    position: int


class _Index:
    """The Bundle's resources, found by "Type/id", by fullUrl, or by an absolute URL's tail."""

    def __init__(self, entries: list[_Entry], full_urls: dict[str, _Entry]) -> None:
        self.entries = entries
        self._by_label = {e.label: e for e in entries}
        self._by_url = full_urls

    def of_type(self, kind: str) -> list[_Entry]:
        return [e for e in self.entries if e.resource.get("resourceType") == kind]

    def resolve(self, reference: object, kind: str) -> _Entry | None:
        ref = reference.get("reference") if isinstance(reference, dict) else None
        if not isinstance(ref, str):
            return None
        found = self._by_url.get(ref) or self._by_label.get(ref)
        if found is None and "/" in ref:  # https://host/fhir/Patient/p1 or Patient/p1/_history/2
            parts = ref.split("/_history/")[0].rstrip("/").split("/")
            found = self._by_label.get("/".join(parts[-2:]))
        if found is None or found.resource.get("resourceType") != kind:
            return None
        return found


def _index(bundle: Json, issues: list[Issue]) -> _Index:
    entries: list[_Entry] = []
    urls: dict[str, _Entry] = {}
    raw = bundle.get("entry")
    if not isinstance(raw, list):
        issues.append(Issue("Bundle.entry", "The Bundle has no entries.", "required"))
        return _Index([], {})
    for i, item in enumerate(raw):
        resource = item.get("resource") if isinstance(item, dict) else None
        if not isinstance(resource, dict) or not isinstance(resource.get("resourceType"), str):
            issues.append(Issue(f"Bundle.entry[{i}]", "The entry has no resource.", "required"))
            continue
        rid = resource.get("id")
        label = (
            f"{resource['resourceType']}/{rid}"
            if isinstance(rid, str) and rid
            else f"Bundle.entry[{i}].resource"
        )
        entry = _Entry(resource, label, i)
        entries.append(entry)
        full_url = item.get("fullUrl")
        if isinstance(full_url, str) and full_url:
            urls[full_url] = entry
    return _Index(entries, urls)


def _counts(resource: Json) -> bool:
    return resource.get("status") not in m.IGNORED_STATUS


# ---------------------------------------------------------------- small readers


def _instant(text: object, zone: ZoneInfo) -> datetime | None:
    """A FHIR dateTime with a time, in the app's zone (no offset: already local time)."""
    if not isinstance(text, str) or "T" not in text:
        return None
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    return when.replace(tzinfo=zone) if when.tzinfo is None else when.astimezone(zone)


def _number_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _codings(concept: object) -> list[Json]:
    found = concept.get("coding") if isinstance(concept, dict) else None
    return [c for c in found if isinstance(c, dict)] if isinstance(found, list) else []


def _key_of(concept: object, table: Mapping[tuple[str, str], Any]) -> tuple[str, str] | None:
    for coding in _codings(concept):
        key = (coding.get("system"), coding.get("code"))
        if key in table:
            return key  # type: ignore[return-value]
    return None


def _words(concept: object) -> list[str]:
    """Every code, display and text of a CodeableConcept, to match against the app's options."""
    words = [str(c.get(k)) for c in _codings(concept) for k in ("code", "display") if c.get(k)]
    if isinstance(concept, dict) and isinstance(concept.get("text"), str):
        words.append(concept["text"])
    return words


def _option(words: list[str], options: tuple[tuple[str, str], ...]) -> str | None:
    for word in words:
        wanted = " ".join(word.split()).lower()
        for value, shown in options:
            if wanted in (value.lower(), shown.lower()):
                return value
    return None


# ---------------------------------------------------------------- Observation values

Filled = dict[str, tuple[str, str]]  # template field -> (text, FHIR element)
ObsReader = Callable[[Json, str, m.ObservationMap], "Filled | Issue"]


def _count(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    if isinstance(obs.get("valueInteger"), int) and not isinstance(obs["valueInteger"], bool):
        return {rule.field: (str(obs["valueInteger"]), f"{at}.valueInteger")}
    quantity = obs.get("valueQuantity")
    if isinstance(quantity, dict) and _is_number(quantity.get("value")):
        return {rule.field: (_number_text(quantity["value"]), f"{at}.valueQuantity")}
    if isinstance(obs.get("valueString"), str):
        return {rule.field: (obs["valueString"].strip(), f"{at}.valueString")}
    return Issue(f"{at}.value[x]", f"Give the {rule.what} as valueInteger.", "required")


def _unit(quantity: Json) -> str:
    return _shown_unit(quantity).lower()


def _shown_unit(quantity: Json) -> str:
    return str(quantity.get("code") or quantity.get("unit") or "").strip()


def _gestation(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    if isinstance(obs.get("valueInteger"), int) and not isinstance(obs["valueInteger"], bool):
        return {rule.field: (str(obs["valueInteger"]), f"{at}.valueInteger")}
    quantity = obs.get("valueQuantity")
    if not isinstance(quantity, dict) or not _is_number(quantity.get("value")):
        return Issue(f"{at}.valueQuantity", "Give the gestational age as a valueQuantity in wk.")
    unit = _unit(quantity) or "wk"
    if unit not in m.GESTATION_UNITS:
        shown = _shown_unit(quantity)
        message = f"Gestational age in “{shown}” is not understood; use wk or d (UCUM)."
        return Issue(f"{at}.valueQuantity.code", message, "not-supported")
    weeks = math.floor(float(quantity["value"]) * m.GESTATION_UNITS[unit] + 1e-9)
    return {rule.field: (str(weeks), f"{at}.valueQuantity")}


def _glucose(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    quantity = obs.get("valueQuantity")
    if not isinstance(quantity, dict) or not _is_number(quantity.get("value")):
        return Issue(f"{at}.valueQuantity", "Give the glucose as a valueQuantity in mmol/L.")
    unit = _unit(quantity)
    if unit not in m.GLUCOSE_UNITS:
        shown = f"“{_shown_unit(quantity)}”" if unit else "no unit"
        message = f"Glucose in {shown} is not understood; use mmol/L or mg/dL (UCUM)."
        return Issue(f"{at}.valueQuantity.code", message, "not-supported")
    mmol = round(float(quantity["value"]) * m.GLUCOSE_UNITS[unit], 1)
    return {rule.field: (f"{mmol:.1f}", f"{at}.valueQuantity")}


def _blood_pressure(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    found: Filled = {}
    components = obs.get("component")
    for i, comp in enumerate(components if isinstance(components, list) else []):
        if not isinstance(comp, dict):
            continue
        codes = {(c.get("system"), c.get("code")) for c in _codings(comp.get("code"))}
        target = "bp_sys" if m.SYSTOLIC in codes else "bp_dia" if m.DIASTOLIC in codes else None
        if target is None:
            continue
        where = f"{at}.component[{i}].valueQuantity"
        quantity = comp.get("valueQuantity")
        if not isinstance(quantity, dict) or not _is_number(quantity.get("value")):
            return Issue(where, "Give the pressure as a valueQuantity in mm[Hg].", "required")
        if _unit(quantity) and _unit(quantity) not in m.BP_UNITS:
            return Issue(f"{where}.code", "Blood pressure must be in mm[Hg].", "not-supported")
        found[target] = (_number_text(quantity["value"]), where)
    if set(found) != {"bp_sys", "bp_dia"}:
        return Issue(
            f"{at}.component",
            "The blood pressure panel needs a systolic (8480-6) and a diastolic (8462-4) "
            "component.",
            "required",
        )
    return found


def _proteinuria(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    if "valueCodeableConcept" in obs:
        words, where = _words(obs["valueCodeableConcept"]), f"{at}.valueCodeableConcept"
    elif isinstance(obs.get("valueString"), str):
        words, where = [obs["valueString"]], f"{at}.valueString"
    else:
        return Issue(f"{at}.value[x]", "Give the dipstick result as a code or text.", "required")
    for word in words:
        stored = PROTEIN_VALUES.get(" ".join(word.split()).lower())
        if stored is not None:
            return {rule.field: (stored, where)}
    shown = ", ".join(v for _, v in labels.PROTEINURIA_OPTIONS)
    return Issue(where, f"“{', '.join(words)}” is not a dipstick result; use one of {shown}.")


def _choice(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    options = CHOICES[rule.field]
    if "valueCodeableConcept" in obs:
        words, where = _words(obs["valueCodeableConcept"]), f"{at}.valueCodeableConcept"
    elif isinstance(obs.get("valueString"), str):
        words, where = [obs["valueString"]], f"{at}.valueString"
    else:
        return Issue(f"{at}.value[x]", f"Give the {rule.what} as a code.", "required")
    value = _option(words, options)
    if value is None:
        allowed = ", ".join(v for v, _ in options)
        return Issue(where, f"“{', '.join(words)}” is not a {rule.what}; use one of {allowed}.")
    return {rule.field: (value, where)}


def _fetuses(obs: Json, at: str, rule: m.ObservationMap) -> Filled | Issue:
    if "valueCodeableConcept" in obs or isinstance(obs.get("valueString"), str):
        return _choice(obs, at, rule)
    counted = _count(obs, at, rule)
    if isinstance(counted, Issue):
        return counted
    text, where = counted[rule.field]
    if not text.isdigit() or int(text) < 1:
        return Issue(where, "The number of fetuses must be 1 or more.")
    return {rule.field: ("1" if int(text) == 1 else "2", where)}


READERS: dict[m.Reader, ObsReader] = {
    m.Reader.COUNT: _count,
    m.Reader.GESTATION: _gestation,
    m.Reader.BLOOD_PRESSURE: _blood_pressure,
    m.Reader.GLUCOSE: _glucose,
    m.Reader.PROTEINURIA: _proteinuria,
    m.Reader.FETUSES: _fetuses,
    m.Reader.CHOICE: _choice,
}


def absent_reason(obs: Json, key: str) -> str:
    """The app's omission reason for a screening Observation with a dataAbsentReason: the reason
    its text names (value or label, any case), else the code's (ABSENT_REASONS), else other."""
    concept = obs.get("dataAbsentReason")
    named = _option(
        [concept["text"]] if isinstance(concept, dict) and isinstance(concept.get("text"), str)
        else [],
        labels.REASONS[key],
    )  # fmt: skip
    if named is not None:
        return named
    for coding in _codings(concept):
        if coding.get("code") in m.ABSENT_REASONS:
            return m.ABSENT_REASONS[str(coding["code"])]
    return "other"


# ---------------------------------------------------------------- one Encounter


@dataclass
class _Draft:
    encounter: _Entry
    patient: _Entry | None
    values: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    flagged: set[str] = field(default_factory=set)
    # mapping field -> (sort key, Observation, its rule): the latest Observation for the field
    latest: dict[str, tuple[tuple[str, int], _Entry, m.ObservationMap]] = field(
        default_factory=dict
    )

    def put(self, name: str, text: str, where: str) -> None:
        self.values[name], self.sources[name] = text, where


def _patient_fields(draft: _Draft, system: str | None, issues: list[Issue], zone: ZoneInfo) -> None:
    patient = draft.patient
    if patient is None:
        return
    res, at = patient.resource, patient.label
    identifiers = [i for i in res.get("identifier") or [] if isinstance(i, dict)]
    chosen = [
        (k, i) for k, i in enumerate(identifiers) if system is None or i.get("system") == system
    ]
    if chosen and isinstance(chosen[0][1].get("value"), str):
        k, ident = chosen[0]
        draft.put("patient_number", ident["value"].strip(), f"{at}.identifier[{k}].value")
    else:
        wanted = f" with system {system}" if system else ""
        message = f"The Patient has no identifier{wanted}."
        issues.append(Issue(f"{at}.identifier", message, "required"))
        draft.flagged.add("patient_number")

    names = [n for n in res.get("name") or [] if isinstance(n, dict)]
    k = next((k for k, n in enumerate(names) if n.get("use") == "official"), 0)
    if names:
        name = names[k]
        given = [g for g in name.get("given") or [] if isinstance(g, str)]
        family = str(name["family"]) if isinstance(name.get("family"), str) else ""
        text = str(name["text"]) if isinstance(name.get("text"), str) else ""
        draft.put("patient_name", text or " ".join([*given, family]).strip(), f"{at}.name[{k}]")

    born = res.get("birthDate")
    admitted = _instant(draft.encounter.resource.get("period", {}).get("start"), zone)
    if isinstance(born, str) and len(born) == 10 and admitted is not None:
        try:
            birth = date.fromisoformat(born)
        except ValueError:
            issues.append(Issue(f"{at}.birthDate", "The birth date is not a date."))
            draft.flagged.add("maternal_age")
            return
        day = admitted.date()
        age = day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))
        draft.put("maternal_age", str(age), f"{at}.birthDate")
    # A partial birth date (year, or year and month) gives no age: the field stays blank.


def _admitted_at(draft: _Draft, issues: list[Issue], zone: ZoneInfo) -> None:
    at = draft.encounter.label
    period = draft.encounter.resource.get("period")
    start = period.get("start") if isinstance(period, dict) else None
    when = _instant(start, zone)
    if start is None:
        issues.append(Issue(f"{at}.period.start", "The Encounter has no start time.", "required"))
    elif when is None:
        issues.append(
            Issue(f"{at}.period.start", "The start must be a date and time, with its offset.")
        )
    else:
        draft.put("admitted_at", when.strftime(TIME_TEXT), f"{at}.period.start")
        return
    draft.flagged.add("admitted_at")


def _sort_key(entry: _Entry, zone: ZoneInfo) -> tuple[str, int]:
    res = entry.resource
    when = _instant(res.get("effectiveDateTime"), zone) or _instant(res.get("issued"), zone)
    return (when.isoformat() if when else "", entry.position)


def _observation(draft: _Draft, rule: m.ObservationMap, entry: _Entry, issues: list[Issue]) -> None:
    obs, at = entry.resource, entry.label
    screening = m.SCREENING.get(rule.field)
    has_value = any(k.startswith("value") for k in obs) or (
        rule.reader is m.Reader.BLOOD_PRESSURE
        and any(isinstance(c, dict) and "valueQuantity" in c for c in obs.get("component") or [])
    )
    if "dataAbsentReason" in obs and not has_value:
        if screening is not None:
            reason = absent_reason(obs, screening)
            draft.put(f"{screening}_reason", reason, f"{at}.dataAbsentReason")
        return  # a non-screening value that was not taken simply stays blank
    found = READERS[rule.reader](obs, at, rule)
    if isinstance(found, Issue):
        issues.append(found)
        draft.flagged.update(SCREENING_FIELDS.get(rule.field, (rule.field,)))
        return
    for name, (text, where) in found.items():
        draft.put(name, text, where)


def _condition(draft: _Draft, name: str, entry: _Entry) -> None:
    res = entry.resource
    verification = {c.get("code") for c in _codings(res.get("verificationStatus"))}
    if "entered-in-error" in verification:
        return
    yes = m.REFUTED not in verification
    draft.put(f"d_{name}", "Yes" if yes else "No", f"{entry.label}.code")


def _missing_screening(draft: _Draft, issues: list[Issue]) -> None:
    for key, fields in SCREENING_FIELDS.items():
        if any(draft.values.get(f) for f in fields) or draft.flagged.intersection(fields):
            continue
        issues.append(
            Issue(
                draft.encounter.label,
                f"No {SCREENING_CODES[key]} Observation for this Encounter. Send the result, or "
                "an Observation with a dataAbsentReason saying why it was not measured.",
                "required",
            )
        )
        draft.flagged.update(fields)


# ---------------------------------------------------------------- the whole Bundle


def _owner(
    entry: _Entry, index: _Index, drafts: dict[str, _Draft], issues: list[Issue]
) -> _Draft | None:
    """The Encounter draft an Observation or Condition belongs to."""
    res, at = entry.resource, entry.label
    if "encounter" in res:
        enc = index.resolve(res["encounter"], "Encounter")
        if enc is None or enc.label not in drafts:
            message = "The Encounter is not in the Bundle."
            issues.append(Issue(f"{at}.encounter", message, "not-found"))
            return None
        return drafts[enc.label]
    patient = index.resolve(res.get("subject"), "Patient")
    if patient is None:
        issues.append(
            Issue(f"{at}.subject", "Say which Encounter or Patient this belongs to.", "required")
        )
        return None
    theirs = [d for d in drafts.values() if d.patient is patient]
    if len(theirs) != 1:
        message = (
            "The Patient has no Encounter in the Bundle."
            if not theirs
            else "The Patient has several Encounters; say which one in encounter."
        )
        issues.append(Issue(f"{at}.encounter", message, "required"))
        return None
    return theirs[0]


def read_bundle(
    bundle: Json,
    zone: ZoneInfo,
    *,
    patient_id_system: str | None = None,
    require_screening: bool = True,
) -> BundleRead:
    """Every Encounter of the Bundle as template values, and every problem found. Resources of
    other types, and codes the mapping does not list, are ignored."""
    issues: list[Issue] = []
    if bundle.get("type") not in BUNDLE_TYPES:
        allowed = ", ".join(sorted(BUNDLE_TYPES))
        return BundleRead([], [Issue("Bundle.type", f"The Bundle type must be {allowed}.")])
    index = _index(bundle, issues)

    drafts: dict[str, _Draft] = {}
    for enc in index.of_type("Encounter"):
        if not _counts(enc.resource):
            continue
        patient = index.resolve(enc.resource.get("subject"), "Patient")
        draft = _Draft(enc, patient)
        if patient is None:
            message = "The Encounter's Patient is not in the Bundle."
            issues.append(Issue(f"{enc.label}.subject", message, "not-found"))
            draft.flagged.update({"patient_number", "patient_name"})
        _admitted_at(draft, issues, zone)
        _patient_fields(draft, patient_id_system, issues, zone)
        drafts[enc.label] = draft
    if not drafts and not issues:
        issues.append(Issue("Bundle.entry", "The Bundle has no Encounter.", "required"))

    for entry in index.of_type("Observation"):
        key = _key_of(entry.resource.get("code"), m.OBSERVATIONS)
        if key is None or not _counts(entry.resource):
            continue
        owner = _owner(entry, index, drafts, issues)
        if owner is None:
            continue
        rule = m.OBSERVATIONS[key]
        rank = _sort_key(entry, zone)
        if rule.field not in owner.latest or owner.latest[rule.field][0] <= rank:
            owner.latest[rule.field] = (rank, entry, rule)
    for draft in drafts.values():
        for _, entry, rule in draft.latest.values():
            _observation(draft, rule, entry, issues)

    for entry in index.of_type("Condition"):
        key = _key_of(entry.resource.get("code"), m.CONDITIONS)
        if key is None:
            continue
        owner = _owner(entry, index, drafts, issues)
        if owner is not None:
            _condition(owner, m.CONDITIONS[key], entry)

    if require_screening:
        for draft in drafts.values():
            _missing_screening(draft, issues)
    admissions = [
        Mapped(
            d.encounter.label,
            d.patient.label if d.patient else f"{d.encounter.label}.subject",
            d.values,
            d.sources,
            frozenset(d.flagged),
        )
        for d in drafts.values()
    ]
    return BundleRead(admissions, issues)
