"""Words for the clinical pages: Robson group names, condition labels, option lists.

Copy rules (ui-mockups design §7): the readiness signal never sits near "risk", "should" or
"indicated"; the probability always appears under its Robson group."""

from __future__ import annotations

from typing import Any

GROUP_NAMES: dict[int, str] = {
    1: "Nulliparous, single cephalic pregnancy, 37 weeks or more, spontaneous labour",
    2: "Nulliparous, single cephalic pregnancy, 37 weeks or more, induced labour or cesarean "
    "before labour",
    3: "Multiparous without a previous cesarean, single cephalic pregnancy, 37 weeks or more, "
    "spontaneous labour",
    4: "Multiparous without a previous cesarean, single cephalic pregnancy, 37 weeks or more, "
    "induced labour or cesarean before labour",
    5: "Multiparous with at least one previous cesarean, single cephalic pregnancy, "
    "37 weeks or more",
    6: "Nulliparous, single breech pregnancy",
    7: "Multiparous, single breech pregnancy, including previous cesarean",
    8: "Multiple pregnancy, including previous cesarean",
    9: "Single pregnancy with a transverse or oblique lie, including previous cesarean",
    10: "Single cephalic pregnancy under 37 weeks, including previous cesarean",
}

SUBGROUP_NAMES: dict[str, str] = {
    "2a": "induced labour",
    "2b": "cesarean before labour",
    "4a": "induced labour",
    "4b": "cesarean before labour",
    "5a": "exactly one previous cesarean",
    "5b": "two or more previous cesareans",
}

# (first group, last group, caption) for the strip under the ten boxes; widths follow the mockup.
STRIP_CAPTIONS: tuple[tuple[str, str], ...] = (
    ("c12", "Nulliparous"),
    ("c35", "Multiparous"),
    ("c67", "Breech"),
    ("c8", "Multiple"),
    ("c9", "Transverse"),
    ("c10", "Preterm"),
)

FIELD_NAMES: dict[str, str] = {
    "parity": "Parity",
    "previous_cs_count": "Previous cesareans",
    "plurality": "Number of fetuses",
    "fetal_presentation": "Fetal presentation",
    "gestational_age_weeks": "Gestational age",
    "onset_of_labour": "Onset of labour",
}

CONSISTENCY_TEXT: dict[str, str] = {
    "nulliparous_with_previous_cs": (
        "A woman with a previous cesarean has given birth before. Check which value is wrong."
    ),
}

PRESENTATION_OPTIONS = (
    ("cephalic", "Cephalic"),
    ("breech", "Breech"),
    ("transverse", "Transverse"),
    ("oblique", "Oblique"),
)
PLURALITY_OPTIONS = (("1", "One"), ("2", "Two or more"))
ONSET_OPTIONS = (
    ("spontaneous", "Spontaneous"),
    ("induced", "Induced"),
    ("prelabour_cs", "No labour (pre-labour cesarean)"),
)
PROTEINURIA_OPTIONS = (
    ("negative", "Negative"),
    ("trace", "Trace"),
    ("+", "+"),
    ("++", "++"),
    ("+++", "+++"),
)
DELIVERY_OPTIONS = (
    ("vaginal_spontaneous", "Vaginal, spontaneous"),
    ("vaginal_assisted", "Assisted vaginal"),
    ("cesarean", "Cesarean"),
)
DELIVERY_LABELS = {
    "vaginal_spontaneous": "Vaginal",
    "vaginal_assisted": "Vaginal",
    "cesarean": "Cesarean",
}

# Screening reasons per field (values are the omission_reason enum).
REASONS: dict[str, tuple[tuple[str, str], ...]] = {
    "bp": (
        ("equipment_unavailable", "Equipment unavailable or broken"),
        ("declined", "Woman declined"),
        ("no_time_before_delivery", "No time before delivery"),
        ("other", "Other"),
    ),
    "protein": (
        ("supplies_out_of_stock", "Reagent strips out of stock"),
        ("declined", "Woman declined"),
        ("no_time_before_delivery", "No time before delivery"),
        ("other", "Other"),
    ),
    "glucose": (
        ("equipment_unavailable", "Glucometer or strips unavailable"),
        ("declined", "Woman declined"),
        ("no_time_before_delivery", "No time before delivery"),
        ("other", "Other"),
    ),
}
REASON_TEXT = {value: text for options in REASONS.values() for value, text in options}
REASON_TEXT["equipment_unavailable"] = "equipment unavailable"
REASON_TEXT["supplies_out_of_stock"] = "reagent strips out of stock"
REASON_TEXT["declined"] = "woman declined"
REASON_TEXT["no_time_before_delivery"] = "no time before delivery"
REASON_TEXT["other"] = "other reason"

SCREENING_NAMES = {"bp": "Blood pressure", "protein": "Proteinuria", "glucose": "Glucose"}

# Model inputs outside the Robson section, in the mockup's order: (name, label, unit).
DETAIL_ORDER: tuple[tuple[str, str, str], ...] = (
    ("height_cm", "Height", "cm"),
    ("weight_kg", "Weight", "kg"),
    ("anc_contacts", "Antenatal visits", ""),
    ("gravidity", "Gravidity", ""),
    ("abortions", "Abortions or miscarriages", ""),
    ("age_first_pregnancy", "Age at first pregnancy", "years"),
    ("living_children_band", "Living children", ""),
    ("died_children", "Children who died", ""),
    ("previous_stillbirth", "Previous stillbirth", ""),
    ("previous_preterm", "Previous preterm births", ""),
    ("hiv_positive", "HIV positive", ""),
    ("preeclampsia_recorded", "Pre-eclampsia recorded", ""),
    ("gdm_recorded", "Gestational diabetes recorded", ""),
    ("ivf", "IVF pregnancy", ""),
    ("antenatal_admission", "Admitted during pregnancy", ""),
    ("malaria_in_pregnancy", "Malaria in pregnancy", ""),
    ("gu_infection", "Genito-urinary infection", ""),
    ("insurance_type", "Insurance", ""),
    ("education_level", "Education", ""),
    ("residency", "Residence", ""),
    ("marital_status", "Marital status", ""),
    ("occupation", "Occupation", ""),
)
DETAIL_LABELS = {name: label for name, label, _ in DETAIL_ORDER}
DETAIL_UNITS = {name: unit for name, _, unit in DETAIL_ORDER if unit}
# Levels shown differently from how the model was trained (the stored value is the trained one).
LEVEL_TEXT = {
    "living_children_band": {"0": "None", "Less than 4": "Fewer than 4", "4 or more": "4 or more"},
}
# Inputs the model schema lists that the form asks for elsewhere or derives.
NOT_DETAILS = frozenset(
    {
        "gestational_age_weeks",
        "ga_band_lower",
        "ga_band_upper",
        "parity",
        "previous_cs_count",
        "plurality",
        "fetal_presentation",
        "maternal_age",
        "onset_of_labour",
        "facility_id",
    }
)

_PRESENTATION_TEXT = {
    "cephalic": "cephalic",
    "breech": "breech",
    "transverse": "transverse",
    "oblique": "oblique",
    "non_cephalic": "not cephalic",
}
_ONSET_TEXT = {
    "spontaneous": "spontaneous",
    "induced": "induced",
    "prelabour_cs": "no labour",
}


def _num(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number == int(number) else str(number)


def condition_label(field: str, op: str, expected: Any) -> str:
    """The sentence for one rule condition, as the mockups word them."""
    members = expected if isinstance(expected, list | tuple) else None
    if field == "parity":
        return "Nulliparous" if op == "eq" and expected == 0 else "Multiparous"
    if field == "previous_cs_count":
        if op == "eq" and expected == 0:
            return "No previous cesarean"
        if op == "eq" and expected == 1:
            return "Exactly one previous cesarean"
        if op == "ge" and expected == 1:
            return "At least one previous cesarean"
        return f"{_num(expected)} or more previous cesareans"
    if field == "plurality":
        return "One fetus" if op == "eq" else "Two or more fetuses"
    if field == "fetal_presentation":
        if members:
            return "Transverse or oblique lie"
        return {"cephalic": "Cephalic presentation", "breech": "Breech presentation"}.get(
            str(expected), f"{expected} presentation"
        )
    if field == "gestational_age_weeks":
        return f"{_num(expected)} weeks or more" if op == "ge" else f"Under {_num(expected)} weeks"
    if field == "onset_of_labour":
        if members:
            return "Induced, or cesarean before labour"
        return {
            "spontaneous": "Spontaneous labour",
            "induced": "Induced labour",
            "prelabour_cs": "Cesarean before labour",
        }.get(str(expected), str(expected))
    return f"{field} {op} {expected}"


def input_value_text(field: str, value: Any) -> str:
    """How an entered Robson input is shown in the 'val' column of the trace."""
    if value is None:
        return "not entered"
    if field == "parity":
        return f"parity {value}"
    if field == "previous_cs_count":
        return "none" if value == 0 else f"{value} previous"
    if field == "plurality":
        return "one" if value == 1 else "two or more"
    if field == "fetal_presentation":
        return _PRESENTATION_TEXT.get(str(value), str(value))
    if field == "gestational_age_weeks":
        return f"{_num(value)} weeks"
    if field == "onset_of_labour":
        return _ONSET_TEXT.get(str(value), str(value))
    return str(value)
