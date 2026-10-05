"""Request and response bodies for the API (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

READINESS_LABEL = (
    "Operational readiness signal: probability of CS under current practice. Not a recommendation."
)

Presentation = Literal["cephalic", "breech", "transverse", "oblique", "non_cephalic"]
Onset = Literal["spontaneous", "induced", "prelabour_cs"]
NotMeasuredReason = Literal[
    "equipment_unavailable", "woman_declined", "no_time_before_delivery", "other"
]
Proteinuria = Literal["negative", "trace", "+", "++", "+++"]

_REASON_HELP = (
    "Why it was not measured. Required when no value is given. "
    "`equipment_unavailable` covers broken equipment and strips or reagents out of stock."
)


class RobsonFields(BaseModel):
    """The six Robson inputs. Leave a field out (or null) when it was not recorded."""

    model_config = ConfigDict(extra="forbid")

    parity: int | None = Field(
        None, ge=0, description="Number of previous births (0 = nulliparous).", examples=[2]
    )
    previous_cs_count: int | None = Field(
        None, ge=0, description="Number of previous cesarean sections.", examples=[1]
    )
    plurality: int | None = Field(
        None, ge=1, description="Number of fetuses (1 = singleton).", examples=[1]
    )
    fetal_presentation: Presentation | None = Field(
        None,
        description=(
            "Fetal presentation. `non_cephalic` is a coarse value meaning breech, transverse "
            "or oblique when the exact one is not known."
        ),
        examples=["cephalic"],
    )
    gestational_age_weeks: float | None = Field(
        None, ge=20, le=45, description="Completed weeks of gestation.", examples=[39]
    )
    onset_of_labour: Onset | None = Field(
        None,
        description="`prelabour_cs` means a cesarean before labour started.",
        examples=["spontaneous"],
    )
    ga_band_lower: float | None = Field(
        None,
        ge=20,
        le=45,
        description="Lower bound of a gestational-age band, used only when the exact GA is "
        "missing. Give both bounds or neither.",
    )
    ga_band_upper: float | None = Field(
        None, ge=20, le=45, description="Upper bound of the gestational-age band."
    )


class BloodPressure(BaseModel):
    """Blood pressure in mmHg, or the reason it was not measured."""

    model_config = ConfigDict(extra="forbid")

    systolic: int | None = Field(None, ge=40, le=300, description="Systolic, mmHg.")
    diastolic: int | None = Field(None, ge=20, le=200, description="Diastolic, mmHg.")
    not_measured_reason: NotMeasuredReason | None = Field(None, description=_REASON_HELP)


class ProteinuriaResult(BaseModel):
    """Urine dipstick protein, or the reason it was not measured."""

    model_config = ConfigDict(extra="forbid")

    value: Proteinuria | None = Field(None, description="Dipstick reading.")
    not_measured_reason: NotMeasuredReason | None = Field(None, description=_REASON_HELP)


class GlucoseResult(BaseModel):
    """Blood glucose in mmol/L, or the reason it was not measured."""

    model_config = ConfigDict(extra="forbid")

    value_mmol_l: float | None = Field(None, gt=0, le=50, description="Glucose, mmol/L.")
    not_measured_reason: NotMeasuredReason | None = Field(None, description=_REASON_HELP)


class Screening(BaseModel):
    """Enforced screening: each field needs a value or a reason for not measuring it."""

    model_config = ConfigDict(extra="forbid")

    blood_pressure: BloodPressure | None = None
    proteinuria: ProteinuriaResult | None = None
    glucose: GlucoseResult | None = None


class AdmissionRequest(RobsonFields):
    """One admission as entered on the form: Robson fields, screening and optional details.

    The optional details only feed the readiness signal; any of them may be left out. Text
    values should use the levels listed by `GET /model` (`categorical_levels`); a level the
    model never saw is treated as unknown and reported back in `unrecognised_inputs`.
    """

    facility_id: str | None = Field(None, description="Facility name.", examples=["Muhima"])
    admitted_at: datetime | None = Field(
        None, description="Admission time (not used by the model)."
    )
    maternal_age: float | None = Field(None, ge=10, le=60, description="Years.", examples=[29])
    screening: Screening | None = Field(
        None, description="Blood pressure, proteinuria and glucose (all three required)."
    )
    height_cm: float | None = Field(None, ge=100, le=220, examples=[160])
    weight_kg: float | None = Field(None, ge=30, le=200, examples=[68])
    anc_contacts: int | None = Field(None, ge=0, description="Antenatal visits.", examples=[4])
    gravidity: int | None = Field(None, ge=1, examples=[3])
    abortions: int | None = Field(None, ge=0, description="Abortions or miscarriages.")
    age_first_pregnancy: float | None = Field(None, ge=10, le=60, description="Years.")
    living_children: int | None = Field(None, ge=0, description="Number of living children.")
    died_children: int | None = Field(None, ge=0, description="Children who died.")
    previous_preterm: int | None = Field(None, ge=0, description="Previous preterm births.")
    previous_stillbirth: bool | None = None
    hiv_positive: bool | None = Field(None, description="null when the status is unknown.")
    preeclampsia_recorded: bool | None = None
    gdm_recorded: bool | None = Field(None, description="Gestational diabetes recorded.")
    ivf: bool | None = Field(None, description="IVF pregnancy.")
    antenatal_admission: bool | None = Field(None, description="Admitted during pregnancy.")
    malaria_in_pregnancy: bool | None = None
    gu_infection: bool | None = Field(None, description="Genito-urinary infection.")
    insurance_type: str | None = Field(None, examples=["Mutuel"])
    education_level: str | None = Field(None, examples=["Completed secondary school"])
    residency: str | None = Field(None, examples=["Urban"])
    marital_status: str | None = Field(None, examples=["Married"])
    occupation: str | None = Field(None, examples=["Full Time Job"])


class TraceItem(BaseModel):
    """One evaluated rule condition."""

    group: int
    subgroup: str | None = Field(description="Set when the condition defines a subgroup.")
    field: str
    op: str = Field(description="eq, ge, lt or in.")
    expected: Any
    outcome: Literal["true", "false", "unknown"]


class Classification(BaseModel):
    """Robson classification of one admission."""

    status: Literal["resolved", "partial", "conflict"] = Field(
        description="`resolved`: one group. `partial`: candidates remain because inputs are "
        "missing. `conflict`: the inputs contradict each other."
    )
    group: int | None = Field(description="Robson group 1-10 when resolved.")
    subgroup: str | None = Field(description="For example `5a`, when the group has subgroups.")
    group_name: str | None = Field(description="Plain description of the resolved group.")
    candidates: list[int] = Field(description="Groups still possible (partial or conflict).")
    resolving_fields: list[str] = Field(
        description="Missing or coarse inputs that would narrow a partial result."
    )
    conflict_fields: list[str] = Field(description="Inputs that contradict each other.")
    consistency_checks_failed: list[str] = Field(
        description="Names of the consistency rules the inputs break."
    )
    trace: list[TraceItem] = Field(description="Every condition evaluated, with its outcome.")
    rule_set_version: str = Field(description="Version of the rule set that produced this result.")


class Readiness(BaseModel):
    """Calibrated probability of cesarean under current practice at facilities like this one."""

    probability: float = Field(ge=0, le=1, examples=[0.71])
    of_100_women_like_her: int = Field(
        description="About how many of 100 similar admissions had a cesarean.", examples=[71]
    )
    label: str = Field(READINESS_LABEL, description="How this number may be used.")
    model_version: str
    inputs_not_given: list[str] = Field(
        description="Optional model inputs left blank; the model imputes them."
    )
    unrecognised_inputs: list[str] = Field(
        description="Inputs whose text value the model never saw; treated as unknown."
    )


class AssessmentResponse(BaseModel):
    """Classification plus the readiness signal, or the reason it is unavailable."""

    classification: Classification
    readiness: Readiness | None
    readiness_unavailable_reason: str | None
    screening: Screening = Field(description="The screening values as recorded.")


class Health(BaseModel):
    """Service status."""

    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None
    model_unavailable_reason: str | None
    rule_set_version: str


class ModelCard(BaseModel):
    """Aggregate description of the active readiness model (no patient-level data)."""

    version_label: str
    algorithm: str
    family: str | None
    feature_set: str
    use_facility: bool | None
    n_features: int
    features: list[str]
    categorical_levels: dict[str, list[str]] = Field(
        description="Text values the model was trained on, per categorical input."
    )
    hyperparameters: dict[str, Any]
    calibration_method: str
    calibration_cv_brier: dict[str, float] | None
    population: str | None
    training_window_start: str | None
    training_window_end: str | None
    n_fit_rows: str | None = Field(description="Small counts are suppressed.")
    n_calibration_rows: str | None
    label: str = READINESS_LABEL


class ErrorDetail(BaseModel):
    """Body of a 422 or 503 error raised by the service itself."""

    error: str
    message: str
    fields: list[str] = []
