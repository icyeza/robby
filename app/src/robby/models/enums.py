"""Enumerations stored as PostgreSQL enum types (spec §3)."""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import Enum as SAEnum


class Role(StrEnum):
    CLINICIAN = "clinician"
    FACILITY_ADMIN = "facility_admin"
    MODEL_CUSTODIAN = "model_custodian"
    NATIONAL_REVIEWER = "national_reviewer"
    SYSTEM_ADMIN = "system_admin"


FACILITY_ROLES = frozenset({Role.CLINICIAN, Role.FACILITY_ADMIN})

ROLE_LABELS: dict[Role, str] = {
    Role.CLINICIAN: "Clinician",
    Role.FACILITY_ADMIN: "Facility administrator",
    Role.MODEL_CUSTODIAN: "Model custodian",
    Role.NATIONAL_REVIEWER: "National reviewer",
    Role.SYSTEM_ADMIN: "System administrator",
}


def role_label(role: Role) -> str:
    """How a role is written on screen."""
    return ROLE_LABELS[role]


class AdmissionSource(StrEnum):
    FORM = "form"
    FHIR = "fhir"
    IMPORT = "import"


class OmissionField(StrEnum):
    BLOOD_PRESSURE = "blood_pressure"
    PROTEINURIA = "proteinuria"
    GLUCOSE = "glucose"


class OmissionReason(StrEnum):
    EQUIPMENT_UNAVAILABLE = "equipment_unavailable"
    SUPPLIES_OUT_OF_STOCK = "supplies_out_of_stock"
    DECLINED = "declined"
    NO_TIME_BEFORE_DELIVERY = "no_time_before_delivery"
    OTHER = "other"


class ClassificationStatus(StrEnum):
    RESOLVED = "resolved"
    PARTIAL = "partial"
    CONFLICT = "conflict"


class ModelState(StrEnum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"


class PredictionMode(StrEnum):
    ACTIVE = "active"
    SHADOW = "shadow"


class DeliveryMode(StrEnum):
    VAGINAL_SPONTANEOUS = "vaginal_spontaneous"
    VAGINAL_ASSISTED = "vaginal_assisted"
    CESAREAN = "cesarean"


class ReferenceKind(StrEnum):
    VOGEL = "vogel"
    C_MODEL = "c_model"


def pg_enum(enum_class: type[StrEnum], name: str) -> SAEnum:
    """A native PostgreSQL enum column type storing the members' values."""
    return SAEnum(
        enum_class,
        name=name,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )
