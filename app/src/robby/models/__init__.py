"""ORM models for every table of the application (spec §3).

All tables are created by sub-project 1; later sub-projects add behaviour, not tables.
"""

from robby.models.base import Base
from robby.models.clinical import (
    Admission,
    AdmissionCounter,
    Classification,
    DeliveryOutcome,
    ReconciledOutcome,
    ScreeningOmission,
)
from robby.models.enums import (
    FACILITY_ROLES,
    ROLE_LABELS,
    AdmissionSource,
    ClassificationStatus,
    DeliveryMode,
    ModelState,
    OmissionField,
    OmissionReason,
    PredictionMode,
    ReferenceKind,
    Role,
    role_label,
)
from robby.models.events import EventLog
from robby.models.facility import Facility
from robby.models.references import ExportRecord, ReferenceSet, ReferenceValue
from robby.models.registry import ModelTransition, ModelVersion, Prediction
from robby.models.rules import RuleSet
from robby.models.users import ApiToken, AppUser, UserSession

__all__ = [
    "FACILITY_ROLES",
    "ROLE_LABELS",
    "Admission",
    "AdmissionCounter",
    "AdmissionSource",
    "ApiToken",
    "AppUser",
    "Base",
    "Classification",
    "ClassificationStatus",
    "DeliveryMode",
    "DeliveryOutcome",
    "EventLog",
    "ExportRecord",
    "Facility",
    "ModelState",
    "ModelTransition",
    "ModelVersion",
    "OmissionField",
    "OmissionReason",
    "Prediction",
    "PredictionMode",
    "ReconciledOutcome",
    "ReferenceKind",
    "ReferenceSet",
    "ReferenceValue",
    "Role",
    "RuleSet",
    "ScreeningOmission",
    "UserSession",
    "role_label",
]
