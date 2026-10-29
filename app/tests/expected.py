"""What the schema must contain (tables, columns, enums, constraints), shared by model and
migration tests."""

from robby.models.enums import (
    AdmissionSource,
    ClassificationStatus,
    DeliveryMode,
    ModelState,
    OmissionField,
    OmissionReason,
    PredictionMode,
    ReferenceKind,
    Role,
)

EXPECTED_TABLES = {
    "facility",
    "app_user",
    "user_session",
    "api_token",
    "rule_set",
    "admission_counter",
    "admission",
    "screening_omission",
    "classification",
    "model_version",
    "model_transition",
    "prediction",
    "delivery_outcome",
    "reconciled_outcome",
    "reference_set",
    "reference_value",
    "event_log",
    "export_record",
}

EXPECTED_ENUMS = {
    "user_role": Role,
    "admission_source": AdmissionSource,
    "omission_field": OmissionField,
    "omission_reason": OmissionReason,
    "classification_status": ClassificationStatus,
    "model_state": ModelState,
    "prediction_mode": PredictionMode,
    "delivery_mode": DeliveryMode,
    "reference_kind": ReferenceKind,
}

IMMUTABLE_TABLES = (
    "admission",
    "screening_omission",
    "classification",
    "prediction",
    "delivery_outcome",
    "reconciled_outcome",
    "event_log",
)
