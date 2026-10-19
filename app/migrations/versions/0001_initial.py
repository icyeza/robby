"""Initial schema: every table of the Robson Readiness application.

Revision ID: 0001
Revises:
Create Date: 2026-10-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ENUMS: dict[str, tuple[str, ...]] = {
    "user_role": (
        "clinician",
        "facility_admin",
        "model_custodian",
        "national_reviewer",
        "system_admin",
    ),
    "admission_source": ("form", "fhir"),
    "omission_field": ("blood_pressure", "proteinuria", "glucose"),
    "omission_reason": (
        "equipment_unavailable",
        "supplies_out_of_stock",
        "declined",
        "no_time_before_delivery",
        "other",
    ),
    "classification_status": ("resolved", "partial", "conflict"),
    "model_state": ("candidate", "shadow", "active", "retired"),
    "prediction_mode": ("active", "shadow"),
    "delivery_mode": ("vaginal_spontaneous", "vaginal_assisted", "cesarean"),
    "reference_kind": ("vogel", "c_model"),
}

TABLES_IN_CREATION_ORDER = (
    "facility",
    "app_user",
    "user_session",
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
)

IMMUTABLE_TABLES = (
    "admission",
    "screening_omission",
    "classification",
    "prediction",
    "delivery_outcome",
    "reconciled_outcome",
    "event_log",
)

# Built with || rather than RAISE's % placeholders so no percent sign passes through the driver.
FORBID_CHANGE_FUNCTION = """
CREATE FUNCTION forbid_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION USING
        MESSAGE = TG_OP || ' on ' || TG_TABLE_NAME || ' is not allowed: rows in this table are immutable',
        ERRCODE = 'restrict_violation';
END;
$$
"""


def enum(name: str) -> pg.ENUM:
    """Column type for an enum created up front (never re-created per table)."""
    return pg.ENUM(*ENUMS[name], name=name, create_type=False)


def id_column() -> sa.Column:
    return sa.Column(
        "id", pg.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), nullable=False
    )


def created_at() -> sa.Column:
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def uuid_column(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, pg.UUID(as_uuid=True), nullable=nullable)


def timestamp(name: str, nullable: bool = False, default_now: bool = False) -> sa.Column:
    default = sa.text("now()") if default_now else None
    return sa.Column(name, sa.DateTime(timezone=True), server_default=default, nullable=nullable)


def text_column(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.Text(), nullable=nullable)


def int_column(name: str, nullable: bool = True) -> sa.Column:
    return sa.Column(name, sa.Integer(), nullable=nullable)


def jsonb(name: str, server_default: str | None = None) -> sa.Column:
    default = sa.text(server_default) if server_default else None
    return sa.Column(name, pg.JSONB(), server_default=default, nullable=False)


def boolean(name: str, default: bool) -> sa.Column:
    return sa.Column(
        name, sa.Boolean(), server_default=sa.true() if default else sa.false(), nullable=False
    )


def pk(table: str, *columns: str) -> sa.PrimaryKeyConstraint:
    return sa.PrimaryKeyConstraint(*(columns or ("id",)), name=op.f(f"pk_{table}"))


def unique(table: str, *columns: str) -> sa.UniqueConstraint:
    return sa.UniqueConstraint(*columns, name=op.f(f"uq_{table}_{'_'.join(columns)}"))


def fk(table: str, column: str, target: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{target}.id"], name=op.f(f"fk_{table}_{column}_{target}")
    )


def check(table: str, name: str, condition: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(condition, name=op.f(f"ck_{table}_{name}"))


def upgrade() -> None:
    bind = op.get_bind()
    for name, values in ENUMS.items():
        pg.ENUM(*values, name=name).create(bind)

    op.create_table(
        "facility",
        id_column(),
        text_column("name"),
        text_column("code"),
        text_column("district"),
        text_column("model_facility_key", nullable=True),
        boolean("active", True),
        created_at(),
        pk("facility"),
        unique("facility", "name"),
        unique("facility", "code"),
        check("facility", "code_format", "code ~ '^[A-Z]{2,6}$'"),
    )

    op.create_table(
        "app_user",
        id_column(),
        text_column("username"),
        text_column("display_name", nullable=True),
        text_column("password_hash"),
        sa.Column("role", enum("user_role"), nullable=False),
        uuid_column("facility_id", nullable=True),
        boolean("active", True),
        boolean("must_change_password", True),
        sa.Column("failed_logins", sa.Integer(), server_default=sa.text("0"), nullable=False),
        timestamp("locked_until", nullable=True),
        timestamp("last_login_at", nullable=True),
        created_at(),
        pk("app_user"),
        unique("app_user", "username"),
        fk("app_user", "facility_id", "facility"),
        check("app_user", "username_lowercase", "username = lower(username)"),
        check(
            "app_user",
            "facility_matches_role",
            "(role IN ('clinician', 'facility_admin')) = (facility_id IS NOT NULL)",
        ),
        check("app_user", "failed_logins_not_negative", "failed_logins >= 0"),
    )

    op.create_table(
        "user_session",
        id_column(),
        uuid_column("user_id"),
        sa.Column("token_hash", sa.CHAR(64), nullable=False),
        created_at(),
        timestamp("last_seen_at"),
        timestamp("expires_at"),
        timestamp("revoked_at", nullable=True),
        text_column("ip", nullable=True),
        text_column("user_agent", nullable=True),
        pk("user_session"),
        unique("user_session", "token_hash"),
        fk("user_session", "user_id", "app_user"),
    )
    op.create_index(op.f("ix_user_session_user_id"), "user_session", ["user_id"])

    op.create_table(
        "rule_set",
        id_column(),
        text_column("version_label"),
        sa.Column("checksum", sa.CHAR(64), nullable=False),
        jsonb("definition"),
        timestamp("loaded_at", default_now=True),
        boolean("is_active", False),
        created_at(),
        pk("rule_set"),
        unique("rule_set", "version_label"),
    )
    op.create_index(
        op.f("uq_rule_set_one_active"),
        "rule_set",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "admission_counter",
        uuid_column("facility_id"),
        sa.Column("next", sa.Integer(), server_default=sa.text("1"), nullable=False),
        created_at(),
        pk("admission_counter", "facility_id"),
        fk("admission_counter", "facility_id", "facility"),
        check("admission_counter", "next_positive", '"next" >= 1'),
    )

    op.create_table(
        "admission",
        id_column(),
        uuid_column("facility_id"),
        text_column("code"),
        timestamp("admitted_at"),
        sa.Column("source", enum("admission_source"), nullable=False),
        uuid_column("entered_by", nullable=True),
        uuid_column("supersedes_id", nullable=True),
        text_column("correction_reason", nullable=True),
        int_column("parity"),
        int_column("previous_cs_count"),
        int_column("plurality"),
        text_column("fetal_presentation", nullable=True),
        sa.Column("gestational_age_weeks", sa.Numeric(4, 1), nullable=True),
        text_column("onset_of_labour", nullable=True),
        int_column("maternal_age"),
        int_column("systolic_bp"),
        int_column("diastolic_bp"),
        text_column("proteinuria", nullable=True),
        sa.Column("glucose_mmol", sa.Numeric(4, 1), nullable=True),
        jsonb("details", server_default="'{}'::jsonb"),
        created_at(),
        pk("admission"),
        unique("admission", "code"),
        fk("admission", "facility_id", "facility"),
        fk("admission", "entered_by", "app_user"),
        fk("admission", "supersedes_id", "admission"),
        check(
            "admission",
            "correction_reason_required",
            "supersedes_id IS NULL OR correction_reason IS NOT NULL",
        ),
        check("admission", "form_entry_has_author", "source = 'fhir' OR entered_by IS NOT NULL"),
    )
    op.create_index(
        op.f("ix_admission_facility_id_admitted_at"), "admission", ["facility_id", "admitted_at"]
    )

    op.create_table(
        "screening_omission",
        uuid_column("admission_id"),
        sa.Column("field", enum("omission_field"), nullable=False),
        sa.Column("reason", enum("omission_reason"), nullable=False),
        text_column("reason_note", nullable=True),
        created_at(),
        pk("screening_omission", "admission_id", "field"),
        fk("screening_omission", "admission_id", "admission"),
    )

    op.create_table(
        "classification",
        id_column(),
        uuid_column("admission_id"),
        uuid_column("rule_set_id"),
        sa.Column("status", enum("classification_status"), nullable=False),
        int_column("robson_group"),
        text_column("subgroup", nullable=True),
        sa.Column(
            "candidates", pg.ARRAY(sa.Integer()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column(
            "resolving_fields", pg.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column(
            "conflict_fields", pg.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        jsonb("trace"),
        uuid_column("supersedes_id", nullable=True),
        created_at(),
        pk("classification"),
        unique("classification", "admission_id", "rule_set_id"),
        fk("classification", "admission_id", "admission"),
        fk("classification", "rule_set_id", "rule_set"),
        fk("classification", "supersedes_id", "classification"),
        check(
            "classification",
            "robson_group_range",
            "robson_group IS NULL OR robson_group BETWEEN 1 AND 10",
        ),
        check(
            "classification",
            "resolved_has_group",
            "status <> 'resolved' OR robson_group IS NOT NULL",
        ),
    )

    op.create_table(
        "model_version",
        id_column(),
        text_column("version_label"),
        text_column("algorithm"),
        text_column("feature_set"),
        jsonb("feature_list"),
        jsonb("hyperparameters"),
        text_column("calibration_method"),
        jsonb("evaluation_metrics"),
        sa.Column("artefact_hash", sa.CHAR(64), nullable=False),
        text_column("artefact_path"),
        jsonb("manifest"),
        sa.Column("training_window_start", sa.Date(), nullable=False),
        sa.Column("training_window_end", sa.Date(), nullable=False),
        sa.Column("state", enum("model_state"), server_default="candidate", nullable=False),
        timestamp("registered_at", default_now=True),
        uuid_column("registered_by", nullable=True),
        uuid_column("approved_by", nullable=True),
        timestamp("approved_at", nullable=True),
        created_at(),
        pk("model_version"),
        unique("model_version", "version_label"),
        fk("model_version", "registered_by", "app_user"),
        fk("model_version", "approved_by", "app_user"),
        check(
            "model_version", "training_window_order", "training_window_start <= training_window_end"
        ),
    )
    op.create_index(
        op.f("uq_model_version_one_active"),
        "model_version",
        ["state"],
        unique=True,
        postgresql_where=sa.text("state = 'active'"),
    )

    op.create_table(
        "model_transition",
        id_column(),
        uuid_column("model_id"),
        sa.Column("from_state", enum("model_state"), nullable=True),
        sa.Column("to_state", enum("model_state"), nullable=False),
        uuid_column("actor_id"),
        text_column("reason"),
        timestamp("at", default_now=True),
        created_at(),
        pk("model_transition"),
        fk("model_transition", "model_id", "model_version"),
        fk("model_transition", "actor_id", "app_user"),
        check("model_transition", "reason_not_blank", "length(btrim(reason)) > 0"),
    )

    op.create_table(
        "prediction",
        id_column(),
        uuid_column("admission_id"),
        uuid_column("model_id"),
        sa.Column("mode", enum("prediction_mode"), nullable=False),
        sa.Column("probability", sa.Numeric(6, 5), nullable=False),
        jsonb("feature_vector"),
        uuid_column("supersedes_id", nullable=True),
        created_at(),
        pk("prediction"),
        unique("prediction", "admission_id", "model_id"),
        fk("prediction", "admission_id", "admission"),
        fk("prediction", "model_id", "model_version"),
        fk("prediction", "supersedes_id", "prediction"),
        check("prediction", "probability_range", "probability >= 0 AND probability <= 1"),
    )
    op.create_index(
        op.f("ix_prediction_model_id_created_at"), "prediction", ["model_id", "created_at"]
    )

    op.create_table(
        "delivery_outcome",
        id_column(),
        uuid_column("admission_id"),
        sa.Column("mode", enum("delivery_mode"), nullable=False),
        timestamp("delivered_at"),
        uuid_column("recorded_by"),
        created_at(),
        pk("delivery_outcome"),
        unique("delivery_outcome", "admission_id"),
        fk("delivery_outcome", "admission_id", "admission"),
        fk("delivery_outcome", "recorded_by", "app_user"),
    )

    op.create_table(
        "reconciled_outcome",
        id_column(),
        uuid_column("prediction_id"),
        uuid_column("delivery_outcome_id"),
        sa.Column("cesarean", sa.Boolean(), nullable=False),
        timestamp("reconciled_at", default_now=True),
        created_at(),
        pk("reconciled_outcome"),
        unique("reconciled_outcome", "prediction_id"),
        fk("reconciled_outcome", "prediction_id", "prediction"),
        fk("reconciled_outcome", "delivery_outcome_id", "delivery_outcome"),
    )

    op.create_table(
        "reference_set",
        id_column(),
        text_column("name"),
        sa.Column("kind", enum("reference_kind"), nullable=False),
        text_column("citation"),
        text_column("population"),
        text_column("version_label"),
        timestamp("loaded_at", default_now=True),
        uuid_column("loaded_by", nullable=True),
        created_at(),
        pk("reference_set"),
        fk("reference_set", "loaded_by", "app_user"),
    )

    op.create_table(
        "reference_value",
        uuid_column("reference_set_id"),
        sa.Column("robson_group", sa.Integer(), nullable=False),
        sa.Column("group_size_pct", sa.Numeric(5, 2), nullable=False),
        sa.Column("cs_rate_pct", sa.Numeric(5, 2), nullable=False),
        created_at(),
        pk("reference_value", "reference_set_id", "robson_group"),
        fk("reference_value", "reference_set_id", "reference_set"),
        check("reference_value", "robson_group_range", "robson_group BETWEEN 1 AND 10"),
        check(
            "reference_value",
            "percentages_range",
            "group_size_pct BETWEEN 0 AND 100 AND cs_rate_pct BETWEEN 0 AND 100",
        ),
    )

    op.create_table(
        "event_log",
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("id", pg.UUID(as_uuid=True), nullable=False),
        timestamp("occurred_at"),
        uuid_column("actor_id", nullable=True),
        text_column("event_type"),
        text_column("entity_type", nullable=True),
        uuid_column("entity_id", nullable=True),
        jsonb("payload"),
        sa.Column("prev_hash", sa.CHAR(64), nullable=False),
        sa.Column("hash", sa.CHAR(64), nullable=False),
        created_at(),
        pk("event_log", "seq"),
        unique("event_log", "id"),
        unique("event_log", "hash"),
        fk("event_log", "actor_id", "app_user"),
    )
    op.create_index(op.f("ix_event_log_occurred_at"), "event_log", ["occurred_at"])
    op.create_index(
        op.f("ix_event_log_entity_type_entity_id"), "event_log", ["entity_type", "entity_id"]
    )

    op.create_table(
        "export_record",
        id_column(),
        uuid_column("facility_id"),
        sa.Column("period_month", sa.Date(), nullable=False),
        timestamp("exported_at", default_now=True),
        uuid_column("exported_by"),
        sa.Column("file_sha256", sa.CHAR(64), nullable=False),
        created_at(),
        pk("export_record"),
        fk("export_record", "facility_id", "facility"),
        fk("export_record", "exported_by", "app_user"),
        check("export_record", "period_is_month_start", "EXTRACT(DAY FROM period_month) = 1"),
    )
    op.execute(FORBID_CHANGE_FUNCTION)
    for table in IMMUTABLE_TABLES:
        op.execute(
            f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION forbid_change()"
        )
        op.execute(
            f"CREATE TRIGGER {table}_no_truncate BEFORE TRUNCATE ON {table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION forbid_change()"
        )


def downgrade() -> None:
    for table in reversed(TABLES_IN_CREATION_ORDER):
        op.drop_table(table)
    op.execute("DROP FUNCTION forbid_change()")
    bind = op.get_bind()
    for name in reversed(ENUMS):
        pg.ENUM(name=name).drop(bind)
