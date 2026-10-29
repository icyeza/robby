"""Smoke tests for sub-project 5: reference CSVs, report arithmetic, unclassified residual,
national aggregates-only, monthly export. All reference numbers here are invented test values."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from robby.audit.counts import awaiting_outcome, birth_counts
from robby.audit.measures import build_report, compare_to_expected
from robby.audit.references import (
    Reference,
    ReferenceFileError,
    ReferenceRow,
    load_reference_set,
    parse_reference_csv,
)
from robby.audit.status import complete_months
from robby.config import Settings
from robby.models import (
    Admission,
    AdmissionSource,
    Classification,
    ClassificationStatus,
    DeliveryMode,
    DeliveryOutcome,
    ExportRecord,
    Facility,
    ReferenceKind,
    Role,
    RuleSet,
)
from robby.web.nav import NAV
from tests.factories import events_of, make_facility, make_rule_set, make_user
from tests.webutil import sign_in_client

ZONE = ZoneInfo("Africa/Kigali")
GOOD_CSV = "robson_group,group_size_pct,cs_rate_pct\n" + "".join(
    f"{g},10,{rate}\n"
    for g, rate in zip(range(1, 11), [20, 30, 10, 20, 70, 80, 80, 50, 100, 40], strict=True)
)


def test_reference_csv_validation(db: Session) -> None:
    assert len(parse_reference_csv(GOOD_CSV)) == 10
    bad_cases = {
        "sum": GOOD_CSV.replace("1,10,20", "1,30,20"),
        "missing group": "\n".join(GOOD_CSV.splitlines()[:-1]) + "\n",
        "rate over 100": GOOD_CSV.replace("2,10,30", "2,10,130"),
        "duplicate": GOOD_CSV + "1,0,0\n",
        "header": "a,b,c\n1,2,3\n",
    }
    for name, text in bad_cases.items():
        with pytest.raises(ReferenceFileError):
            parse_reference_csv(text)
        assert name
    ref = load_reference_set(
        db,
        None,
        csv_bytes=GOOD_CSV.encode(),
        name="Test table",
        kind=ReferenceKind.VOGEL,
        citation="Invented for tests",
        population="Test",
        version_label="t1",
    )
    assert events_of(db, "reference_set.loaded")[-1].entity_id == ref.id
    with pytest.raises(ReferenceFileError):
        load_reference_set(
            db,
            None,
            csv_bytes=GOOD_CSV.encode(),
            name="Test table",
            kind=ReferenceKind.VOGEL,
            citation="x",
            population="x",
            version_label="t1",
        )


def birth(
    db: Session,
    facility: Facility,
    rule_set: RuleSet,
    code: str,
    *,
    group: int | None = None,
    status: ClassificationStatus | None = None,
    cesarean: bool = False,
    delivered: datetime | None = None,
    supersedes: Admission | None = None,
    outcome: bool = True,
    fields: list[str] | None = None,
    patient_number: str | None = None,
    patient_name: str = "Test Patient",
) -> Admission:
    user = make_user(db, role=Role.CLINICIAN, facility=facility)
    delivered = delivered or datetime(2026, 3, 10, 12, tzinfo=UTC)
    adm = Admission(
        facility_id=facility.id,
        code=code,
        admitted_at=delivered - timedelta(hours=6),
        source=AdmissionSource.FORM,
        entered_by=user.id,
        supersedes_id=supersedes.id if supersedes else None,
        correction_reason="fix" if supersedes else None,
        patient_number=patient_number or f"P-{code}",
        patient_name=patient_name,
    )
    db.add(adm)
    db.flush()
    if status is not None or group is not None:
        status = status or ClassificationStatus.RESOLVED
        db.add(
            Classification(
                admission_id=adm.id,
                rule_set_id=rule_set.id,
                status=status,
                robson_group=group if status is ClassificationStatus.RESOLVED else None,
                resolving_fields=fields or [],
                trace={},
            )
        )
    if outcome:
        db.add(
            DeliveryOutcome(
                admission_id=adm.id,
                mode=DeliveryMode.CESAREAN if cesarean else DeliveryMode.VAGINAL_SPONTANEOUS,
                delivered_at=delivered,
                recorded_by=user.id,
            )
        )
    db.flush()
    return adm


def reference() -> Reference:
    rates = [20, 30, 10, 20, 70, 80, 80, 50, 100, 40]
    return Reference(
        "x", "Test table", "vogel", "Invented", "Test", "t1",
        {g: ReferenceRow(g, Decimal(10), Decimal(r)) for g, r in zip(range(1, 11), rates, strict=True)}  # noqa: E501,
    )  # fmt: skip


def small_facility(db: Session) -> Facility:
    facility = make_facility(db)
    rs = make_rule_set(db)
    n = 0

    def add(**kw: object) -> Admission:
        nonlocal n
        n += 1
        return birth(db, facility, rs, f"{facility.code}-{n:04d}", **kw)  # type: ignore[arg-type]

    for i in range(10):  # group 1: 10 births, 4 cesareans
        add(group=1, cesarean=i < 4)
    for _ in range(3):  # group 5: 3 births, 3 cesareans
        add(group=5, cesarean=True)
    add(status=ClassificationStatus.PARTIAL, fields=["gestational_age_weeks"])  # residual
    old = add(group=2, cesarean=True)
    add(group=2, cesarean=False, supersedes=old)  # the correction replaces it
    add(group=1, outcome=False)  # awaiting outcome
    add(group=1, cesarean=True, delivered=datetime(2025, 1, 5, tzinfo=UTC))  # outside the period
    return facility


def test_report_numbers_and_flags(db: Session) -> None:
    facility = small_facility(db)
    start, end = date(2026, 1, 1), date(2026, 12, 31)
    counts = birth_counts(db, start, end, ZONE, facility.id)[facility.id]
    waiting = awaiting_outcome(db, start, end, ZONE, facility.id)[facility.id]
    report = build_report(counts, reference(), waiting)
    assert (report.births, report.cesareans, waiting) == (15, 7, 1)
    g1, g5 = report.rows[0], report.rows[4]
    assert (g1.births, g1.cesareans, g1.rate_pct) == (10, 4, Decimal("40.0"))
    assert g1.contribution_pct == Decimal("26.7")  # 4 / 15
    assert g1.higher_rate and g1.larger_group  # +20.0 pts rate, +56.7 pts size vs invented ref
    assert not g5.higher_rate  # 3 births: below the 10-birth reading threshold
    assert report.rows[1].births == 1  # only the correction counts for group 2
    assert report.residual.births == 1 and report.unclassified_pct == Decimal("6.7")
    comparison = compare_to_expected(counts, reference())
    assert comparison.expected == Decimal("31.4")  # (10 x 20 + 1 x 30 + 3 x 70) / 14
    assert comparison.observed == Decimal("50.0")  # 7 / 14 on resolved births
    # no reference: figures still there, nothing flagged
    plain = build_report(counts, None)
    assert plain.births == 15 and not any(r.higher_rate for r in plain.rows)


def last_month_midpoint(settings: Settings) -> tuple[date, datetime]:
    month = complete_months(datetime.now(settings.zone).date(), 1)[0]
    return month, datetime.combine(month.replace(day=15), datetime.min.time(), tzinfo=UTC)


def test_pages_export_and_national_aggregates_only(
    client: TestClient, db: Session, settings: Settings
) -> None:
    month, mid = last_month_midpoint(settings)
    facility = make_facility(db, name="Test Facility A")
    rs = make_rule_set(db)
    for i in range(4):
        birth(
            db,
            facility,
            rs,
            f"{facility.code}-{i + 1:04d}",
            group=1,
            cesarean=i == 0,
            delivered=mid,
        )
    birth(
        db,
        facility,
        rs,
        f"{facility.code}-0099",
        status=ClassificationStatus.CONFLICT,
        delivered=mid,
    )
    admin = make_user(db, role=Role.FACILITY_ADMIN, facility=facility)
    reviewer = make_user(db, role=Role.NATIONAL_REVIEWER)
    db.commit()

    sign_in_client(client, db, settings, admin)
    page = client.get("/audit")
    assert page.status_code == 200 and "No reference set loaded" in page.text
    assert client.get("/national").status_code == 403

    csv_response = client.get(f"/audit/export.csv?month={month:%Y-%m}")
    assert csv_response.status_code == 200
    lines = csv_response.text.splitlines()
    assert lines[0] == "format_version,facility_code,period_month,robson_group,births,cesareans"
    assert f"v1,{facility.code},{month:%Y-%m},1,4,1" in lines
    assert f"v1,{facility.code},{month:%Y-%m},unclassified,1,0" in lines
    record = db.scalars(select(ExportRecord).where(ExportRecord.facility_id == facility.id)).one()
    assert record.file_sha256 == hashlib.sha256(csv_response.content).hexdigest()
    assert events_of(db, "export.created")
    assert client.get("/audit/export.csv?month=2999-01").status_code == 400

    client.cookies.clear()
    sign_in_client(client, db, settings, reviewer)
    for path in ("/national", f"/national/facilities/{facility.id}", "/national/export.csv"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert f"{facility.code}-0" not in response.text.replace(f'"{facility.code}"', ""), path
    overview = client.get("/national").text
    assert "Test Facility A" in overview and f"{month:%b} exported" in overview
    assert client.get("/audit").status_code == 403


def test_patient_number_reaches_the_facility_admin_and_the_name_nobody_outside_clinical(
    client: TestClient, db: Session, settings: Settings
) -> None:
    month, mid = last_month_midpoint(settings)
    facility = make_facility(db)
    rs = make_rule_set(db)
    birth(db, facility, rs, f"{facility.code}-0001", group=1, delivered=mid)
    birth(
        db,
        facility,
        rs,
        f"{facility.code}-0002",
        status=ClassificationStatus.CONFLICT,
        delivered=mid,
        patient_number="ZQ-99",
        patient_name="Zzyzx Testname",
    )
    users = {role: make_user(db, role=role, facility=facility) for role in Role}
    db.commit()

    def pages_of(role: Role, *extra: str) -> list[str]:
        client.cookies.clear()
        sign_in_client(client, db, settings, users[role])
        paths = [item.path for item in NAV[role]] + list(extra)
        responses = [(path, client.get(path)) for path in paths]
        for path, response in responses:
            assert response.status_code == 200, (role, path)
        return [response.text for _, response in responses]

    *admin_pages, monthly_csv = pages_of(
        Role.FACILITY_ADMIN, f"/audit/export.csv?month={month:%Y-%m}"
    )
    assert all("Zzyzx" not in text for text in [*admin_pages, monthly_csv])
    assert "ZQ-99" not in monthly_csv  # counts only
    for role in (Role.NATIONAL_REVIEWER, Role.MODEL_CUSTODIAN, Role.SYSTEM_ADMIN):
        extra = (
            (f"/national/facilities/{facility.id}", "/national/export.csv")
            if role is Role.NATIONAL_REVIEWER
            else ()
        )
        for text in pages_of(role, *extra):
            assert "ZQ-99" not in text and "Zzyzx" not in text, role
