"""Demo mode smoke tests: load into a throwaway *_demo database (no model), refuse others, purge."""

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from robby import events
from robby.config import Settings
from robby.demo import cli as demo_cli
from robby.demo import loader
from robby.models import (
    Admission,
    AdmissionSource,
    AppUser,
    DeliveryOutcome,
    Facility,
    LabRequest,
    LabResult,
    ReferenceSet,
    Role,
)
from robby.timeutil import utcnow
from tests.dbutil import server_url_from_env

runner = CliRunner()


@pytest.fixture
def demo_url() -> Iterator[URL]:
    url = server_url_from_env().set(database=f"robby_t{uuid.uuid4().hex[:8]}_demo")
    try:
        yield url
    finally:
        loader.drop_database(url)


def assert_demo_patients(db: Session) -> None:
    """Every form admission has a number and a name, entered by a lab technician; numbers are
    unique per facility; outcomes are recorded by clinicians."""
    rows = list(db.scalars(select(Admission).where(Admission.source == AdmissionSource.FORM)))
    assert rows
    assert all(r.patient_number and r.patient_name for r in rows)
    numbers = [(r.facility_id, r.patient_number) for r in rows]
    assert len(numbers) == len(set(numbers))
    assert all(r.supersedes_id is None for r in rows)
    roles = {u.id: u.role for u in db.scalars(select(AppUser))}
    assert {roles[r.entered_by] for r in rows if r.entered_by} == {Role.LAB_TECHNICIAN}
    recorders = {roles[o.recorded_by] for o in db.scalars(select(DeliveryOutcome))}
    assert recorders == {Role.CLINICIAN}


def assert_demo_lab_requests(db: Session) -> None:
    """The lab's list has open requests at every facility, exactly one of them urgent; requests
    are made by clinicians and answered by lab technicians, never in the future."""
    roles = {u.id: u.role for u in db.scalars(select(AppUser))}
    requests = list(db.scalars(select(LabRequest)))
    results = list(db.scalars(select(LabResult)))
    assert {roles[r.requested_by] for r in requests} == {Role.CLINICIAN}
    assert {roles[r.recorded_by] for r in results} == {Role.LAB_TECHNICIAN}
    assert sum(r.urgent for r in requests) == 1
    now = utcnow()
    assert all(r.requested_at <= now for r in requests)
    assert all(r.resulted_at <= now for r in results)


def test_demo_load_refusals_and_purge(
    demo_url: URL, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "get_settings", lambda: settings)
    monkeypatch.setattr(demo_cli, "get_settings", lambda: settings)
    plain = demo_url.render_as_string(hide_password=False)

    summary = loader.load_demo(plain, admissions=30, model_dir=None, say=lambda _: None)
    assert summary.events_ok and summary.events_checked > 60
    assert summary.awaiting == 12 and summary.outcomes >= 10
    assert (summary.lab_open, summary.lab_resulted) == (8, 8)
    assert len(summary.credentials) == 15

    engine = create_engine(demo_url)
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(Facility)) == 4
        assert db.scalar(select(func.count()).select_from(AppUser)) == 15
        assert db.scalar(select(func.count()).select_from(Admission)) == summary.admissions
        assert db.scalar(select(func.count()).select_from(DeliveryOutcome)) == summary.outcomes
        reference = db.scalars(select(ReferenceSet)).one()
        assert reference.name == "DEMO placeholder reference"
        assert events.verify(db).ok
        assert_demo_patients(db)
        assert_demo_lab_requests(db)
    engine.dispose()

    with pytest.raises(loader.DemoError, match="already has data"):
        loader.load_demo(plain, admissions=5, model_dir=None, say=lambda _: None)
    with pytest.raises(loader.DemoError, match="does not end in"):
        loader.load_demo(
            demo_url.set(database="robby_prod").render_as_string(hide_password=False),
            admissions=5,
            model_dir=None,
        )
    refused = runner.invoke(
        demo_cli.demo_app,
        [
            "purge",
            "--yes",
            "--database-url",
            demo_url.set(database="robby").render_as_string(hide_password=False),
        ],
    )
    assert refused.exit_code == 1 and "does not end in" in refused.output
    assert loader.database_exists(demo_url)

    unconfirmed = runner.invoke(demo_cli.demo_app, ["purge", "--database-url", plain])
    assert unconfirmed.exit_code == 1 and loader.database_exists(demo_url)
    done = runner.invoke(demo_cli.demo_app, ["purge", "--yes", "--database-url", plain])
    assert done.exit_code == 0 and not loader.database_exists(demo_url)
