"""Build and remove the demo database. All content goes through the real services, so
classification, predictions, events and the hash chain are genuine.

The demo database is separate from the real one: clinical rows and the event log are immutable
by design, so demo data could never be removed from a shared database."""

from __future__ import annotations

import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from robby import events
from robby.audit.references import load_reference_set
from robby.clinical import scoring
from robby.clinical.service import record_outcome, save_admission
from robby.config import APP_ROOT, get_settings
from robby.db import get_sessionmaker
from robby.demo import generate
from robby.facilities.service import create_facility
from robby.migrate import upgrade
from robby.models import (
    AppUser,
    DeliveryMode,
    Facility,
    ModelState,
    ModelVersion,
    ReferenceKind,
    Role,
)
from robby.rulesets import load_rule_set_file
from robby.timeutil import utcnow
from robby.users.service import create_user

SUFFIX = "_demo"
DEMO_PASSWORD = "demo-password-123"  # demo database only
DEFAULT_SEED = 20261012
DEFAULT_ADMISSIONS = 700
DEFAULT_MODEL_DIR = APP_ROOT.parent / "ML" / "artefacts" / "packaged" / "readiness-v1.0"
CANDIDATE_LABEL = "readiness-v1.1"
DEMO_ARTEFACT_DIR = APP_ROOT / "artefacts" / "demo"  # git-ignored

# DEMO PLACEHOLDER reference numbers: invented, loaded into the demo database only, and not to
# be copied anywhere else in the application. They are NOT a published source.
DEMO_REFERENCE_CSV = (
    "robson_group,group_size_pct,cs_rate_pct\n"
    "1,28.0,24.0\n2,14.0,52.0\n3,26.0,5.0\n4,8.0,24.0\n5,9.0,65.0\n"
    "6,1.0,80.0\n7,1.0,75.0\n8,1.5,70.0\n9,0.5,95.0\n10,11.0,28.0\n"
)
DEMO_FACILITIES = (
    ("Demo Hospital A", "DA", "Demo District 1"),
    ("Demo Hospital B", "DB", "Demo District 1"),
    ("Demo Hospital C", "DC", "Demo District 2"),
    ("Demo Hospital D", "DD", "Demo District 2"),
)


class DemoError(Exception):
    pass


def default_demo_url(real_url: str) -> str:
    url = make_url(real_url)
    name = url.database or "robby"
    if not name.endswith(SUFFIX):
        name += SUFFIX
    return url.set(database=name).render_as_string(hide_password=False)


def require_demo_name(url: URL) -> str:
    name = url.database or ""
    if not name.endswith(SUFFIX):
        raise DemoError(f"Refusing: database '{name}' does not end in '{SUFFIX}'.")
    return name


def _maintenance(url: URL) -> Any:
    return create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")


def database_exists(url: URL) -> bool:
    engine = _maintenance(url)
    try:
        with engine.connect() as conn:
            found = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
            ).scalar()
            return bool(found)
    finally:
        engine.dispose()


def create_database(url: URL) -> None:
    require_demo_name(url)
    engine = _maintenance(url)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"CREATE DATABASE \"{url.database}\" TEMPLATE template0 ENCODING 'UTF8'")
            )
    finally:
        engine.dispose()


def drop_database(url: URL) -> None:
    require_demo_name(url)
    engine = _maintenance(url)
    try:
        with engine.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
    finally:
        engine.dispose()


def has_data(url: URL) -> bool:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT to_regclass('public.event_log')")).scalar() is None:
                return False
            return bool(conn.execute(text("SELECT count(*) FROM event_log")).scalar())
    finally:
        engine.dispose()


@dataclass
class Summary:
    admissions: int = 0
    status: Counter[str] = field(default_factory=Counter)
    outcomes: int = 0
    awaiting: int = 0
    scored: int = 0
    cs_by_facility: dict[str, tuple[int, int]] = field(default_factory=dict)
    events_ok: bool = False
    events_checked: int = 0
    model: str | None = None
    notes: list[str] = field(default_factory=list)
    credentials: list[tuple[str, str, str]] = field(default_factory=list)


def _model_facility_keys(model_dir: Path) -> list[str]:
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    for item in schema.get("inputs", []):
        if item["name"] == "facility_id" and item.get("levels"):
            return list(item["levels"])
    return []


def _register_model(url: str, model_dir: Path, actor: str, *, activate: bool = True) -> None:
    """The same command as `robby-admin model register [--activate]`, run against the demo DB."""
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    shutil.rmtree(DEMO_ARTEFACT_DIR / str(manifest["version_label"]), ignore_errors=True)
    DEMO_ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ROBBY_DATABASE_URL"] = url
    env["ROBBY_ARTEFACT_DIR"] = str(DEMO_ARTEFACT_DIR)
    command = [
        sys.executable, "-c", "from robby.cli import app; app()",
        "model", "register", str(model_dir), "--actor", actor,
    ]  # fmt: skip
    if activate:
        command.append("--activate")
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise DemoError(f"Model registration failed: {done.stderr.strip() or done.stdout.strip()}")


def _register_candidate(url: str, model_dir: Path, actor: str) -> str:
    """A copy of the packaged model registered (not activated) as candidate readiness-v1.1, so
    the Models page always has an Active and a Candidate model for the promote/rollback
    walkthrough."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / CANDIDATE_LABEL
        shutil.copytree(model_dir, copy)
        manifest_path = copy / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["version_label"] = CANDIDATE_LABEL
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _register_model(url, copy, actor, activate=False)
    return CANDIDATE_LABEL


def _create_accounts(
    db: Session, url: URL, facilities: list[Facility], summary: Summary
) -> dict[str, AppUser]:
    require_demo_name(url)  # demo accounts may only be created in a *_demo database
    users: dict[str, AppUser] = {}

    def add(username: str, role: Role, facility: Facility | None = None) -> None:
        created = create_user(
            db,
            None,
            username=username,
            role=role,
            facility_id=facility.id if facility else None,
            display_name=f"Demo {role.value.replace('_', ' ')}",
            password=DEMO_PASSWORD,
        )
        users[username] = created.user
        summary.credentials.append(
            (username, role.value, facility.name if facility else "(national)")
        )

    for f in facilities:
        tag = f.code.lower()
        add(f"lab.{tag}", Role.LAB_TECHNICIAN, f)
        add(f"clinician.{tag}", Role.CLINICIAN, f)
        add(f"admin.{tag}", Role.FACILITY_ADMIN, f)
    add("custodian", Role.MODEL_CUSTODIAN)
    add("reviewer", Role.NATIONAL_REVIEWER)
    add("sysadmin", Role.SYSTEM_ADMIN)
    return users


def _mode(rng: random.Random, cesarean: bool) -> DeliveryMode:
    if cesarean:
        return DeliveryMode.CESAREAN
    if rng.random() < 0.07:
        return DeliveryMode.VAGINAL_ASSISTED
    return DeliveryMode.VAGINAL_SPONTANEOUS


def _fresh_database(url: URL, force: bool, say: Callable[[str], Any]) -> None:
    """An empty demo database (dropped and rebuilt with --force), migrated to the latest schema."""
    if database_exists(url):
        if has_data(url):
            if not force:
                raise DemoError(
                    f"Database '{url.database}' already has data. Use --force to drop and rebuild."
                )
            drop_database(url)
            say(f"Dropped existing database {url.database}.")
            create_database(url)
            say(f"Created database {url.database}.")
        else:
            say(f"Database {url.database} exists and is empty.")
    else:
        create_database(url)
        say(f"Created database {url.database}.")
    upgrade(url.render_as_string(hide_password=False))
    say("Ran migrations.")


def _seed_setup(
    factory: sessionmaker[Session], url: URL, keys: list[str], summary: Summary
) -> None:
    """The rule set, the demo facilities and accounts, and the placeholder reference set."""
    with factory() as db:
        load_rule_set_file(db, get_settings().rule_set_path)
        facilities = [
            create_facility(
                db,
                None,  # type: ignore[arg-type]  # set-up runs before any account exists
                name=name,
                code=code,
                district=district,
                model_facility_key=keys[i] if i < len(keys) else "",
            )
            for i, (name, code, district) in enumerate(DEMO_FACILITIES)
        ]
        users = _create_accounts(db, url, facilities, summary)
        load_reference_set(
            db,
            users["sysadmin"],
            csv_bytes=DEMO_REFERENCE_CSV.encode(),
            name="DEMO placeholder reference",
            kind=ReferenceKind.VOGEL,
            citation="Demo placeholder - not a published source",
            population="synthetic",
            version_label="demo-1",
        )
        db.commit()


def _finish(factory: sessionmaker[Session], summary: Summary) -> None:
    """Reconcile outcomes (optional), verify the event log and note the active model."""
    with factory() as db:
        try:
            reconcile = importlib.import_module("robby.monitoring.reconcile").reconcile
        except (ImportError, AttributeError):
            reconcile = None
        if reconcile is not None:
            try:
                reconcile(db)
                db.commit()
                summary.notes.append("Ran outcome reconciliation.")
            except Exception as exc:  # optional step, never blocks the demo
                db.rollback()
                summary.notes.append(f"Reconciliation skipped ({type(exc).__name__}).")
        result = events.verify(db)
        summary.events_ok, summary.events_checked = result.ok, result.checked
        summary.model = db.scalar(
            select(ModelVersion.version_label).where(ModelVersion.state == ModelState.ACTIVE)
        )


def load_demo(
    database_url: str,
    *,
    admissions: int = DEFAULT_ADMISSIONS,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    model_dir: Path | None = DEFAULT_MODEL_DIR,
    say: Callable[[str], Any] = print,
) -> Summary:
    url = make_url(database_url)
    require_demo_name(url)
    real = make_url(get_settings().database_url)
    if (url.database, url.host, url.port) == (real.database, real.host, real.port):
        raise DemoError("Refusing: the target is the real database.")
    summary = Summary()
    _fresh_database(url, force, say)

    plain = url.render_as_string(hide_password=False)
    factory = get_sessionmaker(plain)
    rng = random.Random(seed)
    use_model = model_dir is not None and (model_dir / "manifest.json").exists()
    keys = _model_facility_keys(model_dir) if use_model and model_dir else []
    _seed_setup(factory, url, keys, summary)
    say("Loaded the rule set, 4 facilities, accounts and the demo reference set.")

    if use_model and model_dir is not None:
        _register_model(plain, model_dir, "custodian")
        say("Registered and activated the packaged model (development activation).")
        _register_candidate(plain, model_dir, "custodian")
        say(f"Registered a copy as candidate {CANDIDATE_LABEL}.")
    else:
        summary.notes.append("No model artefact found: admissions are saved without predictions.")
    runtime = scoring.ModelRuntime(DEMO_ARTEFACT_DIR)

    sampler = generate.DetailSampler(model_dir) if use_model and model_dir else None
    _fill(factory, runtime, rng, admissions, summary, say, sampler, random.Random(seed + 1))
    _finish(factory, summary)
    return summary


def _fill(
    factory: sessionmaker[Session],
    runtime: scoring.ModelRuntime,
    rng: random.Random,
    n: int,
    summary: Summary,
    say: Callable[[str], Any],
    sampler: generate.DetailSampler | None = None,
    name_rng: random.Random | None = None,
) -> None:
    now = utcnow()
    name_rng = name_rng or random.Random()
    issued: Counter[str] = Counter()  # patient numbers issued per facility
    times = sorted(now - timedelta(days=rng.uniform(1.6, 90)) for _ in range(max(n - 12, 0)))
    times += sorted(now - timedelta(hours=rng.uniform(0.5, 36)) for _ in range(min(12, n)))
    awaiting_from = max(n - 12, 0)
    flawed = set(rng.sample(range(n), max(1, round(n * 0.04)))) if n else set()
    cs_count: Counter[str] = Counter()
    totals: Counter[str] = Counter()

    with factory() as db:
        facilities = list(db.scalars(select(Facility).order_by(Facility.code)))

        def staff(role: Role) -> dict[uuid.UUID, AppUser]:
            return {
                f.id: db.scalars(
                    select(AppUser).where(AppUser.facility_id == f.id, AppUser.role == role)
                ).one()
                for f in facilities
            }

        labs, clinicians = staff(Role.LAB_TECHNICIAN), staff(Role.CLINICIAN)
        for i, at in enumerate(times):
            fi = rng.choices(range(len(facilities)), weights=generate.FACILITY_WEIGHTS)[0]
            facility = facilities[fi]
            entry, clinician = labs[facility.id], clinicians[facility.id]
            issued[facility.code] += 1
            parsed = generate.make_parsed(
                rng,
                at,
                fi,
                sampler,
                patient_number=f"{facility.code}-P-{issued[facility.code]:05d}",
                name_rng=name_rng,
                low_anc=at >= now - timedelta(days=generate.SHIFT_DAYS),
            )
            if i in flawed:
                parsed = generate.make_flawed(rng, parsed)
            current = save_admission(db, runtime, entry, parsed)
            summary.admissions += 1
            summary.status[current.classification.status.value] += 1
            if current.prediction is not None:
                summary.scored += 1
            if i >= awaiting_from:
                summary.awaiting += 1
                db.commit()
                continue
            p = (
                float(current.prediction.probability)
                if current.prediction is not None
                else generate.FALLBACK_CS_RATE
            )
            p = min(max(p + generate.FACILITY_OFFSETS[fi], 0.01), 0.99)
            cesarean = rng.random() < p
            labour = current.admission.onset_of_labour == "prelabour_cs"
            hours = rng.uniform(1, 3) if labour else rng.uniform(2, 20)
            delivered = min(
                current.admission.admitted_at + timedelta(hours=hours), now - timedelta(minutes=5)
            )
            record_outcome(db, clinician, current.admission, _mode(rng, cesarean), delivered)
            summary.outcomes += 1
            totals[facility.name] += 1
            cs_count[facility.name] += int(cesarean)
            db.commit()
            if (i + 1) % 100 == 0:
                say(f"  {i + 1} of {n} admissions...")
    summary.cs_by_facility = {k: (cs_count[k], totals[k]) for k in sorted(totals)}
