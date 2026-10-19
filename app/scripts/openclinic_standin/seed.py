"""Fill the stand-in OpenClinic database with invented patients (not real data).

    uv run --with pymysql python scripts/openclinic_standin/seed.py seed        # 60 maternity admissions
    uv run --with pymysql python scripts/openclinic_standin/seed.py add 3       # 3 new admissions, now
    uv run --with pymysql python scripts/openclinic_standin/seed.py add 1 --bad # one with a bad value
    uv run --with pymysql python scripts/openclinic_standin/seed.py reset       # empty the tables

`seed` also adds some non-maternity encounters, which the connector must ignore. Admissions are
spread over the last 20 days (the app only accepts admission times within 30 days). The spread of
parity, previous cesareans, presentation and labour onset covers every Robson group."""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta

import pymysql

FIRST = ["Marie", "Jeanne", "Grace", "Alice", "Claudine", "Beatha", "Rose", "Divine", "Solange",
         "Anita", "Josiane", "Esperance", "Immaculee", "Vestine", "Christine", "Aline", "Yvonne",
         "Francoise", "Providence", "Odette", "Eugenie", "Console", "Daphrose", "Speciose"]  # fmt: skip
LAST = ["Uwase", "Mukamana", "Ingabire", "Nyirahabimana", "Mutesi", "Uwimana", "Niyonsenga",
        "Umutoni", "Kayitesi", "Mukashema", "Nyiransabimana", "Mukeshimana", "Uwamahoro",
        "Iradukunda", "Murekatete", "Mukagasana", "Uwizeye", "Nyirarukundo"]  # fmt: skip

# (weight, para, previous_cs, pregnancy, presentation, onset, gest weeks) per Robson-style case
CASES = [
    (14, "nullip", 0, "Singleton", "Vertex", "Spontaneous", (37, 41)),            # group 1
    (7, "nullip", 0, "Singleton", "Vertex", "Induced", (38, 41)),                 # group 2a
    (4, "nullip", 0, "Singleton", "Vertex", "Elective C-section", (38, 40)),      # group 2b
    (22, "multip", 0, "Singleton", "Vertex", "Spontaneous", (37, 41)),            # group 3
    (7, "multip", 0, "Singleton", "Vertex", "Induced", (38, 41)),                 # group 4a
    (3, "multip", 0, "Singleton", "Vertex", "Elective C-section", (38, 40)),      # group 4b
    (14, "multip", 1, "Singleton", "Vertex", "Spontaneous", (37, 41)),            # group 5
    (3, "multip", 2, "Singleton", "Vertex", "Elective C-section", (37, 40)),      # group 5
    (3, "nullip", 0, "Singleton", "Breech", "Spontaneous", (37, 41)),             # group 6
    (4, "multip", 0, "Singleton", "Breech", "Spontaneous", (37, 41)),             # group 7
    (4, "any", 0, "Twins", "Vertex", "Spontaneous", (34, 39)),                    # group 8
    (2, "any", 0, "Singleton", "Transverse", "Spontaneous", (37, 41)),            # group 9
    (13, "any", 0, "Singleton", "Vertex", "Spontaneous", (28, 36)),               # group 10
]  # fmt: skip


def connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.environ.get("OC_HOST", "127.0.0.1"),
        port=int(os.environ.get("OC_PORT", "3307")),
        user=os.environ.get("OC_ADMIN_USER", "root"),
        password=os.environ.get("OC_ADMIN_PASSWORD", ""),
        database="openclinic",
        autocommit=False,
    )


def gest_text(rng: random.Random, weeks: int) -> str:
    return rng.choice([str(weeks), f"{weeks}+{rng.randint(0, 6)}", f"{weeks}w{rng.randint(0, 6)}d",
                       f"{weeks} weeks"])  # fmt: skip


def add_admission(cur, rng: random.Random, number: int, at: datetime, bad: bool = False) -> None:
    case = rng.choices(CASES, weights=[c[0] for c in CASES])[0]
    _, parity_kind, cs, pregnancy, presentation, onset, (lo, hi) = case
    para = 0 if parity_kind == "nullip" else rng.choice([1, 1, 2, 2, 3, 4]) if parity_kind == "multip" else rng.choice([0, 1, 2])
    para = max(para, cs)
    if cs and para == 0:
        para = cs
    age = rng.randint(17, 42) if para < 3 else rng.randint(26, 44)
    pid = f"OC-{number}"
    cur.execute(
        "INSERT INTO patient VALUES (%s,%s,%s,%s)",
        (pid, rng.choice(FIRST), rng.choice(LAST), (at - timedelta(days=age * 365 + rng.randint(0, 360))).date()),
    )
    cur.execute(
        "INSERT INTO encounter (patient_id, ward, admission_at) VALUES (%s,'Maternity',%s)",
        (pid, at),
    )
    eid = cur.lastrowid
    weeks = rng.randint(lo, hi)
    cur.execute(
        "INSERT INTO obstetric_record VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (eid, para + rng.choice([1, 1, 2]), para, cs, gest_text(rng, weeks), pregnancy,
         "sideways" if bad else presentation, onset),
    )  # fmt: skip
    if rng.random() > 0.08:  # a few admissions have no blood pressure recorded
        for k in range(rng.randint(1, 3)):
            cur.execute(
                "INSERT INTO vitals (encounter_id, measured_at, bp_systolic, bp_diastolic) VALUES (%s,%s,%s,%s)",
                (eid, at + timedelta(minutes=20 * k), rng.randint(100, 150), rng.randint(62, 96)),
            )
    if rng.random() > 0.15:
        cur.execute(
            "INSERT INTO lab_result (encounter_id, test_code, result_text, resulted_at) VALUES (%s,'URINE_PROTEIN',%s,%s)",
            (eid, rng.choices(["neg", "trace", "1+", "2+", "3+"], [60, 12, 14, 9, 5])[0], at + timedelta(minutes=15)),
        )
    if rng.random() > 0.2:
        cur.execute(
            "INSERT INTO lab_result (encounter_id, test_code, result_text, resulted_at) VALUES (%s,'GLUCOSE_MGDL',%s,%s)",
            (eid, str(rng.randint(68, 130)), at + timedelta(minutes=15)),
        )


def add_other_ward(cur, rng: random.Random, number: int, at: datetime) -> None:
    pid = f"OC-{number}"
    cur.execute("INSERT INTO patient VALUES (%s,%s,%s,%s)",
                (pid, rng.choice(FIRST), rng.choice(LAST), datetime(1980, 1, 1).date()))  # fmt: skip
    cur.execute("INSERT INTO encounter (patient_id, ward, admission_at) VALUES (%s,%s,%s)",
                (pid, rng.choice(["General medicine", "Surgery", "Paediatrics"]), at))  # fmt: skip


def next_number(cur) -> int:
    cur.execute("SELECT COALESCE(MAX(CAST(SUBSTRING(patient_id, 4) AS UNSIGNED)), 10000) FROM patient")
    return int(cur.fetchone()[0]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["seed", "add", "reset"])
    parser.add_argument("count", nargs="?", type=int, default=1)
    parser.add_argument("--bad", action="store_true", help="add: give the admission a bad presentation")
    args = parser.parse_args()
    db = connect()
    cur = db.cursor()
    rng = random.Random(2026 if args.command == "seed" else None)
    now = datetime.now().replace(second=0, microsecond=0)
    if args.command == "reset":
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        for table in ("lab_result", "vitals", "obstetric_record", "encounter", "patient"):
            cur.execute(f"TRUNCATE TABLE {table}")
        cur.execute("SET FOREIGN_KEY_CHECKS=1")
        print("Emptied the stand-in OpenClinic database.")
    elif args.command == "seed":
        number = next_number(cur)
        moments = sorted(now - timedelta(hours=rng.uniform(1, 20 * 24)) for _ in range(60))
        for at in moments:
            add_admission(cur, rng, number, at.replace(second=0, microsecond=0))
            number += 1
        for _ in range(15):
            add_other_ward(cur, rng, number, now - timedelta(hours=rng.uniform(1, 20 * 24)))
            number += 1
        print("Seeded 60 maternity admissions and 15 non-maternity encounters.")
    else:
        number = next_number(cur)
        for i in range(args.count):
            add_admission(cur, rng, number + i, now - timedelta(minutes=5 * (args.count - i)), bad=args.bad and i == 0)
        print(f"Added {args.count} maternity admission(s){' (first one has a bad value)' if args.bad else ''}.")
    db.commit()
    db.close()


if __name__ == "__main__":
    main()
