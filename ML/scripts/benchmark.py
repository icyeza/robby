"""Performance and edge-case benchmark of the Robson engine and the API.

Usage, from the ML folder::

    uv run python scripts/benchmark.py [--requests 2000] [--model-dir artefacts/demo-synthetic]

Three parts, all on generated inputs (no patient data):

1. **Exhaustive engine sweep.** Every combination of the six Robson inputs, each including
   "not recorded" and the values at the edges of its domain, is classified. It checks that each
   result is resolved, partial or conflict (never an error), and that a complete,
   non-contradictory admission always gets exactly one group. Reports throughput.
2. **API latency.** ``POST /classify`` and ``POST /admissions/assess`` called in-process
   (FastAPI ``TestClient``, so no network) with random admissions. A fifth of them have
   missing fields and a tenth have screening left blank without a reason, which must be
   rejected with 422. Reports p50, p95, p99 and max per endpoint and the status counts.
3. **Model load time**, the cost of a cold start.

Writes ``reports/benchmark.md`` and prints it.
"""

from __future__ import annotations

import argparse
import itertools
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from robson_engine import RobsonInputs, classify, load_rule_set  # noqa: E402

PARITY = [None, 0, 1, 2, 15]
PREVIOUS_CS = [None, 0, 1, 2, 5]
PLURALITY = [None, 1, 2, 3]
PRESENTATION = [None, "cephalic", "breech", "transverse", "oblique", "non_cephalic"]
GESTATION = [None, 20.0, 36.9, 37.0, 42.0, 45.0]
ONSET = [None, "spontaneous", "induced", "prelabour_cs"]
FIELDS = (
    "parity",
    "previous_cs_count",
    "plurality",
    "fetal_presentation",
    "gestational_age_weeks",
    "onset_of_labour",
)
REASON = "equipment_unavailable"


def engine_sweep() -> dict[str, Any]:
    rule_set = load_rule_set()
    statuses: Counter[str] = Counter()
    complete_unresolved = 0
    complete_total = 0
    rejected = 0
    start = time.perf_counter()
    grid = itertools.product(PARITY, PREVIOUS_CS, PLURALITY, PRESENTATION, GESTATION, ONSET)
    for values in grid:
        try:
            inputs = RobsonInputs(**dict(zip(FIELDS, values, strict=True)))
        except ValueError:
            rejected += 1  # outside the domain (refused before classification)
            continue
        result = classify(inputs, rule_set)
        statuses[result.status] += 1
        complete = None not in values and values[3] != "non_cephalic"
        if complete and result.status != "conflict":
            complete_total += 1
            if result.status != "resolved" or result.group is None:
                complete_unresolved += 1
    elapsed = time.perf_counter() - start
    n = sum(statuses.values())
    return {
        "combinations": n + rejected,
        "classified": n,
        "rejected_by_domain": rejected,
        "statuses": dict(statuses),
        "complete_consistent": complete_total,
        "complete_consistent_not_one_group": complete_unresolved,
        "seconds": elapsed,
        "per_second": n / elapsed,
    }


def random_admission(rng: random.Random) -> tuple[dict[str, Any], bool]:
    """One admission body and whether the screening is (deliberately) invalid."""
    body: dict[str, Any] = {
        "parity": rng.choice([0, 0, 1, 2, 3, 5]),
        "previous_cs_count": rng.choice([0, 0, 0, 1, 2]),
        "plurality": rng.choice([1] * 30 + [2]),
        "fetal_presentation": rng.choice(["cephalic"] * 20 + ["breech", "transverse"]),
        "gestational_age_weeks": round(rng.uniform(28, 42), 1),
        "onset_of_labour": rng.choice(["spontaneous", "spontaneous", "induced", "prelabour_cs"]),
        "maternal_age": rng.randint(15, 45),
        "facility_id": rng.choice(["Site A", "Site B"]),
        "anc_contacts": rng.randint(0, 8),
    }
    if body["parity"] == 0:
        body["previous_cs_count"] = 0
    if rng.random() < 0.2:  # incomplete: one to three inputs not recorded
        for field in rng.sample(FIELDS, rng.randint(1, 3)):
            body[field] = None
    screening: dict[str, Any] = {
        "blood_pressure": {"systolic": rng.randint(90, 170), "diastolic": rng.randint(55, 110)},
        "proteinuria": rng.choice([{"value": "negative"}, {"not_measured_reason": REASON}]),
        "glucose": {"value_mmol_l": round(rng.uniform(3.5, 9.0), 1)},
    }
    invalid = rng.random() < 0.1
    if invalid:
        screening[rng.choice(list(screening))] = {}
    body["screening"] = screening
    return body, invalid


def percentiles(ms: list[float]) -> dict[str, float]:
    q = statistics.quantiles(ms, n=100)
    return {"p50": q[49], "p95": q[94], "p99": q[98], "max": max(ms)}


def api_latency(n: int, model_dir: Path, seed: int) -> dict[str, Any]:
    os.environ["ROBSON_MODEL_DIR"] = str(model_dir)
    from fastapi.testclient import TestClient

    import api.main

    out: dict[str, Any] = {}
    start = time.perf_counter()
    with TestClient(api.main.app) as client:
        out["load_seconds"] = time.perf_counter() - start
        out["model"] = client.get("/model").json().get("version_label")
        rng = random.Random(seed)
        bodies = [random_admission(rng) for _ in range(n)]
        for _ in range(20):  # warm-up
            client.post("/admissions/assess", json=bodies[0][0])
        for path in ("/classify", "/admissions/assess"):
            times: list[float] = []
            codes: Counter[str] = Counter()
            wrong = 0
            for body, invalid in bodies:
                payload = body if path != "/classify" else {f: body[f] for f in FIELDS}
                t0 = time.perf_counter()
                response = client.post(path, json=payload)
                times.append((time.perf_counter() - t0) * 1000)
                key = str(response.status_code)
                if response.status_code == 200:
                    data = response.json()
                    key = f"200 {data.get('classification', data).get('status')}"
                codes[key] += 1
                if path == "/admissions/assess" and invalid != (response.status_code == 422):
                    wrong += 1
            out[path] = {**percentiles(times), "codes": dict(codes), "screening_mismatches": wrong}
    return out


def report(sweep: dict[str, Any], api: dict[str, Any], n: int) -> str:
    lines = [
        "# Benchmark",
        "",
        f"Machine: {platform.platform()}, {platform.processor() or platform.machine()}, "
        f"Python {platform.python_version()}. Model: `{api['model']}`. Generated inputs only.",
        "",
        "## Exhaustive Robson engine sweep",
        "",
        f"* {sweep['combinations']:,} input combinations (every value, including not recorded "
        f"and domain edges); {sweep['rejected_by_domain']:,} refused as out of domain, "
        f"{sweep['classified']:,} classified.",
        f"* Status counts: {sweep['statuses']}. No exceptions.",
        f"* Complete, non-contradictory admissions: {sweep['complete_consistent']:,}; "
        f"not resolved to exactly one group: **{sweep['complete_consistent_not_one_group']}**.",
        f"* Throughput: **{sweep['per_second']:,.0f} classifications/s** "
        f"({sweep['seconds']:.2f} s in total).",
        "",
        f"## API latency ({n:,} requests per endpoint, in-process, sequential)",
        "",
        "| Endpoint | p50 ms | p95 ms | p99 ms | max ms | Responses |",
        "|---|---|---|---|---|---|",
    ]
    for path in ("/classify", "/admissions/assess"):
        r = api[path]
        codes = ", ".join(f"{k}: {v}" for k, v in sorted(r["codes"].items()))
        lines.append(
            f"| `{path}` | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | "
            f"{r['max']:.1f} | {codes} |"
        )
    lines += [
        "",
        "Screening that was left blank without a reason and not rejected with 422 (or valid "
        f"screening that was rejected): **{api['/admissions/assess']['screening_mismatches']}**.",
        "",
        f"Cold start (rule set and model load): {api['load_seconds']:.2f} s.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--model-dir", type=Path, default=ML_DIR / "artefacts" / "demo-synthetic")
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    sweep = engine_sweep()
    api = api_latency(args.requests, args.model_dir, args.seed)
    text = report(sweep, api, args.requests)
    out = ML_DIR / "reports" / "benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
