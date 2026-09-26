"""FAKE reference files for tests and the synthetic notebook mode ONLY.

Every value here is invented to exercise code paths. None is a Vogel (2015) value, a WHO
C-Model coefficient or a published prevalence, and none may ever be copied into
``data/reference/`` or reported. The citations say so explicitly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

FAKE_CITATION = "FAKE TEST FIXTURE - not a published source; never report these values"
# Round, obviously artificial numbers: every group 10% of deliveries, CS rate 5% + 9% x group.
FAKE_VOGEL_GROUPS = {
    g: {"group_size_pct": 10.0, "cs_rate_pct": 5.0 + 9.0 * g, "source": "fake"}
    for g in range(1, 11)
}
FAKE_PREVALENCE_GRID = {
    "preeclampsia": ("preeclampsia_recorded", [30.0, 40.0, 50.0, 60.0]),
    "gestational_diabetes": ("gdm_recorded", [5.0, 10.0, 20.0, 40.0]),
}


def fake_vogel(path: Path) -> Path:
    """Write a fake Vogel reference file at ``path``."""
    data = {
        "citation": FAKE_CITATION,
        "version": "fake-v0",
        "source_table": "fake table",
        "population": "fake population",
        "groups": FAKE_VOGEL_GROUPS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def fake_cmodel(path: Path, absent_variable: bool = False) -> Path:
    """Write a fake C-Model file; with ``absent_variable`` one variable has no column."""
    variables = {
        "nulliparous": {"column": "parity", "source": "fake"},
        "previous_cs": {"column": "previous_cs_count", "source": "fake"},
        "non_cephalic": {"column": "fetal_presentation", "source": "fake"},
        "age": {"column": "maternal_age", "source": "fake"},
    }
    terms = [
        {"name": "nullip", "variable": "nulliparous", "kind": "indicator", "equals": 0,
         "coefficient": 0.1, "source": "fake"},
        {"name": "prev_cs", "variable": "previous_cs", "kind": "indicator", "lower": 1,
         "upper": None, "coefficient": 1.0, "source": "fake"},
        {"name": "malpresentation", "variable": "non_cephalic", "kind": "indicator",
         "equals": ["breech", "transverse", "oblique", "non_cephalic"], "coefficient": 2.0,
         "source": "fake"},
        {"name": "age", "variable": "age", "kind": "linear", "centre": 28.0,
         "coefficient": 0.01, "source": "fake"},
    ]  # fmt: skip
    if absent_variable:
        variables["fake_unrecorded"] = {"column": None, "source": "fake"}
        terms.append(
            {
                "name": "fake_unrecorded",
                "variable": "fake_unrecorded",
                "kind": "indicator",
                "equals": "yes",
                "coefficient": 0.5,
                "source": "fake",
            }
        )
    data = {
        "citation": FAKE_CITATION,
        "version": "fake-v0",
        "link": "logit",
        "intercept": {"coefficient": -1.0, "source": "fake"},
        "variables": variables,
        "terms": terms,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def fake_prevalence(path: Path) -> Path:
    """Write a fake prevalence-grid file (anchors are fake too)."""
    conditions = {
        name: {
            "field": field,
            "grid_pct": grid,
            "anchors": [
                {
                    "value_pct": grid[1],
                    "citation": FAKE_CITATION,
                    "source": "fake",
                    "population": "fake",
                }
            ],
        }
        for name, (field, grid) in FAKE_PREVALENCE_GRID.items()
    }
    data = {"version": "fake-v0", "conditions": conditions}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def write_fake_references(root: Path) -> dict[str, Path]:
    """All three fake files under ``root/data/reference`` (their default relative paths)."""
    base = root / "data" / "reference"
    return {
        "vogel": fake_vogel(base / "vogel2015_v1.yaml"),
        "cmodel": fake_cmodel(base / "cmodel_v1.yaml"),
        "prevalence": fake_prevalence(base / "prevalence_v1.yaml"),
    }
