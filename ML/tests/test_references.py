"""Reference loaders: absent files raise (no fallback), schemas are enforced. All files here
are FAKE fixtures written to temporary directories (tests.synthetic_references)."""

from pathlib import Path

import pytest
import yaml

from robson_ml.references import (
    VOGEL_PATH,
    ReferenceDataMissing,
    ReferenceSchemaError,
    load_prevalence,
    load_vogel,
)
from tests.synthetic_references import FAKE_CITATION, fake_prevalence, fake_vogel


def test_vogel_missing_file_raises_with_path_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "data" / "reference" / "vogel2015_v1.yaml"
    with pytest.raises(ReferenceDataMissing) as error:
        load_vogel(path)
    message = str(error.value)
    assert str(path) in message
    assert "group_size_pct" in message and "cs_rate_pct" in message
    assert "never" in message


def test_default_vogel_path_is_the_spec_path() -> None:
    assert VOGEL_PATH.as_posix() == "data/reference/vogel2015_v1.yaml"


def test_prevalence_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ReferenceDataMissing, match="grid_pct"):
        load_prevalence(tmp_path / "prevalence_v1.yaml")


def test_fake_vogel_loads(tmp_path: Path) -> None:
    ref = load_vogel(fake_vogel(tmp_path / "v.yaml"))
    assert ref.citation == FAKE_CITATION
    assert sorted(ref.groups) == list(range(1, 11))
    assert ref.size(1) == pytest.approx(0.10)
    assert ref.cs_rate(2) == pytest.approx(0.23)
    assert len(ref.sha256) == 64


def _edit(path: Path, change) -> Path:  # type: ignore[no-untyped-def]
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_vogel_rejects_missing_group_and_missing_source(tmp_path: Path) -> None:
    path = _edit(fake_vogel(tmp_path / "a.yaml"), lambda d: d["groups"].pop(10))
    with pytest.raises(ReferenceSchemaError, match="groups 1-10"):
        load_vogel(path)
    path = _edit(fake_vogel(tmp_path / "b.yaml"), lambda d: d["groups"][3].pop("source"))
    with pytest.raises(ReferenceSchemaError, match="source"):
        load_vogel(path)


def test_vogel_rejects_sizes_not_adding_to_100(tmp_path: Path) -> None:
    def inflate(data: dict) -> None:
        data["groups"][1]["group_size_pct"] = 50.0

    with pytest.raises(ReferenceSchemaError, match="add up"):
        load_vogel(_edit(fake_vogel(tmp_path / "v.yaml"), inflate))


def test_vogel_requires_citation_table_population(tmp_path: Path) -> None:
    for key in ("citation", "source_table", "population"):
        path = _edit(fake_vogel(tmp_path / f"{key}.yaml"), lambda d, k=key: d.pop(k))
        with pytest.raises(ReferenceSchemaError, match=key):
            load_vogel(path)


def test_prevalence_grid_must_increase_and_span_anchors(tmp_path: Path) -> None:
    def decreasing(data: dict) -> None:
        data["conditions"]["preeclampsia"]["grid_pct"] = [40.0, 30.0]

    with pytest.raises(ReferenceSchemaError, match="increasing"):
        load_prevalence(_edit(fake_prevalence(tmp_path / "a.yaml"), decreasing))

    def outside(data: dict) -> None:
        data["conditions"]["preeclampsia"]["anchors"][0]["value_pct"] = 99.0

    with pytest.raises(ReferenceSchemaError, match="span"):
        load_prevalence(_edit(fake_prevalence(tmp_path / "b.yaml"), outside))

    def no_anchor(data: dict) -> None:
        data["conditions"]["preeclampsia"]["anchors"] = []

    with pytest.raises(ReferenceSchemaError, match="anchor"):
        load_prevalence(_edit(fake_prevalence(tmp_path / "c.yaml"), no_anchor))


def test_fake_prevalence_loads(tmp_path: Path) -> None:
    ref = load_prevalence(fake_prevalence(tmp_path / "p.yaml"))
    assert ref.conditions["preeclampsia"].field == "preeclampsia_recorded"
    assert ref.conditions["gestational_diabetes"].grid_pct[0] > 0
