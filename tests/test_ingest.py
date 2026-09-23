from pathlib import Path

import pandas as pd
import pytest

from robson_ml.ingest import file_sha256, infer_kind, inventory, read_workbook, select_sheet


def _workbook(path: Path) -> None:
    main = pd.DataFrame(
        {
            "Age": [25, 30, None, 41],
            "Mode": ["CS", "SVD", "CS", "SVD"],
            "Note": ["free a", "free b", "free c", "free d"],
            "__version__": ["v1", "v1", "v2", "v2"],
        }
    )
    child = pd.DataFrame({"Baby weight": [3.1, 2.9], "_parent_index": [1, 1]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        main.to_excel(writer, sheet_name="main", index=False)
        child.to_excel(writer, sheet_name="babies", index=False)


def test_read_all_sheets_raw(tmp_path: Path) -> None:
    path = tmp_path / "raw.xlsx"
    _workbook(path)
    sheets = read_workbook(path)
    assert list(sheets) == ["main", "babies"]
    assert sheets["main"].shape == (4, 4)


def test_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="raw export not found"):
        read_workbook(tmp_path / "absent.xlsx")


def test_infer_kind() -> None:
    assert infer_kind(pd.Series([None, None], dtype=object)) == "empty"
    assert infer_kind(pd.Series([1, "2", 3.5], dtype=object)) == "numeric"
    assert infer_kind(pd.Series(["a", "b", "a"], dtype=object)) == "categorical"
    assert infer_kind(pd.Series([f"t{i}" for i in range(40)], dtype=object)) == "text"
    assert infer_kind(pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))) == "datetime"


def test_inventory_counts_and_versions(tmp_path: Path) -> None:
    path = tmp_path / "raw.xlsx"
    _workbook(path)
    inv = inventory(read_workbook(path))
    main = inv["sheets"][0]
    assert (main["sheet"], main["n_rows"], main["n_cols"]) == ("main", 4, 4)
    age = next(c for c in main["columns"] if c["name"] == "Age")
    assert (age["kind"], age["n_nonnull"]) == ("numeric", 3)
    assert main["versions"]["v2"]["nonnull_by_column"]["Age"] == 1
    assert "versions" not in inv["sheets"][1]


def test_inventory_contains_no_cell_values(tmp_path: Path) -> None:
    path = tmp_path / "raw.xlsx"
    _workbook(path)
    text = str(inventory(read_workbook(path)))
    for value in ["free a", "SVD", "3.1", "41"]:
        assert value not in text


def test_select_sheet(tmp_path: Path) -> None:
    path = tmp_path / "raw.xlsx"
    _workbook(path)
    sheets = read_workbook(path)
    assert select_sheet(sheets, "babies").shape == (2, 2)
    with pytest.raises(ValueError, match="several sheets"):
        select_sheet(sheets, None)


def test_file_sha256(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert file_sha256(path).startswith("ba7816bf")


def test_inventory_handles_duplicate_column_names() -> None:
    df = pd.DataFrame([[1, 2]], columns=["Dup", "Dup"])
    inv = inventory({"s": df})
    names = [c["name"] for c in inv["sheets"][0]["columns"]]
    assert names == ["Dup", "Dup"]
