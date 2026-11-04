"""Tests de la capa de datos: Iris (CSV real del repo) y archivos 'sucios' sintéticos."""
import numpy as np
import pandas as pd
import pytest

from tp_ia.data.decisions import DecisionLog
from tp_ia.data.iris import IRIS_SPEC, default_data_dir, load_iris_dataset
from tp_ia.data.loader import build_dataset, dataset_from_frame, read_table
from tp_ia.data.quality import audit
from tp_ia.data.schema import DatasetSpec, Scenario


# --------------------------------------------------------------------- lectura de archivos sucios
def test_read_table_semicolons_decimal_comma_latin1_and_missing_tokens(tmp_path):
    path = tmp_path / "raro.csv"
    path.write_bytes("edad;peso;ciudad;clase\n30;70,5;Córdoba;a\n41;?;Rosario;b\n 52 ;80,25;  ;a\n".encode("latin-1"))
    df = read_table(path)
    assert list(df.columns) == ["edad", "peso", "ciudad", "clase"]
    assert df["edad"].tolist() == [30, 41, 52]
    assert df["peso"].iloc[0] == 70.5 and np.isnan(df["peso"].iloc[1]) and df["peso"].iloc[2] == 80.25
    assert df["ciudad"].iloc[0] == "Córdoba" and pd.isna(df["ciudad"].iloc[2])


def test_read_table_tab_separated_without_header(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("1\t2.5\tx\n2\t3.5\ty\n3\t?\tx\n", encoding="utf-8")
    df = read_table(path, header=False, columns=["a", "b", "c"])
    assert list(df.columns) == ["a", "b", "c"] and len(df) == 3
    assert np.isnan(df["b"].iloc[2]) and df["c"].tolist() == ["x", "y", "x"]


def test_a_column_with_one_text_value_stays_text(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("a,b\n1,x\n2,y\nfoo,z\n", encoding="utf-8")
    assert read_table(path)["a"].tolist() == ["1", "2", "foo"]


# --------------------------------------------------------------------- Iris
def test_iris_loads_from_csv_with_expected_shape_and_types():
    ds = load_iris_dataset()
    assert ds.X.shape == (150, 4) and len(ds.y) == 150
    assert ds.classes == ["setosa", "versicolor", "virginica"]
    assert ds.report["class_counts"] == {"setosa": 50, "versicolor": 50, "virginica": 50}
    assert set(ds.feature_types.values()) == {"numeric"}
    assert ds.report["missing"] == {} and ds.report["rows_with_missing"] == 0
    assert ds.report["duplicates"] == 1
    assert any("duplicadas" in w for w in ds.report["warnings"])


def test_iris_records_every_decision_with_evidence():
    ds = load_iris_dataset()
    assert {"D-01", "D-02", "D-03", "D-04", "D-05"} <= set(ds.log.ids())
    markdown = ds.log.to_markdown()
    assert "D-04" in markdown and "duplicadas = 1" in markdown
    assert len(ds.log.to_records()) == len(ds.log)
    assert ds.summary()["n_rows"] == 150


def test_duplicates_policy_drop_removes_the_repeated_row():
    ds = load_iris_dataset(duplicates="drop")
    assert len(ds.X) == 149 and ds.report["duplicates"] == 0


def test_missing_policy_keep_vs_drop_rows():
    df = read_table(default_data_dir() / "iris.csv")
    df.loc[[0, 10, 20], "sepal_width"] = np.nan
    kept = build_dataset(df, IRIS_SPEC)
    assert len(kept.X) == 150 and kept.report["missing"]["sepal_width"]["count"] == 3
    dropped = build_dataset(df, IRIS_SPEC, missing_policy="drop_rows")
    assert len(dropped.X) == 147 and len(dropped.y) == 147 and dropped.report["rows_with_missing"] == 0
    with pytest.raises(ValueError):
        build_dataset(df, IRIS_SPEC, missing_policy="otra")
    with pytest.raises(ValueError):
        build_dataset(df, IRIS_SPEC, duplicates="otra")


# --------------------------------------------------------------------- objetivo, tipos y ceros
def test_scenarios_remap_the_target_and_reject_unmapped_values():
    spec = DatasetSpec(name="demo", target="clase", scenarios={
        "binaria": Scenario("binaria", {"a": 0, "b": 1, "c": 1}, {0: "no", 1: "sí"}, "a contra el resto")})
    df = pd.DataFrame({"x": range(12), "clase": list("abc") * 4})
    ds = build_dataset(df, spec, scenario="binaria")
    assert ds.classes == [0, 1] and ds.y.value_counts().to_dict() == {1: 8, 0: 4}
    assert ds.class_names == {0: "no", 1: "sí"}
    with pytest.raises(ValueError):
        build_dataset(df, spec, scenario="inexistente")
    incomplete = DatasetSpec(name="demo", target="clase", scenarios={"x": Scenario("x", {"a": 0})})
    with pytest.raises(ValueError):
        build_dataset(df, incomplete, scenario="x")      # 'b' y 'c' no tienen mapeo


def test_declared_nominal_overrides_numeric_dtype_and_bad_names_are_rejected():
    df = pd.DataFrame({"codigo": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1], "edad": range(10), "clase": list("ab") * 5})
    ds = dataset_from_frame(df, target="clase", nominal=["codigo"])
    assert ds.feature_types == {"codigo": "nominal", "edad": "numeric"}
    with pytest.raises(ValueError):
        dataset_from_frame(df, target="clase", nominal=["no_existe"])
    with pytest.raises(ValueError):
        dataset_from_frame(df, target="no_existe")


def test_zero_is_missing_turns_impossible_zeros_into_missing_values():
    df = pd.DataFrame({"chol": [200, 0, 250, 0, 230, 210, 190, 220, 240, 205], "clase": list("ab") * 5})
    ds = build_dataset(df, DatasetSpec(name="demo", target="clase", zero_is_missing=["chol"]))
    assert int(ds.X["chol"].isna().sum()) == 2 and "D-03" in ds.log.ids()


def test_rows_without_a_target_are_dropped_and_logged():
    df = pd.DataFrame({"x": range(10), "clase": ["a", "b", None, "a", "b", "a", "b", None, "a", "b"]})
    ds = dataset_from_frame(df, target="clase")
    assert len(ds.X) == 8 and any(d.title == "Filas sin objetivo" for d in ds.log)


# --------------------------------------------------------------------- auditoría y registro
def test_audit_flags_small_classes_imbalance_duplicates_constant_and_id_columns():
    n = 40
    base = pd.DataFrame({"id": range(n), "const": [1] * n, "x": np.arange(n) % 7})
    y = pd.Series(["a"] * 35 + ["b"] * 5)
    report = audit(base, y, {"id": "numeric", "const": "numeric", "x": "numeric"})
    assert report["constant_columns"] == ["const"] and report["id_like_columns"] == ["id"]
    assert report["duplicates"] == 0
    text = " ".join(report["warnings"])
    assert "Clase b" in text and "Desbalance" in text

    repeated = pd.concat([base, base.iloc[[3]]], ignore_index=True)
    y_repeated = pd.concat([y, y.iloc[[3]]], ignore_index=True)
    assert audit(repeated, y_repeated, {"id": "numeric", "const": "numeric", "x": "numeric"})["duplicates"] == 1


def test_decision_log_markdown_and_records():
    log = DecisionLog()
    log.add("D-09", "Prueba", "A", rationale="porque sí", alternatives=["B"], evidence={"n": 3})
    markdown = log.to_markdown()
    assert "D-09 · Prueba" in markdown and "n = 3" in markdown and "B" in markdown
    assert log.to_records()[0]["id"] == "D-09" and len(log) == 1
