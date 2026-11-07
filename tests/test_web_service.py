"""Tests de la capa de servicio de la web: todo lo que la interfaz pide, sin abrir ningún navegador."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from tp_ia.web import LabService, to_jsonable


@pytest.fixture()
def lab():
    return LabService()


@pytest.fixture()
def iris(lab):
    lab.select("iris")
    return lab


@pytest.fixture()
def tennis(lab):
    lab.select("play_tennis")
    return lab


def roundtrip(value):
    """Lo que le llega a la interfaz: pasa por JSON y vuelve."""
    return json.loads(json.dumps(value, allow_nan=False))


# --------------------------------------------------------------------- utilidades
def test_to_jsonable_converts_numpy_pandas_and_non_finite_numbers():
    data = {1: np.int64(3), "x": np.float32(2.5), "n": float("nan"), "i": math.inf, "a": np.array([1, 2]),
            "b": np.bool_(True), "t": (1, np.nan), "none": pd.NA, "df": pd.DataFrame({"u": [1]})}
    out = to_jsonable(data)
    assert out == {"1": 3, "x": 2.5, "n": None, "i": None, "a": [1, 2], "b": True, "t": [1, None], "none": None,
                   "df": [{"u": 1}]}
    json.dumps(out, allow_nan=False)


def test_unknown_methods_unknown_datasets_and_missing_selection_are_rejected(lab):
    with pytest.raises(ValueError, match="desconocido"):
        lab.call("os_system", {})
    with pytest.raises(ValueError, match="Dataset desconocido"):
        lab.select("no_existe")
    with pytest.raises(RuntimeError, match="elegí un dataset"):
        lab.call("train")


# --------------------------------------------------------------------- datasets
def test_the_catalog_lists_every_included_dataset(lab):
    keys = [d["key"] for d in lab.call("datasets")["datasets"]]
    assert keys == ["play_tennis", "iris", "wine", "breast_cancer"]
    assert all(d["title"] and d["description"] for d in lab.call("datasets")["datasets"])


def test_iris_info_has_what_the_data_tab_shows(lab):
    info = roundtrip(lab.call("select", {"key": "iris"}))
    assert (info["n_rows"], info["n_features"], info["target"]) == (150, 4, "species")
    assert [c["count"] for c in info["classes"]] == [50, 50, 50]
    assert info["split"] == {"n_train": 105, "n_test": 45, "test_size": 0.3}
    assert {"D-01", "D-02", "D-03", "D-04", "D-05", "D-06"} <= {d["id"] for d in info["decisions"]}
    assert info["quality"]["duplicates"] == 1 and len(info["preview"]) == 8
    width = next(f for f in info["features"] if f["name"] == "sepal_width")
    assert width["kind"] == "numeric" and width["outliers"] == 4 and width["default"] == pytest.approx(3.0)
    assert len(width["hist"]["edges"]) == 13
    totals = {c: sum(counts) for c, counts in width["hist"]["by_class"].items()}
    assert totals == {"setosa": 50, "versicolor": 50, "virginica": 50}


def test_play_tennis_uses_every_case_for_training_and_describes_nominal_features(tennis):
    info = roundtrip(tennis.info())
    assert info["split"] == {"n_train": 14, "n_test": 0, "test_size": 0.0}
    outlook = next(f for f in info["features"] if f["name"] == "Outlook")
    assert outlook["kind"] == "nominal" and outlook["values"] == ["Overcast", "Rain", "Sunny"]
    assert outlook["counts"] == {"Overcast": 4, "Rain": 5, "Sunny": 5} and outlook["default"] in outlook["values"]
    result = roundtrip(tennis.train())
    assert result["metrics"]["test"] is None and result["metrics"]["train"]["accuracy"] == 1.0
    assert result["tree"]["root"]["feature"] == "Outlook" and result["tree"]["n_nodes"] == 8


@pytest.mark.parametrize("key, rows, classes", [("wine", 178, 3), ("breast_cancer", 569, 2)])
def test_scikit_learn_datasets_load_through_the_same_loader(lab, key, rows, classes):
    info = roundtrip(lab.select(key))
    assert info["n_rows"] == rows and len(info["classes"]) == classes
    assert all(f["kind"] == "numeric" for f in info["features"])


def test_a_csv_uploaded_from_the_interface_becomes_the_current_dataset(lab):
    text = "edad;peso;clase\n30;70,5;a\n41;?;b\n52;80,25;a\n33;66,0;b\n45;71,1;a\n38;69,9;b\n"
    info = roundtrip(lab.call("load_csv", {"text": text, "target": "clase", "name": "Mis datos"}))
    assert (info["n_rows"], info["title"]) == (6, "Mis datos") and info["quality"]["missing"]["peso"]["count"] == 1
    assert [c["count"] for c in info["classes"]] == [3, 3]
    with pytest.raises(ValueError, match="objetivo"):
        lab.call("load_csv", {"text": text, "target": "no_existe"})


# --------------------------------------------------------------------- entrenar
def test_training_returns_the_tree_metrics_and_importances(iris):
    result = roundtrip(iris.call("train"))
    assert result["tree"]["root"]["id"] == 0 and result["dot"].startswith("digraph Arbol {")
    assert result["metrics"]["test"]["n"] == 45 and result["metrics"]["train"]["n"] == 105
    assert result["metrics"]["test"]["accuracy"] > 0.85 and result["metrics"]["train"]["accuracy"] >= 0.95
    assert sum(sum(row) for row in result["metrics"]["test"]["confusion"]) == 45
    values = [i["value"] for i in result["importances"]]
    assert values == sorted(values, reverse=True) and sum(values) == pytest.approx(1.0)
    assert result["params"]["criterion"] == "gain_ratio" and result["fit_time"] > 0
    assert "petal" in result["text"]


def test_parameters_are_cleaned_and_pruning_shrinks_the_tree(iris):
    full = iris.train()
    pruned = iris.train({"prune": True, "confidence": 0.05, "foo": 1, "max_depth": 3.0})
    assert pruned["params"]["prune"] is True and pruned["params"]["max_depth"] == 3 and "foo" not in pruned["params"]
    assert pruned["tree"]["n_nodes"] <= full["tree"]["n_nodes"] and pruned["tree"]["depth"] <= 3
    assert iris.train({"max_depth": None})["params"]["max_depth"] is None


def test_the_split_can_change_and_is_reproducible(iris):
    assert iris.call("set_split", {"test_size": 0.5, "seed": 3}) == {"n_train": 75, "n_test": 75, "test_size": 0.5}
    a = iris.train()["metrics"]["test"]["accuracy"]
    iris.call("set_split", {"test_size": 0.5, "seed": 3})
    assert iris.train()["metrics"]["test"]["accuracy"] == a
    for bad in (-0.1, 0.95):
        with pytest.raises(ValueError):
            iris.call("set_split", {"test_size": bad})


def test_the_forest_uses_the_catedra_preset_unless_overridden(iris):
    result = roundtrip(iris.call("train_forest"))
    assert (result["params"]["n_estimators"], result["params"]["max_depth"], result["params"]["random_state"]) == (100, 10, 42)
    assert result["metrics"]["test"]["accuracy"] > 0.85 and result["metrics"]["train"]["accuracy"] > 0.95
    assert iris.train_forest({"n_estimators": 7})["params"]["n_estimators"] == 7


# --------------------------------------------------------------------- predecir un caso
def test_predicting_a_case_returns_both_models_and_the_path(iris):
    row = {f["name"]: f["default"] for f in iris.info()["features"]}
    out = roundtrip(iris.call("predict", {"row": row}))
    assert out["classes"] == ["setosa", "versicolor", "virginica"] and out["missing"] == []
    for model in ("dt", "rf"):
        assert sum(out[model]["proba"]) == pytest.approx(1.0) and out[model]["label"] in out["classes"]
    trace = {int(k): v for k, v in out["dt"]["trace"].items()}
    assert trace[0] == 1.0 and out["dt"]["rules"]
    root_leaves = [i for i, w in trace.items() if w == 1.0]
    assert len(root_leaves) >= 2                                              # el camino completo pesa 1 en cada nodo


def test_a_flower_of_each_species_is_classified_and_missing_values_split_the_path(iris):
    setosa = {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.4, "petal_width": 0.2}
    virginica = {"sepal_length": 6.8, "sepal_width": 3.0, "petal_length": 5.6, "petal_width": 2.1}
    assert iris.call("predict", {"row": setosa})["dt"]["label"] == "setosa"
    assert iris.call("predict", {"row": virginica})["rf"]["label"] == "virginica"
    broken = iris.call("predict", {"row": {**setosa, "petal_length": None, "petal_width": ""}})
    assert broken["missing"] == ["petal_length", "petal_width"]
    assert sum(broken["dt"]["proba"]) == pytest.approx(1.0)
    assert any(0 < w < 1 for w in broken["dt"]["trace"].values())             # el caso se reparte entre ramas


def test_predicting_with_nominal_attributes_and_an_unseen_value(tennis):
    row = {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}
    assert tennis.call("predict", {"row": row})["dt"]["label"] == "No"
    unseen = tennis.call("predict", {"row": {**row, "Outlook": "Foggy"}})
    assert sum(unseen["dt"]["proba"]) == pytest.approx(1.0)


# --------------------------------------------------------------------- comparar y experimentar
def test_compare_runs_both_models_on_the_same_partitions(iris):
    out = roundtrip(iris.call("compare", {"rf_params": {"n_estimators": 20}, "n_splits": 3, "n_repeats": 1}))
    assert out["n_partitions"] == 3 and set(out["models"]) == {"DT", "RF"}
    for model in ("DT", "RF"):
        data = out["models"][model]
        assert len(data["per_split"]) == 3 and data["stats"]["accuracy"]["mean"] > 0.85
        assert data["stats"]["gap"]["mean"] == pytest.approx(
            data["stats"]["train_accuracy"]["mean"] - data["stats"]["accuracy"]["mean"])
        assert np.sum(data["confusion"]) == 150                                  # cada caso se prueba una vez
    assert out["paired"]["accuracy"]["n"] == 3 and out["decision"]["id"] == "D-07"
    assert out["classes"] == ["setosa", "versicolor", "virginica"]


def test_the_number_of_folds_is_capped_by_the_smallest_class(tennis):
    out = tennis.call("compare", {"n_splits": 10, "n_repeats": 1, "rf_params": {"n_estimators": 5}})
    assert out["n_splits"] == 5                                                   # la clase chica de Play Tennis tiene 5 casos


def test_the_parameter_sweep_returns_one_row_per_value_and_marks_the_best(iris):
    out = roundtrip(iris.call("sweep", {"kind": "dt", "param": "max_depth", "values": [1, 2, None],
                                        "n_splits": 3, "n_repeats": 1}))
    assert [r["value"] for r in out["rows"]] == [1, 2, None] and out["metric"] == "macro_f1"
    assert out["rows"][0]["test_mean"] < 0.6 < out["rows"][1]["test_mean"]       # un corte solo separa a setosa
    assert out["best"] in (2, None) and all(r["gap"] >= -1e-9 for r in out["rows"])


def test_the_sweep_validates_the_parameter_and_prunes_when_sweeping_the_confidence(iris):
    with pytest.raises(ValueError, match="no permitido"):
        iris.call("sweep", {"kind": "dt", "param": "criterion", "values": ["gain"]})
    with pytest.raises(ValueError, match="no permitido"):
        iris.call("sweep", {"kind": "xx", "param": "max_depth", "values": [2]})
    out = iris.call("sweep", {"kind": "dt", "param": "confidence", "values": [0.05, 0.5], "n_splits": 3, "n_repeats": 1})
    assert out["base"]["prune"] is True
    forest = iris.call("sweep", {"kind": "rf", "param": "n_estimators", "values": [3, 20], "n_splits": 3, "n_repeats": 1})
    assert [r["value"] for r in forest["rows"]] == [3, 20] and "n_estimators" not in forest["base"]


def test_robustness_to_missing_values(iris):
    out = roundtrip(iris.call("robustness", {"fractions": [0, 0.4], "n_splits": 3, "n_repeats": 1,
                                             "rf_params": {"n_estimators": 20}}))
    assert [r["fraction"] for r in out["rows"]] == [0.0, 0.4] and out["metric"] == "macro_f1"
    clean, damaged = out["rows"]
    assert 0 < damaged["DT"] < clean["DT"] <= 1 and 0 < damaged["RF"] < clean["RF"] <= 1
    assert clean["DT_std"] is not None and out["seeds"] == [0]


def test_decisions_are_available_as_records_and_markdown(iris):
    iris.compare(rf_params={"n_estimators": 5}, n_splits=3, n_repeats=1)
    out = iris.call("decisions")
    ids = [d["id"] for d in out["records"]]
    assert ids == sorted(ids) and "D-07" in ids and "## D-06" in out["markdown"]
