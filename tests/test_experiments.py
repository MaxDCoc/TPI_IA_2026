"""Tests de los experimentos: barrido de parámetros, validación cruzada anidada y robustez a faltantes (sobre Iris)."""
import dataclasses
import json

import pytest

from tp_ia.data import load_iris_dataset
from tp_ia.eval import make_splits, run_experiment
from tp_ia.eval.experiments import (inject_missing, missing_robustness, nested_cv, parameter_sweep,
                                    robustness_table, selection_frequency)

DT = {"DT": ("dt", {})}
RF_SMALL = {"n_estimators": 15, "random_state": 0}
GRID = {"max_depth": [2, None], "min_samples_leaf": [2, 5]}


@pytest.fixture(scope="module")
def iris():
    return load_iris_dataset()


@pytest.fixture(scope="module")
def splits(iris):
    return make_splits(iris.y, n_splits=3, n_repeats=1, seed=0)


# --------------------------------------------------------------------- barrido de un parámetro
def test_sweep_shows_a_stump_cannot_separate_three_classes_and_reports_the_gap(iris, splits):
    sweep = parameter_sweep(iris, "dt", "max_depth", [1, 2, None], splits)
    assert list(sweep["value"]) == [1, 2, None]                   # se conservan los valores originales
    assert list(sweep.columns[:3]) == ["param", "value", "n_partitions"]
    assert (sweep["n_partitions"] == len(splits)).all()
    one, two, _ = sweep["test_macro_f1_mean"]
    assert one < 0.6 < 0.9 < two                                  # un solo corte separa a setosa y nada más
    assert (sweep["gap"] >= -1e-9).all()                          # en train se rinde igual o mejor que en test
    assert sweep["gap"].iloc[-1] > sweep["gap"].iloc[0]           # el árbol sin límite es el que más sobreajusta


def test_sweep_passes_base_params_and_works_for_the_random_forest(iris, splits):
    sweep = parameter_sweep(iris, "rf", "n_estimators", [3, 25], splits, base_params={"random_state": 0},
                            metric="accuracy")
    assert list(sweep["value"]) == [3, 25] and (sweep["test_accuracy_mean"] > 0.85).all()
    assert "test_macro_f1_mean" not in sweep.columns             # solo la métrica pedida


def test_sweep_rejects_repeated_values_empty_values_and_unknown_metrics(iris, splits):
    with pytest.raises(ValueError, match="repetir"):
        parameter_sweep(iris, "dt", "max_depth", [2, 2], splits)
    with pytest.raises(ValueError):
        parameter_sweep(iris, "dt", "max_depth", [], splits)
    with pytest.raises(ValueError, match="metric"):
        parameter_sweep(iris, "dt", "max_depth", [2], splits, metric="roc_auc")


# --------------------------------------------------------------------- validación cruzada anidada
def test_nested_cv_returns_one_row_per_outer_split_with_a_choice_from_the_grid(iris):
    outer = make_splits(iris.y, n_splits=3, n_repeats=1, seed=1)
    nested = nested_cv(iris, "dt", GRID, outer, inner_splits=3)
    assert len(nested) == 3 and list(nested["outer_split"]) == [s.id for s in outer]
    for chosen in nested["chosen"]:
        params = json.loads(chosen)
        assert set(params) == set(GRID) and all(params[name] in GRID[name] for name in GRID)
    assert nested["inner_score"].between(0, 1).all() and nested["test_macro_f1"].between(0, 1).all()
    assert (nested["test_accuracy"] > 0.8).all()
    assert selection_frequency(nested).sum() == 3


def test_the_outer_test_fold_never_influences_the_choice_and_runs_are_reproducible(iris):
    outer = make_splits(iris.y, n_splits=3, n_repeats=1, seed=2)[:1]
    altered_labels = iris.y.copy()
    altered_labels.iloc[outer[0].test] = iris.y.iloc[outer[0].test].to_numpy()[::-1]   # se rompe SOLO el test externo
    assert not altered_labels.equals(iris.y)
    original = nested_cv(iris, "dt", GRID, outer)
    altered = nested_cv(dataclasses.replace(iris, y=altered_labels), "dt", GRID, outer)
    assert original["chosen"][0] == altered["chosen"][0]
    assert original["inner_score"][0] == altered["inner_score"][0]           # la elección no vio el test
    assert original["test_accuracy"][0] != altered["test_accuracy"][0]       # y el resultado sí cambió
    again = nested_cv(iris, "dt", GRID, outer)
    assert again["chosen"][0] == original["chosen"][0] and again["test_accuracy"][0] == original["test_accuracy"][0]


def test_a_tie_in_the_inner_cv_goes_to_the_first_combination_of_the_grid(iris):
    outer = make_splits(iris.y, n_splits=3, n_repeats=1, seed=3)[:2]
    # max_depth None y 50 producen exactamente el mismo árbol (Iris no llega a 50 niveles): empate perfecto
    first = nested_cv(iris, "dt", {"max_depth": [None, 50]}, outer)
    second = nested_cv(iris, "dt", {"max_depth": [50, None]}, outer)
    assert {json.loads(c)["max_depth"] for c in first["chosen"]} == {None}
    assert {json.loads(c)["max_depth"] for c in second["chosen"]} == {50}


def test_nested_cv_rejects_an_empty_grid_a_parameter_without_values_and_a_bad_metric(iris, splits):
    for grid in ({}, {"max_depth": []}):
        with pytest.raises(ValueError, match="grid"):
            nested_cv(iris, "dt", grid, splits)
    with pytest.raises(ValueError, match="metric"):
        nested_cv(iris, "dt", {"max_depth": [2]}, splits, metric="kappa")


# --------------------------------------------------------------------- faltantes inyectados
def test_inject_missing_deletes_about_the_requested_fraction_and_leaves_the_original_intact(iris):
    out = inject_missing(iris, 0.3, seed=1)
    assert abs(out.X.isna().to_numpy().mean() - 0.3) < 0.06
    assert int(iris.X.isna().sum().sum()) == 0 and out.y.equals(iris.y)    # el original y la clase no se tocan
    assert out.X.shape == iris.X.shape and "faltantes" in out.name
    kept = out.X.notna().to_numpy()
    assert (out.X.to_numpy()[kept] == iris.X.to_numpy()[kept]).all()        # lo que no se borró queda igual


def test_inject_missing_is_reproducible_nested_and_validated(iris):
    a, b, c = (inject_missing(iris, 0.2, seed=s) for s in (5, 5, 6))
    assert a.X.isna().equals(b.X.isna()) and not a.X.isna().equals(c.X.isna())
    low, high = inject_missing(iris, 0.1, seed=5), inject_missing(iris, 0.3, seed=5)
    assert (low.X.isna().to_numpy() <= high.X.isna().to_numpy()).all()      # lo borrado al 10 % sigue borrado al 30 %
    assert inject_missing(iris, 0).X.equals(iris.X)
    only = inject_missing(iris, 0.5, seed=1, columns=["sepal_width"])
    assert only.X["sepal_width"].isna().any()
    assert only.X.drop(columns="sepal_width").equals(iris.X.drop(columns="sepal_width"))
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            inject_missing(iris, bad)
    with pytest.raises(ValueError, match="no existen"):
        inject_missing(iris, 0.1, columns=["inexistente"])


# --------------------------------------------------------------------- robustez a faltantes
def test_robustness_at_zero_matches_the_plain_experiment_and_degrades_with_more_missing(iris, splits):
    models = {**DT, "RF": ("rf", RF_SMALL)}
    robust = missing_robustness(iris, models, [0, 0.4], splits)
    plain = run_experiment(iris, models, splits, register_decision=False)
    zero = robust[robust["missing_fraction"] == 0]
    assert zero["test_accuracy"].tolist() == plain["test_accuracy"].tolist()
    table = robustness_table(robust)
    assert list(table.index) == [0, 0.4] and list(table.columns) == ["DT", "RF"]
    assert (table.loc[0.4] < table.loc[0]).all()                            # quitar información no puede ayudar
    assert robust["mask_seed"].unique().tolist() == [0]


def test_robustness_can_average_over_several_mask_seeds(iris, splits):
    robust = missing_robustness(iris, DT, [0.2], splits, seeds=(0, 1, 2))
    assert sorted(robust["mask_seed"].unique()) == [0, 1, 2] and len(robust) == 3 * len(splits)
    assert robust.groupby("mask_seed")["test_macro_f1"].mean().nunique() > 1   # máscaras distintas, resultados distintos
    assert robustness_table(robust).iloc[0, 0] == pytest.approx(robust["test_macro_f1"].mean())
