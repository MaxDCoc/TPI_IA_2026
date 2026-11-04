"""Tests de la evaluación: particiones, métricas, Random Forest y corredor. Todo sobre Iris."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from tp_ia.c45 import C45DecisionTree
from tp_ia.data.fixtures import play_tennis
from tp_ia.data.iris import load_iris_dataset
from tp_ia.eval import (RandomForestModel, confusion, environment_info, evaluate, make_model, make_splits,
                        max_stratified_splits, paired_comparison, run_experiment, save_results, summarize)
from tp_ia.eval.runner import _align_proba


@pytest.fixture(scope="module")
def iris():
    return load_iris_dataset()


# --------------------------------------------------------------------- particiones
def test_splits_cover_every_sample_once_per_repeat_and_are_stratified(iris):
    y = iris.y.to_numpy()
    splits = make_splits(y, n_splits=5, n_repeats=2, seed=1)
    assert len(splits) == 10
    for repeat in (0, 1):
        tests = np.concatenate([s.test for s in splits if s.repeat == repeat])
        assert sorted(tests.tolist()) == list(range(150))
    for s in splits:
        assert not set(s.train) & set(s.test)
        counts = pd.Series(y[s.test]).value_counts()
        assert counts.min() == counts.max() == 10          # 150 / 5 pliegues = 30 por pliegue, 10 por clase


def test_splits_are_deterministic_and_depend_on_the_seed(iris):
    y = iris.y.to_numpy()
    a, b, c = make_splits(y, seed=3), make_splits(y, seed=3), make_splits(y, seed=4)
    assert all(np.array_equal(x.test, z.test) for x, z in zip(a, b))
    assert not all(np.array_equal(x.test, z.test) for x, z in zip(a, c))


def test_splits_reject_more_folds_than_the_smallest_class():
    y = np.array([0] * 20 + [1] * 3)
    assert max_stratified_splits(y) == 3
    with pytest.raises(ValueError, match="clase más chica"):
        make_splits(y, n_splits=5)
    with pytest.raises(ValueError):
        make_splits(y, n_splits=1)


def test_group_splits_never_separate_a_group():
    y = np.array([0] * 50 + [1] * 30 + [2] * 20)
    groups = np.arange(100) // 2                       # pares de filas "repetidas"
    for s in make_splits(y, n_splits=5, n_repeats=2, groups=groups):
        assert not set(groups[s.train]) & set(groups[s.test])


def test_holdout_scheme_and_invalid_scheme(iris):
    y = iris.y.to_numpy()
    splits = make_splits(y, scheme="holdout", n_repeats=4, test_size=0.3)
    assert len(splits) == 4 and all(len(s.test) == 45 for s in splits)
    with pytest.raises(ValueError):
        make_splits(y, scheme="otro")
    with pytest.raises(ValueError):
        make_splits(y, scheme="holdout", groups=np.arange(150))


# --------------------------------------------------------------------- métricas
def test_evaluate_matches_hand_computed_multiclass_values():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    m = evaluate(y_true, y_pred, classes=["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert (m["precision_a"], m["precision_b"], m["precision_c"]) == pytest.approx((0.5, 2 / 3, 1.0))
    assert (m["recall_a"], m["recall_b"], m["recall_c"]) == pytest.approx((0.5, 1.0, 0.5))
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_binary_sensitivity_specificity_and_auc_match_sklearn():
    m = evaluate(np.array([0, 0, 0, 1, 1]), np.array([0, 0, 1, 1, 0]), classes=[0, 1])
    assert m["sensitivity"] == pytest.approx(0.5) and m["specificity"] == pytest.approx(2 / 3)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    p1 = rng.random(60)
    scored = evaluate(y, (p1 > 0.5).astype(int), np.c_[1 - p1, p1], [0, 1])
    assert scored["roc_auc"] == pytest.approx(roc_auc_score(y, p1))


def test_evaluate_survives_a_class_missing_from_the_fold():
    proba = np.array([[0.9, 0.1, 0.0], [0.4, 0.6, 0.0], [0.2, 0.8, 0.0], [0.1, 0.9, 0.0]])
    m = evaluate([0, 0, 1, 1], [0, 1, 1, 1], proba, [0, 1, 2])
    assert np.isfinite(m["roc_auc"]) and m["support_2"] == 0 and m["recall_2"] == 0


def test_confusion_rows_are_the_true_class():
    assert confusion(["a", "b", "b"], ["a", "a", "b"], ["a", "b"]).tolist() == [[1, 0], [1, 1]]


# --------------------------------------------------------------------- modelos
def test_random_forest_on_iris_with_missing_values(iris):
    X = iris.X.copy()
    X.iloc[::9, 1] = np.nan
    rf = RandomForestModel(n_estimators=30, random_state=0).fit(X, iris.y, feature_types=iris.feature_types)
    np.testing.assert_allclose(rf.predict_proba(X).sum(axis=1), 1.0)
    assert list(rf.classes_) == iris.classes
    assert (rf.predict(X) == iris.y.to_numpy()).mean() > 0.9
    assert set(rf.feature_importances_) == set(X.columns)
    assert sum(rf.feature_importances_.values()) == pytest.approx(1.0)


def test_random_forest_handles_nominal_attributes_and_unseen_values():
    X, y = play_tennis()
    X = X.copy()
    X.loc[[0, 3], "Humidity"] = np.nan
    rf = RandomForestModel(n_estimators=20, random_state=0).fit(
        X, y, feature_types={c: "nominal" for c in X.columns})
    new = X.head(3).copy()
    new["Outlook"] = "Foggy"                           # valor que no vio al entrenar
    np.testing.assert_allclose(rf.predict_proba(new).sum(axis=1), 1.0)
    assert set(rf.feature_importances_) == set(X.columns)


def test_make_model_and_params():
    dt = make_model("dt", max_depth=2)
    assert isinstance(dt, C45DecisionTree) and dt.max_depth == 2
    rf = make_model("rf", n_estimators=7)
    assert rf.get_params()["n_estimators"] == 7 and rf.set_params(max_depth=3) is rf
    with pytest.raises(ValueError):
        make_model("otro")
    with pytest.raises(ValueError):
        rf.set_params(no_existe=1)


# --------------------------------------------------------------------- corredor
MODELS = {"DT": ("dt", {}), "DT-prof2": ("dt", {"max_depth": 2}), "RF": ("rf", {"n_estimators": 25, "random_state": 0})}


@pytest.fixture(scope="module")
def results(iris):
    return run_experiment(iris, MODELS, make_splits(iris.y, n_splits=3, n_repeats=2, seed=0))


def test_run_experiment_returns_one_row_per_split_and_model(results):
    assert len(results) == 6 * 3
    for column in ("model", "split", "fit_time", "predict_time", "test_accuracy", "train_accuracy",
                   "test_macro_f1", "test_roc_auc", "confusion", "params"):
        assert column in results.columns
    assert results["test_accuracy"].between(0, 1).all()
    assert results.attrs["env"]["sklearn"]


def test_summarize_orders_models_and_gives_sensible_scores(results):
    table = summarize(results)
    assert list(table.index) == ["DT", "DT-prof2", "RF"]
    assert table.loc["DT", "accuracy_mean"] > 0.9 and table.loc["RF", "accuracy_mean"] > 0.9
    assert "accuracy_std" in table.columns and "fit_time_mean" in table.columns


def test_a_deeper_tree_never_fits_the_training_data_worse(results):
    train = results.groupby("model")["train_accuracy"].mean()
    assert train["DT"] >= train["DT-prof2"]


def test_runs_are_reproducible(iris):
    splits = make_splits(iris.y, n_splits=3, n_repeats=1, seed=5)
    a = run_experiment(iris, MODELS, splits)
    b = run_experiment(iris, MODELS, splits)
    assert a["test_accuracy"].tolist() == b["test_accuracy"].tolist()


def test_paired_comparison_counts_and_p_value(results):
    comparison = paired_comparison(results, "DT", "DT-prof2")
    assert comparison["n"] == 6
    assert comparison["wins_a"] + comparison["wins_b"] + comparison["ties"] == 6
    assert 0.0 <= comparison["p_wilcoxon"] <= 1.0
    same = paired_comparison(results, "DT", "DT")
    assert same["mean_diff"] == 0 and same["p_wilcoxon"] == 1.0


def test_align_proba_fills_absent_classes_with_zero():
    out = _align_proba(np.array([[0.7, 0.3], [0.2, 0.8]]), [0, 2], [0, 1, 2])
    assert out.tolist() == [[0.7, 0.0, 0.3], [0.2, 0.0, 0.8]]


def test_results_roundtrip_through_csv_and_environment_info(results, tmp_path):
    path = tmp_path / "resultados.csv"
    save_results(results, path)
    back = pd.read_csv(path)
    assert len(back) == len(results) and "test_accuracy" in back.columns
    assert {"python", "numpy", "pandas", "scipy", "sklearn"} <= set(environment_info())
