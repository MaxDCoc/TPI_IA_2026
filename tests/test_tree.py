"""Tests del árbol C4.5: estructura esperada, propiedades, faltantes, parámetros y validaciones.

Iris se usa solo como DATOS de prueba (sklearn.datasets); el árbol que se prueba es el nuestro.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from tp_ia.c45 import C45DecisionTree
from tp_ia.data.fixtures import liu_loans, play_tennis


@pytest.fixture(scope="module")
def iris():
    data = load_iris(as_frame=True)
    return data.data, data.target_names[data.target.to_numpy()]


def leaf_label(tree, node):
    return tree.classes_[int(np.argmax(node.class_counts))]


def child_by_value(node, value):
    return dict(zip(node.values, node.children))[value]


# --------------------------------------------------------------------- estructura esperada
def test_play_tennis_matches_textbook_tree():
    X, y = play_tennis()
    tree = C45DecisionTree().fit(X, y)
    root = tree.tree_
    assert tree.feature_names_[root.feature] == "Outlook"

    overcast = child_by_value(root, "Overcast")
    assert overcast.is_leaf and leaf_label(tree, overcast) == "Yes"

    sunny = child_by_value(root, "Sunny")
    assert tree.feature_names_[sunny.feature] == "Humidity"
    assert leaf_label(tree, child_by_value(sunny, "High")) == "No"
    assert leaf_label(tree, child_by_value(sunny, "Normal")) == "Yes"

    rain = child_by_value(root, "Rain")
    assert tree.feature_names_[rain.feature] == "Wind"
    assert leaf_label(tree, child_by_value(rain, "Strong")) == "No"
    assert leaf_label(tree, child_by_value(rain, "Weak")) == "Yes"

    assert tree.score(X, y) == 1.0
    assert tree.n_leaves_ == 5 and tree.depth_ == 2


def test_liu_loans_tree_matches_book():
    X, y = liu_loans()
    tree = C45DecisionTree().fit(X, y)
    root = tree.tree_
    assert tree.feature_names_[root.feature] == "Own_house"
    assert leaf_label(tree, child_by_value(root, "true")) == "Yes"
    no_house = child_by_value(root, "false")
    assert tree.feature_names_[no_house.feature] == "Has_job"
    assert leaf_label(tree, child_by_value(no_house, "true")) == "Yes"
    assert leaf_label(tree, child_by_value(no_house, "false")) == "No"
    assert tree.score(X, y) == 1.0


def test_iris_root_isolates_setosa(iris):
    X, y = iris
    tree = C45DecisionTree().fit(X, y)
    root = tree.tree_
    assert root.kind == "numeric"
    assert tree.feature_names_[root.feature] == "petal length (cm)"   # empate con petal width: gana el de menor índice
    assert root.threshold == pytest.approx(1.9)                      # modo "c45": valor observado <= punto medio
    assert root.children[0].class_counts.tolist() == [50, 0, 0]      # rama "<=": solo setosa
    assert root.children[1].class_counts.tolist() == [0, 50, 50]


def test_iris_training_accuracy_is_high(iris):
    X, y = iris
    assert C45DecisionTree().fit(X, y).score(X, y) >= 0.97


# --------------------------------------------------------------------- propiedades
def test_predict_proba_rows_sum_to_one(iris):
    X, y = iris
    proba = C45DecisionTree().fit(X, y).predict_proba(X)
    assert proba.shape == (150, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_fit_is_deterministic_and_invariant_to_row_order(iris):
    X, y = iris
    a = C45DecisionTree().fit(X, y)
    b = C45DecisionTree().fit(X, y)
    perm = np.random.default_rng(0).permutation(len(X))
    c = C45DecisionTree().fit(X.iloc[perm], y[perm])
    assert a.export_text() == b.export_text() == c.export_text()


def test_sample_weight_equals_duplicated_rows():
    X, y = play_tennis()
    weights = np.ones(len(X))
    weights[[0, 5]] = 2
    weighted = C45DecisionTree().fit(X, y, sample_weight=weights)
    duplicated = C45DecisionTree().fit(pd.concat([X, X.iloc[[0, 5]]], ignore_index=True),
                                       pd.concat([y, y.iloc[[0, 5]]], ignore_index=True))
    assert weighted.export_text() == duplicated.export_text()
    np.testing.assert_allclose(weighted.predict_proba(X), duplicated.predict_proba(X))


def test_sample_weight_equals_duplicated_rows_numeric(iris):
    X, y = iris
    weights = np.ones(len(X))
    weights[:40] = 2
    weighted = C45DecisionTree().fit(X, y, sample_weight=weights)
    dup_idx = list(range(len(X))) + list(range(40))
    duplicated = C45DecisionTree().fit(X.iloc[dup_idx], y[dup_idx])
    assert weighted.export_text() == duplicated.export_text()


def test_feature_importances_are_a_distribution(iris):
    X, y = iris
    tree = C45DecisionTree().fit(X, y)
    assert len(tree.feature_importances_) == X.shape[1]
    assert tree.feature_importances_.sum() == pytest.approx(1.0)
    assert (tree.feature_importances_ >= 0).all()


# --------------------------------------------------------------------- parámetros
def test_max_depth_and_min_samples_leaf_limit_the_tree(iris):
    X, y = iris
    stump = C45DecisionTree(max_depth=1).fit(X, y)
    assert stump.depth_ == 1 and stump.n_leaves_ == 2
    assert C45DecisionTree(max_depth=0).fit(X, y).n_nodes_ == 1
    assert C45DecisionTree(min_samples_leaf=80).fit(X, y).n_nodes_ == 1   # ningún corte deja 80+80 casos
    assert C45DecisionTree(min_gain=5.0).fit(X, y).n_nodes_ == 1          # ninguna ganancia llega a 5 bits


def test_more_restrictive_params_never_grow_the_tree(iris):
    X, y = iris
    sizes = [C45DecisionTree(min_samples_leaf=m).fit(X, y).n_nodes_ for m in (1, 2, 5, 10, 30)]
    assert sizes == sorted(sizes, reverse=True)


def test_criterion_gain_and_threshold_modes(iris):
    X, y = iris
    pure_gain = C45DecisionTree(criterion="gain").fit(*play_tennis())
    assert pure_gain.feature_names_[pure_gain.tree_.feature] == "Outlook"

    midpoint = C45DecisionTree(threshold_mode="midpoint").fit(X, y)
    assert midpoint.tree_.threshold == pytest.approx(2.45)               # (1.9 + 3.0) / 2
    corrected = C45DecisionTree(numeric_gain_correction=True).fit(X, y)
    assert corrected.score(X, y) >= 0.95                                  # la corrección no rompe el árbol


def test_get_and_set_params():
    tree = C45DecisionTree(max_depth=3)
    assert tree.get_params()["max_depth"] == 3
    assert tree.set_params(min_samples_leaf=5) is tree and tree.min_samples_leaf == 5
    with pytest.raises(ValueError):
        tree.set_params(no_existe=1)


# --------------------------------------------------------------------- faltantes y valores nuevos
def test_missing_values_in_training_conserve_weight_and_still_classify():
    X, y = play_tennis()
    X = X.copy()
    X.loc[[0, 5, 8], "Humidity"] = np.nan
    X.loc[[2], "Outlook"] = None
    tree = C45DecisionTree().fit(X, y)
    leaves = [n for n in tree._iter_nodes() if n.is_leaf]
    assert sum(n.n_weight for n in leaves) == pytest.approx(14.0)         # los pesos fraccionarios suman 14
    proba = tree.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    # Con 4 valores faltantes el árbol ya no es perfecto, pero debe superar al clasificador trivial
    # que siempre responde la clase mayoritaria (9 Yes de 14 = 0.643).
    assert tree.score(X, y) > 9 / 14


def test_missing_values_in_numeric_attributes(iris):
    X, y = iris
    X = X.copy()
    X.iloc[::7, 2] = np.nan          # faltantes en petal length (el atributo de la raíz)
    X.iloc[3::11, 0] = np.nan
    tree = C45DecisionTree().fit(X, y)
    leaves = [n for n in tree._iter_nodes() if n.is_leaf]
    assert sum(n.n_weight for n in leaves) == pytest.approx(150.0)
    proba = tree.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert tree.score(X, y) > 0.9
    all_missing = pd.DataFrame([{c: np.nan for c in X.columns}])
    assert tree.predict_proba(all_missing)[0].sum() == pytest.approx(1.0)


def test_missing_values_at_prediction_mix_the_branches():
    X, y = play_tennis()
    tree = C45DecisionTree().fit(X, y)
    row = pd.DataFrame([{"Outlook": None, "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}])
    proba = tree.predict_proba(row)[0]
    assert proba.sum() == pytest.approx(1.0)
    assert 0 < proba[list(tree.classes_).index("Yes")] < 1                # mezcla de ramas, no una certeza


def test_unseen_category_is_treated_as_missing():
    X, y = play_tennis()
    tree = C45DecisionTree().fit(X, y)
    row = pd.DataFrame([{"Outlook": "Foggy", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}])
    proba = tree.predict_proba(row)[0]
    assert proba.sum() == pytest.approx(1.0) and (proba >= 0).all()


# --------------------------------------------------------------------- datos mixtos
def test_numeric_threshold_is_found_on_mixed_data():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"x": rng.uniform(0, 1, n), "color": rng.choice(["red", "blue", "green"], n)})
    y = np.where(X["x"] > 0.5, "A", "B")
    tree = C45DecisionTree().fit(X, y)
    assert tree.score(X, y) == 1.0
    assert tree.feature_names_[tree.tree_.feature] == "x"
    assert 0.4 < tree.tree_.threshold < 0.6


def test_nominal_split_is_multiway_on_mixed_data():
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"x": rng.uniform(0, 1, n), "color": rng.choice(["red", "blue", "green"], n)})
    y = X["color"].map({"red": "R", "blue": "B", "green": "G"}).to_numpy()
    tree = C45DecisionTree().fit(X, y)
    assert tree.score(X, y) == 1.0
    assert tree.tree_.kind == "nominal" and len(tree.tree_.children) == 3


def test_integer_coded_category_can_be_declared_nominal():
    # El código 1/2/3 no tiene orden: como numérico habría que cortar "x <= 1", "x <= 2"...
    X = pd.DataFrame({"code": [1, 2, 3] * 20})
    y = np.where(X["code"] == 2, "mid", "edge")
    as_nominal = C45DecisionTree().fit(X, y, feature_types={"code": "nominal"})
    assert as_nominal.tree_.kind == "nominal" and as_nominal.score(X, y) == 1.0
    as_numeric = C45DecisionTree().fit(X, y)
    assert as_numeric.tree_.kind == "numeric" and as_numeric.depth_ == 2   # necesita dos cortes


# --------------------------------------------------------------------- texto y explicación
def test_export_text_and_explain_on_play_tennis():
    X, y = play_tennis()
    tree = C45DecisionTree().fit(X, y)
    text = tree.export_text()
    assert "Outlook = Overcast: Yes (4)" in text
    assert "|   Humidity = High: No (3)" in text
    info = tree.explain({"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"})
    assert info["label"] == "No"
    assert info["rules"] == ["Outlook = Sunny", "Humidity = High"]
    assert info["proba"]["No"] == pytest.approx(1.0)


# --------------------------------------------------------------------- validaciones
def test_input_validation():
    X, y = play_tennis()
    with pytest.raises(ValueError):
        C45DecisionTree().fit(pd.DataFrame({"a": []}), [])                       # datos vacíos
    with pytest.raises(ValueError):
        C45DecisionTree().fit(X, y.iloc[:-1])                                    # largos distintos
    with pytest.raises(ValueError):
        C45DecisionTree().fit(X, y.where(y.index != 0, None))                    # y con faltantes
    with pytest.raises(ValueError):
        C45DecisionTree().fit(X, y, sample_weight=-np.ones(len(X)))              # pesos negativos
    with pytest.raises(ValueError):
        C45DecisionTree().fit(X, y, feature_types={"Outlook": "ordinal"})        # tipo inválido
    with pytest.raises(ValueError):
        C45DecisionTree(criterion="otro").fit(X, y)                              # parámetro inválido
    with pytest.raises(RuntimeError):
        C45DecisionTree().predict(X)                                             # sin entrenar
    tree = C45DecisionTree().fit(X, y)
    with pytest.raises(ValueError):
        tree.predict(X.drop(columns="Wind"))                                     # falta una columna


def test_float_coded_categories_are_shown_as_integers():
    # Una columna de códigos con NaN llega como float (3.0, 7.0): el texto debe decir "thal = 3".
    X = pd.DataFrame({"thal": [3.0, 7.0, 3.0, 7.0, np.nan, 3.0, 7.0, 3.0]})
    y = np.where(X["thal"] == 7.0, "sick", "ok")
    tree = C45DecisionTree().fit(X, y, feature_types={"thal": "nominal"})
    text = tree.export_text()
    assert "thal = 3" in text and "3.0" not in text
    assert tree.predict(pd.DataFrame({"thal": [3.0, 7.0]})).tolist() == ["ok", "sick"]


def test_single_class_gives_a_single_leaf():
    X, _ = play_tennis()
    tree = C45DecisionTree().fit(X, ["Yes"] * len(X))
    assert tree.n_nodes_ == 1 and set(tree.predict(X)) == {"Yes"}
