"""Tests de la poda por error pesimista de C4.5: fórmulas, ejemplo de Quinlan, árboles armados a mano y datos reales."""
import numpy as np
import pandas as pd
import pytest

from tp_ia.c45 import (C45DecisionTree, Node, added_errors, confidence_coefficient, estimated_errors,
                       prune_tree)
from tp_ia.data import load_iris_dataset
from tp_ia.data.fixtures import liu_loans, play_tennis
from tp_ia.eval import make_model, make_splits, parameter_sweep, run_experiment


# --------------------------------------------------------------------- árboles armados a mano
def leaf(*counts):
    c = np.array(counts, dtype=float)
    return Node(n_weight=float(c.sum()), class_counts=c, depth=1)


def split(children, depth=0):
    counts = sum(child.class_counts for child in children)
    return Node(n_weight=float(counts.sum()), class_counts=counts, depth=depth, feature=0, kind="nominal",
                values=list(range(len(children))), children=list(children),
                branch_probs=np.full(len(children), 1.0 / len(children)))


def quinlan_example():
    """16 casos: tres hojas (6, 0 errores), (9, 0 errores) y (1 caso de la otra clase)."""
    return split([leaf(6, 0), leaf(9, 0), leaf(0, 1)])


# --------------------------------------------------------------------- fórmulas
def test_the_coefficient_interpolates_quinlans_table():
    assert confidence_coefficient(0.25) == pytest.approx(0.6925 ** 2)     # entre 0.20 (0.84) y 0.40 (0.25)
    assert confidence_coefficient(0.10) == pytest.approx(1.28 ** 2)       # puntos de la tabla
    assert confidence_coefficient(0.05) == pytest.approx(1.65 ** 2)
    assert confidence_coefficient(1.0) == 0.0
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            confidence_coefficient(bad)


def test_pessimistic_error_rates_match_the_values_in_quinlans_book():
    # U_25%(E, N): 0.206 está confirmado en una cita del libro; los otros tres son de su mismo ejemplo (de memoria)
    for (n, e), book in {(6, 0): 0.206, (9, 0): 0.143, (1, 0): 0.750, (16, 1): 0.157}.items():
        assert estimated_errors(n, e) / n == pytest.approx(book, abs=5e-4)


def test_added_errors_branches():
    cf = 0.25
    for n in (1, 6, 40):
        assert added_errors(n, 0, cf) == pytest.approx(n * (1 - cf ** (1 / n)))        # sin errores: binomial exacto
    for n in (5, 30):                                                                   # errores fraccionarios: interpolación
        assert added_errors(n, 0.5, cf) == pytest.approx((added_errors(n, 0, cf) + added_errors(n, 1, cf)) / 2)
    assert added_errors(2, 1.6, cf) == pytest.approx(0.67 * 0.4)                        # caso de borde e + 0.5 >= N
    assert added_errors(0, 0, cf) == 0.0 and added_errors(-1, 0, cf) == 0.0
    assert added_errors(10, 0.99989, cf) == pytest.approx(added_errors(10, 1.0, cf), abs=1e-3)   # continuidad


def test_added_errors_behaves_like_an_upper_confidence_limit():
    assert all(added_errors(n, e) > 0 for n in (3, 10, 50) for e in (0, 0.5, 1, 2))
    # más confianza exigida (CF más chico) -> límite superior más alto
    assert added_errors(20, 2, 0.05) > added_errors(20, 2, 0.25) > added_errors(20, 2, 0.5)
    # el mismo porcentaje de error con más casos se estima más ajustado
    assert added_errors(10, 0) / 10 > added_errors(100, 0) / 100
    assert added_errors(20, 1) / 20 < added_errors(20, 4) / 20 < added_errors(20, 8) / 20


# --------------------------------------------------------------------- poda sobre nodos armados a mano
def test_quinlans_example_replaces_the_three_leaf_subtree_by_a_single_leaf():
    root = quinlan_example()
    subtree = sum(estimated_errors(c.n_weight, c.errors) for c in root.children)
    as_leaf = estimated_errors(root.n_weight, root.errors)
    assert subtree == pytest.approx(3.273, abs=2e-3)                      # 6*0.206 + 9*0.143 + 1*0.750
    assert as_leaf == pytest.approx(2.512, abs=1e-2)                      # 16 * 0.157
    assert prune_tree(root, 0.25) == 1
    assert root.is_leaf and root.feature is None and root.threshold is None and root.branch_probs is None
    assert root.proba == pytest.approx([15 / 16, 1 / 16])                 # conserva la distribución de clases


def test_a_clean_split_is_kept():
    root = split([leaf(50, 0), leaf(0, 50)])
    assert prune_tree(root) == 0 and not root.is_leaf and len(root.children) == 2


def test_pruning_is_bottom_up_and_uses_the_pruned_estimate_of_the_children():
    inner = split([leaf(0, 25), leaf(0, 25)], depth=1)        # dos hojas puras de la misma clase: sobra el corte
    root = split([leaf(50, 0), inner])
    assert prune_tree(root) == 1
    assert inner.is_leaf and not root.is_leaf and root.children[1] is inner   # cae el de abajo, la raíz se queda


def test_tolerance_and_confidence_arguments():
    never = quinlan_example()
    assert prune_tree(never, 0.25, tolerance=-1e9) == 0 and not never.is_leaf     # exigir una mejora enorme: nada se poda
    everything = split([leaf(50, 0), leaf(0, 50)])
    assert prune_tree(everything, 0.25, tolerance=1e9) == 1 and everything.is_leaf
    for bad in (0, 0.6, -0.2):
        with pytest.raises(ValueError):
            prune_tree(quinlan_example(), bad)


# --------------------------------------------------------------------- el árbol completo
def test_prune_is_off_by_default_and_the_parameters_are_validated():
    tree = C45DecisionTree()
    assert tree.prune is False and tree.confidence == 0.25
    assert {"prune", "confidence"} <= set(tree.get_params())
    assert tree.set_params(prune=True, confidence=0.1) is tree and tree.prune is True
    X, y = play_tennis()
    plain = C45DecisionTree().fit(X, y)
    assert plain.n_pruned_ == 0 and plain.n_nodes_unpruned_ == plain.n_nodes_
    for kwargs in ({"confidence": 0}, {"confidence": -0.1}, {"confidence": 0.6}, {"prune": "si"}):
        with pytest.raises(ValueError):
            C45DecisionTree(**kwargs).fit(X, y)


def test_iris_pruned_tree_has_the_leaves_of_the_pruned_tree_reported_by_j48():
    iris = load_iris_dataset()
    full = C45DecisionTree().fit(iris.X, iris.y)
    pruned = C45DecisionTree(prune=True).fit(iris.X, iris.y)
    assert full.n_nodes_ == 13 and pruned.n_nodes_ == 9 and pruned.n_leaves_ == 5
    assert pruned.n_nodes_unpruned_ == 13 and pruned.n_pruned_ == 2
    leaves = sorted((round(n.n_weight), round(n.errors)) for n in pruned._iter_nodes() if n.is_leaf)
    # J48 (Weka) reporta para Iris las hojas 50, 48/1, 3, 3/1 y 46/1; solo cambia qué atributo separa a setosa
    # (petal_width <= 0.6 allá, petal_length <= 1.9 acá: empate de gain ratio que se resuelve por menor índice)
    assert leaves == sorted([(50, 0), (48, 1), (3, 0), (3, 1), (46, 1)])
    assert pruned.score(iris.X, iris.y) == pytest.approx(full.score(iris.X, iris.y))   # no pierde nada en train


def test_clean_toy_trees_are_not_pruned():
    X, y = play_tennis()
    types = {c: "nominal" for c in X.columns}
    a, b = C45DecisionTree().fit(X, y, feature_types=types), C45DecisionTree(prune=True).fit(X, y, feature_types=types)
    assert b.n_pruned_ == 0 and a.export_text() == b.export_text()
    X, y = liu_loans()
    a, b = C45DecisionTree().fit(X, y), C45DecisionTree(prune=True).fit(X, y)
    assert b.n_pruned_ == 0 and a.export_text() == b.export_text()


def noisy_data(n, noise, seed):
    """y = (x0 > 0) con una fracción `noise` de etiquetas invertidas, más 4 atributos que no sirven."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    clean = (X["x0"] > 0).astype(int).to_numpy()
    return X, np.where(rng.random(n) < noise, 1 - clean, clean), clean


def test_pruning_removes_the_noise_a_full_tree_memorizes():
    X, y, _ = noisy_data(300, 0.2, seed=0)
    X_test, _, y_test = noisy_data(2000, 0.0, seed=99)                      # prueba sin ruido: la regla verdadera
    full = C45DecisionTree().fit(X, y)
    pruned = C45DecisionTree(prune=True, confidence=0.1).fit(X, y)
    assert full.n_nodes_ > 50 and pruned.n_nodes_ <= 5
    assert pruned.feature_names_[pruned.tree_.feature] == "x0"              # lo único que queda es el corte que sirve
    assert pruned.feature_importances_[0] == pytest.approx(1.0)
    assert pruned.score(X, y) <= full.score(X, y)                           # en train rinde menos...
    assert pruned.score(X_test, y_test) > full.score(X_test, y_test) + 0.15  # ...y generaliza mucho mejor
    np.testing.assert_allclose(pruned.predict_proba(X_test).sum(axis=1), 1.0)


def test_a_smaller_confidence_prunes_more():
    X, y, _ = noisy_data(300, 0.2, seed=0)
    sizes = [C45DecisionTree(prune=True, confidence=cf).fit(X, y).n_nodes_ for cf in (0.01, 0.1, 0.25, 0.5)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1] <= C45DecisionTree().fit(X, y).n_nodes_


def test_pruning_works_with_missing_values_and_is_deterministic():
    X, y, _ = noisy_data(300, 0.2, seed=1)
    X = X.mask(np.random.default_rng(5).random(X.shape) < 0.1)             # 10 % de faltantes
    a = C45DecisionTree(prune=True, confidence=0.1).fit(X, y)
    b = C45DecisionTree(prune=True, confidence=0.1).fit(X, y)
    assert a.export_text() == b.export_text()
    assert a.n_nodes_ <= C45DecisionTree().fit(X, y).n_nodes_ and a.n_pruned_ > 0
    np.testing.assert_allclose(a.predict_proba(X).sum(axis=1), 1.0)
    X_clean, _, y_clean = noisy_data(1000, 0.0, seed=7)
    assert a.score(X_clean, y_clean) > 0.85


# --------------------------------------------------------------------- con el resto del sistema
def test_pruning_runs_through_the_evaluation_pipeline_and_the_parameter_sweep():
    iris = load_iris_dataset()
    splits = make_splits(iris.y, n_splits=3, n_repeats=1, seed=0)
    assert make_model("dt", prune=True, confidence=0.1).prune is True
    results = run_experiment(iris, {"DT": ("dt", {}), "DT-poda": ("dt", {"prune": True})}, splits,
                             register_decision=False)
    means = results.groupby("model")["test_accuracy"].mean()
    assert means["DT-poda"] > 0.9
    sweep = parameter_sweep(iris, "dt", "confidence", [0.05, 0.25, 0.5], splits, base_params={"prune": True})
    assert list(sweep["value"]) == [0.05, 0.25, 0.5] and (sweep["test_macro_f1_mean"] > 0.85).all()
