"""Tests de los exportadores del árbol: estructura anidada, DOT de Graphviz y traza de un caso."""
import json

import numpy as np
import pandas as pd
import pytest

from tp_ia.c45 import C45DecisionTree
from tp_ia.data import load_iris_dataset
from tp_ia.data.fixtures import play_tennis


def walk(node):
    yield node
    for child in node.get("children", []):
        yield from walk(child)


@pytest.fixture(scope="module")
def tennis():
    X, y = play_tennis()
    tree = C45DecisionTree().fit(X, y, feature_types={c: "nominal" for c in X.columns})
    return tree, X, y


@pytest.fixture(scope="module")
def iris():
    ds = load_iris_dataset()
    return C45DecisionTree().fit(ds.X, ds.y), ds


# --------------------------------------------------------------------- estructura anidada
def test_the_dict_reproduces_the_tree_in_the_catedra_slide_for_play_tennis(tennis):
    tree, _, _ = tennis
    root = tree.to_dict()["root"]
    assert (root["feature"], root["n"]) == ("Outlook", 14)
    assert root["entropy"] == pytest.approx(0.940, abs=5e-4)                      # lámina 9: entropía 0.940
    assert root["gain"] == pytest.approx(0.247, abs=5e-4)                         # ganancia 0.247
    assert root["gain_ratio"] == pytest.approx(0.156, abs=5e-4)                   # razón de ganancia 0.156
    assert root["edges"] == ["Overcast", "Rain", "Sunny"]
    overcast, rain, sunny = root["children"]
    assert overcast["leaf"] and (overcast["label"], overcast["n"]) == ("Yes", 4)
    assert (rain["feature"], rain["n"], rain["edges"]) == ("Wind", 5, ["Strong", "Weak"])
    assert (sunny["feature"], sunny["n"], sunny["edges"]) == ("Humidity", 5, ["High", "Normal"])
    for inner in (rain, sunny):                                                   # entropía 0.971, ganancia 0.971, razón 1.000
        assert inner["entropy"] == pytest.approx(0.971, abs=5e-4)
        assert inner["gain"] == pytest.approx(0.971, abs=5e-4)
        assert inner["gain_ratio"] == pytest.approx(1.0, abs=5e-4)
    assert [(c["label"], c["n"]) for c in rain["children"]] == [("No", 2), ("Yes", 3)]
    assert [(c["label"], c["n"]) for c in sunny["children"]] == [("No", 3), ("Yes", 2)]


def test_ids_are_preorder_and_everything_is_json_safe(tennis):
    tree, _, _ = tennis
    data = tree.to_dict()
    json.dumps(data)                                                              # sin tipos de numpy
    nodes = list(walk(data["root"]))
    assert [n["id"] for n in nodes] == list(range(len(nodes))) and data["n_nodes"] == len(nodes)
    for node in nodes:
        if not node["leaf"]:
            assert node["children"][0]["id"] == node["id"] + 1                    # el primer hijo va pegado al padre
    assert data["n_leaves"] == sum(n["leaf"] for n in nodes) and data["depth"] == tree.depth_


def test_counts_probabilities_and_weights_are_consistent(tennis):
    tree, _, _ = tennis
    for node in walk(tree.to_dict()["root"]):
        assert sum(node["counts"]) == pytest.approx(node["n"])
        assert sum(node["proba"]) == pytest.approx(1.0)
        if not node["leaf"]:
            assert sum(c["n"] for c in node["children"]) == pytest.approx(node["n"])
            assert sum(node["branch_probs"]) == pytest.approx(1.0)


def test_numeric_edges_class_names_and_pruning_totals_on_iris(iris):
    full, ds = iris
    named = full.to_dict(class_names={"setosa": "Setosa", "virginica": "Virginica"})
    assert [c["name"] for c in named["classes"]] == ["Setosa", "versicolor", "Virginica"]
    assert [c["value"] for c in named["classes"]] == ["setosa", "versicolor", "virginica"]
    assert named["root"]["kind"] == "numeric" and named["root"]["edges"] == ["≤ 1.9", "> 1.9"]
    assert named["root"]["children"][0]["label"] == "Setosa"
    pruned = C45DecisionTree(prune=True).fit(ds.X, ds.y).to_dict()
    assert (pruned["n_nodes"], pruned["n_nodes_unpruned"], pruned["n_pruned"]) == (9, 13, 2)


def test_exporting_an_unfitted_tree_fails_clearly():
    with pytest.raises(RuntimeError):
        C45DecisionTree().to_dict()


# --------------------------------------------------------------------- DOT de Graphviz
def test_dot_follows_the_slide_format(tennis):
    tree, _, _ = tennis
    dot = tree.to_dot()
    nodes = list(walk(tree.to_dict()["root"]))
    assert dot.startswith("digraph Arbol {") and dot.endswith("}")
    assert dot.count(" -> ") == len(nodes) - 1
    for text in ("Outlook\\nentropía = 0.940\\nganancia = 0.247\\nrazón ganancia = 0.156\\nn = 14",
                 'label="Overcast"', 'label="Sunny"', "Yes\\nn = 4"):
        assert text in dot
    assert dot.count('fillcolor="#FFF2AE"') == sum(not n["leaf"] for n in nodes)    # nodos internos amarillos
    yes_color = [line for line in dot.splitlines() if 'label="Yes\\nn = 4"' in line][0]
    no_color = [line for line in dot.splitlines() if 'label="No\\nn = 2"' in line][0]
    assert yes_color.split('fillcolor="')[1] != no_color.split('fillcolor="')[1]   # una clase, un color


def test_dot_uses_display_names_and_escapes_special_characters():
    X = pd.DataFrame({'a"b': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "c": range(8)})
    y = ["no"] * 4 + ["si"] * 4
    dot = C45DecisionTree().fit(X, y).to_dot(class_names={"no": "Sin enfermedad", "si": "Con enfermedad"})
    assert 'a\\"b' in dot and "Sin enfermedad" in dot and "Con enfermedad" in dot


# --------------------------------------------------------------------- traza de un caso
def leaf_probabilities(tree):
    return {n["id"]: np.array(n["proba"]) for n in walk(tree.to_dict()["root"]) if n["leaf"]}


def test_a_complete_case_follows_a_single_path(tennis):
    tree, _, _ = tennis
    weights = tree.trace({"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"})
    nodes = {n["id"]: n for n in walk(tree.to_dict()["root"])}
    visited = {i for i, w in weights.items() if w > 0}
    assert all(weights[i] == 1.0 for i in visited) and len(weights) == len(nodes)
    leaf = [nodes[i] for i in visited if nodes[i]["leaf"]]
    assert len(leaf) == 1 and leaf[0]["label"] == "No"                              # soleado + humedad alta -> no juega
    assert sum(weights[i] for i, n in nodes.items() if n["leaf"]) == pytest.approx(1.0)


def test_a_missing_or_unseen_value_splits_the_case_by_branch_probabilities(tennis):
    tree, _, _ = tennis
    root = tree.to_dict()["root"]
    expected = [c["id"] for c in root["children"]], root["branch_probs"]
    for outlook in (None, np.nan, "Foggy"):                                          # faltante y nominal no visto
        weights = tree.trace({"Outlook": outlook, "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"})
        assert [weights[i] for i in expected[0]] == pytest.approx(expected[1])
        assert weights[0] == 1.0


def test_the_trace_is_consistent_with_predict_proba(tennis, iris):
    tree, X, _ = tennis
    leaves = leaf_probabilities(tree)
    rows = [dict(r) for _, r in X.iterrows()]
    rows += [{**rows[0], "Outlook": None}, {**rows[3], "Wind": None, "Humidity": None}, {**rows[5], "Outlook": "Foggy"}]
    for row in rows:
        weights = tree.trace(row)
        mixed = sum(weights[i] * proba for i, proba in leaves.items())
        np.testing.assert_allclose(mixed, tree.predict_proba(pd.DataFrame([row]))[0], atol=1e-12)
    full, ds = iris                                                                  # y lo mismo con atributos numéricos
    leaves = leaf_probabilities(full)
    for _, r in ds.X.sample(25, random_state=0).iterrows():
        for broken in ({}, {"petal_length": np.nan}, {"petal_width": np.nan, "sepal_width": np.nan}):
            row = {**r.to_dict(), **broken}
            mixed = sum(full.trace(row)[i] * proba for i, proba in leaves.items())
            np.testing.assert_allclose(mixed, full.predict_proba(pd.DataFrame([row]))[0], atol=1e-12)
