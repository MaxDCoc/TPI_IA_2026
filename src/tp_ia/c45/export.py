"""
Exportadores del árbol: estructura anidada (para la interfaz web), DOT de Graphviz y traza de un caso.

GUÍA DE LECTURA
---------------
  - tree_to_dict(): el árbol como diccionario anidado con solo tipos de Python (se puede pasar a json.dumps).
    Cada nodo lleva un `id` en PREORDEN (raíz = 0, luego el subárbol del primer hijo, etc.): la interfaz y
    trace_case() usan ese mismo id para hablar del mismo nodo.
  - tree_to_dot(): el árbol en el lenguaje de Graphviz con el formato de las láminas de la cátedra: nodos
    internos en amarillo con entropía, ganancia, razón de ganancia y n; hojas coloreadas por clase.
  - trace_case(): por qué nodos pasa UN caso y con qué peso. Si un valor falta (o es un nominal que el árbol
    no vio), el caso se reparte entre las ramas según branch_probs, igual que en predict_proba.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .criteria import entropy

LEAF_COLORS = ("#9ECAE1", "#B7E2A3", "#F4B8A0", "#D6C2E8", "#F2D78C", "#BFE3DC")   # una por clase, en orden
INTERNAL_COLOR = "#FFF2AE"


def _py(value):
    """Escalar de numpy -> escalar de Python (para que json.dumps no falle)."""
    return value.item() if hasattr(value, "item") else value


def _preorder(root) -> list:
    nodes, stack = [], [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(reversed(node.children))      # el primer hijo se visita primero
    return nodes


def _class_names(tree, class_names) -> list:
    names = class_names or {}
    return [str(names.get(_py(c), _py(c))) for c in tree.classes_]


def _edge_labels(node) -> list:
    if node.kind == "numeric":
        return [f"≤ {node.threshold:g}", f"> {node.threshold:g}"]
    return [str(_py(v)) for v in node.values]


def tree_to_dict(tree, class_names=None) -> dict:
    """El árbol entrenado como estructura anidada JSON-segura. `class_names` = {clase: nombre para mostrar}."""
    tree._check_fitted()
    names = _class_names(tree, class_names)
    ids = {id(node): i for i, node in enumerate(_preorder(tree.tree_))}

    def build(node) -> dict:
        total = float(node.class_counts.sum())
        out = {
            "id": ids[id(node)], "depth": int(node.depth), "n": float(node.n_weight), "leaf": node.is_leaf,
            "counts": [float(c) for c in node.class_counts],
            "proba": [float(p) for p in node.proba],
            "label": names[int(np.argmax(node.class_counts))],
            "entropy": float(entropy(node.class_counts)) if total > 0 else 0.0,
            "errors": float(node.errors),
        }
        if not node.is_leaf:
            out.update({
                "feature": tree.feature_names_[node.feature], "feature_index": int(node.feature), "kind": node.kind,
                "threshold": None if node.threshold is None else float(node.threshold),
                "edges": _edge_labels(node),
                "branch_probs": [float(p) for p in node.branch_probs],
                "gain": float(node.gain), "gain_ratio": float(node.gain_ratio),
                "children": [build(child) for child in node.children],
            })
        return out

    return {"classes": [{"value": _py(c), "name": n} for c, n in zip(tree.classes_, names)],
            "root": build(tree.tree_), "n_nodes": int(tree.n_nodes_), "n_leaves": int(tree.n_leaves_),
            "depth": int(tree.depth_), "n_pruned": int(getattr(tree, "n_pruned_", 0)),
            "n_nodes_unpruned": int(getattr(tree, "n_nodes_unpruned_", tree.n_nodes_))}


def _dot_escape(text) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def tree_to_dot(tree, class_names=None) -> str:
    """El árbol en formato DOT (Graphviz), con el estilo de las láminas de la cátedra."""
    data = tree_to_dict(tree, class_names)
    names = [c["name"] for c in data["classes"]]
    lines = ["digraph Arbol {",
             '  graph [rankdir=TB, fontname="Helvetica"];',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", color="#222222"];',
             '  edge [fontname="Helvetica", fontsize=10];']
    pending = [data["root"]]
    while pending:
        node = pending.pop(0)
        if node["leaf"]:
            label = f"{node['label']}\nn = {node['n']:g}"
            if node["errors"] > 1e-9:
                label += f"\nerrores = {node['errors']:g}"
            color = LEAF_COLORS[names.index(node["label"]) % len(LEAF_COLORS)]
        else:
            label = (f"{node['feature']}\nentropía = {node['entropy']:.3f}\nganancia = {node['gain']:.3f}\n"
                     f"razón ganancia = {node['gain_ratio']:.3f}\nn = {node['n']:g}")
            color = INTERNAL_COLOR
        lines.append(f'  n{node["id"]} [label="{_dot_escape(label)}", fillcolor="{color}"];')
        for edge, child in zip(node.get("edges", []), node.get("children", [])):
            lines.append(f'  n{node["id"]} -> n{child["id"]} [label="{_dot_escape(edge)}"];')
            pending.append(child)
    lines.append("}")
    return "\n".join(lines)


def trace_case(tree, row) -> dict:
    """{id de nodo: peso} con que UN caso pasa por cada nodo (un caso completo recorre un solo camino).

    `row`: dict {atributo: valor}; un valor faltante es None o NaN. Si falta (o es un nominal no visto) en un
    nodo, el caso se reparte entre sus ramas según branch_probs: los pesos de los hijos suman el del padre.
    """
    tree._check_fitted()
    ids = {id(node): i for i, node in enumerate(_preorder(tree.tree_))}
    frame = tree._as_frame(pd.DataFrame([dict(row)]))
    cols = tree._columns(frame)
    weights: dict = {}

    def visit(node, weight: float) -> None:
        weights[ids[id(node)]] = weight
        if node.is_leaf:
            return
        value = cols[node.feature][0]
        branch = None
        if node.kind == "numeric":
            if not np.isnan(value):
                branch = 0 if value <= node.threshold else 1
        elif value is not None and value in node.values:
            branch = node.values.index(value)
        for i, child in enumerate(node.children):
            if branch is None:
                visit(child, weight * float(node.branch_probs[i]))
            elif i == branch:
                visit(child, weight)
            else:
                skip(child)

    def skip(node) -> None:                   # las ramas que el caso no toca figuran con peso 0
        weights[ids[id(node)]] = 0.0
        for child in node.children:
            skip(child)

    visit(tree.tree_, 1.0)
    return weights
