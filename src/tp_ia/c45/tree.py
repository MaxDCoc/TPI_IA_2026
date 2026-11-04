"""
Árbol de decisión C4.5 - implementación propia (sin ninguna librería de árboles de decisión).

GUÍA DE LECTURA (para el coloquio)
----------------------------------
Idea general (Quinlan, 1993): construir el árbol de arriba hacia abajo. En cada nodo:

  1. Si el nodo es "puro" (una sola clase), o tiene muy pocos casos, o llegó a max_depth -> HOJA.
  2. Si no, para cada atributo se calcula su mejor división (ver splits.py) y se elige una con la
     regla de C4.5:
        a) quedarse con los atributos cuya ganancia es >= la ganancia PROMEDIO, y
        b) entre ellos, el de mayor GAIN RATIO.
     (La parte (a) evita que el gain ratio elija un atributo de ganancia ínfima solo porque su
      SplitInfo es chico.)
  3. Si la mejor ganancia no supera min_gain -> HOJA (no vale la pena dividir).
  4. Se reparten los casos entre las ramas y se repite recursivamente.

VALORES FALTANTES (C4.5)
  - Entrenamiento: un caso al que le falta el atributo elegido se manda a TODAS las ramas con peso
    fraccionario (proporcional al peso de cada rama). Por eso todo el código trabaja con pesos.
  - Predicción: igual, el caso viaja por todas las ramas y se mezclan las distribuciones de clases
    de las hojas, ponderadas por la probabilidad de cada rama.
  - Un valor nominal NO visto en el entrenamiento se trata como faltante.

SALIDA
  - Cada hoja guarda la distribución ponderada de clases. predict_proba la normaliza y predict
    devuelve la clase más probable (empates: la primera en orden -> resultado determinista).

DIFERENCIAS CON C4.5 COMPLETO (para la tabla "implementado vs omitido" del informe)
  - Implementado: gain ratio + heurística de ganancia promedio, umbrales numéricos, ramas
    multivalor, faltantes con reparto fraccionario, min_samples_leaf (-m), corrección release 8
    (opcional). Pendiente: poda por error pesimista (HU-26), agrupación de valores nominales.
  - Diferencia conocida: la ganancia PROMEDIO se calcula solo sobre los atributos con ganancia
    real (> min_gain); C4.5 promedia sobre todos los atributos evaluados.
  - min_samples_split=2 por defecto casi no actúa: min_samples_leaf=2 ya exige peso >= 4 para dividir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .criteria import EPS
from .splits import SplitCandidate, evaluate_nominal, evaluate_numeric

CRITERIA = ("gain_ratio", "gain")
THRESHOLD_MODES = ("c45", "midpoint")
FEATURE_KINDS = ("numeric", "nominal")
_PARAM_NAMES = ("criterion", "max_depth", "min_samples_split", "min_samples_leaf", "min_gain",
                "use_avg_gain", "threshold_mode", "numeric_gain_correction")


def _tidy(value):
    """Un nominal numérico como 3.0 (la columna era float porque tenía NaN) se compara y se muestra como 3."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


@dataclass
class Node:
    """Nodo del árbol. Es hoja si no tiene hijos."""

    n_weight: float                       # peso total de casos que llegaron al nodo
    class_counts: np.ndarray              # distribución ponderada de clases en el nodo
    depth: int
    feature: int | None = None            # índice de la columna usada para dividir (nodos internos)
    kind: str | None = None               # "numeric" | "nominal"
    threshold: float | None = None        # numérico: rama 0 si x <= threshold, rama 1 si x > threshold
    values: list = field(default_factory=list)   # nominal: valor que corresponde a cada rama
    children: list = field(default_factory=list)
    branch_probs: np.ndarray | None = None       # probabilidad de cada rama (reparte los faltantes)
    gain: float = 0.0
    gain_ratio: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def proba(self) -> np.ndarray:
        total = float(self.class_counts.sum())
        if total <= EPS:
            return np.full(len(self.class_counts), 1.0 / len(self.class_counts))
        return self.class_counts / total

    @property
    def errors(self) -> float:
        """Peso mal clasificado si este nodo fuera hoja (base de la poda por error pesimista)."""
        return float(self.n_weight - self.class_counts.max())


class C45DecisionTree:
    """Clasificador C4.5.

    Parámetros (todos ajustables desde el "laboratorio de parámetros" de la interfaz):
      criterion              "gain_ratio" (C4.5) o "gain" (ganancia de información pura, estilo ID3).
      max_depth              profundidad máxima (None = sin límite).
      min_samples_split      peso mínimo de un nodo para intentar dividirlo.
      min_samples_leaf       peso mínimo que deben tener al menos dos ramas (parámetro -m de C4.5).
      min_gain               ganancia mínima para aceptar una división (pre-poda).
      use_avg_gain           aplicar la heurística de ganancia promedio de C4.5.
      threshold_mode         "c45" (valor observado <= punto medio) o "midpoint" (punto medio).
      numeric_gain_correction  corrección log2(N-1)/|D| de C4.5 release 8 para atributos numéricos.
    """

    def __init__(self, criterion="gain_ratio", max_depth=None, min_samples_split=2,
                 min_samples_leaf=2, min_gain=0.0, use_avg_gain=True, threshold_mode="c45",
                 numeric_gain_correction=False):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.use_avg_gain = use_avg_gain
        self.threshold_mode = threshold_mode
        self.numeric_gain_correction = numeric_gain_correction

    # ------------------------------------------------------------------ parámetros
    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params):
        for name, value in params.items():
            if name not in _PARAM_NAMES:
                raise ValueError(f"Parámetro desconocido: {name!r}. Válidos: {_PARAM_NAMES}")
            setattr(self, name, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"C45DecisionTree({args})"

    def _validate_params(self):
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion debe ser uno de {CRITERIA}, no {self.criterion!r}")
        if self.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(f"threshold_mode debe ser uno de {THRESHOLD_MODES}, no {self.threshold_mode!r}")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth debe ser >= 0 o None")
        if self.min_samples_split < 1:
            raise ValueError("min_samples_split debe ser >= 1")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf debe ser >= 1")
        if self.min_gain < 0:
            raise ValueError("min_gain debe ser >= 0")

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, X, y, feature_types=None, sample_weight=None):
        """Entrena el árbol.

        X: DataFrame (o matriz) con los atributos. Los faltantes se marcan con NaN/None/pd.NA.
        y: etiquetas (cualquier tipo comparable: texto o números).
        feature_types: dict opcional {columna: "numeric" | "nominal"}. Si se omite, se infiere por
            dtype. IMPORTANTE: una columna numérica que en realidad es una categoría (p. ej. un
            código 1/2/3/4) debe declararse "nominal" para que el árbol ramifique por valor.
        sample_weight: pesos por caso (opcional).
        """
        self._validate_params()
        X = self._as_frame(X, fitting=True)
        y = np.asarray(y)
        if len(X) == 0:
            raise ValueError("No hay datos para entrenar (X está vacío).")
        if len(X) != len(y):
            raise ValueError(f"X tiene {len(X)} filas pero y tiene {len(y)}.")
        if pd.isna(y).any():
            raise ValueError("y tiene valores faltantes; elimine esas filas antes de entrenar.")
        weights = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(X),) or (weights < 0).any():
            raise ValueError("sample_weight debe tener un valor >= 0 por fila.")

        self.feature_names_ = list(X.columns)
        self.feature_types_ = self._resolve_types(X, feature_types)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        y_idx = np.asarray(y_idx).reshape(-1)
        self.n_classes_ = len(self.classes_)

        keep = np.nonzero(weights > 0)[0]
        if keep.size == 0:
            raise ValueError("Todos los pesos son cero.")
        cols = self._columns(X)
        self.tree_ = self._grow(cols, y_idx, keep, weights[keep], depth=0)
        self._compute_importances()
        return self

    def _grow(self, cols, y, idx, w, depth) -> Node:
        """Construye recursivamente el subárbol de los casos `idx` (con pesos `w`)."""
        counts = np.bincount(y[idx], weights=w, minlength=self.n_classes_)
        node = Node(n_weight=float(w.sum()), class_counts=counts, depth=depth)

        # Paso 1: criterios de parada -> hoja.
        pure = np.count_nonzero(counts > EPS) <= 1
        too_small = node.n_weight < self.min_samples_split - EPS
        too_deep = self.max_depth is not None and depth >= self.max_depth
        if pure or too_small or too_deep:
            return node

        # Paso 2: elegir la mejor división; si ninguna sirve -> hoja.
        cand = self._best_split(cols, y, idx, w, node.n_weight)
        if cand is None:
            return node

        # Paso 3: repartir los casos (los faltantes van a todas las ramas, con peso fraccionario).
        parts, probs = self._partition(cand, cols[cand.feature][idx], idx, w)
        node.feature, node.kind, node.threshold = cand.feature, cand.kind, cand.threshold
        node.values = list(cand.values) if cand.values is not None else []
        node.branch_probs = probs
        node.gain, node.gain_ratio = cand.gain, cand.gain_ratio

        # Paso 4: recursión sobre cada rama.
        node.children = [self._grow(cols, y, ci, cw, depth + 1) for ci, cw in parts]
        return node

    def _best_split(self, cols, y, idx, w, total_weight) -> SplitCandidate | None:
        """Regla de selección de C4.5 entre las mejores divisiones de cada atributo."""
        candidates = []
        for j, kind in enumerate(self.feature_types_):
            x = cols[j][idx]
            if kind == "numeric":
                cand = evaluate_numeric(j, x, y[idx], w, self.n_classes_, total_weight,
                                        self.min_samples_leaf, self.threshold_mode,
                                        self.numeric_gain_correction)
            else:
                cand = evaluate_nominal(j, x, y[idx], w, self.n_classes_, total_weight,
                                        self.min_samples_leaf)
            # Solo cuentan las divisiones con ganancia real y que no sean degeneradas (SplitInfo > 0).
            if cand is not None and cand.gain > self.min_gain + EPS and cand.split_info > EPS:
                candidates.append(cand)
        if not candidates:
            return None

        if self.criterion == "gain":                      # ganancia pura (estilo ID3)
            pool, key = candidates, (lambda c: c.gain)
        else:                                             # gain ratio (C4.5)
            pool = candidates
            if self.use_avg_gain:                         # heurística: ganancia >= promedio
                average = float(np.mean([c.gain for c in candidates]))
                pool = [c for c in candidates if c.gain >= average - 1e-9]
            key = lambda c: c.gain_ratio

        best = pool[0]                                    # empate -> gana el atributo de menor índice
        for c in pool[1:]:
            if key(c) > key(best) + 1e-12:
                best = c
        return best

    @staticmethod
    def _partition(cand, x, idx, w):
        """Reparte (idx, w) entre las ramas. Devuelve ([(idx_rama, w_rama), ...], prob. de cada rama)."""
        if cand.kind == "numeric":
            known = ~np.isnan(x)
            masks = [known & (x <= cand.threshold), known & (x > cand.threshold)]
        else:
            known = np.array([v is not None for v in x], dtype=bool)
            masks = [known & (x == v) for v in cand.values]
        probs = cand.branch_weights / cand.branch_weights.sum()
        missing = ~known
        parts = []
        for mask, p in zip(masks, probs):
            child_idx, child_w = idx[mask], w[mask]
            if missing.any():  # los faltantes pasan por todas las ramas con peso w * p
                child_idx = np.concatenate([child_idx, idx[missing]])
                child_w = np.concatenate([child_w, w[missing] * p])
            parts.append((child_idx, child_w))
        return parts, probs

    # ------------------------------------------------------------------ predicción
    def predict_proba(self, X) -> np.ndarray:
        """Probabilidad de cada clase (columnas en el orden de `classes_`)."""
        self._check_fitted()
        X = self._as_frame(X)
        cols = self._columns(X)
        n = len(X)
        out = np.zeros((n, self.n_classes_))
        self._route(self.tree_, cols, np.arange(n), np.ones(n), out)
        return out

    def predict(self, X):
        self._check_fitted()
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _route(self, node, cols, idx, w, out):
        """Lleva un lote de casos por el árbol acumulando las probabilidades en `out`.

        Un caso con valor conocido sigue UNA rama (peso completo). Uno con valor faltante (o nominal
        no visto) sigue TODAS las ramas con peso w * prob_rama: así se mezclan las hojas alcanzadas.
        """
        if idx.size == 0:
            return
        if node.is_leaf:
            out[idx] += w[:, None] * node.proba
            return
        x = cols[node.feature][idx]
        if node.kind == "numeric":
            known = ~np.isnan(x)
            masks = [known & (x <= node.threshold), known & (x > node.threshold)]
        else:
            lookup = {v: i for i, v in enumerate(node.values)}
            branch = np.array([lookup.get(v, -1) if v is not None else -1 for v in x])
            known = branch >= 0
            masks = [branch == i for i in range(len(node.values))]
        missing = ~known
        for mask, child, p in zip(masks, node.children, node.branch_probs):
            take = mask | missing
            weights = np.where(mask[take], w[take], w[take] * p)
            self._route(child, cols, idx[take], weights, out)

    # ------------------------------------------------------------------ interpretación
    def export_text(self) -> str:
        """El árbol como texto (estilo J48/Weka). Entre paréntesis: casos (y errores) de la hoja."""
        self._check_fitted()

        def leaf_text(node):
            label = self.classes_[int(np.argmax(node.class_counts))]
            errors = f"/{node.errors:g}" if node.errors > 1e-9 else ""
            return f"{label} ({node.n_weight:g}{errors})"

        def conditions(node):
            name = self.feature_names_[node.feature]
            if node.kind == "numeric":
                return [f"{name} <= {node.threshold:g}", f"{name} > {node.threshold:g}"]
            return [f"{name} = {v}" for v in node.values]

        if self.tree_.is_leaf:
            return f": {leaf_text(self.tree_)}"
        lines = []

        def walk(node, depth):
            for cond, child in zip(conditions(node), node.children):
                prefix = "|   " * depth
                if child.is_leaf:
                    lines.append(f"{prefix}{cond}: {leaf_text(child)}")
                else:
                    lines.append(f"{prefix}{cond}")
                    walk(child, depth + 1)

        walk(self.tree_, 0)
        return "\n".join(lines)

    def explain(self, row) -> dict:
        """Explica la predicción de UN caso: reglas recorridas, clase y probabilidades.

        Si en algún nodo el valor falta (o es un nominal no visto), el recorrido se corta ahí y se
        avisa: la predicción final igual mezcla las ramas (ver _route).
        """
        self._check_fitted()
        frame = self._as_frame(pd.DataFrame([dict(row)]))
        cols = self._columns(frame)
        node, rules = self.tree_, []
        while not node.is_leaf:
            name = self.feature_names_[node.feature]
            value = cols[node.feature][0]
            if node.kind == "numeric":
                if np.isnan(value):
                    rules.append(f"{name}: valor faltante -> se reparte entre las ramas")
                    break
                go_left = value <= node.threshold
                rules.append(f"{name} {'<=' if go_left else '>'} {node.threshold:g}")
                node = node.children[0 if go_left else 1]
            else:
                if value is None or value not in node.values:
                    rules.append(f"{name}: valor faltante o no visto -> se reparte entre las ramas")
                    break
                rules.append(f"{name} = {value}")
                node = node.children[node.values.index(value)]
        proba = self.predict_proba(frame)[0]
        return {"rules": rules, "label": self.classes_[int(np.argmax(proba))],
                "proba": {str(c): float(p) for c, p in zip(self.classes_, proba)}}

    # ------------------------------------------------------------------ estructura
    def _iter_nodes(self):
        stack = [self.tree_]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children)

    @property
    def depth_(self) -> int:
        self._check_fitted()
        return max(n.depth for n in self._iter_nodes())

    @property
    def n_nodes_(self) -> int:
        self._check_fitted()
        return sum(1 for _ in self._iter_nodes())

    @property
    def n_leaves_(self) -> int:
        self._check_fitted()
        return sum(1 for n in self._iter_nodes() if n.is_leaf)

    def _compute_importances(self):
        """Importancia de un atributo = suma de (peso del nodo x ganancia) en los nodos que lo usan."""
        importance = np.zeros(len(self.feature_names_))
        for node in self._iter_nodes():
            if not node.is_leaf:
                importance[node.feature] += node.n_weight * node.gain
        total = importance.sum()
        self.feature_importances_ = importance / total if total > EPS else importance

    # ------------------------------------------------------------------ utilidades internas
    def _check_fitted(self):
        if not hasattr(self, "tree_"):
            raise RuntimeError("El árbol no está entrenado: llame a fit() primero.")

    def _as_frame(self, X, fitting=False) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.reset_index(drop=True).copy()
        else:
            frame = pd.DataFrame(X, columns=None if fitting else self.feature_names_)
        frame.columns = [str(c) for c in frame.columns]
        if not fitting:
            absent = [c for c in self.feature_names_ if c not in frame.columns]
            if absent:
                raise ValueError(f"Faltan columnas en X: {absent}")
            frame = frame[self.feature_names_]
        return frame

    @staticmethod
    def _resolve_types(X, feature_types):
        given = {str(k): v for k, v in (feature_types or {}).items()}
        unknown = set(given) - set(X.columns)
        if unknown:
            raise ValueError(f"feature_types menciona columnas inexistentes: {sorted(unknown)}")
        types = []
        for column in X.columns:
            if column in given:
                kind = given[column]
                if kind not in FEATURE_KINDS:
                    raise ValueError(f"Tipo inválido para {column!r}: {kind!r}. Use {FEATURE_KINDS}.")
            else:
                s = X[column]
                is_number = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                kind = "numeric" if is_number else "nominal"
            types.append(kind)
        return types

    def _columns(self, X):
        """Una columna por atributo: numéricos como float (NaN = faltante), nominales como objetos
        (None = faltante). Un texto no numérico en una columna numérica (p. ej. '?') pasa a NaN."""
        cols = []
        for column, kind in zip(self.feature_names_, self.feature_types_):
            s = X[column]
            if kind == "numeric":
                cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
            else:
                cols.append(np.array([None if pd.isna(v) else _tidy(v) for v in s], dtype=object))
        return cols
