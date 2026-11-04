"""
Evaluación de divisiones candidatas para un nodo: atributos numéricos y nominales.

GUÍA DE LECTURA (para el coloquio)
----------------------------------
Para cada atributo A calculamos *la mejor división posible* y la resumimos en un SplitCandidate:

  - Atributo NUMÉRICO  -> división binaria  "A <= t"  /  "A > t".
        Se ordenan los valores y se prueban todos los cortes entre dos valores distintos
        consecutivos. Con sumas acumuladas por clase, probar todos los cortes cuesta O(n log n)
        en vez de O(n^2). El corte elegido es el de mayor ganancia (como hace C4.5).
  - Atributo NOMINAL   -> división multirrama: una rama por cada valor observado.

VALORES FALTANTES (idea de C4.5)
  - La ganancia se calcula solo con los casos que tienen el valor conocido y se multiplica por F, la
    fracción (en peso) de casos conocidos: Gain = F * Gain_conocidos.
  - Para SplitInfo los faltantes cuentan como una rama más.

MÍNIMO POR RAMA (parámetro -m de C4.5)
  - Una división solo es válida si al menos DOS ramas tienen peso >= min_leaf. Evita divisiones que
    dejan una hoja con un solo caso suelto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .criteria import EPS, entropy, entropy_rows, gain_ratio, split_info


@dataclass
class SplitCandidate:
    """Mejor división encontrada para un atributo."""

    feature: int                    # índice de columna
    kind: str                       # "numeric" | "nominal"
    gain: float                     # ganancia de C4.5 (ya multiplicada por la fracción de conocidos)
    gain_ratio: float
    split_info: float
    branch_weights: np.ndarray      # peso de los casos con valor conocido en cada rama
    threshold: float | None = None  # solo numérico: corte "x <= threshold"
    values: list | None = None      # solo nominal: valor que corresponde a cada rama


def _factorize(values: np.ndarray):
    """Devuelve (códigos, valores únicos ordenados). El orden fijo hace el árbol determinista."""
    uniques = list(dict.fromkeys(values.tolist()))
    try:
        uniques.sort()
    except TypeError:  # tipos mezclados que no se pueden comparar
        uniques.sort(key=str)
    index = {v: i for i, v in enumerate(uniques)}
    codes = np.fromiter((index[v] for v in values.tolist()), dtype=int, count=len(values))
    return codes, uniques


def evaluate_numeric(feature, x, y, w, n_classes, total_weight, min_leaf,
                     threshold_mode="c45", gain_correction=False):
    """Mejor corte binario de un atributo numérico (NaN = faltante). None si no hay corte válido."""
    known = ~np.isnan(x)
    if known.sum() < 2:
        return None
    xs, ys, ws = x[known], y[known], w[known]

    # Paso 1: ordenar por valor del atributo.
    order = np.argsort(xs, kind="stable")
    xs, ys, ws = xs[order], ys[order], ws[order]
    n = len(xs)

    # Paso 2: conteos acumulados por clase. cum[i] = conteos de los casos 0..i (los de la izquierda
    # si cortamos justo después de i). Así cada corte candidato se evalúa sin recorrer los datos.
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), ys] = ws
    cum = np.cumsum(onehot, axis=0)
    total_counts = cum[-1]
    known_weight = float(total_counts.sum())

    # Paso 3: solo tiene sentido cortar entre dos valores DISTINTOS consecutivos.
    cut = np.nonzero(xs[:-1] < xs[1:])[0]
    if cut.size == 0:
        return None
    left_counts = cum[cut]
    right_counts = total_counts - left_counts
    left_w = left_counts.sum(axis=1)
    right_w = known_weight - left_w

    # Paso 4: descartar cortes que dejan una rama con menos de min_leaf de peso.
    valid = (left_w >= min_leaf - EPS) & (right_w >= min_leaf - EPS)
    if not valid.any():
        return None

    # Paso 5: ganancia de cada corte = H(padre) - promedio ponderado de H(hijos).
    child_info = (left_w * entropy_rows(left_counts) + right_w * entropy_rows(right_counts)) / known_weight
    gains_known = np.where(valid, entropy(total_counts) - child_info, -np.inf)
    best = int(np.argmax(gains_known))  # primer máximo -> desempate determinista (corte más bajo)
    gain_known = float(gains_known[best])

    # (Opcional) corrección de C4.5 release 8: penaliza a los numéricos por la cantidad de cortes
    # que "tuvieron oportunidad" de ganar, log2(N-1)/|D|. Evita preferirlos solo por tener muchos valores.
    if gain_correction:
        gain_known -= float(np.log2(cut.size)) / known_weight
    gain_known = max(gain_known, 0.0)

    # Paso 6: umbral. "c45" = mayor valor observado <= punto medio (el valor inferior del par);
    # "midpoint" = punto medio. Ambos separan exactamente los mismos casos de entrenamiento.
    i = int(cut[best])
    lower, upper = float(xs[i]), float(xs[i + 1])
    threshold = lower if threshold_mode == "c45" else (lower + upper) / 2.0

    gain = (known_weight / total_weight) * gain_known
    branch_weights = np.array([left_w[best], right_w[best]])
    missing_weight = total_weight - known_weight
    si_weights = np.append(branch_weights, missing_weight) if missing_weight > EPS else branch_weights
    si = split_info(si_weights)
    return SplitCandidate(feature, "numeric", gain, gain_ratio(gain, si), si, branch_weights,
                          threshold=threshold)


def evaluate_nominal(feature, x, y, w, n_classes, total_weight, min_leaf):
    """División multirrama de un atributo nominal (None = faltante). None si no hay división válida."""
    known = np.array([v is not None for v in x], dtype=bool)
    if known.sum() < 2:
        return None
    codes, uniques = _factorize(x[known])
    if len(uniques) < 2:
        return None  # un solo valor: no separa nada
    ys, ws = y[known], w[known]

    # Conteos (valor x clase) en una sola pasada.
    counts = np.zeros((len(uniques), n_classes))
    np.add.at(counts, (codes, ys), ws)
    branch_w = counts.sum(axis=1)
    if (branch_w >= min_leaf - EPS).sum() < 2:
        return None  # regla de C4.5: al menos dos ramas "con cuerpo"

    known_weight = float(branch_w.sum())
    child_info = float((branch_w * entropy_rows(counts)).sum()) / known_weight
    gain_known = max(entropy(counts.sum(axis=0)) - child_info, 0.0)
    gain = (known_weight / total_weight) * gain_known

    missing_weight = total_weight - known_weight
    si_weights = np.append(branch_w, missing_weight) if missing_weight > EPS else branch_w
    si = split_info(si_weights)
    return SplitCandidate(feature, "nominal", gain, gain_ratio(gain, si), si, branch_w,
                          values=uniques)
