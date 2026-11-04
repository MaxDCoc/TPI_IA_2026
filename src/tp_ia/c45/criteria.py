"""
Criterios de C4.5: entropía, ganancia de información, split info y gain ratio.

GUÍA DE LECTURA (para el coloquio)
----------------------------------
Todo se calcula sobre *conteos ponderados de clases*: un vector de largo k (k = nº de clases)
donde cada posición es la suma de pesos de los casos de esa clase.

  - Usamos pesos y no simples cuentas porque C4.5 reparte "fracciones de caso" cuando un valor
    falta (ver tree.py). Sin faltantes, cada caso pesa 1 y todo coincide con las fórmulas del libro.

Fórmulas (Quinlan, 1993):
  Entropía        H(S)          = - sum_c p_c * log2(p_c)
  Ganancia        Gain(S, A)    = H(S) - sum_b (|S_b| / |S|) * H(S_b)
  Split info      SplitInfo(A)  = - sum_b (|S_b| / |S|) * log2(|S_b| / |S|)
  Gain ratio      GainRatio(A)  = Gain(S, A) / SplitInfo(A)

Por qué el gain ratio: la ganancia "premia" a los atributos con muchas ramas (un atributo tipo ID
separa todo y gana mucho sin servir). Dividir por SplitInfo penaliza justamente eso.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12  # tolerancia numérica para comparar con cero


def entropy(counts) -> float:
    """Entropía de Shannon (base 2) de un vector de conteos (pueden ser pesos fraccionarios)."""
    c = np.asarray(counts, dtype=float)
    total = c.sum()
    if total <= EPS:
        return 0.0
    p = c[c > 0] / total  # solo las clases presentes: 0 * log2(0) se toma como 0
    return float(-(p * np.log2(p)).sum())


def entropy_rows(count_matrix) -> np.ndarray:
    """Entropía de cada fila de una matriz de conteos (m filas x k clases), sin bucles de Python.

    Se usa para evaluar *todos* los umbrales de un atributo numérico de una sola vez.
    """
    m = np.asarray(count_matrix, dtype=float)
    total = m.sum(axis=1, keepdims=True)
    p = m / np.where(total > EPS, total, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=1)


def information_gain(parent_counts, child_counts) -> float:
    """Ganancia de información de una partición.

    parent_counts: conteos de clases del nodo padre (vector de largo k).
    child_counts:  matriz (ramas x k) con los conteos de cada rama hija.
    """
    parent = np.asarray(parent_counts, dtype=float)
    child = np.asarray(child_counts, dtype=float)
    total = parent.sum()
    if total <= EPS:
        return 0.0
    branch_share = child.sum(axis=1) / total
    gain = entropy(parent) - float((branch_share * entropy_rows(child)).sum())
    return max(gain, 0.0)  # evita -1e-16 por redondeo


def split_info(branch_weights) -> float:
    """Entropía de los *tamaños* de las ramas (ignora las clases). Es 0 si hay una sola rama."""
    return entropy(branch_weights)


def gain_ratio(gain: float, split_information: float) -> float:
    """Gain / SplitInfo. Si SplitInfo es ~0 la división es degenerada y devolvemos 0."""
    if split_information <= EPS:
        return 0.0
    return gain / split_information
