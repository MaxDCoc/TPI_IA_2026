"""
Experimentos: barrido de un parámetro, selección con validación cruzada anidada y robustez a faltantes.

GUÍA DE LECTURA
---------------
  - parameter_sweep(): varía UN parámetro y deja los demás fijos. Devuelve, por valor, el rendimiento en test
    y en train: la brecha train - test mide el sobreajuste (es la "curva de complejidad" del informe).
  - nested_cv(): elegir los parámetros con los mismos datos con que después se mide el resultado lo infla,
    porque el test "participó" de la elección. Por eso, dentro de CADA partición externa se elige la mejor
    combinación de la grilla con una validación cruzada interna que ve SOLO el train externo; recién después
    se evalúa en el test externo. Ante un empate gana la primera combinación de la grilla: poner primero la
    más simple.
  - inject_missing() / missing_robustness(): borra al azar una fracción de los valores (faltantes completamente
    al azar) y mide cuánto cae cada modelo. Es donde más se diferencian los dos: el árbol C4.5 reparte fracciones
    de caso por las ramas, mientras que el Random Forest tiene que imputar con la mediana o la moda.
  - Todo se apoya en run_experiment y make_splits: mismas particiones, mismas métricas, mismo formato de tabla.
"""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from .runner import run_experiment
from .splits import make_splits

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")   # las que se calculan en train y en test


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"metric debe ser una de {METRICS}, no {metric!r}")


# ------------------------------------------------------------------ un parámetro a la vez
def parameter_sweep(dataset, kind: str, param: str, values, splits, base_params=None,
                    metric: str = "macro_f1") -> pd.DataFrame:
    """Rendimiento en test y en train al variar UN parámetro. `gap` = train - test (sobreajuste)."""
    _check_metric(metric)
    values = list(values)
    labels = [str(v) for v in values]
    if not values or len(set(labels)) != len(labels):
        raise ValueError("values no puede estar vacío ni repetir valores")
    base = dict(base_params or {})
    models = {label: (kind, {**base, param: value}) for label, value in zip(labels, values)}
    results = run_experiment(dataset, models, splits, register_decision=False)
    rows = []
    for label, value in zip(labels, values):
        part = results[results["model"] == label]
        test, train = part[f"test_{metric}"], part[f"train_{metric}"]
        rows.append({"param": param, "value": value, "n_partitions": len(part),
                     f"test_{metric}_mean": test.mean(), f"test_{metric}_std": test.std(ddof=1),
                     f"train_{metric}_mean": train.mean(), "gap": train.mean() - test.mean(),
                     "fit_time_mean": part["fit_time"].mean()})
    table = pd.DataFrame(rows)
    table["value"] = pd.Series(values, dtype=object)    # sin esto 1 pasa a 1.0 y None a NaN
    return table


# ------------------------------------------------------------------ validación cruzada anidada
def _combinations(grid: dict) -> list:
    names = list(grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*(list(grid[n]) for n in names))]


def nested_cv(dataset, kind: str, grid: dict, outer_splits, inner_splits: int = 3, inner_repeats: int = 1,
              metric: str = "macro_f1", base_params=None, seed: int = 0) -> pd.DataFrame:
    """Una fila por partición externa: combinación elegida con la CV interna y su resultado en el test externo."""
    _check_metric(metric)
    if not grid or any(len(list(v)) == 0 for v in grid.values()):
        raise ValueError("grid no puede estar vacía ni tener parámetros sin valores")
    base = dict(base_params or {})
    combos = _combinations(grid)
    candidates = {str(i): (kind, {**base, **combo}) for i, combo in enumerate(combos)}
    rows = []
    for outer in outer_splits:
        # La CV interna trabaja con una copia que contiene SOLO el train externo: el test externo no existe para ella.
        train_only = replace(dataset, X=dataset.X.iloc[outer.train].reset_index(drop=True),
                             y=dataset.y.iloc[outer.train].reset_index(drop=True))
        inner = make_splits(train_only.y, n_splits=inner_splits, n_repeats=inner_repeats, seed=seed + outer.id)
        inner_results = run_experiment(train_only, candidates, inner, register_decision=False)
        scores = inner_results.groupby("model")[f"test_{metric}"].mean().reindex(list(candidates))
        best = int(np.argmax(scores.to_numpy()))        # ante un empate, la primera de la grilla
        final = run_experiment(dataset, {"elegido": (kind, {**base, **combos[best]})}, [outer],
                               register_decision=False).iloc[0]
        row = {"outer_split": outer.id, "repeat": outer.repeat, "fold": outer.fold,
               "chosen": json.dumps(combos[best], sort_keys=True, default=str),
               "inner_score": float(scores.iloc[best])}
        row.update({k: final[k] for k in final.index if k.startswith(("test_", "train_"))})
        rows.append(row)
    return pd.DataFrame(rows)


def selection_frequency(nested: pd.DataFrame) -> pd.Series:
    """Cuántas veces eligió cada combinación la CV interna: si cambia mucho, la elección es inestable."""
    return nested["chosen"].value_counts()


# ------------------------------------------------------------------ robustez a faltantes
def inject_missing(dataset, fraction: float, seed: int = 0, columns=None):
    """Copia del dataset con una fracción de los valores de los atributos borrada al azar. No toca la clase."""
    if not 0 <= fraction < 1:
        raise ValueError("fraction debe estar en [0, 1)")
    columns = list(columns) if columns is not None else list(dataset.X.columns)
    unknown = set(columns) - set(dataset.X.columns)
    if unknown:
        raise ValueError(f"Columnas que no existen en los datos: {sorted(map(str, unknown))}")
    mask = np.random.default_rng(seed).random((len(dataset.X), len(columns))) < fraction
    X = dataset.X.copy()
    X[columns] = X[columns].mask(pd.DataFrame(mask, index=X.index, columns=columns))
    return replace(dataset, name=f"{dataset.name} (+{fraction:.0%} faltantes)", X=X, log=copy.deepcopy(dataset.log))


def missing_robustness(dataset, models: dict, fractions, splits, seeds=(0,)) -> pd.DataFrame:
    """run_experiment repetido con fracciones crecientes de faltantes (en train y en test), con las mismas particiones.

    Cada semilla de `seeds` es una máscara de faltantes distinta: promediar varias evita que el resultado dependa
    de qué celdas cayeron justo. Con la misma semilla las máscaras son anidadas (las celdas borradas al 10 % siguen
    borradas al 20 %), así que las fracciones son comparables entre sí.
    """
    frames = []
    for seed in seeds:
        for fraction in fractions:
            results = run_experiment(inject_missing(dataset, fraction, seed=seed), models, splits,
                                     register_decision=False)
            results["missing_fraction"] = fraction
            results["mask_seed"] = seed
            frames.append(results)
    return pd.concat(frames, ignore_index=True)


def robustness_table(results: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    """Media de la métrica en test: una fila por fracción de faltantes y una columna por modelo."""
    table = results.pivot_table(index="missing_fraction", columns="model", values=f"test_{metric}", aggfunc="mean")
    return table[list(dict.fromkeys(results["model"]))]
