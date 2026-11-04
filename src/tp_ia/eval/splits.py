"""
Particiones train/test reproducibles y estratificadas.

GUÍA DE LECTURA
---------------
  - Con pocos datos, UNA sola partición es ruido: un error de más mueve la accuracy varios puntos.
    Por eso el esquema por defecto es "repeated_kfold": k pliegues, repetidos n_repeats veces con
    semillas distintas (seed + repetición), y se reporta media ± desvío.
  - ESTRATIFICADO: cada pliegue conserva la proporción de clases. Requiere k <= casos de la clase
    más chica (si no, se avisa con un error claro en vez de fallar adentro de sklearn).
  - groups: si hay filas repetidas (o del mismo paciente), pasarlas con el mismo grupo evita que una
    caiga en train y su copia en test, que infla las métricas.
  - Las MISMAS particiones se usan para DT y RF: la comparación es sobre los mismos datos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

SCHEMES = ("repeated_kfold", "holdout")


@dataclass(frozen=True)
class Split:
    id: int
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def max_stratified_splits(y) -> int:
    """Cantidad máxima de pliegues estratificados que admite y (casos de la clase más chica)."""
    return int(pd.Series(np.asarray(y)).value_counts().min())


def make_splits(y, scheme="repeated_kfold", n_splits=5, n_repeats=3, test_size=0.3, seed=0, groups=None):
    y = np.asarray(y)
    if scheme not in SCHEMES:
        raise ValueError(f"scheme debe ser uno de {SCHEMES}, no {scheme!r}")
    placeholder = np.zeros(len(y))
    splits: list = []
    if scheme == "repeated_kfold":
        smallest = max_stratified_splits(y)
        if n_splits < 2:
            raise ValueError("n_splits debe ser >= 2")
        if n_splits > smallest:
            raise ValueError(f"n_splits={n_splits} supera los casos de la clase más chica ({smallest}). "
                             f"Use n_splits <= {smallest} o agrupe clases.")
        for repeat in range(n_repeats):
            state = seed + repeat
            if groups is None:
                folds = StratifiedKFold(n_splits, shuffle=True, random_state=state).split(placeholder, y)
            else:
                folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=state).split(placeholder, y, groups)
            for fold, (train, test) in enumerate(folds):
                splits.append(Split(len(splits), repeat, fold, train, test))
    else:
        if groups is not None:
            raise ValueError("holdout no admite groups; use repeated_kfold con groups.")
        shuffle = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=seed)
        for repeat, (train, test) in enumerate(shuffle.split(placeholder, y)):
            splits.append(Split(repeat, repeat, 0, train, test))
    return splits
