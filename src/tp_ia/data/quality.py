"""
Auditoría de calidad de un dataset ya cargado: faltantes, duplicados, clases, columnas raras.

GUÍA DE LECTURA
---------------
audit() devuelve un diccionario con los números y una lista de `warnings` en castellano. Es lo que
muestra la pantalla "Datos" de la interfaz y lo que respalda las decisiones D-03 (faltantes) y D-04
(duplicados). Los umbrales son parámetros: no hay números mágicos escondidos.
"""
from __future__ import annotations

import pandas as pd


def _py(value):
    return value.item() if hasattr(value, "item") else value


def audit(X: pd.DataFrame, y: pd.Series, feature_types: dict, small_class: int = 15,
          high_missing: float = 0.30, imbalance_ratio: float = 5.0) -> dict:
    n = len(X)
    counts = y.value_counts().sort_index()
    class_counts = {_py(k): int(v) for k, v in counts.items()}

    missing = {}
    for column in X.columns:
        k = int(X[column].isna().sum())
        if k:
            missing[str(column)] = {"count": k, "pct": round(100.0 * k / n, 1)}
    rows_with_missing = int(X.isna().any(axis=1).sum())

    # Un duplicado es una fila idéntica en atributos Y clase.
    both = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename("__objetivo__")], axis=1)
    duplicates = int(both.duplicated().sum())

    constant = [str(c) for c in X.columns if X[c].nunique(dropna=True) <= 1]
    id_like = []
    for column in X.columns:
        values = X[column].dropna()
        if n >= 10 and values.nunique() == n:
            integer_valued = pd.api.types.is_numeric_dtype(values) and bool((values % 1 == 0).all())
            if feature_types.get(column) == "nominal" or integer_valued:
                id_like.append(str(column))

    numeric_summary = {}
    for column in X.columns:
        if feature_types.get(column) == "numeric":
            values = pd.to_numeric(X[column], errors="coerce").dropna()
            if len(values):
                numeric_summary[str(column)] = {"min": round(float(values.min()), 3),
                                                "max": round(float(values.max()), 3),
                                                "mean": round(float(values.mean()), 3)}
    cardinality = {str(c): int(X[c].nunique(dropna=True)) for c in X.columns if feature_types.get(c) == "nominal"}

    warnings = []
    for label, count in class_counts.items():
        if count < small_class:
            warnings.append(f"Clase {label}: solo {count} casos. Los pliegues estratificados necesitan k <= {count} "
                            "y las métricas de esa clase van a ser muy ruidosas (agrupar clases o usar menos pliegues).")
    if len(class_counts) > 1:
        ratio = max(class_counts.values()) / max(min(class_counts.values()), 1)
        if ratio > imbalance_ratio:
            warnings.append(f"Desbalance de clases {ratio:.1f}:1. Usar macro-F1 y accuracy balanceada, no solo accuracy.")
    for column, info in missing.items():
        if info["count"] / n > high_missing:
            warnings.append(f"{column}: {info['pct']}% de faltantes.")
    if duplicates:
        warnings.append(f"{duplicates} filas duplicadas: si caen en train y en test las métricas salen infladas.")
    if constant:
        warnings.append(f"Columnas constantes (no aportan): {constant}.")
    if id_like:
        warnings.append(f"Columnas que parecen un identificador (un valor distinto por fila): {id_like}.")

    return {
        "n_rows": n, "n_features": X.shape[1], "class_counts": class_counts,
        "missing": missing, "rows_with_missing": rows_with_missing, "duplicates": duplicates,
        "constant_columns": constant, "id_like_columns": id_like,
        "numeric_summary": numeric_summary, "cardinality": cardinality, "warnings": warnings,
    }
