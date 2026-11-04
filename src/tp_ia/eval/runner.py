"""
Corredor de experimentos: entrena cada modelo en cada partición y devuelve UNA tabla.

GUÍA DE LECTURA
---------------
  - Entrada: un Dataset, un diccionario {nombre: (tipo, parámetros)} y las particiones (splits.py).
  - Salida: un DataFrame con una fila por (partición, modelo): parámetros, tiempos de fit y de predict,
    métricas en TEST y en TRAIN (la diferencia entre ambas es la medida de sobreajuste) y la matriz de
    confusión. Se guarda como CSV con la versión de las librerías: con eso el experimento se reproduce.
  - summarize(): media y desvío por modelo. paired_comparison(): diferencia entre dos modelos sobre las
    MISMAS particiones, con la prueba de Wilcoxon. Las particiones de una validación cruzada se solapan,
    así que ese p-valor es DESCRIPTIVO (optimista), no una prueba formal.
  - Los tiempos del DT (Python puro) y del RF (código compilado) no son comparables en valor absoluto.
"""
from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .metrics import confusion, evaluate
from .models import make_model


def environment_info() -> dict:
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def _align_proba(proba, model_classes, classes) -> np.ndarray:
    """Lleva las probabilidades al conjunto GLOBAL de clases (un pliegue de train puede no tener una clase rara)."""
    out = np.zeros((len(proba), len(classes)))
    index = {c: i for i, c in enumerate(classes)}
    for j, c in enumerate(model_classes):
        out[:, index[c]] = proba[:, j]
    return out


def run_experiment(dataset, models: dict, splits) -> pd.DataFrame:
    X, y = dataset.X, dataset.y.to_numpy()
    classes = list(dataset.classes)
    labels = np.asarray(classes)
    rows = []
    for split in splits:
        X_train, X_test = X.iloc[split.train], X.iloc[split.test]
        y_train, y_test = y[split.train], y[split.test]
        for name, (kind, params) in models.items():
            model = make_model(kind, **params)
            start = time.perf_counter()
            model.fit(X_train, y_train, feature_types=dataset.feature_types)
            fit_time = time.perf_counter() - start
            start = time.perf_counter()
            proba_test = _align_proba(model.predict_proba(X_test), model.classes_, classes)
            predict_time = time.perf_counter() - start
            proba_train = _align_proba(model.predict_proba(X_train), model.classes_, classes)
            pred_test, pred_train = labels[proba_test.argmax(axis=1)], labels[proba_train.argmax(axis=1)]
            test = evaluate(y_test, pred_test, proba_test, classes)
            train = evaluate(y_train, pred_train, proba_train, classes)
            row = {"dataset": dataset.name, "scenario": dataset.scenario, "split": split.id,
                   "repeat": split.repeat, "fold": split.fold, "model": name, "kind": kind,
                   "params": json.dumps(params, sort_keys=True, default=str),
                   "n_train": len(split.train), "n_test": len(split.test),
                   "fit_time": fit_time, "predict_time": predict_time,
                   "confusion": json.dumps(confusion(y_test, pred_test, classes).tolist())}
            row.update({f"test_{k}": v for k, v in test.items()})
            row.update({f"train_{k}": train[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")})
            rows.append(row)
    results = pd.DataFrame(rows)
    results.attrs["env"] = environment_info()
    return results


def _column(results, metric):
    return f"test_{metric}" if f"test_{metric}" in results.columns else metric


def summarize(results, metrics=("accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "fit_time")) -> pd.DataFrame:
    """Media y desvío por modelo (en el orden en que se definieron)."""
    order = list(dict.fromkeys(results["model"]))
    grouped = results.groupby("model")
    table = pd.DataFrame(index=order)
    for metric in metrics:
        column = _column(results, metric)
        if column in results.columns:
            table[f"{metric}_mean"] = grouped[column].mean()
            table[f"{metric}_std"] = grouped[column].std(ddof=1)
    return table


def paired_comparison(results, a: str, b: str, metric: str = "accuracy") -> dict:
    """Diferencia a - b sobre las mismas particiones, con Wilcoxon (descriptivo: ver nota del módulo)."""
    column = _column(results, metric)
    a_scores = results[results["model"] == a].set_index("split")[column]
    b_scores = results[results["model"] == b].set_index("split")[column]
    diff = (a_scores - b_scores).dropna()
    p_value = 1.0 if (len(diff) == 0 or np.allclose(diff, 0)) else float(wilcoxon(diff).pvalue)
    return {"metric": metric, "a": a, "b": b, "n": int(len(diff)), "mean_diff": float(diff.mean()),
            "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
            "p_wilcoxon": p_value}


def save_results(results, path) -> None:
    results.to_csv(path, index=False)
