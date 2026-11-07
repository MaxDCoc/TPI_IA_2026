"""
Capa de servicio de la interfaz web: toda la lógica del laboratorio, sin nada de interfaz.

GUÍA DE LECTURA
---------------
La interfaz (JavaScript dentro de un widget de Colab) solo pide cosas y dibuja lo que vuelve. Esta clase hace el
trabajo: elige el dataset, entrena el árbol propio (C4.5) y el Random Forest, compara, barre parámetros y mide la
robustez a faltantes. Cada método devuelve diccionarios con tipos de Python puros (se pasan a JSON sin cambios),
así que se puede probar con pytest sin abrir ningún navegador.

  - LabService.call(método, parámetros) es el único punto de entrada para la interfaz: despacha a un método
    permitido y limpia el resultado para JSON. Un método que no está en la lista no se ejecuta.
  - El árbol de la pestaña "Árbol" se entrena con una partición estratificada train/test (por defecto 30 % de test;
    Play Tennis usa todos los casos, como en el libro). La pestaña "Comparar" usa validación cruzada repetida.
  - Los datasets se registran con register_dataset(): así se puede sumar Heart Disease o un CSV subido sin tocar nada más.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..c45 import C45DecisionTree
from ..data import load_iris_dataset
from ..data.fixtures import play_tennis
from ..data.loader import Dataset, build_dataset, read_table
from ..data.schema import DatasetSpec
from ..eval import (RandomForestModel, evaluate, make_splits, max_stratified_splits, missing_robustness,
                    paired_comparison, parameter_sweep, robustness_table, run_experiment)
from ..eval.metrics import confusion
from ..eval.runner import _align_proba

DT_PARAMS = ("criterion", "max_depth", "min_samples_split", "min_samples_leaf", "min_gain", "use_avg_gain",
             "threshold_mode", "numeric_gain_correction", "prune", "confidence")
RF_PARAMS = ("n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf", "criterion",
             "class_weight", "random_state")
FOREST_PRESET = {"n_estimators": 100, "max_depth": 10, "random_state": 42}      # el RF de ejemplo de la cátedra
SWEEPABLE = {"dt": ("max_depth", "min_samples_leaf", "min_samples_split", "min_gain", "confidence"),
             "rf": ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split")}
INT_PARAMS = ("max_depth", "min_samples_split", "n_estimators", "random_state")
METHODS = ("datasets", "select", "set_split", "train", "train_forest", "predict", "compare", "sweep",
           "robustness", "decisions", "load_csv")


# ------------------------------------------------------------------ utilidades
def to_jsonable(obj):
    """Convierte (recursivamente) tipos de numpy/pandas a tipos de Python; NaN e infinitos pasan a None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict("records"))
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    return obj if isinstance(obj, str) else str(obj)


def _tidy(value):
    """Un nominal como 3.0 se muestra como 3; los escalares de numpy pasan a Python."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass
class DatasetEntry:
    key: str
    title: str
    description: str
    loader: Callable[[], Dataset]
    source: str = ""
    default_test_size: float = 0.3
    disclaimer: str = ""


# ------------------------------------------------------------------ datasets incluidos
def _play_tennis() -> Dataset:
    X, y = play_tennis()
    frame = X.copy()
    frame["Play"] = np.asarray(y)
    spec = DatasetSpec(name="Play Tennis", target="Play", nominal=list(X.columns),
                       source="Ejemplo clásico de Quinlan (1986), 14 casos con atributos discretos",
                       citation="Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81-106.")
    return build_dataset(frame, spec, source="Play Tennis")


def _sklearn_dataset(title: str, loader, target_label: str, source: str) -> Dataset:
    bunch = loader(as_frame=True)
    frame = bunch.frame.copy()
    frame[target_label] = [bunch.target_names[int(i)] for i in frame.pop("target")]
    return build_dataset(frame, DatasetSpec(name=title, target=target_label, source=source), source=title)


def _wine() -> Dataset:
    from sklearn.datasets import load_wine
    return _sklearn_dataset("Wine", load_wine, "variedad",
                            "UCI Machine Learning Repository, copia incluida en scikit-learn (load_wine)")


def _breast_cancer() -> Dataset:
    from sklearn.datasets import load_breast_cancer
    return _sklearn_dataset("Breast Cancer Wisconsin", load_breast_cancer, "diagnóstico",
                            "UCI Machine Learning Repository, copia incluida en scikit-learn (load_breast_cancer)")


def default_entries() -> list:
    return [
        DatasetEntry("play_tennis", "Play Tennis", "14 casos con atributos discretos: el ejemplo del libro de Quinlan.",
                     _play_tennis, "Quinlan (1986)", default_test_size=0.0),
        DatasetEntry("iris", "Iris", "150 flores, 4 medidas numéricas y 3 especies: el dataset de desarrollo.",
                     load_iris_dataset, "Fisher (1936)"),
        DatasetEntry("wine", "Wine", "178 vinos, 13 atributos químicos y 3 variedades.", _wine, "UCI vía scikit-learn"),
        DatasetEntry("breast_cancer", "Breast Cancer Wisconsin", "569 casos, 30 medidas numéricas y 2 diagnósticos.",
                     _breast_cancer, "UCI vía scikit-learn",
                     disclaimer="Herramienta educativa: no es un diagnóstico médico."),
    ]


# ------------------------------------------------------------------ el servicio
class LabService:
    def __init__(self, entries=None, seed: int = 0):
        self.seed = seed
        self.entries: dict = {}
        for entry in (entries if entries is not None else default_entries()):
            self.register_dataset(entry)
        self.dataset: Dataset | None = None
        self.entry: DatasetEntry | None = None
        self.tree: C45DecisionTree | None = None
        self.forest: RandomForestModel | None = None
        self.test_size = 0.3

    # ------------------------------------------------------------ registro y selección
    def register_dataset(self, entry: DatasetEntry) -> None:
        self.entries[entry.key] = entry

    def datasets(self) -> dict:
        return {"datasets": [{"key": e.key, "title": e.title, "description": e.description, "source": e.source,
                              "disclaimer": e.disclaimer} for e in self.entries.values()]}

    def select(self, key: str) -> dict:
        if key not in self.entries:
            raise ValueError(f"Dataset desconocido {key!r}. Disponibles: {sorted(self.entries)}")
        self.entry = self.entries[key]
        self.dataset = self.entry.loader()
        self.tree = self.forest = None
        self.test_size = self.entry.default_test_size
        self._split()
        return self.info()

    def load_csv(self, text: str, target: str, name: str = "Mi archivo", nominal=()) -> dict:
        """Carga un CSV subido desde la interfaz (texto) como dataset actual."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "datos.csv"
            path.write_text(text, encoding="utf-8")
            frame = read_table(path)
        spec = DatasetSpec(name=name, target=target, nominal=list(nominal), source="Archivo subido por el usuario")
        entry = DatasetEntry("subido", name, "Archivo subido por el usuario.",
                             lambda: build_dataset(frame, spec, source=name))
        self.register_dataset(entry)
        return self.select("subido")

    def _need_dataset(self) -> Dataset:
        if self.dataset is None:
            raise RuntimeError("Primero elegí un dataset.")
        return self.dataset

    def _split(self) -> None:
        ds = self._need_dataset()
        n = len(ds.y)
        if self.test_size and self.test_size > 0:
            split = make_splits(ds.y, scheme="holdout", n_repeats=1, test_size=self.test_size, seed=self.seed)[0]
            self.train_idx, self.test_idx = split.train, split.test
        else:
            self.train_idx, self.test_idx = np.arange(n), np.array([], dtype=int)
        self.tree = self.forest = None

    def set_split(self, test_size: float = 0.3, seed: int = 0) -> dict:
        if not 0 <= test_size < 0.9:
            raise ValueError("test_size debe estar entre 0 y 0.9")
        self.test_size, self.seed = float(test_size), int(seed)
        self._split()
        return {"n_train": int(len(self.train_idx)), "n_test": int(len(self.test_idx)), "test_size": self.test_size}

    # ------------------------------------------------------------ información del dataset
    def class_names(self) -> dict:
        return {c: n for c, n in self._need_dataset().class_names.items()}

    def info(self) -> dict:
        ds = self._need_dataset()
        report = ds.report
        names = self.class_names()
        classes = [{"value": _tidy(c), "name": str(names.get(c, c)), "count": int(report["class_counts"].get(c, 0))}
                   for c in ds.classes]
        head = ds.X.head(8)
        return {
            "key": self.entry.key, "title": self.entry.title, "description": self.entry.description,
            "source": self.entry.source, "disclaimer": self.entry.disclaimer,
            "n_rows": int(len(ds.X)), "n_features": int(ds.X.shape[1]), "classes": classes,
            "target": ds.target, "features": self._features(ds),
            "quality": {"duplicates": report["duplicates"], "rows_with_missing": report["rows_with_missing"],
                        "missing": report["missing"], "warnings": report["warnings"],
                        "constant_columns": report["constant_columns"], "id_like_columns": report["id_like_columns"]},
            "split": {"n_train": int(len(self.train_idx)), "n_test": int(len(self.test_idx)),
                      "test_size": self.test_size},
            "preview": head.astype(object).where(head.notna(), None).to_dict("records"),
            "decisions": ds.log.to_records(),
        }

    def _features(self, ds: Dataset) -> list:
        out = []
        for column in ds.X.columns:
            series, kind = ds.X[column], ds.feature_types[column]
            item = {"name": str(column), "kind": kind, "missing": int(series.isna().sum())}
            if kind == "numeric":
                values = pd.to_numeric(series, errors="coerce").dropna()
                edges = np.histogram_bin_edges(values, bins=12)
                by_class = {}
                for c in ds.classes:
                    in_class = pd.to_numeric(series[ds.y == c], errors="coerce").dropna()
                    by_class[str(_tidy(c))] = np.histogram(in_class, bins=edges)[0].tolist()
                item.update(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                            median=float(values.median()), q1=float(values.quantile(0.25)),
                            q3=float(values.quantile(0.75)), default=float(values.median()),
                            outliers=int(ds.report["outliers"].get(str(column), {}).get("count", 0)),
                            hist={"edges": edges.tolist(), "by_class": by_class})
            else:
                counts = series.dropna().map(_tidy).value_counts()
                values = sorted(counts.index, key=str)
                item.update(values=[_tidy(v) for v in values], counts={str(_tidy(v)): int(counts[v]) for v in values},
                            default=_tidy(counts.index[0]) if len(counts) else None)
            out.append(item)
        return out

    def decisions(self) -> dict:
        ds = self._need_dataset()
        return {"records": ds.log.to_records(), "markdown": ds.log.to_markdown()}

    # ------------------------------------------------------------ entrenamiento
    @staticmethod
    def _coerce(params: dict | None, allowed: tuple) -> dict:
        clean = {}
        for key, value in (params or {}).items():
            if key not in allowed:
                continue
            if key in INT_PARAMS and value is not None:
                value = int(value)
            elif key in ("min_samples_leaf", "min_gain", "confidence") and value is not None:
                value = float(value)
            clean[key] = value
        return clean

    def _train_test(self):
        ds = self._need_dataset()
        X, y = ds.X, ds.y.to_numpy()
        return (X.iloc[self.train_idx], y[self.train_idx], X.iloc[self.test_idx], y[self.test_idx])

    def _metrics(self, model, X, y) -> dict | None:
        if len(X) == 0:
            return None
        ds = self._need_dataset()
        proba = _align_proba(model.predict_proba(X), model.classes_, ds.classes)
        labels = np.asarray(ds.classes)
        pred = labels[proba.argmax(axis=1)]
        scores = evaluate(y, pred, proba, ds.classes)
        keep = {k: scores[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc") if k in scores}
        keep["confusion"] = confusion(y, pred, ds.classes).tolist()
        keep["n"] = int(len(X))
        return keep

    def train(self, params: dict | None = None) -> dict:
        """Entrena el árbol C4.5 propio con la partición actual y devuelve el árbol y sus métricas."""
        ds = self._need_dataset()
        X_train, y_train, X_test, y_test = self._train_test()
        clean = self._coerce(params, DT_PARAMS)
        tree = C45DecisionTree(**clean)
        start = time.perf_counter()
        tree.fit(X_train, y_train, feature_types=ds.feature_types)
        fit_time = time.perf_counter() - start
        self.tree = tree
        names = self.class_names()
        order = np.argsort(-tree.feature_importances_)
        return {"params": {**C45DecisionTree().get_params(), **clean}, "tree": tree.to_dict(names),
                "fit_time": fit_time, "text": tree.export_text(), "dot": tree.to_dot(names),
                "metrics": {"train": self._metrics(tree, X_train, y_train),
                            "test": self._metrics(tree, X_test, y_test)},
                "importances": [{"feature": tree.feature_names_[i], "value": float(tree.feature_importances_[i])}
                                for i in order if tree.feature_importances_[i] > 0],
                "classes": [{"value": _tidy(c), "name": str(names.get(c, c))} for c in ds.classes]}

    def train_forest(self, params: dict | None = None) -> dict:
        ds = self._need_dataset()
        X_train, y_train, X_test, y_test = self._train_test()
        clean = {**FOREST_PRESET, **self._coerce(params, RF_PARAMS)}
        forest = RandomForestModel(**clean)
        start = time.perf_counter()
        forest.fit(X_train, y_train, feature_types=ds.feature_types)
        fit_time = time.perf_counter() - start
        self.forest = forest
        order = sorted(forest.feature_importances_.items(), key=lambda kv: -kv[1])
        return {"params": forest.get_params(), "fit_time": fit_time,
                "metrics": {"train": self._metrics(forest, X_train, y_train),
                            "test": self._metrics(forest, X_test, y_test)},
                "importances": [{"feature": f, "value": float(v)} for f, v in order if v > 0]}

    # ------------------------------------------------------------ predecir un caso
    def _clean_row(self, row: dict) -> dict:
        ds = self._need_dataset()
        clean = {}
        for column in ds.X.columns:
            value = (row or {}).get(str(column))
            if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
                clean[str(column)] = None
            elif ds.feature_types[column] == "numeric":
                clean[str(column)] = float(value)
            else:
                clean[str(column)] = value
        return clean

    def predict(self, row: dict) -> dict:
        """Qué dicen el árbol y el bosque de UN caso; con el camino del árbol (los ids son los de `train`)."""
        ds = self._need_dataset()
        if self.tree is None:
            self.train({})
        if self.forest is None:
            self.train_forest({})
        clean = self._clean_row(row)
        frame = pd.DataFrame([clean])
        names = self.class_names()
        labels = [str(names.get(c, c)) for c in ds.classes]
        out = {"classes": labels, "row": clean, "missing": [k for k, v in clean.items() if v is None]}
        for key, model in (("dt", self.tree), ("rf", self.forest)):
            proba = _align_proba(model.predict_proba(frame), model.classes_, ds.classes)[0]
            out[key] = {"proba": proba.tolist(), "label": labels[int(np.argmax(proba))]}
        out["dt"]["rules"] = self.tree.explain(clean)["rules"]
        out["dt"]["trace"] = self.tree.trace(clean)
        return out

    # ------------------------------------------------------------ comparar y experimentar
    def _splits(self, n_splits: int, n_repeats: int, seed: int):
        ds = self._need_dataset()
        k = max(2, min(int(n_splits), max_stratified_splits(ds.y)))
        return make_splits(ds.y, n_splits=k, n_repeats=max(1, min(int(n_repeats), 10)), seed=int(seed)), k

    def compare(self, dt_params=None, rf_params=None, n_splits: int = 5, n_repeats: int = 3, seed: int = 0) -> dict:
        """Árbol propio contra Random Forest sobre las MISMAS particiones (validación cruzada repetida)."""
        ds = self._need_dataset()
        splits, k = self._splits(n_splits, n_repeats, seed)
        dt = self._coerce(dt_params, DT_PARAMS)
        rf = {**FOREST_PRESET, **self._coerce(rf_params, RF_PARAMS)}
        results = run_experiment(ds, {"DT": ("dt", dt), "RF": ("rf", rf)}, splits)
        out = {"n_splits": k, "n_repeats": len(splits) // k, "n_partitions": len(splits),
               "params": {"DT": dt, "RF": rf}, "models": {}}
        for model in ("DT", "RF"):
            part = results[results["model"] == model]
            stats = {}
            for column, label in (("test_accuracy", "accuracy"), ("test_balanced_accuracy", "balanced_accuracy"),
                                  ("test_macro_f1", "macro_f1"), ("test_roc_auc", "roc_auc"), ("fit_time", "fit_time"),
                                  ("predict_time", "predict_time"), ("train_accuracy", "train_accuracy")):
                if column in part:
                    stats[label] = {"mean": float(part[column].mean()), "std": float(part[column].std(ddof=1))}
            gap = part["train_accuracy"] - part["test_accuracy"]
            stats["gap"] = {"mean": float(gap.mean()), "std": float(gap.std(ddof=1))}
            matrix = np.sum([np.array(json.loads(text)) for text in part["confusion"]], axis=0)
            out["models"][model] = {"stats": stats, "per_split": part["test_accuracy"].tolist(),
                                    "per_split_f1": part["test_macro_f1"].tolist(), "confusion": matrix.tolist()}
        out["paired"] = {"accuracy": paired_comparison(results, "RF", "DT", "accuracy"),
                         "macro_f1": paired_comparison(results, "RF", "DT", "macro_f1")}
        names = self.class_names()
        out["classes"] = [str(names.get(c, c)) for c in ds.classes]
        out["decision"] = next((d for d in reversed(ds.log.to_records()) if d["id"] == "D-07"), None)
        return out

    def sweep(self, kind: str, param: str, values, base_params=None, metric: str = "macro_f1",
              n_splits: int = 5, n_repeats: int = 2, seed: int = 0) -> dict:
        """Cómo cambia el rendimiento al mover UN parámetro (los demás quedan fijos)."""
        ds = self._need_dataset()
        if kind not in SWEEPABLE or param not in SWEEPABLE[kind]:
            raise ValueError(f"Parámetro no permitido: {kind}/{param}. Permitidos: {SWEEPABLE}")
        splits, _ = self._splits(n_splits, n_repeats, seed)
        allowed = DT_PARAMS if kind == "dt" else RF_PARAMS
        base = self._coerce(base_params, allowed)
        if kind == "rf":
            base = {**FOREST_PRESET, **base}
            base.pop(param, None)
        if kind == "dt" and param == "confidence":
            base["prune"] = True                         # la confianza solo actúa si se poda
        coerced = [self._coerce({param: v}, allowed)[param] for v in values]
        table = parameter_sweep(ds, kind, param, coerced, splits, base_params=base, metric=metric)
        rows = [{"value": _tidy(r["value"]), "test_mean": float(r[f"test_{metric}_mean"]),
                 "test_std": float(r[f"test_{metric}_std"]), "train_mean": float(r[f"train_{metric}_mean"]),
                 "gap": float(r["gap"]), "fit_time": float(r["fit_time_mean"])} for _, r in table.iterrows()]
        best = max(rows, key=lambda r: r["test_mean"])
        return {"kind": kind, "param": param, "metric": metric, "rows": rows, "best": best["value"],
                "n_partitions": len(splits), "base": base}

    def robustness(self, fractions=(0, 0.1, 0.2, 0.3, 0.4), seeds=(0,), n_splits: int = 5, n_repeats: int = 1,
                   dt_params=None, rf_params=None, metric: str = "macro_f1") -> dict:
        """Borra al azar una fracción de los valores (en train y en test) y mide cuánto cae cada modelo."""
        ds = self._need_dataset()
        splits, _ = self._splits(n_splits, n_repeats, 0)
        dt = self._coerce(dt_params, DT_PARAMS)
        rf = {**FOREST_PRESET, **self._coerce(rf_params, RF_PARAMS)}
        results = missing_robustness(ds, {"DT": ("dt", dt), "RF": ("rf", rf)}, [float(f) for f in fractions], splits,
                                     seeds=[int(s) for s in seeds])
        table = robustness_table(results, metric)
        spread = results.groupby(["missing_fraction", "model"])[f"test_{metric}"].std(ddof=1).unstack()
        rows = [{"fraction": float(f), "DT": float(table.loc[f, "DT"]), "RF": float(table.loc[f, "RF"]),
                 "DT_std": float(spread.loc[f, "DT"]), "RF_std": float(spread.loc[f, "RF"])} for f in table.index]
        return {"metric": metric, "rows": rows, "n_partitions": len(splits), "seeds": [int(s) for s in seeds]}

    # ------------------------------------------------------------ punto de entrada para la interfaz
    def call(self, method: str, params: dict | None = None):
        """Despacha un método permitido y devuelve el resultado listo para JSON."""
        if method not in METHODS:
            raise ValueError(f"Método desconocido: {method!r}")
        return to_jsonable(getattr(self, method)(**(params or {})))

    def handle(self, message: dict) -> dict:
        """Un pedido de la interfaz {id, method, params} -> {id, ok, result} o {id, ok: False, error}.

        Es el protocolo que comparten el widget de Colab y el servidor de desarrollo: la interfaz no sabe
        con cuál está hablando. Un error nunca tumba la sesión: vuelve como texto para mostrarlo en pantalla.
        """
        request_id = message.get("id") if isinstance(message, dict) else None
        try:
            if not isinstance(message, dict):
                raise ValueError("El pedido debe ser un objeto {id, method, params}")
            return {"id": request_id, "ok": True, "result": self.call(message.get("method", ""), message.get("params"))}
        except Exception as error:                                   # se informa en la interfaz, no se propaga
            return {"id": request_id, "ok": False, "error": f"{type(error).__name__}: {error}"}
