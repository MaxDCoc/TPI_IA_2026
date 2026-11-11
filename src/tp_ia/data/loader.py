"""
Cargador de datasets AGNÓSTICO: lee casi cualquier CSV y deja registrado qué decidió.

GUÍA DE LECTURA
---------------
Dos etapas, a propósito separadas:
  1. read_table(): archivo -> DataFrame. Adivina separador (, ; tab |), coma decimal y codificación
     (utf-8 o latin-1), recorta espacios y convierte los textos de "falta dato" (?, NA, vacío, ...)
     en faltantes reales. Una columna se vuelve numérica solo si TODOS sus valores lo son.
  2. build_dataset(): DataFrame + DatasetSpec -> Dataset (X, y, tipos, auditoría, registro de
     decisiones). Aplica el escenario de objetivo, los ceros-faltantes, la política de faltantes
     y de duplicados.

El dataset de la cátedra, uno subido desde la interfaz o los archivos de UCI pasan por el mismo camino.
"""
from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .decisions import DecisionLog
from .quality import audit
from .schema import DEFAULT_MISSING_TOKENS, DatasetSpec

_SEP_CANDIDATES = (",", ";", "\t", "|")
MISSING_POLICIES = ("keep", "drop_rows")
DUPLICATE_POLICIES = ("keep", "drop")


# ------------------------------------------------------------------ etapa 1: archivo -> DataFrame
def _read_sample(path, nbytes: int = 65536):
    raw = Path(path).read_bytes()[:nbytes]
    try:  # decodificador incremental: tolera que la muestra corte un carácter a la mitad
        return codecs.getincrementaldecoder("utf-8-sig")().decode(raw, final=False), "utf-8-sig"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def _detect_sep(sample: str) -> str:
    """El separador es el que aparece la MISMA cantidad de veces en todas las primeras líneas."""
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    if not lines:
        return ","
    best, best_count = ",", 0
    for sep in _SEP_CANDIDATES:
        counts = [line.count(sep) for line in lines]
        if counts[0] > 0 and len(set(counts)) == 1 and counts[0] > best_count:
            best, best_count = sep, counts[0]
    if best_count == 0:  # ninguno es consistente: nos quedamos con el más frecuente
        totals = {sep: sum(line.count(sep) for line in lines) for sep in _SEP_CANDIDATES}
        best = max(totals, key=totals.get) if max(totals.values()) > 0 else ","
    return best


def _detect_decimal(sample: str, sep: str) -> str:
    if sep != "," and re.search(r"\d,\d", sample) and not re.search(r"\d\.\d", sample):
        return ","
    return "."


def _convert_column(series: pd.Series, tokens: set, decimal: str) -> pd.Series:
    # Se trabaja con listas de Python (None = faltante): pandas 3 representa los faltantes de una columna
    # de texto como NaN y pandas 2 como None; así el resultado no depende de la versión.
    values = [None if (pd.isna(v) or str(v).strip() in tokens) else str(v).strip() for v in series]
    numeric_text = values if decimal == "." else [None if v is None else v.replace(",", ".") for v in values]
    try:
        return pd.Series(pd.to_numeric(pd.Series(numeric_text, dtype=object)).to_numpy(), index=series.index)
    except (ValueError, TypeError):
        return pd.Series(values, index=series.index, dtype=object)   # hay texto: queda como texto (nominal)


def read_table(path, header: bool = True, columns=None, missing_tokens=DEFAULT_MISSING_TOKENS,
               sep=None, decimal=None, encoding=None) -> pd.DataFrame:
    """Lee un CSV/TSV 'sucio' y lo devuelve con faltantes reales (NaN/None) y tipos razonables."""
    sample, detected = _read_sample(path)
    encoding = encoding or detected
    sep = sep or _detect_sep(sample)
    decimal = decimal or _detect_decimal(sample, sep)

    def read(enc):
        return pd.read_csv(path, sep=sep, encoding=enc, header=0 if header else None,
                           names=None if header else columns, dtype=str, keep_default_na=False,
                           skipinitialspace=True)
    try:
        raw = read(encoding)
    except UnicodeDecodeError:   # había bytes inválidos más allá de la muestra
        raw = read("latin-1")
    raw.columns = [str(c).strip() for c in raw.columns]
    tokens = {t.strip() for t in missing_tokens}
    return pd.DataFrame({c: _convert_column(raw[c], tokens, decimal) for c in raw.columns})


# ------------------------------------------------------------------ etapa 2: DataFrame -> Dataset
@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    feature_types: dict
    target: str
    scenario: str | None
    class_names: dict
    classes: list
    log: DecisionLog
    report: dict
    source: str = ""
    descriptions: dict = field(default_factory=dict)   # columna -> qué mide (para la interfaz)

    def summary(self) -> dict:
        return {"name": self.name, "scenario": self.scenario, "n_rows": len(self.X),
                "n_features": self.X.shape[1], "classes": self.report["class_counts"],
                "missing_cells": int(self.X.isna().sum().sum()), "duplicates": self.report["duplicates"],
                "warnings": len(self.report["warnings"])}


def _resolve_types(X: pd.DataFrame, nominal) -> dict:
    unknown = set(nominal) - set(map(str, X.columns))
    if unknown:
        raise ValueError(f"Columnas nominales que no existen en los datos: {sorted(unknown)}")
    types = {}
    for column in X.columns:
        series = X[column]
        if str(column) in nominal:
            types[str(column)] = "nominal"
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            types[str(column)] = "numeric"
        else:
            types[str(column)] = "nominal"
    return types


def build_dataset(df: pd.DataFrame, spec: DatasetSpec, scenario=None, missing_policy="keep",
                  duplicates="keep", log=None, source="") -> Dataset:
    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"missing_policy debe ser uno de {MISSING_POLICIES}, no {missing_policy!r}")
    if duplicates not in DUPLICATE_POLICIES:
        raise ValueError(f"duplicates debe ser uno de {DUPLICATE_POLICIES}, no {duplicates!r}")
    log = log if log is not None else DecisionLog()
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    if spec.target not in df.columns:
        raise ValueError(f"La columna objetivo {spec.target!r} no está en los datos. Columnas: {list(df.columns)}")

    file_name = Path(str(source)).name
    chosen = f"usar el dataset «{spec.name}»" + (f" (archivo {file_name})" if "." in file_name else "")
    why = [spec.purpose, f"Los datos provienen de {spec.source}." if spec.source else ""]
    log.add("D-01", "Fuente de los datos", chosen, rationale=" ".join(p for p in why if p),
            evidence={"filas": len(df), "columnas": df.shape[1], "referencia": spec.citation or "—"})

    if spec.drop:
        df = df.drop(columns=[c for c in spec.drop if c in df.columns])

    # --- códigos con nombre: 4 -> "sin síntomas" (se registra en D-05) ---
    translated = {}
    for column, labels in spec.value_labels.items():
        if column in df.columns:
            df[column] = df[column].map(lambda v, labels=labels: _label(v, labels))
            translated[column] = len(labels)

    # --- D-03: ceros que en realidad son faltantes ---
    for column in spec.zero_is_missing:
        if column in df.columns:
            zeros = df[column] == 0
            if int(zeros.sum()):
                df[column] = df[column].astype(float).mask(zeros)
                log.add("D-03", f"Ceros en «{column}»", f"tratar los ceros de «{column}» como valores faltantes",
                        rationale="En esta variable, un 0 es un valor imposible. Por eso se interpreta como un dato faltante que se registró como 0.",
                        alternatives=["considerarlos valores reales"], evidence={"filas afectadas": int(zeros.sum())})

    # --- D-02: objetivo y escenario ---
    y_raw = df[spec.target]
    has_target = y_raw.notna()
    if (~has_target).any():
        log.add("D-02", "Filas sin objetivo", "eliminar las filas que no tienen la clase", rationale="No se puede entrenar ni evaluar un modelo con casos que no tienen la clase.",
                evidence={"filas eliminadas": int((~has_target).sum())})
        df, y_raw = df[has_target], y_raw[has_target]
    scenario_obj = None
    y = y_raw
    if scenario is not None:
        scenario_obj = spec.scenarios.get(scenario)
        if scenario_obj is None:
            raise ValueError(f"Escenario {scenario!r} desconocido. Disponibles: {sorted(spec.scenarios)}")
        if scenario_obj.mapping is not None:
            unknown = sorted(set(y_raw.unique()) - set(scenario_obj.mapping), key=str)
            if unknown:
                raise ValueError(f"Valores del objetivo sin mapeo en el escenario {scenario!r}: {unknown}")
            y = y_raw.map(scenario_obj.mapping)
    X = df.drop(columns=[spec.target])

    # --- D-05: tipos de atributo ---
    feature_types = _resolve_types(X, set(spec.nominal))
    log.add("D-05", "Tipos de atributo", "dividir los atributos nominales por valor y los numéricos por umbral",
            rationale="Un código sin orden, como 1, 2, 3 o 4 para distintas categorías, no debe tratarse como un número. "
                      "Si se tratara así, el árbol haría cortes sin sentido, como «categoría ≤ 2».",
            alternatives=["deducir el tipo de cada atributo a partir del tipo de dato"],
            evidence={"atributos nominales": [c for c, t in feature_types.items() if t == "nominal"],
                      "atributos numéricos": [c for c, t in feature_types.items() if t == "numeric"],
                      **({"códigos reemplazados por su significado": list(translated)} if translated else {})})

    # --- D-03: política de faltantes ---
    n_missing_rows = int(X.isna().any(axis=1).sum())
    per_column = {c: int(k) for c, k in X.isna().sum().items() if k}
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    if missing_policy == "drop_rows":
        keep = ~X.isna().any(axis=1)
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        log.add("D-03", "Valores faltantes", "eliminar las filas con algún valor faltante",
                rationale="Es la opción más simple, pero se pierden datos y los resultados pueden sesgarse si los "
                          "faltantes no aparecen al azar.",
                alternatives=["conservarlas, ya que el árbol puede repartir esos casos entre las ramas y el bosque puede "
                              "completar los datos usando solo el entrenamiento"],
                evidence={"filas con faltantes": n_missing_rows, "filas eliminadas": int((~keep).sum()),
                          "faltantes por columna": per_column})
    else:
        log.add("D-03", "Valores faltantes", "conservar las filas con valores faltantes",
                rationale="El árbol propio maneja los faltantes como C4.5, repartiendo el caso entre las ramas. El bosque, en "
                          "cambio, completa los datos faltantes con valores calculados solo con el conjunto de entrenamiento, "
                          "para no usar información de las pruebas.",
                alternatives=["eliminar las filas con valores faltantes"],
                evidence={"filas con faltantes": n_missing_rows, "faltantes por columna": per_column})

    # --- D-04: duplicados ---
    both = pd.concat([X, y.rename("__objetivo__")], axis=1)
    n_dup = int(both.duplicated().sum())
    if duplicates == "drop" and n_dup:
        keep = ~both.duplicated()
        X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    dropped = duplicates == "drop"
    log.add("D-04", "Filas duplicadas", "eliminar las filas duplicadas" if dropped else "conservar las filas duplicadas",
            rationale=("Si una fila repetida queda a la vez en el entrenamiento y en la prueba, el modelo ya «vio» ese caso "
                       "y las métricas salen mejores de lo que realmente son."
                       if dropped else
                       "Pueden ser casos reales que se repiten, así que borrarlos cambiaría los datos originales. Hay que "
                       "tener en cuenta que, si una copia queda en el entrenamiento y otra en la prueba, las métricas "
                       "pueden salir algo mejores de lo que realmente son."),
            alternatives=["conservarlas" if dropped else "eliminarlas", "mantener las copias juntas al dividir los datos"],
            evidence={"filas duplicadas": n_dup})

    classes = sorted(y.unique().tolist())
    report = audit(X, y, feature_types)
    log.add("D-06", "Valores extremos", "conservar los valores extremos",
            rationale="Un valor extremo pero posible suele ser una medición real, y borrarlo o recortarlo sin conocer el "
                      "dominio sería inventar datos. Además, los árboles dividen por umbrales, así que un valor muy "
                      "grande cae en la misma rama que uno grande y no distorsiona el modelo. Los valores imposibles, "
                      "como un 0 donde no puede haberlo, no se consideran extremos sino faltantes, y se tratan en la "
                      "decisión sobre valores faltantes.",
            alternatives=["eliminar las filas con valores extremos", "recortarlos hasta un valor límite (winsorización)"],
            evidence={"criterio": f"se considera extremo un valor fuera de [Q1 - {report['outlier_factor']}·IQR, "
                                  f"Q3 + {report['outlier_factor']}·IQR] (regla de Tukey)",
                      "valores extremos": sum(i["count"] for i in report["outliers"].values()),
                      "por columna": {c: i["count"] for c, i in report["outliers"].items()}})
    log.add("D-02", "Variable objetivo",
            (scenario_obj.goal or f"usar el escenario «{scenario}» como variable objetivo") if scenario
            else f"predecir la columna «{spec.target}» con sus clases originales",
            rationale=scenario_obj.description if scenario_obj
            else "Las clases originales ya sirven para el problema, así que no hace falta agruparlas ni transformarlas.",
            alternatives=[f"usar el escenario «{s}»" for s in (sorted(spec.scenarios) if spec.scenarios else []) if s != scenario],
            evidence={"casos por clase": report["class_counts"]})
    return Dataset(name=spec.name, X=X, y=y, feature_types=feature_types, target=spec.target, scenario=scenario,
                   class_names=dict(scenario_obj.class_names or {}) if scenario_obj else {}, classes=classes,
                   log=log, report=report, source=str(source),
                   descriptions={c: d for c, d in spec.descriptions.items() if c in X.columns})


def _label(value, labels: dict):
    """Reemplaza un código por su significado. 4.0 busca la clave 4; un código desconocido o un faltante quedan igual."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    key = int(value) if isinstance(value, float) and value.is_integer() else value
    return labels.get(key, value)


def load_dataset(path, spec: DatasetSpec, scenario=None, missing_policy="keep", duplicates="keep",
                 log=None) -> Dataset:
    """Archivo + especificación -> Dataset listo para entrenar."""
    df = read_table(path, header=spec.header, columns=spec.columns, missing_tokens=spec.missing_tokens)
    return build_dataset(df, spec, scenario, missing_policy, duplicates, log, source=str(path))


def dataset_from_frame(df: pd.DataFrame, target: str, nominal=None, name="datos", **kwargs) -> Dataset:
    """Atajo para un DataFrame que ya está en memoria (por ejemplo, un archivo subido desde la interfaz)."""
    spec = DatasetSpec(name=name, target=target, nominal=list(nominal or []))
    return build_dataset(df, spec, **kwargs)
