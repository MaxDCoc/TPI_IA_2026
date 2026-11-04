"""
Modelos con una interfaz común para compararlos: fit(X, y, feature_types) / predict / predict_proba.

GUÍA DE LECTURA
---------------
  - El árbol propio (C45DecisionTree) ya tiene esa interfaz.
  - RandomForestModel envuelve el RandomForestClassifier de scikit-learn (que el enunciado permite) con
    el preprocesamiento que necesita, ajustado SOLO con el train de cada pliegue (sin filtrar el test):
        numéricos -> imputación por mediana
        nominales -> imputación por la moda + one-hot (los valores nuevos en test se ignoran)
    Así el RF no se beneficia de información del test y el DT y el RF ven los mismos datos.
  - Diferencia a documentar en el informe: el RF de sklearn usa árboles binarios tipo CART; nuestro DT
    usa divisiones multirrama C4.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..c45 import C45DecisionTree

_RF_PARAMS = ("n_estimators", "max_depth", "max_features", "min_samples_split", "min_samples_leaf",
              "criterion", "class_weight", "random_state")


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt", min_samples_split=2,
                 min_samples_leaf=1, criterion="gini", class_weight=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.random_state = random_state

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in _RF_PARAMS}

    def set_params(self, **params):
        for name, value in params.items():
            if name not in _RF_PARAMS:
                raise ValueError(f"Parámetro desconocido: {name!r}. Válidos: {_RF_PARAMS}")
            setattr(self, name, value)
        return self

    def _prepare(self, X):
        frame = X[self.feature_names_].copy().reset_index(drop=True)
        for column in self.numeric_:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in self.nominal_:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
        return frame

    def fit(self, X, y, feature_types=None, sample_weight=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        types = feature_types or {}
        self.feature_names_ = [str(c) for c in X.columns]
        X = X.copy()
        X.columns = self.feature_names_
        self.nominal_ = [c for c in self.feature_names_ if types.get(c) == "nominal" or
                         (c not in types and not pd.api.types.is_numeric_dtype(X[c]))]
        self.numeric_ = [c for c in self.feature_names_ if c not in self.nominal_]
        preprocess = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), self.numeric_),
            ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), self.nominal_),
        ])
        forest = RandomForestClassifier(**self.get_params())
        self.pipeline_ = Pipeline([("pre", preprocess), ("rf", forest)])
        fit_params = {} if sample_weight is None else {"rf__sample_weight": np.asarray(sample_weight)}
        self.pipeline_.fit(self._prepare(X), np.asarray(y), **fit_params)
        self.classes_ = self.pipeline_.classes_
        self.feature_importances_ = self._importances_by_feature()
        return self

    def _importances_by_feature(self) -> dict:
        """Importancia por atributo ORIGINAL: se suman las columnas one-hot de cada nominal."""
        raw = self.pipeline_.named_steps["rf"].feature_importances_
        result = dict(zip(self.numeric_, raw[:len(self.numeric_)]))
        position = len(self.numeric_)
        if self.nominal_:
            encoder = self.pipeline_.named_steps["pre"].named_transformers_["nom"].named_steps["oh"]
            for column, categories in zip(self.nominal_, encoder.categories_):
                result[column] = raw[position:position + len(categories)].sum()
                position += len(categories)
        return {c: float(result[c]) for c in self.feature_names_}

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(self._prepare(X))

    def predict(self, X):
        return self.pipeline_.predict(self._prepare(X))


def make_model(kind: str, **params):
    """'dt' = árbol C4.5 propio; 'rf' = Random Forest de scikit-learn."""
    if kind == "dt":
        return C45DecisionTree(**params)
    if kind == "rf":
        return RandomForestModel(**params)
    raise ValueError(f"Modelo desconocido {kind!r}: use 'dt' o 'rf'.")
