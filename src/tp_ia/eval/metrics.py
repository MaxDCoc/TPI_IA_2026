"""
Métricas de clasificación (binaria y multiclase) con las precauciones que importan con pocos datos.

GUÍA DE LECTURA
---------------
  - accuracy sola engaña con clases desbalanceadas: se reportan también balanced_accuracy, macro-F1,
    kappa y MCC. Las métricas "macro" promedian las clases por igual (una clase rara pesa lo mismo).
  - Por clase: precision, recall, F1 y soporte. En binario: sensibilidad (recall de la clase 1) y
    especificidad (recall de la clase 0).
  - ROC-AUC uno-contra-resto (OvR) promediado, calculado SOLO sobre las clases que tienen positivos y
    negativos en el test (en un pliegue chico puede faltar una clase; en ese caso se la omite en vez de fallar).
  - Los pesos de las métricas "macro" se calculan sobre las clases presentes en y_true o y_pred.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn import metrics as skm


def confusion(y_true, y_pred, classes):
    """Matriz de confusión con filas = clase real y columnas = predicha, en el orden de `classes`."""
    return skm.confusion_matrix(y_true, y_pred, labels=list(classes))


def ovr_auc(y_true, proba, classes) -> float:
    y_true = np.asarray(y_true)
    aucs = []
    for k, c in enumerate(classes):
        positive = y_true == c
        if 0 < positive.sum() < len(y_true):
            aucs.append(skm.roc_auc_score(positive, proba[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate(y_true, y_pred, proba=None, classes=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = list(classes) if classes is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    seen = set(y_true.tolist()) | set(y_pred.tolist())
    present = [c for c in classes if c in seen]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {
            "accuracy": skm.accuracy_score(y_true, y_pred),
            "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
            "macro_f1": skm.f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
            "weighted_f1": skm.f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0),
            "macro_precision": skm.precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
            "macro_recall": skm.recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
            "kappa": skm.cohen_kappa_score(y_true, y_pred, labels=present),
            "mcc": skm.matthews_corrcoef(y_true, y_pred),
        }
        precision, recall, f1, support = skm.precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0)
    for i, c in enumerate(classes):
        out[f"precision_{c}"], out[f"recall_{c}"] = precision[i], recall[i]
        out[f"f1_{c}"], out[f"support_{c}"] = f1[i], support[i]
    if len(classes) == 2:
        out["sensitivity"], out["specificity"] = recall[1], recall[0]
    if proba is not None:
        out["roc_auc"] = ovr_auc(y_true, np.asarray(proba), classes)
    return {k: float(v) for k, v in out.items()}
