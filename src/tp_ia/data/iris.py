"""
Iris (Fisher, 1936): dataset de DESARROLLO y validación del pipeline.

GUÍA DE LECTURA
---------------
150 flores, 4 atributos numéricos, 3 especies (50 de cada una), sin faltantes y con 1 fila duplicada.
Sirve para probar todo el camino (cargar -> particionar -> entrenar DT y RF -> métricas) antes de usar
el dataset real. Se guarda como CSV en data/iris/ y se lee con el mismo cargador que cualquier otro archivo.
"""
from __future__ import annotations

from pathlib import Path

from .loader import Dataset, load_dataset
from .schema import DatasetSpec

CITATION = ("Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. "
            "Annals of Eugenics, 7(2), 179-188.")

IRIS_SPEC = DatasetSpec(
    name="Iris", target="species",
    source="Copia incluida en scikit-learn (datasets.load_iris), guardada como CSV en data/iris/",
    citation=CITATION)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "iris"


def load_iris_dataset(data_dir=None, **kwargs) -> Dataset:
    path = (Path(data_dir) if data_dir else default_data_dir()) / "iris.csv"
    return load_dataset(path, IRIS_SPEC, **kwargs)
