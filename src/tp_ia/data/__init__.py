"""Datos: cargador agnóstico, auditoría, registro de decisiones, Iris y fixtures de juguete."""
from .decisions import DecisionLog
from .fixtures import liu_loans, play_tennis
from .iris import load_iris_dataset
from .loader import Dataset, build_dataset, dataset_from_frame, load_dataset, read_table
from .quality import audit
from .schema import DatasetSpec, Scenario

__all__ = ["Dataset", "DatasetSpec", "DecisionLog", "Scenario", "audit", "build_dataset",
           "dataset_from_frame", "liu_loans", "load_dataset", "load_iris_dataset", "play_tennis", "read_table"]
