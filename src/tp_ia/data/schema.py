"""
Descripción DECLARATIVA de un dataset: qué columnas hay, cuál es el objetivo, cuáles son nominales,
qué valores significan "faltante" y qué escenarios de objetivo se pueden armar.

GUÍA DE LECTURA
---------------
Cuando llegue un dataset nuevo (el de la cátedra, uno subido desde la interfaz) no se toca el código:
se escribe un DatasetSpec y listo. El cargador (loader.py) hace el resto y deja registrado en un
DecisionLog cada decisión que tomó (qué se trató como faltante, qué columnas son nominales, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Textos que, en un CSV, suelen querer decir "no hay dato".
DEFAULT_MISSING_TOKENS = ("?", "", "NA", "N/A", "NaN", "nan", "null", "NULL", "None", "-", "--")


@dataclass(frozen=True)
class Scenario:
    """Una forma de definir la variable objetivo a partir de la columna original.

    mapping:      valor original -> clase nueva (None = se deja la columna como viene).
    class_names:  clase nueva -> nombre legible para tablas y gráficos.
    """

    name: str
    mapping: dict | None = None
    class_names: dict | None = None
    description: str = ""


@dataclass
class DatasetSpec:
    name: str
    target: str                                   # nombre de la columna objetivo
    columns: list | None = None                   # nombres, si el archivo NO trae encabezado
    header: bool = True                           # ¿la primera fila es el encabezado?
    nominal: list = field(default_factory=list)   # columnas nominales (códigos sin orden)
    zero_is_missing: list = field(default_factory=list)  # columnas donde un 0 es un faltante disfrazado
    drop: list = field(default_factory=list)      # columnas a descartar (IDs, nombres, ...)
    missing_tokens: tuple = DEFAULT_MISSING_TOKENS
    scenarios: dict = field(default_factory=dict)  # nombre -> Scenario
    source: str = ""
    citation: str = ""
