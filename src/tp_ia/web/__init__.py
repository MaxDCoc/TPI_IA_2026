"""Interfaz web del laboratorio: servicio en Python (service.py) y widget de Colab (widget.py, requiere anywidget)."""
from .service import DatasetEntry, LabService, default_entries, to_jsonable

__all__ = ["DatasetEntry", "LabService", "default_entries", "to_jsonable"]
