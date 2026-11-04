"""
Registro de decisiones de datos.

GUÍA DE LECTURA
---------------
La cátedra pide que las decisiones técnicas queden anotadas en el informe. En vez de reconstruirlas
de memoria, cada paso del cargador las va registrando solo: qué se decidió, por qué, qué alternativas
había y con qué evidencia (conteos reales). Se puede mostrar en la interfaz o pegar en el informe.

Numeración (la misma del plan): D-01 fuente, D-02 objetivo, D-03 faltantes, D-04 duplicados,
D-05 tipos de atributo, D-06 valores extremos, D-07 evaluación.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class Decision:
    id: str
    title: str
    chosen: str
    rationale: str = ""
    alternatives: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    when: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class DecisionLog:
    def __init__(self):
        self._items: list[Decision] = []

    def add(self, id, title, chosen, rationale="", alternatives=None, evidence=None) -> Decision:
        decision = Decision(id, title, chosen, rationale, list(alternatives or []), dict(evidence or {}))
        self._items.append(decision)
        return decision

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def ids(self) -> list:
        return [d.id for d in self._items]

    def to_records(self) -> list:
        return [asdict(d) for d in self._items]

    def to_markdown(self) -> str:
        lines = ["# Registro de decisiones de datos", ""]
        for d in self._items:
            lines += [f"## {d.id} · {d.title}", f"- **Decisión:** {d.chosen}"]
            if d.rationale:
                lines.append(f"- **Por qué:** {d.rationale}")
            if d.alternatives:
                lines.append("- **Alternativas:** " + "; ".join(d.alternatives))
            if d.evidence:
                lines.append("- **Evidencia:** " + ", ".join(f"{k} = {v}" for k, v in d.evidence.items()))
            lines += [f"- _Registrada: {d.when}_", ""]
        return "\n".join(lines)
