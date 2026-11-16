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

    def replace(self, id, title, chosen, rationale="", alternatives=None, evidence=None) -> Decision:
        """Como add(), pero descarta antes lo registrado con ese id (para decisiones que se re-registran)."""
        self._items = [d for d in self._items if d.id != id]
        return self.add(id, title, chosen, rationale, alternatives, evidence)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def ids(self) -> list:
        return [d.id for d in self._items]

    def _ordered(self) -> list:
        """Por número de decisión (D-01, D-02, ...); a igual id se conserva el orden en que se registraron."""
        return sorted(self._items, key=lambda d: d.id)

    def to_records(self) -> list:
        return [asdict(d) for d in self._ordered()]

    def to_markdown(self) -> str:
        """El registro como texto para el informe: cada decisión numerada en palabras, con su motivo y sus datos."""
        lines = ["# Registro de decisiones sobre los datos", "",
                 "Mientras se cargan y se evalúan los datos, el programa anota cada decisión que se tomó, junto con el "
                 "motivo y los números que la respaldan.", ""]
        for number, d in enumerate(self._ordered(), start=1):
            lines += [f"## {decision_label(number)}: {d.chosen}", f"*Tema: {d.title} (código {d.id})*", ""]
            if d.rationale:
                lines += [f"**Por qué:** {d.rationale}", ""]
            if d.alternatives:
                lines += [f"**Otras opciones posibles:** {join_or(d.alternatives)}.", ""]
            if d.evidence:
                lines.append("**En qué se basa:**")
                lines += [f"- {k[:1].upper()}{k[1:]}: {_friendly(v)}" for k, v in d.evidence.items()]
                lines.append("")
            lines += [f"<sub>Se registró el {_friendly_date(d.when)}.</sub>", ""]
        return "\n".join(lines)


ORDINALS = ["Primera", "Segunda", "Tercera", "Cuarta", "Quinta", "Sexta", "Séptima", "Octava", "Novena", "Décima",
            "Undécima", "Duodécima"]


def decision_label(number: int) -> str:
    """1 -> "Primera decisión", 2 -> "Segunda decisión", ...; desde la 13, "Decisión n.º 13"."""
    return f"{ORDINALS[number - 1]} decisión" if 1 <= number <= len(ORDINALS) else f"Decisión n.º {number}"


def join_or(items: list) -> str:
    """["a", "b", "c"] -> "a, b o c" (como se diría en voz alta)."""
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " o " + items[-1]


def _friendly(value) -> str:
    """Muestra listas y diccionarios como texto corrido, sin corchetes ni llaves."""
    if isinstance(value, dict):
        parts = [f"{k}: {_friendly(v)}" for k, v in value.items()]
        # si algún valor ya tiene comas, se separa con punto y coma para que no se mezclen
        return ("; " if any("," in p for p in parts) else ", ").join(parts) if parts else "ninguno"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value)) if value else "ninguno"
    return str(value)


def _friendly_date(iso: str) -> str:
    """'2026-11-11T02:08:13' -> '11/11/2026 a las 02:08'."""
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return when.strftime("%d/%m/%Y a las %H:%M")
