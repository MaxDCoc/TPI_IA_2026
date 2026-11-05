"""
Poda por error pesimista de C4.5 (Quinlan, 1993, cap. 4).

GUÍA DE LECTURA (para el coloquio)
----------------------------------
El problema: un árbol que crece hasta clasificar bien TODOS los casos de entrenamiento memoriza el ruido.
Cada hoja "perfecta" que cubre pocos casos es en realidad una apuesta frágil.

La idea de Quinlan: el error medido sobre los mismos casos con que se construyó el árbol es OPTIMISTA.
Si una hoja cubre N casos y falla en E, el error verdadero probablemente sea mayor. Se lo estima con el
LÍMITE SUPERIOR de un intervalo de confianza binomial (nivel CF, 25 % por defecto): U_CF(E, N). Los
"errores estimados" de la hoja son N * U_CF(E, N) = E + AddErrs(N, E).
  - Con pocos casos el límite es muy holgado (una hoja de 1 caso sin errores estima 0.75 errores); con
    muchos casos se acerca al error observado. Por eso la poda castiga sobre todo a las hojas chicas.
  - Bajar CF poda MÁS (el límite superior se vuelve más pesimista); subirlo poda menos.

El algoritmo, de abajo hacia arriba, en cada nodo interno:
  (a) errores estimados si el nodo fuera una HOJA   -> E_nodo + AddErrs(N_nodo, E_nodo)
  (b) errores estimados del SUBÁRBOL                -> suma de lo estimado en cada hijo (ya podado)
  Si (a) <= (b) + 0.1 el subárbol se reemplaza por una hoja. El 0.1 es la tolerancia de C4.5: ante un
  empate técnico se prefiere el árbol más chico.
No hacen falta datos aparte: se usa la distribución de clases de entrenamiento que ya guarda cada nodo
(con los pesos fraccionarios de los faltantes, igual que en la construcción).

AddErrs (versión del código de C4.5 release 8)
  - E = 0:          N * (1 - CF^(1/N))     límite binomial exacto cuando no hubo errores.
  - 0 < E < 1:      interpolación lineal entre el caso E = 0 y el caso E = 1 (hay errores fraccionarios
                    cuando los faltantes reparten casos).
  - E + 0.5 >= N:   0.67 * (N - E)         caso de borde.
  - resto:          aproximación normal con corrección de continuidad. El coeficiente z sale de interpolar
                    una tabla de desvíos normales (para CF = 25 % da z = 0.6925; el cuantil exacto sería 0.6745).

QUÉ ESTÁ VERIFICADO Y QUÉ NO (para la tabla del informe)
  - U_25%(0, 6) = 0.206 coincide con el valor que cita Quinlan en su libro (visto en una cita pública).
    U_25%(0, 9) = 0.143, U_25%(0, 1) = 0.750 y U_25%(1, 16) = 0.157 son los otros valores de su ejemplo,
    tomados de memoria: esta fórmula los reproduce a tres decimales, pero no se contrastaron con el libro.
  - La tolerancia 0.1 y la tabla de desvíos son las de C4.5 tal como las recordamos (Weka/J48 usa también
    0.1, pero con z = 0.6745 exacto y otro caso de borde, así que sus podas pueden diferir apenas).
  - La constante 0.67 del caso de borde NO se pudo contrastar con el código original. Solo pesa en nodos
    con peso fraccionario muy chico, y se deja como constante con nombre (EDGE_FACTOR) para poder cambiarla.

DIFERENCIA CON C4.5 COMPLETO
  - No se implementa "subtree raising" (reemplazar un nodo por su rama más poblada), que C4.5 aplica
    además de reemplazar un subárbol por una hoja.
"""
from __future__ import annotations

import math

import numpy as np

from .criteria import EPS

CF_TABLE = (0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00)    # nivel de confianza (cola superior)
Z_TABLE = (4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00)       # desvío normal correspondiente
EDGE_FACTOR = 0.67        # caso e + 0.5 >= N (sin contrastar con el código original: ver arriba)
TOLERANCE = 0.1           # el subárbol sobrevive solo si mejora a la hoja por más de esto
DEFAULT_CONFIDENCE = 0.25


def confidence_coefficient(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """z² para el nivel de confianza dado, interpolando linealmente la tabla de C4.5."""
    if not 0 < confidence <= 1:
        raise ValueError("confidence debe estar en (0, 1]")
    z = float(np.interp(confidence, CF_TABLE, Z_TABLE))
    return z * z


def added_errors(n: float, e: float, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """AddErrs de C4.5: errores que se SUMAN a los E observados en N casos para obtener el límite superior."""
    if n <= EPS:
        return 0.0
    if e < 1e-6:                                    # sin errores: límite binomial exacto
        return n * (1.0 - confidence ** (1.0 / n))
    if e < 0.9999:                                  # error fraccionario: interpolar entre e = 0 y e = 1
        base = n * (1.0 - confidence ** (1.0 / n))
        return base + e * (added_errors(n, 1.0, confidence) - base)
    if e + 0.5 >= n:                                # casi todos mal clasificados
        return EDGE_FACTOR * (n - e)
    z2 = confidence_coefficient(confidence)
    upper = (e + 0.5 + z2 / 2 + math.sqrt(z2 * ((e + 0.5) * (1 - (e + 0.5) / n) + z2 / 4))) / (n + z2)
    return n * upper - e


def estimated_errors(n: float, e: float, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Errores estimados de una hoja con N casos y E errores observados: E + AddErrs(N, E)."""
    return e + added_errors(n, e, confidence)


def _make_leaf(node) -> None:
    """Convierte un nodo interno en hoja conservando su distribución de clases."""
    node.children = []
    node.feature = node.kind = node.threshold = node.branch_probs = None
    node.values = []
    node.gain = node.gain_ratio = 0.0


def prune_tree(root, confidence: float = DEFAULT_CONFIDENCE, tolerance: float = TOLERANCE) -> int:
    """Poda `root` en el lugar (de abajo hacia arriba). Devuelve cuántos subárboles se reemplazaron por una hoja."""
    if not 0 < confidence <= 0.5:
        raise ValueError("confidence debe estar en (0, 0.5]")
    collapsed = 0

    def visit(node) -> float:
        nonlocal collapsed
        as_leaf = estimated_errors(node.n_weight, node.errors, confidence)
        if node.is_leaf:
            return as_leaf
        as_subtree = sum(visit(child) for child in node.children)    # los hijos ya vienen podados
        if as_leaf <= as_subtree + tolerance:
            _make_leaf(node)
            collapsed += 1
            return as_leaf
        return as_subtree

    visit(root)
    return collapsed
