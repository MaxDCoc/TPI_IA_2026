"""Tests de los criterios de C4.5 contra valores calculados a mano y contra un oráculo externo.

El oráculo (scipy.stats.entropy) se usa SOLO acá, para comprobar nuestras fórmulas; el árbol no
depende de scipy para nada.
"""
import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy

from tp_ia.c45.criteria import entropy, entropy_rows, gain_ratio, information_gain, split_info
from tp_ia.data.fixtures import liu_loans, play_tennis


def class_counts(y, classes):
    return np.array([(y == c).sum() for c in classes], dtype=float)


def gain_of(X, y, column):
    """Ganancia de dividir por `column` (una rama por valor) usando nuestras funciones."""
    classes = sorted(y.unique())
    parent = class_counts(y, classes)
    children = np.array([class_counts(y[X[column] == v], classes) for v in sorted(X[column].unique())])
    return information_gain(parent, children)


def test_entropy_known_values():
    assert entropy([5, 0]) == 0.0                              # conjunto puro
    assert entropy([0, 0]) == 0.0                              # vacío
    assert entropy([3, 3]) == pytest.approx(1.0)               # equilibrado, 2 clases = 1 bit
    assert entropy([1, 1, 1, 1]) == pytest.approx(2.0)         # 4 clases uniformes = 2 bits
    assert entropy([9, 5]) == pytest.approx(0.9402859586706309, abs=1e-9)  # Play Tennis


def test_entropy_matches_scipy_oracle():
    rng = np.random.default_rng(0)
    for _ in range(100):
        counts = rng.integers(0, 20, size=rng.integers(2, 7)).astype(float)
        if counts.sum() == 0:
            continue
        assert entropy(counts) == pytest.approx(scipy_entropy(counts, base=2), abs=1e-9)


def test_entropy_bounds_and_scale_invariance():
    rng = np.random.default_rng(1)
    for _ in range(50):
        counts = rng.random(rng.integers(2, 8)) * 10
        h = entropy(counts)
        assert -1e-12 <= h <= np.log2(len(counts)) + 1e-12
        assert entropy(counts * 3.7) == pytest.approx(h)       # pesos fraccionarios: no importa la escala


def test_entropy_rows_matches_scalar_version():
    m = np.array([[5, 0, 0], [3, 3, 0], [1, 1, 1], [0, 0, 0]], dtype=float)
    expected = [entropy(row) for row in m]
    np.testing.assert_allclose(entropy_rows(m), expected, atol=1e-12)


def test_play_tennis_gains_match_textbook():
    X, y = play_tennis()
    expected = {"Outlook": 0.246, "Humidity": 0.151, "Wind": 0.048, "Temperature": 0.029}
    for column, value in expected.items():
        assert gain_of(X, y, column) == pytest.approx(value, abs=1e-3)


def test_liu_gains_match_book():
    X, y = liu_loans()
    expected = {"Age": 0.083, "Has_job": 0.324, "Own_house": 0.420, "Credit_rating": 0.363}
    for column, value in expected.items():
        assert gain_of(X, y, column) == pytest.approx(value, abs=1e-3)


def test_gain_ratio_of_outlook():
    X, y = play_tennis()
    gain = gain_of(X, y, "Outlook")
    si = split_info([5, 4, 5])                                  # Outlook: 5 Sunny, 4 Overcast, 5 Rain
    assert si == pytest.approx(scipy_entropy([5, 4, 5], base=2))
    assert gain_ratio(gain, si) == pytest.approx(0.156, abs=1e-3)


def test_degenerate_split_has_zero_gain_ratio():
    assert split_info([14]) == 0.0                              # una sola rama
    assert gain_ratio(0.3, 0.0) == 0.0                          # sin división por cero


def test_gain_is_never_negative_nor_above_parent_entropy():
    rng = np.random.default_rng(2)
    for _ in range(100):
        parent = rng.integers(1, 30, size=3).astype(float)
        branches = rng.integers(2, 5)
        # cada clase se reparte al azar entre las ramas -> los hijos suman exactamente al padre
        children = np.array([rng.multinomial(int(c), np.ones(branches) / branches) for c in parent]).T
        g = information_gain(parent, children)
        assert -1e-12 <= g <= entropy(parent) + 1e-12
