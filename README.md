# TPI_IA_2026

Trabajo práctico de Inteligencia Artificial 2026 (5.º nivel ISI): **árbol de decisión C4.5 implementado desde cero vs. Random Forest (scikit-learn)**, comparados sobre el dataset *Heart Disease* de UCI.

El árbol no usa ninguna librería de árboles de decisión: todo está en `src/tp_ia/c45/`, con comentarios guía en cada archivo.

## Estado

- [x] Árbol C4.5 propio: gain ratio, atributos numéricos y nominales, faltantes con reparto fraccionario, parámetros, árbol en texto y explicación de cada predicción.
- [x] Tests con Play Tennis, el ejemplo de Liu, Iris y propiedades.
- [ ] Capa de datos agnóstica al dataset y registro de decisiones.
- [ ] Random Forest, métricas y corredor de experimentos.
- [ ] Sitio web embebido en el notebook de Colab.
- [ ] Poda, notebook final e informe.

## Estructura

```
src/tp_ia/c45/     el árbol: criteria.py (entropía, ganancia) · splits.py (mejor corte) · tree.py
src/tp_ia/data/    fixtures de juguete (Play Tennis, Liu); cargadores del dataset real, próximamente
tests/             pytest
data/uci_heart/    archivos procesados de UCI (ver DATOS.md)
DATOS.md           fuente, cita, licencia, columnas y auditoría de los datos
```

## Cómo correr los tests

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
pytest
```

## Uso del árbol

Desde la raíz del repo, con `PYTHONPATH=src`:

```python
from tp_ia.c45 import C45DecisionTree
from tp_ia.data.fixtures import play_tennis

X, y = play_tennis()
tree = C45DecisionTree(max_depth=3).fit(X, y)
print(tree.export_text())          # el árbol como texto
print(tree.predict_proba(X.head(2)))
print(tree.explain(X.iloc[0]))     # reglas recorridas por un caso
```

Una columna numérica que en realidad es una categoría (por ejemplo un código 1/2/3/4) se declara con `fit(X, y, feature_types={"cp": "nominal"})`.

### Parámetros

| Parámetro | Por defecto | Qué hace |
|---|---|---|
| `criterion` | `"gain_ratio"` | `"gain_ratio"` (C4.5) o `"gain"` (ganancia pura) |
| `max_depth` | `None` | profundidad máxima |
| `min_samples_split` | `2` | peso mínimo para intentar dividir un nodo |
| `min_samples_leaf` | `2` | al menos dos ramas deben tener este peso (parámetro `-m` de C4.5) |
| `min_gain` | `0.0` | ganancia mínima para aceptar una división |
| `use_avg_gain` | `True` | heurística de C4.5: solo atributos con ganancia ≥ promedio |
| `threshold_mode` | `"c45"` | umbral numérico: valor observado ≤ punto medio, o `"midpoint"` |
| `numeric_gain_correction` | `False` | corrección de la versión 8 de C4.5 para atributos numéricos |

## Datos

Ver [DATOS.md](DATOS.md). Fuente: UCI Machine Learning Repository, *Heart Disease*, licencia CC BY 4.0.
