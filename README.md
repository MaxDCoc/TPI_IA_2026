# TPI_IA_2026

Trabajo práctico de Inteligencia Artificial 2026 (5.º nivel ISI): **árbol de decisión C4.5 implementado desde cero vs. Random Forest (scikit-learn)**, comparados sobre el dataset *Heart Disease* de UCI. El pipeline se desarrolla y se valida primero sobre Iris.

El árbol no usa ninguna librería de árboles de decisión: todo está en `src/tp_ia/c45/`, con comentarios guía en cada archivo.

## Estado

- [x] Árbol C4.5 propio: gain ratio, atributos numéricos y nominales, faltantes con reparto fraccionario, parámetros, árbol en texto y explicación de cada predicción.
- [x] Tests con Play Tennis, el ejemplo de Liu, Iris y propiedades.
- [x] Capa de datos agnóstica al dataset: cargador, auditoría y registro de decisiones (D-01 a D-07). Probada sobre Iris.
- [x] Random Forest, métricas y corredor de experimentos. Probados sobre Iris.
- [ ] Heart Disease: verificar el cargador y los escenarios, una vez confirmado todo con Iris.
- [ ] Sitio web embebido en el notebook de Colab.
- [ ] Poda, notebook final e informe.

## Estructura

```
src/tp_ia/c45/     el árbol: criteria.py (entropía, ganancia) · splits.py (mejor corte) · tree.py
src/tp_ia/data/    cargador agnóstico, auditoría, registro de decisiones, Iris y fixtures de juguete (Play Tennis, Liu)
src/tp_ia/eval/    particiones, métricas, Random Forest (scikit-learn) y corredor de experimentos
tests/             pytest
data/iris/         iris.csv: dataset de desarrollo y validación (ver DATOS.md)
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

## Pipeline completo sobre Iris

Todo se construye y se prueba primero sobre Iris. Mismas particiones para el árbol y el Random Forest, y cada decisión sobre los datos queda registrada con su evidencia:

```python
from tp_ia.data import load_iris_dataset
from tp_ia.eval import make_splits, paired_comparison, run_experiment, summarize

ds = load_iris_dataset()
splits = make_splits(ds.y, n_splits=5, n_repeats=4, seed=0)      # validación cruzada estratificada repetida
models = {"DT": ("dt", {}), "RF": ("rf", {"n_estimators": 100, "random_state": 0})}
results = run_experiment(ds, models, splits)                     # una fila por partición y modelo
print(summarize(results).round(3))                               # media y desvío de cada métrica
print(paired_comparison(results, "RF", "DT"))                    # diferencia sobre las mismas particiones
print(ds.log.to_markdown())                                      # decisiones D-01 a D-07 con su evidencia
```

`load_dataset(ruta, DatasetSpec(...))` hace lo mismo con cualquier CSV: detecta separador, decimal, codificación y marcadores de faltante (`?`, `NA`, vacío).

## Datos

Ver [DATOS.md](DATOS.md). Iris (Fisher, 1936) se usa para desarrollar y validar el pipeline. El dataset de análisis es UCI Machine Learning Repository, *Heart Disease*, licencia CC BY 4.0, y se usa después de confirmar todo con Iris.
