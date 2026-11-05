# TPI_IA_2026

Trabajo práctico de Inteligencia Artificial 2026 (5.º nivel ISI): **árbol de decisión C4.5 implementado desde cero vs. Random Forest (scikit-learn)**, comparados sobre el dataset *Heart Disease* de UCI. El pipeline se desarrolla y se valida primero sobre Iris.

El árbol no usa ninguna librería de árboles de decisión: todo está en `src/tp_ia/c45/`, con comentarios guía en cada archivo.

## Estado

- [x] Árbol C4.5 propio: gain ratio, atributos numéricos y nominales, faltantes con reparto fraccionario, parámetros, árbol en texto y explicación de cada predicción.
- [x] Tests con Play Tennis, el ejemplo de Liu, Iris y propiedades.
- [x] Capa de datos agnóstica al dataset: cargador, auditoría y registro de decisiones (D-01 a D-07). Probada sobre Iris.
- [x] Random Forest, métricas y corredor de experimentos. Probados sobre Iris.
- [x] Experimentos: barrido de un parámetro, validación cruzada anidada y robustez a faltantes. Probados sobre Iris.
- [ ] Heart Disease: verificar el cargador y los escenarios, una vez confirmado todo con Iris.
- [ ] Sitio web embebido en el notebook de Colab.
- [x] Poda por error pesimista de C4.5 (`prune=True`). Probada sobre Iris.
- [ ] Notebook final e informe.

## Estructura

```
src/tp_ia/c45/     el árbol: criteria.py (entropía, ganancia) · splits.py (mejor corte) · tree.py
src/tp_ia/data/    cargador agnóstico, auditoría, registro de decisiones, Iris y fixtures de juguete (Play Tennis, Liu)
src/tp_ia/eval/    particiones, métricas, Random Forest (scikit-learn), corredor y experimentos
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
| `prune` | `False` | poda por error pesimista de C4.5 (post-poda, ver más abajo) |
| `confidence` | `0.25` | nivel de confianza CF de la poda, en (0, 0.5]: más chico poda más |

### Poda

La poda reemplaza un subárbol por una hoja cuando el error estimado de la hoja no es peor que el del subárbol (más una tolerancia de 0.1). El error estimado es el límite superior de un intervalo de confianza binomial (Quinlan, 1993, cap. 4), así que castiga a las hojas chicas que memorizan ruido. No usa datos aparte: usa las distribuciones de clase del entrenamiento que ya guarda cada nodo. Los comentarios guía de [pruning.py](src/tp_ia/c45/pruning.py) explican la fórmula.

```python
from tp_ia.c45 import C45DecisionTree
from tp_ia.data import load_iris_dataset

ds = load_iris_dataset()
full = C45DecisionTree().fit(ds.X, ds.y)
pruned = C45DecisionTree(prune=True, confidence=0.25).fit(ds.X, ds.y)
print(full.n_nodes_, "->", pruned.n_nodes_, "nodos")      # 13 -> 9
print(pruned.export_text())                               # las hojas muestran (casos/errores)
```

- **Decisión:** en C4.5 la poda viene activa por defecto. Acá es opcional (`prune=False`) para poder mostrar y comparar el árbol completo y el podado.
- **Qué está verificado:** las tasas U₂₅%(0,6) = 0.206 (confirmada en una cita del libro), 0.143, 0.750 y 0.157 del ejemplo de Quinlan, y que ese ejemplo se poda. En Iris la poda deja las mismas cinco hojas que J48 (50, 48/1, 3, 3/1 y 46/1); solo cambia qué atributo separa a setosa (empate de gain ratio).
- **Qué no está verificado:** la constante 0.67 del caso de borde `e + 0.5 ≥ N` salió del código de C4.5 tal como lo recordamos y no pudo contrastarse con el original (solo pesa con pesos fraccionarios muy chicos).
- **Diferencia con C4.5 completo:** no implementa *subtree raising* (reemplazar un nodo por su rama más poblada).
- **Ojo con el valor por defecto:** con datos muy ruidosos, `confidence=0.25` puede podar poco. En un conjunto sintético con 20 % de etiquetas invertidas pasó de 105 a 77 nodos, mientras que con `confidence=0.1` quedó en 3 nodos y generalizó mucho mejor. Conviene barrer el parámetro con `parameter_sweep(ds, "dt", "confidence", [...], splits, base_params={"prune": True})`.

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

### Experimentos

Tres experimentos listos, todos con las mismas particiones y el mismo formato de tabla (tardan unos 45 segundos en total sobre Iris):

```python
from tp_ia.eval import (make_splits, missing_robustness, nested_cv, parameter_sweep, robustness_table,
                        selection_frequency)

splits = make_splits(ds.y, n_splits=5, n_repeats=2, seed=0)

# 1. Un parámetro a la vez: rendimiento en test, en train y la brecha entre ambos (sobreajuste)
print(parameter_sweep(ds, "dt", "max_depth", [1, 2, 3, None], splits).round(3))

# 2. Elegir parámetros sin hacer trampa: dentro de cada partición, la CV interna ve SOLO el train
nested = nested_cv(ds, "dt", {"max_depth": [2, 3, None], "min_samples_leaf": [2, 5]}, splits)
print(nested[["chosen", "inner_score", "test_macro_f1"]].round(3))
print(selection_frequency(nested))          # si la elección cambia mucho, es inestable

# 3. Robustez: se borra al azar una fracción de los valores (en train y en test) y se mide cuánto cae cada modelo
models = {"DT": ("dt", {}), "RF": ("rf", {"n_estimators": 100, "random_state": 0})}
print(robustness_table(missing_robustness(ds, models, [0, 0.2, 0.4], splits, seeds=(0, 1, 2))).round(3))
```

En la selección anidada, ante un empate gana la primera combinación de la grilla: conviene poner primero la más simple.

## Datos

Ver [DATOS.md](DATOS.md). Iris (Fisher, 1936) se usa para desarrollar y validar el pipeline. El dataset de análisis es UCI Machine Learning Repository, *Heart Disease*, licencia CC BY 4.0, y se usa después de confirmar todo con Iris.
