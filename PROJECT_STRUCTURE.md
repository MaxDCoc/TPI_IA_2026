# Estructura del proyecto

El proyecto se organiza en cuatro bloques de código dentro de `src/tp_ia/`, más los datos, un notebook y los tests:

```
src/tp_ia/
├── c45/    → el árbol C4.5 hecho desde cero
├── data/   → carga y preparación de los datos
├── eval/   → Random Forest y la comparación entre los dos modelos
└── web/    → interfaz interactiva para el notebook
```

## 1. `c45/`: el árbol de decisión

Es la parte central, porque no usa ninguna librería de árboles.

- `criteria.py`: calcula la entropía, la ganancia y el *gain ratio* (las medidas que usa el árbol para decidir por qué atributo dividir).
- `splits.py`: busca el mejor corte en cada nodo, tanto para atributos numéricos (con un umbral) como nominales.
- `tree.py`: arma el árbol completo (`C45DecisionTree`) y sabe predecir y explicar cada predicción.
- `pruning.py`: poda por error pesimista, tal como la propone Quinlan.
- `export.py`: exporta el árbol como estructura de datos, como código para Graphviz y como recorrido de un caso (lo usan la interfaz y el notebook).

## 2. `data/`: los datos

- `loader.py`, `schema.py`: un cargador que sirve para cualquier dataset.
- `quality.py`: revisa la calidad de los datos (faltantes, tipos, etc.).
- `decisions.py`: deja registradas las decisiones tomadas sobre los datos (D-01 a D-07).
- `heart.py`: carga Heart Disease (Cleveland), con nombres en español y los códigos reemplazados por su significado.
- `iris.py`, `fixtures.py`: Iris y ejemplos chicos (Play Tennis, el ejemplo de Liu) para probar el árbol.

## 3. `eval/`: la comparación

- `models.py`: el Random Forest de scikit-learn.
- `splits.py`, `metrics.py`: separan los datos en entrenamiento y prueba y calculan las métricas.
- `runner.py`, `experiments.py`: corren los experimentos (cambiar un parámetro y ver qué pasa, validación cruzada anidada y robustez ante datos faltantes).

## 4. `web/`: la interfaz

Es un widget hecho con *anywidget* para usar dentro del notebook. Tiene una parte en Python (`service.py`, `widget.py`) y una parte visual (`static/`, con JS y CSS).

## Fuera de `src/`

- `data/`: los datasets. Iris se usa para desarrollar y validar, y *Heart Disease* de UCI es el dataset de la comparación final.
- `notebooks/laboratorio_arboles_c45.ipynb`: el notebook para Colab, donde se usa todo lo anterior. Es autocontenido: lleva adentro el código y los datos, y es lo único que se entrega.
- `scripts/`: `build_colab_notebook.py` arma ese notebook a partir de los archivos del repositorio, y `run_notebook_offline.py` lo ejecuta en una carpeta vacía para comprobar que no depende de nada más.
- `tests/`: tests con pytest, uno por módulo, y `test_notebook.py`, que comprueba el notebook.
- `README.md` y `DATOS.md`: el estado del proyecto y la documentación de los datasets.

## Recorrido general

`data/` carga los datos, `c45/` y el Random Forest de `eval/` se entrenan con ellos, `eval/` compara los resultados, y `web/` junto con el notebook los muestran.

## Estado

Heart Disease ya está incorporado y verificado (cargador, tests e interfaz), y la interfaz y el notebook se probaron dentro de Colab el 16/11/2026. El informe LNCS se entrega aparte: no está en este repositorio.
