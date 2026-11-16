# -*- coding: utf-8 -*-
"""
Arma el notebook de Colab autocontenido.

GUÍA DE LECTURA
---------------
El entregable del trabajo es UN solo archivo: notebooks/laboratorio_arboles_c45.ipynb. Tiene que funcionar en un Colab
vacío, sin clonar ni descargar nada del repositorio. Para lograrlo, este script copia dentro del notebook:

  - cada archivo del programa (src/tp_ia/...), como una celda "%%writefile ruta" que lo escribe en la carpeta de trabajo;
  - los dos archivos de datos (Iris y Heart Disease, Cleveland);
  - los tests y pytest.ini, para la verificación opcional;
  - el texto explicativo y las celdas que usan el programa (interfaz, árbol desde código, pruebas).

El repositorio sigue siendo la única fuente del código: el notebook se GENERA, no se edita a mano.

    python scripts/build_colab_notebook.py          # regenera el notebook
    python scripts/build_colab_notebook.py --check  # falla si el notebook quedó desactualizado

tests/test_notebook.py comprueba que el notebook esté al día, que no baje código de ningún lado y que funcione desde
una carpeta vacía.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "laboratorio_arboles_c45.ipynb"
THIS_TEST = "tests/test_notebook.py"        # prueba el notebook, así que no viaja dentro de él

# --------------------------------------------------------------------------------------------- qué hace cada archivo
NOTES = {
    "src/tp_ia/__init__.py": "el paquete `tp_ia`, que reúne todo el programa.",
    "src/tp_ia/c45/__init__.py": "lo que se puede importar del árbol.",
    "src/tp_ia/c45/criteria.py": "entropía, ganancia de información y razón de ganancia.",
    "src/tp_ia/c45/splits.py": "busca el mejor corte de cada nodo: por valor en los atributos nominales y por umbral en los numéricos.",
    "src/tp_ia/c45/tree.py": "el árbol completo (`C45DecisionTree`): se entrena con `fit`, predice, da probabilidades y explica cada predicción.",
    "src/tp_ia/c45/pruning.py": "la poda por error pesimista, como la propone Quinlan.",
    "src/tp_ia/c45/export.py": "exporta el árbol como estructura de datos, como código para Graphviz y como recorrido de un caso.",
    "src/tp_ia/data/__init__.py": "lo que se puede importar del paquete de datos.",
    "src/tp_ia/data/schema.py": "cómo se describe un dataset: columnas, tipos de atributo y escenarios de la variable objetivo.",
    "src/tp_ia/data/loader.py": "un cargador que sirve para cualquier dataset: detecta el separador, los decimales y los faltantes.",
    "src/tp_ia/data/quality.py": "revisa la calidad de los datos: faltantes, duplicados y valores extremos.",
    "src/tp_ia/data/decisions.py": "el registro de las decisiones que se toman sobre los datos (D-01 a D-07).",
    "src/tp_ia/data/fixtures.py": "ejemplos chicos para probar el árbol: Play Tennis y el ejemplo de Liu.",
    "src/tp_ia/data/iris.py": "carga Iris.",
    "src/tp_ia/data/heart.py": "carga Heart Disease (Cleveland), con nombres en español y los códigos reemplazados por su significado.",
    "src/tp_ia/eval/__init__.py": "lo que se puede importar de la evaluación.",
    "src/tp_ia/eval/models.py": "el Random Forest de scikit-learn.",
    "src/tp_ia/eval/splits.py": "las particiones de entrenamiento y prueba (validación cruzada estratificada repetida).",
    "src/tp_ia/eval/metrics.py": "las métricas: exactitud, exactitud balanceada, F1 macro, AUC y matriz de confusión.",
    "src/tp_ia/eval/runner.py": "corre los experimentos con las mismas particiones para los dos modelos.",
    "src/tp_ia/eval/experiments.py": "el barrido de un parámetro, la validación cruzada anidada y la robustez a datos faltantes.",
    "src/tp_ia/web/__init__.py": "lo que se puede importar de la interfaz.",
    "src/tp_ia/web/service.py": "toda la lógica de la interfaz, sin necesidad de un navegador.",
    "src/tp_ia/web/widget.py": "el widget de anywidget que se muestra en la celda.",
    "src/tp_ia/web/static/app.js": "la interfaz que se dibuja en el navegador (JavaScript).",
    "src/tp_ia/web/static/app.css": "los estilos de la interfaz.",
    "src/tp_ia/web/devserver.py": "un servidor local para probar la interfaz fuera de Colab. En este cuaderno no se usa.",
    "src/tp_ia/web/static/dev.html": "la página que usa ese servidor local.",
    "data/uci_heart/processed.cleveland.data": "los 303 pacientes de Cleveland (UCI Heart Disease).",
    "data/iris/iris.csv": "las 150 flores de Iris.",
}

SOURCE_SECTIONS = [
    ("### 1.1 El árbol C4.5 (`tp_ia.c45`)",
     "Es la parte central del trabajo y no usa ninguna librería de árboles de decisión.",
     ["src/tp_ia/__init__.py", "src/tp_ia/c45/__init__.py", "src/tp_ia/c45/criteria.py", "src/tp_ia/c45/splits.py",
      "src/tp_ia/c45/tree.py", "src/tp_ia/c45/pruning.py", "src/tp_ia/c45/export.py"]),
    ("### 1.2 Los datos (`tp_ia.data`)",
     "Cargan y revisan los datos, y dejan escrito qué se decidió sobre ellos.",
     ["src/tp_ia/data/__init__.py", "src/tp_ia/data/schema.py", "src/tp_ia/data/loader.py", "src/tp_ia/data/quality.py",
      "src/tp_ia/data/decisions.py", "src/tp_ia/data/fixtures.py", "src/tp_ia/data/iris.py", "src/tp_ia/data/heart.py"]),
    ("### 1.3 La comparación (`tp_ia.eval`)",
     "Entrenan el Random Forest y comparan los dos modelos sobre las mismas particiones.",
     ["src/tp_ia/eval/__init__.py", "src/tp_ia/eval/models.py", "src/tp_ia/eval/splits.py", "src/tp_ia/eval/metrics.py",
      "src/tp_ia/eval/runner.py", "src/tp_ia/eval/experiments.py"]),
    ("### 1.4 La interfaz (`tp_ia.web`)",
     "Una parte en Python (los cálculos) y otra en JavaScript y CSS (lo que se ve).",
     ["src/tp_ia/web/__init__.py", "src/tp_ia/web/service.py", "src/tp_ia/web/widget.py", "src/tp_ia/web/static/app.js",
      "src/tp_ia/web/static/app.css", "src/tp_ia/web/devserver.py", "src/tp_ia/web/static/dev.html"]),
    ("### 1.5 Los datos del trabajo",
     "Los dos archivos están dentro del cuaderno, así que no se descarga nada.",
     ["data/uci_heart/processed.cleveland.data", "data/iris/iris.csv"]),
]

# --------------------------------------------------------------------------------------------------------- textos
INTRO = """# Árbol de decisión C4.5 contra Random Forest

**Inteligencia Artificial · 5.º nivel ISI · Ciclo lectivo 2026**

En este cuaderno se compara un **árbol de decisión C4.5**, programado íntegramente por el equipo sin usar librerías
de árboles, con el **Random Forest** de la librería scikit-learn. Los dos modelos se entrenan y se evalúan con los
mismos datos, para que la comparación sea justa.

**Este cuaderno es autocontenido:** incluye todo el código del programa y los datos que usa, así que no descarga ningún
repositorio. Lo único que instala es la librería [anywidget](https://anywidget.dev) (con `pip`), que permite mostrar la
interfaz dentro del cuaderno.

| Objetivo de la consigna | Dónde se cumple |
|---|---|
| Implementar el algoritmo de árbol de decisión | El código está en la sección **1.1** y se usa en las secciones 2 y 3. |
| Comparar el árbol con el Random Forest | En la interfaz, en la pestaña **Comparar**. |
| Analizar cómo cambian las métricas al modificar los parámetros | En la interfaz, en las pestañas **Árbol** y **Experimentos**. |

### Contenido

0. **Preparación:** instala anywidget y crea la carpeta de trabajo.
1. **Código del proyecto:** el programa completo y los datos. Cada celda escribe un archivo en la carpeta de trabajo.
2. **Interfaz interactiva:** permite usar todo el trabajo desde una sola pantalla, sin escribir código.
3. **El árbol desde código:** muestra cómo usar el mismo modelo directamente desde Python.
4. **Verificación:** ejecuta las pruebas automáticas del proyecto. Es opcional.

> **Cómo usarlo:** ejecutá las celdas en orden. En Colab, podés usar el menú *Entorno de ejecución → Ejecutar todas*."""

PREPARATION = """## 0. Preparación

Esta celda instala anywidget, crea la carpeta de trabajo (`tp_ia_2026`) y deja a Python listo para encontrar el código
que escriben las celdas de la sección 1. Si el cuaderno se vuelve a ejecutar en la misma sesión, también descarta
cualquier versión anterior del código que Python tuviera cargada."""

SETUP_CODE = r'''# Prepara el entorno: instala anywidget y crea la carpeta de trabajo donde se escriben los archivos del proyecto.
import os, subprocess, sys

def run_quiet(command, **kwargs):
    """Ejecuta un comando sin mostrar su salida; si falla, la muestra para poder ver qué pasó."""
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:], sep="\n")
        raise RuntimeError("Falló el comando: " + " ".join(command))
    return result

ROOT = os.path.join("/content" if os.path.isdir("/content") else os.getcwd(), "tp_ia_2026")
os.makedirs(ROOT, exist_ok=True)
os.chdir(ROOT)
for folder in __FOLDERS__:
    os.makedirs(folder, exist_ok=True)

run_quiet([sys.executable, "-m", "pip", "install", "-q", "anywidget"])
sys.path.insert(0, os.path.join(ROOT, "src"))
for name in [m for m in sys.modules if m == "tp_ia" or m.startswith("tp_ia.")]:
    del sys.modules[name]                  # por si esta sesión ya había cargado una versión anterior del código
print(f"✔ Listo: carpeta de trabajo {ROOT}")'''

CODE_INTRO = """## 1. Código del proyecto

Cada celda de esta sección escribe un archivo del programa en la carpeta de trabajo. La primera línea de la celda
(`%%writefile`) dice qué archivo es, y el resto es su código completo. No hace falta leerlas para usar el cuaderno, pero
son el programa entero: no hay nada afuera.

El programa se organiza en cuatro bloques (secciones 1.1 a 1.4) y usa dos archivos de datos (sección 1.5). Las pruebas
automáticas están en la sección 4."""

DATA_NOTE = """- **Enfermedad cardíaca** (`processed.cleveland.data`): es el archivo de los pacientes de la Cleveland Clinic del dataset
  *Heart Disease* del UCI Machine Learning Repository, con licencia CC BY 4.0. Cita: Janosi, A., Steinbrunn, W.,
  Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C52P4X
- **Iris** (`iris.csv`): Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of
  Eugenics, 7*(2), 179-188.
- **Play Tennis** y el ejemplo de Liu están escritos en el código (`fixtures.py`). **Wine** y **Breast Cancer
  Wisconsin** vienen incluidos en la librería scikit-learn."""

INTERFACE = """## 2. Interfaz interactiva

La interfaz permite explorar los datos, entrenar el árbol, compararlo con el Random Forest y hacer experimentos,
todo sin escribir código. Los cálculos se hacen en Python y los resultados se dibujan en el navegador.

Al iniciar, la interfaz muestra el dataset de enfermedad cardíaca. Para usar otro, se elige en la lista de arriba a la derecha. También se puede cargar un archivo CSV propio desde la pestaña **Datos**.

### Conjuntos de datos disponibles

| Conjunto de datos | Qué contiene | Para qué sirve |
|---|---|---|
| **Enfermedad cardíaca** | 303 pacientes de la Cleveland Clinic, descritos con 13 atributos clínicos, como la edad, el colesterol o el tipo de dolor de pecho. | Es el dataset principal del trabajo: un problema real que mezcla atributos numéricos y nominales, y que tiene algunos valores faltantes. Se predice si el paciente tiene o no una enfermedad cardíaca. |
| **Play Tennis** | 14 días descritos por el clima. | Es el ejemplo clásico del libro de Quinlan, así que permite comparar el árbol obtenido con el del libro. |
| **Iris** | 150 flores de 3 especies, descritas con 4 medidas. | Es el dataset con el que se desarrolló y se probó el algoritmo. |
| **Wine** | 178 vinos de 3 variedades, descritos con 13 atributos químicos. | Muestra cómo se comportan los modelos cuando hay más atributos. |
| **Breast Cancer Wisconsin** | 569 casos con 30 medidas, clasificados en 2 diagnósticos. | Es otro caso real, más exigente por la cantidad de atributos. |

> En el dataset de enfermedad cardíaca, los atributos tienen nombres en español y sus códigos se reemplazaron por su significado (por ejemplo, *sin síntomas* en lugar de *4*). Así el árbol se puede leer sin consultar un diccionario. En la pestaña **Datos**, cada atributo muestra una breve explicación de lo que mide.

| Pestaña | Qué se puede hacer |
|---|---|
| **1. Datos** | Ver el conjunto de datos elegido: sus atributos, la calidad de los datos y las decisiones que se tomaron sobre ellos. |
| **2. Árbol** | Ver el árbol C4.5 dibujado. Al cambiar los parámetros, el árbol se vuelve a entrenar al instante. También se puede describir un caso y ver qué clase predice cada modelo. |
| **3. Comparar** | Comparar el árbol con el Random Forest sobre las mismas particiones de los datos, con el promedio y el desvío de cada métrica. |
| **4. Experimentos** | Probar varios valores de un parámetro para ver cuándo aparece el sobreajuste, y medir qué pasa cuando faltan datos. |

### Recorrido sugerido

1. **Datos:** con *Enfermedad cardíaca*, que es el dataset que se muestra al iniciar, leé qué mide cada atributo y fijate, por los colores de las barras, qué valores aparecen más en los pacientes con enfermedad.
2. **Árbol:** activá *Podar por error pesimista* y compará la cantidad de nodos y la exactitud antes y después de podar.
3. **Árbol:** en *Predecir un caso*, mové los valores. Después marcá un atributo como *falta* y fijate cómo el árbol reparte el caso entre las ramas.
4. **Comparar:** tocá **Comparar** con los valores que propone la cátedra (100 árboles y profundidad 10).
5. **Experimentos:** probá distintos valores de *Profundidad máxima*, del 1 al 8, y buscá a partir de cuál el árbol empieza a sobreajustar."""

INTERFACE_CODE = """# Abre la interfaz interactiva.
from tp_ia.web.widget import show

lab = show()"""

TREE_FROM_CODE = """## 3. El árbol desde código

Esta sección entrena el árbol C4.5 directamente desde Python, sin pasar por la interfaz. Muestra que la implementación
es una clase que se puede usar en cualquier programa: se crea el árbol, se lo entrena con `fit` y queda listo para
predecir.

Se usa el dataset de enfermedad cardíaca con la poda activada y la misma partición que usa la interfaz al iniciar:
el 70 % de los pacientes para entrenar y el 30 % restante para medir la exactitud con casos que el árbol no vio.
Por eso el árbol que aparece acá es idéntico al de la pestaña **Árbol** con la poda activada. Para probar otros
parámetros o consultar las decisiones sobre los datos conviene usar la interfaz, porque las pestañas **Árbol** y
**Datos** muestran esa información de forma interactiva."""

TREE_CODE = r'''# Entrena el árbol propio con poda sobre el dataset de enfermedad cardíaca y lo dibuja con Graphviz,
# con el formato de las láminas de la cátedra.
from IPython.display import SVG, display
from tp_ia.c45 import C45DecisionTree
from tp_ia.data import load_heart_dataset
from tp_ia.eval import make_splits

ds = load_heart_dataset()

# La misma partición que la interfaz: 70 % para entrenar y 30 % para evaluar, estratificada y con la semilla 0.
split = make_splits(ds.y, scheme="holdout", n_repeats=1, test_size=0.3, seed=0)[0]
y = ds.y.to_numpy()
X_train, y_train = ds.X.iloc[split.train], y[split.train]
X_test, y_test = ds.X.iloc[split.test], y[split.test]

tree = C45DecisionTree(prune=True).fit(X_train, y_train, feature_types=ds.feature_types)
print(f"Entrenado con {len(y_train)} pacientes y evaluado con {len(y_test)}.")
print(f"{tree.n_nodes_} nodos (se podaron {tree.n_pruned_}), profundidad {tree.depth_}")
print(f"Exactitud con los datos de prueba: {tree.score(X_test, y_test):.1%}")

try:    # se dibuja acá mismo para que un error de Graphviz se pueda atrapar y explicar
    from graphviz import Source
    display(SVG(Source(tree.to_dot()).pipe(format="svg").decode("utf-8")))
except Exception as error:
    print("\nNo se pudo dibujar con Graphviz:", error)
    print("Hace falta tener instalado Graphviz. Mientras tanto, el árbol se puede ver como texto con print(tree.export_text()).")'''

VERIFICATION = """## 4. Verificación (opcional)

Primero se escriben los archivos de prueba (carpeta `tests/`) y el archivo de configuración de pytest. Después, la
última celda ejecuta las pruebas automáticas del proyecto. Las pruebas comprueban que funcionen correctamente el
algoritmo C4.5, la carga de los datos, la evaluación de los modelos y la interfaz."""

TESTS_CODE = r'''# Corre las pruebas automáticas del proyecto y muestra solo el resumen (o el detalle, si algo falla).
run_quiet([sys.executable, "-m", "pip", "install", "-q", "pytest"])
result = subprocess.run([sys.executable, "-m", "pytest", "--color=no", "-p", "no:cacheprovider"],
                        cwd=ROOT, capture_output=True, text=True)
summary = result.stdout.strip().splitlines()
if result.returncode == 0:
    print("✔", summary[-1])
else:
    print(result.stdout[-4000:] or result.stderr[-4000:])'''


# ------------------------------------------------------------------------------------------------------- utilidades
def tracked(prefix: str) -> list:
    """Archivos versionados bajo un prefijo (así el notebook refleja el repositorio y no basura local)."""
    out = subprocess.run(["git", "ls-files", prefix], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if line)


def read_text(path: str) -> str:
    """El contenido de un archivo con saltos de línea \\n y exactamente un salto al final."""
    text = (REPO / path).read_bytes().decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
    return text.rstrip("\n") + "\n"


def lines(text: str) -> list:
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(text)}


def file_cell(path: str) -> dict:
    return code(f"%%writefile {path}\n" + read_text(path))


def section_header(title: str, intro: str, paths: list) -> str:
    bullets = "\n".join(f"- `{Path(p).name}`: {NOTES[p]}" for p in paths)
    return f"{title}\n\n{intro}\n\n{bullets}"


def project_files() -> tuple:
    """(archivos del programa y datos, archivos de prueba), comprobando que no falte ni sobre ninguno."""
    listed = [p for _, _, paths in SOURCE_SECTIONS for p in paths]
    missing = sorted(set(tracked("src")) - set(listed))
    unknown = sorted(p for p in listed if p.startswith("src/") and p not in set(tracked("src")))
    if missing or unknown:
        raise SystemExit("Hay que actualizar SOURCE_SECTIONS en scripts/build_colab_notebook.py.\n"
                         f"  Archivos de src/ que faltan en el notebook: {missing}\n"
                         f"  Archivos del notebook que ya no existen: {unknown}")
    for path in listed:
        if path not in NOTES:
            raise SystemExit(f"Falta describir {path} en NOTES (scripts/build_colab_notebook.py).")
    tests = [p for p in tracked("tests") if p.endswith(".py") and p != THIS_TEST]
    return listed, tests


# ------------------------------------------------------------------------------------------------------- el notebook
def build_notebook() -> dict:
    listed, tests = project_files()
    folders = sorted({str(Path(p).parent.as_posix()) for p in listed + tests + ["pytest.ini"]} - {"."})
    setup = SETUP_CODE.replace("__FOLDERS__", "[\n" + "".join(f'    "{f}",\n' for f in folders) + "]")

    cells = [md(INTRO), md(PREPARATION), code(setup), md(CODE_INTRO)]
    for title, intro, paths in SOURCE_SECTIONS:
        header = section_header(title, intro, paths)
        if title.startswith("### 1.5"):
            header += "\n\n" + DATA_NOTE
        cells.append(md(header))
        cells.extend(file_cell(p) for p in paths)
    cells += [md(INTERFACE), code(INTERFACE_CODE), md(TREE_FROM_CODE), code(TREE_CODE), md(VERIFICATION),
              file_cell("pytest.ini")]
    cells.extend(file_cell(p) for p in tests)
    cells.append(code(TESTS_CODE))
    return {
        "cells": cells,
        "metadata": {"colab": {"name": "laboratorio_arboles_c45.ipynb", "provenance": [], "toc_visible": True},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def render(notebook: dict) -> str:
    return json.dumps(notebook, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def main(argv: list) -> int:
    notebook = build_notebook()
    text = render(notebook)
    if "--check" in argv:
        # Se comparan los datos y no el texto: Git en Windows puede cambiar los saltos de línea del archivo.
        current = json.loads(NOTEBOOK.read_text(encoding="utf-8")) if NOTEBOOK.exists() else None
        if current != notebook:
            print("El notebook está desactualizado. Regeneralo con: python scripts/build_colab_notebook.py")
            return 1
        print("El notebook está al día.")
        return 0
    NOTEBOOK.write_text(text, encoding="utf-8", newline="\n")
    n_code = sum(c["cell_type"] == "code" for c in notebook["cells"])
    print(f"Notebook escrito: {NOTEBOOK.relative_to(REPO)} ({len(text) // 1024} KB, {n_code} celdas de código)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
