# Datos

## Iris (desarrollo y validación del pipeline)

Es el dataset con el que se construye y se prueba todo primero. Cuando el pipeline funcione bien con Iris, se pasa a Heart Disease.

- **Cita:** Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2), 179–188.
- **Copia usada:** la que incluye scikit-learn (`sklearn.datasets.load_iris`), guardada como `data/iris/iris.csv` para que el cargador la lea como a cualquier otro archivo. scikit-learn documenta que corrigió dos datos erróneos según el artículo de Fisher, y que esa versión coincide con la de R pero no con la del repositorio de UCI.
- **Contenido:** 150 filas, 4 atributos numéricos en cm (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) y la clase `species` (setosa, versicolor y virginica, 50 de cada una). Sin faltantes.
- **Auditoría (la hace el cargador):** 1 fila duplicada (las filas 101 y 142 contando desde 0, ambas virginica 5.8 / 2.7 / 5.1 / 1.9) y 4 valores extremos según la regla de Tukey, todos en `sepal_width` (2.0, 4.1, 4.2 y 4.4).
- **Uso:** desarrollo y pruebas. El dataset de análisis del informe es Heart Disease (a confirmar con la cátedra).

## Decisiones que registra el cargador

`load_iris_dataset().log.to_markdown()` (o el de cualquier dataset) devuelve este registro con la evidencia real, listo para pegar en el informe. D-07 se anota al correr un experimento.

| Id | Tema | Qué se hace por defecto |
|---|---|---|
| D-01 | Fuente | se registran la fuente, la cita y el tamaño |
| D-02 | Variable objetivo | la columna original, o un escenario (por ejemplo binario o multiclase) |
| D-03 | Faltantes | se conservan: el árbol reparte fracciones de caso y el RF imputa con estadísticos del train. Opción `missing_policy="drop_rows"` |
| D-04 | Duplicados | se conservan y se informan. Opción `duplicates="drop"` |
| D-05 | Tipos de atributo | los declarados nominales ramifican por valor y los numéricos por umbral |
| D-06 | Valores extremos | se conservan. La regla de Tukey (1.5·IQR) solo informa cuántos hay |
| D-07 | Evaluación | validación cruzada estratificada repetida, con las mismas particiones para todos los modelos |

## UCI Heart Disease (dataset principal, a confirmar con la cátedra)

- **Fuente:** UCI Machine Learning Repository, *Heart Disease* (id 45): <https://archive.ics.uci.edu/dataset/45/heart+disease>
- **Cita** (formato que da UCI): Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
- **Artículo de referencia:** Detrano, R., Jánosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.
- **Licencia:** CC BY 4.0 (se puede redistribuir citando la fuente).
- **Descarga:** 04/11/2026, desde `https://archive.ics.uci.edu/static/public/45/heart+disease.zip` (126 KB).

## Archivos incluidos en `data/uci_heart/`

El contenido completo del zip de UCI, que se puede redistribuir citando la fuente (CC BY 4.0): los cuatro archivos procesados (`processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`), los archivos originales (`cleveland.data`, `hungarian.data`, `switzerland.data`, `long-beach-va.data`, `new.data`, `reprocessed.hungarian.data`, `cleve.mod`, `bak`, `Index`, `ask-detrano` y la carpeta `costs/`), `heart-disease.names` (documentación) y `WARNING`. El trabajo usa solo `processed.cleveland.data`; el resto queda como material de consulta.

No se versiona el archivo zip. El `WARNING` de los autores indica que el crudo `cleveland.data` quedó dañado y que `processed.cleveland.data` es utilizable para la situación de 14 atributos.

## Formato

Sin encabezado, 14 columnas separadas por coma, `?` = valor faltante.

| Columna | Qué es | Tipo propuesto | Codificación |
|---|---|---|---|
| `age` | edad (años) | numérico | |
| `sex` | sexo | nominal | 1 = varón, 0 = mujer |
| `cp` | tipo de dolor de pecho | nominal | 1 angina típica, 2 atípica, 3 no anginoso, 4 asintomático |
| `trestbps` | presión en reposo (mm Hg) | numérico | |
| `chol` | colesterol sérico (mg/dl) | numérico | |
| `fbs` | glucemia en ayunas > 120 mg/dl | nominal | 1 = sí |
| `restecg` | electrocardiograma en reposo | nominal | 0 normal, 1 anomalía ST-T, 2 hipertrofia ventricular |
| `thalach` | frecuencia cardíaca máxima | numérico | |
| `exang` | angina inducida por ejercicio | nominal | 1 = sí |
| `oldpeak` | depresión del ST inducida por ejercicio | numérico | |
| `slope` | pendiente del segmento ST | nominal | 1 ascendente, 2 plana, 3 descendente |
| `ca` | vasos principales coloreados (0–3) | numérico (conteo) | |
| `thal` | resultado del estudio de talio | nominal | 3 normal, 6 defecto fijo, 7 defecto reversible |
| `num` | **objetivo**: diagnóstico | clase | 0 = ausencia; 1–4 = presencia de enfermedad |

Los códigos de `cp`, `restecg`, `slope` y `thal` **no tienen orden**: se declaran nominales para que el árbol ramifique por valor y no corte con umbrales sobre un código.

## Auditoría de los archivos (04/11/2026)

| Archivo | Filas | Clases (`num` 0 / 1 / 2 / 3 / 4) | Faltantes destacados |
|---|---|---|---|
| Cleveland | 303 | 164 / 55 / 36 / 35 / 13 | `ca`: 4, `thal`: 2. Sin duplicados. |
| Hungría | 294 | 188 / 106 / – / – / – | `ca`: 291, `thal`: 266, `slope`: 190 |
| Suiza | 123 | 8 / 48 / 32 / 30 / 5 | `chol` vale 0 en las 123 filas (faltante disfrazado), `ca`: 118 |
| VA Long Beach | 200 | 51 / 56 / 41 / 42 / 10 | `chol` vale 0 en 49 filas, `trestbps`: 56, `ca`: 198 |

Total: 920 filas. Cleveland es el único archivo prácticamente completo.

## Cómo se carga (`src/tp_ia/data/heart.py`)

- **Archivo:** solo `processed.cleveland.data` (303 filas), por el `WARNING` de los autores y por la auditoría de arriba.
- **Objetivo:** por defecto **binario**: 0 = sin enfermedad (164 casos) y 1 a 4 = con enfermedad (139 casos). Es lo que indica `heart-disease.names`: los experimentos con Cleveland se concentraron en distinguir presencia (1–4) de ausencia (0). También está el escenario `multiclase`, con los cinco valores originales (`load_heart_dataset(scenario="multiclase")`).
- **Nombres en español:** `age` → `edad`, `sex` → `sexo`, `cp` → `tipo_dolor`, `trestbps` → `presion_reposo`, `chol` → `colesterol`, `fbs` → `glucemia_ayunas`, `restecg` → `ecg_reposo`, `thalach` → `frecuencia_max`, `exang` → `angina_ejercicio`, `oldpeak` → `depresion_st`, `slope` → `pendiente_st`, `ca` → `vasos_coloreados`, `thal` → `talio`, `num` → `diagnostico`.
- **Códigos traducidos:** los siete atributos nominales muestran su significado en lugar del código (por ejemplo, `tipo_dolor = sin síntomas` en lugar de `cp = 4`), según la tabla de arriba. `vasos_coloreados` queda numérico porque es un conteo.
- **Faltantes:** se conservan (4 en `vasos_coloreados` y 2 en `talio`), como el resto de los datasets.

Cada una de estas decisiones queda anotada, con su evidencia, en el registro de decisiones del dataset.

## Otra copia: Kaggle (no se usa como principal)

La copia de Kaggle «Heart Disease Dataset» (usuario `johnsmith88`) tiene según su página 1025 filas, objetivo binario y licencia «Unknown». Las discusiones de esa página reportan 723 filas duplicadas y, según un hilo, que son solo datos de Cleveland con el objetivo invertido. Esto último no está verificado contra el archivo.
